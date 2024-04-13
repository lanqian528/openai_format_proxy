import base64
import io
import json
import math
import random
import string
import time

import httpx
import tiktoken
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse, JSONResponse, Response

from Logger import logger

app = FastAPI()

model_proxy = {
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k-0613",
    "gpt-4": "gpt-4-0613",
    "gpt-4-32k": "gpt-4-32k-0613",
    "gpt-4-turbo-preview": "gpt-4-0125-preview",
    "gpt-4-vision-preview": "gpt-4-1106-vision-preview",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09"
}

model_system_fingerprint = {
    "gpt-3.5-turbo-0125": ["fp_b28b39ffa8"],
    "gpt-3.5-turbo-1106": ["fp_592ef5907d"],
    "gpt-4-0125-preview": ["fp_f38f4d6482", "fp_2f57f81c11", "fp_a7daf7c51e", "fp_a865e8ede4", "fp_13c70b9f70",
                           "fp_b77cb481ed"],
    "gpt-4-1106-preview": ["fp_e467c31c3d", "fp_d986a8d1ba", "fp_99a5a401bb", "fp_123d5a9f90", "fp_0d1affc7a6",
                           "fp_5c95a4634e"],
    "gpt-4-turbo-2024-04-09": ["fp_d1bac968b4", "fp_6fcb0929db", "fp_e37a4b9591", "fp_446ee9db59"]
}


def decode_base64_image(base64_string):
    if "base64," in base64_string:
        base64_str = base64_string.split("base64,")[1]
    image_data = base64.b64decode(base64_str)
    image_file = io.BytesIO(image_data)
    with Image.open(image_file) as img:
        img.load()
        return img.size


def fetch_and_open_image(url):
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
    partial_image_data = io.BytesIO(response.content)
    with Image.open(partial_image_data) as img:
        img.load()
        return img.size


def calculate_image_tokens(image_url):
    url = image_url.get("url")
    detail = image_url.get("detail", "auto")
    if detail == "low":
        return 85
    else:
        if url.startswith("data:image"):
            size = decode_base64_image(url)
        else:
            size = fetch_and_open_image(url)
        width, height = size

        max_dimension = max(width, height)
        if max_dimension > 2048:
            scale_factor = 2048 / max_dimension
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            new_width = width
            new_height = height

        width, height = new_width, new_height
        min_dimension = min(width, height)
        if min_dimension > 768:
            scale_factor = 768 / min_dimension
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            new_width = width
            new_height = height

        width, height = new_width, new_height
        num_masks_w = math.ceil(width / 512)
        num_masks_h = math.ceil(height / 512)
        total_masks = num_masks_w * num_masks_h

        tokens_per_mask = 170
        total_tokens = total_masks * tokens_per_mask + 85

        return total_tokens


def num_tokens_from_messages(messages, model=None):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
    else:
        tokens_per_message = 3
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if isinstance(value, list):
                for item in value:
                    if item.get("type") == "text":
                        num_tokens += len(encoding.encode(item.get("text")))
                    if item.get("type") == "image_url":
                        num_tokens += calculate_image_tokens(item.get("image_url"))
            else:
                num_tokens += len(encoding.encode(value))
    num_tokens += 3
    return num_tokens


def num_tokens_from_content(content, model=None):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    encoded_content = encoding.encode(content)
    len_encoded_content = len(encoded_content)
    return len_encoded_content


def split_tokens_from_content(content, max_tokens, model=None):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    encoded_content = encoding.encode(content)
    len_encoded_content = len(encoded_content)
    if len_encoded_content >= max_tokens:
        content = encoding.decode(encoded_content[:max_tokens])
        return content, max_tokens, "length"
    else:
        return content, len_encoded_content, "stop"


async def stream_response(response, model, max_tokens):
    radnom_chat_id = f"chatcmpl-{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(29))}"
    system_fingerprint_list = model_system_fingerprint.get(model, None)
    system_fingerprint = random.choice(system_fingerprint_list) if system_fingerprint_list else None
    chat_object = "chat.completion.chunk"
    chat_id = radnom_chat_id
    created_time = int(time.time())
    completion_tokens = -1
    end = False
    async for chunk in response.aiter_lines():
        if end:
            yield "data: [DONE]\n\n"
            break
        try:
            if chunk == "data: [DONE]":
                yield "data: [DONE]\n\n"
            elif not chunk.startswith("data:"):
                continue
            else:
                chunk_old_data = json.loads(chunk[6:])
                if not chunk_old_data.get("choices"):
                    continue
                index = chunk_old_data["choices"][0].get("index", 0)
                delta = chunk_old_data["choices"][0].get("delta", {})
                logprobs = chunk_old_data["choices"][0].get("logprobs", None)
                finish_reason = chunk_old_data["choices"][0].get("finish_reason", None)
                if finish_reason == "stop" or finish_reason == "length":
                    end = True
                elif completion_tokens >= max_tokens:
                    delta = {}
                    finish_reason = "length"
                    end = True
                chunk_new_data = {
                    "id": chat_id,
                    "object": chat_object,
                    "created": created_time,
                    "model": model,
                    "choices": [
                        {
                            "index": index,
                            "delta": delta,
                            "logprobs": logprobs,
                            "finish_reason": finish_reason
                        }
                    ],
                    "system_fingerprint": system_fingerprint
                }
                completion_tokens += 1
                yield f"data: {json.dumps(chunk_new_data)}\n\n"
        except Exception as e:
            logger.error(f"Error: {chunk}")
            continue


def chat_response(resp, model, send_message, max_tokens):
    choices = []

    chat_id = f"chatcmpl-{''.join(random.choice(string.ascii_letters + string.digits) for _ in range(29))}"
    chat_object = "chat.completion"
    created_time = int(time.time())
    total_completion_tokens = 0
    for i in range(len(resp["choices"])):
        index = resp["choices"][i].get("index", 0)
        message = resp["choices"][i].get("message", None)
        logprobs = resp["choices"][i].get("logprobs", None)
        message_content, completion_tokens, finish_reason = split_tokens_from_content(message["content"], max_tokens,
                                                                                      model)
        message["content"] = message_content

        choices.append({
            "index": index,
            "message": message,
            "logprobs": logprobs,
            "finish_reason": finish_reason
        })

        total_completion_tokens += completion_tokens
    usage = resp.get("usage", None)
    if not usage or not usage.get("total_tokens", 0):
        prompt_tokens = num_tokens_from_messages(send_message, model)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens
        }
    system_fingerprint_list = model_system_fingerprint.get(model, None)
    system_fingerprint = random.choice(system_fingerprint_list) if system_fingerprint_list else None
    chat_response_json = {
        "id": chat_id,
        "object": chat_object,
        "created": created_time,
        "model": model,
        "choices": choices,
        "usage": usage,
        "system_fingerprint": system_fingerprint
    }
    return chat_response_json


def image_response(resp):
    created_time = resp.get("created", int(time.time()))
    revised_prompt = resp["data"][0].get("revised_prompt", None)
    image_url = resp["data"][0].get("url", None)
    image_response_json = {
        "created": created_time,
        "data": [{"url": image_url}]}
    if revised_prompt:
        resp["data"][0]["revised_prompt"] = revised_prompt
    return image_response_json


@app.api_route("/{proxy_url}/v1/images/generations", methods=["POST"])
async def proxy_v1_images_generations(request: Request, proxy_url: str):
    url = f"http://{proxy_url}/v1/images/generations" if ":" in proxy_url else f"https://{proxy_url}/v1/images/generations"
    params = request.query_params
    headers = {'Authorization': request.headers.get('Authorization', '')}
    try:
        s = httpx.AsyncClient()
        try:
            data = await request.json()
        except:
            data = {}
        response = await s.post(url, json=data, params=params, headers=headers, timeout=600)
        if response.status_code == 200:
            resp = response.json()
            return JSONResponse(content=image_response(resp), media_type=response.headers['Content-Type'],
                                status_code=response.status_code, background=BackgroundTask(response.aclose))

        return Response(content=response.content, media_type=response.headers['Content-Type'],
                        status_code=response.status_code, background=BackgroundTask(response.aclose))
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(content={"error": "Something went wrong."}, status_code=500)


@app.api_route("/{proxy_url}/v1/chat/completions", methods=["POST"])
async def proxy_v1_chat_completions(request: Request, proxy_url: str):
    url = f"http://{proxy_url}/v1/chat/completions" if ":" in proxy_url else f"https://{proxy_url}/v1/chat/completions"
    params = request.query_params
    headers = {'Authorization': request.headers.get('Authorization', '')}
    try:
        s = httpx.AsyncClient()
        try:
            data = await request.json()
        except:
            data = {}
        model = data.get("model", "gpt-3.5-turbo")
        model = model_proxy.get(model, model)
        send_message = data.get("messages", [])
        max_tokens = data.get("max_tokens", 2147483647)
        stream = data.get("stream", False)
        response = await s.send(
            s.build_request("POST", url, json=data, params=params, headers=headers, timeout=600), stream=True)
        if stream and response.status_code == 200:
            return StreamingResponse(stream_response(response, model, max_tokens),
                                     media_type=response.headers['Content-Type'],
                                     status_code=response.status_code, background=BackgroundTask(response.aclose))
        else:
            await response.aread()
            if response.status_code == 200:
                resp = response.json()
                return JSONResponse(content=chat_response(resp, model, send_message, max_tokens),
                                    media_type=response.headers['Content-Type'],
                                    status_code=response.status_code, background=BackgroundTask(response.aclose))

            return Response(content=response.content, media_type=response.headers['Content-Type'],
                            status_code=response.status_code, background=BackgroundTask(response.aclose))
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(content={"error": "Something went wrong."}, status_code=500)


@app.api_route("/{proxy_url}/{path:path}", methods=["GET", "POST"])
async def proxy_others(request: Request, proxy_url: str, path: str):
    if ":" in proxy_url:
        url = f"http://{proxy_url}/{path}"
    else:
        url = f"https://{proxy_url}/{path}"
    params = request.query_params
    headers = {'Authorization': request.headers.get('Authorization', '')}
    try:
        if request.method == "GET":
            s = httpx.AsyncClient()
            response = await s.get(url, params=params, headers=headers, timeout=600)
            return Response(content=response.content, media_type=response.headers['Content-Type'],
                            status_code=response.status_code,
                            background=BackgroundTask(response.aclose))
        if request.method == "POST":
            s = httpx.AsyncClient()
            try:
                data = await request.json()
            except:
                data = {}
            response = await s.post(url, json=data, params=params, headers=headers, timeout=600)
            return Response(content=response.content, media_type=response.headers['Content-Type'],
                            status_code=response.status_code,
                            background=BackgroundTask(response.aclose))
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(content={"error": "Something went wrong."}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("format_proxy:app", host="0.0.0.0", port=5003)
