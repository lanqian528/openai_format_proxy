import json
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse, JSONResponse, Response

from Logger import Logger

app = FastAPI()


async def stream_response(response, model):
    async for chunk in response.aiter_lines():
        try:
            if chunk == "data: [DONE]":
                yield f"data: [DONE]\n\n"
            elif not chunk.startswith("data:"):
                continue
            else:
                chunk_old_data = json.loads(chunk[6:])
                chat_id = chunk_old_data.get("id", "")
                if not chat_id:
                    continue
                chat_object = chunk_old_data.get("object", "chat.completion.chunk")
                created_time = chunk_old_data.get("created", int(time.time()))
                index = chunk_old_data["choices"][0].get("index", 0)
                delta = chunk_old_data["choices"][0].get("delta", None)
                finish_reason = chunk_old_data["choices"][0].get("finish_reason", None)
                system_fingerprint = chunk_old_data.get("system_fingerprint", "fp_3bc1b5746c")
                chunk_new_data = {
                    "id": chat_id,
                    "object": chat_object,
                    "created": created_time,
                    "model": model,
                    "choices": [
                        {
                            "index": index,
                            "delta": delta,
                            "finish_reason": finish_reason
                        }
                    ],
                    "system_fingerprint": system_fingerprint
                }
                yield f"data: {json.dumps(chunk_new_data)}\n\n"
        except Exception as e:
            Logger.error(f"Error: {str(e)}")
            Logger.error(f"Error: {chunk}")
            continue


def chat_response(resp, model):
    chat_id = resp.get("id", "")
    chat_object = resp.get("object", "chat.completion")
    created_time = resp.get("created", int(time.time()))
    index = resp["choices"][0].get("index", 0)
    message = resp["choices"][0].get("message", None)
    finish_reason = resp["choices"][0].get("finish_reason", None)
    usage = resp.get("usage", None)
    system_fingerprint = resp.get("system_fingerprint", "fp_3bc1b5746c")
    chat_response_json = {
        "id": chat_id,
        "object": chat_object,
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": index,
                "message": message,
                "finish_reason": finish_reason
            }
        ],
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
        "data": [
            {
                "url": image_url
            }
        ]
    }
    if revised_prompt:
        resp["data"][0]["revised_prompt"] = revised_prompt
    return image_response_json


@app.api_route("/{proxy_url}/v1/images/generations", methods=["POST"])
async def proxy_v1_images_generations(request: Request, proxy_url: str):
    if ":" in proxy_url:
        url = f"http://{proxy_url}/v1/images/generations"
    else:
        url = f"https://{proxy_url}/v1/images/generations"

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
        Logger.error(str(e))
        return JSONResponse(content={"error": "Something went wrong."}, status_code=500)


@app.api_route("/{proxy_url}/v1/chat/completions", methods=["POST"])
async def proxy_v1_chat_completions(request: Request, proxy_url: str):
    if ":" in proxy_url:
        url = f"http://{proxy_url}/v1/chat/completions"
    else:
        url = f"https://{proxy_url}/v1/chat/completions"
    params = request.query_params
    headers = {'Authorization': request.headers.get('Authorization', '')}

    try:
        s = httpx.AsyncClient()
        try:
            data = await request.json()
        except:
            data = {}
        model = data.get("model", "")
        stream = data.get("stream", False)
        if stream:
            response = await s.send(
                s.build_request("POST", url, json=data, params=params, headers=headers, timeout=600), stream=True)
            if response.status_code == 200:
                return StreamingResponse(stream_response(response, model), media_type=response.headers['Content-Type'],
                                         status_code=response.status_code, background=BackgroundTask(response.aclose))
        else:
            response = await s.post(url, json=data, params=params, headers=headers, timeout=600)
            if response.status_code == 200:
                resp = response.json()
                return JSONResponse(content=chat_response(resp, model), media_type=response.headers['Content-Type'],
                                    status_code=response.status_code, background=BackgroundTask(response.aclose))

            return Response(content=response.content, media_type=response.headers['Content-Type'],
                            status_code=response.status_code, background=BackgroundTask(response.aclose))
    except Exception as e:
        Logger.error(str(e))
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
        Logger.error(str(e))
        return JSONResponse(content={"error": "Something went wrong."}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("format_proxy:app", host="0.0.0.0", port=5003)
