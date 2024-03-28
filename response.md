## Response Body

### stream: False
```
{
    "id": "chatcmpl-96dMyLdF3kSYH14rnhMi8HzbiurGJ",
    "object": "chat.completion",
    "created": 1711368332,
    "model": "gpt-3.5-turbo-0125",
    "choices": [
        {
            "index": 0,
            "message": {
                "content": "xxxxxx",
                "role": "assistant"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "completion_tokens": 1,
        "prompt_tokens": 10,
        "total_tokens": 11
    },
    "system_fingerprint": "fp_2f57f81c11"
}
```

### stream: True
```
{
    "id": "chatcmpl-96dMyLdF3kSYH14rnhMi8HzbiurGJ",
    "object": "chat.completion.chunk",
    "created": 1711368332,
    "model": "gpt-3.5-turbo-0125",
    "system_fingerprint": "fp_3bc1b5746c",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": ""
            },
            "logprobs": null,
            "finish_reason": null
        }
    ]
}
{
    "id": "chatcmpl-96dMyLdF3kSYH14rnhMi8HzbiurGJ",
    "object": "chat.completion.chunk",
    "created": 1711368332,
    "model": "gpt-3.5-turbo-0125",
    "system_fingerprint": "fp_3bc1b5746c",
    "choices": [
        {
            "index": 0,
            "delta": {
                "content": "xxxxxx"
            },
            "logprobs": null,
            "finish_reason": null
        }
    ]
}
{
    "id": "chatcmpl-96dMyLdF3kSYH14rnhMi8HzbiurGJ",
    "object": "chat.completion.chunk",
    "created": 1711368332,
    "model": "gpt-3.5-turbo-0125",
    "system_fingerprint": "fp_3bc1b5746c",
    "choices": [
        {
            "index": 0,
            "delta": {},
            "logprobs": null,
            "finish_reason": "stop"
        }
    ]
}
```

### image
```
{
    "created": 1711572163,
    "data": [
        {
            "revised_prompt": "xxxxxx", // dall-e-2 not support revised_prompt
            "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/xxxxxx"
        }
    ]
}
```