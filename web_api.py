"""
Web API server for LLaMA model inference.
"""

import json
import random
import time
from pathlib import Path

from bottle import Bottle, HTTPResponse, abort, request, response, static_file
from tinygrad import GlobalCounters, Tensor


def create_web_api(model, tokenizer, device, args):
    """Create and configure the web API"""
    app = Bottle()

    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Authorization",
        "Access-Control-Allow-Credentials": "true",
    }

    @app.hook("before_request")
    def handle_options():
        if request.method == "OPTIONS":
            raise HTTPResponse(headers=cors_headers)

    @app.hook("after_request")
    def enable_cors():
        for key, value in cors_headers.items():
            response.set_header(key, value)

    # Static file serving
    @app.route("/<filename>")
    def server_static(filename):
        return static_file(filename, root=(Path(__file__).parent / "tinychat").as_posix())

    @app.route("/assets/<filename:path>")
    def server_assets(filename):
        return static_file(filename, root=(Path(__file__).parent / "tinychat" / "assets").as_posix())

    @app.route("/")
    def index():
        return static_file("index.html", root=(Path(__file__).parent / "tinychat").as_posix())

    # API endpoints
    @app.get("/v1/models")
    def models():
        return json.dumps([str(args.model)])

    @app.post("/v1/internal/token-count")
    def token_count():
        rjson = json.loads(request.body.read())
        return json.dumps(len(tokenizer.encode(rjson.get("text", ""))))

    @app.post("/v1/token/encode")
    def token_encode():
        rjson = json.loads(request.body.read())
        return json.dumps(tokenizer.encode(rjson.get("text", "")))

    @app.post("/v1/completions")
    def completions():
        from generation import ALPHA_F, ALPHA_P, TEMPERATURE, TOP_K, TOP_P, prefill

        rjson = json.loads(request.body.read())

        if rjson.get("stream", False):
            response.content_type = "text/event-stream"
            response.set_header("Cache-Control", "no-cache")
        else:
            abort(400, "streaming required")

        toks = [tokenizer.bos_id] + tokenizer.encode(rjson.get("prompt", ""), allow_special=True)

        start_pos = prefill(model, toks[:-1])
        last_tok = toks[-1]
        while True:
            GlobalCounters.reset()
            tok = model(
                Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P
            ).item()
            start_pos += 1
            last_tok = tok
            if tok in tokenizer.stop_tokens:
                break

            res = {
                "choices": [
                    {
                        "text": tokenizer.decode([tok]),
                    }
                ]
            }
            yield f"data: {json.dumps(res)}\n\n"

    @app.post("/v1/chat/token/encode")
    def chat_token_encode():
        from generation import encode_message, encode_role

        rjson = json.loads(request.body.read())
        if "messages" not in rjson:
            abort(400, "messages required")
        toks = [tokenizer.bos_id]
        for message in rjson["messages"]:
            toks += encode_message(message["role"], message["content"], tokenizer)
        if len(rjson["messages"]) > 0 and message["role"] == "user":
            toks += encode_role("assistant", tokenizer)
        return json.dumps(toks)

    @app.post("/v1/chat/completions")
    def chat_completions():
        from generation import (
            ALPHA_F,
            ALPHA_P,
            TEMPERATURE,
            TOP_K,
            TOP_P,
            encode_message,
            encode_role,
            last_seen_toks,
            prefill,
        )

        rjson = json.loads(request.body.read())
        if "messages" not in rjson:
            abort(400, "messages required")

        if rjson.get("stream", False):
            response.content_type = "text/event-stream"
            response.set_header("Cache-Control", "no-cache")
        else:
            abort(400, "streaming required")

        toks = [tokenizer.bos_id]
        for message in rjson["messages"]:
            toks += encode_message(message["role"], message["content"], tokenizer)
        if message["role"] != "user":
            abort(400, "last message must be a user message")
        toks += encode_role("assistant", tokenizer)

        random_id = random.randbytes(16).hex()

        start_pos = prefill(model, toks[:-1])
        last_tok = toks[-1]
        last_seen_toks.append(last_tok)
        while True:
            GlobalCounters.reset()
            tok = model(
                Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P
            ).item()
            start_pos += 1
            last_tok = tok
            last_seen_toks.append(tok)
            if tok in tokenizer.stop_tokens:
                break

            res = {
                "id": random_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": str(args.model),
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": tokenizer.decode([tok]),
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(res)}\n\n"

        res = {
            "id": random_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": str(args.model),
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(res)}\n\n"

    return app
