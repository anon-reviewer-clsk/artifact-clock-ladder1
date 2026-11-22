"""Loopback mock of an OpenAI-compatible streaming endpoint, for offline testing.

Binds 127.0.0.1 only. Never reaches the network. Behaviour is selected by the
requested model name so a single test run can exercise every branch:

    mock-chunky   40 small content chunks (token-by-token style)
    mock-batched   4 large chunks (server-side batching style)
    mock-single    1 chunk (whole answer at once)
    mock-429       HTTP 429 with Retry-After: 2
    mock-trunc     stops at finish_reason=length, no [DONE]
    mock-nousage   streams normally but omits the usage block

Every response starts with a role-only SSE chunk, so a collector that conflates
"first SSE line" with "first token" is detectable.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega").split()


def _content_chunks(model: str, target_bytes: int = 420) -> list[str]:
    text = " ".join(WORDS)
    while len(text.encode()) < target_bytes:
        text += " " + " ".join(WORDS)
    text = text[:target_bytes]
    n = 40 if "chunky" in model else 4 if "batched" in model else 1
    size = max(1, len(text) // n)
    return [text[i:i + size] for i in range(0, len(text), size)]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep test output clean
        pass

    def _json_error(self, code: int, msg: str, retry_after: str | None = None):
        body = json.dumps({"error": {"message": msg, "code": code}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if retry_after:
            self.send_header("Retry-After", retry_after)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json_error(400, "bad json")

        model = str(body.get("model", ""))
        if "429" in model:
            return self._json_error(429, "rate limited", retry_after="2")

        chunks = _content_chunks(model)
        trunc = "trunc" in model
        no_usage = "nousage" in model
        rid = f"mock-{abs(hash((model, time.time()))) % 10**8}"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj: dict):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        time.sleep(0.03)
        emit({"id": rid, "model": f"{model}-served", "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

        for i, piece in enumerate(chunks):
            time.sleep(0.004)
            last = i == len(chunks) - 1
            obj = {"id": rid, "model": f"{model}-served", "choices": [
                {"index": 0,
                 "delta": {"content": piece},
                 "finish_reason": ("length" if trunc else "stop") if last else None}]}
            if last and not no_usage:
                obj["usage"] = {"prompt_tokens": 29, "completion_tokens": 96,
                                "total_tokens": 125}
            emit(obj)

        if trunc:
            self.wfile.write(b": stream aborted\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        if no_usage:
            emit({"id": rid, "choices": [{"index": 0, "delta": {},
                                          "finish_reason": None}]})

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def start(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, url


if __name__ == "__main__":
    s, u = start()
    print("mock listening:", u)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        s.shutdown()
