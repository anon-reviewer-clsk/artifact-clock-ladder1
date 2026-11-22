#!/usr/bin/env python3
"""a3_server.py — 在 TD 内跑的极简推理服务（单线程，串行处理，便于测时延）。

启动时加载一个模型，预热 3 次，打印 READY 与元数据；POST /infer 触发一次推理，
只回 {"ok":1}（不报时，时间由客户端测）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8089
SEED = 20261121

_t0 = time.perf_counter()
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess = ort.InferenceSession(MODEL, sess_options=so, providers=["CPUExecutionProvider"])
LOAD_MS = (time.perf_counter() - _t0) * 1e3

_rng = np.random.default_rng(SEED)
_IN = sess.get_inputs()[0]
_SHAPE = [1 if (d is None or isinstance(d, str)) else int(d) for d in _IN.shape]
OUT = [o.name for o in sess.get_outputs()]

def _mk(kind):
    n = int(np.prod(_SHAPE))
    if kind == "zeros":
        a = np.zeros(_SHAPE, np.float32)
    elif kind == "ones":
        a = np.ones(_SHAPE, np.float32)
    elif kind == "small":
        a = np.full(_SHAPE, 1e-6, np.float32)
    elif kind.startswith("mag:"):
        mag = float(kind.split(":", 1)[1])
        g = np.random.default_rng(SEED).standard_normal(_SHAPE).astype(np.float32)
        a = (g / np.abs(g).max() * mag).astype(np.float32)
    elif kind == "checker":
        a = (np.arange(n).reshape(_SHAPE) % 2 * 2 - 1).astype(np.float32)
    else:
        a = np.random.default_rng(SEED).standard_normal(_SHAPE).astype(np.float32)
    return {_IN.name: a}

KINDS = ("rand", "zeros", "ones", "small", "checker",
         "mag:0.1", "mag:0.8", "mag:1.6", "mag:3.2", "mag:6.0")
FEEDS = {k: _mk(k) for k in KINDS}

for k in KINDS:
    for _ in range(2):
        sess.run(OUT, FEEDS[k])


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        # 只回 ok，不做推理：用于测隧道/网络基线
        body = b'{"ok":1}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            kind = json.loads(raw or b"{}").get("kind", "rand")
        except Exception:
            kind = "rand"
        res = sess.run(OUT, FEEDS.get(kind, FEEDS["rand"]))
        payload = {"ok": 1}
        if len(res) > 1:
            payload["idx"] = [int(v) for v in np.array(res[1]).ravel()[:2]]
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    meta = dict(model=os.path.basename(MODEL), path=MODEL,
                sha256=sha256_file(MODEL), size=os.path.getsize(MODEL),
                load_ms=round(LOAD_MS, 1), ort=ort.__version__,
                inputs={k: list(v.shape) for k, v in FEEDS["rand"].items()})
    print("META " + json.dumps(meta), flush=True)
    print("READY", flush=True)
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
