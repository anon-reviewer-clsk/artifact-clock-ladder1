#!/usr/bin/env python3
"""a3_input_run.py — 权重固定、只变输入内容：时延是否随输入变化。

在 TD 内本机回环跑（隔离网络）。每种输入 5 block × 30 请求，block 内轮转输入顺序。
"""
from __future__ import annotations

import http.client
import json
import os
import signal
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/root/a3")
MODELS = ["resnet18.onnx", "dsv1_downstream.onnx"]
KINDS = ["rand", "zeros", "ones", "small", "checker"]
BLOCKS = 5
REQ = 30
PORT = 8093


def wait_ready(proc, logf, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.05)
        if "READY" in Path(logf).read_text(errors="ignore"):
            return True
        if proc.poll() is not None:
            return False
    return False


def main():
    rows = []
    for m in MODELS:
        logf = OUT / f"inp_{m}.log"
        with open(logf, "w") as lf:
            proc = subprocess.Popen([sys.executable, "/root/a3/a3_server.py",
                                     f"/root/models/{m}", str(PORT)],
                                    stdout=lf, stderr=subprocess.STDOUT,
                                    preexec_fn=os.setsid)
        try:
            if not wait_ready(proc, logf):
                print(f"[skip] {m}", flush=True)
                continue
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
            for b in range(1, BLOCKS + 1):
                order = KINDS[b % len(KINDS):] + KINDS[: b % len(KINDS)]
                for kind in order:
                    body = json.dumps({"kind": kind}).encode()
                    rtts = []
                    for _ in range(REQ):
                        t0 = time.perf_counter()
                        conn.request("POST", "/infer", body=body,
                                     headers={"Content-Type": "application/json"})
                        r = conn.getresponse(); r.read()
                        rtts.append((time.perf_counter() - t0) * 1e3)
                    rows.append(dict(model=m, kind=kind, block=b, rtts=rtts,
                                     med=st.median(rtts)))
                    print(f"[{m}] b{b} {kind:8s} med={st.median(rtts):8.3f}ms", flush=True)
            conn.close()
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            proc.wait(timeout=10)
    json.dump(rows, open(OUT / "input_raw.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
