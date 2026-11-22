#!/usr/bin/env python3
"""a3_run.py — 按 PREREG_a3.md 采集：每模型两档 × 5 block × 60 请求。

block 间重启服务切换权重（G0 已证换权重不改 quote）。客户端用 keep-alive 连接，
RTT 由 perf_counter 测量，服务端不参与计时。
"""
from __future__ import annotations

import http.client
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/root/a3")
MODELS = ["dsv1_downstream.onnx", "resnet18.onnx", "dsv1_tok_v4.onnx"]
BLOCKS_PER_VARIANT = 5
REQS_PER_BLOCK = 60
PORT = 8089
SERVER = "/root/a3/a3_server.py"


def wait_ready(proc, logf, timeout=120):
    t0 = time.time()
    meta = None
    while time.time() - t0 < timeout:
        time.sleep(0.05)
        txt = Path(logf).read_text(errors="ignore")
        for line in txt.splitlines():
            if line.startswith("META "):
                meta = json.loads(line[5:])
        if "READY" in txt:
            return meta
        if proc.poll() is not None:
            return None
    return None


def run_block(model_path, variant, block_idx, model_name):
    logf = OUT / f"srv_{model_name}_{variant}_{block_idx}.log"
    with open(logf, "w") as lf:
        proc = subprocess.Popen([sys.executable, SERVER, model_path, str(PORT)],
                                stdout=lf, stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid)
    try:
        meta = wait_ready(proc, logf)
        if meta is None:
            print(f"[skip] {model_name} {variant} b{block_idx}: server not ready", flush=True)
            return None
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
        payload = b'{"i":1}'
        rtts = []
        for i in range(REQS_PER_BLOCK):
            t0 = time.perf_counter()
            conn.request("POST", "/infer", body=payload,
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            r.read()
            rtts.append((time.perf_counter() - t0) * 1e3)
        conn.close()
        return dict(model=model_name, variant=variant, block=block_idx,
                    rtts=rtts, meta=meta)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        proc.wait(timeout=10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mi, m in enumerate(MODELS):
        start_with_int8 = (mi % 2 == 1)   # 起始档按模型轮换，抵消顺序效应
        for block in range(1, BLOCKS_PER_VARIANT * 2 + 1):
            is_int8 = ((block - 1) % 2 == 1) ^ start_with_int8
            variant = "int8" if is_int8 else "fp32"
            path = f"/root/models/{'q_' if is_int8 else ''}{m}"
            r = run_block(path, variant, block, m)
            if r is None:
                continue
            rows.append(r)
            rtts = sorted(r["rtts"])
            med = rtts[len(rtts) // 2]
            print(f"[{m}] b{block:02d} {variant:4s} median={med:8.3f}ms "
                  f"min={rtts[0]:.2f} max={rtts[-1]:.2f}", flush=True)
    with open(OUT / "a3_raw.json", "w") as f:
        json.dump(rows, f)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
