#!/usr/bin/env python3
"""wan_client.py — 只做时延测量（本地运行，经 SSH 隧道打到 TD 内的服务）。

用法：python wan_client.py --tag "model:variant:block" --n 60 --port 18089
输出：向 results.jsonl 追加一行，含每次请求的 RTT、ping 基线、tag。
"""
from __future__ import annotations

import argparse
import http.client
import json
import statistics as st
import time
from pathlib import Path

OUT = Path(__file__).with_name("wan_results.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--port", type=int, default=18089)
    ap.add_argument("--pings", type=int, default=10)
    a = ap.parse_args()

    conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=120)
    # 隧道基线：/ping 不触发推理
    pings = []
    for _ in range(a.pings):
        t0 = time.perf_counter()
        conn.request("GET", "/ping")
        r = conn.getresponse()
        r.read()
        pings.append((time.perf_counter() - t0) * 1e3)

    rtts = []
    for _ in range(a.n):
        t0 = time.perf_counter()
        conn.request("POST", "/infer", body=b'{"i":1}',
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        r.read()
        rtts.append((time.perf_counter() - t0) * 1e3)
    conn.close()

    row = dict(tag=a.tag, n=a.n, rtts=rtts, pings=pings,
               ping_med=round(st.median(pings), 2),
               rtt_med=round(st.median(rtts), 2))
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"{a.tag:38s} ping={row['ping_med']:7.2f}ms  rtt={row['rtt_med']:8.2f}ms")


if __name__ == "__main__":
    main()
