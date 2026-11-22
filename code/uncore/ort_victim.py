#!/usr/bin/env python3
"""ort_victim.py — 未修改的正常 ONNX 推理进程（阶段 1 的 victim）。

行为：
  1) 加载指定 .onnx，创建单线程 CPU 会话；
  2) 用确定性输入做一次 warmup 与 sanity check（输出形状 + 有限性 + 校验和）；
  3) 打印一行 METADATA（含 sha256、大小、输入规格、RSS、输入缓冲物理地址）；
  4) 等待 SIGUSR1，然后执行 N 次**正常推理**，记录每次延迟；
  5) 打印一行 RESULT。

不做：clflush、CR0.CD、任何减速。输入固定、种子固定。
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import sys
import time

import numpy as np
import onnxruntime as ort

GO = 0


def _h(sig, _frm):
    global GO
    GO = 1


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rss_kb() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def _pa_of(ptr: int) -> tuple[int, int]:
    """返回 (PA_of_page, page_count_present) —— 需要 root 读 pagemap。"""
    with open("/proc/self/pagemap", "rb") as f:
        f.seek((ptr // 4096) * 8)
        entry = int.from_bytes(f.read(8), "little")
    if not (entry >> 63) & 1:
        return -1, 0
    return ((entry & 0x7FFFFFFFFFFFFF) << 12) + (ptr % 4096), 1


def _dtype_of(meta) -> np.dtype:
    return {"tensor(float)": np.float32, "tensor(float16)": np.float16,
            "tensor(double)": np.float64, "tensor(int64)": np.int64,
            "tensor(int32)": np.int32, "tensor(uint8)": np.uint8}.get(meta.type, np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261120)
    a = ap.parse_args()

    signal.signal(signal.SIGUSR1, _h)

    t0 = time.perf_counter()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(a.model, sess_options=so,
                                providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1e3

    feeds = {}
    spec = []
    rng = np.random.default_rng(a.seed)
    for i in sess.get_inputs():
        shape = [1 if (d is None or isinstance(d, str)) else int(d) for d in i.shape]
        dt = _dtype_of(i)
        arr = (rng.standard_normal(shape).astype(dt) if np.issubdtype(dt, np.floating)
               else np.zeros(shape, dtype=dt))
        feeds[i.name] = arr
        spec.append({"name": i.name, "shape": shape, "dtype": str(dt)})

    out_names = [o.name for o in sess.get_outputs()]
    t1 = time.perf_counter()
    outs = sess.run(out_names, feeds)
    first_ms = (time.perf_counter() - t1) * 1e3

    checks = []
    ok = True
    for name, arr in zip(out_names, outs):
        finite = bool(np.isfinite(arr).all()) if np.issubdtype(arr.dtype, np.floating) else True
        checks.append({"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype),
                       "finite": finite, "sum": float(np.nansum(arr.astype(np.float64)))})
        ok = ok and finite and arr.size > 0

    # 输入缓冲物理地址（用于 PKT_MATCH 目标地址；victim 每个 iteration 都会读它）
    inbuf = next(iter(feeds.values()))
    pa_in, present = _pa_of(inbuf.ctypes.data)

    meta = dict(
        model=os.path.basename(a.model), path=a.model,
        sha256=sha256_file(a.model), size_bytes=os.path.getsize(a.model),
        load_ms=round(load_ms, 1), first_infer_ms=round(first_ms, 2),
        inputs=spec, outputs=checks, sanity_ok=bool(ok),
        rss_kb=_rss_kb(), rss_after_load_kb=_rss_kb(),
        input_pa=pa_in, input_pa_present=present,
        ort=ort.__version__, threads=1, seed=a.seed, iters=a.iters,
    )
    print("METADATA " + json.dumps(meta), flush=True)
    if not ok:
        print("SANITY_FAIL", flush=True)
        return 2

    print("READY", flush=True)
    while not GO:
        time.sleep(0.01)

    lat = []
    t_all = time.perf_counter()
    for _ in range(a.iters):
        t = time.perf_counter()
        sess.run(out_names, feeds)
        lat.append((time.perf_counter() - t) * 1e3)
    total_ms = (time.perf_counter() - t_all) * 1e3

    res = dict(model=os.path.basename(a.model), iters=a.iters,
               total_ms=round(total_ms, 2),
               p50_ms=round(float(np.percentile(lat, 50)), 3),
               p95_ms=round(float(np.percentile(lat, 95)), 3),
               rate_per_s=round(a.iters / (total_ms / 1e3), 2) if total_ms else None,
               rss_kb=_rss_kb())
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
