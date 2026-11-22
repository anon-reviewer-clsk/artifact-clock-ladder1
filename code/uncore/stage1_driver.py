#!/usr/bin/env python3
"""stage1_driver.py — 阶段 1 采集：正常推理期间核外 PMC 是否可重复观测。

每个 (模型, run) 做三件事：
  A 推理前空载 3 s
  B 正常推理 3 s（victim 收到 SIGUSR1 后跑 iters 次推理）
  C 结束后空载 3 s
两套过滤：
  coarse  —— ADDRMASK=0xFFFFFFFF（不比较地址，计该 box 全部 PKT）
  filtered—— 精确指向 victim 输入缓冲的物理地址（对照组指向 +64 MiB 的非目标地址）

victim 是未修改的正常推理进程；不做 clflush、不改 CR0.CD、不减速。
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

MODELS = os.environ.get(
    "S1_MODELS",
    "dsv1_tok_v4.onnx,dsv1_tok_v4_l1.onnx,resnet18.onnx,dsv1_downstream.onnx").split(",")
BOXES = ["fe:0c.0", "fe:0d.0", "fe:0e.0", "fe:0f.0"]
EV = ",".join(f"uncore_m2m_{i}/event=0x4c,umask=0x3/" for i in range(4))
WINDOW = 3.0
ITERS = 200
RUNS = int(os.environ.get("S1_RUNS", "3"))
FILTERED_MODEL = os.environ.get("S1_FILTERED", "resnet18.onnx")
OUT = Path(os.environ.get("S1_OUT", "/root/phase1"))


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def prog(box, addr=None, coarse=False):
    if coarse:
        lo, hi, mlo, mhi = "00000000", "00000000", "ffffffff", "ffffffff"
    else:
        lo = f"{addr & 0xFFFFFFC0:08x}"
        hi = f"{addr >> 32:08x}"
        mlo = mhi = "00000000"
    sh(["setpci", "-s", box, f"0x490.L={lo}"])
    sh(["setpci", "-s", box, f"0x494.L={hi}"])
    sh(["setpci", "-s", box, f"0x498.L={mlo}"])
    sh(["setpci", "-s", box, f"0x49c.L={mhi}"])
    sh(["setpci", "-s", box, "0x4a0.L=000000f0"])


def unprog(box):
    sh(["setpci", "-s", box, "0x4a0.L=00000000"])
    sh(["setpci", "-s", box, "0x490.L=00000000"])


def perf(tag="", window=WINDOW):
    r = sh(["perf", "stat", "-e", EV, "-a", "--", "sleep", str(window)])
    txt = r.stdout + r.stderr
    if tag:
        (OUT / f"perf_{tag}.log").write_text(txt, encoding="utf-8")
    vals = []
    for i in range(4):
        m = re.search(rf"^\s*([\d,]+)\s+uncore_m2m_{i}/", txt, re.M)
        vals.append(int(m.group(1).replace(",", "")) if m else -1)
    return vals, txt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tsv = OUT / "stage1.tsv"
    if tsv.exists():
        tsv.unlink()
    tsv.write_text(
        "model\trun\tfilter\ttarget\tphase\tbox0\tbox1\tbox2\tbox3\t"
        "rtt_p50_ms\trss_kb\tinput_pa\talive\n", encoding="utf-8")

    plan = [(m, r, "coarse") for m in MODELS for r in range(1, RUNS + 1)]
    for r in range(1, RUNS + 1):
        plan.append((FILTERED_MODEL, r, "filtered"))
        plan.append((FILTERED_MODEL, r, "filtered_neg"))

    for i, (model, run, kind) in enumerate(plan):
        # 交错顺序：模型按 Latin 轮转（不同 run 的起点不同）
        vfile = OUT / f"v_{model}_{run}_{kind}.log"
        vf = open(vfile, "w")
        v = subprocess.Popen(
            ["numactl", "--membind=1", "--physcpubind=40",
             "python3", "/root/phase1/ort_victim.py",
             "--model", f"/root/models/{model}", "--iters", str(ITERS)],
            stdout=vf, stderr=subprocess.STDOUT)
        meta = None
        for _ in range(200):
            time.sleep(0.1)
            txt = vfile.read_text(errors="ignore") if vfile.exists() else ""
            mm = re.search(r"^METADATA (\{.*\})$", txt, re.M)
            if mm:
                meta = json.loads(mm.group(1))
            if "READY" in txt:
                break
        if not meta:
            print(f"[skip] {model} run{run}: no METADATA", flush=True)
            v.kill()
            continue

        pa = meta.get("input_pa", -1)
        target = pa if kind == "filtered" else (pa + 64 * 1024 * 1024 if kind == "filtered_neg" else 0)
        for box in BOXES:
            prog(box, addr=target, coarse=(kind == "coarse"))

        idle, _ = perf(f"{model}_{run}_{kind}_idle")   # A: 推理前空载
        with open(tsv, "a", encoding="utf-8") as f:
            f.write(f"{model}\t{run}\t{kind}\t{target}\tidle_before\t" +
                    "\t".join(map(str, idle)) + f"\t\t{meta['rss_kb']}\t{pa}\t-\n")

        p = subprocess.Popen(["perf", "stat", "-e", EV, "-a", "--", "sleep", str(WINDOW)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(0.3)
        t_infer0 = time.time()
        v.send_signal(signal.SIGUSR1)
        pout = p.communicate()[0]
        (OUT / f"perf_{model}_{run}_{kind}_infer.log").write_text(pout, encoding="utf-8")
        vals = []
        for k in range(4):
            m = re.search(rf"^\s*([\d,]+)\s+uncore_m2m_{k}/", pout, re.M)
            vals.append(int(m.group(1).replace(",", "")) if m else -1)
        try:
            v.wait(timeout=60)
            alive = 0
        except subprocess.TimeoutExpired:
            v.kill()
            alive = 1
        txt = vfile.read_text(errors="ignore")
        mr = re.search(r"^RESULT (\{.*\})$", txt, re.M)
        rtt = json.loads(mr.group(1))["p50_ms"] if mr else ""
        with open(tsv, "a", encoding="utf-8") as f:
            f.write(f"{model}\t{run}\t{kind}\t{target}\tinference\t" +
                    "\t".join(map(str, vals)) + f"\t{rtt}\t{meta['rss_kb']}\t{pa}\t{alive}\n")

        if kind == "coarse":
            after, _ = perf(f"{model}_{run}_{kind}_idle_after")
            with open(tsv, "a", encoding="utf-8") as f:
                f.write(f"{model}\t{run}\t{kind}\t{target}\tidle_after\t" +
                        "\t".join(map(str, after)) + f"\t\t{meta['rss_kb']}\t{pa}\t-\n")

        for box in BOXES:
            unprog(box)
        vf.close()
        print(f"[{i+1}/{len(plan)}] {model} run{run} {kind} "
              f"idle={sum(idle)} infer={sum(vals)}", flush=True)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
