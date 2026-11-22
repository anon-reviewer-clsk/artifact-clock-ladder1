#!/usr/bin/env python3
"""stage2_driver.py — 阶段 2 数据采集：交错、以"推理运行"为独立单元。

与阶段 1 的差别：
  * 只采粗粒度（地址定向臂在阶段 1 已证明不稳定，不重复消耗）；
  * 每轮把 4 个模型**打乱顺序**各跑一次，共 8 轮 —— 消除模型与时间/热度的共线；
  * 每次运行记录四 box 分别的计数、RTT p50、RSS、输入 PA、模型大小；
  * 每段窗口固定 3.0 s。

采样点不是独立样本：独立单元是"一次推理运行"（每个 run 只有一行摘要）。
"""
from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from stage1_driver import BOXES, EV, ITERS, WINDOW, prog, unprog

MODELS = os.environ.get(
    "S2_MODELS",
    "dsv1_tok_v4.onnx,dsv1_tok_v4_l1.onnx,resnet18.onnx,dsv1_downstream.onnx").split(",")
ROUNDS = int(os.environ.get("S2_ROUNDS", "8"))
SEED = int(os.environ.get("S2_SEED", "20261120"))
OUT = Path(os.environ.get("S2_OUT", "/root/phase2"))


def parse_perf(txt):
    vals = []
    for k in range(4):
        m = re.search(rf"^\s*([\d,]+)\s+uncore_m2m_{k}/", txt, re.M)
        vals.append(int(m.group(1).replace(",", "")) if m else -1)
    return vals


def perf_coarse(tag):
    r = subprocess.run(["perf", "stat", "-e", EV, "-a", "--", "sleep", str(WINDOW)],
                       capture_output=True, text=True)
    txt = r.stdout + r.stderr
    (OUT / f"perf_{tag}.log").write_text(txt, encoding="utf-8")
    return parse_perf(txt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tsv = OUT / "stage2.tsv"
    tsv.write_text("round\torder_pos\tmodel\tsize_bytes\tbox0\tbox1\tbox2\tbox3\t"
                   "idle_before\trtt_p50_ms\trss_kb\tinput_pa\talive\n", encoding="utf-8")
    rng = random.Random(SEED)
    total = ROUNDS * len(MODELS)
    i = 0
    for rnd in range(1, ROUNDS + 1):
        order = MODELS[:]
        rng.shuffle(order)
        for pos, model in enumerate(order):
            i += 1
            vfile = OUT / f"v_r{rnd}_{model}.log"
            vf = open(vfile, "w")
            v = subprocess.Popen(
                ["numactl", "--membind=1", "--physcpubind=40",
                 "python3", "/root/phase1/ort_victim.py",
                 "--model", f"/root/models/{model}", "--iters", str(ITERS)],
                stdout=vf, stderr=subprocess.STDOUT)
            meta = None
            for _ in range(300):
                time.sleep(0.1)
                txt = vfile.read_text(errors="ignore") if vfile.exists() else ""
                mm = re.search(r"^METADATA (\{.*\})$", txt, re.M)
                if mm:
                    meta = json.loads(mm.group(1))
                if "READY" in txt:
                    break
            if not meta:
                print(f"[skip] {model} r{rnd}: no METADATA", flush=True)
                v.kill(); vf.close(); continue

            for box in BOXES:
                prog(box, coarse=True)
            idle = sum(perf_coarse(f"r{rnd}_{model}_idle"))

            p = subprocess.Popen(["perf", "stat", "-e", EV, "-a", "--", "sleep", str(WINDOW)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            time.sleep(0.3)
            v.send_signal(signal.SIGUSR1)
            pout = p.communicate()[0]
            (OUT / f"perf_r{rnd}_{model}_infer.log").write_text(pout, encoding="utf-8")
            vals = parse_perf(pout)
            try:
                v.wait(timeout=5); alive = 0
            except subprocess.TimeoutExpired:
                v.kill(); alive = 1
            txt = vfile.read_text(errors="ignore")
            mr = re.search(r"^RESULT (\{.*\})$", txt, re.M)
            rtt = json.loads(mr.group(1))["p50_ms"] if mr else ""
            with open(tsv, "a", encoding="utf-8") as f:
                f.write(f"{rnd}\t{pos}\t{model}\t{meta['size_bytes']}\t" +
                        "\t".join(map(str, vals)) +
                        f"\t{idle}\t{rtt}\t{meta['rss_kb']}\t{meta['input_pa']}\t{alive}\n")
            for box in BOXES:
                unprog(box)
            vf.close()
            print(f"[{i}/{total}] r{rnd} {model} idle={idle} infer={sum(vals)} rtt={rtt}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
