"""Offline analysis of the saved GATE-2 run (12 rows). No requests are made.

Reports, in this order:
  1. the FORMAL verdict under AMENDMENT_2 as approved (no re-interpretation);
  2. why that criterion misfires, stated without leaning on the result;
  3. post-hoc descriptives, explicitly labelled post-hoc and not usable as a gate.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "out_gate" / "wild_full_GATE2.csv"


def cv(xs):
    xs = [x for x in xs if x]
    return st.pstdev(xs) / st.mean(xs) if len(xs) > 1 and st.mean(xs) else float("nan")


def main():
    d = pd.read_csv(CSV)
    ok = d[d.http_status == 200].copy()
    failed = d[d.http_status != 200].copy()

    per, cv_len_p, cv_chunk_p = [], [], []
    for p in sorted(ok.prompt_id.unique()):
        sub = ok[ok.prompt_id == p]
        a = cv(sub.output_text_bytes.tolist())
        b = cv(sub.content_chunk_count.tolist())
        cv_len_p.append(a)
        cv_chunk_p.append(b)
        per.append(dict(
            prompt=p,
            bytes=[int(x) for x in sub.output_text_bytes],
            chunks=[int(x) for x in sub.content_chunk_count],
            bytes_per_chunk=[round(x / y, 2) for x, y in
                             zip(sub.output_text_bytes, sub.content_chunk_count)],
            models=list(sub.model_requested),
            cv_len=round(a, 4), cv_chunk=round(b, 4)))

    gate_len = round(st.median(cv_len_p), 4)
    gate_chunk = round(st.median(cv_chunk_p), 4)

    if gate_chunk >= 0.10 and gate_len >= 0.05:
        verdict = "PASS"
    elif gate_chunk < 0.05 and gate_len > 0.2:
        verdict = "FAIL-GATEWAY"
    else:
        verdict = "INCONCLUSIVE"

    within = []
    for row in per:
        within.append(dict(
            prompt=row["prompt"],
            cv_bytes_per_chunk=round(cv(row["bytes_per_chunk"]), 4),
            chunk_ratio=round(max(row["chunks"]) / min(row["chunks"]), 2),
            bytes_ratio=round(max(row["bytes"]) / max(1, min(row["bytes"])), 2)))

    res = dict(
        n=int(len(d)), ok=int(len(ok)), failed=int(len(failed)),
        gate_len=gate_len, gate_chunk=gate_chunk, verdict_formal=verdict,
        per_prompt=per, posthoc_within=within,
        usage_rate=round(float(ok.usage_available.astype(bool).mean()), 3),
        usage_by_model=ok.groupby("model_requested")["usage_available"]
        .apply(lambda s: float(s.astype(bool).mean())).round(3).to_dict(),
        failed_rows=failed[["model_requested", "prompt_id", "http_status"]].to_dict("records"),
    )
    (HERE / "out_gate" / "GATE2_ANALYSIS.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    md = [
        "# GATE-2 复验结果", "",
        "- 12 请求（4 槽 × 3 prompt）：**%d 成功 / %d 失败**" % (len(ok), len(failed)),
        "- 失败行：`vanchin/deepseek-v4-pro` × 3，全部 **HTTP 400**"
        "（三种参数变体都失败，含完全不加额外参数的变体）",
        "",
        "## 1 正式判定（严格按 AMENDMENT_2 执行，未改口径）", "",
        "| 量 | 值 | 阈值 | 是否满足 |", "|---|---|---|---|",
        "| `gate_chunk` | **%s** | ≥ 0.10 | %s |" % (gate_chunk, "是" if gate_chunk >= 0.10 else "否"),
        "| `gate_len` | **%s** | ≥ 0.05 | %s |" % (gate_len, "是" if gate_len >= 0.05 else "否"),
        "",
        "**判定：`%s`** —— 按已批准的判据，`gate_len` 不达标，不得宣布通过。" % verdict,
        "",
        "## 2 逐 prompt 实测", "",
        "| prompt | 输出字节 | 块数 | 每块文本字节 | CV_len | CV_chunk |",
        "|---|---|---|---|---|---|",
    ]
    for r in per:
        md.append("| %s | %s | %s | %s | %s | %s |" % (
            r["prompt"], r["bytes"], r["chunks"], r["bytes_per_chunk"],
            r["cv_len"], r["cv_chunk"]))

    md += [
        "",
        "## 3 判据本身的问题（不是数据的问题）", "",
        "`gate_len ≥ 0.05` 这条约束在逻辑上是反的。它的本意是「输出长度必须有变化，"
        "才能判断块数是否跟着长度走」；但恰恰是**长度不变而块数变化**，"
        "才是「分块不由字节决定」的直接证据。",
        "",
        "本批最干净的一组正是 P6：三个模型输出长度几乎相同（CV 2%），"
        "块数却是 18 / 28 / 38（相差 2.1 倍）。也就是说，"
        "**判据把最有信息量的那组数据当成了不合格条件**。",
        "",
        "我要说清楚：这是我在写 AMENDMENT_2 时的逻辑错误，不是事后为了通过而改口。"
        "上述观测只作为**事后描述**，不构成 G4 的通过结论。",
        "",
        "## 4 事后描述（不入判定）", "",
        "| prompt | 块数最大/最小 | 字节最大/最小 | 每块字节 CV |", "|---|---|---|---|",
    ]
    for w in within:
        md.append("| %s | %s× | %s× | %s |" % (
            w["prompt"], w["chunk_ratio"], w["bytes_ratio"], w["cv_bytes_per_chunk"]))

    md += [
        "",
        "若网关按固定字节 k 重打包，则「每块文本字节」应在所有行上恒等于 k、"
        "且块数应与字节数成正比。实测每块文本字节在 **1.8–19.1** 之间跨 10 倍，"
        "块数比与字节比明显脱钩（P6：块数 2.1× 而字节 1.03×）。",
        "⇒ 描述性上支持「分块边界由模型决定」，但**须按修订判据在新请求上正式复验**。",
        "",
        "## 5 usage：上一轮的结论被推翻", "",
        "本轮 `usage` 回传率 **%s（9/9）**，按模型 %s。" % (res["usage_rate"], res["usage_by_model"]),
        "上一轮 `qwen3.5-27b` 缺 usage，我当时判定为「模型属性」，**这个判断是错的**："
        "本轮加了 `stream_options.include_usage=true` 后该模型正常回传。",
        "⇒ usage 缺失是**请求参数**问题，不是模型属性。AMENDMENT_2 §3 的表述需据此更正。",
        "",
        "## 6 D2：列出的模型不等于可用", "",
        "`vanchin/deepseek-v4-pro` 在 `/models` 列表中，但三种参数变体"
        "（含完全不加额外参数）都是 400。",
        "⇒ 目录 ≠ 可用；D2 预检**未完成**，需换配对："
        "`deepseek-v4-pro-0813` vs `vanchin/deepseek-v4-pro-0813`，"
        "或 `siliconflow/deepseek-v3.1-terminus` vs `vanchin/deepseek-v3.1-terminus`。",
        "按 §3.1 记 `cross_provider: not_executed`，**不得用 400 的空数据冒充「provider 无差异」**。",
        "",
    ]
    (HERE / "GATE2_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("formal verdict:", verdict, "| gate_len", gate_len, "| gate_chunk", gate_chunk)
    print("usage:", res["usage_rate"], res["usage_by_model"])
    for r in per:
        print(r["prompt"], r["bytes"], r["chunks"], r["bytes_per_chunk"],
              "cv_len", r["cv_len"], "cv_chunk", r["cv_chunk"])


if __name__ == "__main__":
    main()
