"""S1 quality + G4 analysis. Reads one session CSV, writes tables and a report.

Two jobs:
  1. sample ledger: raw / valid / failed / undetermined, by model x prompt x round,
     with error classes. Nothing is dropped silently.
  2. G4 matched-length test, frozen before S1 was collected (STATUS_AND_S1_PLAN §4):
     among rows with identical (prompt_id, output_text_bytes), does content_chunk_count
     ever differ across models? Zero such groups across the whole session means the
     gateway locks chunking to byte length -> gateway_repacking = True.

No cross-session inference here: with one session there is nothing to transfer.
"""
from __future__ import annotations

import itertools
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def cv(xs):
    xs = [float(x) for x in xs]
    return st.pstdev(xs) / st.mean(xs) if len(xs) > 1 and st.mean(xs) else float("nan")


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append("%.3f" % v if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(csv_name: str):
    p = OUT / csv_name
    d = pd.read_csv(p)
    d["valid"] = (d.http_status == 200) & d.done_seen.astype(str).str.lower().eq("true") \
        & d.completion_tokens.notna()
    d["undetermined"] = (d.http_status == 200) & (~d["valid"])

    # ---------- ledger ----------
    led = (d.groupby(["model_requested", "http_status"], as_index=False)
             .size().rename(columns={"size": "n"}))
    per_model = []
    for m, g in d.groupby("model_requested"):
        per_model.append(dict(
            model=m, designed=len(g),
            http200=int((g.http_status == 200).sum()),
            valid=int(g.valid.sum()),
            undetermined=int(g.undetermined.sum()),
            failed=int(len(g) - (g.http_status == 200).sum()),
            rounds=int(g["round"].nunique()),
            prompts=int(g.prompt_id.nunique()),
            usage=int(g.completion_tokens.notna().sum()),
            errors=";".join(sorted({str(x) for x in g.error_class.dropna() if str(x)})) or "-",
        ))
    led_m = pd.DataFrame(per_model)
    led_m.to_csv(OUT / "TABLE_s1_ledger.csv", index=False, encoding="utf-8")
    led.to_csv(OUT / "TABLE_s1_ledger_cells.csv", index=False, encoding="utf-8")

    # ---------- per model x prompt medians ----------
    ok = d[d.valid].copy()
    feat = []
    for (m, pr), g in ok.groupby(["model_requested", "prompt_id"]):
        b = g.output_text_bytes.tolist()
        c = g.content_chunk_count.tolist()
        feat.append(dict(
            model=m, prompt=pr, n=len(g),
            bytes_med=float(pd.Series(b).median()),
            chunks_med=float(pd.Series(c).median()),
            bytes_per_chunk=round(float(pd.Series(b).median()) /
                                  max(1e-9, float(pd.Series(c).median())), 2),
            bytes_iqr=round(float(pd.Series(b).quantile(.75) - pd.Series(b).quantile(.25)), 1),
            chunks_min=int(min(c)) if c else 0, chunks_max=int(max(c)) if c else 0,
            ttft_content_med=round(float(pd.to_numeric(g.ttft_content_ms, errors="coerce").median()), 1),
            total_med=round(float(pd.to_numeric(g.total_ms, errors="coerce").median()), 1),
        ))
    feat_d = pd.DataFrame(feat).sort_values(["prompt", "model"])
    feat_d.to_csv(OUT / "TABLE_s1_features.csv", index=False, encoding="utf-8")

    # ---------- G4 matched-length test ----------
    g4, matched_total, groups_total = [], 0, 0
    for pr, g in ok.groupby("prompt_id"):
        vals = list(zip(g.output_text_bytes, g.content_chunk_count, g.model_requested))
        pairs = [(a, b) for a, b in itertools.combinations(vals, 2) if a[0] == b[0]]
        differ = [(a, b) for a, b in pairs if a[1] != b[1]]
        groups_total += len(pairs)
        matched_total += len(differ)
        g4.append(dict(prompt=pr, equal_length_pairs=len(pairs), chunk_differs=len(differ),
                       example=(f"{differ[0][0][1]} vs {differ[0][1][1]} chunks at "
                                f"{differ[0][0][0]} bytes" if differ else "-")))
    g4_d = pd.DataFrame(g4)
    g4_d.to_csv(OUT / "TABLE_s1_g4.csv", index=False, encoding="utf-8")

    # 同字节多块数：另做一个更严的版本，只看同一轮同 prompt（消除时间因素）
    per_round = []
    for (rd, pr), g in ok.groupby(["round", "prompt_id"]):
        vals = list(zip(g.output_text_bytes, g.content_chunk_count))
        if len(vals) < 2:
            continue
        eq = [(a, b) for a, b in itertools.combinations(vals, 2) if a[0] == b[0]]
        per_round.append(dict(round=int(rd), prompt=pr, equal_pairs=len(eq),
                              differs=sum(1 for a, b in eq if a[1] != b[1])))
    pr_d = pd.DataFrame(per_round)

    verdict = "PASS" if matched_total > 0 else ("FAIL-GATEWAY" if groups_total > 0 else "INCONCLUSIVE")

    summary = dict(
        csv=str(p), rows=int(len(d)),
        designed=int(len(d)), valid=int(d.valid.sum()), undetermined=int(d.undetermined.sum()),
        failed=int(len(d) - (d.http_status == 200).sum()),
        g4_equal_length_pairs=groups_total, g4_chunk_differs=matched_total,
        g4_verdict=verdict,
        g4_same_round=dict(pairs=int(pr_d.equal_pairs.sum()) if len(pr_d) else 0,
                           differs=int(pr_d.differs.sum()) if len(pr_d) else 0),
        usage_rate=round(float(d.completion_tokens.notna().mean()), 3),
        trunc_rate=round(float(d.truncated.astype(str).str.lower().eq("true").mean()), 3),
        reason_rate=round(float((pd.to_numeric(d.reasoning_chars, errors="coerce").fillna(0) > 0).mean()), 3),
    )
    (OUT / "S1_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    md = ["# S1 质量报告与 G4 判定", "",
          "- 行数 %d｜design 408｜valid **%d**｜undetermined %d｜failed %d" % (
              summary["rows"], summary["valid"], summary["undetermined"], summary["failed"]),
          "- usage 回传率 %.3f｜截断率 %.3f｜推理内容出现率 %.3f" % (
              summary["usage_rate"], summary["trunc_rate"], summary["reason_rate"]),
          "", "## 逐模型账本", "",
          md_table(led_m), "",
          "## 逐模型 x prompt 中位数", "",
          md_table(feat_d), "",
          "## G4 等长检验（跨全会话）", "",
          "- 等长对总数 **%d**，其中块数不同 **%d**" % (groups_total, matched_total),
          "- 同轮同 prompt 内：等长对 %d，块数不同 %d" % (
              summary["g4_same_round"]["pairs"], summary["g4_same_round"]["differs"]),
          "", "**判定：%s**" % verdict, ""]
    if verdict == "FAIL-GATEWAY":
        md.append("⇒ 判 `gateway_repacking = true`，停止实验，写阴性结论。")
    (HERE / "S1_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "wild_full_S1-20261119T1330.csv"))
