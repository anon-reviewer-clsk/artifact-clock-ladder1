"""Gate re-test under AMENDMENT_2 (12 requests: 4 model slots x 3 prompts).

Changes vs smoke.py, all per AMENDMENT_2:
  * gate statistic is now WITHIN-prompt, across models, then median over prompts
    (the old "median across prompts then CV" was degenerate by construction);
  * C1 drops `first_chunk_bytes` (it encoded the echoed model-name length);
  * `usage_available` is a first-class field; missing usage is a model property,
    never padded with zeros;
  * `stream_options.include_usage` is requested, because the metadata baseline
    needs usage wherever the provider is willing to give it.

Extra, pre-specified here (not post-hoc): the DeepSeek twin pair is scored against
the cross-model spread. If the twin's distance is smaller than the typical
cross-model distance, chunk structure tracks the *model*, not the *provider*.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics as st
from pathlib import Path

import collector as C

KEY_DIR = Path(r"<data-root>/api key")
OUT = Path(__file__).resolve().parent / "out_gate"
HERE = Path(__file__).resolve().parent

MODELS = [
    {"request_name": "qwen3.5-27b", "provider": "ali-native", "pair": None},
    {"request_name": "qwen3.5-122b-a10b", "provider": "ali-native", "pair": None},
    {"request_name": "deepseek-v4-pro", "provider": "deepseek-native", "pair": "P"},
    {"request_name": "vanchin/deepseek-v4-pro", "provider": "vanchin", "pair": "P"},
]
PROMPTS = [
    {"id": "P2", "text": "List the integers 1 to 5, comma separated, nothing else."},
    {"id": "P4", "text": "Name the three primary colors, comma separated, no other text."},
    {"id": "P6", "text": "Summarise the plot of an invented film in about 60 words."},
]
VARIANTS = [
    {"enable_thinking": False, "stream_options": {"include_usage": True}},
    {"enable_thinking": False},
    {},
]


def parse_key_file(p: Path):
    t = p.read_text(encoding="utf-8", errors="replace")
    key = next(l.strip() for l in t.splitlines() if l.strip().startswith("sk-"))
    url = re.search(r"https?://\S+", t).group(0)
    return key, url.rstrip("/")


def cv(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return (st.pstdev(xs) / st.mean(xs)) if len(xs) > 1 and st.mean(xs) else float("nan")


def main() -> int:
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    key, base = parse_key_file(KEY_DIR / "ali.txt")
    url = base + "/chat/completions"
    print("host:", re.sub(r"https?://([^/]+).*", r"\1", base))

    rows = []
    for m in MODELS:
        for p in PROMPTS:
            chosen, note = None, ""
            for i, extra in enumerate(VARIANTS):
                plan = C.Plan(endpoint=url, models=[m], prompts=[p], rpm=600,
                              max_tokens=160, temperature=0.0, timeout_s=120)
                plan.extra_body = extra
                r = C.probe(url, key, m["request_name"], p["text"], plan)
                if r["http_status"] == 200:
                    chosen = (r, i, extra)
                    break
                note = f"variant{i}->{r['http_status']}"
            if chosen is None:
                r, vi, extra = r, len(VARIANTS) - 1, {}
            else:
                r, vi, extra = chosen
            r.update(session_id="GATE2", round=0, seq_in_round=len(rows),
                     model_requested=m["request_name"], provider=m["provider"],
                     prompt_id=p["id"],
                     prompt_sha256=hashlib.sha256(p["text"].encode()).hexdigest()[:16],
                     usage_available=bool(r["completion_tokens"] != ""),
                     param_variant=vi, note=note, pair=m["pair"] or "")
            rows.append(r)
            print(f"  {m['request_name']:24s} {p['id']} st={r['http_status']} "
                  f"chunks={r['content_chunk_count']:3d} bytes={r['output_text_bytes']:4d} "
                  f"reason={r['reasoning_chars']:4d} usage={r['usage_available']} v={vi}")

    df = pd.DataFrame(rows)
    csv_path = OUT / "wild_full_GATE2.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    ok = df[df.http_status == 200].copy()
    res = {"n": len(df), "ok": len(ok)}

    if len(ok) >= 6:
        per = ok.pivot_table(index="prompt_id", columns="model_requested",
                             values=["output_text_bytes", "content_chunk_count"], aggfunc="first")
        cv_len_p, cv_chunk_p, detail = [], [], []
        for p in ok.prompt_id.unique():
            sub = ok[ok.prompt_id == p]
            a, b = cv(sub.output_text_bytes.tolist()), cv(sub.content_chunk_count.tolist())
            cv_len_p.append(a)
            cv_chunk_p.append(b)
            detail.append(dict(prompt=p, cv_len=round(a, 4), cv_chunk=round(b, 4),
                               bytes=sub.output_text_bytes.tolist(),
                               chunks=sub.content_chunk_count.tolist()))
        gate_len = round(st.median(cv_len_p), 4)
        gate_chunk = round(st.median(cv_chunk_p), 4)

        # 同规模敏感性：只用 3 个不同模型（排除 D2 孪生）
        distinct = ok[ok["pair"] != "P"]
        gate_chunk3 = round(st.median([cv(distinct[distinct.prompt_id == p].content_chunk_count.tolist())
                                       for p in distinct.prompt_id.unique()]), 4)

        # 孪生距离 vs 跨模型距离
        twin = []
        for p in ok.prompt_id.unique():
            sub = ok[ok.prompt_id == p]
            a = sub[sub.model_requested == "deepseek-v4-pro"].content_chunk_count.iloc[0]
            b = sub[sub.model_requested == "vanchin/deepseek-v4-pro"].content_chunk_count.iloc[0]
            others = sub[~sub.model_requested.str.contains("deepseek")].content_chunk_count.tolist()
            cross = st.mean([abs(a - o) for o in others] + [abs(b - o) for o in others]) if others else float("nan")
            twin.append(dict(prompt=p, twin_delta=abs(a - b), cross_mean_delta=round(cross, 2)))

        if gate_chunk >= 0.10 and gate_len >= 0.05:
            verdict = "PASS — 分块边界未被字节锁定，C1 可用"
        elif gate_chunk < 0.05 and gate_len > 0.2:
            verdict = "FAIL-GATEWAY — 定长重打包"
        else:
            verdict = "INCONCLUSIVE"

        res.update(gate_len=gate_len, gate_chunk=gate_chunk, gate_chunk_distinct3=gate_chunk3,
                   per_prompt=detail, twin=twin, verdict=verdict,
                   usage_rate=round(float(ok.usage_available.mean()), 3),
                   usage_by_model=ok.groupby("model_requested").usage_available.mean().round(3).to_dict(),
                   reasoning_rate=round(float((ok.reasoning_chars > 0).mean()), 3),
                   truncated_rate=round(float(ok.truncated.astype(str).str.lower().eq("true").mean()), 3))
        print("\nper-prompt:", json.dumps(detail, ensure_ascii=False))
        print("gate_len=%.4f gate_chunk=%.4f (distinct3=%.4f)" % (gate_len, gate_chunk, gate_chunk3))
        print("twin:", json.dumps(twin, ensure_ascii=False))
        print("VERDICT:", verdict)
    else:
        res["verdict"] = "NO-DATA — 成功不足 6 条，不判定"
        print("VERDICT:", res["verdict"])

    (OUT / "GATE2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# GATE-2 复验结果（AMENDMENT_2 已批准的判据）", "",
          f"- 请求：{res['n']} 条（4 模型槽 × 3 prompt），成功 {res['ok']}",
          "- 判据：同 prompt 内跨模型 CV，取各 prompt 中位数（`AMENDMENT_2` §1）", ""]
    if "gate_chunk" in res:
        md += ["## 判据量", "",
               "| 量 | 值 | 阈值 |", "|---|---|---|",
               f"| `gate_len` | {res['gate_len']} | ≥ 0.05 |",
               f"| `gate_chunk`（4 槽） | {res['gate_chunk']} | ≥ 0.10 |",
               f"| `gate_chunk`（3 个不同模型，排除孪生） | {res['gate_chunk_distinct3']} | 敏感性 |",
               "", "## 逐 prompt", "",
               "| prompt | 输出字节 | CV_len | 块数 | CV_chunk |", "|---|---|---|---|---|"]
        for d in res["per_prompt"]:
            md.append(f"| {d['prompt']} | {d['bytes']} | {d['cv_len']} | {d['chunks']} | {d['cv_chunk']} |")
        md += ["", "## D2 孪生距离 vs 跨模型距离", "",
               "| prompt | 孪生块数差 | 跨模型平均块数差 |", "|---|---|---|"]
        for t in res["twin"]:
            md.append(f"| {t['prompt']} | {t['twin_delta']} | {t['cross_mean_delta']} |")
        md += ["", f"`usage` 回传率 {res.get('usage_rate')}；按模型 {res.get('usage_by_model')}", "",
               "## 判定", "", f"**{res['verdict']}**"]
    else:
        md.append(f"**{res['verdict']}**")
    (HERE / "GATE2_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote GATE2_RESULT.md, out_gate/GATE2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
