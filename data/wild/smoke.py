"""Smoke test (G1) against the approved endpoint. 9 inference requests, max_tokens=160.

Purpose, fixed before running (DECISION_platform.md §4):
  CV_len   = coefficient of variation of output_text_bytes across model medians
  CV_chunk = coefficient of variation of content_chunk_count across model medians

  CV_chunk > 0.15 and CV_len > 0.2  -> gateway passes chunks through; C1 usable
  CV_chunk < 0.05 and CV_len > 0.3  -> gateway repacks to fixed-size chunks; C1 dies

Also records what can only be learned from a live provider: whether `usage` is
returned on a stream, whether the response `model` echoes the requested id, and
whether the model emits `reasoning_content` (thinking mode).

No key is printed. Only hosts and derived numbers are written.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics as st
from pathlib import Path

import collector as C

KEY_DIR = Path(r"<data-root>/api key")
OUT = Path(__file__).resolve().parent / "out_smoke"

MODELS = [
    {"request_name": "qwen3.5-27b", "provider": "ali-native"},
    {"request_name": "qwen3.5-122b-a10b", "provider": "ali-native"},
    {"request_name": "qwen3.5-397b-a17b", "provider": "ali-native"},
]
PROMPTS = [
    {"id": "P2", "text": "List the integers 1 to 5, comma separated, nothing else."},
    {"id": "P4", "text": "Name the three primary colors, comma separated, no other text."},
    {"id": "P6", "text": "Summarise the plot of an invented film in about 60 words."},
]


def parse_key_file(p: Path):
    t = p.read_text(encoding="utf-8", errors="replace")
    key = next(l.strip() for l in t.splitlines() if l.strip().startswith("sk-"))
    url = re.search(r"https?://\S+", t).group(0)
    return key, url.rstrip("/")


def main() -> int:
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    key, base = parse_key_file(KEY_DIR / "ali.txt")
    url = base + "/chat/completions"
    print("endpoint host:", re.sub(r"https?://([^/]+).*", r"\1", base))
    print("models:", [m["request_name"] for m in MODELS], "prompts:", len(PROMPTS))

    rows = []
    for m in MODELS:
        for p in PROMPTS:
            plan = C.Plan(endpoint=url, models=[m], prompts=[p], rpm=600,
                          max_tokens=160, temperature=0.0, timeout_s=120)
            plan.extra_body = {"enable_thinking": False}
            r = C.probe(url, key, m["request_name"], p["text"], plan)
            note = ""
            if r["http_status"] >= 400:
                # 可能是 enable_thinking 不被支持：去掉重试一次
                plan.extra_body = {}
                r2 = C.probe(url, key, m["request_name"], p["text"], plan)
                note = f"retry_without_enable_thinking(status {r['http_status']}->{r2['http_status']})"
                r = r2
            r.update(session_id="SMOKE", round=0, seq_in_round=len(rows),
                     model_requested=m["request_name"], provider=m["provider"],
                     prompt_id=p["id"],
                     prompt_sha256=hashlib.sha256(p["text"].encode()).hexdigest()[:16])
            r["note"] = note
            rows.append(r)
            print(f"  {m['request_name']:22s} {p['id']}  status={r['http_status']:3d} "
                  f"chunks={r['content_chunk_count']:3d} out_b={r['output_text_bytes']:4d} "
                  f"reason={r['reasoning_chars']:4d} usage_ct={r['completion_tokens']} {note}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wild_full_SMOKE.csv", index=False, encoding="utf-8")

    ok = df[df.http_status == 200].copy()
    diag = {}
    if len(ok):
        med = ok.groupby("model_requested").agg(
            n=("output_text_bytes", "size"),
            out_bytes=("output_text_bytes", "median"),
            chunks=("content_chunk_count", "median"),
            reason=("reasoning_chars", "median"),
        )
        def cv(x):
            x = [v for v in x if v and v > 0]
            return (st.pstdev(x) / st.mean(x)) if len(x) > 1 and st.mean(x) else float("nan")
        diag = dict(
            cv_len=round(cv(med.out_bytes.tolist()), 4),
            cv_chunk=round(cv(med.chunks.tolist()), 4),
            usage_rate=round(float((ok.completion_tokens != "").mean()), 3),
            model_echo_rate=round(float((ok.model_served != "").mean()), 3),
            reasoning_rate=round(float((ok.reasoning_chars > 0).mean()), 3),
            done_rate=round(float(ok.done_seen.astype(str).str.lower().eq("true").mean()), 3),
            truncated_rate=round(float(ok.truncated.astype(str).str.lower().eq("true").mean()), 3),
        )
        print("\nper-model medians:\n", med.to_string())
        print("\ndiagnostics:", json.dumps(diag, ensure_ascii=False))

        if diag["cv_chunk"] > 0.15 and diag["cv_len"] > 0.2:
            verdict = "PASS — 网关透传，C1 可用，按本端点继续"
        elif diag["cv_chunk"] < 0.05 and diag["cv_len"] > 0.3:
            verdict = "FAIL-GATEWAY — 疑定长重打包，主平台需换方舟/硅基流动"
        else:
            verdict = "INCONCLUSIVE — 输出被强约束或样本太少，需改 prompt/加样本后再判"
    else:
        verdict = "NO-SUCCESS — 冒烟无成功响应，按 G1 停止"
        diag = dict(cv_len=None, cv_chunk=None)

    print("\nVERDICT:", verdict)

    md = ["# 冒烟结果（G1）", "",
          f"- 端点 host：`{re.sub(r'https?://([^/]+).*', r'\\1', base)}`（**未记录完整 URL**）",
          f"- 请求：{len(df)} 条（{len(MODELS)} 模型 × {len(PROMPTS)} prompt），max_tokens=160，temp=0，stream",
          f"- 成功：{len(ok)}/{len(df)}", "",
          "## 逐请求", "",
          "| 模型 | prompt | status | 内容块 | 输出字节 | 推理字符 | completion_tokens | 备注 |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['model_requested']} | {r['prompt_id']} | {r['http_status']} | "
                  f"{r['content_chunk_count']} | {r['output_text_bytes']} | {r['reasoning_chars']} | "
                  f"{r['completion_tokens']} | {r.get('note','')} |")
    md += ["", "## 诊断量（先于结果固定）", ""]
    for k, v in diag.items():
        md.append(f"- `{k}` = {v}")
    md += ["", "## 判定", "", f"**{verdict}**", "",
           "判据见 `DECISION_platform.md` §4：`CV_chunk>0.15 且 CV_len>0.2` 为通过；"
           "`CV_chunk<0.05 且 CV_len>0.3` 为网关重打包。", ""]
    (Path(__file__).resolve().parent / "SMOKE_RESULT.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "SMOKE_DIAG.json").write_text(
        json.dumps(dict(diag=diag, verdict=verdict), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote SMOKE_RESULT.md, out_smoke/SMOKE_DIAG.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
