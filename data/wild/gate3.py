"""GATE-3 under AMENDMENT_3: matched-length counting test + provider-prefix probe.

11 requests total:
  9 = 3 known-good models x 3 NEW prompts (P1/P3/P5, not used by GATE-2)
  2 = one request each to siliconflow/ and vanchin/ DeepSeek terminus, to learn
      which provider prefixes are actually servable (vanchin/deepseek-v4-pro was 400)

Verdict rule (pre-registered, counting, no tunable threshold):
  matched(p) = pairs with identical output_text_bytes but different content_chunk_count
  PASS          : matched >= 1 on at least 2 prompts
  FAIL-GATEWAY  : at least 2 prompts have equal-length pairs, but matched == 0 everywhere
  INCONCLUSIVE  : not enough equal-length pairs to decide
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import statistics as st
from pathlib import Path

import collector as C

KEY_DIR = Path(r"<data-root>/api key")
HERE = Path(__file__).resolve().parent
OUT = HERE / "out_gate3"

MODELS = ["qwen3.5-27b", "qwen3.5-122b-a10b", "deepseek-v4-pro"]
PROVIDER_PROBES = ["siliconflow/deepseek-v3.1-terminus", "vanchin/deepseek-v3.1-terminus"]
PROMPTS = [
    {"id": "P1", "text": "Reply with exactly one word: ok."},
    {"id": "P3", "text": "Answer with a single digit: 7*6=?"},
    {"id": "P5", "text": "Write exactly two sentences on why caching helps web servers."},
]
VARIANTS = [{"enable_thinking": False, "stream_options": {"include_usage": True}},
            {"enable_thinking": False}, {}]


def parse_key_file(p: Path):
    t = p.read_text(encoding="utf-8", errors="replace")
    key = next(l.strip() for l in t.splitlines() if l.strip().startswith("sk-"))
    url = re.search(r"https?://\S+", t).group(0)
    return key, url.rstrip("/")


def call(url, key, model, prompt, tag):
    last = None
    for i, extra in enumerate(VARIANTS):
        plan = C.Plan(endpoint=url, models=[{"request_name": model}], prompts=[prompt],
                      rpm=600, max_tokens=160, temperature=0.0, timeout_s=120)
        plan.extra_body = extra
        r = C.probe(url, key, model, prompt["text"], plan)
        last = r
        if r["http_status"] == 200:
            break
    last.update(model_requested=model, prompt_id=prompt["id"], kind=tag,
                usage_available=bool(last["completion_tokens"] != ""),
                prompt_sha256=hashlib.sha256(prompt["text"].encode()).hexdigest()[:16])
    return last


def main() -> int:
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    key, base = parse_key_file(KEY_DIR / "ali.txt")
    url = base + "/chat/completions"
    print("host:", re.sub(r"https?://([^/]+).*", r"\1", base))

    rows = []
    for m in MODELS:
        for p in PROMPTS:
            r = call(url, key, m, p, "gate")
            rows.append(r)
            print("  %-22s %s st=%s chunks=%2s bytes=%3s" % (
                m, p["id"], r["http_status"], r["content_chunk_count"], r["output_text_bytes"]))
    for m in PROVIDER_PROBES:
        r = call(url, key, m, PROMPTS[1], "provider_probe")
        rows.append(r)
        print("  %-34s probe st=%s" % (m, r["http_status"]))

    df = pd.DataFrame(rows)
    csv = OUT / "wild_full_GATE3.csv"
    df.to_csv(csv, index=False, encoding="utf-8")

    g = df[df.kind == "gate"]
    ok = g[g.http_status == 200]
    per, n_pass = [], 0
    for p in [x["id"] for x in PROMPTS]:
        sub = ok[ok.prompt_id == p]
        vals = list(zip(sub.output_text_bytes, sub.content_chunk_count, sub.model_requested))
        pairs = [a for a, b in itertools.combinations(vals, 2) if a[0] == b[0]]
        matched = [a for a, b in itertools.combinations(vals, 2)
                   if a[0] == b[0] and a[1] != b[1]]
        if matched:
            n_pass += 1
        per.append(dict(prompt=p,
                        bytes=[int(v[0]) for v in vals],
                        chunks=[int(v[1]) for v in vals],
                        models=[v[2] for v in vals],
                        equal_length_pairs=len(pairs),
                        matched_pairs=len(matched),
                        equal_pair_chunks=[(int(a[1]), int(b[1])) for a, b in
                                           itertools.combinations(vals, 2) if a[0] == b[0]]))

    have_pairs = sum(1 for x in per if x["equal_length_pairs"] > 0)
    if n_pass >= 2:
        verdict = "PASS"
    elif have_pairs >= 2 and n_pass == 0:
        verdict = "FAIL-GATEWAY"
    else:
        verdict = "INCONCLUSIVE"

    probes = df[df.kind == "provider_probe"][["model_requested", "http_status"]].to_dict("records")
    res = dict(n=int(len(df)), gate_ok=int(len(ok)), per_prompt=per, verdict=verdict,
               n_pass_prompts=n_pass, have_equal_pairs=have_pairs, provider_probes=probes,
               usage_rate=round(float(ok.usage_available.mean()), 3) if len(ok) else None)
    (OUT / "GATE3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# GATE-3 结果（AMENDMENT_3 等长检验）", "",
          "- 请求：%d（9 等长检验 + 2 provider 探测），等长检验成功 %d" % (len(df), len(ok)),
          "- 判据：输出字节相等而行块数不同 → 计数判据，无连续阈值", "",
          "## 逐 prompt", "",
          "| prompt | 输出字节 | 内容块数 | 等长对 | 其中块数不同 | 等长对的块数 |",
          "|---|---|---|---|---|---|"]
    for x in per:
        md.append("| %s | %s | %s | %d | **%d** | %s |" % (
            x["prompt"], x["bytes"], x["chunks"], x["equal_length_pairs"],
            x["matched_pairs"], x["equal_pair_chunks"]))
    md += ["", "## 判定", "", "**%s**" % verdict,
           "", "满足条件的 prompt 数：%d / %d；存在等长对的 prompt 数：%d" % (
               n_pass, len(per), have_pairs), "",
           "## provider 前缀探测", "", "| 模型 | HTTP |", "|---|---|"]
    for p in probes:
        md.append("| `%s` | %s |" % (p["model_requested"], p["http_status"]))
    md += ["", "（400 = 该前缀当前不可服务；不得记为 provider 无差异）", ""]
    (HERE / "GATE3_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\nper-prompt:", json.dumps(per, ensure_ascii=False))
    print("VERDICT:", verdict, "| n_pass", n_pass, "| have_pairs", have_pairs)
    print("probes:", probes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
