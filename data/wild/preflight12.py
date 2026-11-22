"""Preflight for the 12-model design (B). 16 requests, one per candidate.

Selection rule, fixed BEFORE looking at results:
  usable := status 200  AND  content_chunk_count >= 1  AND  done_seen
  excluded(u) otherwise, with the reason recorded.

Then the 12 slots are filled in priority order:
  H1 hard trio   : qwen3.5-27b, qwen3.6-27b, qwen3.8-27b      (same size, 3 generations)
  H2 hard pair   : qwen3.5-35b-a3b, qwen3.6-35b-a3b           (same size + arch, 2 generations)
  S  scale       : qwen3.5-122b-a10b, qwen3.5-397b-a17b
  X  cross-family: deepseek-v4-pro, glm-5.3, kimi-k3, MiniMax-M2.5
  R  reserve     : qwen3.8-flash, deepseek-v4.1-flash, qwen3.6-flash, kimi-k2.6
Missing slots are filled from the reserve list in order; nothing is chosen on outcome.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import collector as C

KEY = Path(r"<data-root>/api key/ali.txt")
HERE = Path(__file__).resolve().parent
OUT = HERE / "out_preflight"

SLOTS = [
    ("H1", "qwen3.5-27b"), ("H1", "qwen3.6-27b"), ("H1", "qwen3.8-27b"),
    ("H2", "qwen3.5-35b-a3b"), ("H2", "qwen3.6-35b-a3b"),
    ("S", "qwen3.5-122b-a10b"), ("S", "qwen3.5-397b-a17b"),
    ("X", "deepseek-v4-pro"), ("X", "glm-5.3"), ("X", "kimi-k3"), ("X", "MiniMax-M2.5"),
    ("R", "qwen3.8-flash"), ("R", "deepseek-v4.1-flash"), ("R", "qwen3.6-flash"),
    ("R", "kimi-k2.6"),
]
PROMPT = "Answer with a single digit: 7*6=?"
VARIANTS = [{"enable_thinking": False, "stream_options": {"include_usage": True}},
            {"enable_thinking": False}, {}]


def main():
    import pandas as pd

    t = KEY.read_text(encoding="utf-8", errors="replace")
    key = next(l.strip() for l in t.splitlines() if l.strip().startswith("sk-"))
    url = re.search(r"https?://\S+", t).group(0).rstrip("/") + "/chat/completions"
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for slot, m in SLOTS:
        r, vi = None, -1
        for i, extra in enumerate(VARIANTS):
            plan = C.Plan(endpoint=url, models=[{"request_name": m}], prompts=[],
                          rpm=600, max_tokens=160, temperature=0.0, timeout_s=90)
            plan.extra_body = extra
            r = C.probe(url, key, m, PROMPT, plan)
            vi = i
            if r["http_status"] == 200:
                break
        usable = (r["http_status"] == 200 and r["content_chunk_count"] >= 1
                  and bool(r["done_seen"]))
        rows.append(dict(slot=slot, model=m, status=r["http_status"],
                         variant=vi, error=r["error_class"],
                         chunks=r["content_chunk_count"], out_bytes=r["output_text_bytes"],
                         usage=bool(r["completion_tokens"] != ""),
                         done=bool(r["done_seen"]), truncated=bool(r["truncated"]),
                         reasoning=r["reasoning_chars"], usable=usable))
        print("  %-2s %-24s st=%-3s v=%d chunks=%-2s bytes=%-4s done=%-5s trunc=%-5s usable=%s %s" % (
            slot, m, r["http_status"], vi, r["content_chunk_count"], r["output_text_bytes"],
            bool(r["done_seen"]), bool(r["truncated"]), usable, r["error_class"] or ""))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "preflight12.csv", index=False, encoding="utf-8")

    # fill 12 slots: H1,H2,S,X first (in listed order), then R
    order = {"H1": 0, "H2": 1, "S": 2, "X": 3, "R": 4}
    df["prio"] = df.slot.map(order)
    chosen = (df[df.usable].sort_values(["prio"]).head(12)
                .sort_values("prio", ascending=True, kind="stable"))
    picked = chosen.model.tolist()
    print("\nusable:", int(df.usable.sum()), "/", len(df))
    print("selected 12:", picked)

    json.dump(dict(selected=picked, rows=rows), open(OUT / "preflight12.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
