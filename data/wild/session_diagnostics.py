"""Per-prompt diagnostic identification for one session CSV.

This is a DIAGNOSTIC, not main evidence (PREREG §6.2): same-session splits cannot
substitute for cross-session transfer. Standardisation uses the whole session here,
which is why it stays diagnostic; the cross-session analysis fits scalers on the
training session only.

Features per PREREG §2 after AMENDMENT_2:
  C1 = [content_chunk_count, chunk_bytes_cv]
  C3 = [output_text_bytes]
  C2 = [ttft_content_ms, total_ms]
Null baselines: wall_ts, collection order, prompt_id (should be constant per cell).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

FEATS = {
    "C1_packet": ["content_chunk_count", "chunk_bytes_cv"],
    "C3_length": ["output_text_bytes"],
    "C2_timing": ["ttft_content_ms", "total_ms"],
    "null_wall_ts": ["wall_ts"],
    "null_order": ["seq_in_round"],
}


def loo_acc(X: np.ndarray, y: np.ndarray) -> float:
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    models = sorted(set(y))
    ok = 0
    for i in range(len(Z)):
        mask = np.ones(len(Z), bool)
        mask[i] = False
        cent = np.vstack([Z[mask & (y == m)].mean(0) for m in models])
        ok += models[int(np.argmin(((Z[i] - cent) ** 2).sum(1)))] == y[i]
    return ok / len(Z)


def main(csv_name: str):
    d = pd.read_csv(OUT / csv_name)
    d = d[d.http_status == 200].copy()
    rows = []
    for pr, g in d.groupby("prompt_id"):
        r = {"prompt": pr, "n": len(g), "models": g.model_requested.nunique()}
        for name, cols in FEATS.items():
            if not set(cols).issubset(g.columns):
                continue
            X = g[cols].astype(float).fillna(0).values
            r[name] = round(loo_acc(X, g.model_requested.values), 3)
        rows.append(r)
    t = pd.DataFrame(rows)
    out = OUT / "TABLE_session_diag.csv"
    t.to_csv(out, index=False, encoding="utf-8")
    print(t.to_string(index=False))
    print("\nchance = %.4f" % (1 / d.model_requested.nunique()))
    (OUT / "session_diag.json").write_text(
        json.dumps(dict(csv=csv_name, chance=round(1 / d.model_requested.nunique(), 4),
                        rows=t.to_dict("records")), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
