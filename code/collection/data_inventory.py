"""Create raw/valid data inventories without altering source CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from freeze_inputs import CSVS, ROOT

OUT = ROOT / "results/inventory"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["scenario", "model_id", "rep", "sample_id"]


def main() -> None:
    legs, models = [], []
    for path in CSVS:
        d = pd.read_csv(path)
        d["__valid"] = (d.enclave_us > 0) & (d.roundtrip_us > 0)
        for scenario, s in d.groupby("scenario"):
            valid = s[s.__valid]
            rounds = int(s.rep.nunique())
            n_models = int(s.model_id.nunique())
            expected = rounds * 10 * n_models
            rt = valid.roundtrip_us.to_numpy(dtype=float) / 1000
            wall_start = wall_end = ""
            if "wall_ts" in s and s.wall_ts.notna().any():
                wall_start = str(pd.to_datetime(s.wall_ts.min(), unit="s", utc=True))
                wall_end = str(pd.to_datetime(s.wall_ts.max(), unit="s", utc=True))
            legs.append(dict(
                file=str(path), scenario=scenario, raw_n=len(s), valid_n=len(valid),
                bad_n=int((~s.__valid).sum()), missing_slots=expected - len(s),
                expected_slots=expected, models=n_models, rounds=rounds,
                duplicate_request_keys=int(s.duplicated(KEYS).sum()),
                exact_duplicate_rows=int(s.duplicated().sum()),
                wall_start_utc=wall_start, wall_end_utc=wall_end,
                rt_p50_ms=np.median(rt) if len(rt) else np.nan,
                rt_p95_ms=np.percentile(rt, 95) if len(rt) else np.nan,
                rt_p99_ms=np.percentile(rt, 99) if len(rt) else np.nan,
                rt_max_ms=np.max(rt) if len(rt) else np.nan,
            ))
            for model, g in s.groupby("model_id"):
                v = g[g.__valid]
                models.append(dict(
                    scenario=scenario, model_id=model, raw_n=len(g), valid_n=len(v),
                    bad_n=int((~g.__valid).sum()), rounds=int(g.rep.nunique()),
                    rt_p50_ms=v.roundtrip_us.median() / 1000 if len(v) else np.nan,
                    rt_p99_ms=v.roundtrip_us.quantile(0.99) / 1000 if len(v) else np.nan,
                    rt_max_ms=v.roundtrip_us.max() / 1000 if len(v) else np.nan,
                ))
    pd.DataFrame(legs).to_csv(OUT / "data_inventory_by_leg.csv", index=False)
    pd.DataFrame(models).to_csv(OUT / "data_inventory_by_model.csv", index=False)
    print(f"wrote {OUT}: {len(legs)} scenario legs, {len(models)} model cells")


if __name__ == "__main__":
    main()
