"""Hash and inventory read-only source inputs for the isolated review."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(r"E:/111-workspace/workbuddy/2026-11-12-16-03-14/sgx-review")
RESULTS = Path(r"<data-root>/Results")
CODE = Path(r"<data-root>/Analysis")

CSVS = [
    RESULTS / "timing_s2_interleaved.csv",
    RESULTS / "timing_s3_interleaved.csv",
    RESULTS / "timing_s4_interleaved.csv",
    RESULTS / "timing_wan_paired_dual.csv",
    RESULTS / "timing_s4_interleaved_rep.csv",
    RESULTS / "timing_plain_loopback_pos.csv",
    RESULTS / "timing_plain_loopback_pos_s7.csv",
    RESULTS / "timing_plain_loopback_sgx.csv",
    RESULTS / "timing_plain_loopback_sgx_s7.csv",
    RESULTS / "timing_plain_wan_pos.csv",
    RESULTS / "models_real/legs/timing_r2_loopback_pos_s20261110.csv",
    RESULTS / "models_real/legs/timing_r2_loopback_pos_s20261117.csv",
    RESULTS / "models_real/legs/timing_r2_loopback_sgx_s20261110.csv",
    RESULTS / "models_real/legs/timing_r2_loopback_sgx_s20261117.csv",
    RESULTS / "models_real/legs/timing_r3_wan_s20261111.csv",
    RESULTS / "models_real/legs/timing_wan_real_rep_s20261112.csv",
    RESULTS / "models_real/legs/timing_t1_tdx_wan_s20261111.csv",
    RESULTS / "models_real/legs/timing_t2_tdx_loopback_s20261111.csv",
]

SOURCES = [
    CODE / "models_real/framework_core.py",
    CODE / "models_real/framework_open_defense.py",
    CODE / "models_real/review_stats.py",
    CODE / "models_real/make_fig_criterion.py",
    CODE / "network_attack/collect_interleaved.py",
    CODE / "network_attack/test_nshot_confound_probe.py",
    CODE / "sprint1/tvla_suite.py",
    CODE / "sprint1/snr_complexity.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    records = []
    for path in CSVS:
        d = pd.read_csv(path)
        valid = d[(d["enclave_us"] > 0) & (d["roundtrip_us"] > 0)]
        records.append({
            "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "raw_rows": int(len(d)), "valid_rows": int(len(valid)),
            "models": sorted(d["model_id"].unique().tolist()),
            "scenarios": sorted(d["scenario"].unique().tolist()),
        })
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(RESULTS),
        "inputs": records,
        "source_code": [{"path": str(p), "sha256": sha256(p)} for p in SOURCES],
    }
    out = ROOT / "results/input_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(f"raw={sum(x['raw_rows'] for x in records)} valid={sum(x['valid_rows'] for x in records)}")


if __name__ == "__main__":
    main()
