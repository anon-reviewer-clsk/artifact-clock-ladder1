"""Collect machine-readable key results and selected paper-value comparisons."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from freeze_inputs import ROOT

RESULTS = ROOT / "results"
FRAMEWORK = RESULTS / "framework"
CROSS = RESULTS / "cross_session"


def main() -> None:
    manifest = json.loads((RESULTS / "input_manifest.json").read_text(encoding="utf-8"))
    criterion = pd.read_csv(FRAMEWORK / "criterion_prospective.csv")
    core = pd.read_csv(FRAMEWORK / "sequential_error_control.csv")
    declarations = pd.read_csv(FRAMEWORK / "declaration_counts.csv")
    budgets = pd.read_csv(FRAMEWORK / "same_budget.csv")
    open_power = pd.read_csv(FRAMEWORK / "open_set_power.csv")
    cross_rf = pd.read_csv(CROSS / "cross_session_rf.csv")
    cross_remote = pd.read_csv(CROSS / "cross_session_remoteid_budgets.csv")
    snr = pd.read_csv(RESULTS / "sprint1/snr_predicted_n.csv")

    key = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_raw_rows": sum(x["raw_rows"] for x in manifest["inputs"]),
        "input_valid_rows": sum(x["valid_rows"] for x in manifest["inputs"]),
        "criterion_loo_mae": float(criterion.abs_err.mean()),
        "core": core.to_dict(orient="records"),
        "declaration_counts": declarations.to_dict(orient="records"),
        "cross_session_rf_n100": cross_rf[cross_rf.n == 100].to_dict(orient="records"),
        "cross_session_remoteid_8192": cross_remote[cross_remote.budget == 8192].to_dict(orient="records"),
        "open_set_power_2048": open_power[open_power.t == 2048].groupby("leg").power.mean().to_dict(),
        "snr_uf": snr[snr.stat == "uf_proportion"].to_dict(orient="records"),
    }
    tvla_path = RESULTS / "sprint1/tvla_all.csv"
    if tvla_path.exists():
        tvla = pd.read_csv(tvla_path)
        key["tvla_verdicts"] = [
            {"leg": leg, "feature": feature, "verdict": verdict}
            for (leg, feature), verdict in tvla.groupby(["leg", "feature"]).verdict.first().items()
        ]
    (RESULTS / "key_results.json").write_text(json.dumps(key, indent=2, default=str), encoding="utf-8")

    comparisons = []

    def add(metric, paper, repaired, note):
        comparisons.append({"metric": metric, "paper": paper, "repaired": repaired, "note": note})

    add("Table 7 LOO MAE", "0.14", f"{criterion.abs_err.mean():.6f}",
        "Nine legs including real_wan_rep; dynamically recomputed")
    for leg, paper_pairs in (("cal_loopback", "61/72"), ("cal_quant", "57/72"),
                             ("tdx_wan", "18/42"), ("tdx_loopback", "38/42")):
        repaired = core.loc[core.leg == leg, "resolvable_pairs"].iloc[0]
        add(f"Table 5 resolvable {leg}", paper_pairs, repaired,
            "Exact CI-disjoint definition plus split calibration")
    for leg, paper_power in (("cal_loopback", "0.42"), ("real_wan", "0.14"),
                             ("tdx_wan", "0.14"), ("tdx_loopback", "0.41")):
        repaired = open_power[(open_power.leg == leg) & (open_power.t == 2048)].power.mean()
        add(f"Open-set power@2048 {leg}", paper_power, f"{repaired:.3f}",
            "Correct 2048 checkpoint; unknown excluded from thresholds")
    for leg, paper_acc in (("real_wan", "0.867"), ("tdx_wan", "0.867")):
        value = budgets[(budgets.leg == leg) & (budgets.budget == 128)
                        & (budgets.method == "rf_window")].correct_rate.iloc[0]
        add(f"Matched-budget RF@128 {leg}", paper_acc, f"{value:.3f}",
            "Shared calibration/test rounds and collection-order windows")
    repaired_snr = snr[(snr.leg == "SGX_S4_int") & (snr.stat == "uf_proportion")].n_star_gauss.iloc[0]
    add("WAN uf Gaussian n*", "79", f"{repaired_snr:.3f}",
        "Two-sample and two-sided formula; bootstrap remains 500")
    rep_aobs = criterion.loc[criterion.leg == "real_wan_rep", "a_obs"].iloc[0]
    add("real_wan_rep nearest-centroid a_obs", "0.286", f"{rep_aobs:.3f}",
        "Uniform valid-row filtering and nine-leg core pipeline")
    pd.DataFrame(comparisons).to_csv(RESULTS / "paper_comparison.csv", index=False)

    code_manifest = []
    for path in sorted((ROOT / "code").glob("*.py")):
        code_manifest.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    run_manifest = {
        "generated_utc": key["generated_utc"],
        "input_manifest": "results/input_manifest.json",
        "tests": "tests/test_review.py",
        "code": code_manifest,
    }
    (RESULTS / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS / 'key_results.json'}")
    print(f"wrote {RESULTS / 'paper_comparison.csv'}")


if __name__ == "__main__":
    main()
