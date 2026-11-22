"""Generate the criterion figure entirely from RemoteID v3 output files."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(r"E:/111-workspace/workbuddy/2026-11-12-16-03-14/sgx-review")
IN = ROOT / "results/framework"
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(IN / "criterion_features.csv")
pred = pd.read_csv(IN / "criterion_prospective.csv")
d = features.merge(pred[["leg", "a_pred", "abs_err"]], on="leg", validate="one_to_one")


def fleet(leg: str) -> str:
    if leg.startswith("cal_"):
        return "cal"
    if leg.startswith("tdx_"):
        return "tdx"
    return "real"


d["fleet"] = d.leg.map(fleet)
marks = {"cal": "o", "real": "s", "tdx": "^"}
fig, (ax, ax_res) = plt.subplots(1, 2, figsize=(7.0, 2.8))
for group, marker in marks.items():
    part = d[d.fleet == group]
    ax.scatter(part.r, part.a_obs, marker=marker, s=30, facecolors="none",
               edgecolors="black", linewidths=1.1, label=group)
    for row in part.itertuples():
        ax.annotate(row.leg, (row.r, row.a_obs), textcoords="offset points",
                    xytext=(3, 3), fontsize=5)
ax.axhline(1 / 9, ls=":", lw=0.8, c="0.3")
ax.axhline(1 / 7, ls="--", lw=0.8, c="0.3")
ax.set_xscale("log")
ax.set_xlabel("model-time span / path jitter (gap/$\\sigma$)")
ax.set_ylabel("nearest-centroid single-shot accuracy")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=6)
ax_res.barh(d.leg, d.a_obs - d.a_pred, color="white", edgecolor="black")
ax_res.axvline(0, color="black", lw=0.8)
ax_res.set_xlabel("observed $-$ held-out prediction")
ax_res.tick_params(axis="y", labelsize=6)
for axis in (ax, ax_res):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
fig.suptitle(f"Leave-one-leg-out MAE = {d.abs_err.mean():.3f}", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_criterion_nearest_centroid.pdf")
fig.savefig(OUT / "fig_criterion_nearest_centroid.png", dpi=200)
print(f"wrote {OUT}")
