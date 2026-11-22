"""Plot bidirectional cross-session RF and RemoteID results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(r"E:/111-workspace/workbuddy/2026-11-12-16-03-14/sgx-review")
IN = ROOT / "results/cross_session"
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)

rf = pd.read_csv(IN / "cross_session_rf.csv")
remote = pd.read_csv(IN / "cross_session_remoteid_budgets.csv")
rf = rf[rf.variant == "cross_session"]
remote = remote[remote.variant == "cross_session"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for direction, style in (("R3_to_R3rep", "o"), ("R3rep_to_R3", "s")):
    for mode, linestyle in (("sequential", "-"), ("random", "--")):
        d = rf[(rf.direction == direction) & (rf.window_mode == mode)]
        axes[0].plot(d.n, d.accuracy, marker=style, ms=3, lw=1, ls=linestyle,
                     label=f"{direction}, {mode}")
axes[0].set_xscale("log", base=2)
axes[0].axhline(1 / 7, color="0.3", ls=":", lw=0.9)
axes[0].set_xlabel("requests per window")
axes[0].set_ylabel("RF accuracy")
axes[0].set_ylim(0, 0.85)
axes[0].legend(fontsize=5, frameon=False)

for direction, marker in (("R3_to_R3rep", "o"), ("R3rep_to_R3", "s")):
    d = remote[remote.direction == direction]
    axes[1].plot(d.budget, d.decl_rate, marker=marker, ms=3, lw=1,
                 label=f"{direction}, declared")
    axes[1].plot(d.budget, d.correct_rate, marker=marker, ms=3, lw=1, ls="--",
                 label=f"{direction}, correct")
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("query budget")
axes[1].set_ylabel("fraction of runs")
axes[1].set_ylim(0, 1.02)
axes[1].legend(fontsize=5, frameon=False)
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_cross_session.pdf")
fig.savefig(OUT / "fig_cross_session.png", dpi=200)
print(f"wrote {OUT}")
