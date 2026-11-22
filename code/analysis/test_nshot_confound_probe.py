"""Is the n-shot recovery curve model timing, or per-collection-block transport state?

attack_nshot_per_scenario.py scores each window with 10 distributional features
(mean, std, median, min, max, p25, p75, IQR, skew, kurtosis). Per-model within-model
dispersion differs by 2-3x, so fast recovery at n~50 is plausibly driven by
heteroscedasticity rather than by the mean. That raises a validity question: the
original data was collected in contiguous per-model blocks, so the transport/OS state
during a block is aliased with model_id, and dispersion is exactly the statistic that
state leaks into.

This reruns the identical windowing + feature + scoring pipeline on three columns:

  enclave_us   : trusted in-enclave work time. Upper bound on what model compute
                 alone can deliver, and only observable to a verifier that reads
                 the clock inside the TCB.
  roundtrip_us : what the published curve uses (client-observed end-to-end).
  overhead_us  : roundtrip_us - enclave_us. The confound probe.

  NOTE on overhead_us: it is NOT "free of model compute by construction". Enclave
  work time contributes to the end-to-end time and the subtraction is imperfect
  (within-model corr(overhead, enclave) = +0.25 for S2, +0.15 for S3), so a
  nonzero overhead-only accuracy can still be partly real model signal. Read the
  number against its own permutation null, not against 1/9.

VERDICT RULE (revised after the mock validation, see test_interleaved_mock.py):
  chance level 1/9 is NOT the right floor for a 10-feature Random Forest on a few
  hundred windows -- RF overfits and its null is measurably above 1/9. Each cell
  therefore gets its own label-permutation null (PERM permutations, same windows,
  same split), and the reported statistic is z = (acc - null_mean) / null_sd.

  original contiguous run : z(overhead) large  => block-state artifact confirmed
  interleaved re-run      : z(overhead) ~ 0 and z(enclave) large => the ladder is
                           real, and the artifact was the collection order.

Usage:
  python test_nshot_confound_probe.py                      # published data only
  python test_nshot_confound_probe.py \
      --inputs <data-root>/Results/timing_all_scenarios.csv,<data-root>/Results/timing_s2_interleaved.csv \
      --scenarios S2_loopback,S2_interleaved --out .../nshot_confound_paired.csv

  --per-model N   cap every scenario at N rows/model. Without it the block curve
      (4000/model) and the interleaved curve (990/model) use different numbers of
      windows at every n, so a delta could be window count rather than layout.
      With it, both layouts see identical n and identical window counts per cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DEFAULT_INPUTS = [Path(r"<data-root>/Results/timing_all_scenarios.csv")]
DEFAULT_OUT = Path(r"E:/111-workspace/workbuddy/2026-11-12-16-03-14/sgx-review/results/network_attack/nshot_confound_probe.csv")
DEFAULT_SCENARIOS = ["S2_loopback", "S3_LAN_simulated", "S4_WAN_real"]

MODEL_ORDER = ["cfc_h16", "cfc_h32", "cfc_h64",
               "lstm_h16", "lstm_h32", "lstm_h64",
               "mlp_h16", "mlp_h32", "mlp_h64"]
SEED = 42
MIN_WINDOWS_PER_MODEL = 3


def downsample_per_model(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """Exactly n rows per model, write order preserved (see test_order_confound)."""
    keep = []
    rng = np.random.default_rng(SEED)
    for _, g in d.groupby("model_id"):
        if len(g) <= n:
            keep.append(g)
        else:
            idx = rng.choice(len(g), size=n, replace=False)
            keep.append(g.iloc[np.sort(idx)])
    return pd.concat(keep).sort_index()


def _feats(win: np.ndarray) -> list[float]:
    """Finite features for singleton and constant windows."""
    win = np.asarray(win, dtype=float)
    if win.size == 0:
        raise ValueError("empty latency window")
    sd = float(np.std(win, ddof=1)) if win.size > 1 else 0.0
    constant = win.size == 1 or bool(np.all(win == win[0]))
    skew = 0.0 if constant else float(sps.skew(win))
    kurtosis = 0.0 if constant else float(sps.kurtosis(win))
    return [float(np.mean(win)), sd, float(np.median(win)),
            float(np.min(win)), float(np.max(win)),
            float(np.percentile(win, 25)), float(np.percentile(win, 75)),
            float(np.percentile(win, 75) - np.percentile(win, 25)), skew, kurtosis]


def _windows(lat_by_model: dict[str, np.ndarray], n: int,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X_rows, y = [], []
    label_to_int = {m: i for i, m in enumerate(MODEL_ORDER)}
    for mid, lat in lat_by_model.items():
        idx = np.arange(len(lat)); rng.shuffle(idx)
        lat = lat[idx]
        nwin = len(lat) // n
        for w in range(nwin):
            X_rows.append(_feats(lat[w * n:(w + 1) * n]))
            y.append(label_to_int[mid])
    return np.array(X_rows, dtype=float), np.array(y, dtype=int)


def sequential_windows(d: pd.DataFrame, models: list[str], n: int,
                       value_col: str = "roundtrip_us"):
    """Non-overlapping collection-order windows for independent train/test legs."""
    X_rows, y = [], []
    label_to_int = {m: i for i, m in enumerate(models)}
    for model in models:
        values = d.loc[d.model_id == model, value_col].to_numpy(dtype=float)
        for start in range(0, len(values) - n + 1, n):
            X_rows.append(_feats(values[start:start + n]))
            y.append(label_to_int[model])
    return np.asarray(X_rows, dtype=float), np.asarray(y, dtype=int)


def random_windows(d: pd.DataFrame, models: list[str], n: int, seed: int,
                   value_col: str = "roundtrip_us"):
    """Shuffled non-overlapping windows, with an explicit model list."""
    rng = np.random.default_rng(seed)
    X_rows, y = [], []
    label_to_int = {m: i for i, m in enumerate(models)}
    for model in models:
        values = d.loc[d.model_id == model, value_col].to_numpy(dtype=float, copy=True)
        rng.shuffle(values)
        for start in range(0, len(values) - n + 1, n):
            X_rows.append(_feats(values[start:start + n]))
            y.append(label_to_int[model])
    return np.asarray(X_rows, dtype=float), np.asarray(y, dtype=int)


def rf_train_test(train: pd.DataFrame, test: pd.DataFrame, models: list[str], n: int,
                  value_col: str = "roundtrip_us", n_estimators: int = 300,
                  window_mode: str = "sequential"):
    if window_mode == "sequential":
        X_train, y_train = sequential_windows(train, models, n, value_col)
        X_test, y_test = sequential_windows(test, models, n, value_col)
    elif window_mode == "random":
        X_train, y_train = random_windows(train, models, n, 42, value_col)
        X_test, y_test = random_windows(test, models, n, 43, value_col)
    else:
        raise ValueError(window_mode)
    if len(X_train) == 0 or len(X_test) == 0:
        return float("nan"), np.array([]), y_test
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return float(accuracy_score(y_test, pred)), pred, y_test


def _split_indices(y: np.ndarray):
    indices = np.arange(len(y))
    return train_test_split(indices, test_size=0.3, random_state=SEED, stratify=y)


def _fit_score(X: np.ndarray, y: np.ndarray, n_estimators: int = 300,
               split=None) -> float:
    if len(X) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    tr, te = split if split is not None else _split_indices(y)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=SEED, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    return float(accuracy_score(y[te], clf.predict(X[te])))


def _null(X: np.ndarray, y: np.ndarray, perms: int) -> tuple[float, float]:
    """Permute labels while holding windows, features, and row split fixed."""
    rng = np.random.default_rng(SEED)
    split = _split_indices(y)
    vals = [_fit_score(X, rng.permutation(y), n_estimators=100, split=split)
            for _ in range(perms)]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", default=None,
                    help="comma-separated CSVs; concatenated, scenario taken from column")
    ap.add_argument("--scenarios", default=None,
                    help="comma-separated scenario labels to score (default: the three "
                         "published network scenarios)")
    ap.add_argument("--ns", default="1,5,10,50,100,500",
                    help="comma-separated window sizes")
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--per-model", type=int, default=0,
                    help="cap rows per model in every scenario so both layouts "
                         "see identical window counts (0 = use all)")
    ap.add_argument("--drop-failed", action="store_true",
                    help="also require enclave_us > 0 (changes the curve; the "
                         "published run did not)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    paths = ([Path(p) for p in a.inputs.split(",")] if a.inputs else list(DEFAULT_INPUTS))
    frames = []
    for p in paths:
        if not p.exists():
            print(f"[probe] missing input {p}")
            return 1
        d = pd.read_csv(p)
        d["__src"] = p.name
        frames.append(d)
        print(f"[probe] {p.name}: {len(d)} rows, scenarios={sorted(d['scenario'].unique())}")
    df = pd.concat(frames, ignore_index=True)
    # Uniform failure filtering: failed responses must not become latency or
    # overhead features. The legacy flag is accepted only for CLI compatibility.
    before = len(df)
    df = df[(df["roundtrip_us"] > 0) & (df["enclave_us"] > 0)].copy()
    print(f"[probe] valid timing rows: {before} -> {len(df)}")
    df["overhead_us"] = df["roundtrip_us"] - df["enclave_us"]

    scenarios = a.scenarios.split(",") if a.scenarios else DEFAULT_SCENARIOS
    ns_all = [int(x) for x in a.ns.split(",")]
    # overhead_resid_us removes a pooled linear dependence on enclave time, so a
    # curve that survives it is transport state; a curve that dies with it was
    # compute leaking through the subtraction.
    features = ["enclave_us", "roundtrip_us", "overhead_us", "overhead_resid_us"]
    # roundtrip_us is the published F-4 curve, so it needs its own null too: at
    # n=100 windows RF overfits, and 1/9 is not the floor for any of these columns.
    nulled = ("roundtrip_us", "overhead_us", "overhead_resid_us")

    rows = []
    for scen in scenarios:
        d = df[df["scenario"] == scen].copy()
        if len(d) == 0:
            print(f"[probe] no rows for {scen}, skip")
            continue
        if a.per_model:
            before = len(d)
            d = downsample_per_model(d, a.per_model)
            print(f"[probe] {scen}: --per-model {a.per_model} -> "
                  f"{before} -> {len(d)} rows")
        slope, intercept, _r, _p, _se = sps.linregress(d["enclave_us"], d["overhead_us"])
        d["overhead_resid_us"] = d["overhead_us"] - (slope * d["enclave_us"] + intercept)
        print(f"[probe] {scen}: pooled overhead ~ {slope:+.3f}*enclave {intercept:+.0f} us")
        counts = d.groupby("model_id").size()
        print(f"[probe] === {scen} === {len(d)} rows, "
              f"{counts.min()}-{counts.max()} per model")
        for col in features:
            cells = []
            for n in ns_all:
                if counts.min() // n < MIN_WINDOWS_PER_MODEL:
                    cells.append(f"n={n}:  -- ")
                    continue
                rng = np.random.default_rng(SEED)
                lat = {m: g[col].to_numpy() for m, g in d.groupby("model_id")}
                X, y = _windows(lat, n, rng)
                acc = _fit_score(X, y)
                if col in nulled and a.perms:
                    nm, ns = _null(X, y, a.perms)
                    z = (acc - nm) / ns if ns and ns > 0 else float("nan")
                else:
                    nm, ns, z = float("nan"), float("nan"), float("nan")
                cells.append(f"n={n}:{acc:.3f}")
                rows.append({"scenario": scen, "feature": col, "n": n,
                             "n_windows": int(len(X)), "accuracy": round(acc, 4),
                             "n_classes": int(len(np.unique(y))),
                             "chance": round(1 / len(np.unique(y)), 4),
                             "null_mean": None if np.isnan(nm) else round(nm, 4),
                             "null_sd": None if np.isnan(ns) else round(ns, 4),
                             "z_vs_null": None if np.isnan(z) else round(z, 2),
                             "source_file": d["__src"].iloc[0]})
            print(f"[probe] {col:18s} " + "  ".join(cells))

    oh = pd.DataFrame([r for r in rows if r["feature"] in nulled
                       and r["z_vs_null"] is not None])
    if len(oh):
        print("\n[probe] overhead-feature confound summary (z>2.5 == real leakage "
              "beyond RF overfit floor)")
        print(oh[["scenario", "feature", "n", "accuracy", "null_mean", "null_sd",
                  "z_vs_null"]].to_string(index=False))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\n[probe] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
