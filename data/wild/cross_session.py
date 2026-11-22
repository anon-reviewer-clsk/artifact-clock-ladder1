"""Cross-session identification (main evidence, PREREG §6.2).

Train on session A, test on session B. Nothing from B touches scaling or centroids.

Rules (fixed before any cross-session data existed):
  * scaler and centroids fitted on the TRAIN session only, per prompt;
  * nearest-centroid over the models present in both sessions;
  * budget k: non-overlapping blocks of k consecutive rounds within each
    (model, prompt) cell, features averaged inside the block. 17 rounds/cell, so
    k in {1, 5, 17};
  * chance = 1 / n_models;
  * null: permute model labels WITHIN (prompt, block) groups, so each group keeps
    all 12 models; B = 2000;
  * CI: cluster bootstrap over rounds, B = 2000, percentile.

Efficiency note: a test unit's predicted label depends only on its features, never
on its label. Predictions are therefore computed once per cell and the permutation
only reshuffles labels against a fixed prediction vector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RNG = np.random.default_rng(20261119)
B = 2000

FEATS = {
    "C1_packet": ["content_chunk_count", "chunk_bytes_cv"],
    "C3_length": ["output_text_bytes"],
    "C2_timing": ["ttft_content_ms", "total_ms"],
    "null_wall_ts": ["wall_ts"],
    "null_order": ["seq_in_round"],
    "null_prompt_tokens": ["prompt_tokens"],
}


def load(name: str) -> pd.DataFrame:
    d = pd.read_csv(OUT / name)
    d = d[(d.http_status == 200) & d.done_seen.astype(str).str.lower().eq("true")].copy()
    for c in set(sum(FEATS.values(), [])) | {"round", "seq_in_round"}:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def make_units(d: pd.DataFrame, k: int, cols: list[str]) -> pd.DataFrame:
    if k == 1:
        u = d.copy()
        u["block"] = u["round"].astype(int)
        keep = ["model_requested", "prompt_id", "block", "round"] + cols
        return u[[c for c in keep if c in u.columns]].reset_index(drop=True)
    parts = []
    for _, g in d.groupby(["model_requested", "prompt_id"]):
        g = g.sort_values("round").reset_index(drop=True)
        g["block"] = g.index // k
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    u = d.groupby(["model_requested", "prompt_id", "block"], as_index=False)[cols].mean()
    rnd = d.groupby(["model_requested", "prompt_id", "block"], as_index=False)["round"].median()
    u = u.merge(rnd, on=["model_requested", "prompt_id", "block"], how="left")
    return u[["model_requested", "prompt_id", "block", "round"] + cols].reset_index(drop=True)


def fit(train: pd.DataFrame, cols: list[str]):
    st, cen = {}, {}
    for pr, g in train.groupby("prompt_id"):
        mu, sd = g[cols].mean(), g[cols].std().replace(0, 1e-9)
        st[pr] = (mu.values, sd.values)
        z = (g[cols].values - mu.values) / sd.values
        cen[pr] = {m: z[g.model_requested.values == m].mean(0)
                   for m in sorted(g.model_requested.unique())}
    return st, cen


def predict(units: pd.DataFrame, cols: list[str], st, cen) -> np.ndarray:
    pred = np.array(["UNSEEN"] * len(units), dtype=object)
    for pr in units.prompt_id.unique():
        mask = (units.prompt_id == pr).values
        if pr not in cen:
            continue
        mu, sd = st[pr]
        ms = sorted(cen[pr])
        C = np.vstack([cen[pr][m] for m in ms])
        Z = (units.loc[mask, cols].values - mu) / sd
        idx = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
        pred[mask] = np.array(ms, dtype=object)[idx]
    return pred


def permutation_p(pred: np.ndarray, y: np.ndarray, groups: np.ndarray, obs: float) -> float:
    ge = 1
    idx = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    for _ in range(B):
        yp = y.copy()
        for ii in idx.values():
            yp[ii] = RNG.permutation(yp[ii])
        ge += float(np.mean(pred == yp)) >= obs
    return ge / (1 + B)


def cluster_ci(pred: np.ndarray, y: np.ndarray, rounds: np.ndarray):
    urs = np.unique(rounds)
    if len(urs) < 2:
        return float("nan"), float("nan")
    accs = []
    for _ in range(B):
        pick = RNG.choice(urs, size=len(urs), replace=True)
        sel = np.concatenate([np.where(rounds == r)[0] for r in pick])
        accs.append(float(np.mean(pred[sel] == y[sel])))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def one_direction(tr_name, te_name, tag, budgets=(1, 5, 17)):
    tr, te = load(tr_name), load(te_name)
    common = sorted(set(tr.model_requested) & set(te.model_requested))
    tr = tr[tr.model_requested.isin(common)]
    te = te[te.model_requested.isin(common)]
    chance = 1 / len(common)
    rows = []
    for fname, cols in FEATS.items():
        if not set(cols).issubset(tr.columns):
            continue
        st, cen = fit(tr, cols)
        for k in budgets:
            ute = make_units(te, k, cols)
            per = []
            for pr in sorted(ute.prompt_id.unique()):
                sub = ute[ute.prompt_id == pr].reset_index(drop=True)
                if len(sub) < len(common):
                    continue
                pred = predict(sub, cols, st, cen)
                y = sub.model_requested.values
                obs = float(np.mean(pred == y))
                p = permutation_p(pred, y, sub["block"].values, obs)
                lo, hi = cluster_ci(pred, y, sub["round"].values)
                per.append(dict(direction=tag, feature=fname, k=k, prompt=pr,
                                n=len(sub), acc=round(obs, 3), p=round(p, 4),
                                ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                                chance=round(chance, 4)))
            if per:
                med = float(np.median([x["acc"] for x in per]))
                rows.append(dict(direction=tag, feature=fname, k=k, prompt="MEDIAN",
                                 n=int(sum(x["n"] for x in per)), acc=round(med, 3),
                                 p=round(float(np.median([x["p"] for x in per])), 4),
                                 ci_lo=round(float(np.median([x["ci_lo"] for x in per])), 3),
                                 ci_hi=round(float(np.median([x["ci_hi"] for x in per])), 3),
                                 chance=round(chance, 4)))
                rows.extend(per)
    return pd.DataFrame(rows)


def main(a: str, b: str):
    ta, tb = Path(a).stem.split("_")[-1], Path(b).stem.split("_")[-1]
    frames = [one_direction(a, b, f"{ta}->{tb}")]
    if a != b:
        frames.append(one_direction(b, a, f"{tb}->{ta}"))
    t = pd.concat(frames, ignore_index=True)
    t.to_csv(OUT / "TABLE_cross_session.csv", index=False, encoding="utf-8")
    head = t[t.prompt == "MEDIAN"].sort_values(["feature", "k", "direction"])
    print(head.to_string(index=False))
    (OUT / "cross_session.json").write_text(
        json.dumps(dict(a=a, b=b, rows=t.to_dict("records")), ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
