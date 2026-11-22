"""analyze_phase2.py — 按 PREREG_phase2.md 执行，不做任何事后扩充。

A = rounds 1-4，B = rounds 5-8；标准化只在训练 session 拟合。
置换：打乱测试 session 的模型标签，B=2000。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = pd.read_csv(HERE / "raw" / "stage2.tsv", sep="\t")
RNG = np.random.default_rng(20261120)
B = 2000
FEATS = {"PMC": ["box0", "box1", "box2", "box3"], "SIZE": ["size_bytes"],
         "RTT": ["rtt_p50_ms"], "LAYOUT": ["lay"]}
D["lay"] = (D.input_pa // (1 << 24)).astype(float)


def nc_acc(tr, te, cols):
    mu, sd = tr[cols].mean().values, tr[cols].std().replace(0, 1e-9).values
    Ztr = (tr[cols].values - mu) / sd
    Zte = (te[cols].values - mu) / sd
    models = sorted(tr.model.unique())
    C = np.vstack([Ztr[tr.model.values == m].mean(0) for m in models])
    pred = np.array(models)[((Zte[:, None, :] - C[None]) ** 2).sum(2).argmin(1)]
    return float((pred == te.model.values).mean()), pred


def perm_p(y, pred, obs):
    ge = 1
    for _ in range(B):
        ge += float((pred == RNG.permutation(y)).mean()) >= obs
    return ge / (1 + B)


def boot_ci(y, pred):
    idx = np.arange(len(y))
    accs = []
    for _ in range(B):
        s = RNG.choice(idx, len(idx), replace=True)
        accs.append(float((pred[s] == y[s]).mean()))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


rows = []
A, Bs = D[D["round"] <= 4], D[D["round"] > 4]
for name, cols in [("PMC", FEATS["PMC"]), ("SIZE", FEATS["SIZE"]),
                   ("PMC+SIZE", FEATS["PMC"] + FEATS["SIZE"]), ("LAYOUT", FEATS["LAYOUT"])]:
    for tag, tr, te in [("A->B", A, Bs), ("B->A", Bs, A)]:
        acc, pred = nc_acc(tr, te, cols)
        p = perm_p(te.model.values, pred, acc)
        lo, hi = boot_ci(te.model.values, pred)
        rows.append(dict(set=name, direction=tag, n=len(te), acc=round(acc, 3),
                         p=round(p, 4), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))

# near-null pair（只用有 per-run RTT 的两个 tok 模型）
pair = D[D.model.isin(["dsv1_tok_v4.onnx", "dsv1_tok_v4_l1.onnx"])]
pa, pb = pair[pair["round"] <= 4], pair[pair["round"] > 4]
for name, cols in [("PMC", FEATS["PMC"]), ("RTT", FEATS["RTT"]),
                   ("SIZE", FEATS["SIZE"]), ("PMC+RTT", FEATS["PMC"] + FEATS["RTT"])]:
    for tag, tr, te in [("A->B", pa, pb), ("B->A", pb, pa)]:
        acc, pred = nc_acc(tr.dropna(subset=cols), te.dropna(subset=cols), cols)
        te2 = te.dropna(subset=cols)
        p = perm_p(te2.model.values, pred, acc)
        lo, hi = boot_ci(te2.model.values, pred)
        rows.append(dict(set=f"PAIR:{name}", direction=tag, n=len(te2), acc=round(acc, 3),
                         p=round(p, 4), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))

res = pd.DataFrame(rows)
res.to_csv(HERE / "phase2_results.csv", index=False, encoding="utf-8")
print(res.to_string(index=False))

# 判据
def get(s, d="A->B"):
    r = res[(res.set == s) & (res.direction == d)]
    return float(r.acc.iloc[0]) if len(r) else float("nan")

P1 = get("PMC") <= get("SIZE")
P2 = get("PMC+SIZE") <= get("SIZE") + 1e-9
pair_acc = get("PAIR:PMC")
pair_p = float(res[(res.set == "PAIR:PMC") & (res.direction == "A->B")].p.iloc[0])
P3 = (pair_acc <= 0.5) or (pair_p > 0.05)
P4 = get("LAYOUT") <= 0.375  # 不显著高于 chance 的宽松上界

verdict = dict(P1=bool(P1), P2=bool(P2), P3=bool(P3), P4=bool(P4),
               PMC=get("PMC"), SIZE=get("SIZE"), PMCpSIZE=get("PMC+SIZE"),
               LAYOUT=get("LAYOUT"), pair_PMC=pair_acc, pair_p=pair_p)
json.dump(verdict, open(HERE / "phase2_verdict.json", "w"), ensure_ascii=False, indent=1)

print("\n== 判据 ==")
print(f"P1 PMC≤SIZE: {P1}  (PMC={get('PMC'):.3f} vs SIZE={get('SIZE'):.3f})")
print(f"P2 PMC+SIZE≤SIZE: {P2}  (PMC+SIZE={get('PMC+SIZE'):.3f})")
print(f"P3 near-null 不可分: {P3}  (pair acc={pair_acc:.3f}, p={pair_p:.4f})")
print(f"P4 LAYOUT 不显著: {P4}  (LAYOUT={get('LAYOUT'):.3f})")
