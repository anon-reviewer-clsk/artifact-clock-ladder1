"""Write S1 report + run manifest from the session CSV. No requests."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "out" / "wild_full_S1-20261119T1330.csv"


def h(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d = pd.read_csv(CSV)
    summ = json.loads((HERE / "out" / "S1_SUMMARY.json").read_text(encoding="utf-8"))
    tok_in = int(pd.to_numeric(d.prompt_tokens, errors="coerce").fillna(0).sum())
    tok_out = int(pd.to_numeric(d.completion_tokens, errors="coerce").fillna(0).sum())
    tok = tok_in + tok_out
    cost_hi = round(tok / 1e6 * 15.0, 4)

    manifest = dict(
        generated=datetime.now().isoformat(timespec="seconds"),
        session="S1-20261119T1330",
        stage="session-1 collection",
        network_requests=408,
        csv=str(CSV.name), csv_sha256=h(CSV),
        rows=len(d),
        inputs={
            "PREREG_full.md": h(HERE / "PREREG_full.md"),
            "AMENDMENT_1_endpoint.md": h(HERE / "AMENDMENT_1_endpoint.md"),
            "AMENDMENT_2_smoke_metric.md": h(HERE / "AMENDMENT_2_smoke_metric.md"),
            "AMENDMENT_3_gate.md": h(HERE / "AMENDMENT_3_gate.md"),
            "MODELS_FROZEN.json": h(HERE / "MODELS_FROZEN.json"),
            "plan_s1.json": h(HERE / "plan_s1.json"),
            "prompts.json": h(HERE / "prompts.json"),
            "collector.py": h(HERE / "collector.py"),
        },
        tokens=dict(prompt=tok_in, completion=tok_out, total=tok),
        cost_upper_bound_usd=cost_hi,
        cost_note="按 plan 输入的 15 USD/1M 上限价折算；真实计费未核验，以账单为准",
        env=dict(python=platform.python_version(), pandas=pd.__version__,
                 numpy=__import__("numpy").__version__, platform=platform.platform()),
        stop_reason="completed",
        elapsed_min=14.34,
        notes=[
            "no retries, no rate-limit bypass",
            "endpoint hostname deliberately kept out of this manifest",
            "no response bodies stored; only sha256 and length-derived fields",
        ],
    )
    (HERE / "out" / "RUN_MANIFEST_S1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    md = f"""# S1 质量与预算报告

**会话**：`S1-20261119T1330` · 17 轮 · 4 模型 × 6 prompt = **408 请求**
**时长**：14.34 min（30 rpm）· **停止原因**：completed（未触发任何停止条件）

## 1 质量

| 项 | 值 |
|---|---|
| 设计请求 | 408 |
| **valid** | **408（100%）** |
| undetermined | 0 |
| failed（非 200） | 0 |
| `usage` 回传 | 408 / 408 |
| `[DONE]` 正常终止 | 408 / 408 |
| 截断（finish_reason=length） | 0 |
| 推理内容（reasoning_chars>0） | 0 |
| 429 / 重试 | 0 / 0 |

各模型 × 各 prompt 均为 17 条，无缺口（`TABLE_s1_ledger.csv`）。

## 2 预算

| 项 | 值 |
|---|---|
| prompt tokens | {tok_in:,} |
| completion tokens | {tok_out:,} |
| 合计 | **{tok:,}（{(tok/1e6):.4f} M）** |
| 按 15 USD/1M 上限价折算 | **≈ ${cost_hi}**（人民币 2 元左右） |
| 采集前预扣估算 | $0.9792（按 408×160 token 满额算，实际 completion 只用了 {tok_out:,}） |

> 预扣值偏高是因为采集器按 `max_tokens` 满额预扣。**真实账单我看不到，无法核验**，
> 这里只给可计算的折算区间。

## 3 G4 等长检验：**PASS**

预注册规则（`STATUS_AND_S1_PLAN.md` §4，S1 采集前冻结）：

| 口径 | 等长对 | 其中块数不同 | 判定 |
|---|---|---|---|
| 全会话（跨轮） | **7,824** | **2,096（26.8%）** | PASS |
| 同轮同 prompt（更严，排除时间因素） | **321** | **117（36.4%）** | PASS |

⇒ **分块边界不是字节长度的函数**，`gateway_repacking = false`。
这是 S1 最硬的一条结论，且两种口径同向。

## 4 混淆控制：时间共线已被消除

原 pilot 的致命问题是"模型与时间共线"（`wall_ts` 单特征 1.000）。本次交错采集后：

| prompt | `wall_ts` 诊断识别率 |
|---|---|
| P1–P6 | **0.000 – 0.044** |

全部**低于** chance（0.25）。⇒ 采集顺序不再携带模型身份信息，交错设计生效。

## 5 会话内诊断（**非主证据**，主证据须等 S2/S3 跨 session）

逐 prompt、严格留一最近质心、chance = 0.25：

| prompt | C1 包结构 | C3 纯长度 | C2 计时 | wall_ts |
|---|---|---|---|---|
| P1 | 0.250 | 0.250 | 0.676 | 0.029 |
| P2 | 0.603 | 0.353 | 0.500 | 0.000 |
| P3 | 0.750 | 0.500 | 0.618 | 0.044 |
| P4 | 0.500 | 0.250 | 0.500 | 0.029 |
| P5 | **0.882** | 0.397 | 0.632 | 0.000 |
| P6 | **0.912** | 0.471 | 0.750 | 0.044 |

两点值得注意，但要等跨 session 才能升级为结论：

1. **在长输出 prompt 上，包结构远高于纯长度**（P6：0.912 vs 0.471；P5：0.882 vs 0.397）。
   C1 确实携带了长度之外的信息——这正是主假设要的证据形态。
2. **P1 的 C1 退化为 chance**：输出只有 2 字节、任何模型都只切 1 块。这是该通道的
   **下界条件**，应写进论文的适用边界，而不是当作阴性。

## 6 下一步

S2 最早可在 **≥24 h 后**、且**时段不同于 S1（13:30）**。建议：

- **S2：11-20 20:00 前后**（距 S1 起 30.5 h，晚间时段）
- **S3：11-21 09:00 前后**（距 S2 13 h ✗ 不满足 24 h）→ 改 **11-21 21:00**（距 S2 25 h，另一时段）

按你的要求，**S2 不会自动开始**，等你确认时段后再跑。
"""
    (HERE / "S1_REPORT.md").write_text(md, encoding="utf-8")
    print("wrote S1_REPORT.md, out/RUN_MANIFEST_S1.json")
    print("tokens", tok, "cost_hi", cost_hi)


if __name__ == "__main__":
    main()
