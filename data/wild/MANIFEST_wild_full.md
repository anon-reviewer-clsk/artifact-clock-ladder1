# MANIFEST — 在野模型识别（wild_full）

生成：2026-11-20 · 根目录 `<data-root>/Analysis/reports/wild_full/`

**你截图里的 4 个文件只是最终交接面**：`FINDINGS_wild_full.md` + `TABLE_cross_session.csv` + `out_cross.log` + `AMENDMENT_5_same_day.md`。
原始数据、预注册、代码、账本、清单**都不在其中**。本清单是完整集合。

## 顶层文件

| 文件 | 字节 | 角色 |
|---|---|---|
| `AMENDMENT_1_endpoint.md` | 3,565 | 协议：端点性质与 D2 可得性勘误 |
| `AMENDMENT_2_smoke_metric.md` | 4,192 | 协议：闸门判据修订（含 usage 结论更正） |
| `AMENDMENT_3_gate.md` | 3,247 | 协议：等长检验判据 |
| `AMENDMENT_4_scale12.md` | 3,430 | 协议：扩到 12 模型、S1 重跑 |
| `AMENDMENT_5_same_day.md` | 2,227 | 协议：同日复现的偏离声明 |
| `ASSESS_importance.md` | 4,081 | 报告：重要性评估与三个方案 |
| `BUDGET.md` | 3,695 | 报告：预算框架与包络 |
| `collector.py` | 16,432 | 代码：采集器（默认 dry-run） |
| `CONFIRM_CHECKLIST.md` | 4,216 | 报告：采集前待批清单 |
| `cross_session.py` | 7,331 | 代码：跨 session 主分析（置换 + 块级 bootstrap） |
| `DECISION_platform.md` | 5,420 | 报告：平台选型分析 |
| `FINDINGS_wild_full.md` | 4,034 | 报告：最终结论（可说/不可说） |
| `gate2.py` | 8,888 | — |
| `gate2_analyze.py` | 7,252 | — |
| `GATE2_RESULT.md` | 3,198 | 报告：GATE-2 判定与判据缺陷说明 |
| `gate3.py` | 6,367 | 代码：GATE-3 等长检验 |
| `GATE3_RESULT.md` | 875 | 报告：GATE-3 等长检验 |
| `HASHES_precollection.txt` | 399 | 完整性：采集前哈希 v1 |
| `HASHES_precollection_v3.txt` | 584 | 完整性：协议+修订案哈希链 |
| `mock_server.py` | 4,524 | 代码：本地 SSE mock |
| `MOCK_TEST_RESULTS.md` | 2,116 | 报告：mock 测试 15 项 |
| `MODELS_FROZEN.json` | 2,520 | 输入：4 模型 pilot 的模型集与排除记录 |
| `MODELS_FROZEN_v2.json` | 3,811 | 输入：12 模型正式集与排除记录 |
| `out_cross.log` | 4,033 | 日志：跨 session 分析完整输出 |
| `out_preflight_run.log` | 1,718 | 日志：12 模型预检 |
| `out_s1v2.log` | 612 | 日志：S1v2 采集 |
| `out_s2v2.log` | 612 | 日志：S2v2 采集 |
| `plan_s1.json` | 689 | 输入：4 模型采集计划 |
| `plan_s1v2.json` | 1,207 | 输入：12 模型 S1v2 计划 |
| `plan_s2v2.json` | 1,516 | 输入：12 模型 S2v2 计划 |
| `preflight12.py` | 3,885 | 代码：12 模型可用性预检 |
| `PREREG_full.md` | 16,456 | 协议：预注册 v1（冻结于采集前） |
| `prompts.json` | 898 | 输入：6 条 prompt + 采样参数（冻结） |
| `README.md` | 2,324 | 说明：目录导航 |
| `s1_analyze.py` | 7,592 | 代码：单会话质量 + G4 |
| `S1_REPORT.md` | 8,210 | 报告：4 模型 pilot 质量与预算 |
| `s1_report.py` | 5,828 | 代码：S1 报告生成 |
| `S1v2_REPORT.md` | 3,239 | 报告：12 模型 S1v2 质量与诊断 |
| `session_diagnostics.py` | 2,421 | 代码：逐 prompt 会话内诊断 |
| `smoke.py` | 6,819 | 代码：冒烟 9 请求 |
| `SMOKE_REPORT.md` | 4,952 | 报告：冒烟 9 请求 |
| `SMOKE_RESULT.md` | 1,251 | 报告：冒烟自动判定 |
| `STATUS_AND_S1_PLAN.md` | 3,869 | 报告：状态与 S1 计划 |
| `test_mock.py` | 9,469 | 代码：离线测试（15/15） |

## out/ 目录

| 文件 | 字节 | 角色 |
|---|---|---|
| `out/cross_session.json` | 60,621 | 结果：同上，完整 JSON |
| `out/RUN_MANIFEST_S1.json` | 1,621 | 完整性：S1 清单（哈希/token） |
| `out/RUN_MANIFEST_S1v2.json` | 361 | 完整性：S1v2 清单 |
| `out/RUN_MANIFEST_S2v2.json` | 360 | 完整性：S2v2 清单 |
| `out/S1_SUMMARY.json` | 395 | 结果：S1 汇总 |
| `out/session_diag.json` | 1,188 | 结果：同上 JSON |
| `out/TABLE_cross_session.csv` | 22,519 | 结果：跨 session 双向 × 6 特征 × 3 预算 × 逐 prompt |
| `out/TABLE_s1_features.csv` | 4,198 | 结果：逐模型×prompt 中位数 |
| `out/TABLE_s1_g4.csv` | 268 | 结果：G4 等长检验逐 prompt |
| `out/TABLE_s1_ledger.csv` | 586 | 结果：样本账本（逐模型 raw/valid/failed） |
| `out/TABLE_s1_ledger_cells.csv` | 311 | 结果：账本明细（模型×状态） |
| `out/TABLE_session_diag.csv` | 305 | 结果：会话内逐 prompt 诊断 |
| `out/wild_full_S1-20261119T1330.csv` | 151,200 | 数据：4 模型 pilot 请求级原始（408 行） |
| `out/wild_full_S1v2-20261119T1950.csv` | 452,688 | 数据：12 模型 S1v2 请求级原始（1224 行） |
| `out/wild_full_S2v2-20261119T2100.csv` | 452,629 | 数据：12 模型 S2v2 请求级原始（1224 行） |

**合计**：1,339,254 字节（约 1.28 MB）

## 未包含（有意排除）

- `<data-root>/api key/`：**任何情况下都不打包**。计划文件里只有 `FROM_KEY_FILE` 占位符。
- 响应正文：从未保存，只有 `content_sha256` 与长度派生量。
- 工作空间主机名：已从 `SMOKE_REPORT.md` 抹除，其余文件不含。

## 另外两个目录（本包之外）

| 目录 | 内容 |
|---|---|
| `wild/` | 旧 4 模型 pilot 原始（60 请求，31 成功）+ 原 findings，**只读未改** |
| `wild_fix/` | 对 pilot 的方法学修复（伪 LOO、样本删失、时间基线） |

## 复现路径

```bash
python test_mock.py                      # 15/15，无需网络
python cross_session.py \
  wild_full_S1v2-20261119T1950.csv \
  wild_full_S2v2-20261119T2100.csv      # 主结果，约 3.5 min
python session_diagnostics.py wild_full_S1v2-20261119T1950.csv
```
