# wild_full — 在野模型服务的计时/包结构识别（完整实验）

**状态**：协议与采集器就绪，**尚未采集任何真实数据**。
**真实网络请求数**：0（仅本地 loopback mock 测试）。

## 目录

| 文件 | 作用 | 状态 |
|---|---|---|
| `PREREG_full.md` | 预注册 v1：设计、通道、基线、统计、闸门、预算、治理 | 已冻结（见 `HASHES_precollection.txt`） |
| `prompts.json` | 6 条 prompt + 冻结参数（temp=0, max_tokens=160, stream） | 已冻结 |
| `collector.py` | 采集器：默认 dry-run，需 `--execute --i-have-approval` | 已通过 mock 测试 |
| `mock_server.py` | 本地 SSE mock（仅 127.0.0.1），按模型名切换分块/429/截断/缺 usage | — |
| `test_mock.py` | 离线测试，产出 `MOCK_TEST_RESULTS.md` | **15/15 通过** |
| `MOCK_TEST_RESULTS.md` | 测试明细 | — |
| `BUDGET.md` | 请求规模、token 上界、费用包络、时间成本（价格待填） | 待你填价 |
| `CONFIRM_CHECKLIST.md` | **采集前必须你批准的 7 项** | 待批 |
| `HASHES_precollection.txt` | 采集前哈希（证明协议先于数据） | — |
| `out_mock/` | mock 运行产物（72 请求的请求级 CSV + 测试结果 JSON） | — |

## 未做的事（有意）

- 未发任何真实请求、未注册账户、未联系 provider、未产生费用。
- 未实例化最终模型列表：模型可用性无法离线核验，`PREREG_full.md` §3.1 冻结的是
  **选择规则与候选池**，实例化在冒烟阶段产出并写入 `MODELS_FROZEN.json`。
- 未做 TEE：无 attestation、无替换 ground truth，按 §11 不开展。

## 跑法

```bash
# 离线自测（只连 127.0.0.1）
python test_mock.py

# 采集前的 dry-run（不发请求）
python collector.py --plan plan.json --session S1-YYYYMMDDTHHMM --out out --key-file <path>

# 真实采集（需两项显式开关，且需先获批准）
python collector.py --plan plan.json --session S1-... --out out \
    --key-file <path> --execute --i-have-approval
```

## 下一步

1. 你回填 `CONFIRM_CHECKLIST.md` 的 7 项。
2. 我据此产出冒烟计划（12 请求）→ 你确认 → 执行 → 报可用模型与真实 usage 行为。
3. 冻结 `MODELS_FROZEN.json` → S1/S2/S3 逐 session 采集，每个 session 结束先汇报再继续。
