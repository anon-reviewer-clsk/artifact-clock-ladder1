# 冒烟结果（G1）

- 端点 host：`\1`（**未记录完整 URL**）
- 请求：9 条（3 模型 × 3 prompt），max_tokens=160，temp=0，stream
- 成功：9/9

## 逐请求

| 模型 | prompt | status | 内容块 | 输出字节 | 推理字符 | completion_tokens | 备注 |
|---|---|---|---|---|---|---|---|
| qwen3.5-27b | P2 | 200 | 5 | 13 | 0 |  |  |
| qwen3.5-27b | P4 | 200 | 3 | 17 | 0 |  |  |
| qwen3.5-27b | P6 | 200 | 38 | 353 | 0 |  |  |
| qwen3.5-122b-a10b | P2 | 200 | 5 | 13 | 0 | 13 |  |
| qwen3.5-122b-a10b | P4 | 200 | 2 | 17 | 0 | 5 |  |
| qwen3.5-122b-a10b | P6 | 200 | 27 | 386 | 0 | 82 |  |
| qwen3.5-397b-a17b | P2 | 200 | 4 | 13 | 0 | 13 |  |
| qwen3.5-397b-a17b | P4 | 200 | 3 | 17 | 0 | 5 |  |
| qwen3.5-397b-a17b | P6 | 200 | 23 | 319 | 0 | 62 |  |

## 诊断量（先于结果固定）

- `cv_len` = 0.0
- `cv_chunk` = 0.101
- `usage_rate` = 0.667
- `model_echo_rate` = 1.0
- `reasoning_rate` = 0.0
- `done_rate` = 1.0
- `truncated_rate` = 0.0

## 判定

**INCONCLUSIVE — 输出被强约束或样本太少，需改 prompt/加样本后再判**

判据见 `DECISION_platform.md` §4：`CV_chunk>0.15 且 CV_len>0.2` 为通过；`CV_chunk<0.05 且 CV_len>0.3` 为网关重打包。
