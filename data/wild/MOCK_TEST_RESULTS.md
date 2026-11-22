# Mock 测试结果（离线，仅 loopback）

- 请求数：72  ·  mock 端点：`http://127.0.0.1:56469/v1/chat/completions`
- 通过：**15/15**

| # | 检查项 | 期望 | 实测 | 结果 |
|---|---|---|---|---|
| T1 | T1 CSV 列与 PREREG §4 完全一致 | 43 | 43 | ✅ PASS |
| T2 | T2a content_chunk_count 反映分块粒度（chunky>batched>single=1） | 40>4>1 | 42>4>1 | ✅ PASS |
| T3 | T2b chunk_count = 内容块 + role 块 + 终止块 | content+2 | 44 vs 42+2 | ✅ PASS |
| T4 | T3 ttft_role_ms < ttft_content_ms（未把 role chunk 当首 token） | True | True | ✅ PASS |
| T5 | T4 sse_overhead_bytes>0 且 sse_bytes_total>output_text_bytes | True | True | ✅ PASS |
| T6 | T5 正常流 done_seen=True 且 content_sha256 为 64 位 | True | True | ✅ PASS |
| T7 | T6 429: status=429 / error_class / Retry-After / attempts=1 | 429/rate_limited/2/1 | 429/429:rate_limited/2/1 | ✅ PASS |
| T8 | T7 截断: truncated=True / finish_reason=length / done_seen=False | True/length/False | True/length/False | ✅ PASS |
| T9 | T8 缺 usage: completion_tokens 为空（应判 invalid） | '' | '' | ✅ PASS |
| T10 | T9 每轮 = M×P 且 (模型,prompt) 不重复 | 36/round | {0: 36, 1: 36} | ✅ PASS |
| T11 | T10 轮间顺序不同（未按模型分块） | >=2 种顺序 | 2 种 | ✅ PASS |
| T12 | T11 CSV 不含 Bearer/key/公网 IP | none | none | ✅ PASS |
| T13 | T12 预算闸门触发并停止 | budget_cap 且早停 | budget_cap n=3 | ✅ PASS |
| T14 | T13 拒绝非 https 外部端点 | SystemExit | SystemExit | ✅ PASS |
| T15 | T14 默认 max_retries=0（所有行 attempts=1） | 1 | 1 | ✅ PASS |

## 覆盖率

已验证：SSE 分块粒度、role/内容 chunk 分离、SSE 封装开销、`[DONE]`、截断、缺 usage、429 与 Retry-After、不重试、轮内随机与轮间轮转、预算闸门、端点协议守卫、密钥不入盘。

**未验证（需真实服务）**：真实 provider 的 `usage` 是否回传、真实 chunk 粒度、跨 provider 差异、真实 429 策略、attestation 字段。这些只能在冒烟阶段确认。
