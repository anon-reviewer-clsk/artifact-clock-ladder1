# PREREG_full — 在野模型服务的计时/包结构识别（完整实验）

**版本**：v1（2026-11-18 冻结）
**状态**：**未采集**。本文件在任何真实请求发出之前写死；一经哈希，采集期间不得修改。
**范围**：仅第三方**非 TEE** 模型服务。TEE / 替换审计**不在本版范围内**（见 §11）。
**前序**：`wild/`（原 pilot）与 `wild_fix/`（阶段一修复）只读复用，不产生新结论。

---

## 0 为什么必须重采，以及本版不重复哪些错误

原 pilot 的五条致命缺陷，本协议逐条对应消除：

| 原缺陷 | 本版对策 | 对应条款 |
|---|---|---|
| 按模型分块采集，模型与时间完全共线 | 每轮含全部模型，轮内随机 + Latin 轮转 | §3.3 |
| 单 session | 3 个独立日期/session，间隔 ≥24 h | §3.2 |
| 伪 LOO（质心含自身） | 跨 session 留出为主证据；同 session 仅作诊断 | §6.2 |
| `dropna` 静默删失样本 | 统一有效样本规则 + 双口径并报 | §6.1 |
| 异常 200 未记录、无法裁定 | 逐条 `undetermined` 记账，不裁定、不删除 | §6.3 |

**本版不承诺阳性结果。** 阴性即按阴性报告，不因结果调整阈值、删模型或挑 session。

---

## 1 研究问题

- **RQ1**：真实第三方服务上，**SSE 包结构**（chunk 数、每 chunk 字节、SSE/JSON 封装占比）能否在
  **跨 session** 条件下识别模型？
- **RQ2**：该信号能否与以下因素区分：采集时间、采集顺序、provider、prompt、限流重试、
  动态批处理、序列化实现、`usage` 显式元数据、纯输出长度？
- **RQ3**：在同模型跨 provider、不同量化档位、同架构相近计算量的**困难对**上，信号是否仍然成立？
- **RQ4**（条件性）：只有在拿到**真实 TEE 响应 + 可验证 attestation + 已知替换 ground truth**
  时才扩展；否则不开展、不暗示。本版不开展。

**主假设 H1**：包结构特征在 session A 上建立的模型画像，能在 session B 上以高于 chance 且高于
全部基线的准确率识别模型。
**零假设 H0**：不能；或能，但其判别力可被时间/顺序/provider/prompt/元数据基线完全解释。

---

## 2 通道定义（预先指定，不可事后扩充）

| 通道 | 特征向量 | 角色 |
|---|---|---|
| **C1 包结构**（主） | `[content_chunk_count, chunk_bytes_mean, chunk_bytes_cv, sse_overhead_ratio, first_chunk_bytes]` | 主检验 |
| **C2 计时**（次） | `[ttft_content_ms, total_ms, inter_chunk_gap_mean]` | 次要，Holm 校正 |
| **C3 输出长度**（第三） | `[output_text_bytes, output_chars]` | 次要，Holm 校正 |
| **C4 纯长度对照** | `[output_text_bytes]` 单独 | 用于检验 C1 是否只是在测长度 |

- `sse_overhead_ratio = (sse_bytes_total - output_text_bytes) / sse_bytes_total`
- C1/C2/C3 特征在**训练 session 内部**做 z-score（均值/标准差仅用训练集），测试 session 用同一组参数。
- **C1 的成绩必须在控制 C4 后仍有增量**，否则只能称"输出长度信号"。

---

## 3 设计

### 3.1 模型选择规则（先定规则，冒烟后实例化）

要求 4–6 个模型，且**尽可能**覆盖下列对比。每一类对比都必须可核验；**无法核验即记“未执行”**，
不得用近似项顶替后声称已做。

| 对比 | 要求 | 若无法满足 |
|---|---|---|
| D1 规模/架构差异 | ≥2 个明显不同规模或家族 | 至少保留 2 个不同家族 |
| D2 同模型跨 provider | 同一 model id 在 ≥2 个 provider 可用，且 provider 可被显式指定 | 记 `cross_provider: not_executed` |
| D3 量化档位 | provider **公开文档**写明量化档（fp16/int8/awq 等） | 记 `quant: unverified`，不做量化结论 |
| D4 困难对 | 同架构、参数规模相邻（比值 ≤2×），或同模型两档量化 | 记 `hard_pair: not_executed` |

**候选池（可用性全部未核验，禁止默认可用）**

| 候选 | 来源 | 已知历史 | 状态 |
|---|---|---|---|
| Nemotron 系列（原 `nvidia/nemotron-3.5-lightning:free`） | 原 pilot | 20/20 成功 | 候选，需冒烟复核 |
| Liquid 系列（原 `liquid/lfm-3-1.7b:free`） | 原 pilot | 11/20 成功 | 候选，需冒烟复核 |
| Gemma 系列（原 `google/gemma-4-31b-it:free`） | 原 pilot | **0/20 成功（20/20 = 429）** | **不得默认可用**，仅作候选 |

> 说明：模型身份无法在离线状态核验。因此本节冻结的是**选择规则、候选池与替换顺序**，
> 最终实例化列表在冒烟阶段（§9 G1）产出并写入 `MODELS_FROZEN.json`，同时记录该文件哈希。
> 这是"规则先冻结、身份后实例化"，不是事后挑模型：实例化只能按本节规则与顺序进行，
> 且任何未满足的对比必须显式记为 `not_executed`。

### 3.2 Session

- **3 个 session**：`S1 / S2 / S3`，相邻间隔 **≥24 h**，且**采集时段不同**（上午 / 下午 / 晚间各一），
  以削弱昼夜负载与路由策略的共线。
- 每个 session 有独立 `session_id`（形如 `S1-20261119T0930`），写入每一行。
- 不允许把两个 session 的数据合并成"一个 session"再分析。

### 3.3 Round 与随机化

- 每轮 = **全部模型 × 全部 prompt 各一次**（`M × P` 个请求）。
- **禁止按模型分块**：轮内请求顺序用固定种子的置换随机化。
- **模型顺序 Latin 轮转**：模型数 M，则第 r 轮的模型次序循环左移 `r mod M`，
  使每个模型在每个序位上出现次数相等；轮内 prompt 顺序独立随机化。
- 轮数 `R = ceil(100 / P)`。取 `P = 6` ⇒ `R = 17` ⇒ 每模型每 session `6×17 = 102` 次请求。
- 每 session 请求数 = `M × P × R`。M=5 时 510；M=6 时 612。
- 目标：每模型每 session **≥100 条成功**；每模型分布在 **17 个独立 round**（满足块级统计）。
- 若某模型 429 率过高致成功数 <100，**不允许延长该模型的采集**（会重新引入分块），
  按 §9 G1 处理并如实报为 `insufficient_success`。

### 3.4 请求参数（冻结）

`stream=true`、`temperature=0.0`、`max_tokens=160`、`top_p` 不设、不加 system prompt。
每个 session 的 round 顺序固定（不随机化轮次顺序），但记录 `round` 与 `wall_ts`。

### 3.5 Prompt 集合（冻结，6 条）

`prompts.json` 单独存放并于采集前哈希。设计为 2 短 / 2 中 / 2 开放，用于检验 prompt 是否为混淆项：

| id | 类型 | 内容 |
|---|---|---|
| P1 | 短-约束 | `Reply with exactly one word: ok.` |
| P2 | 短-约束 | `List the integers 1 to 5, comma separated, nothing else.` |
| P3 | 短-约束 | `Answer with a single digit: 7*6=?` |
| P4 | 中-约束 | `Name the three primary colors, comma separated, no other text.` |
| P5 | 中-开放 | `Write exactly two sentences on why caching helps web servers.` |
| P6 | 开放 | `Summarise the plot of an invented film in about 60 words.` |

- 6 条 prompt 全部参与主分析；**另报**"仅约束型（P1–P4）"与"仅开放型（P5–P6）"两个敏感性视图。
- 不保存 prompt 之外的任何输入；prompt 全文可公开，无敏感内容。

---

## 4 记录字段（逐请求，CSV 一行）

**必填，缺一即该行 `invalid`：**

| 组 | 字段 |
|---|---|
| 身份 | `session_id, round, seq_in_round, model_requested, provider, model_served, request_id` |
| 输入 | `prompt_id, prompt_sha256` |
| 时间 | `wall_ts, t_start_mono, t_end_mono`（单调时钟，纳秒） |
| 状态 | `http_status, http_attempts, retry_after_s, error_class` |
| 计时 | `ttfb_ms, ttft_role_ms, ttft_content_ms, total_ms, inter_chunk_gap_mean_ms` |
| 包结构 | `chunk_count`（全部 SSE data 行，含 role 与终止块）, `content_chunk_count`（仅内容块）, `sse_bytes_total, sse_overhead_bytes, chunk_bytes_min, chunk_bytes_max, chunk_bytes_mean, chunk_bytes_cv, first_chunk_bytes` |
| 输出 | `output_text_bytes, output_chars, completion_tokens, prompt_tokens, finish_reason, done_seen, truncated, content_sha256` |
| TEE（本版恒为空） | `attest_present, attest_verified, attest_bound_to` |

- `ttfb_ms` = 首字节；`ttft_role_ms` = 第一条 SSE `data:` 行（常为空 role chunk）；
  `ttft_content_ms` = 第一条 `delta.content` 非空的 chunk。三者分开记录，避免把 role chunk 当首 token。
- `resp_bytes` 直接使用 `sse_bytes_total`，并在报告中**明确其含 SSE/JSON 封装**。
- **不保存**输出正文，只保存 `content_sha256` 与长度类派生量。
- **不保存** API key、账户、公网 IP、host 名。

---

## 5 baseline（必报，逐项）

1. `wall_ts`（session 内时间戳，以及跨 session 的绝对时间）
2. 采集顺序（session 内 seq；跨 session 的绝对序号）
3. `provider`（若 provider 与模型一一对应，则此项**必然**与模型共线，必须显式声明）
4. `prompt_tokens` / `completion_tokens`（**显式元数据，单列**，不计入侧信道）
5. `prompt_id`（检验 prompt 本身是否泄露模型）
6. `content_chunk_count` 与 `sse_overhead_ratio`（用于判定 C1 是否只是"输出长度"或"分块实现"的代理）

判定规则：
- 任一通道 `acc ≤ max(所有 baseline acc)` ⇒ 判定"无可归因于模型身份的增量"，不得称弱信号。
- `provider` 与模型一对一时，**该 session 的 provider 对比作废**，只报"不可分离"。

---

## 6 分析纪律

### 6.1 有效样本规则（统一，先于结果）

`valid := http_status==200 且 done_seen 且 completion_tokens 非空 且 finish_reason 非空`

- **主表**：`valid` 全含（**不剔除任何异常**）。
- **敏感性表**：额外剔除 `truncated==true` 与 `undetermined` 行。
- 二表必须并报；**不得只报其一**。`truncated` 行参与 C1/C2，但**不参与 C3**（长度被 `max_tokens` 截断，
  属删失），此规则先于结果固定。
- 所有 `dropna` 必须打印被丢弃的行数与 `session×model×reason` 分布，写入失败账本。

### 6.2 主证据：跨 session

- 主结果 = `S1→S2`、`S2→S1`、`S1→S3`、`S2→S3`、`S3→S1`、`S3→S2` 六个方向的**留出识别**，
  以及三 session 的 leave-one-session-out（LOSO）。
- **同 session 内随机划分只作诊断，不作为主证据，不得单独引用。**
- 每个方向报告：n=1 单发、n=10/50/100 预算下的累计准确率、混淆矩阵、弃权率（若用阈值规则）。
- **逐模型**与**逐 pair**（含 D4 困难对）分别报告，不只报总体。

### 6.3 异常与失败

- `undetermined` = HTTP 200 但 `done_seen==False` 或 usage 缺失或 `finish_reason` 缺失；
  逐条记录，不猜测成因，不删除，双口径并报。
- 失败账本按 `session × model × status × error_class` 汇总，并给出 429 的时间分布
  （分钟级直方图），用于检验限流是否与模型共线。
- HTTP 尝试次数逐条记录；**默认 `max_retries=0`**。若启用重试（需在 §10 预算中显式批准），
  必须遵守 `Retry-After`、上限 1 次，并在 `http_attempts` 与 `retry_after_s` 中体现。

### 6.4 统计

- **区间**：session 内按 **round 块级 bootstrap**（重采样 round，不重采样请求），B=2000，固定种子。
  请求级 Wilson 只作描述性并标注"忽略聚类，不可作推断"。
- **检验**：round 内**标签置换**（每个 round 内打乱模型标签，保持轮结构与特征分布），B=2000。
  该零分布对应"profile 无法越过轮结构识别模型"，正是 H0。
- **最小可达 p**：每 session 17 个 round、M 个模型 ⇒ 置换空间足够，不存在 n_eff=2 的退化。
  报告实际 p 及 Monte-Carlo 误差（p̂ ± 1/√B）。
- **多重比较**：主检验唯一（C1 × n=1 × `S1→S2`），α=0.05，不校正。
  其余（C2/C3/C4、其他方向、其他预算）在同一 family 内 Holm 校正，family 定义见 `RESULTS` 表头。
- **方向一致性**：H1 要求六个方向中 ≥5 个同向且显著，且 LOSO 平均不低于单方向中位数。
  少于该门槛 ⇒ 判为"方向不稳定"，不得择优报告。

### 6.5 REMOTEID（若使用）

必须重新检验标定平稳性：在 `S1` 上标定，在 `S2/S3` 上跑，报告**判定率、错误声明率、弃权率**。
跨 session 漂移导致错误声明，如实报告，不得只报同 session 重放。

---

## 7 功效（设计目标，含未核验假设）

- chance = 1/M（M=5 时 0.2）。
- 每 session 17 轮，每模型每轮 6 条（每 prompt 1 条）。
- **假设 ICC(轮内) ≈ 0.1**（未核验，采集后用实际 ICC 复核并写入报告）。
- 在该假设与 n=102/模型/session 下，检测 `acc=0.45` 对比 chance=0.2、α=0.05 双侧，
  预期功效 ≥0.8；检测 `acc=0.35` 时功效约 0.5（不足）。
- 结论：本设计能检出**中等以上**效应；对小效应**不具功效**，届时只能报"未检出"，
  不得解释为"不存在"。

---

## 8 停止条件（任一触发即停当前 session，并报告）

| 条件 | 阈值 |
|---|---|
| 累计估算费用 | ≥ 单 session 预算 |
| 单 session 时长 | ≥ 90 min |
| 连续 429 | ≥ 20 次 |
| 429 率（滚动 50 次） | ≥ 60% |
| 失败率（滚动 50 次） | ≥ 50% |
| 任一模型成功数 | 达到 140 次（上限保护） |
| 手动中止 | — |

停止后**不得**为凑数而在同日续采；该 session 按实际数据入账，缺口如实报。

---

## 9 Go / No-Go

| 闸门 | 判据 | 不通过时 |
|---|---|---|
| **G1 冒烟**（每模型 3 条 × 6 prompt 中抽 3） | 至少 4 个模型取得 ≥2 条成功响应 | 停止，不进入正式采集，不扩大 |
| **G2 共线检查** | 时间/顺序基线不得 ≥ 主通道；provider 若与模型一一对应须声明 | 先修设计，不进入分析 |
| **G3 跨 session** | 六方向结果不得全部 ≤ chance 或 ≤ 基线 | 报阴性，停止，不调参续采 |
| **G4 包结构可分性** | C1 与 C4 必须可分 | 不可分 ⇒ 只称"SSE 包结构信号"或"长度信号"，二者不可混称 |
| **G5 TEE** | 需真实 attestation + 替换 ground truth | 无则**禁止**任何 TEE 替换结论 |
| **G6 attestation 绑定** | 绑定环境 ≠ 绑定权重 | 只研究行为变化，不得声称证明确切权重替换或恶意掉包 |

---

## 10 预算（框架，数值待批）

- 单 session 预算上限、总预算上限、是否允许付费探测：**待用户填写**（见 `CONFIRM_CHECKLIST.md`）。
- 预估式：`cost ≈ Σ_requests (prompt_tokens + completion_tokens)/1e6 × 单价`。
  由于 token 数在响应后才知，采集器按 `prompt 估计 + max_tokens` 做**保守预扣**，达到上限即停。
- 最坏失败费用 = 已发出请求的实际计费之和（429 通常不计费，但可能计费，按**全部计费**估计）。
- 明细见 `BUDGET.md`。

---

## 11 TEE：本版明确不做

- 触发条件（三者同时满足）才扩展：真实 TEE 响应、**可验证** attestation、**已知替换 ground truth**。
- 若 attestation 只绑定运行环境而不绑定权重：只能研究行为一致性与变化检测；
  **不得**声称"证明确切权重替换"或"恶意掉包"。
- 本版所有行 `attest_*` 字段恒为空；报告中不得出现任何 TEE 结论或暗示。

---

## 12 数据治理

- 不保存 API key、账户、公网 IP、host 名、响应正文；只保存 `content_sha256` 与长度派生量。
- 每条请求只保存一个 prompt 的**公开文本哈希**，不保存用户输入。
- 原始 CSV 逐行落盘（flush），崩溃不丢数据；每 session 结束计算哈希写入 `RUN_MANIFEST.json`。
- 发布制品前扫描并剔除任何可识别信息（含路径、主机名、账户片段）。

---

## 13 冻结声明

本文件在首次真实请求之前冻结，哈希写入 `PREREG_full.sha256`，并记入每个 session 的
`RUN_MANIFEST.json`。采集开始后，任何改动必须以 `AMENDMENT_n.md` 形式单独提交并说明
"改动发生在第几个 session 之后"，且不得改动 §6 的分析规则与 §9 的闸门判据。
