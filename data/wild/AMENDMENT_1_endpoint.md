# AMENDMENT_1 — 端点性质与 D2 可得性（采集前）

**时间**：2026-11-18 19:0x，**在首次推理请求之前**
**依据**：`GET {base}/compatible-mode/v1/models` 返回 200，184 个模型 id（只读、零费用）
**变更范围**：§3.1（对照可得性）、§4（新增一个字段）。**§6 分析规则与 §9 闸门判据未改动。**

---

## 1 事实更正：所选端点是一个**多 provider 聚合端点**

原判断（`DECISION_platform.md`）假设百炼是"单厂模型 + 自有网关"，D2 需要第二家平台（火山方舟）。
实测该 workspace 端点的模型目录里同时存在**同一模型的多个上游 provider 前缀**：

| 同一模型 | provider A | provider B |
|---|---|---|
| DeepSeek V4 Pro | `deepseek-v4-pro`（原生） | `vanchin/deepseek-v4-pro` |
| DeepSeek V4 Pro 0813 | `deepseek-v4-pro-0813` | `vanchin/deepseek-v4-pro-0813` |
| DeepSeek V4.1 Flash | `deepseek-v4.1-flash` | `vanchin/deepseek-v4.1-flash` |
| DeepSeek V4.1 Flash 1017 | `deepseek-v4.1-flash-1017` | `vanchin/deepseek-v4.1-flash-1017` |
| DeepSeek V3.1 Terminus | `siliconflow/deepseek-v3.1-terminus` | `vanchin/deepseek-v3.1-terminus` |

**含义**：D2（同模型跨 provider）可在**单一端点、单一鉴权、单一网络路径**下完成。
这比原设计更强：网关、TLS 路径、客户端环境全部被固定，唯一变化的是上游 provider，
因此 C1 的 provider 敏感度可以被干净地测出，不掺入跨平台/跨地域的网络差异。

**代价**：本端点无法提供"同模型跨**网关**"的对照。若论文要区分"网关效应"，仍需第二家平台
（火山方舟/硅基流动直连）。这条记为 `gateway_effect: not_executed`，不得省略。

## 2 D1 / D4 的可得性（依模型 id，未做文档核验）

Qwen 开放权重阶梯在同一端点内可得，尺寸写在 id 里：

| 用途 | 候选 id | 说明 |
|---|---|---|
| D1 规模跨度 | `qwen3.5-27b` → `qwen3.5-397b-a17b` | 同族、同 tokenizer、尺寸相差 ~15× |
| D4 困难对 | `qwen3.5-27b` vs `qwen3.5-35b-a3b` | 规模最接近，但**架构不同**（dense vs MoE），只能称"近邻对"，**不得称同架构困难对** |
| D4 备选 | `qwen3.5-122b-a10b` vs `qwen3.5-397b-a17b` | 同为 MoE，规模 3.3× —— 也**不是**相邻规模 |

> 结论：**本端点无法提供严格意义上的 D4 同架构相邻规模困难对。** 按 §3.1 记为
> `hard_pair: not_executed`（或用 `qwen3.5-27b`/`35b-a3b` 作"近邻对"并明确标注架构差异）。
> 不得用"名字看起来接近"顶替。

## 3 字段变更（§4）

新增一列：`reasoning_chars`（推理型模型在 `delta.reasoning_content` 中输出的字符数）。

理由：若模型默认开启思考，`max_tokens=160` 可能被思考过程吃光，正文为空；
此时 `output_text_bytes` 会趋近 0，若不加区分会把"思考截断"误判成"网关定长重打包"。
该列在采集前加入，采集后不得再改。

## 4 未改动

- §2 通道定义、§5 基线清单、§6 有效样本规则与统计、§9 闸门判据：**逐字未改**。
- §11 TEE：仍不开展。

## 5 需你知情的两点

1. 我把"端点性质"从"单厂网关"更正为"多厂聚合网关"，这影响论文表述：
   结果只能称"**单一聚合网关下**的包结构信号"，不能称"跨平台在野"。
2. 火山方舟/硅基流动暂时**没有 key**（`<data-root>\api key\` 下只有一个阿里文件），
   因此 `gateway_effect` 与"跨平台外部效度"这两条，本轮仍然做不了。
