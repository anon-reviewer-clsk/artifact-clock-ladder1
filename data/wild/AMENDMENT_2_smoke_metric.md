# AMENDMENT_2（提案，待批准）— 闸门判据与 C1 特征修订

**状态**：**已批准（2026-11-18 22:05）**。按本文件在新请求上复验，不用已跑过的 9 条改判。
**批准范围**：§1 判据统计、§2 C1 去 `first_chunk_bytes`、§3 usage 作模型属性、§4 prompt_tokens 存疑标注、§5 12 请求复验。
**触发原因**：`SMOKE_REPORT.md` §2、§3 暴露了两个设计缺陷。二者都属"判据/特征定义"层面，
不涉及结果方向，但**若不改，正式采集会拿到不可用的数据**。

---

## 1 缺陷一：闸门统计量退化

**现行（PREREG §7 / DECISION §4）**：跨 3 个 prompt 取中位数，再算跨模型 CV。

**问题**：P2/P4 是约束型 prompt，输出长度天然固定（13 B / 17 B）。3 个 prompt 的中位数必落在
约束型区间，`CV_len` 恒为 0，判据永远落在"两者皆低"或不可判分支。这是一个**恒假判据**。

**修订**：改为**同 prompt 内、跨模型**的统计，并对每个 prompt 单独计算后汇总：

```
对每个 prompt p:
    CV_len(p)   = std_over_models(output_text_bytes) / mean_over_models(...)
    CV_chunk(p) = std_over_models(content_chunk_count) / mean_over_models(...)
gate_len   = median_p CV_len(p)
gate_chunk = median_p CV_chunk(p)
```

判定：
- `gate_chunk ≥ 0.10` 且 `gate_len ≥ 0.05` → 分块边界未被字节数锁定，C1 可用；
- `gate_chunk < 0.05` 且 `gate_len > 0.2` → 定长重打包，C1 失效；
- 其余 → 不可判，加 prompt 或加模型再测。

**为什么阈值可以改**：原阈值是为"跨 prompt 中位数"设计的，与修订后的统计量不同尺度，
不能沿用。修订后的阈值由**同一批 9 条数据的事后分布**给出量级（gate_chunk≈0.10–0.29、
gate_len≈0–0.10），因此设 `0.10 / 0.05` 是**宽松**取法，倾向于"判为可用"——
这意味着若复验仍不通过，"网关重打包"的结论是稳的，不会被我的阈值选择质疑。

## 2 缺陷二：`first_chunk_bytes` 泄漏模型名

**问题**：首块 252 B vs 258 B，差值 ≈ 回显模型名的长度差。该特征进 C1 等于用模型名字符串长度分类。

**修订**：从 C1 移除 `first_chunk_bytes`。C1 变为：

```
C1 = [content_chunk_count, chunk_bytes_mean, chunk_bytes_cv, sse_overhead_ratio]
```

`chunk_bytes_mean` 与 `sse_overhead_ratio` 高度共线（封装 0.96–0.99，主要由块数决定），
故**统计上只用 `content_chunk_count` 与 `chunk_bytes_cv` 两个自由度**，另两个作描述性报告，
不计入多重比较。

## 3 新增：`usage` 可用性是模型属性

> **更正（2026-11-18 22:1x，GATE-2 实测后）**：本条**被证伪**。GATE-2 加上
> `stream_options.include_usage=true` 后，`qwen3.5-27b` 正常回传 usage，回传率 9/9。
> 因此 usage 缺失是**请求参数**问题，不是模型属性。本条的"模型属性"表述作废，
> 但"缺失不补 0、单列报告"的记账规则保留。

**修订**：新增字段 `usage_available`（bool）。规则：
- 缺失 usage **不视为 invalid**，也不补 0；
- 元数据基线（§5 第 4 项）只在 `usage_available==True` 的子集上计算，并显式报告子集大小；
- 若某模型 usage 全缺，该模型的"元数据基线"记 `not_computable`。

## 4 新增：`prompt_tokens` 异常低的成因未知

实测 3 条长度差异明显的 prompt 都是 25–28 token，疑似平台注入固定前缀。
**修订**：在 §5 基线表中新增一行 `platform_prefix_suspected`（无法核验，仅记录），
并在报告中声明"`prompt_tokens` 的语义未经核验，不得用于推断 prompt 长度"。

## 5 复验方案（批准后执行）

- 12 请求：4 模型 × 3 prompt，其中新增 1 个跨 provider 同模型
  （`deepseek-v4-pro` vs `vanchin/deepseek-v4-pro`）作为 D2 的**可行性预检**。
- 用修订后的判据计算 `gate_chunk` / `gate_len` 并出正式 G4 判定。
- 成本：约 12 × 220 token，**低于 ¥0.02**。

## 6 未改动

§6 统计方法、§9 其余闸门、§11 TEE：逐字未改。
