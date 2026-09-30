# 通用灵活术语注入方案（设计）

[配置](configuration.md) · [翻译流程](pipeline.md)

目标：**同一套算法**适配短书与长书，控制术语抽取成本，并保持译名对齐质量。本文为设计口径，落地以实现与测试为准。

## 问题

| 现状 | 问题 |
|---|---|
| 抽取时注入「全书已有术语」 | 表越大每次提示词越贵 |
| 固定「前 N 条」 | 不相关词占位，后面的词反而进不去 |
| 章末再全量 LLM 抽 | 与批抽重复，成本接近翻倍 |
| 抽取提示词带 Note | 备注只服务译文阶段，浪费抽取预算 |

## 设计原则

1. **不按书长分支**：短书/长书共用打分与预算。
2. **相关度优先**：本批命中与核心实体优先于全库平铺。
3. **预算封顶**：成本有上界，不随术语表线性膨胀。
4. **兜底不裸奔**：命中为 0 或预算极小时仍保留最小对照。
5. **抽取瘦身**：抽取提示词无 Note；Note 仅用于翻译/润色/审校。

## 总流程

```text
每批译文落盘
    │
    ▼
智能注入集 = score + budget 装填
    │
    ▼
extract：本批原文+译文 + 注入集（无 Note）
    │
    ▼
入库 / 更新 / conflict / auto-lock
    │
章末（可选本地收尾）
    │
    ▼
finalize：补空译名 + auto-lock（默认不调 LLM）
```

翻译侧按章过滤 + Note 的逻辑**保持不变**。

## 打分模型

对已有术语表中的每条 `term`，在一次抽取请求前计算分数：

| 因子 | 判定 | 分值 | 说明 |
|---|---|---|---|
| `batch_hit` | `source`/`aliases` 出现在本批 `source_text` 或 `target_text` | +100 | 最强相关 |
| `core` | `type ∈ always-on`（默认 person）且全书出现 ≥ `core_min_occurrences` | +60 | 主角/核心实体 |
| `recent` | 入库位置属于最近 `recent_max` 条或最近 K 批 | +40 | 刚抽完、下一批常用 |
| `frequency` | 全书出现次数 | +min(count, 20) | 高频更需统一 |
| `type_bonus` | person/place/organization | +15 | 实体优先 |
| `type_low` | speech/fixed_expression 且非 hit | +0 | 易挤出 |
| `open_conflict` | 存在未决 conflict | 强制置顶 | 必须对照 |

同分按插入序（`rowid`）稳定排序，保证可复现。

伪代码：

```text
score(term) = 100*hit + 60*core + 40*recent + min(freq,20) + 15*entity_type
order = sort_by([-score, insert_index])
```

## 预算与装填

| 参数 | 默认 | 含义 |
|---|---|---|
| `glossary_extract_budget_chars` | 4000 | 注入集渲染后字符上限（约 1k–2k tokens） |
| `glossary_extract_core_max` | 12 | core 层最多条数 |
| `glossary_extract_recent_max` | 20 | recent 层最多条数 |
| `glossary_extract_min_terms` | 5 | 最少注入条数 |
| `glossary_extract_inject` | `smart` | `smart` \| `all` \| `hit_only` |

装填顺序：

1. **未决冲突** source 全放（若单项超预算则截断渲染，不丢 source）。  
2. **batch_hit** 按分数放入，直至预算将满。  
3. **core / recent** 补位（respect core_max / recent_max）。  
4. 若总数 &lt; `min_terms` 或命中为 0：从全表按分数补到 `min_terms`（复发词优先）。  
5. 预算写满即停；短书通常填不满，则**自然接近全量**。

### 模式

| 模式 | 行为 |
|---|---|
| `smart` | 上述默认 |
| `all` | 全量注入（调试/对照） |
| `hit_only` | 仅 batch_hit + 强制冲突，最省 |

## 渲染格式（仅抽取）

```text
- 田中 → 田中 (person)
- バー → 酒吧 (place) [Aliases: bar]
```

- **无 Note**。  
- aliases 合并同一行。  
- 类型保留。  
- 预算按渲染后字符串长度计。

翻译/润色/审校继续：`render_glossary(..., include_note=True, max_note_chars=glossary_note_chars)`。

## 章末收尾

为避免「批抽 + 章末全量 LLM」双倍成本：

| 步骤 | 是否 LLM | 内容 |
|---|---|---|
| 批抽取 | 是 | 智能注入 + extract |
| 章末 finalize | **否** | 用历史对齐填空译名；对已有译名做 auto-lock 门控 |

若需要章末补抽（地址/长程称呼），应单独开关 `glossary_chapter_reextract: false` 默认关，且仍使用智能注入。

## 配置草案

```yaml
pipeline:
  glossary_extract_inject: smart
  glossary_extract_budget_chars: 4000
  glossary_extract_core_max: 12
  glossary_extract_recent_max: 20
  glossary_extract_min_terms: 5
  # glossary_note_chars 仅作用于翻译/审校渲染
  glossary_note_chars: 120
```

实现时以 `config.py` 模型为准，并同步中英文 configuration 文档。

## 成本预期

| 场景 | 相对 main |
|---|---|
| 短书、术语少 | 预算未满 ≈ 全量，**持平或略省** |
| 长书、术语多 | 注入有上界，**明显更省** |
| 章末 | 去掉全量 LLM 重抽，**再省一截** |

主要对比基线是 **main**（全量复发词 + 每章重抽）。相对「死板前 80」可能略贵一点（相关词更多进预算），对齐质量更好。

## 验收标准

1. 短书：注入条数 ≈ 可用全表（未触顶预算）。  
2. 长书：渲染字符 ≤ `budget_chars`；本批命中词绝大多数在注入集内。  
3. `glossary.extract` 用量相对全文翻译 **不高于 main**；长书更低。  
4. 重复入库与 conflict 率不劣于 main。  
5. 单测：命中优先、core/recent 上限、预算截断、0 命中兜底、模式切换。  
6. 翻译提示词仍含 Note（回归锁定）。

## 实现切片

| 优先级 | 内容 | 改代码 |
|---|---|---|
| P1 | `select_extraction_terms(...)` 纯函数 + 单测 | 是 |
| P2 | 抽取 prompt 接入智能注入，去 Note | 是 |
| P3 | 章末本地 finalize（无 LLM） | 是（若尚未落地） |
| P4 | 配置项 + 中英文文档 | 是 |
| P5 | 事件记 `injected_count` / 估算字符 | 可选 |

**不做**：固定集 A/B、按书长开关、ML 重排、抽取侧保留 Note。

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 匹配漏导致旧词当新词 | recent 层 + core 层 + min_terms 兜底 |
| 预算过小 | 默认 4000 字符可配；min_terms 保底 |
| 结果不稳定 | 同分稳定排序；模式可退 `all` |
| 长书后期仍贵 | 优先收紧 budget / 改 `hit_only` |

## 与相关文档

- 术语判断标准：抽取提示词中的实体 / 称呼 / 固定表达三类。  
- Auto-lock、always-on：见 glossary 与 autonomous-quality 相关说明。  
- 质量验收：`autonomous-quality.md` 中 L2 一致性依赖术语表质量。
