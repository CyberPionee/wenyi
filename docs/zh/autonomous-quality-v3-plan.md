# 自治质量 v3 执行计划

[方案设计](autonomous-quality.md) · 分支 `autonomous-quality-v3`

目标：在 `feat/quality-autonomy-v2` 之上实现最小验收套件与自治修复闭环；人最少干预；需要界面的用 Web 呈现。不在此计划内部署 Docker。

## 范围

**做：**

1. 评估报告（L0–L3）与机器验收门
2. 风险门控回译（定向，不整本）
3. C-batch / 回译低分入 Autofix 候选链
4. 可选译文决策锚
5. 红线失败自动局部重做的调度入口（评估未过 → 重做队列）
6. Web：评估报告、风险段、决策锚、验收状态

**不做：**

- Docker 部署/构建验证
- 固定集 A/B
- 整本回译默认路径
- 人工逐项验收 UI 门禁
- BabelDOC / SRT 专项

## 模块落点

| 能力 | 主要落点 | 说明 |
|---|---|---|
| L0 已有 | `assemble/report.py`、`review/sweep.py` | 扩展汇总字段，不重写规则 |
| L1 风险选择 | `pipeline/evaluation.py`（新） | 选段：对话/术语/长句/sweep/低置信 |
| L1 回译 | `agents/quality_pass.py` | 复用 back_translation，加风险选段 |
| L3 judge | `agents/quality_judge.py`（新） | 固定 rubric，通顺/文风 1–5 |
| 机器门 | `pipeline/evaluation.py` | L0–L3 阈值 → passed/blocking |
| 决策锚 | `pipeline/decision_anchors.py`（新） | 从早期定稿译文凝锚，写入 analysis |
| 锚注入 | `pipeline/translation_batch.py`、`agents/translator.py` | 与 style 并列 |
| C-batch→Autofix | `pipeline/autofix_candidates.py` | quality_pass / 回译低分映射候选 |
| 报告产出 | `pipeline/finalization.py`、`assemble/report.py` | evaluation 块 + 验收结论 |
| API | `apps/api/wenyi_api/` | evaluation / anchors 读接口 |
| Web | `apps/web/src/features/evaluation/` | 报告页、风险段、锚、验收 |

## 实现切片

### 切片 1 · 评估核心（P0）

1. `config.py` 增加评估策略字段（阈值、是否开 L3、回译风险比例等），同步默认 YAML 与测试。
2. `pipeline/evaluation.py`：
   - `select_risk_segments(...)`：规则选高风险/抽样段
   - `run_back_translation_checks(...)`：仅选中段回译并打分
   - `run_quality_judge(...)`：抽样 L3 评分
   - `build_machine_gate(...)`：L0–L3 → `{passed, blocking, counts, items}`
3. `assemble/report.py` 合并 `evaluation` 与 `auto_qa`，扩展 `build_report`。
4. `finalization.py` 导出前写 evaluation 事件；`auto_qa_strict` 或评估 blocking 时失败导出。

### 切片 2 · 风险回译 + C-batch 入链（P2）

1. 评估产生的低分段、C-batch revision 差异段映射为 Autofix 候选（复用 `autofix_candidates.py` 契约）。
2. 不新增 Autofix 专属 prompt；仍走 fix → verify → publish。
3. 空译文仍只进 auto_qa，不自动填。

### 切片 3 · 可选译文决策锚（P1）

1. `pipeline/decision_anchors.py`：从已完成段落凝「必用/禁用/样例」短表。
2. 配置 `pipeline.decision_anchors: off|auto|risk`。
3. 翻译批次 prompt 注入锚（可为空）。
4. Web 风格/质量区展示锚（只读 + 手动清空可选）。

### 切片 4 · Web 与 API

1. 共享 schema：`EvaluationReport`、`DecisionAnchor`、`RiskSegment`。
2. API：`GET .../report` 已有则扩 evaluation；`GET .../decision-anchors`。
3. 页面：
   - 进度页：机器验收门状态（L0–L3 计数与 passed/blocking）
   - 评估页或风格页分区：风险段列表、回译低分、L3 分
   - 决策锚卡片
4. i18n：en + zh-CN。

### 切片 5 · 验证

1. 单测：选段、机器门、锚凝练、候选映射、report 合并。
2. `ruff check/format`、`pytest` 受影响模块、`pnpm typecheck`。
3. 不跑 Docker；Web 以 typecheck/build 为准。

## 配置草案

```yaml
pipeline:
  auto_qa_strict: true          # 自治·准默认（文档口径）
  evaluation:
    enabled: true
    risk_back_translation: true # L1 定向回译
    risk_sample_ratio: 0.08     # 每章抽样上限
    quality_judge: true         # L3
    judge_sample_ratio: 0.05
    max_auto_redo_rounds: 2
  decision_anchors: off         # off | auto | risk
```

阈值字段以 `config.py` 实际模型为准；实现时与文档对齐并更新 configuration 文档。

## 验收（本计划交付）

1. 导出报告含 `evaluation` 与 `machine_gate`。
2. 风险回译只覆盖选中段，有日志与计数。
3. C-batch/回译低分可进入 Autofix 候选（有测试）。
4. 决策锚可选生成并进入翻译 prompt（有测试）。
5. Web 可见验收门与风险/锚信息；typecheck + build 通过。
6. 受影响 pytest 通过；无 Docker 步骤。

## 顺序

```text
T5 计划 → T6 分支 → T7 评估核心 → T8 回译/入链 → T9 决策锚 → T10 Web → T11 测试
```

跨切片共享：操作 ID 注册、事件、usage 记账沿用现有 `llm/operations.py` 与账本规则，评估调用计入对应 operation，不重复计费。
