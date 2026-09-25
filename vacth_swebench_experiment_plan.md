# VACTH 多智能体 SWE-bench 实验扩展计划

版本：v0.1  
目标：从当前 13-instance progress set 扩展到可投稿、可复现、可解释的实验体系  
当前基础结果：`vacth_optimized_iter14b` 在 13 个 SWE-bench Lite progress 样本上达到 11/13 resolved，与最强非 VACTH baseline `summary` 持平，但 tokens/resolved 降低 75.2%，rework 降低 88.5%，cost/resolved 降低 70.2%。

---

## 1. 当前实验结果的定位

当前 13 个样本的结果非常适合作为 development/progress result，用来说明 VACTH 的优化方向是有希望的。但是它还不适合作为最终论文主结果，原因有三点：

1. 样本量较小，13 个样本中每多解一个样本就会带来 7.7 个百分点的 resolved rate 波动，因此 resolved rate 上不能过度宣称显著提升。
2. `vacth_optimized_iter14b` 是 iter13 结果与 iter14b 部分替换结果的组合，如果替换规则是在看到 official resolved/unresolved 结果后形成的，那么它更适合作为开发集结果，而不是严格的 frozen evaluation result。
3. `vacth_optimized_iter14b` 当前沿用了 `vacth_full` 的 mechanism-suite score，说明它属于同一 VACTH 机制家族，但还不能直接宣称 optimized runner 本身也取得了 0.9454 的 mechanism overall，需要单独复跑机制评测。

因此，当前结果最稳妥的论文表述应是：

> On a 13-instance SWE-bench Lite progress set, VACTH-Optimized matches the strongest non-VACTH baseline in resolved rate while substantially reducing token usage, rework, and cost. This preliminary result motivates a larger frozen-protocol evaluation.

中文表述可以是：

> 在 13 个 SWE-bench Lite progress 样本上，VACTH-Optimized 在 resolved 数量上追平最强非 VACTH baseline，但显著降低了通信 token、返工次数和成本。这说明 VACTH 当前最强的证据不是“解题数量显著更高”，而是在不损失成功率的情况下大幅压缩冗余通信和错误返工。

---

## 2. 实验总目标

后续实验需要回答四个核心问题。

### G1：端到端效果

验证 VACTH-Optimized 是否能在更大的 SWE-bench Lite / Verified 子集上保持与强 baseline 相同或更高的 resolved rate。

### G2：效率收益

验证 VACTH-Optimized 的 token、cost、rework 优势是否在更大样本上稳定存在，而不是 13 个样本的偶然结果。

### G3：机制归因

拆分 VACTH 机制与 compact/hybrid runner 工程优化的贡献，证明收益不是简单来自更短 prompt、更少轮数或更激进的 stopping rule。

### G4：错误传播控制

证明 VACTH 的 typed handoff、state item selection、provenance-aware aggregation 和 targeted re-ask 能减少 false fact propagation、stale tool state usage、missing constraint error 和 wrong-branch repair。

---

## 3. 实验问题设计

### RQ1：VACTH-Optimized 是否能在更大 SWE-bench 子集上保持当前效率优势？

比较对象：

- `summary`
- `autogen_broadcast`
- `compact_summary` 或 `optimized_non_vacth`
- `vacth_optimized`

主要指标：

- resolved rate
- tokens / instance
- tokens / resolved
- cost / instance
- cost / resolved
- rework
- patch apply rate
- official unresolved failure type

预期结论：

> VACTH-Optimized 在 resolved rate 不下降的前提下，显著降低 tokens/resolved、cost/resolved 和 rework。

---

### RQ2：VACTH 的收益来自机制本身，还是来自 compact/hybrid runner？

这是最关键的消融问题。需要补一个非 VACTH 的 compact baseline。

建议 2x2 实验矩阵：

| Setting | VACTH typed handoff | Compact / optimized runner | Gate / fallback | 目的 |
|---|---:|---:|---:|---|
| `summary` | 否 | 否 | 否 | 最强非 VACTH baseline |
| `compact_summary` / `optimized_non_vacth` | 否 | 是 | 是/否 | 测 runner 优化本身的作用 |
| `vacth_full` | 是 | 否 | 否 | 测朴素 VACTH 机制 |
| `vacth_optimized` | 是 | 是 | 是 | 完整方法 |

关键判断：

- 如果 `compact_summary` 已经接近 `vacth_optimized`，说明主要收益来自 runner 工程优化。
- 如果 `vacth_optimized` 明显优于 `compact_summary`，说明 VACTH typed state handoff 对端到端任务确实有贡献。
- 如果 `vacth_full` mechanism score 高但 resolved rate 低，说明高保真 handoff 还需要 compact/action-oriented rendering 才能转化为软件修复能力。

---

### RQ3：VACTH-Optimized 是否仍然保持 VACTH 机制保真？

当前 `vacth_optimized_iter14b` 的 mechanism overall 使用的是 `vacth_full` 的 0.9454。下一步必须单独复跑：

```text
vacth_optimized on mechanism suite
```

机制指标建议包括：

- Fact Retention
- Constraint Retention
- Hypothesis Calibration
- Tool-State Freshness
- Evidence Attribution Accuracy
- Contradiction Detection Accuracy
- Invalidated Item Accuracy
- Useful Re-ask Rate

可能出现两种结果：

1. 如果 `vacth_optimized` 的 mechanism score 仍然接近 `vacth_full`，说明 compact optimization 没有牺牲 typed state fidelity。
2. 如果 `vacth_optimized` 的 mechanism score 明显下降，但 SWE-bench resolved 和 cost 更好，说明当前 mechanism suite 和 executable repair workflow 之间还存在 gap，需要重新调整 mechanism suite 的权重，使它更贴近软件工程任务。

---

### RQ4：VACTH 是否减少错误传播和返工？

仅看 resolved rate 不足以证明 VACTH 的机制价值。因此需要增加 failure/cascade 诊断。

建议统计：

- false fact propagation：未验证假设被下游 Agent 当成事实继续使用的次数。
- missing constraint error：早期约束在后续修复中丢失，导致错误 patch 的次数。
- stale tool state usage：使用过期测试结果或过期错误日志指导下一步修复的次数。
- wrong-branch repair：基于错误根因假设进行无效修改的次数。
- repeated localization：重复定位同一文件/同一函数的次数。
- rework count：重复 patch、撤销 patch、修复失败后重新定位的次数。

当前结果里 `summary` 的 rework=26，而 `vacth_optimized_iter14b` 的 rework=3，这是非常强的信号。后续需要通过 per-instance log 证明这种 rework 降低确实来自更清晰的 state handoff，而不是单纯少跑了轮数。

---

## 4. 数据集与评测层级

### 4.1 当前 progress set

当前已经完成：

```text
N = 13
Methods = autogen_broadcast, sliding_window, structured_summary, summary, vector_memory, vacth_full, vacth_optimized_iter14b
Evaluation = SWE-bench Lite official corrected reports
```

用途：

- development result
- sanity check
- per-instance failure analysis
- 确定下一轮 baseline 和消融设计

不建议用途：

- 不作为最终主结果
- 不用于宣称 statistically significant resolve-rate gain
- 不用于后验选择最终方法后直接投稿

---

### 4.2 SWE-bench Lite frozen subset

建议下一步从 SWE-bench Lite 中选一个新的 frozen subset：

```text
N = 30 或 50
```

选择原则：

- 不包含当前 13 个 progress instances。
- 在运行前固定 instance list。
- 方法、prompt、gate rule、selection rule 在运行前冻结。
- 所有方法使用相同模型、相同最大轮数、相同工具、相同文件定位设置。

用途：

- 第一个正式扩展实验
- 验证当前 13 个样本上的效率优势是否可迁移
- 控制实验成本

---

### 4.3 SWE-bench Lite larger subset

在 N=30/50 结果稳定后，再扩展到：

```text
N = 100
```

建议只跑最关键方法：

- `summary`
- `compact_summary` / `optimized_non_vacth`
- `vacth_optimized`
- 可选：`autogen_broadcast`

用途：

- 作为论文主表的主要 SWE-bench Lite 结果
- 观察 resolved rate、token、cost、rework 的稳定性

---

### 4.4 SWE-bench Verified Mini / Verified subset

SWE-bench Verified 是人工过滤的 500-instance 子集，问题描述、测试补丁和可解性经过人工审查。Verified Mini 是 50 个 Verified 样本的轻量子集，适合作为成本受控的补充评测。

建议顺序：

```text
Stage 1: Verified Mini 50
Stage 2: Verified subset 100, if budget allows
```

用途：

- 验证方法在更可靠人工筛选样本上的稳定性
- 避免只在 SWE-bench Lite 上调参造成过拟合
- 作为论文中的 external validation

---

### 4.5 Mechanism Suite / State Handoff Suite

机制评测不需要很大，但需要标注精确。建议构造或扩充一个 StateHandoffBench：

```text
N = 100–300 handoff samples
```

每个样本包含：

- facts
- constraints
- hypotheses
- tool states
- artifact references
- conflicts
- stale states
- duplicate claims
- unresolved issues
- expected re-ask target

用途：

- 证明 VACTH 不是普通 summary
- 量化 state fidelity
- 支撑 mechanism overall
- 解释 SWE-bench 中 rework 降低的原因

---

## 5. 方法与 baseline 设计

### 5.1 已有方法

当前已跑：

| Method | 说明 | 后续是否保留 |
|---|---|---|
| `autogen_broadcast` | 完整上下文广播 | 保留，作为高 token 上界 |
| `sliding_window` | 最近窗口记忆 | 可保留，但后续可以减少运行规模 |
| `structured_summary` | 结构化摘要 | 保留，证明结构化本身不等于 VACTH |
| `summary` | 普通摘要，当前最强非 VACTH baseline | 必须保留 |
| `vector_memory` | 检索式记忆 | 可在小规模保留，大规模可选 |
| `vacth_full` | 原始完整 VACTH | 保留用于 ablation，不一定大规模跑 |
| `vacth_optimized_iter14b` | 当前最优开发版本 | 需要改名并冻结为正式方法 |

---

### 5.2 必须新增的方法

#### 5.2.1 `compact_summary` / `optimized_non_vacth`

目的：隔离 compact/hybrid runner 的工程收益。

实现：

- 使用与 `vacth_optimized` 相同的 compact runner、gate/fallback、最大轮数和 stopping rule。
- 不使用 VACTH typed state item、PAA、provenance graph、targeted re-ask。
- 上下文管理使用普通 summary 或 role-specific summary。

这是后续最重要的 baseline。

---

#### 5.2.2 `vacth_compact_no_gate`

目的：判断 gate/fallback 对结果的贡献。

实现：

- 使用 VACTH compact rendering。
- 不使用跨版本 patch replacement 或 gate fallback。
- 每个 instance 只输出一个固定流程生成的 patch。

---

#### 5.2.3 `vacth_no_paa`

目的：证明 provenance-aware aggregation 的价值。

实现：

- 保留 typed capsule 和 CVE selection。
- 不维护全局 state graph。
- 每次 handoff 独立渲染。

关注指标：

- conflict handling
- stale tool state usage
- duplicate claim rate
- rework

---

#### 5.2.4 `vacth_no_reask`

目的：证明 targeted re-ask 的价值。

实现：

- 保留 VACTH state graph。
- 关闭 unresolved issue 触发的 targeted re-ask。

关注指标：

- useful re-ask rate
- unresolved issue resolution rate
- wrong-branch repair

---

#### 5.2.5 `vacth_no_type`

目的：证明 epistemic typing 的价值。

实现：

- 使用 compact state items。
- 不区分 fact / hypothesis / constraint / decision / tool_state。
- 或者把所有 state item 渲染成普通 bullet summary。

关注指标：

- hypothesis calibration
- false fact propagation
- constraint retention

---

## 6. 指标体系

### 6.1 端到端指标

| 指标 | 定义 | 说明 |
|---|---|---|
| Resolved | official SWE-bench resolved 数量 | 主指标 |
| Resolved % | Resolved / N | 样本小的时候同时报告分子分母 |
| Patch apply rate | patch 是否能被 harness 成功应用 | 区分格式失败和能力失败 |
| FAIL_TO_PASS pass rate | 目标失败测试是否被修复 | 判断是否解决 issue |
| PASS_TO_PASS preservation | 原本通过测试是否保持通过 | 判断是否引入 regression |
| Average repair rounds | 平均修复轮次 | 反映返工和执行效率 |

---

### 6.2 效率指标

| 指标 | 定义 | 说明 |
|---|---|---|
| Total tokens | 总 token | 包含 input/output，建议同时拆分 |
| Tokens / instance | Total tokens / N | 避免 tokens/resolved 因 resolved 数不同而偏移 |
| Tokens / resolved | Total tokens / resolved | 表示解决一个任务的 token 成本 |
| Cost | 总 API 成本 | 用统一价格计算 |
| Cost / instance | Cost / N | 成本基础指标 |
| Cost / resolved | Cost / resolved | 解决一个任务的美元成本 |
| LLM calls | 调用次数 | 区分少 token 和少调用 |
| Tool calls | 工具调用次数 | 反映执行复杂度 |
| Wall-clock time | 端到端耗时 | 如果并行评测，需注明 |

---

### 6.3 返工与错误传播指标

| 指标 | 定义 |
|---|---|
| Rework | 需要撤销、重复定位、重复 patch 或再次修复的次数 |
| Wrong-branch repair | 基于错误根因假设进行的无效修改次数 |
| Repeated localization | 重复搜索/阅读同一文件但没有新增信息的次数 |
| False fact propagation | 未验证假设被下游当成事实继续使用的次数 |
| Missing constraint error | 早期约束丢失导致错误修改的次数 |
| Stale tool state usage | 使用过期测试结果或过期日志指导行动的次数 |
| Cascade error length | 一个错误 state item 影响后续多少轮/多少 Agent |

---

### 6.4 机制指标

| 指标 | 定义 |
|---|---|
| Fact Retention | 关键事实在 handoff 后是否保留 |
| Constraint Retention | 高优先级约束是否保留 |
| Hypothesis Calibration | hypothesis 是否保持为 hypothesis，而不是被升级为 fact |
| Tool-State Freshness | 下游看到的测试/工具状态是否最新 |
| Evidence Attribution Accuracy | state item 是否能追溯到正确代码片段、日志或测试输出 |
| Contradiction Detection Accuracy | 冲突 state item 是否被识别 |
| Invalidated Item Accuracy | 过期/被推翻信息是否被 invalidated |
| Useful Re-ask Rate | 触发的 re-ask 是否解决 blocking uncertainty |

---

## 7. 实验阶段规划

### Phase 0：当前 13-instance 结果整理与复核

目标：把已有结果整理成可复现的 development result。

任务：

1. 固定当前 13 个 instance list。
2. 保存所有方法的 predictions.jsonl、patch、official report、run log、agent transcript。
3. 做 per-instance breakdown。
4. 对 `psf__requests-1963` 和 `astropy__astropy-7746` 做失败分析。
5. 明确 `vacth_optimized_iter14b` 的组合规则是否使用 official result。如果使用了，则标记为 dev-selected result。

输出：

- `results/progress13/main_table.md`
- `results/progress13/per_instance.csv`
- `results/progress13/failure_analysis.md`
- `results/progress13/patches/`
- `results/progress13/logs/`

优先级：最高。

---

### Phase 1：机制评测复跑

目标：确认 optimized runner 是否仍然保持 VACTH 机制优势。

实验：

| Method | Mechanism Suite |
|---|---:|
| `structured_summary` | rerun or keep existing |
| `vacth_full` | existing 0.9454 |
| `vacth_optimized` | must rerun |
| `vacth_no_paa` | optional |
| `vacth_no_reask` | optional |

重点指标：

- mechanism overall
- fact retention
- constraint retention
- hypothesis calibration
- evidence attribution
- useful re-ask rate

输出：

- `results/mechanism_suite/main_table.md`
- `results/mechanism_suite/error_cases.md`

优先级：最高。

---

### Phase 2：2x2 归因消融

目标：回答“收益来自 VACTH 机制还是 compact runner”。

数据：

```text
progress13 + new small frozen subset, N = 20–30
```

方法：

| Method | VACTH | Compact Runner | Gate |
|---|---:|---:|---:|
| `summary` | ✗ | ✗ | ✗ |
| `compact_summary` | ✗ | ✓ | ✓/✗ |
| `vacth_full` | ✓ | ✗ | ✗ |
| `vacth_optimized` | ✓ | ✓ | ✓ |

指标：

- resolved
- tokens / instance
- tokens / resolved
- cost / resolved
- rework
- mechanism score, if available

输出：

- `results/ablation_2x2/main_table.md`
- `results/ablation_2x2/interpretation.md`

优先级：最高。

---

### Phase 3：Frozen subset 扩展实验

目标：在新样本上验证结果稳定性。

数据：

```text
SWE-bench Lite frozen subset, N = 30 或 50
```

方法：

- `summary`
- `compact_summary`
- `vacth_optimized`
- `autogen_broadcast`, if budget allows

实验要求：

1. 运行前固定 method config。
2. 不允许根据 official result 替换 patch。
3. 所有方法统一模型、temperature、max iterations、tool limit、time limit。
4. 每个 method 输出完整 predictions.jsonl。
5. 使用 official corrected reports。

输出：

- `results/frozen50/main_table.md`
- `results/frozen50/per_instance.csv`
- `results/frozen50/official_reports/`
- `results/frozen50/failure_analysis.md`

优先级：高。

---

### Phase 4：SWE-bench Lite 100-instance 主实验

目标：形成论文主表。

数据：

```text
SWE-bench Lite subset, N = 100
```

方法：

- `summary`
- `compact_summary`
- `vacth_optimized`
- 可选：`autogen_broadcast`
- 可选：`vacth_no_paa` 或 `vacth_no_reask`, if budget allows

主表建议：

| Method | N | Resolved | Resolved % | Tokens / inst. | Tokens / resolved | Rework | Cost / inst. | Cost / resolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

输出：

- `results/lite100/main_table.md`
- `results/lite100/per_instance.csv`
- `results/lite100/paired_analysis.md`

优先级：高。

---

### Phase 5：Verified Mini / Verified subset 外部验证

目标：证明方法不只对 SWE-bench Lite progress/frozen set 有效。

数据：

```text
SWE-bench Verified Mini 50
或 Verified subset 50–100
```

方法：

- `summary`
- `compact_summary`
- `vacth_optimized`

重点：

- 不需要跑所有历史 baseline。
- 重点验证 efficiency gain 和 resolved rate 不下降。
- 如果 Verified 上 resolved rate 下降，也要分析是否因为任务更复杂导致 VACTH state selection 丢失信息。

输出：

- `results/verified50/main_table.md`
- `results/verified50/failure_analysis.md`

优先级：中高。

---

### Phase 6：扰动与鲁棒性实验

目标：直接证明 VACTH 减少错误传播。

数据：

```text
StateHandoffBench 或 SWE-bench trajectories with perturbations
```

扰动类型：

1. false hypothesis：上游给出错误根因假设。
2. stale test state：旧测试结果与新测试结果冲突。
3. missing constraint：关键约束只在早期出现一次。
4. corrupted evidence pointer：claim 指向错误文件或错误日志。
5. duplicate irrelevant claims：重复背景信息占据上下文。
6. conflicting facts：两个 Agent 给出互相矛盾的事实。

方法：

- `summary`
- `structured_summary`
- `compact_summary`
- `vacth_optimized`
- `vacth_no_paa`
- `vacth_no_type`

指标：

- false fact propagation
- constraint retention
- contradiction detection
- stale state usage
- useful re-ask rate
- downstream decision correctness

输出：

- `results/robustness/main_table.md`
- `results/robustness/case_studies.md`

优先级：中高。

---

## 8. 统计分析方案

### 8.1 resolved rate

由于每个方法都跑同一批 instance，应该尽量做 paired analysis，而不是只比较 aggregate percentage。

建议报告：

- resolved count and percentage
- per-instance win/loss/tie matrix
- McNemar test, if sample size sufficient
- Wilson confidence interval, especially for N=13/30/50

注意：

- N=13 时不做显著性宣称。
- N=50 以后可以开始报告置信区间。
- N=100 更适合作为主结果。

---

### 8.2 token / cost / rework

这些指标应使用 paired bootstrap 或 per-instance paired difference。

建议报告：

- mean tokens / instance
- median tokens / instance
- mean cost / instance
- median cost / instance
- tokens / resolved
- cost / resolved
- rework mean and total
- bootstrap confidence interval for reduction rate

重点不要只报 tokens/resolved，因为它会受 resolved 数量影响。必须同时报告 tokens/instance 和 cost/instance。

---

### 8.3 机制指标

机制指标建议报告：

- overall score
- sub-metric score
- per-category score: fact / constraint / hypothesis / tool_state / evidence / conflict / re-ask
- confusion cases

对于 mechanism suite，可以更强调 precision/recall/F1，而不是只给一个总分。

---

## 9. 结果表设计

### 表 1：当前 progress set 结果

用于展示当前进展，但标题应写清楚：

```text
Preliminary Results on the 13-Instance SWE-bench Lite Progress Set
```

建议列：

| Method | N | Resolved | Resolved % | Tokens | Tokens / inst. | Tokens / resolved | Rework | Cost | Cost / inst. | Cost / resolved | Mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

---

### 表 2：相对最强 baseline 的效率提升

| Comparison | Resolved change | Tokens / resolved reduction | Cost / resolved reduction | Rework reduction |
|---|---:|---:|---:|---:|
| VACTH-Opt vs Summary | 0 | 75.2% | 70.2% | 88.5% |
| VACTH-Opt vs Broadcast | +1 | 79.5% | 75.1% | 87.0% |
| VACTH-Opt vs VACTH-Full | +3 | 82.8% | 79.3% | 87.5% |

---

### 表 3：2x2 消融

| Method | VACTH typed state | Compact runner | Gate | Resolved | Tokens / inst. | Rework | Cost / resolved |
|---|---:|---:|---:|---:|---:|---:|---:|

---

### 表 4：机制评测

| Method | Overall | Fact Ret. | Constraint Ret. | Hyp. Calib. | Tool Fresh. | Evidence Attr. | Re-ask Useful |
|---|---:|---:|---:|---:|---:|---:|---:|

---

### 表 5：错误传播与鲁棒性

| Method | False Fact Prop. | Missing Constraint | Stale Tool State | Wrong Branch | Rework | Useful Re-ask |
|---|---:|---:|---:|---:|---:|---:|

---

## 10. Per-instance 分析模板

每个 instance 建议记录：

```text
instance_id
repo
method
resolved
patch_apply_success
fail_to_pass_passed
pass_to_pass_passed
total_tokens
input_tokens
output_tokens
cost
llm_calls
tool_calls
rework_count
repair_rounds
localization_success
main_modified_files
failure_type
notes
```

推荐 failure_type 分类：

```text
FORMAT_ERROR
PATCH_APPLY_FAILED
LOCALIZATION_FAILED
WRONG_ROOT_CAUSE
INCOMPLETE_PATCH
REGRESSION
TEST_TIMEOUT
ENV_ERROR
OVER_COMPRESSION
STALE_STATE
MISSING_CONSTRAINT
UNKNOWN
```

---

## 11. 运行与日志规范

每次实验必须保存：

```text
config.yaml
instance_ids.txt
predictions.jsonl
official_report.json
summary_metrics.json
per_instance_metrics.csv
patches/{instance_id}.patch
logs/{instance_id}/agent_transcript.jsonl
logs/{instance_id}/tool_calls.jsonl
logs/{instance_id}/state_graph.json
logs/{instance_id}/handoff_capsules.jsonl
logs/{instance_id}/reask_events.jsonl
```

每次 run 都必须有唯一 run_id，例如：

```text
vacth_opt_lite50_frozen_2026_05_23_r1
summary_lite50_frozen_2026_05_23_r1
compact_summary_lite50_frozen_2026_05_23_r1
```

---

## 12. 方法冻结规则

为了避免后验选择，正式评测前必须冻结：

1. prompt
2. model name and version
3. temperature
4. max turns
5. max tool calls
6. max repair rounds
7. token budget
8. gate / fallback rule
9. candidate patch selection rule
10. timeout setting
11. dataset instance list

禁止事项：

- 看到 official result 后替换某个 instance 的 patch。
- 对不同方法使用不同最大轮数或不同工具权限。
- 对某个方法手工修 patch。
- 使用 official hidden result 作为 gate 的选择信号。

允许事项：

- 使用 internal validation tests 选择 candidate patch，但规则必须提前固定。
- 使用同一套 runner 对所有方法评测。
- 在 progress set 上调参，但不能把 progress set 当 final test。

---

## 13. 推荐时间安排

### 第 1 周：结果整理与机制复跑

- 完成 progress13 per-instance breakdown。
- 分析两个 unresolved cases。
- 复跑 `vacth_optimized` mechanism suite。
- 固定正式方法名，例如 `VACTH-Compact` 或 `VACTH-Gated`。

### 第 2 周：补关键消融

- 实现并跑 `compact_summary` / `optimized_non_vacth`。
- 跑 2x2 消融。
- 做机制指标与 SWE-bench 指标的相关性分析。

### 第 3–4 周：Lite frozen subset

- 选定新的 N=30 或 N=50 frozen subset。
- 跑 `summary`、`compact_summary`、`vacth_optimized`。
- 如果预算允许，补 `broadcast`。
- 完成 paired analysis。

### 第 5–6 周：Lite 100 或 Verified Mini

- 跑 Lite 100 主实验，或 Verified Mini 50 外部验证。
- 汇总 cost、token、rework、resolved。
- 完成 case study。

### 第 7 周以后：鲁棒性与投稿版整理

- 完成 perturbation / StateHandoffBench。
- 完成最终实验表。
- 写实验章节和 related work 对比。

---

## 14. 投稿前最低证据链

如果要形成一篇比较稳的论文，最低需要以下证据：

1. `vacth_optimized` 在机制 suite 上有独立结果，而不是沿用 `vacth_full`。
2. 至少一个非 VACTH compact baseline，用来隔离 runner 优化。
3. 至少一个 frozen held-out SWE-bench subset，不使用后验 patch replacement。
4. per-instance breakdown，证明效率收益不是某几个 instance 偶然造成。
5. 错误传播或 rework 分析，证明 VACTH 的机制确实减少了 wrong-branch repair。
6. 明确报告 tokens/instance、cost/instance，而不仅是 tokens/resolved。
7. 对所有方法使用相同 model、tool、budget、最大轮数和 official harness。

---

## 15. 最终论文 claim 建议

当前阶段不建议写：

> VACTH significantly improves SWE-bench Lite resolved rate.

更建议写：

> VACTH improves the efficiency and reliability of multi-agent software-engineering collaboration by replacing free-form message passing with typed, provenance-aware state handoff. On the SWE-bench Lite progress set, VACTH-Optimized matches the strongest summary baseline in resolved rate while reducing tokens per resolved instance by 75.2%, cost per resolved instance by 70.2%, and rework by 88.5%. Larger frozen-subset evaluation and mechanism-suite analysis are used to validate whether these efficiency gains generalize beyond the progress set.

中文对应：

> VACTH 的主要贡献不是简单提高 resolved rate，而是在保持相同任务成功率的同时，显著减少多智能体协作中的冗余通信、错误分支和返工成本。后续实验需要通过 frozen subset、机制评测和消融实验，证明这种收益来自 typed state handoff，而不是单纯 runner 工程优化。

---

## 16. 附：当前 progress result 推荐表述

当前结果可以作为 preliminary table：

| Method | N | Resolved | Resolved % | Corrected tokens | Tokens / resolved | Rework | Cost ($) | Cost / resolved ($) | Mechanism overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| autogen_broadcast | 13 | 10 | 76.9 | 4,010,067 | 401,006.70 | 23 | 10.524668 | 1.052467 | - |
| sliding_window | 13 | 8 | 61.5 | 3,014,634 | 376,829.25 | 23 | 8.139985 | 1.017498 | - |
| structured_summary | 13 | 10 | 76.9 | 4,380,539 | 438,053.90 | 23 | 11.493187 | 1.149319 | 0.2911 |
| summary | 13 | 11 | 84.6 | 3,646,955 | 331,541.36 | 26 | 9.670676 | 0.879152 | 0.0217 |
| vector_memory | 13 | 9 | 69.2 | 4,720,739 | 524,526.56 | 23 | 12.300599 | 1.366733 | - |
| vacth_full | 13 | 8 | 61.5 | 3,825,668 | 478,208.50 | 24 | 10.150421 | 1.268803 | 0.9454 |
| vacth_optimized_iter14b | 13 | 11 | 84.6 | 904,519 | 82,229.00 | 3 | 2.885024 | 0.262275 | inherited 0.9454 |

注意：最后一行 mechanism overall 应标注为 inherited / not separately rerun，而不是正式 measured score。

---

## 17. 参考资料

- SWE-bench Lite official page: https://www.swebench.com/lite.html
- SWE-bench Verified official page: https://www.swebench.com/verified.html
- SWE-bench GitHub and harness: https://github.com/swe-bench/SWE-bench
- SWE-bench FAQ: https://www.swebench.com/SWE-bench/faq/
- SWE-bench Verified Mini: https://hal.cs.princeton.edu/swebench_verified_mini
