# VACTHBench

VACTHBench is a controlled AutoGenBench harness for evaluating multi-agent
communication protocols under identical agents, tasks, tools, and budgets.

The benchmark is intentionally staged. E0 provides the shared experiment
scaffolding, schema, logging, and smoke tasks. Later experiment units add
AutoGen broadcast, summary, sliding-window, structured-summary, VACTH, and
ablation runtimes behind the same scenario interface.

## E0 Scope

The initial harness creates deterministic smoke tasks that do not require an
LLM. This keeps the benchmark runnable in a fresh environment while validating
the parts that every later method depends on:

- task JSONL generation
- template substitution
- unified experiment configuration
- unified result and log files
- custom tabulation
- result schema validation

## E1 Scope

E1 replaces the `software_repair_autogen_broadcast` smoke implementation with a
real AutoGen AgentChat baseline. The scenario now uses five
`AssistantAgent`s inside a `RoundRobinGroupChat`:

- `planner`: proposes the repair plan
- `retriever`: reads files, searches the workspace, and can run tests
- `coder`: applies the minimal source patch
- `tester`: runs the unittest suite and reports verification evidence
- `reviewer`: checks the final diff/tests and emits `FINAL_ANSWER: resolved`
  or `FINAL_ANSWER: unresolved`

The collaboration mode is native AutoGen broadcast. After each participant
produces its final `BaseChatMessage`, `RoundRobinGroupChat` publishes that
message to the shared group topic, so later agents receive the prior agents'
messages through their own model context. Tool call requests, tool execution
events, and final tool summaries are streamed and logged, and downstream agents
can see the final agent messages that summarize those tool results. E1 does not
use VACTH routing, capsules, structured summaries, or token-budget selection;
`state_items.json`, `routing_decisions.json`, and `provenance_graph.json` remain
empty for this baseline.

For auditability, E1 wraps each real model client with a recorder. Every
`create()` call stores the exact LLM-visible message list in
`visible_contexts.json`, while AutoGen stream outputs are written to
`raw_messages.json` and parsed tool activity is written to `tool_calls.json`.

## E2 Scope

E2 adds the `software_repair_summary` baseline. It keeps the same software
repair task, agent roles, model configuration, and tools as E1, but replaces
native broadcast with a free-text handoff summary:

1. one agent acts from the original task or current handoff summary;
2. `summary_runtime` updates a natural-language summary from that turn;
3. the next agent receives the original task plus the latest summary, not the
   full transcript.

This baseline intentionally avoids typed schemas, provenance edges, state-item
merge keys, and receiver-specific routing. It writes `summaries.json` alongside
the standard result files and records `summary_count`, `summary_tokens`, and
`summary_compression_ratio` in `metrics`.

## E3 Scope

E3 adds the `software_repair_sliding_window` baseline. It keeps the E1 native
round-robin group chat, the same agent roles, and the same tools, but assigns
each `AssistantAgent` a `BufferedChatCompletionContext`. The smoke
configuration uses `context_window_messages=4`, so each agent sends only the
latest four model-context messages to the LLM.

The recorder still captures the exact LLM-visible messages in
`visible_contexts.json`. For auditability, sliding-window context records also
include a `window_policy` object and an estimated `dropped_messages` list.
Metrics include `context_window_messages`, `dropped_message_count`, and
`max_visible_context_messages`.

## E4 Scope

E4 adds the `software_repair_vector_memory` baseline. It keeps the same five
software-repair agents, tools, model configuration, and task as E1-E3, but
replaces broadcast transcript sharing with a lightweight retrieval memory:

1. before each agent acts, the runtime builds a query from the original task,
   receiver role, and previous agent;
2. prior assistant messages and tool events are ranked with lexical cosine
   similarity over token counts;
3. the next agent receives only the original task plus the top-k retrieved
   memory entries, not the full transcript.

This is intentionally a baseline, not VACTH: it does not create typed state
items, provenance edges, merge keys, CVE routing, or capsule budgets. Retrieval
decisions are recorded in `routing_decisions.json` with
`retrieval_method=lexical_cosine`, and the memory store plus retrieval log are
written to `vector_memory.json`. Metrics include `memory_item_count`,
`retrieval_count`, `retrieval_top_k`, `retrieved_tokens`, and
`avg_retrieval_score`.

## E5 Scope

E5 adds the `software_repair_structured_summary` baseline. It keeps the same
agents, task, model configuration, and tools, but replaces free-text handoff
with a single global JSON summary:

```json
{
  "facts": [],
  "constraints": [],
  "decisions": [],
  "hypotheses": [],
  "tool_states": [],
  "open_issues": []
}
```

After each non-reviewer agent turn, `structured_summary_runtime` asks the LLM
to update that JSON object. The next agent receives the original task plus the
latest global JSON, not the full transcript. This is a strong structured
baseline, but it intentionally does not use VACTH provenance graphs, evidence
pointers, merge keys, conflict/supersede edges, receiver-specific routing, or
capsules. The global JSON snapshots are written to `structured_summary.json`.
Metrics include `structured_update_count`, `structured_state_item_count`,
`structured_tokens`, `structured_parse_error_count`, and
`structured_schema_valid`.

## E6 Scope

E6 adds the first runnable `software_repair_vacth_full` system. The reusable
VACTH core lives under `autogen_agentchat.vacth` and is separate from the
AutoGenBench scenario:

- `schema.py`: typed state items, provenance edges, and capsules
- `extractor.py`: THC prompt helpers and extraction parsing
- `aggregator.py`: provenance-aware aggregation with duplicate/refine/supersede
  edges
- `router.py`: CVE-style greedy budgeted routing
- `renderer.py`: role-specific capsule rendering
- `runtime.py`: the core runtime that combines aggregation, routing, and
  rendering

The software-repair scenario uses the same five agents and tools as previous
methods. Before each agent acts, VACTH builds a receiver-specific capsule from
the shared typed state graph under `capsule_token_budget`. After each agent
turn, `vacth_extractor` extracts typed state deltas and provenance edges from
the raw output, and PAA merges them into the graph. The downstream agents do
not receive the raw transcript.

E6 writes `capsules.json` and `vacth_extractions.json` in addition to the
standard `state_items.json`, `routing_decisions.json`, and
`provenance_graph.json`. Metrics include `vacth_extraction_count`,
`vacth_parse_error_count`, `vacth_capsule_count`, `vacth_selected_item_count`,
`vacth_cve_routing_count`, and `vacth_avg_capsule_tokens`.

## E7 Scope

E7 turns the VACTH ablation names into runnable software-repair baselines. All
ablation methods keep the same five AutoGen agents, tools, task, model
configuration, and result contract as `vacth_full`; only the communication
mechanism switches change:

- `vacth_wo_cve`: disables contextual-value routing and sends a chronological
  global capsule under the same token budget
- `vacth_wo_thc`: disables LLM typed handoff extraction and uses deterministic
  heuristic state extraction from the latest turn
- `vacth_wo_paa`: disables provenance-aware aggregation, so extracted items are
  appended without duplicate/refine/supersede merging
- `vacth_wo_provenance`: keeps typed items but removes evidence pointers and
  provenance edges
- `vacth_wo_reask`: disables the repair re-ask when THC output is not valid
  JSON
- `vacth_global_capsule`: keeps CVE scoring but uses a role-neutral global
  capsule instead of receiver-specific routing

The generated task file is `Tasks/software_repair_vacth_ablations.jsonl`.
Each run records the active switches in `experiment_config` and metrics such as
`vacth_enable_thc`, `vacth_enable_cve`, `vacth_enable_paa`,
`vacth_enable_provenance`, `vacth_enable_reask`,
`vacth_role_specific_routing`, `vacth_heuristic_extraction_count`, and
`vacth_reask_count`.

## E8 Scope

E8 adds a lightweight VACTH-Bench mechanism suite. This is separate from the
main software-repair endpoint benchmark: it evaluates whether a communication
mechanism preserves and routes the internal information VACTH is designed to
model.

The initial E8-mini data release contains 10 hand-authored samples for each
mechanism group:

- `extraction`: typed state extraction, epistemic status, and evidence pointers
- `routing`: budgeted receiver-specific item selection
- `aggregation`: provenance-aware merge, duplicate/refine/supersede behavior
- `reask`: targeted repair of malformed typed handoff deltas

The generated tasks compare `summary`, `structured_summary`, and `vacth_full`
using the same gold labels. Scores are written into `metrics` and aggregated by
`Scripts/custom_tabulate.py` as mechanism score groups:

- extraction item F1, slot F1, status accuracy, evidence F1
- routing recall, routing NDCG, budget-used ratio
- active item F1, edge F1, supersession accuracy
- re-ask parse success, target accuracy, and attempt count

The first 10 samples in each JSONL file are hand-authored E8-mini records. The
current scale-up data keeps those records and appends deterministic generated
samples so that each mechanism group has 50 records:

- 10 hand-authored seed samples
- 40 generated samples spanning software repair, tool/API workflows, database
  migration constraints, privacy/security constraints, and evidence-conflict
  verification

Generated sample ids are prefixed with `gen_`. Regenerate the scale-up data and
tasks with:

```bash
python Scripts/scale_mechanism_data.py --target 50
python Scripts/init_tasks.py
```

This produces 200-task suites for each method:

- `Tasks/mechanism_suite_summary.jsonl`
- `Tasks/mechanism_suite_structured_summary.jsonl`
- `Tasks/mechanism_suite_vacth_full.jsonl`

It also produces 4-task scale-up smoke files, one generated sample per
mechanism group:

- `Tasks/mechanism_scaleup_smoke_summary.jsonl`
- `Tasks/mechanism_scaleup_smoke_structured_summary.jsonl`
- `Tasks/mechanism_scaleup_smoke_vacth_full.jsonl`

The E8-mini full results are preserved under `Results/mechanism_e8_summary.*`.
The scale-up smoke results are under `Results/mechanism_scaleup_smoke_*`.

## E8 Ablation Extension

This extension adds the 200-task mechanism suite to the VACTH ablations introduced in
E7. The same 50 extraction, 50 routing, 50 aggregation, and 50 reask samples are
generated for:

- `vacth_wo_cve`
- `vacth_wo_thc`
- `vacth_wo_paa`
- `vacth_wo_provenance`
- `vacth_wo_reask`
- `vacth_global_capsule`

Each ablation uses the same deterministic mechanism gold labels as
`vacth_full`; only the communication-mechanism switches change. Full results
are under `Results/mechanism_suite_<method>/`, and the combined comparison is
written to:

- `Results/mechanism_ablation_full_summary.md`
- `Results/mechanism_ablation_full_summary.csv`

Regenerate the summary after rerunning the suites with:

```bash
python Scripts/summarize_mechanism_ablation.py
```

## E8 Statistical Extension

This extension turns the full mechanism ablation runs into paper-oriented statistical
artifacts. It does not rerun experiments; it reads the existing
`Results/mechanism_suite_<method>/metrics.csv` files and pairs each ablation
run with the matching `vacth_full` sample by
`(mechanism_task_type, mechanism_sample_id)`.

The analysis reports:

- paired mean loss against `vacth_full`
- nonparametric bootstrap 95% confidence intervals over paired samples
- worse/tied/better paired counts
- component-attribution losses for extraction, routing, aggregation, and reask
- a compact SVG heatmap of weighted component losses

Generated files:

- `Results/mechanism_ablation_paired_stats.md`
- `Results/mechanism_ablation_paired_stats.csv`
- `Results/mechanism_ablation_component_losses.md`
- `Results/mechanism_ablation_component_losses.csv`
- `Results/mechanism_ablation_loss_heatmap.svg`

Regenerate them with:

```bash
python Scripts/analyze_mechanism_ablation_stats.py
```

## E9 Scope

E9 adds a local endpoint software-repair suite for the actual
`VACTH vs baseline` comparison layer. This is separate from the deterministic
mechanism suite: each task creates a small buggy repository in `./workspace`,
agents inspect files and tests, the coder edits source, the tester/reviewer
verify the repair, and success requires all of the following:

- the unittest suite passes
- the reviewer emits `FINAL_ANSWER: resolved`
- no test files are modified
- the expected source file is touched

The current local suite has 8 repair tasks:

- `calc_add`
- `csv_quoted_commas`
- `slug_whitespace`
- `refund_boundary`
- `safe_divide_zero`
- `dedupe_order`
- `iso_date_format`
- `inventory_equal_stock`

Generated task files:

- `Tasks/software_repair_suite_autogen_broadcast.jsonl`
- `Tasks/software_repair_suite_summary.jsonl`
- `Tasks/software_repair_suite_sliding_window.jsonl`
- `Tasks/software_repair_suite_vector_memory.jsonl`
- `Tasks/software_repair_suite_structured_summary.jsonl`
- `Tasks/software_repair_suite_vacth_full.jsonl`
- `Tasks/software_repair_suite_all.jsonl`

The completed local E9 results are written to:

- `Results/software_repair_suite_summary.md`
- `Results/software_repair_suite_summary.csv`
- `Results/software_repair_suite_task_matrix.csv`

Regenerate the task files and summary with:

```bash
python Scripts/init_tasks.py
python Scripts/summarize_software_repair_suite.py
```

Run one method's endpoint suite with:

```bash
AGBENCH_ALLOW_NATIVE=Yes agbench run --native --parallel 2 Tasks/software_repair_suite_vacth_full.jsonl
agbench tabulate Results/software_repair_suite_vacth_full
python Scripts/check_result_schema.py Results/software_repair_suite_vacth_full
```

The current local E9 suite is intentionally lightweight; it validates the
end-to-end comparison machinery before moving to real SWE/BugsInPy-scale tasks.
Because the tasks are small, all methods currently solve all 8 tasks. The
high-difficulty SWE/BugsInPy adapter should be treated as the next endpoint
experiment step, not as already claimed by this local suite.

## External Repair Benchmarks

The benchmark now includes adapters for real endpoint repair benchmarks:

- SWE-bench Lite metadata from `SWE-bench/SWE-bench_Lite`
- SWE-bench Verified metadata from `SWE-bench/SWE-bench_Verified`
- BugsInPy checkout metadata and framework from `soarsmu/BugsInPy`

Download or refresh the local copies with:

```bash
python Scripts/download_external_benchmarks.py
```

This writes:

- `data/external/swebench_lite/index_dev.jsonl`
- `data/external/swebench_lite/index_test.jsonl`
- `data/external/swebench_verified/index_test.jsonl`
- `data/external/BugsInPy/index.jsonl`

Build small experiment JSONL files with:

```bash
python Scripts/build_external_repair_tasks.py \
  --benchmark swebench_lite \
  --split test \
  --limit 3 \
  --with-setup \
  --repo psf/requests \
  --repo pallets/flask \
  --repo mwaskom/seaborn

python Scripts/build_external_repair_tasks.py \
  --benchmark swebench_verified \
  --split test \
  --limit 3 \
  --with-setup \
  --repo psf/requests \
  --repo pallets/flask \
  --repo mwaskom/seaborn

python Scripts/build_external_repair_tasks.py \
  --benchmark bugsinpy \
  --limit 3 \
  --compile-bugsinpy
```

Generated task files follow the pattern:

- `Tasks/external_swebench_lite_<method>.jsonl`
- `Tasks/external_swebench_verified_<method>.jsonl`
- `Tasks/external_bugsinpy_<method>.jsonl`

Run one external method subset with:

```bash
AGBENCH_ALLOW_NATIVE=Yes agbench run --native --parallel 1 Tasks/external_swebench_lite_sliding_window.jsonl
agbench tabulate Results/external_swebench_lite_sliding_window
python Scripts/check_result_schema.py Results/external_swebench_lite_sliding_window
python Scripts/summarize_external_repair.py swebench_lite
```

Implementation notes:

- SWE-bench tasks shallow-fetch the target GitHub repo at `base_commit`, apply
  the SWE-bench test patch, optionally run `pip install -e .`, and then run the
  selected `FAIL_TO_PASS` tests.
- BugsInPy tasks use official checkout metadata, then create a local `env` venv
  in the checked-out project with a macOS-compatible native setup path. The test
  command directly runs `bugsinpy_run_test.sh` inside that venv.
- Real benchmark execution is much heavier than the local E9 suite. Before
  running all methods, first run one method on a small subset and filter out
  instances whose failing tests do not reproduce in the current Python/macOS
  environment.

## Baseline Comparison

The E1-E6 software-repair baseline results are summarized separately from the
E8 mechanism-suite results. The original software-repair smoke comparison is a
single calculator sanity run. The E9 endpoint suite above is the multi-task
software-repair comparison for `autogen_broadcast`, `summary`,
`sliding_window`, `vector_memory`, `structured_summary`, and `vacth_full`.
The mechanism-suite comparison uses the 200-task suites and compares `summary`,
`structured_summary`, and `vacth_full` on typed mechanism metrics.

Generated files:

- `Results/baseline_comparison_summary.md`
- `Results/software_repair_baseline_comparison.csv`
- `Results/mechanism_baseline_comparison.csv`
- `Results/software_repair_suite_summary.md`
- `Results/software_repair_suite_summary.csv`
- `Results/software_repair_suite_task_matrix.csv`

Regenerate them with:

```bash
python Scripts/summarize_baseline_comparison.py
python Scripts/summarize_software_repair_suite.py
```

## Layout

```text
VACTHBench/
  data/
    extraction.jsonl
    routing.jsonl
    aggregation.jsonl
    reask.jsonl
  Scripts/
    init_tasks.py
    custom_tabulate.py
    check_result_schema.py
    summarize_mechanism_ablation.py
    analyze_mechanism_ablation_stats.py
    summarize_baseline_comparison.py
    summarize_software_repair_suite.py
    download_external_benchmarks.py
    build_external_repair_tasks.py
    summarize_external_repair.py
  Templates/
    common/
      model_client_factory.py
    software_repair/
      scenario.py
      prompt.txt
      requirements.txt
    fact_verification/
      scenario.py
      prompt.txt
      requirements.txt
    tool_use/
      scenario.py
      prompt.txt
      requirements.txt
    mechanism/
      scenario.py
      prompt.txt
      requirements.txt
  Tasks/
    generated by Scripts/init_tasks.py
  Results/
    generated by agbench run
```

## Quick Start

Run from this directory:

```bash
python Scripts/init_tasks.py
agbench run --native Tasks/software_repair_autogen_broadcast.jsonl --subsample 1
agbench tabulate Results/software_repair_autogen_broadcast
python Scripts/check_result_schema.py Results/software_repair_autogen_broadcast
```

Run the mechanism suite:

```bash
python Scripts/init_tasks.py
agbench run --native Tasks/mechanism_suite_vacth_full.jsonl --subsample 4
agbench tabulate Results/mechanism_suite_vacth_full
python Scripts/check_result_schema.py Results/mechanism_suite_vacth_full
```

Run the generated scale-up smoke:

```bash
python Scripts/scale_mechanism_data.py --target 50
python Scripts/init_tasks.py
agbench run --native --parallel 4 Tasks/mechanism_scaleup_smoke_vacth_full.jsonl
agbench tabulate Results/mechanism_scaleup_smoke_vacth_full
python Scripts/check_result_schema.py Results/mechanism_scaleup_smoke_vacth_full
```

The native smoke run creates an isolated `.agbench_venv` inside each run
folder. Docker execution is still the recommended mode for full experiments.

## Dashboard

The lightweight dashboard reads result artifacts directly from `Results/` and
does not require a build step or database:

```bash
python Scripts/dashboard.py --port 8765
```

Open `http://127.0.0.1:8765` to inspect:

- run-level metrics and final answers
- raw multi-agent message streams
- exact LLM-visible contexts captured before each model call
- parsed tool calls and tool results
- side files and repaired workspace files

Reload the page after running new `agbench` tasks; the dashboard rescans
`Results/**/result.json` on demand.

## Model Provider

All LLM-based scenarios should load the shared model configuration from the
benchmark-level `config.yaml`. The default configuration targets an
OpenAI-compatible proxy:

```text
http://207.56.225.111:8317/v1
```

The helper copied into each run folder is `model_client_factory.py`:

```python
from model_client_factory import load_model_client

model_client = load_model_client("config.yaml")
```

The default config uses `OpenAIChatCompletionClient`, so the proxy must expose
the Chat Completions-compatible route `/v1/chat/completions`. Override values
without editing the file:

```bash
export VACTHBENCH_MODEL="gpt-5.5"
export VACTHBENCH_OPENAI_BASE_URL="http://207.56.225.111:8317/v1"
export VACTHBENCH_OPENAI_API_KEY="placeholder"
```

For a proxy that does not require authentication, `placeholder` is sufficient.
For a key-protected proxy, set `VACTHBENCH_OPENAI_API_KEY`.

## Unified Result Files

Each scenario run writes the following files:

- `raw_messages.json`
- `visible_contexts.json`
- `tool_calls.json`
- `summaries.json`
- `structured_summary.json`
- `capsules.json`
- `vacth_extractions.json`
- `vector_memory.json`
- `state_items.json`
- `routing_decisions.json`
- `provenance_graph.json`
- `result.json`
- `metrics.csv`

`result.json` is the stable contract consumed by `custom_tabulate.py` and
`Scripts/check_result_schema.py`.

## Methods

E0 generates task files for the method names used by later experiment units:

- `autogen_broadcast`
- `summary`
- `sliding_window`
- `vector_memory`
- `structured_summary`
- `vacth_full`
- VACTH ablations, stored in `software_repair_vacth_ablations.jsonl`

E1-E7 replace the communication runtime while preserving the task schema and
result contract, so method-level comparisons differ only in what information is
handed from one agent to the next.
