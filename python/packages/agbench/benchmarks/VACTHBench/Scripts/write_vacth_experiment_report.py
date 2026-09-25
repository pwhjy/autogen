from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"

CORE_METHODS = [
    "autogen_broadcast",
    "summary",
    "sliding_window",
    "vector_memory",
    "structured_summary",
    "vacth_full",
    "vacth_optimized",
]

ABLATION_METHODS = [
    "vacth_wo_cve",
    "vacth_wo_thc",
    "vacth_wo_paa",
    "vacth_wo_provenance",
    "vacth_wo_reask",
    "vacth_global_capsule",
]

METHOD_LABELS = {
    "autogen_broadcast": "AutoGen broadcast",
    "summary": "普通 summary",
    "sliding_window": "最近窗口",
    "vector_memory": "向量记忆",
    "structured_summary": "结构化 summary",
    "vacth_full": "Full VACTH",
    "vacth_optimized": "VACTH optimized",
    "vacth_wo_cve": "w/o value scoring",
    "vacth_wo_thc": "w/o typed handoff",
    "vacth_wo_paa": "w/o PAA",
    "vacth_wo_provenance": "w/o provenance",
    "vacth_wo_reask": "w/o re-ask",
    "vacth_global_capsule": "global capsule",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return str(value) if value is not None else ""
    return f"{number:.{digits}f}"


def intish(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return str(value)
    return str(int(round(number)))


def method_rows(path: Path, methods: list[str]) -> list[dict[str, str]]:
    rows = read_csv(path)
    by_method = {row["method"]: row for row in rows}
    return [by_method[method] for method in methods if method in by_method]


def table(rows: list[dict[str, Any]], columns: list[str], headers: dict[str, str] | None = None) -> list[str]:
    headers = headers or {}
    text_columns = {"method", "budget", "software_task_id", "category", "dominant_component", "note", "experiment"}
    lines = [
        "| " + " | ".join(headers.get(column, column) for column in columns) + " |",
        "| " + " | ".join("---" if column in text_columns else "---:" for column in columns) + " |",
    ]
    for row in rows:
        rendered: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if column == "method":
                rendered.append(METHOD_LABELS.get(str(value), str(value)))
            elif column == "runs":
                rendered.append(intish(value))
            elif column in text_columns:
                rendered.append(str(value))
            else:
                rendered.append(fmt(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def select(rows: list[dict[str, str]], **filters: str) -> list[dict[str, str]]:
    return [row for row in rows if all(row.get(key) == value for key, value in filters.items())]


def add_delta_vs_full(rows: list[dict[str, str]], full_success: float) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        success = as_float(item.get("success_rate"))
        item["delta_vs_full_vacth"] = "" if success is None else str(success - full_success)
        output.append(item)
    return output


def matrix_rows(path: Path, methods: list[str]) -> list[dict[str, str]]:
    rows = read_csv(path)
    for row in rows:
        for method in methods:
            row[method] = row.get(method, "")
    return rows


def compact_matrix(rows: list[dict[str, str]], methods: list[str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        item = {
            "software_task_id": row.get("software_task_id", ""),
            "category": row.get("category", ""),
        }
        for method in methods:
            item[method] = row.get(method, "")
        output.append(item)
    return output


def category_summary(core_raw: Path, ablation_raw: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for label, path in [("core", core_raw), ("ablation", ablation_raw)]:
        for row in read_csv(path):
            category = row.get("software_task_category") or "unspecified"
            item = grouped.setdefault(
                category,
                {
                    "category": category,
                    "core_runs": 0,
                    "core_success": 0.0,
                    "ablation_runs": 0,
                    "ablation_success": 0.0,
                },
            )
            runs_key = f"{label}_runs"
            success_key = f"{label}_success"
            item[runs_key] += 1
            item[success_key] += as_float(row.get("success")) or 0.0

    rows: list[dict[str, Any]] = []
    for item in grouped.values():
        core_runs = item["core_runs"]
        ablation_runs = item["ablation_runs"]
        rows.append(
            {
                "category": item["category"],
                "core_runs": core_runs,
                "core_success_rate": "" if core_runs == 0 else item["core_success"] / core_runs,
                "ablation_runs": ablation_runs,
                "ablation_success_rate": "" if ablation_runs == 0 else item["ablation_success"] / ablation_runs,
            }
        )
    return sorted(rows, key=lambda row: str(row["category"]))


def main() -> None:
    core_summary = method_rows(RESULTS_DIR / "llm_complete_20260630_core_core_summary.csv", CORE_METHODS)
    core_matrix = compact_matrix(
        matrix_rows(RESULTS_DIR / "llm_complete_20260630_core_core_task_matrix.csv", CORE_METHODS),
        CORE_METHODS,
    )
    full_success = as_float(next(row["success_rate"] for row in core_summary if row["method"] == "vacth_full")) or 0.0

    ablation_summary = add_delta_vs_full(
        method_rows(
            RESULTS_DIR / "llm_complete_20260630_ablation_full_ablation_summary.csv",
            ABLATION_METHODS,
        ),
        full_success,
    )
    ablation_matrix = compact_matrix(
        matrix_rows(
            RESULTS_DIR / "llm_complete_20260630_ablation_full_ablation_task_matrix.csv",
            ABLATION_METHODS,
        ),
        ABLATION_METHODS,
    )
    grouped_by_category = category_summary(
        RESULTS_DIR / "llm_complete_20260630_core_raw_results.csv",
        RESULTS_DIR / "llm_complete_20260630_ablation_full_raw_results.csv",
    )

    mechanism_rows = select(read_csv(RESULTS_DIR / "mechanism_ablation_full_summary.csv"), mechanism_task_type="all")
    mechanism_rows = [row for row in mechanism_rows if row["method"] in ["vacth_full", *ABLATION_METHODS]]
    loss_rows = method_rows(RESULTS_DIR / "mechanism_ablation_component_losses.csv", ABLATION_METHODS)
    budget_rows = [
        row
        for row in read_csv(RESULTS_DIR / "budget_sensitivity_summary.csv")
        if row["budget"] in {"128", "256", "512", "1024", "2048", "4096", "unlimited"}
        and row["method"] in {"vacth_value_aware", "recent_state", "random_state"}
    ]
    state_rows = read_csv(RESULTS_DIR / "state_transfer_quality_summary.csv")

    lines: list[str] = [
        "# VACTH 小规模 LLM 实验报告",
        "",
        "## 实验范围",
        "",
        "本轮实验不使用 SWE-bench Lite，也不包含 LuluVerse 平台压测。主实验改为本地小型软件修复任务，每个任务都经过 LLM 代理流程，包含代码修改、测试运行和 reviewer 判定。控制实验用于解释状态传递、预算路由和模块消融的机制，不把它们等同于真实仓库修复能力。",
        "",
        "| 实验 | 规模 | 是否经过 LLM | 作用 |",
        "| --- | ---: | --- | --- |",
        "| 本地软件修复核心 baseline | 7 方法 x 8 题 = 56 条 | 是 | 比较 VACTH 与普通 summary、向量记忆、最近窗口等 baseline |",
        "| 本地软件修复 VACTH 消融 | 6 变体 x 8 题 = 48 条 | 是 | 检查 typed handoff、value scoring、provenance、PAA、re-ask 等模块在 LLM 修复流程中的影响 |",
        "| 受控机制消融 | 7 个 VACTH 配置 x 200 条机制样本 | 否 | 定位模块损失来自 extraction、routing、aggregation 还是 re-ask |",
        "| 上下文预算敏感性 | 3 策略 x 7 档预算 | 否 | 判断胶囊预算压缩到何处开始丢失关键状态 |",
        "| 状态传递质量 | 3 方法 x 200 条机制样本 | 否 | 检查传给下游智能体的状态是否保留约束、证据、状态类型和修复目标 |",
        "",
        "软件修复任务共 8 个，覆盖运算符错误、CSV 解析边界、字符串归一化、边界条件、异常处理、数据结构语义和日期格式。成功标准为测试通过、reviewer 判定 resolved、未修改测试文件，并触及预期源码文件。",
        "",
        "## LLM 核心 baseline 结果",
        "",
        "56 条核心 LLM 运行全部完成并通过结果结构校验。除 AutoGen broadcast 在 `slug_whitespace` 任务上 reviewer 未判 resolved 外，其余方法在 8 个任务上均满足成功标准。",
        "",
        *table(
            core_summary,
            [
                "method",
                "runs",
                "success_rate",
                "reviewer_resolved_rate",
                "avg_turns",
                "avg_tool_calls",
                "avg_estimated_tokens",
                "avg_wall_time_sec",
                "avg_vacth_capsule_count",
                "avg_vacth_reask_count",
            ],
            {
                "success_rate": "success",
                "reviewer_resolved_rate": "reviewer",
                "avg_turns": "turns",
                "avg_tool_calls": "tools",
                "avg_estimated_tokens": "tokens",
                "avg_wall_time_sec": "wall_s",
                "avg_vacth_capsule_count": "capsules",
                "avg_vacth_reask_count": "reask",
            },
        ),
        "",
        "从成功率看，本地小题整体偏容易，除 AutoGen broadcast 外各方法均为 1.000。成本指标更有区分度。Full VACTH 的平均估计 token 为 36720.750，高于普通 summary 和 sliding window；VACTH optimized 降至 17947.000，平均耗时为 93.068 秒，低于 Full VACTH 的 338.894 秒。该结果说明，在当前小型任务上，优化版保留了成功率，同时降低了上下文处理成本。",
        "",
        "### 核心任务矩阵",
        "",
        *table(
            core_matrix,
            ["software_task_id", "category", *CORE_METHODS],
            {method: METHOD_LABELS[method] for method in CORE_METHODS},
        ),
        "",
        "## LLM 真实任务消融结果",
        "",
        "消融实验在同一组 8 个软件修复任务上运行 6 个 VACTH 变体，共 48 条 LLM 结果。所有结果文件通过 schema 校验。`w/o typed handoff` 在 CSV 引号逗号解析任务上失败，其余消融变体在这组任务上均达到 1.000 成功率。",
        "",
        *table(
            ablation_summary,
            [
                "method",
                "runs",
                "success_rate",
                "delta_vs_full_vacth",
                "reviewer_resolved_rate",
                "avg_turns",
                "avg_tool_calls",
                "avg_estimated_tokens",
                "avg_wall_time_sec",
                "avg_constraint_violations",
                "avg_invalid_patch_count",
                "avg_vacth_capsule_count",
                "avg_vacth_reask_count",
            ],
            {
                "success_rate": "success",
                "delta_vs_full_vacth": "delta",
                "reviewer_resolved_rate": "reviewer",
                "avg_turns": "turns",
                "avg_tool_calls": "tools",
                "avg_estimated_tokens": "tokens",
                "avg_wall_time_sec": "wall_s",
                "avg_constraint_violations": "violations",
                "avg_invalid_patch_count": "invalid_patch",
                "avg_vacth_capsule_count": "capsules",
                "avg_vacth_reask_count": "reask",
            },
        ),
        "",
        "LLM 消融表明，typed handoff 的移除会在解析边界类任务上造成一次失败。其他模块在 8 个小题上的成功率未下降，但 token、工具调用和机制指标存在差异。由于任务规模仍小，真实任务消融适合作为最低限度的 LLM 证据；模块贡献的主要解释依据放在下一节的受控机制消融中。",
        "",
        "### 消融任务矩阵",
        "",
        *table(
            ablation_matrix,
            ["software_task_id", "category", *ABLATION_METHODS],
            {method: METHOD_LABELS[method] for method in ABLATION_METHODS},
        ),
        "",
        "## 任务类别分组",
        "",
        "本地任务集均为单文件、短轨迹修复，因此不能形成多文件和长轨迹分组。当前可用的难度代理是缺陷类别。CSV 解析边界类任务在核心 baseline 中全部通过，但在 `w/o typed handoff` 消融中失败一次；字符串归一化任务暴露了 AutoGen broadcast 的 reviewer 失败。其他类别在这组小题上未形成成功率差异。",
        "",
        *table(
            grouped_by_category,
            ["category", "core_runs", "core_success_rate", "ablation_runs", "ablation_success_rate"],
            {
                "core_runs": "core_n",
                "core_success_rate": "core_success",
                "ablation_runs": "ablation_n",
                "ablation_success_rate": "ablation_success",
            },
        ),
        "",
        "## 受控机制消融",
        "",
        "受控机制实验覆盖 extraction、routing、aggregation 和 re-ask 四类样本，每类 50 条，共 200 条。该实验不依赖 LLM 成功率，而是直接检查状态抽取、预算路由、证据链和目标追问是否正确。",
        "",
        *table(
            mechanism_rows,
            [
                "method",
                "runs",
                "overall",
                "delta_vs_vacth_full",
                "extract_item_f1",
                "routing_recall",
                "active_item_f1",
                "edge_f1",
                "reask_target",
            ],
            {
                "delta_vs_vacth_full": "delta",
                "extract_item_f1": "extract_f1",
                "routing_recall": "route_recall",
                "active_item_f1": "active_f1",
                "reask_target": "reask_target",
            },
        ),
        "",
        "组件损失归因如下。`w/o typed handoff` 的总损失为 0.458，主要来自 extraction 和 re-ask；`w/o re-ask` 的损失集中在 re-ask；`w/o provenance` 与 `w/o PAA` 的损失主要出现在 aggregation；`w/o value scoring` 的损失集中在 routing。",
        "",
        *table(
            loss_rows,
            [
                "method",
                "total_loss",
                "dominant_component",
                "extraction_share",
                "routing_share",
                "aggregation_share",
                "reask_share",
            ],
            {
                "total_loss": "loss",
                "dominant_component": "dominant",
                "extraction_share": "extract",
                "routing_share": "routing",
                "aggregation_share": "aggregation",
                "reask_share": "reask",
            },
        ),
        "",
        "## 上下文预算敏感性",
        "",
        "预算实验固定 routing 样本，只改变胶囊预算，并加入最近状态和随机状态两个对照。VACTH value-aware routing 在 512 tokens 时召回率为 0.907，1024 tokens 时达到 0.993。随机状态在 4096 tokens 仍只有 0.733 召回，最近状态在 512 tokens 后基本不再提升。",
        "",
        *table(
            budget_rows,
            [
                "method",
                "budget",
                "runs",
                "routing_recall",
                "routing_precision",
                "routing_ndcg",
                "all_gold_selected_rate",
                "avg_used_tokens",
            ],
            {
                "routing_recall": "recall",
                "routing_precision": "precision",
                "routing_ndcg": "ndcg",
                "all_gold_selected_rate": "all_gold",
                "avg_used_tokens": "used_tokens",
            },
        ),
        "",
        "结果支持两个判断。第一，价值感知路由在中低预算下比随机选择稳定得多。第二，1024 tokens 左右已接近该任务集的收益上限；继续增加预算主要提高冗余状态覆盖，关键状态召回提升有限。",
        "",
        "## 状态传递质量",
        "",
        "状态传递质量实验检查用户约束、当前失败测试、已做决策、未验证假设、最新工具状态和证据来源是否被保留。普通 summary 几乎不保留结构化状态，structured summary 对 active-state 有帮助，但缺少 provenance 和 repair target。VACTH 在类型化状态、证据保留、provenance 和修复目标上均接近满分。",
        "",
        *table(
            state_rows,
            [
                "method",
                "runs",
                "typed_item_retention",
                "slot_type_accuracy",
                "evidence_retention",
                "budgeted_routing_recall",
                "provenance_edge_f1",
                "repair_target_accuracy",
                "state_transfer_quality",
            ],
            {
                "typed_item_retention": "typed",
                "slot_type_accuracy": "slot",
                "evidence_retention": "evidence",
                "budgeted_routing_recall": "route_recall",
                "provenance_edge_f1": "prov_f1",
                "repair_target_accuracy": "repair_target",
                "state_transfer_quality": "quality",
            },
        ),
        "",
        "## 错误传播与修复覆盖",
        "",
        "错误传播实验在本轮以半自动状态质量指标呈现。该部分不重复运行 SWE-bench 轨迹，而是把容易传播的错误拆成可检查的状态项和证据链指标。",
        "",
        "| 错误类型 | 对应指标 | 实验信号 |",
        "| --- | --- | --- |",
        "| 用户约束遗漏 | typed、slot、evidence | 普通 summary 在 typed、slot、evidence 上均为 0.000，VACTH 均为 1.000 |",
        "| 旧测试结果误用 | active_state_f1、stale_state_handling | VACTH 的 active_state_f1 为 0.780，stale_state_handling 为 0.900 |",
        "| 未验证假设被当作事实 | epistemic_status_accuracy | VACTH 为 1.000，summary 为 0.000 |",
        "| 补丁决策缺少证据 | evidence、prov_f1 | VACTH 的 evidence 和 prov_f1 均为 1.000，structured summary 的 prov_f1 为 0.000 |",
        "| 审查阶段缺少最新测试结果 | repair_target、reask_target | VACTH 的 repair_target 为 1.000，`w/o re-ask` 在机制实验中的 reask_target 为 0.000 |",
        "",
        "## 实验结论",
        "",
        "本轮实验给出三点结果。其一，所有核心 baseline 都完成了 LLM 运行，VACTH full 与 VACTH optimized 在 8 个本地修复题上均达到 1.000 成功率；优化版在 token 和耗时上低于 full 版本。其二，真实任务消融中，去掉 typed handoff 后在 CSV 解析边界任务上失败，与受控机制实验中 typed extraction 和 re-ask target 的大幅下降相互对应。其三，预算实验显示 512 tokens 是明显压缩区间，1024 tokens 后关键状态召回接近饱和。",
        "",
        "这些结果适合支撑论文中的机制论证。它们不能替代 SWE-bench 的 resolved rate，但能在 12 小时窗口内说明 VACTH 的核心设计是否在 LLM 流程中可运行、是否覆盖所有 baseline，以及各模块在受控机制上对应哪些损失。",
        "",
        "## 局限",
        "",
        "- 本地软件修复题规模较小，成功率较高，不能直接外推到大型真实仓库。",
        "- LLM 消融只有 8 个任务，模块贡献的主要统计依据仍来自 200 条受控机制样本。",
        "- token 为实验脚本估计值，不是 API 账单 token。",
        "- wall time 受代理服务响应速度影响，部分任务存在长尾等待。",
        "",
        "## 产物路径",
        "",
        "- 核心 LLM 汇总：`Results/llm_complete_20260630_core_software_repair_summary.md`",
        "- LLM 消融汇总：`Results/llm_complete_20260630_ablation_full_software_repair_summary.md`",
        "- 机制消融：`Results/mechanism_ablation_full_summary.md`",
        "- 组件损失：`Results/mechanism_ablation_component_losses.md`",
        "- 预算敏感性：`Results/budget_sensitivity_summary.md`",
        "- 状态传递质量：`Results/state_transfer_quality_summary.md`",
        "- 本报告：`Results/vacth_full_llm_experiment_report_20260630.md`",
        "",
        "## 验证记录",
        "",
        "- `Scripts/summarize_llm_rerun_results.py Results/llm_full_20260630_all --prefix llm_complete_20260630_core` 汇总 56 个 result 文件。",
        "- `Scripts/summarize_llm_rerun_results.py Results/llm_full_20260630_ablation_full --prefix llm_complete_20260630_ablation_full` 汇总 48 个 result 文件。",
        "- `Scripts/check_result_schema.py Results/llm_full_20260630_all` 校验 56 个 result instance。",
        "- `Scripts/check_result_schema.py Results/llm_full_20260630_ablation_full` 校验 48 个 result instance。",
    ]

    main_output = RESULTS_DIR / "vacth_full_llm_experiment_report_20260630.md"
    legacy_output = RESULTS_DIR / "vacth_fast_experiment_report_20260630.md"
    content = "\n".join(lines) + "\n"
    main_output.write_text(content, encoding="utf-8")
    legacy_output.write_text(content, encoding="utf-8")
    print(f"Wrote {main_output}")
    print(f"Wrote {legacy_output}")


if __name__ == "__main__":
    main()
