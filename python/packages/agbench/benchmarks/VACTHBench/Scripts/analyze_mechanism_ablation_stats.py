from __future__ import annotations

import csv
import html
import math
import random
from pathlib import Path
from typing import Any

from mechanism_score_calibration import calibrated_row

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"
BASELINE = "vacth_full"
ABLATIONS = [
    "vacth_wo_cve",
    "vacth_global_capsule",
    "vacth_wo_thc",
    "vacth_wo_paa",
    "vacth_wo_provenance",
    "vacth_wo_reask",
]
TASK_TYPES = ["all", "extraction", "routing", "aggregation", "reask"]
COMPONENT_TASK_TYPES = ["extraction", "routing", "aggregation", "reask"]
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260503


def read_rows(method: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"mechanism_suite_{method}" / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [calibrated_row(row) for row in csv.DictReader(fh)]


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed_suffix: str,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(f"{BOOTSTRAP_SEED}:{seed_suffix}")
    n = len(values)
    boot_means = []
    for _ in range(repeats):
        boot_means.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    boot_means.sort()
    return percentile(boot_means, 0.025), percentile(boot_means, 0.975)


def scores_by_key(
    rows: list[dict[str, Any]], task_type: str
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for row in rows:
        row_task_type = str(row.get("mechanism_task_type", ""))
        if task_type != "all" and row_task_type != task_type:
            continue
        sample_id = str(row.get("mechanism_sample_id", ""))
        score = as_float(row.get("mechanism_overall_score"))
        if not row_task_type or not sample_id or score is None:
            continue
        scores[(row_task_type, sample_id)] = score
    return scores


def paired_scores(
    baseline_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    task_type: str,
) -> list[tuple[float, float]]:
    baseline = scores_by_key(baseline_rows, task_type)
    method = scores_by_key(method_rows, task_type)
    keys = sorted(set(baseline).intersection(method))
    return [(baseline[key], method[key]) for key in keys]


def paired_stats(
    baseline_rows: list[dict[str, Any]],
    by_method: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ABLATIONS:
        for task_type in TASK_TYPES:
            pairs = paired_scores(baseline_rows, by_method[method], task_type)
            baseline_scores = [baseline for baseline, _ in pairs]
            method_scores = [score for _, score in pairs]
            deltas = [score - baseline for baseline, score in pairs]
            losses = [baseline - score for baseline, score in pairs]
            delta_ci_low, delta_ci_high = bootstrap_mean_ci(
                deltas, seed_suffix=f"{method}:{task_type}:delta"
            )
            loss_ci_low, loss_ci_high = bootstrap_mean_ci(
                losses, seed_suffix=f"{method}:{task_type}:loss"
            )
            worse_count = len([delta for delta in deltas if delta < -1e-12])
            tied_count = len([delta for delta in deltas if abs(delta) <= 1e-12])
            better_count = len([delta for delta in deltas if delta > 1e-12])
            rows.append(
                {
                    "method": method,
                    "task_type": task_type,
                    "paired_n": len(pairs),
                    "vacth_full_mean": mean(baseline_scores),
                    "method_mean": mean(method_scores),
                    "paired_delta_mean": mean(deltas),
                    "paired_delta_ci_low": delta_ci_low,
                    "paired_delta_ci_high": delta_ci_high,
                    "loss_mean": mean(losses),
                    "loss_ci_low": loss_ci_low,
                    "loss_ci_high": loss_ci_high,
                    "worse_count": worse_count,
                    "tied_count": tied_count,
                    "better_count": better_count,
                    "loss_ci_excludes_zero": loss_ci_low > 0 or loss_ci_high < 0,
                }
            )
    return rows


def row_for(stats_rows: list[dict[str, Any]], method: str, task_type: str) -> dict[str, Any]:
    for row in stats_rows:
        if row["method"] == method and row["task_type"] == task_type:
            return row
    raise KeyError(f"Missing stats row for {method}/{task_type}")


def component_losses(stats_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in ABLATIONS:
        all_row = row_for(stats_rows, method, "all")
        weighted_losses = {
            task_type: float(row_for(stats_rows, method, task_type)["loss_mean"]) / 4
            for task_type in COMPONENT_TASK_TYPES
        }
        total_loss = float(all_row["loss_mean"])
        dominant = max(weighted_losses, key=lambda key: weighted_losses[key])
        record: dict[str, Any] = {
            "method": method,
            "total_loss": total_loss,
            "total_loss_ci_low": all_row["loss_ci_low"],
            "total_loss_ci_high": all_row["loss_ci_high"],
            "dominant_component": dominant,
        }
        for task_type in COMPONENT_TASK_TYPES:
            raw_loss = float(row_for(stats_rows, method, task_type)["loss_mean"])
            weighted = weighted_losses[task_type]
            share = weighted / total_loss if total_loss else 0.0
            record[f"{task_type}_raw_loss"] = raw_loss
            record[f"{task_type}_weighted_loss"] = weighted
            record[f"{task_type}_share"] = share
        rows.append(record)
    return rows


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_paired_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "method",
        "task_type",
        "paired_n",
        "vacth_full_mean",
        "method_mean",
        "loss_mean",
        "loss_ci_low",
        "loss_ci_high",
        "worse_count",
        "tied_count",
        "better_count",
    ]
    headers = {
        "method": "method",
        "task_type": "task_type",
        "paired_n": "n",
        "vacth_full_mean": "full",
        "method_mean": "method",
        "loss_mean": "loss",
        "loss_ci_low": "loss_ci_low",
        "loss_ci_high": "loss_ci_high",
        "worse_count": "worse",
        "tied_count": "tied",
        "better_count": "better",
    }
    lines = [
        "# Mechanism Ablation Paired Statistics",
        "",
        "Paired by `(mechanism_task_type, mechanism_sample_id)` against `vacth_full` using stress-calibrated continuous scores.",
        "Confidence intervals are nonparametric bootstrap 95% CIs over paired samples.",
        "",
        "| " + " | ".join(headers[column] for column in columns) + " |",
        "| "
        + " | ".join("---" if column in {"method", "task_type"} else "---:" for column in columns)
        + " |",
    ]
    for row in rows:
        rendered = []
        for column in columns:
            if column in {"method", "task_type"}:
                rendered.append(str(row[column]))
            elif column in {"paired_n", "worse_count", "tied_count", "better_count"}:
                rendered.append(str(row[column]))
            else:
                rendered.append(fmt(row[column]))
        lines.append("| " + " | ".join(rendered) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_component_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "method",
        "total_loss",
        "total_loss_ci_low",
        "total_loss_ci_high",
        "extraction_weighted_loss",
        "routing_weighted_loss",
        "aggregation_weighted_loss",
        "reask_weighted_loss",
        "dominant_component",
    ]
    headers = {
        "method": "method",
        "total_loss": "total_loss",
        "total_loss_ci_low": "ci_low",
        "total_loss_ci_high": "ci_high",
        "extraction_weighted_loss": "extraction",
        "routing_weighted_loss": "routing",
        "aggregation_weighted_loss": "aggregation",
        "reask_weighted_loss": "reask",
        "dominant_component": "dominant",
    }
    lines = [
        "# Mechanism Ablation Component Losses",
        "",
        "Loss is `vacth_full - method` under stress-calibrated continuous scores. Component columns are weighted by the",
        "suite composition, so the four component losses sum to `total_loss`.",
        "",
        "| " + " | ".join(headers[column] for column in columns) + " |",
        "| "
        + " | ".join("---" if column in {"method", "dominant_component"} else "---:" for column in columns)
        + " |",
    ]
    for row in rows:
        rendered = []
        for column in columns:
            if column in {"method", "dominant_component"}:
                rendered.append(str(row[column]))
            else:
                rendered.append(fmt(row[column]))
        lines.append("| " + " | ".join(rendered) + " |")

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `vacth_wo_thc` loses mostly extraction plus targeted reask repair.",
            "- `vacth_wo_paa` and `vacth_wo_provenance` lose mostly aggregation.",
            "- `vacth_wo_cve` loses only routing quality.",
            "- `vacth_global_capsule` is close to full VACTH on this deterministic suite.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpolate_color(value: float, max_value: float) -> str:
    if max_value <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, value / max_value))
    start = (255, 245, 240)
    end = (203, 24, 29)
    rgb = [
        round(start[index] + (end[index] - start[index]) * ratio)
        for index in range(3)
    ]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def write_heatmap_svg(rows: list[dict[str, Any]], path: Path) -> None:
    cell_w = 118
    cell_h = 34
    left_w = 190
    top_h = 72
    width = left_w + cell_w * len(COMPONENT_TASK_TYPES) + 30
    height = top_h + cell_h * len(rows) + 45
    max_loss = max(
        float(row[f"{task_type}_weighted_loss"])
        for row in rows
        for task_type in COMPONENT_TASK_TYPES
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="Arial, sans-serif" '
        'font-size="18" font-weight="700">Mechanism Ablation Weighted Loss</text>',
        '<text x="20" y="50" font-family="Arial, sans-serif" '
        'font-size="12" fill="#555">Cell value = component contribution to total loss</text>',
    ]
    for col, task_type in enumerate(COMPONENT_TASK_TYPES):
        x = left_w + col * cell_w
        lines.append(
            f'<text x="{x + cell_w / 2:.1f}" y="66" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" font-weight="700">'
            f"{html.escape(task_type)}</text>"
        )
    for row_index, row in enumerate(rows):
        y = top_h + row_index * cell_h
        lines.append(
            f'<text x="20" y="{y + 22}" font-family="Arial, sans-serif" '
            f'font-size="12">{html.escape(str(row["method"]))}</text>'
        )
        for col, task_type in enumerate(COMPONENT_TASK_TYPES):
            value = float(row[f"{task_type}_weighted_loss"])
            x = left_w + col * cell_w
            fill = interpolate_color(value, max_loss)
            text_fill = "#ffffff" if value / max_loss > 0.65 else "#111111"
            lines.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_w - 4}" '
                    f'height="{cell_h - 4}" rx="3" fill="{fill}" stroke="#d0d0d0"/>',
                    f'<text x="{x + (cell_w - 4) / 2:.1f}" y="{y + 20}" '
                    f'text-anchor="middle" font-family="Arial, sans-serif" '
                    f'font-size="12" fill="{text_fill}">{value:.4f}</text>',
                ]
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    by_method = {method: read_rows(method) for method in [BASELINE, *ABLATIONS]}
    stats_rows = paired_stats(by_method[BASELINE], by_method)
    loss_rows = component_losses(stats_rows)

    paired_csv = RESULTS_DIR / "mechanism_ablation_paired_stats.csv"
    paired_md = RESULTS_DIR / "mechanism_ablation_paired_stats.md"
    losses_csv = RESULTS_DIR / "mechanism_ablation_component_losses.csv"
    losses_md = RESULTS_DIR / "mechanism_ablation_component_losses.md"
    heatmap_svg = RESULTS_DIR / "mechanism_ablation_loss_heatmap.svg"

    write_csv(stats_rows, paired_csv)
    write_paired_markdown(stats_rows, paired_md)
    write_csv(loss_rows, losses_csv)
    write_component_markdown(loss_rows, losses_md)
    write_heatmap_svg(loss_rows, heatmap_svg)

    print(f"Wrote {paired_csv}")
    print(f"Wrote {paired_md}")
    print(f"Wrote {losses_csv}")
    print(f"Wrote {losses_md}")
    print(f"Wrote {heatmap_svg}")


if __name__ == "__main__":
    main()
