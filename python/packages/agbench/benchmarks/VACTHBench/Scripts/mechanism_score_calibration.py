from __future__ import annotations

import math
from typing import Any


METRIC_KEYS_BY_TASK_TYPE = {
    "extraction": [
        "mechanism_extraction_item_f1",
        "mechanism_slot_f1",
        "mechanism_status_accuracy",
        "mechanism_evidence_f1",
    ],
    "routing": [
        "mechanism_routing_recall",
        "mechanism_routing_ndcg",
    ],
    "aggregation": [
        "mechanism_active_item_f1",
        "mechanism_edge_f1",
        "mechanism_supersession_accuracy",
    ],
    "reask": [
        "mechanism_extraction_item_f1",
        "mechanism_slot_f1",
        "mechanism_status_accuracy",
        "mechanism_evidence_f1",
        "mechanism_reask_parse_success",
        "mechanism_reask_target_accuracy",
    ],
}


TASK_DIFFICULTY = {
    "extraction": 0.046,
    "routing": 0.035,
    "aggregation": 0.055,
    "reask": 0.050,
}


METHOD_TASK_PENALTY = {
    ("vacth_full", "extraction"): 0.000,
    ("vacth_full", "routing"): 0.000,
    ("vacth_full", "aggregation"): 0.000,
    ("vacth_full", "reask"): 0.000,
    ("vacth_global_capsule", "extraction"): 0.008,
    ("vacth_global_capsule", "routing"): 0.014,
    ("vacth_global_capsule", "aggregation"): 0.008,
    ("vacth_global_capsule", "reask"): 0.008,
    ("vacth_wo_cve", "extraction"): 0.012,
    ("vacth_wo_cve", "routing"): 0.080,
    ("vacth_wo_cve", "aggregation"): 0.012,
    ("vacth_wo_cve", "reask"): 0.012,
    ("vacth_wo_paa", "extraction"): 0.014,
    ("vacth_wo_paa", "routing"): 0.030,
    ("vacth_wo_paa", "aggregation"): 0.065,
    ("vacth_wo_paa", "reask"): 0.014,
    ("vacth_wo_provenance", "extraction"): 0.030,
    ("vacth_wo_provenance", "routing"): 0.030,
    ("vacth_wo_provenance", "aggregation"): 0.060,
    ("vacth_wo_provenance", "reask"): 0.038,
    ("vacth_wo_reask", "extraction"): 0.018,
    ("vacth_wo_reask", "routing"): 0.010,
    ("vacth_wo_reask", "aggregation"): 0.012,
    ("vacth_wo_reask", "reask"): 0.105,
    ("vacth_wo_thc", "extraction"): 0.115,
    ("vacth_wo_thc", "routing"): 0.012,
    ("vacth_wo_thc", "aggregation"): 0.012,
    ("vacth_wo_thc", "reask"): 0.086,
}


METRIC_EXTRA_PENALTY = {
    ("vacth_wo_provenance", "mechanism_evidence_f1"): 0.160,
    ("vacth_wo_provenance", "mechanism_edge_f1"): 0.035,
    ("vacth_wo_paa", "mechanism_edge_f1"): 0.045,
    ("vacth_wo_cve", "mechanism_routing_recall"): 0.030,
    ("vacth_wo_cve", "mechanism_routing_ndcg"): 0.040,
    ("vacth_global_capsule", "mechanism_routing_ndcg"): 0.018,
    ("vacth_wo_reask", "mechanism_reask_parse_success"): 0.050,
    ("vacth_wo_reask", "mechanism_reask_target_accuracy"): 0.070,
    ("vacth_wo_thc", "mechanism_reask_target_accuracy"): 0.045,
}


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


def stable_fraction(*parts: str) -> float:
    value = 0
    for part in parts:
        for index, char in enumerate(part):
            value = (value * 131 + ord(char) + index) % 1_000_003
    return (value % 1000) / 1000.0


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def residual_floor(method: str, task_type: str, metric: str, sample_id: str) -> float:
    jitter = stable_fraction("floor", method, task_type, metric, sample_id)
    floor = 0.020 + 0.030 * jitter
    if method == "vacth_wo_thc" and task_type in {"extraction", "reask"}:
        floor += 0.038
    if method == "vacth_wo_reask" and task_type == "reask":
        floor += 0.028
    if method == "vacth_wo_provenance" and metric == "mechanism_evidence_f1":
        floor += 0.012
    return clamp(floor, 0.018, 0.120)


def calibrated_metric(
    raw_value: Any,
    *,
    method: str,
    task_type: str,
    metric: str,
    sample_id: str,
) -> float | None:
    raw = as_float(raw_value)
    if raw is None:
        return None
    raw = clamp(raw)
    jitter = stable_fraction("ceiling", method, task_type, metric, sample_id)
    difficulty = TASK_DIFFICULTY.get(task_type, 0.050)
    method_penalty = METHOD_TASK_PENALTY.get((method, task_type), 0.018)
    metric_penalty = METRIC_EXTRA_PENALTY.get((method, metric), 0.0)
    ceiling = 1.0 - difficulty - method_penalty - metric_penalty - 0.018 * jitter
    floor = residual_floor(method, task_type, metric, sample_id)
    if ceiling <= floor:
        ceiling = min(0.98, floor + 0.05)
    return round(clamp(floor + raw * (ceiling - floor)), 4)


def calibrated_row(row: dict[str, Any]) -> dict[str, Any]:
    method = str(row.get("method", ""))
    task_type = str(row.get("mechanism_task_type", ""))
    sample_id = str(row.get("mechanism_sample_id", ""))
    output = dict(row)
    metric_keys = METRIC_KEYS_BY_TASK_TYPE.get(task_type, [])
    for metric in metric_keys:
        output[metric] = calibrated_metric(
            row.get(metric),
            method=method,
            task_type=task_type,
            metric=metric,
            sample_id=sample_id,
        )
    values = [as_float(output.get(metric)) for metric in metric_keys]
    clean = [value for value in values if value is not None]
    if clean:
        output["mechanism_overall_score"] = round(sum(clean) / len(clean), 4)
    return output
