from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"
OUT_PREFIX = "chapter5_context_suite"
MAIN_CONTEXT_BUDGET = 512
KNOWLEDGE_CONTEXT_BUDGET = 384
ENTERPRISE_CONTEXT_BUDGET = 768
HARDENING_SEEDS = (17, 29, 43, 71, 101)
BUDGETS: tuple[int | None, ...] = (128, 256, 512, 1024, 2048, 4096, None)
SERIALIZATION_COST_SCALE = {
    "sliding_window": 1.0,
    "random_state": 1.0,
    "summary": 0.55,
    "structured_summary": 0.8,
    "vector_memory": 1.0,
    "autogen_broadcast": 1.0,
    "vacth": 0.9,
}


SLOTS = {
    "fact",
    "constraint",
    "decision",
    "hypothesis",
    "tool_state_delta",
    "artifact",
    "unresolved_issue",
}
PRIORITY_VALUE = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
STATUS_VALUE = {
    "hypothesis": 0.35,
    "observed": 0.55,
    "active_constraint": 0.9,
    "verified": 1.0,
    "invalidated": 0.8,
    "superseded": 0.1,
}
ROLE_SLOT_VALUE = {
    "global": {
        "constraint": 1.0,
        "unresolved_issue": 0.95,
        "tool_state_delta": 0.95,
        "decision": 0.85,
        "artifact": 0.8,
        "fact": 0.8,
        "hypothesis": 0.55,
    },
    "planner": {
        "constraint": 1.0,
        "unresolved_issue": 0.95,
        "hypothesis": 0.7,
        "fact": 0.55,
        "decision": 0.6,
        "tool_state_delta": 0.45,
        "artifact": 0.45,
    },
    "executor": {
        "decision": 0.95,
        "artifact": 0.9,
        "constraint": 0.85,
        "fact": 0.75,
        "unresolved_issue": 0.7,
        "tool_state_delta": 0.55,
        "hypothesis": 0.35,
    },
    "tester": {
        "tool_state_delta": 1.0,
        "artifact": 0.9,
        "unresolved_issue": 0.8,
        "constraint": 0.75,
        "decision": 0.55,
        "fact": 0.45,
        "hypothesis": 0.25,
    },
    "reviewer": {
        "constraint": 1.0,
        "tool_state_delta": 1.0,
        "artifact": 0.9,
        "decision": 0.85,
        "unresolved_issue": 0.85,
        "fact": 0.75,
        "hypothesis": 0.35,
    },
    "analyst": {
        "fact": 1.0,
        "hypothesis": 0.85,
        "unresolved_issue": 0.8,
        "constraint": 0.65,
        "decision": 0.4,
        "tool_state_delta": 0.4,
        "artifact": 0.35,
    },
    "approver": {
        "constraint": 1.0,
        "decision": 0.95,
        "tool_state_delta": 0.9,
        "fact": 0.75,
        "unresolved_issue": 0.7,
        "artifact": 0.55,
        "hypothesis": 0.25,
    },
    "auditor": {
        "constraint": 1.0,
        "decision": 0.95,
        "tool_state_delta": 0.9,
        "fact": 0.85,
        "artifact": 0.75,
        "unresolved_issue": 0.7,
        "hypothesis": 0.35,
    },
    "external_vendor": {
        "constraint": 0.85,
        "decision": 0.55,
        "tool_state_delta": 0.4,
        "artifact": 0.35,
        "fact": 0.35,
        "unresolved_issue": 0.3,
        "hypothesis": 0.2,
    },
}


@dataclass(frozen=True)
class StateItem:
    item_id: str
    scenario_id: str
    slot: str
    content: str
    status: str
    priority: str
    created_at: int
    source: str
    evidence_id: str | None
    active: bool
    gold: bool
    roles: tuple[str, ...]
    domain: str
    token_cost: int
    supersedes: str | None = None
    contradicts: str | None = None
    sensitive: bool = False
    allowed_roles: tuple[str, ...] = ()
    # Hidden evaluation truth. The ordinary fields above are what a method observes.
    gold_active: bool | None = None
    gold_roles: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    domain: str
    description: str
    receiver_roles: tuple[str, ...]
    items: tuple[StateItem, ...]
    gold_edges: tuple[tuple[str, str, str], ...] = ()
    injected_errors: tuple[str, ...] = ()
    difficulty: str = "base"
    stress_seed: int | None = None


@dataclass(frozen=True)
class MethodPolicy:
    method: str
    display_name: str
    type_quality: float
    evidence_quality: float
    status_quality: float
    relation_quality: float
    summary_compression: float
    base_noise: float
    repair_detection: float
    repair_success: float
    repair_turns: float
    rework: float
    unauthorized_filter: float


@dataclass
class Selection:
    selected: list[StateItem]
    used_tokens: int
    total_candidate_tokens: int
    budget_exceeded: bool = False


METHODS = [
    "sliding_window",
    "random_state",
    "summary",
    "structured_summary",
    "vector_memory",
    "autogen_broadcast",
    "vacth",
]
POLICIES: dict[str, MethodPolicy] = {
    "sliding_window": MethodPolicy(
        method="sliding_window",
        display_name="Sliding window",
        type_quality=0.45,
        evidence_quality=0.35,
        status_quality=0.35,
        relation_quality=0.15,
        summary_compression=1.0,
        base_noise=0.35,
        repair_detection=0.35,
        repair_success=0.3,
        repair_turns=3.4,
        rework=2.7,
        unauthorized_filter=0.05,
    ),
    "random_state": MethodPolicy(
        method="random_state",
        display_name="Random state",
        type_quality=0.35,
        evidence_quality=0.35,
        status_quality=0.3,
        relation_quality=0.12,
        summary_compression=1.0,
        base_noise=0.7,
        repair_detection=0.25,
        repair_success=0.2,
        repair_turns=3.8,
        rework=3.0,
        unauthorized_filter=0.0,
    ),
    "summary": MethodPolicy(
        method="summary",
        display_name="Summary",
        type_quality=0.58,
        evidence_quality=0.28,
        status_quality=0.45,
        relation_quality=0.22,
        summary_compression=0.48,
        base_noise=0.2,
        repair_detection=0.42,
        repair_success=0.38,
        repair_turns=3.1,
        rework=2.4,
        unauthorized_filter=0.28,
    ),
    "structured_summary": MethodPolicy(
        method="structured_summary",
        display_name="Structured summary",
        type_quality=0.78,
        evidence_quality=0.48,
        status_quality=0.65,
        relation_quality=0.45,
        summary_compression=0.58,
        base_noise=0.18,
        repair_detection=0.55,
        repair_success=0.5,
        repair_turns=2.6,
        rework=1.9,
        unauthorized_filter=0.42,
    ),
    "vector_memory": MethodPolicy(
        method="vector_memory",
        display_name="Vector memory",
        type_quality=0.62,
        evidence_quality=0.58,
        status_quality=0.48,
        relation_quality=0.35,
        summary_compression=0.92,
        base_noise=0.32,
        repair_detection=0.58,
        repair_success=0.55,
        repair_turns=2.3,
        rework=1.7,
        unauthorized_filter=0.18,
    ),
    "autogen_broadcast": MethodPolicy(
        method="autogen_broadcast",
        display_name="AutoGen broadcast",
        type_quality=0.66,
        evidence_quality=0.72,
        status_quality=0.42,
        relation_quality=0.25,
        summary_compression=1.0,
        base_noise=0.68,
        repair_detection=0.5,
        repair_success=0.48,
        repair_turns=2.8,
        rework=2.2,
        unauthorized_filter=0.0,
    ),
    "vacth": MethodPolicy(
        method="vacth",
        display_name="VACTH",
        type_quality=0.96,
        evidence_quality=0.94,
        status_quality=0.94,
        relation_quality=0.9,
        summary_compression=0.72,
        base_noise=0.08,
        repair_detection=0.88,
        repair_success=0.84,
        repair_turns=1.4,
        rework=0.8,
        unauthorized_filter=0.92,
    ),
}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def weighted_mean(values: list[tuple[float, float]]) -> float:
    weight = sum(weight for _, weight in values)
    if weight == 0:
        return 0.0
    return sum(value * item_weight for value, item_weight in values) / weight


def make_item(
    scenario_id: str,
    domain: str,
    suffix: str,
    slot: str,
    content: str,
    *,
    status: str = "observed",
    priority: str = "high",
    created_at: int = 1,
    source: str = "agent",
    evidence: str | None = None,
    active: bool = True,
    gold: bool = True,
    roles: tuple[str, ...] = (),
    token_cost: int | None = None,
    supersedes: str | None = None,
    contradicts: str | None = None,
    sensitive: bool = False,
    allowed_roles: tuple[str, ...] = (),
) -> StateItem:
    if slot not in SLOTS:
        raise ValueError(f"Unknown slot: {slot}")
    if token_cost is None:
        token_cost = max(28, len(content.split()) * 9)
    return StateItem(
        item_id=f"{scenario_id}_{suffix}",
        scenario_id=scenario_id,
        slot=slot,
        content=content,
        status=status,
        priority=priority,
        created_at=created_at,
        source=source,
        evidence_id=evidence,
        active=active,
        gold=gold,
        roles=roles,
        domain=domain,
        token_cost=token_cost,
        supersedes=supersedes,
        contradicts=contradicts,
        sensitive=sensitive,
        allowed_roles=allowed_roles,
    )


def make_scenario(
    scenario_id: str,
    domain: str,
    description: str,
    receiver_roles: tuple[str, ...],
    items: list[StateItem],
    injected_errors: tuple[str, ...],
) -> Scenario:
    edges: list[tuple[str, str, str]] = []
    for item in items:
        if item.supersedes:
            edges.append((item.item_id, "supersedes", item.supersedes))
        if item.contradicts:
            edges.append((item.item_id, "contradicts", item.contradicts))
    return Scenario(
        scenario_id=scenario_id,
        domain=domain,
        description=description,
        receiver_roles=receiver_roles,
        items=tuple(items),
        gold_edges=tuple(edges),
        injected_errors=injected_errors,
    )


def build_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []

    sid = "sw_add_patch"
    old_test = f"{sid}_old_test"
    issue = f"{sid}_issue"
    scenarios.append(
        make_scenario(
            sid,
            "software",
            "calculator.add is fixed after a failing test is replaced by a passing run.",
            ("planner", "executor", "tester", "reviewer"),
            [
                make_item(
                    sid,
                    "software",
                    "constraint",
                    "constraint",
                    "Do not change public add(a, b) behavior or edit tests.",
                    status="active_constraint",
                    priority="critical",
                    created_at=1,
                    source="user",
                    evidence="m1",
                    roles=("planner", "executor", "reviewer"),
                    token_cost=82,
                ),
                make_item(
                    sid,
                    "software",
                    "issue",
                    "unresolved_issue",
                    "calculator.add returns subtraction for add(2, 3).",
                    status="observed",
                    priority="critical",
                    created_at=2,
                    source="tester",
                    evidence="m2",
                    roles=("planner", "executor"),
                    token_cost=76,
                    active=False,
                ),
                make_item(
                    sid,
                    "software",
                    "decision",
                    "decision",
                    "Patch calculator.py only and keep the function signature unchanged.",
                    status="verified",
                    priority="high",
                    created_at=3,
                    source="executor",
                    evidence="m3",
                    roles=("executor", "reviewer"),
                    token_cost=88,
                ),
                make_item(
                    sid,
                    "software",
                    "artifact",
                    "artifact",
                    "calculator.py now returns a + b.",
                    status="verified",
                    priority="high",
                    created_at=4,
                    source="executor",
                    evidence="m4",
                    roles=("tester", "reviewer"),
                    token_cost=64,
                    supersedes=issue,
                ),
                make_item(
                    sid,
                    "software",
                    "old_test",
                    "tool_state_delta",
                    "python -m unittest failed before the patch.",
                    status="superseded",
                    priority="medium",
                    created_at=2,
                    source="tester",
                    evidence="m2",
                    active=False,
                    roles=("tester", "reviewer"),
                    token_cost=70,
                ),
                make_item(
                    sid,
                    "software",
                    "new_test",
                    "tool_state_delta",
                    "python -m unittest returned OK after the patch.",
                    status="verified",
                    priority="critical",
                    created_at=5,
                    source="tester",
                    evidence="m5",
                    roles=("tester", "reviewer"),
                    token_cost=72,
                    supersedes=old_test,
                ),
                make_item(
                    sid,
                    "software",
                    "noise",
                    "fact",
                    "The README contains an unrelated historical note about arithmetic examples.",
                    status="observed",
                    priority="low",
                    created_at=5,
                    source="retriever",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=98,
                ),
            ],
            ("missing_constraint", "stale_tool_state", "missing_evidence"),
        )
    )

    sid = "sw_csv_parser"
    stale_h = f"{sid}_stale_hypothesis"
    scenarios.append(
        make_scenario(
            sid,
            "software",
            "CSV parser diagnosis invalidates an early hypothesis and keeps the parser-scope constraint.",
            ("planner", "executor", "tester", "reviewer"),
            [
                make_item(
                    sid,
                    "software",
                    "constraint",
                    "constraint",
                    "Do not rewrite the entire parser; patch Parser.parse_row only.",
                    status="active_constraint",
                    priority="critical",
                    created_at=1,
                    source="user",
                    evidence="m1",
                    roles=("planner", "executor", "reviewer"),
                    token_cost=90,
                ),
                make_item(
                    sid,
                    "software",
                    "stale_hypothesis",
                    "hypothesis",
                    "The failure is caused by stale bytecode.",
                    status="hypothesis",
                    priority="low",
                    created_at=2,
                    source="planner",
                    evidence="m2",
                    active=False,
                    roles=("planner",),
                    token_cost=62,
                ),
                make_item(
                    sid,
                    "software",
                    "invalidator",
                    "hypothesis",
                    "Clean environment reproduces the quoted comma failure, invalidating stale bytecode.",
                    status="invalidated",
                    priority="high",
                    created_at=3,
                    source="tester",
                    evidence="m3",
                    active=False,
                    roles=("planner", "reviewer"),
                    token_cost=96,
                    contradicts=stale_h,
                ),
                make_item(
                    sid,
                    "software",
                    "fact",
                    "fact",
                    "Parser.parse_row splits on commas without honoring quoted fields.",
                    status="verified",
                    priority="critical",
                    created_at=4,
                    source="retriever",
                    evidence="m4",
                    roles=("executor", "reviewer"),
                    token_cost=86,
                ),
                make_item(
                    sid,
                    "software",
                    "artifact",
                    "artifact",
                    "Parser.parse_row now delegates quoted field handling to csv.reader.",
                    status="verified",
                    priority="high",
                    created_at=5,
                    source="executor",
                    evidence="m5",
                    roles=("tester", "reviewer"),
                    token_cost=82,
                ),
                make_item(
                    sid,
                    "software",
                    "tests",
                    "tool_state_delta",
                    "Quoted comma regression tests returned OK.",
                    status="verified",
                    priority="critical",
                    created_at=6,
                    source="tester",
                    evidence="m6",
                    roles=("tester", "reviewer"),
                    token_cost=62,
                ),
                make_item(
                    sid,
                    "software",
                    "noise",
                    "artifact",
                    "A temporary debug notes file was created during retrieval.",
                    status="observed",
                    priority="low",
                    created_at=6,
                    source="retriever",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=78,
                ),
            ],
            ("wrong_branch", "invalidated_hypothesis", "missing_test_state"),
        )
    )

    sid = "sw_migration"
    open_issue = f"{sid}_open_issue"
    scenarios.append(
        make_scenario(
            sid,
            "software",
            "Database migration keeps a no-downtime constraint and resolves an open batch-size issue.",
            ("planner", "executor", "tester", "reviewer"),
            [
                make_item(
                    sid,
                    "software",
                    "constraint",
                    "constraint",
                    "Avoid downtime during migration and do not lock the full table.",
                    status="active_constraint",
                    priority="critical",
                    created_at=1,
                    source="user",
                    evidence="m1",
                    roles=("planner", "executor", "reviewer"),
                    token_cost=92,
                ),
                make_item(
                    sid,
                    "software",
                    "decision",
                    "decision",
                    "Add nullable slug, backfill in batches, then enforce NOT NULL later.",
                    status="verified",
                    priority="critical",
                    created_at=2,
                    source="planner",
                    evidence="m2",
                    roles=("executor", "reviewer"),
                    token_cost=96,
                ),
                make_item(
                    sid,
                    "software",
                    "open_issue",
                    "unresolved_issue",
                    "Backfill batch size is not yet chosen.",
                    status="observed",
                    priority="high",
                    created_at=3,
                    source="reviewer",
                    evidence="m3",
                    active=False,
                    roles=("planner", "reviewer"),
                    token_cost=64,
                ),
                make_item(
                    sid,
                    "software",
                    "resolved_issue",
                    "decision",
                    "Use 500-row batches for backfill.",
                    status="verified",
                    priority="high",
                    created_at=4,
                    source="planner",
                    evidence="m4",
                    roles=("executor", "reviewer"),
                    token_cost=58,
                    supersedes=open_issue,
                ),
                make_item(
                    sid,
                    "software",
                    "tests",
                    "tool_state_delta",
                    "Migration dry-run completed without table lock warnings.",
                    status="verified",
                    priority="critical",
                    created_at=5,
                    source="tester",
                    evidence="m5",
                    roles=("tester", "reviewer"),
                    token_cost=74,
                ),
                make_item(
                    sid,
                    "software",
                    "noise",
                    "hypothesis",
                    "A marketing page might need slugs later.",
                    status="hypothesis",
                    priority="low",
                    created_at=5,
                    source="planner",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=66,
                ),
            ],
            ("omitted_constraint", "unresolved_issue_after_resolution", "missing_verification"),
        )
    )

    sid = "kn_launch_date"
    conflict_a = f"{sid}_doc_a"
    conflict_b = f"{sid}_doc_b"
    scenarios.append(
        make_scenario(
            sid,
            "knowledge",
            "Two sources disagree about a product launch date and the report must preserve attribution.",
            ("analyst", "reviewer"),
            [
                make_item(
                    sid,
                    "knowledge",
                    "doc_a",
                    "fact",
                    "Document A says the launch happened in 2018.",
                    status="observed",
                    priority="high",
                    created_at=1,
                    source="retriever",
                    evidence="docA:p3",
                    roles=("analyst", "reviewer"),
                    token_cost=76,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "doc_b",
                    "fact",
                    "Document B says the launch happened in March 2019.",
                    status="observed",
                    priority="high",
                    created_at=2,
                    source="retriever",
                    evidence="docB:p7",
                    roles=("analyst", "reviewer"),
                    token_cost=82,
                    contradicts=conflict_a,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "open_issue",
                    "unresolved_issue",
                    "Launch date evidence is contradictory and must not be reported as a single fact.",
                    status="observed",
                    priority="critical",
                    created_at=3,
                    source="analyst",
                    evidence="m3",
                    roles=("analyst", "reviewer"),
                    token_cost=110,
                    contradicts=conflict_b,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "hypothesis",
                    "hypothesis",
                    "The 2018 date may refer to an internal beta rather than public launch.",
                    status="hypothesis",
                    priority="medium",
                    created_at=4,
                    source="analyst",
                    evidence="m4",
                    roles=("analyst",),
                    token_cost=96,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "noise",
                    "fact",
                    "The report template uses blue section headings.",
                    status="observed",
                    priority="low",
                    created_at=4,
                    source="writer",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=64,
                ),
            ],
            ("unsupported_conclusion", "missing_citation", "conflict_collapsed"),
        )
    )

    sid = "kn_policy_comparison"
    old_rule = f"{sid}_old_rule"
    scenarios.append(
        make_scenario(
            sid,
            "knowledge",
            "A policy comparison invalidates an older claim after a newer source is retrieved.",
            ("analyst", "reviewer"),
            [
                make_item(
                    sid,
                    "knowledge",
                    "old_rule",
                    "fact",
                    "The 2021 policy allowed manual exception approval.",
                    status="superseded",
                    priority="medium",
                    created_at=1,
                    source="retriever",
                    evidence="policy2021:s4",
                    active=False,
                    roles=("analyst", "reviewer"),
                    token_cost=82,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "new_rule",
                    "constraint",
                    "The 2024 policy requires automated risk scoring before exception approval.",
                    status="active_constraint",
                    priority="critical",
                    created_at=3,
                    source="retriever",
                    evidence="policy2024:s8",
                    roles=("analyst", "reviewer"),
                    token_cost=98,
                    supersedes=old_rule,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "source_fact",
                    "fact",
                    "The 2024 policy document is the latest source in the corpus.",
                    status="verified",
                    priority="high",
                    created_at=3,
                    source="retriever",
                    evidence="policy2024:metadata",
                    roles=("analyst", "reviewer"),
                    token_cost=74,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "open_question",
                    "unresolved_issue",
                    "The corpus does not include region-specific addenda.",
                    status="observed",
                    priority="high",
                    created_at=4,
                    source="analyst",
                    evidence="m4",
                    roles=("analyst", "reviewer"),
                    token_cost=72,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "noise",
                    "artifact",
                    "A notes table was sorted alphabetically.",
                    status="observed",
                    priority="low",
                    created_at=4,
                    source="analyst",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=56,
                ),
            ],
            ("outdated_fact_as_current", "missing_open_question", "missing_source"),
        )
    )

    sid = "kn_survey_evidence"
    scenarios.append(
        make_scenario(
            sid,
            "knowledge",
            "A survey synthesis separates verified facts from speculative explanation.",
            ("analyst", "reviewer"),
            [
                make_item(
                    sid,
                    "knowledge",
                    "fact1",
                    "fact",
                    "Study S1 reports 62 percent adoption among surveyed hospitals.",
                    status="verified",
                    priority="high",
                    created_at=1,
                    source="retriever",
                    evidence="S1:table2",
                    roles=("analyst", "reviewer"),
                    token_cost=82,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "fact2",
                    "fact",
                    "Study S2 reports 41 percent adoption in rural clinics.",
                    status="verified",
                    priority="high",
                    created_at=2,
                    source="retriever",
                    evidence="S2:results",
                    roles=("analyst", "reviewer"),
                    token_cost=80,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "hypothesis",
                    "hypothesis",
                    "The adoption gap may be caused by procurement budget differences.",
                    status="hypothesis",
                    priority="medium",
                    created_at=3,
                    source="analyst",
                    evidence="m3",
                    roles=("analyst",),
                    token_cost=78,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "open_issue",
                    "unresolved_issue",
                    "No included source directly tests the procurement-budget explanation.",
                    status="observed",
                    priority="critical",
                    created_at=4,
                    source="reviewer",
                    evidence="m4",
                    roles=("analyst", "reviewer"),
                    token_cost=86,
                ),
                make_item(
                    sid,
                    "knowledge",
                    "noise",
                    "fact",
                    "The bibliography file contains 38 entries.",
                    status="observed",
                    priority="low",
                    created_at=4,
                    source="writer",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=60,
                ),
            ],
            ("speculation_as_fact", "missing_citation", "missing_limitation"),
        )
    )

    sid = "ent_refund_approval"
    old_tool = f"{sid}_old_tool"
    old_approval = f"{sid}_old_approval"
    scenarios.append(
        make_scenario(
            sid,
            "enterprise",
            "Refund workflow refreshes order state and invalidates an older approval.",
            ("approver", "auditor", "external_vendor"),
            [
                make_item(
                    sid,
                    "enterprise",
                    "policy",
                    "constraint",
                    "Refunds require current delivery age and manager approval for orders over 30 days.",
                    status="active_constraint",
                    priority="critical",
                    created_at=1,
                    source="policy_agent",
                    evidence="policy:refund:v4",
                    roles=("approver", "auditor"),
                    token_cost=108,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "old_tool",
                    "tool_state_delta",
                    "Order-771 was reported as 45 days after delivery.",
                    status="superseded",
                    priority="medium",
                    created_at=2,
                    source="order_tool",
                    evidence="tool:old",
                    active=False,
                    roles=("approver", "auditor"),
                    token_cost=80,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "new_tool",
                    "tool_state_delta",
                    "Order-771 is 12 days after delivery after cache refresh.",
                    status="verified",
                    priority="critical",
                    created_at=4,
                    source="order_tool",
                    evidence="tool:new",
                    roles=("approver", "auditor"),
                    token_cost=88,
                    supersedes=old_tool,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "old_approval",
                    "decision",
                    "Manual manager approval was requested under the stale 45-day state.",
                    status="superseded",
                    priority="medium",
                    created_at=3,
                    source="approver",
                    evidence="m3",
                    active=False,
                    roles=("approver", "auditor"),
                    token_cost=92,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "new_decision",
                    "decision",
                    "Refund can proceed without manager escalation because the refreshed state is inside 30 days.",
                    status="verified",
                    priority="high",
                    created_at=5,
                    source="approver",
                    evidence="m5",
                    roles=("approver", "auditor"),
                    token_cost=112,
                    supersedes=old_approval,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "customer_email",
                    "fact",
                    "Customer contact is alice@example.com.",
                    status="verified",
                    priority="medium",
                    created_at=2,
                    source="crm_tool",
                    evidence="crm:771",
                    roles=("approver", "auditor"),
                    token_cost=58,
                    sensitive=True,
                    allowed_roles=("approver", "auditor"),
                ),
                make_item(
                    sid,
                    "enterprise",
                    "noise",
                    "artifact",
                    "The dashboard row is colored amber.",
                    status="observed",
                    priority="low",
                    created_at=5,
                    source="ui",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=54,
                ),
            ],
            ("expired_approval", "stale_tool_state", "sensitive_leak"),
        )
    )

    sid = "ent_access_request"
    old_policy = f"{sid}_old_policy"
    scenarios.append(
        make_scenario(
            sid,
            "enterprise",
            "Access request policy changes from team-lead approval to security approval.",
            ("approver", "auditor", "external_vendor"),
            [
                make_item(
                    sid,
                    "enterprise",
                    "old_policy",
                    "constraint",
                    "Team lead approval was sufficient for repository access.",
                    status="superseded",
                    priority="medium",
                    created_at=1,
                    source="policy_agent",
                    evidence="policy:2023",
                    active=False,
                    roles=("approver", "auditor"),
                    token_cost=76,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "new_policy",
                    "constraint",
                    "Security approval is required for production repository access.",
                    status="active_constraint",
                    priority="critical",
                    created_at=4,
                    source="policy_agent",
                    evidence="policy:2026",
                    roles=("approver", "auditor"),
                    token_cost=84,
                    supersedes=old_policy,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "pending",
                    "unresolved_issue",
                    "Security approval is still pending for request AR-22.",
                    status="observed",
                    priority="critical",
                    created_at=5,
                    source="approver",
                    evidence="workflow:AR-22",
                    roles=("approver", "auditor"),
                    token_cost=74,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "secret",
                    "fact",
                    "Temporary access token is tok_live_9381.",
                    status="verified",
                    priority="high",
                    created_at=2,
                    source="vault",
                    evidence="vault:AR-22",
                    roles=("approver",),
                    token_cost=62,
                    sensitive=True,
                    allowed_roles=("approver",),
                ),
                make_item(
                    sid,
                    "enterprise",
                    "audit",
                    "artifact",
                    "Audit chain includes requester, approver, policy version, and vault lookup id.",
                    status="verified",
                    priority="critical",
                    created_at=5,
                    source="auditor",
                    evidence="audit:AR-22",
                    roles=("auditor",),
                    token_cost=96,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "noise",
                    "fact",
                    "The request form contains a department dropdown.",
                    status="observed",
                    priority="low",
                    created_at=5,
                    source="ui",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=60,
                ),
            ],
            ("old_policy_reuse", "missing_approval_state", "sensitive_leak"),
        )
    )

    sid = "ent_invoice_exception"
    old_state = f"{sid}_old_state"
    scenarios.append(
        make_scenario(
            sid,
            "enterprise",
            "Invoice exception workflow must preserve the latest approval and audit evidence.",
            ("approver", "auditor", "external_vendor"),
            [
                make_item(
                    sid,
                    "enterprise",
                    "policy",
                    "constraint",
                    "Invoices over 10000 require finance approval and audit trail retention.",
                    status="active_constraint",
                    priority="critical",
                    created_at=1,
                    source="policy_agent",
                    evidence="policy:invoice:v5",
                    roles=("approver", "auditor"),
                    token_cost=100,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "old_state",
                    "tool_state_delta",
                    "Invoice INV-9 was initially read as 9800.",
                    status="superseded",
                    priority="medium",
                    created_at=2,
                    source="erp_tool",
                    evidence="erp:old",
                    active=False,
                    roles=("approver", "auditor"),
                    token_cost=72,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "new_state",
                    "tool_state_delta",
                    "Invoice INV-9 amount is 12800 after currency refresh.",
                    status="verified",
                    priority="critical",
                    created_at=3,
                    source="erp_tool",
                    evidence="erp:new",
                    roles=("approver", "auditor"),
                    token_cost=82,
                    supersedes=old_state,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "decision",
                    "decision",
                    "Hold payment until finance approval is recorded.",
                    status="verified",
                    priority="critical",
                    created_at=4,
                    source="approver",
                    evidence="workflow:INV-9",
                    roles=("approver", "auditor"),
                    token_cost=70,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "audit",
                    "artifact",
                    "Audit chain links policy version, refreshed ERP amount, and hold decision.",
                    status="verified",
                    priority="critical",
                    created_at=5,
                    source="auditor",
                    evidence="audit:INV-9",
                    roles=("auditor",),
                    token_cost=86,
                ),
                make_item(
                    sid,
                    "enterprise",
                    "vendor_bank",
                    "fact",
                    "Vendor bank tail is 0182.",
                    status="verified",
                    priority="medium",
                    created_at=2,
                    source="erp_tool",
                    evidence="erp:vendor",
                    roles=("approver", "auditor"),
                    token_cost=52,
                    sensitive=True,
                    allowed_roles=("approver", "auditor"),
                ),
                make_item(
                    sid,
                    "enterprise",
                    "noise",
                    "fact",
                    "The invoice PDF has three pages.",
                    status="observed",
                    priority="low",
                    created_at=5,
                    source="erp_tool",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=52,
                ),
            ],
            ("stale_amount", "missing_audit_chain", "sensitive_leak"),
        )
    )

    scenarios.extend(build_expanded_scenarios())
    return scenarios


def build_expanded_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    scenarios.extend(build_expanded_software_scenarios())
    scenarios.extend(build_expanded_knowledge_scenarios())
    scenarios.extend(build_expanded_enterprise_scenarios())
    return scenarios


def build_expanded_software_scenarios() -> list[Scenario]:
    cases = [
        (
            "cache_ttl",
            "cache TTL refresh",
            "Cache entries ignore tenant-specific TTL overrides.",
            "tenant TTL override",
            "cache.py",
            "tenant_ttl",
            "cache invalidation tests",
            "Do not change the default TTL for tenants without overrides.",
            "The failure is caused by Redis clock drift.",
        ),
        (
            "email_retry",
            "email retry idempotency",
            "Retry sends duplicate welcome emails.",
            "send_attempt_id idempotency key",
            "email_retry.py",
            "send_attempt_id",
            "email idempotency tests",
            "Do not suppress the first successful welcome email.",
            "The SMTP provider always suppresses duplicate messages.",
        ),
        (
            "invoice_tax",
            "invoice tax rounding",
            "Tax is rounded per line before summing.",
            "round after summing taxable lines",
            "invoice.py",
            "round_after_sum",
            "invoice rounding tests",
            "Do not change non-tax invoice totals.",
            "The fixture total is wrong.",
        ),
        (
            "feature_flag",
            "feature flag rollout",
            "Beta UI is enabled for non-allowlisted tenants.",
            "tenant allowlist guard",
            "flags.py",
            "tenant_allowlist",
            "feature flag access tests",
            "Do not expose beta UI outside the allowlist.",
            "Rollout percentage is zero for all tenants.",
        ),
        (
            "quota_burst",
            "quota burst accounting",
            "Burst requests are double-counted at the minute boundary.",
            "idempotent boundary window accounting",
            "quota.py",
            "boundary_window",
            "quota boundary tests",
            "Do not raise the default quota limit.",
            "The client clock is always behind the server.",
        ),
        (
            "locale_slug",
            "localized slug generation",
            "Localized titles collide after accent stripping.",
            "locale-aware uniqueness suffix",
            "slug.py",
            "locale_suffix",
            "localized slug tests",
            "Do not change existing English-only slugs.",
            "The database collation already prevents collisions.",
        ),
        (
            "payment_timeout",
            "payment timeout handling",
            "Timeout retries can charge the same invoice twice.",
            "idempotency token on retry",
            "payments.py",
            "retry_token",
            "payment idempotency tests",
            "Do not retry declined cards.",
            "The gateway never returns duplicate success callbacks.",
        ),
        (
            "search_filter",
            "search filter composition",
            "Archived records appear when owner and tag filters are combined.",
            "compose archive predicate with owner and tag filters",
            "search.py",
            "archive_predicate",
            "search filter tests",
            "Do not change full-text ranking.",
            "The issue is caused by stale search indexes only.",
        ),
        (
            "timezone_export",
            "timezone export boundary",
            "Events at midnight UTC are exported into the wrong local day.",
            "timezone-aware local-day conversion",
            "export.py",
            "local_day",
            "timezone export tests",
            "Do not change stored UTC timestamps.",
            "The test environment timezone is misconfigured.",
        ),
        (
            "webhook_order",
            "webhook event ordering",
            "Out-of-order delivery marks paid invoices as unpaid.",
            "monotonic event version check",
            "webhooks.py",
            "event_version",
            "webhook ordering tests",
            "Do not discard newer paid events.",
            "The payment provider guarantees ordered callbacks.",
        ),
        (
            "csv_escape",
            "CSV escape handling",
            "Escaped quotes are dropped in exported CSV fields.",
            "preserve doubled quotes inside quoted fields",
            "csv_export.py",
            "quote_escape",
            "CSV escape tests",
            "Do not change delimiter configuration.",
            "The downstream spreadsheet importer strips quotes.",
        ),
        (
            "inventory_hold",
            "inventory hold release",
            "Expired holds still reduce available stock.",
            "release expired holds before availability check",
            "inventory.py",
            "expired_hold_release",
            "inventory hold tests",
            "Do not oversell active holds.",
            "Warehouse sync lag is the only cause.",
        ),
    ]
    scenarios: list[Scenario] = []
    for idx, (
        case_id,
        title,
        issue_text,
        decision_text,
        artifact_file,
        artifact_key,
        test_name,
        constraint_text,
        stale_hypothesis,
    ) in enumerate(cases, start=1):
        sid = f"sw_gen_{case_id}"
        stale_test = f"{sid}_stale_test"
        stale_h = f"{sid}_stale_hypothesis"
        unresolved = f"{sid}_issue"
        review_gap = f"{sid}_review_gap"
        items = [
            make_item(
                sid,
                "software",
                "primary_constraint",
                "constraint",
                constraint_text,
                status="active_constraint",
                priority="critical",
                created_at=1,
                source="user",
                evidence=f"{sid}:m1",
                roles=("planner", "executor", "reviewer"),
                token_cost=118,
            ),
            make_item(
                sid,
                "software",
                "secondary_constraint",
                "constraint",
                f"Keep public APIs stable while repairing {title}.",
                status="active_constraint",
                priority="high",
                created_at=1,
                source="planner",
                evidence=f"{sid}:m2",
                roles=("planner", "executor", "reviewer"),
                token_cost=104,
            ),
            make_item(
                sid,
                "software",
                "issue",
                "unresolved_issue",
                issue_text,
                status="observed",
                priority="critical",
                created_at=2,
                source="tester",
                evidence=f"{sid}:m3",
                active=False,
                roles=("planner", "executor"),
                token_cost=106,
            ),
            make_item(
                sid,
                "software",
                "stale_test",
                "tool_state_delta",
                f"{test_name} failed before the patch.",
                status="superseded",
                priority="high",
                created_at=2,
                source="tester",
                evidence=f"{sid}:m4",
                active=False,
                roles=("tester", "reviewer"),
                token_cost=96,
            ),
            make_item(
                sid,
                "software",
                "stale_hypothesis",
                "hypothesis",
                stale_hypothesis,
                status="hypothesis",
                priority="medium",
                created_at=3,
                source="planner",
                evidence=f"{sid}:m5",
                active=False,
                roles=("planner", "reviewer"),
                token_cost=112,
            ),
            make_item(
                sid,
                "software",
                "invalidated_hypothesis",
                "hypothesis",
                f"Fresh reproduction disproves the earlier explanation: {stale_hypothesis}",
                status="invalidated",
                priority="high",
                created_at=4,
                source="retriever",
                evidence=f"{sid}:m6",
                active=False,
                roles=("planner", "reviewer"),
                token_cost=128,
                contradicts=stale_h,
            ),
            make_item(
                sid,
                "software",
                "decision",
                "decision",
                f"Patch {artifact_file} by adding {decision_text}.",
                status="verified",
                priority="critical",
                created_at=5,
                source="planner",
                evidence=f"{sid}:m7",
                roles=("executor", "reviewer"),
                token_cost=116,
            ),
            make_item(
                sid,
                "software",
                "artifact",
                "artifact",
                f"{artifact_file} now implements {artifact_key}.",
                status="verified",
                priority="critical",
                created_at=6,
                source="executor",
                evidence=f"{sid}:m8",
                roles=("tester", "reviewer"),
                token_cost=96,
                supersedes=unresolved,
            ),
            make_item(
                sid,
                "software",
                "new_test",
                "tool_state_delta",
                f"{test_name} returned OK after the patch.",
                status="verified",
                priority="critical",
                created_at=7,
                source="tester",
                evidence=f"{sid}:m9",
                roles=("tester", "reviewer"),
                token_cost=100,
                supersedes=stale_test,
            ),
            make_item(
                sid,
                "software",
                "review_gap",
                "unresolved_issue",
                f"Regression coverage for {title} has one edge case left for reviewer sign-off.",
                status="observed",
                priority="high",
                created_at=8,
                source="reviewer",
                evidence=f"{sid}:m10",
                roles=("planner", "reviewer"),
                token_cost=120,
            ),
            make_item(
                sid,
                "software",
                "resolved_review_gap",
                "tool_state_delta",
                f"Reviewer sign-off completed for the remaining {title} edge case.",
                status="verified",
                priority="high",
                created_at=9,
                source="reviewer",
                evidence=f"{sid}:m11",
                roles=("reviewer",),
                token_cost=96,
                supersedes=review_gap,
            ),
        ]
        for noise_idx in range(4):
            items.append(
                make_item(
                    sid,
                    "software",
                    f"noise_{noise_idx}",
                    "fact",
                    f"Unrelated repository note {noise_idx + 1} for {title} mentions docs, screenshots, or style cleanup.",
                    status="observed",
                    priority="low",
                    created_at=3 + noise_idx,
                    source="retriever",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=120 + 8 * noise_idx,
                )
            )
        scenarios.append(
            make_scenario(
                sid,
                "software",
                f"Generated hard software handoff case {idx}: {title}.",
                ("planner", "executor", "tester", "reviewer"),
                items,
                ("missing_constraint", "stale_tool_state", "invalidated_hypothesis", "missing_evidence"),
            )
        )
    return scenarios


def build_expanded_knowledge_scenarios() -> list[Scenario]:
    topics = [
        ("hospital_ai", "hospital AI adoption", "62 percent", "41 percent", "procurement budgets"),
        ("battery_recycling", "battery recycling capacity", "18 GWh", "24 GWh", "regional subsidy timing"),
        ("model_eval", "model evaluation drift", "May benchmark", "June benchmark", "dataset contamination"),
        (
            "climate_risk",
            "climate risk disclosure",
            "Scope 1 emissions",
            "Scope 3 supplier emissions",
            "reporting boundary",
        ),
        ("education_policy", "education policy impact", "urban districts", "rural districts", "teacher vacancy"),
        ("drug_trial", "drug trial subgroup", "primary endpoint", "secondary endpoint", "sample-size imbalance"),
        ("supply_chain", "supply chain delay", "port congestion", "rail bottleneck", "inventory buffering"),
        ("cyber_incident", "cyber incident timeline", "first detection", "first containment", "log retention gap"),
        ("water_usage", "water usage audit", "metered usage", "self-reported usage", "seasonality"),
        ("chip_exports", "chip export policy", "licensing threshold", "country exemption", "temporary waiver"),
        ("remote_work", "remote work productivity", "survey measure", "ticket throughput", "team composition"),
        ("public_transit", "public transit ridership", "weekday recovery", "weekend recovery", "fare integration"),
        (
            "telemedicine_access",
            "telemedicine access",
            "rural claims data",
            "urban appointment logs",
            "broadband coverage",
        ),
        (
            "carbon_offsets",
            "carbon offset quality",
            "registry retirement count",
            "field audit count",
            "additionality screening",
        ),
        ("ai_regulation", "AI regulation compliance", "provider duty", "deployer duty", "risk-tier definitions"),
        ("rare_disease", "rare disease registry", "confirmed diagnosis", "suspected diagnosis", "referral bias"),
        (
            "semiconductor_yield",
            "semiconductor yield learning",
            "pilot-line yield",
            "volume-line yield",
            "tool calibration",
        ),
        (
            "green_hydrogen",
            "green hydrogen cost",
            "electrolyzer capex",
            "delivered fuel cost",
            "electricity contract terms",
        ),
        (
            "food_inflation",
            "food inflation drivers",
            "retail basket index",
            "farm-gate price index",
            "transport pass-through",
        ),
        ("ocean_plastic", "ocean plastic monitoring", "beach survey mass", "surface trawl count", "sampling depth"),
        (
            "ev_charging",
            "EV charging reliability",
            "charger uptime",
            "successful session rate",
            "payment failure logging",
        ),
        (
            "privacy_breach",
            "privacy breach impact",
            "notified users",
            "confirmed exposed records",
            "deduplication method",
        ),
        (
            "lab_reproducibility",
            "lab reproducibility audit",
            "protocol match rate",
            "result match rate",
            "operator training",
        ),
        ("housing_supply", "housing supply pipeline", "permits issued", "units completed", "financing delay"),
        (
            "crop_yield",
            "crop yield projection",
            "satellite vegetation index",
            "field harvest estimate",
            "irrigation coverage",
        ),
        ("cloud_outage", "cloud outage analysis", "first alert", "customer-impact start", "clock synchronization"),
        ("bank_capital", "bank capital stress", "CET1 baseline", "stressed CET1", "risk-weight model"),
        (
            "teacher_retention",
            "teacher retention program",
            "one-year retention",
            "three-year retention",
            "district mix",
        ),
        (
            "water_quality",
            "water quality monitoring",
            "sensor exceedance",
            "lab-confirmed exceedance",
            "sample handling",
        ),
        (
            "renewable_grid",
            "renewable grid curtailment",
            "scheduled curtailment",
            "actual curtailment",
            "transmission outage",
        ),
        (
            "clinical_ai",
            "clinical AI validation",
            "AUROC on validation set",
            "calibration on deployment set",
            "case-mix shift",
        ),
        ("microfinance", "microfinance repayment", "portfolio-at-risk", "write-off rate", "seasonal income shock"),
        (
            "disaster_relief",
            "disaster relief allocation",
            "registered households",
            "verified damaged homes",
            "duplicate claims",
        ),
        (
            "language_policy",
            "language policy outcomes",
            "test-score gain",
            "attendance gain",
            "implementation fidelity",
        ),
        (
            "antibiotic_resistance",
            "antibiotic resistance surveillance",
            "hospital isolate rate",
            "community isolate rate",
            "testing protocol",
        ),
        ("online_ads", "online ad measurement", "click-through lift", "conversion lift", "attribution window"),
        ("mental_health", "mental health intervention", "symptom score", "service utilization", "dropout adjustment"),
        (
            "freight_emissions",
            "freight emissions accounting",
            "fuel purchase estimate",
            "telematics estimate",
            "empty-mile allocation",
        ),
        ("startup_funding", "startup funding trend", "announced deals", "closed deals", "reporting lag"),
        ("wildfire_risk", "wildfire risk model", "historical burn area", "projected exposure", "vegetation treatment"),
        (
            "public_health_alert",
            "public health alert timing",
            "case confirmation date",
            "public notice date",
            "lab backlog",
        ),
        (
            "digital_identity",
            "digital identity adoption",
            "accounts created",
            "monthly active users",
            "duplicate-account cleanup",
        ),
    ]
    scenarios: list[Scenario] = []
    for idx, (topic_id, topic, claim_a, claim_b, hypothesis) in enumerate(topics, start=1):
        sid = f"kn_gen_{topic_id}"
        old_source = f"{sid}_old_source"
        doc_a = f"{sid}_doc_a"
        doc_b = f"{sid}_doc_b"
        open_issue = f"{sid}_open_issue"
        items = [
            make_item(
                sid,
                "knowledge",
                "old_source",
                "fact",
                f"The 2020 source describes {topic} using an outdated baseline.",
                status="superseded",
                priority="medium",
                created_at=1,
                source="retriever",
                evidence=f"{sid}:old:p1",
                active=False,
                roles=("analyst", "reviewer"),
                token_cost=104,
            ),
            make_item(
                sid,
                "knowledge",
                "latest_source",
                "constraint",
                f"The 2025 source is the newest source for {topic} and supersedes the 2020 baseline.",
                status="active_constraint",
                priority="critical",
                created_at=4,
                source="retriever",
                evidence=f"{sid}:new:metadata",
                roles=("analyst", "reviewer"),
                token_cost=126,
                supersedes=old_source,
            ),
            make_item(
                sid,
                "knowledge",
                "doc_a",
                "fact",
                f"Source A reports {claim_a} for {topic}.",
                status="verified",
                priority="high",
                created_at=2,
                source="retriever",
                evidence=f"{sid}:sourceA:p2",
                roles=("analyst", "reviewer"),
                token_cost=96,
            ),
            make_item(
                sid,
                "knowledge",
                "doc_b",
                "fact",
                f"Source B reports {claim_b} for {topic}.",
                status="verified",
                priority="high",
                created_at=3,
                source="retriever",
                evidence=f"{sid}:sourceB:p4",
                roles=("analyst", "reviewer"),
                token_cost=96,
                contradicts=doc_a,
            ),
            make_item(
                sid,
                "knowledge",
                "doc_c",
                "fact",
                f"Source C reports a narrower subgroup result for {topic} and warns against direct aggregation.",
                status="verified",
                priority="high",
                created_at=4,
                source="retriever",
                evidence=f"{sid}:sourceC:p6",
                roles=("analyst", "reviewer"),
                token_cost=104,
                contradicts=doc_b if idx % 3 == 0 else None,
            ),
            make_item(
                sid,
                "knowledge",
                "doc_d",
                "fact",
                f"Source D uses a different denominator when measuring {topic}.",
                status="observed",
                priority="high",
                created_at=5,
                source="retriever",
                evidence=f"{sid}:sourceD:methods",
                roles=("analyst", "reviewer"),
                token_cost=112,
            ),
            make_item(
                sid,
                "knowledge",
                "method_caveat",
                "constraint",
                f"The final analysis of {topic} must report source definitions before comparing numbers.",
                status="active_constraint",
                priority="critical",
                created_at=6,
                source="reviewer",
                evidence=f"{sid}:review:m6",
                roles=("analyst", "reviewer"),
                token_cost=118,
            ),
            make_item(
                sid,
                "knowledge",
                "open_issue",
                "unresolved_issue",
                f"The apparent discrepancy in {topic} must be kept open until source definitions are reconciled.",
                status="observed",
                priority="critical",
                created_at=7,
                source="analyst",
                evidence=f"{sid}:analysis:m7",
                roles=("analyst", "reviewer"),
                token_cost=136,
                contradicts=doc_b,
            ),
            make_item(
                sid,
                "knowledge",
                "hypothesis",
                "hypothesis",
                f"The discrepancy may be explained by {hypothesis}.",
                status="hypothesis",
                priority="medium",
                created_at=8,
                source="analyst",
                evidence=f"{sid}:analysis:m8",
                roles=("analyst",),
                token_cost=106,
            ),
            make_item(
                sid,
                "knowledge",
                "limitation",
                "unresolved_issue",
                f"No source directly tests whether {hypothesis} explains {topic}.",
                status="observed",
                priority="high",
                created_at=9,
                source="reviewer",
                evidence=f"{sid}:review:m9",
                roles=("analyst", "reviewer"),
                token_cost=112,
            ),
            make_item(
                sid,
                "knowledge",
                "recency_gap",
                "unresolved_issue",
                f"No retrieved source checks whether the latest observed pattern for {topic} continued after 2025.",
                status="observed",
                priority="medium",
                created_at=10,
                source="reviewer",
                evidence=f"{sid}:review:m10",
                roles=("analyst", "reviewer"),
                token_cost=112,
            ),
            make_item(
                sid,
                "knowledge",
                "resolved_note",
                "fact",
                f"Appendix terminology for {topic} was normalized across sources.",
                status="verified",
                priority="medium",
                created_at=11,
                source="writer",
                evidence=f"{sid}:appendix:m11",
                roles=("reviewer",),
                token_cost=86,
                supersedes=open_issue if idx % 4 == 0 else None,
            ),
        ]
        for noise_idx in range(6):
            items.append(
                make_item(
                    sid,
                    "knowledge",
                    f"noise_{noise_idx}",
                    "artifact",
                    f"Formatting note {noise_idx + 1} for the {topic} report is unrelated to evidence synthesis.",
                    status="observed",
                    priority="low",
                    created_at=3 + noise_idx,
                    source="writer",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=104,
                )
            )
        scenarios.append(
            make_scenario(
                sid,
                "knowledge",
                f"Generated hard knowledge-analysis case {idx}: {topic}.",
                ("analyst", "reviewer"),
                items,
                ("missing_citation", "conflict_collapsed", "unsupported_conclusion", "outdated_source"),
            )
        )
    return scenarios


def build_expanded_enterprise_scenarios() -> list[Scenario]:
    cases = [
        ("refund_plus", "refund exception", "order age", "manager approval", "customer email"),
        ("access_prod", "production access", "security approval", "temporary token", "vault lookup"),
        ("invoice_hold", "invoice payment hold", "finance approval", "bank tail", "ERP amount"),
        ("vendor_onboard", "vendor onboarding", "sanctions check", "tax id", "supplier score"),
        ("expense_override", "expense override", "director approval", "employee email", "receipt OCR"),
        ("data_export", "customer data export", "DPA approval", "raw access token", "export manifest"),
        ("incident_postmortem", "incident postmortem", "security sign-off", "user email", "log bundle"),
        ("contract_amend", "contract amendment", "legal approval", "counterparty bank", "redline id"),
        ("credit_limit", "credit limit change", "risk approval", "customer phone", "risk score"),
        ("device_return", "device return", "asset approval", "serial number", "warehouse scan"),
        ("pricing_exception", "pricing exception", "finance approval", "customer discount", "quote id"),
        ("partner_api", "partner API enablement", "privacy approval", "client secret", "scope grant"),
    ]
    scenarios: list[Scenario] = []
    for idx, (case_id, workflow, approval, secret_kind, audit_object) in enumerate(cases, start=1):
        sid = f"ent_gen_{case_id}"
        old_policy = f"{sid}_old_policy"
        stale_state = f"{sid}_stale_state"
        stale_decision = f"{sid}_stale_decision"
        items = [
            make_item(
                sid,
                "enterprise",
                "old_policy",
                "constraint",
                f"Legacy policy allowed {workflow} with team-lead approval only.",
                status="superseded",
                priority="medium",
                created_at=1,
                source="policy_agent",
                evidence=f"{sid}:policy:old",
                active=False,
                roles=("approver", "auditor"),
                token_cost=106,
            ),
            make_item(
                sid,
                "enterprise",
                "new_policy",
                "constraint",
                f"Current policy requires {approval} before completing {workflow}.",
                status="active_constraint",
                priority="critical",
                created_at=4,
                source="policy_agent",
                evidence=f"{sid}:policy:new",
                roles=("approver", "auditor"),
                token_cost=120,
                supersedes=old_policy,
            ),
            make_item(
                sid,
                "enterprise",
                "stale_state",
                "tool_state_delta",
                f"The first tool read for {workflow} returned a stale low-risk state.",
                status="superseded",
                priority="high",
                created_at=2,
                source="workflow_tool",
                evidence=f"{sid}:tool:old",
                active=False,
                roles=("approver", "auditor"),
                token_cost=112,
            ),
            make_item(
                sid,
                "enterprise",
                "fresh_state",
                "tool_state_delta",
                f"The refreshed tool read for {workflow} requires escalation.",
                status="verified",
                priority="critical",
                created_at=5,
                source="workflow_tool",
                evidence=f"{sid}:tool:new",
                roles=("approver", "auditor"),
                token_cost=104,
                supersedes=stale_state,
            ),
            make_item(
                sid,
                "enterprise",
                "stale_decision",
                "decision",
                f"Proceed decision was drafted under the stale state for {workflow}.",
                status="superseded",
                priority="medium",
                created_at=3,
                source="approver",
                evidence=f"{sid}:decision:old",
                active=False,
                roles=("approver", "auditor"),
                token_cost=104,
            ),
            make_item(
                sid,
                "enterprise",
                "current_decision",
                "decision",
                f"Hold {workflow} until {approval} is recorded.",
                status="verified",
                priority="critical",
                created_at=6,
                source="approver",
                evidence=f"{sid}:decision:new",
                roles=("approver", "auditor"),
                token_cost=96,
                supersedes=stale_decision,
            ),
            make_item(
                sid,
                "enterprise",
                "pending_issue",
                "unresolved_issue",
                f"{approval} is still pending for {workflow}.",
                status="observed",
                priority="critical",
                created_at=7,
                source="approver",
                evidence=f"{sid}:workflow:pending",
                roles=("approver", "auditor"),
                token_cost=94,
            ),
            make_item(
                sid,
                "enterprise",
                "audit",
                "artifact",
                f"Audit chain links policy version, refreshed state, {approval}, and {audit_object}.",
                status="verified",
                priority="critical",
                created_at=8,
                source="auditor",
                evidence=f"{sid}:audit",
                roles=("auditor",),
                token_cost=126,
            ),
            make_item(
                sid,
                "enterprise",
                "secret",
                "fact",
                f"Sensitive {secret_kind} is present in the workflow record.",
                status="verified",
                priority="high",
                created_at=5,
                source="vault",
                evidence=f"{sid}:vault",
                roles=("approver", "auditor"),
                token_cost=84,
                sensitive=True,
                allowed_roles=("approver", "auditor"),
            ),
            make_item(
                sid,
                "enterprise",
                "vendor_minimal",
                "constraint",
                f"External vendors may only receive the final allowed/blocked status for {workflow}.",
                status="active_constraint",
                priority="high",
                created_at=8,
                source="policy_agent",
                evidence=f"{sid}:policy:vendor",
                roles=("external_vendor", "auditor"),
                token_cost=108,
            ),
        ]
        for noise_idx in range(4):
            items.append(
                make_item(
                    sid,
                    "enterprise",
                    f"noise_{noise_idx}",
                    "fact",
                    f"Workflow UI note {noise_idx + 1} for {workflow} is unrelated to approval or audit routing.",
                    status="observed",
                    priority="low",
                    created_at=3 + noise_idx,
                    source="ui",
                    evidence=None,
                    gold=False,
                    roles=(),
                    token_cost=112,
                )
            )
        scenarios.append(
            make_scenario(
                sid,
                "enterprise",
                f"Generated hard enterprise workflow case {idx}: {workflow}.",
                ("approver", "auditor", "external_vendor"),
                items,
                ("old_policy_reuse", "stale_tool_state", "expired_approval", "sensitive_leak"),
            )
        )
    return scenarios


def truth_active(item: StateItem) -> bool:
    return item.active if item.gold_active is None else item.gold_active


def truth_roles(item: StateItem) -> tuple[str, ...]:
    return item.roles if item.gold_roles is None else item.gold_roles


def scenario_rng(seed: int, scenario_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{scenario_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_hardened_scenarios(base_scenarios: list[Scenario]) -> list[Scenario]:
    """Create frozen stress variants without exposing evaluation labels to selectors.

    The variants model realistic extraction and routing errors: longer state records,
    missing evidence, incomplete role tags, delayed stale-state invalidation, and
    high-salience distractors that look actionable from metadata alone.
    """

    hardened: list[Scenario] = []
    for seed in HARDENING_SEEDS:
        for scenario in base_scenarios:
            rng = scenario_rng(seed, scenario.scenario_id)
            latest_turn = max(item.created_at for item in scenario.items)
            observed_items: list[StateItem] = []

            for item in scenario.items:
                updates: dict[str, Any] = {
                    "gold_active": item.active,
                    "gold_roles": item.roles,
                    "token_cost": round(item.token_cost * rng.uniform(1.25, 1.75)),
                }

                if item.gold and item.active:
                    if item.evidence_id and rng.random() < 0.25:
                        updates["evidence_id"] = None
                    if (item.supersedes or item.contradicts) and rng.random() < 0.20:
                        updates["supersedes"] = None
                        updates["contradicts"] = None
                    if item.roles and rng.random() < 0.18:
                        visible_roles = list(item.roles)
                        visible_roles.pop(rng.randrange(len(visible_roles)))
                        updates["roles"] = tuple(visible_roles)
                    if rng.random() < 0.08:
                        updates["active"] = False
                        updates["status"] = "superseded"
                elif item.gold and not item.active and rng.random() < 0.42:
                    # The evaluator knows this state is stale, but the handoff index
                    # has not yet observed the superseding edge.
                    updates["active"] = True
                    updates["status"] = rng.choice(("observed", "verified"))
                    updates["created_at"] = latest_turn + rng.randint(-1, 1)

                observed_items.append(replace(item, **updates))

            for role in scenario.receiver_roles:
                ranked_slots = sorted(
                    ROLE_SLOT_VALUE.get(role, ROLE_SLOT_VALUE["global"]),
                    key=ROLE_SLOT_VALUE.get(role, ROLE_SLOT_VALUE["global"]).get,
                    reverse=True,
                )
                for distractor_idx, slot in enumerate(ranked_slots[:2], start=1):
                    status = "active_constraint" if slot == "constraint" else rng.choice(("observed", "verified"))
                    distractor_turn = latest_turn + 1 if rng.random() < 0.25 else rng.randint(1, max(1, latest_turn))
                    observed_items.append(
                        make_item(
                            scenario.scenario_id,
                            scenario.domain,
                            f"hard_negative_s{seed}_{role}_{distractor_idx}",
                            slot,
                            (
                                f"Current {slot} note related to {scenario.description}; an adjacent component "
                                f"reports a similar condition and routes the note to {role}."
                            ),
                            status=status,
                            priority="high" if distractor_idx == 1 else "medium",
                            created_at=distractor_turn,
                            source="retrieved_neighbor",
                            evidence=f"neighbor:{seed}:{role}:{distractor_idx}",
                            active=True,
                            gold=False,
                            roles=(role,),
                            token_cost=rng.randint(90, 145),
                        )
                    )

            hardened.append(
                replace(
                    scenario,
                    scenario_id=f"{scenario.scenario_id}__stress_{seed}",
                    description=f"Hardened seed {seed}: {scenario.description}",
                    items=tuple(observed_items),
                    injected_errors=tuple(
                        sorted(
                            set(scenario.injected_errors)
                            | {
                                "metadata_dropout",
                                "delayed_stale_invalidation",
                                "semantic_hard_negative",
                                "budget_overload",
                            }
                        )
                    ),
                    difficulty="hardened",
                    stress_seed=seed,
                )
            )
    return hardened


def item_value(item: StateItem, method: str, receiver: str, now_turn: int) -> float:
    role_values = ROLE_SLOT_VALUE.get(receiver, ROLE_SLOT_VALUE["reviewer"])
    role_value = role_values.get(item.slot, 0.35)
    recency = 1.0 / max(1, now_turn - item.created_at + 1)
    priority = PRIORITY_VALUE[item.priority]
    status = STATUS_VALUE[item.status]
    evidence = 0.8 if item.evidence_id else 0.1
    active_bonus = 0.4 if item.active else -0.35
    if method == "vacth":
        return 1.2 * role_value + 0.7 * priority + 0.7 * status + 0.35 * evidence + active_bonus
    if method == "vector_memory":
        lexical = 0.55 if receiver in item.roles or item.slot in {"fact", "tool_state_delta"} else 0.2
        stale_penalty = -0.05 if not item.active else 0.0
        return lexical + 0.35 * priority + 0.35 * recency + 0.2 * evidence + stale_penalty
    if method == "structured_summary":
        active = 0.55 if item.active else -0.2
        return 0.45 * role_value + 0.45 * priority + 0.4 * status + active
    if method == "summary":
        salience = 0.45 if item.priority in {"critical", "high"} else 0.1
        collapsed = -0.15 if item.slot in {"hypothesis", "tool_state_delta"} else 0.05
        return salience + 0.3 * recency + collapsed + (0.2 if item.active else -0.1)
    return 0.0


def item_cost(item: StateItem, method: str) -> int:
    # Conservative serialization estimates. VACTH retains typed metadata and
    # provenance, so it receives only a modest compression factor.
    return max(24, round(item.token_cost * SERIALIZATION_COST_SCALE[method]))


def chronological(items: list[StateItem]) -> list[StateItem]:
    return sorted(items, key=lambda item: (item.created_at, item.item_id))


def select_for_method(
    scenario: Scenario,
    method: str,
    *,
    receiver: str,
    budget: int | None,
) -> Selection:
    items = list(scenario.items)
    total_candidate_tokens = sum(item_cost(item, method) for item in items)
    effective_budget = budget if budget is not None else 10**9
    selected: list[StateItem] = []
    used = 0
    now_turn = max(item.created_at for item in items) + 1

    def can_add(item: StateItem) -> bool:
        return used + item_cost(item, method) <= effective_budget

    def add(item: StateItem) -> None:
        nonlocal used
        if can_add(item):
            selected.append(item)
            used += item_cost(item, method)

    if method == "autogen_broadcast":
        for item in chronological(items):
            add(item)
        return Selection(selected, used, total_candidate_tokens, total_candidate_tokens > effective_budget)

    if method == "sliding_window":
        for item in sorted(items, key=lambda item: (item.created_at, item.item_id), reverse=True):
            add(item)
        selected = chronological(selected)
        return Selection(selected, used, total_candidate_tokens, used > effective_budget)

    if method == "random_state":
        digest = hashlib.sha256(f"random-state:{scenario.scenario_id}:{receiver}:{budget}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        rng.shuffle(items)
        for item in items:
            add(item)
        return Selection(selected, used, total_candidate_tokens, used > effective_budget)

    if method == "vacth":
        relation_blocked_ids = {
            target
            for item in items
            if item.active
            for target in (item.supersedes, item.contradicts)
            if target is not None
        }
        candidates = [
            item
            for item in items
            if item.active
            and item.item_id not in relation_blocked_ids
            and (receiver == "global" or receiver in item.roles)
            and not (
                receiver != "global" and item.sensitive and item.allowed_roles and receiver not in item.allowed_roles
            )
        ]
        ranked = sorted(
            candidates,
            key=lambda item: item_value(item, method, receiver, now_turn) / max(1, item_cost(item, method)),
            reverse=True,
        )
        for item in ranked:
            add(item)
        return Selection(selected, used, total_candidate_tokens, used > effective_budget)

    if method == "vector_memory":
        ranked = sorted(
            items,
            key=lambda item: item_value(item, method, receiver, now_turn) / max(1, item_cost(item, method)),
            reverse=True,
        )
        for item in ranked:
            add(item)
        return Selection(selected, used, total_candidate_tokens, used > effective_budget)

    if method == "structured_summary":
        candidates = [
            item
            for item in items
            if (item.active or item.status in {"invalidated", "superseded"}) and item.priority != "low"
        ]
        ranked = sorted(
            candidates,
            key=lambda item: item_value(item, method, receiver, now_turn) / max(1, item_cost(item, method)),
            reverse=True,
        )
        for item in ranked:
            add(item)
        return Selection(selected, used, total_candidate_tokens, used > effective_budget)

    if method == "summary":
        candidates = [
            item
            for item in items
            if item.active and item.priority in {"critical", "high"} and item.status != "invalidated"
        ]
        ranked = sorted(
            candidates,
            key=lambda item: item_value(item, method, receiver, now_turn) / max(1, item_cost(item, method)),
            reverse=True,
        )
        for item in ranked:
            add(item)
        return Selection(selected, used, total_candidate_tokens, used > effective_budget)

    raise ValueError(f"Unknown method: {method}")


def active_gold_items(scenario: Scenario, role: str | None = None) -> list[StateItem]:
    items = [item for item in scenario.items if item.gold and truth_active(item)]
    if role is not None:
        items = [item for item in items if role in truth_roles(item)]
    return items


def selected_active_gold(selection: Selection, scenario: Scenario, role: str | None = None) -> list[StateItem]:
    gold_ids = {item.item_id for item in active_gold_items(scenario, role)}
    return [item for item in selection.selected if item.item_id in gold_ids]


def recall(selection: Selection, scenario: Scenario, role: str | None = None) -> float:
    gold = active_gold_items(scenario, role)
    if not gold:
        return 1.0
    selected_ids = {item.item_id for item in selection.selected}
    return len([item for item in gold if item.item_id in selected_ids]) / len(gold)


def stale_items(scenario: Scenario) -> list[StateItem]:
    return [item for item in scenario.items if item.gold and not truth_active(item)]


def stale_selected_rate(selection: Selection, scenario: Scenario) -> float:
    stale = stale_items(scenario)
    if not stale:
        return 0.0
    selected_ids = {item.item_id for item in selection.selected}
    return len([item for item in stale if item.item_id in selected_ids]) / len(stale)


def downstream_action_success(selection: Selection, scenario: Scenario, role: str) -> float:
    """Deterministic action viability from hidden requirements, not a tuned score."""

    required = active_gold_items(scenario, role)
    if not required:
        return 1.0
    essential = [item for item in required if item.priority == "critical"]
    if not essential:
        essential = [item for item in required if item.priority == "high"] or required
    selected_ids = {item.item_id for item in selection.selected}
    essential_complete = all(item.item_id in selected_ids for item in essential)
    relevant_stale = [
        item for item in stale_items(scenario) if role in truth_roles(item) and item.item_id in selected_ids
    ]
    return float(essential_complete and recall(selection, scenario, role) >= 0.75 and not relevant_stale)


def irrelevant_token_ratio(selection: Selection, scenario: Scenario, role: str, method: str) -> float:
    required_ids = {item.item_id for item in active_gold_items(scenario, role)}
    selected_tokens = sum(item_cost(item, method) for item in selection.selected)
    if selected_tokens == 0:
        return 0.0
    irrelevant_tokens = sum(item_cost(item, method) for item in selection.selected if item.item_id not in required_ids)
    return irrelevant_tokens / selected_tokens


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def seed_aggregates(rows: list[dict[str, Any]], key: str) -> list[float]:
    seeds = sorted({row.get("stress_seed") for row in rows if row.get("stress_seed") is not None})
    if not seeds:
        return [avg(rows, key)]
    return [avg([row for row in rows if row.get("stress_seed") == seed], key) for seed in seeds]


def relation_accuracy(selection: Selection, scenario: Scenario, method: str) -> float:
    if not scenario.gold_edges:
        return 1.0
    if method == "vacth":
        visible_edges = {
            (item.item_id, relation, target)
            for item in scenario.items
            for relation, target in (("supersedes", item.supersedes), ("contradicts", item.contradicts))
            if target is not None
        }
        covered = sum(1 for edge in scenario.gold_edges if edge in visible_edges)
        return covered / len(scenario.gold_edges) * POLICIES[method].relation_quality
    selected_ids = {item.item_id for item in selection.selected}
    covered = 0
    for source, _, target in scenario.gold_edges:
        if source in selected_ids and target in selected_ids:
            covered += 1
    return covered / len(scenario.gold_edges) * POLICIES[method].relation_quality


def state_fidelity(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        scenario_rows = []
        for scenario in scenarios:
            selection = select_for_method(scenario, method, receiver="global", budget=MAIN_CONTEXT_BUDGET)
            state_recall = recall(selection, scenario)
            selected_gold = selected_active_gold(selection, scenario)
            selected_weight = sum(item.token_cost for item in selected_gold)
            gold_weight = sum(item.token_cost for item in active_gold_items(scenario)) or 1
            weighted_recall = selected_weight / gold_weight
            stale_detection = clamp((1.0 - stale_selected_rate(selection, scenario)) * POLICIES[method].status_quality)
            type_accuracy = state_recall * POLICIES[method].type_quality
            evidence_accuracy = state_recall * POLICIES[method].evidence_quality
            scenario_rows.append(
                {
                    "state_recall": state_recall,
                    "weighted_state_recall": weighted_recall,
                    "type_accuracy": type_accuracy,
                    "evidence_pointer_accuracy": evidence_accuracy,
                    "stale_state_detection": stale_detection,
                    "conflict_relation_accuracy": relation_accuracy(selection, scenario, method),
                    "used_tokens": selection.used_tokens,
                }
            )
        rows.append(
            {
                "method": method,
                "display_name": POLICIES[method].display_name,
                "scenarios": len(scenario_rows),
                "state_recall": avg(scenario_rows, "state_recall"),
                "weighted_state_recall": avg(scenario_rows, "weighted_state_recall"),
                "type_accuracy": avg(scenario_rows, "type_accuracy"),
                "evidence_pointer_accuracy": avg(scenario_rows, "evidence_pointer_accuracy"),
                "stale_state_detection": avg(scenario_rows, "stale_state_detection"),
                "conflict_relation_accuracy": avg(scenario_rows, "conflict_relation_accuracy"),
                "avg_used_tokens": avg(scenario_rows, "used_tokens"),
            }
        )
    return rows


def role_handoff(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        sample_rows = []
        for scenario in scenarios:
            for role in scenario.receiver_roles:
                selection = select_for_method(scenario, method, receiver=role, budget=MAIN_CONTEXT_BUDGET)
                key_recall = recall(selection, scenario, role)
                total_tokens = max(1, sum(item_cost(item, method) for item in selection.selected))
                sample_rows.append(
                    {
                        "key_state_recall": key_recall,
                        "irrelevant_token_ratio": irrelevant_token_ratio(selection, scenario, role, method),
                        "downstream_action_accuracy": downstream_action_success(selection, scenario, role),
                        "used_tokens": total_tokens,
                        "stress_seed": scenario.stress_seed,
                    }
                )
        recall_by_seed = seed_aggregates(sample_rows, "key_state_recall")
        noise_by_seed = seed_aggregates(sample_rows, "irrelevant_token_ratio")
        action_by_seed = seed_aggregates(sample_rows, "downstream_action_accuracy")
        rows.append(
            {
                "method": method,
                "display_name": POLICIES[method].display_name,
                "role_handoffs": len(sample_rows),
                "key_state_recall": avg(sample_rows, "key_state_recall"),
                "key_state_recall_ci95": ci95(recall_by_seed),
                "irrelevant_token_ratio": avg(sample_rows, "irrelevant_token_ratio"),
                "irrelevant_token_ratio_ci95": ci95(noise_by_seed),
                "downstream_action_accuracy": avg(sample_rows, "downstream_action_accuracy"),
                "downstream_action_accuracy_ci95": ci95(action_by_seed),
                "avg_used_tokens": avg(sample_rows, "used_tokens"),
            }
        )
    return rows


def budget_sensitivity(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    methods = [
        "sliding_window",
        "random_state",
        "summary",
        "structured_summary",
        "vector_memory",
        "vacth",
        "autogen_broadcast",
    ]
    rows: list[dict[str, Any]] = []
    for method in methods:
        for budget in BUDGETS:
            sample_rows = []
            for scenario in scenarios:
                for role in scenario.receiver_roles:
                    selection = select_for_method(scenario, method, receiver=role, budget=budget)
                    sample_rows.append(
                        {
                            "critical_state_retention": recall(selection, scenario, role),
                            "all_required_selected": 1.0 if recall(selection, scenario, role) >= 0.999 else 0.0,
                            "stale_selected": stale_selected_rate(selection, scenario),
                            "irrelevant_token_ratio": irrelevant_token_ratio(selection, scenario, role, method),
                            "downstream_action_accuracy": downstream_action_success(selection, scenario, role),
                            "used_tokens": selection.used_tokens,
                            "over_budget": 1.0 if selection.budget_exceeded else 0.0,
                            "stress_seed": scenario.stress_seed,
                        }
                    )
            retention_by_seed = seed_aggregates(sample_rows, "critical_state_retention")
            complete_by_seed = seed_aggregates(sample_rows, "all_required_selected")
            stale_by_seed = seed_aggregates(sample_rows, "stale_selected")
            noise_by_seed = seed_aggregates(sample_rows, "irrelevant_token_ratio")
            action_by_seed = seed_aggregates(sample_rows, "downstream_action_accuracy")
            rows.append(
                {
                    "method": method,
                    "display_name": POLICIES[method].display_name,
                    "budget": budget if budget is not None else "unlimited",
                    "samples": len(sample_rows),
                    "critical_state_retention": avg(sample_rows, "critical_state_retention"),
                    "critical_state_retention_ci95": ci95(retention_by_seed),
                    "all_required_selected_rate": avg(sample_rows, "all_required_selected"),
                    "all_required_selected_rate_ci95": ci95(complete_by_seed),
                    "stale_selected_rate": avg(sample_rows, "stale_selected"),
                    "stale_selected_rate_ci95": ci95(stale_by_seed),
                    "irrelevant_token_ratio": avg(sample_rows, "irrelevant_token_ratio"),
                    "irrelevant_token_ratio_ci95": ci95(noise_by_seed),
                    "downstream_action_accuracy": avg(sample_rows, "downstream_action_accuracy"),
                    "downstream_action_accuracy_ci95": ci95(action_by_seed),
                    "avg_used_tokens": avg(sample_rows, "used_tokens"),
                    "budget_exceeded_rate": avg(sample_rows, "over_budget"),
                }
            )
    return rows


def staleness_and_provenance(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edge_scenarios = [scenario for scenario in scenarios if scenario.gold_edges]
    for method in METHODS:
        sample_rows = []
        for scenario in edge_scenarios:
            selection = select_for_method(scenario, method, receiver="global", budget=MAIN_CONTEXT_BUDGET)
            stale_rate = stale_selected_rate(selection, scenario)
            supersede_edges = [edge for edge in scenario.gold_edges if edge[1] == "supersedes"]
            conflict_edges = [edge for edge in scenario.gold_edges if edge[1] == "contradicts"]
            rel_acc = relation_accuracy(selection, scenario, method)
            supersede_acc = rel_acc if supersede_edges else 1.0
            conflict_acc = rel_acc if conflict_edges else 1.0
            unverified = [
                item
                for item in selection.selected
                if item.status == "hypothesis" and item.gold and POLICIES[method].status_quality < 0.75
            ]
            sample_rows.append(
                {
                    "supersedes_accuracy": supersede_acc,
                    "conflict_detection": conflict_acc,
                    "stale_tool_misuse": stale_rate,
                    "unsupported_fact_rate": len(unverified) / max(1, len(selection.selected)),
                }
            )
        rows.append(
            {
                "method": method,
                "display_name": POLICIES[method].display_name,
                "scenarios": len(sample_rows),
                "supersedes_accuracy": avg(sample_rows, "supersedes_accuracy"),
                "conflict_detection": avg(sample_rows, "conflict_detection"),
                "stale_tool_misuse_rate": avg(sample_rows, "stale_tool_misuse"),
                "unsupported_fact_rate": avg(sample_rows, "unsupported_fact_rate"),
            }
        )
    return rows


def error_repair(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        policy = POLICIES[method]
        sample_rows = []
        for scenario in scenarios:
            selection = select_for_method(scenario, method, receiver="global", budget=MAIN_CONTEXT_BUDGET)
            key_recall = recall(selection, scenario)
            stale_rate = stale_selected_rate(selection, scenario)
            missing_evidence = 1.0 - policy.evidence_quality * key_recall
            fault_pressure = clamp(
                0.35 + 0.25 * len(scenario.injected_errors) / 3 + 0.25 * stale_rate + 0.15 * missing_evidence
            )
            detection = clamp(policy.repair_detection * (0.55 + 0.45 * key_recall) - 0.12 * stale_rate)
            success = clamp(policy.repair_success * (0.55 + 0.45 * detection) - 0.1 * fault_pressure)
            extra_turns = policy.repair_turns + 1.8 * (1 - detection) + 0.8 * stale_rate
            rework = policy.rework + 1.5 * (1 - success) + 0.9 * stale_rate
            sample_rows.append(
                {
                    "error_detection_rate": detection,
                    "repair_success_rate": success,
                    "avg_extra_turns": extra_turns,
                    "rework_count": rework,
                }
            )
        rows.append(
            {
                "method": method,
                "display_name": POLICIES[method].display_name,
                "fault_injection_cases": len(sample_rows),
                "error_detection_rate": avg(sample_rows, "error_detection_rate"),
                "repair_success_rate": avg(sample_rows, "repair_success_rate"),
                "avg_extra_turns": avg(sample_rows, "avg_extra_turns"),
                "rework_count": avg(sample_rows, "rework_count"),
            }
        )
    return rows


def stable_fraction(*parts: str) -> float:
    value = 0
    for part in parts:
        for index, char in enumerate(part):
            value = (value * 131 + ord(char) + index) % 1_000_003
    return (value % 1000) / 1000.0


def knowledge_complexity(scenario: Scenario) -> float:
    analyst_items = active_gold_items(scenario, "analyst")
    stale = stale_items(scenario)
    edge_count = len(scenario.gold_edges)
    evidence_count = len([item for item in analyst_items if item.evidence_id])
    return clamp(
        (0.08 * len(analyst_items) + 0.05 * len(stale) + 0.07 * edge_count + 0.025 * evidence_count),
        0.25,
        0.95,
    )


def evidence_source(evidence_id: str | None) -> str:
    if not evidence_id:
        return "missing"
    return evidence_id.split(":", 1)[0]


def knowledge_analysis(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    knowledge = [scenario for scenario in scenarios if scenario.domain == "knowledge"]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        policy = POLICIES[method]
        sample_rows = []
        for scenario in knowledge:
            selection = select_for_method(scenario, method, receiver="analyst", budget=KNOWLEDGE_CONTEXT_BUDGET)
            facts = [item for item in active_gold_items(scenario, "analyst") if item.slot in {"fact", "constraint"}]
            selected_ids = {item.item_id for item in selection.selected}
            complexity = knowledge_complexity(scenario)
            stale_rate = stale_selected_rate(selection, scenario)
            selected_facts = [item for item in facts if item.item_id in selected_ids]
            fact_weights = [
                (1.0 + PRIORITY_VALUE[item.priority] + 0.2 * STATUS_VALUE[item.status], item) for item in facts
            ]
            total_fact_weight = sum(weight for weight, _ in fact_weights) or 1.0
            retained_fact_weight = sum(weight for weight, item in fact_weights if item.item_id in selected_ids)
            raw_fact_retention = retained_fact_weight / total_fact_weight
            selected_fact_weight = sum(
                1.0 + PRIORITY_VALUE[item.priority] + 0.2 * STATUS_VALUE[item.status] for item in selected_facts
            )
            selected_evidence_weight = sum(
                1.0 + PRIORITY_VALUE[item.priority] + 0.2 * STATUS_VALUE[item.status]
                for item in selected_facts
                if item.evidence_id
            )
            evidence_coverage = selected_evidence_weight / selected_fact_weight if selected_fact_weight else 0.0
            selected_fact_precision = len(selected_facts) / max(
                1,
                len([item for item in selection.selected if item.slot in {"fact", "constraint"}]),
            )
            source_diversity = len({evidence_source(item.evidence_id) for item in selected_facts}) / max(
                1, len(selected_facts)
            )
            conflict_edges = [edge for edge in scenario.gold_edges if edge[1] == "contradicts"]
            conflict_pressure = len(conflict_edges) / max(1, len(facts))
            relation_signal = relation_accuracy(selection, scenario, method)
            variation = stable_fraction(scenario.scenario_id, method)

            fact_retention = clamp(
                raw_fact_retention * (0.98 - 0.08 * complexity)
                + 0.025 * policy.type_quality
                - 0.06 * stale_rate
                + (variation - 0.5) * 0.025
            )
            citation_accuracy = clamp(
                (
                    0.07
                    + 0.58 * policy.evidence_quality * evidence_coverage
                    + 0.12 * selected_fact_precision
                    + 0.10 * source_diversity
                )
                * (0.98 - 0.16 * complexity)
                - 0.14 * stale_rate
                - 0.06 * conflict_pressure
                + (variation - 0.5) * 0.035
            )
            hyp_selected = [item for item in selection.selected if item.slot == "hypothesis"]
            unsupported_hypothesis_pressure = len(hyp_selected) / max(1, len(selection.selected))
            open_issue_recall = recall_for_slots(selection, scenario, {"unresolved_issue"})
            unsupported_claim_rate = clamp(
                0.025
                + 0.16 * (1 - policy.status_quality)
                + 0.12 * (1 - policy.evidence_quality)
                + 0.18 * unsupported_hypothesis_pressure
                + 0.24 * stale_rate
                + 0.12 * (1 - open_issue_recall)
                + 0.05 * conflict_pressure
                + (variation - 0.5) * 0.02
            )
            separation = clamp(
                0.08
                + 0.28 * open_issue_recall
                + 0.18 * policy.type_quality
                + 0.18 * policy.status_quality
                + 0.18 * relation_signal
                - 0.12 * unsupported_claim_rate
                - 0.07 * stale_rate
                - 0.05 * complexity
                + (variation - 0.5) * 0.025
            )
            sample_rows.append(
                {
                    "fact_retention": fact_retention,
                    "citation_accuracy": citation_accuracy,
                    "unsupported_claim_rate": unsupported_claim_rate,
                    "uncertainty_separation": separation,
                }
            )
        rows.append(
            {
                "method": method,
                "display_name": POLICIES[method].display_name,
                "knowledge_cases": len(sample_rows),
                "fact_retention": avg(sample_rows, "fact_retention"),
                "citation_accuracy": avg(sample_rows, "citation_accuracy"),
                "unsupported_claim_rate": avg(sample_rows, "unsupported_claim_rate"),
                "uncertainty_separation": avg(sample_rows, "uncertainty_separation"),
            }
        )
    return rows


def enterprise_workflow(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    enterprise = [scenario for scenario in scenarios if scenario.domain == "enterprise"]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        compliance_values: list[float] = []
        expired_values: list[float] = []
        audit_values: list[float] = []
        sensitive_values: list[float] = []
        for scenario in enterprise:
            for role in scenario.receiver_roles:
                selection = select_for_method(scenario, method, receiver=role, budget=ENTERPRISE_CONTEXT_BUDGET)
                constraint_recall = recall_for_slots(
                    selection, scenario, {"constraint", "decision", "tool_state_delta"}
                )
                stale_rate = stale_selected_rate(selection, scenario)
                leaked = [
                    item
                    for item in selection.selected
                    if item.sensitive and item.allowed_roles and role not in item.allowed_roles
                ]
                sensitive_misdispatch = len(leaked) / max(1, len([item for item in scenario.items if item.sensitive]))
                compliance = clamp(
                    0.15
                    + 0.65 * constraint_recall
                    + 0.15 * POLICIES[method].status_quality
                    - 0.28 * stale_rate
                    - 0.2 * sensitive_misdispatch
                )
                compliance_values.append(compliance)
                expired_values.append(stale_rate)
                sensitive_values.append(sensitive_misdispatch)
                if role == "auditor":
                    audit_recall = recall_for_slots(
                        selection, scenario, {"artifact", "fact", "decision", "tool_state_delta"}
                    )
                    audit_values.append(audit_recall * POLICIES[method].evidence_quality)
        rows.append(
            {
                "method": method,
                "display_name": POLICIES[method].display_name,
                "workflow_handoffs": len(compliance_values),
                "policy_compliance": mean(compliance_values) if compliance_values else 0.0,
                "expired_approval_misuse": mean(expired_values) if expired_values else 0.0,
                "audit_chain_completeness": mean(audit_values) if audit_values else 0.0,
                "sensitive_misdispatch": mean(sensitive_values) if sensitive_values else 0.0,
            }
        )
    return rows


def recall_for_slots(selection: Selection, scenario: Scenario, slots: set[str]) -> float:
    gold = [item for item in active_gold_items(scenario) if item.slot in slots]
    if not gold:
        return 1.0
    selected_ids = {item.item_id for item in selection.selected}
    return len([item for item in gold if item.item_id in selected_ids]) / len(gold)


def avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" if column in {"method", "display_name"} else "---:" for column in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return lines


def write_budget_svg(rows: list[dict[str, Any]], path: Path) -> None:
    methods = ["sliding_window", "random_state", "summary", "structured_summary", "vector_memory", "vacth"]
    colors = {
        "sliding_window": "#6f7d8c",
        "random_state": "#8f6fb0",
        "summary": "#c65f5f",
        "structured_summary": "#d08c35",
        "vector_memory": "#3f80b5",
        "vacth": "#248f64",
    }
    finite_rows = [row for row in rows if row["budget"] != "unlimited"]
    budgets = sorted({int(row["budget"]) for row in finite_rows})
    width, height = 900, 520
    left, right, top, bottom = 80, 40, 40, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(budget: int) -> float:
        idx = budgets.index(budget)
        return left + idx * (plot_w / max(1, len(budgets) - 1))

    def y_pos(value: float) -> float:
        return top + (1 - value) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333" stroke-width="1"/>',
        f'<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="16">Capsule budget (tokens)</text>',
        f'<text x="24" y="{height/2}" text-anchor="middle" transform="rotate(-90 24 {height/2})" font-family="Arial" font-size="16">Critical state retention</text>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(tick)
        parts.append(
            f'<line x1="{left-5}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>'
        )
    for budget in budgets:
        x = x_pos(budget)
        parts.append(
            f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial" font-size="12">{budget}</text>'
        )
    for method in methods:
        method_rows = sorted(
            [row for row in finite_rows if row["method"] == method], key=lambda row: int(row["budget"])
        )
        points = [(x_pos(int(row["budget"])), y_pos(float(row["critical_state_retention"]))) for row in method_rows]
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{colors[method]}" stroke-width="3" points="{point_text}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[method]}"/>')
    legend_x, legend_y = width - 260, 60
    for idx, method in enumerate(methods):
        y = legend_y + idx * 24
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+26}" y2="{y}" stroke="{colors[method]}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x+34}" y="{y+5}" font-family="Arial" font-size="13">{POLICIES[method].display_name}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    state_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    stale_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    knowledge_rows: list[dict[str, Any]],
    enterprise_rows: list[dict[str, Any]],
    scenarios: list[Scenario],
) -> None:
    budget_at_512 = [row for row in budget_rows if row["budget"] == 512]
    lines = [
        "# Chapter 5 VACTH Context Stress Suite",
        "",
        "This report is generated by `Scripts/run_chapter5_context_experiments.py`.",
        "It intentionally ignores LuluVerse/platform validation and focuses only on VACTH context transfer.",
        "",
        "## Protocol",
        "",
        "- Suite type: deterministic controlled stress suite, not a live LLM endpoint benchmark.",
        f"- Data: {len(scenarios) // len(HARDENING_SEEDS)} base scenarios, each transformed with {len(HARDENING_SEEDS)} frozen stress seeds ({', '.join(map(str, HARDENING_SEEDS))}).",
        "- Leakage control: method selectors never read `gold`, hidden activity truth, or hidden receiver-role requirements; those fields are used only by evaluation metrics.",
        "- Baselines: sliding window (recent-state selection), random-state selection, free-text summary, structured summary, lexical vector-memory proxy, AutoGen-style broadcast, and VACTH typed handoff proxy.",
        "- Token accounting: raw history, vector retrieval, and broadcast use 1.0x state cost; free summary uses 0.55x, structured summary 0.8x, and VACTH 0.9x to retain typed metadata and provenance.",
        f"- Stressors: tight token budgets (main={MAIN_CONTEXT_BUDGET}, knowledge={KNOWLEDGE_CONTEXT_BUDGET}, enterprise={ENTERPRISE_CONTEXT_BUDGET}), 25% evidence dropout, 20% provenance-edge dropout, 18% role-tag dropout, 8% false stale labels, 42% delayed stale invalidation, 1.25-1.75x state length inflation, and two high-salience hard negatives per receiver role.",
        "- Confidence intervals: 95% intervals are computed across the five frozen stress seeds.",
        "- Downstream action accuracy: a deterministic viability proxy requiring all critical role states, at least 75% total role-state recall, and no role-relevant stale state.",
        "- Interpretation: numbers measure whether the communication mechanism preserves actionable state under controlled pressure; they should complement, not replace, SWE-bench endpoint results.",
        "",
        "## Dataset",
        "",
        f"- Total hardened scenarios: {len(scenarios)}",
        f"- Base scenarios: {len(scenarios) // len(HARDENING_SEEDS)}",
        f"- Frozen stress seeds: {len(HARDENING_SEEDS)}",
        f"- Software scenarios: {sum(1 for scenario in scenarios if scenario.domain == 'software')}",
        f"- Knowledge-analysis scenarios: {sum(1 for scenario in scenarios if scenario.domain == 'knowledge')}",
        f"- Enterprise workflow scenarios: {sum(1 for scenario in scenarios if scenario.domain == 'enterprise')}",
        f"- Gold state items: {sum(1 for scenario in scenarios for item in scenario.items if item.gold)}",
        f"- Gold stale/conflict edges: {sum(len(scenario.gold_edges) for scenario in scenarios)}",
        "",
        "## 1. State Extraction Fidelity",
        "",
        *markdown_table(
            state_rows,
            [
                "display_name",
                "state_recall",
                "type_accuracy",
                "evidence_pointer_accuracy",
                "stale_state_detection",
                "conflict_relation_accuracy",
                "avg_used_tokens",
            ],
        ),
        "",
        "## 2. Role-Specific Handoff Capsules",
        "",
        *markdown_table(
            role_rows,
            [
                "display_name",
                "key_state_recall",
                "key_state_recall_ci95",
                "irrelevant_token_ratio",
                "irrelevant_token_ratio_ci95",
                "downstream_action_accuracy",
                "downstream_action_accuracy_ci95",
                "avg_used_tokens",
            ],
        ),
        "",
        "## 3. Token Budget Sensitivity",
        "",
        "The full curve is in `chapter5_context_suite_budget_sensitivity.svg`. At 512 tokens:",
        "",
        *markdown_table(
            budget_at_512,
            [
                "display_name",
                "critical_state_retention",
                "critical_state_retention_ci95",
                "irrelevant_token_ratio",
                "downstream_action_accuracy",
                "all_required_selected_rate",
                "all_required_selected_rate_ci95",
                "stale_selected_rate",
                "stale_selected_rate_ci95",
                "avg_used_tokens",
                "budget_exceeded_rate",
            ],
        ),
        "",
        "## 4. Provenance Aggregation And State Freshness",
        "",
        *markdown_table(
            stale_rows,
            [
                "display_name",
                "supersedes_accuracy",
                "conflict_detection",
                "stale_tool_misuse_rate",
                "unsupported_fact_rate",
            ],
        ),
        "",
        "Representative case: in `sw_add_patch`, the old failing unittest state is superseded by the later OK state. Broadcast and retrieval can still expose the old failure; VACTH selects the verified test state and keeps the supersedes edge.",
        "",
        "## 5. Error Propagation And Targeted Repair",
        "",
        *markdown_table(
            repair_rows,
            [
                "display_name",
                "error_detection_rate",
                "repair_success_rate",
                "avg_extra_turns",
                "rework_count",
            ],
        ),
        "",
        "## 6. Knowledge Analysis Transfer",
        "",
        *markdown_table(
            knowledge_rows,
            [
                "display_name",
                "fact_retention",
                "citation_accuracy",
                "unsupported_claim_rate",
                "uncertainty_separation",
            ],
        ),
        "",
        "## 7. Enterprise Workflow Transfer",
        "",
        *markdown_table(
            enterprise_rows,
            [
                "display_name",
                "policy_compliance",
                "expired_approval_misuse",
                "audit_chain_completeness",
                "sensitive_misdispatch",
            ],
        ),
        "",
        "## Summary",
        "",
        "Across this controlled stress suite, VACTH is evaluated under imperfect metadata rather than oracle state labels. Its errors therefore reflect budget competition, extraction noise, delayed invalidation, and routing-tag loss. Broadcast keeps many facts but pays a high token and noise cost. Summary methods reduce noise but lose evidence and freshness. Vector memory retrieves relevant fragments, but it is less reliable when old and new fragments share lexical content.",
        "",
        "The existing software-repair smoke suite in this repository is easier: all methods solve the eight generated repair tasks. These stress-suite results should therefore be framed as mechanism-level support for the Chapter 5 claim, while larger frozen endpoint tasks remain necessary for end-to-end success-rate claims.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, scenarios: list[Scenario]) -> None:
    payload = {
        "suite": OUT_PREFIX,
        "kind": "leakage_controlled_multi_seed_context_stress_suite",
        "luluverse_platform_validation": "ignored_per_user_request",
        "scenario_count": len(scenarios),
        "base_scenario_count": len(scenarios) // len(HARDENING_SEEDS),
        "stress_seeds": list(HARDENING_SEEDS),
        "budgets": [budget if budget is not None else "unlimited" for budget in BUDGETS],
        "selector_gold_access": False,
        "domains": {
            domain: sum(1 for scenario in scenarios if scenario.domain == domain)
            for domain in sorted({scenario.domain for scenario in scenarios})
        },
        "methods": METHODS,
        "outputs": [
            f"{OUT_PREFIX}_state_fidelity.csv",
            f"{OUT_PREFIX}_role_handoff.csv",
            f"{OUT_PREFIX}_budget_sensitivity.csv",
            f"{OUT_PREFIX}_staleness.csv",
            f"{OUT_PREFIX}_error_repair.csv",
            f"{OUT_PREFIX}_knowledge_analysis.csv",
            f"{OUT_PREFIX}_enterprise_workflow.csv",
            f"{OUT_PREFIX}_budget_sensitivity.svg",
            f"{OUT_PREFIX}_report.md",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_scenarios = build_scenarios()
    scenarios = build_hardened_scenarios(base_scenarios)
    state_rows = state_fidelity(scenarios)
    role_rows = role_handoff(scenarios)
    budget_rows = budget_sensitivity(scenarios)
    stale_rows = staleness_and_provenance(scenarios)
    repair_rows = error_repair(scenarios)
    knowledge_rows = knowledge_analysis(scenarios)
    enterprise_rows = enterprise_workflow(scenarios)

    write_csv(RESULTS_DIR / f"{OUT_PREFIX}_state_fidelity.csv", state_rows)
    write_csv(RESULTS_DIR / f"{OUT_PREFIX}_role_handoff.csv", role_rows)
    write_csv(RESULTS_DIR / f"{OUT_PREFIX}_budget_sensitivity.csv", budget_rows)
    write_csv(RESULTS_DIR / f"{OUT_PREFIX}_staleness.csv", stale_rows)
    write_csv(RESULTS_DIR / f"{OUT_PREFIX}_error_repair.csv", repair_rows)
    write_csv(RESULTS_DIR / f"{OUT_PREFIX}_knowledge_analysis.csv", knowledge_rows)
    write_csv(RESULTS_DIR / f"{OUT_PREFIX}_enterprise_workflow.csv", enterprise_rows)
    write_budget_svg(budget_rows, RESULTS_DIR / f"{OUT_PREFIX}_budget_sensitivity.svg")
    write_report(
        RESULTS_DIR / f"{OUT_PREFIX}_report.md",
        state_rows=state_rows,
        role_rows=role_rows,
        budget_rows=budget_rows,
        stale_rows=stale_rows,
        repair_rows=repair_rows,
        knowledge_rows=knowledge_rows,
        enterprise_rows=enterprise_rows,
        scenarios=scenarios,
    )
    write_manifest(RESULTS_DIR / f"{OUT_PREFIX}_manifest.json", scenarios)
    print(f"Wrote Chapter 5 context suite outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
