import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DATA_DIR = BENCHMARK_DIR / "data"
TARGET_SAMPLES_PER_TYPE = 50

MECHANISM_TYPES = ("extraction", "routing", "aggregation", "reask")

FAMILIES: list[dict[str, str]] = [
    {
        "id": "calc_add",
        "component": "calculator.add",
        "issue": "calculator.add subtracts for add(2, 3).",
        "issue_key": "unresolved_issue:calc_add_subtracts",
        "constraint": "Keep add(a, b) public and do not change tests.",
        "constraint_key": "constraint:calc_add_public_no_test_changes",
        "decision": "Patch calculator.py only.",
        "decision_key": "decision:calc_patch_calculator_only",
        "artifact": "calculator.py now returns a + b.",
        "artifact_key": "artifact:calc_returns_a_plus_b",
        "test": "python -m unittest returned OK for calculator tests.",
        "test_key": "tool_state_delta:calc_tests_ok",
        "fact": "test_calculator.py calls calculator.add.",
        "fact_key": "fact:calc_add_called_by_tests",
        "hypothesis": "The failure is caused by stale bytecode.",
        "hypothesis_key": "hypothesis:calc_stale_bytecode",
        "invalidating_fact": "A clean test run still fails before the patch.",
    },
    {
        "id": "refund_policy",
        "component": "refund_tool",
        "issue": "refund_tool rejects orders inside the 30 day window.",
        "issue_key": "unresolved_issue:refund_tool_rejects_valid_window",
        "constraint": "Refunds are allowed within 30 days after delivery.",
        "constraint_key": "constraint:refund_30_day_window",
        "decision": "Use delivery_date instead of order_date for eligibility.",
        "decision_key": "decision:refund_use_delivery_date",
        "artifact": "refund_tool.py now checks delivery_date.",
        "artifact_key": "artifact:refund_tool_delivery_date",
        "test": "refund eligibility tests returned OK.",
        "test_key": "tool_state_delta:refund_tests_ok",
        "fact": "order-001 was delivered 12 days ago.",
        "fact_key": "fact:order_001_delivered_12_days_ago",
        "hypothesis": "The policy window is 7 days.",
        "hypothesis_key": "hypothesis:refund_window_7_days",
        "invalidating_fact": "The policy document says 30 days after delivery.",
    },
    {
        "id": "slug_migration",
        "component": "users.slug migration",
        "issue": "users.slug migration would lock the users table.",
        "issue_key": "unresolved_issue:slug_migration_table_lock",
        "constraint": "Avoid downtime during the users.slug migration.",
        "constraint_key": "constraint:slug_migration_no_downtime",
        "decision": "Add nullable slug, backfill, then enforce NOT NULL.",
        "decision_key": "decision:slug_two_phase_migration",
        "artifact": "migration adds slug in two phases.",
        "artifact_key": "artifact:slug_two_phase_migration",
        "test": "migration dry run completed without table lock.",
        "test_key": "tool_state_delta:slug_migration_dry_run_ok",
        "fact": "users has 8 million rows.",
        "fact_key": "fact:users_8_million_rows",
        "hypothesis": "A single NOT NULL migration is safe.",
        "hypothesis_key": "hypothesis:slug_single_not_null_safe",
        "invalidating_fact": "The database plan shows an exclusive table lock.",
    },
    {
        "id": "csv_parser",
        "component": "csv parser",
        "issue": "csv parser splits quoted commas incorrectly.",
        "issue_key": "unresolved_issue:csv_quoted_commas",
        "constraint": "Preserve quoted fields when parsing CSV.",
        "constraint_key": "constraint:csv_preserve_quoted_fields",
        "decision": "Use csv.reader instead of split(',').",
        "decision_key": "decision:csv_use_reader",
        "artifact": "parser.py now uses csv.reader.",
        "artifact_key": "artifact:parser_uses_csv_reader",
        "test": "quoted comma parser tests returned OK.",
        "test_key": "tool_state_delta:csv_parser_tests_ok",
        "fact": "sample row contains a quoted comma.",
        "fact_key": "fact:csv_row_has_quoted_comma",
        "hypothesis": "The delimiter is a semicolon.",
        "hypothesis_key": "hypothesis:csv_semicolon_delimiter",
        "invalidating_fact": "The fixture uses comma delimiters with quotes.",
    },
    {
        "id": "token_logging",
        "component": "auth logging",
        "issue": "auth logging prints raw OAuth access tokens.",
        "issue_key": "unresolved_issue:auth_logs_raw_tokens",
        "constraint": "Never print raw OAuth access tokens.",
        "constraint_key": "constraint:no_raw_oauth_tokens",
        "decision": "Redact Authorization headers before logging.",
        "decision_key": "decision:redact_authorization_header",
        "artifact": "logger.py redacts Authorization headers.",
        "artifact_key": "artifact:logger_redacts_authorization",
        "test": "security log scan found no raw tokens.",
        "test_key": "tool_state_delta:token_log_scan_ok",
        "fact": "debug logs include Authorization headers.",
        "fact_key": "fact:debug_logs_authorization_header",
        "hypothesis": "The logs only contain request IDs.",
        "hypothesis_key": "hypothesis:logs_only_request_ids",
        "invalidating_fact": "The debug sample includes a bearer token.",
    },
    {
        "id": "search_cache",
        "component": "search cache",
        "issue": "search cache returns stale rankings after reindex.",
        "issue_key": "unresolved_issue:search_cache_stale_rankings",
        "constraint": "Invalidate cached rankings after every reindex.",
        "constraint_key": "constraint:invalidate_cache_after_reindex",
        "decision": "Include index_version in the cache key.",
        "decision_key": "decision:cache_key_index_version",
        "artifact": "search_cache.py keys entries by index_version.",
        "artifact_key": "artifact:search_cache_index_version",
        "test": "reindex cache invalidation tests returned OK.",
        "test_key": "tool_state_delta:search_cache_tests_ok",
        "fact": "index_version changes after reindex.",
        "fact_key": "fact:index_version_changes_after_reindex",
        "hypothesis": "The stale ranking is caused by tokenizer drift.",
        "hypothesis_key": "hypothesis:search_tokenizer_drift",
        "invalidating_fact": "The tokenizer hash is unchanged across the run.",
    },
    {
        "id": "upload_dedupe",
        "component": "upload dedupe",
        "issue": "upload dedupe drops rows with distinct external IDs.",
        "issue_key": "unresolved_issue:upload_dedupe_external_ids",
        "constraint": "Deduplicate uploads by external_id and source.",
        "constraint_key": "constraint:dedupe_by_external_id_and_source",
        "decision": "Use a composite external_id plus source key.",
        "decision_key": "decision:dedupe_composite_key",
        "artifact": "dedupe.py now uses the composite key.",
        "artifact_key": "artifact:dedupe_composite_key",
        "test": "upload dedupe regression tests returned OK.",
        "test_key": "tool_state_delta:upload_dedupe_tests_ok",
        "fact": "two sources can reuse the same external_id.",
        "fact_key": "fact:external_id_reused_by_sources",
        "hypothesis": "Rows are duplicates when names match.",
        "hypothesis_key": "hypothesis:dedupe_by_name",
        "invalidating_fact": "The product spec says name matches are not duplicates.",
    },
    {
        "id": "email_retry",
        "component": "email retry",
        "issue": "email retry sends duplicate welcome messages.",
        "issue_key": "unresolved_issue:email_retry_duplicates",
        "constraint": "Welcome email sends must be idempotent per user.",
        "constraint_key": "constraint:welcome_email_idempotent",
        "decision": "Store a send_attempt_id before retrying.",
        "decision_key": "decision:email_retry_attempt_id",
        "artifact": "email_retry.py records send_attempt_id.",
        "artifact_key": "artifact:email_retry_attempt_id",
        "test": "email retry idempotency tests returned OK.",
        "test_key": "tool_state_delta:email_retry_tests_ok",
        "fact": "network timeout occurs after the first send.",
        "fact_key": "fact:email_timeout_after_first_send",
        "hypothesis": "The SMTP provider suppresses duplicates.",
        "hypothesis_key": "hypothesis:smtp_suppresses_duplicates",
        "invalidating_fact": "The provider delivered two welcome messages.",
    },
    {
        "id": "feature_flag",
        "component": "feature flag rollout",
        "issue": "feature flag rollout enables beta UI for all tenants.",
        "issue_key": "unresolved_issue:feature_flag_all_tenants",
        "constraint": "Enable beta UI only for allowlisted tenants.",
        "constraint_key": "constraint:beta_ui_allowlist_only",
        "decision": "Check tenant_id against the allowlist before enabling.",
        "decision_key": "decision:feature_flag_allowlist_check",
        "artifact": "flags.py checks tenant_id allowlist.",
        "artifact_key": "artifact:flags_allowlist_check",
        "test": "feature flag rollout tests returned OK.",
        "test_key": "tool_state_delta:feature_flag_tests_ok",
        "fact": "tenant beta-42 is allowlisted.",
        "fact_key": "fact:tenant_beta_42_allowlisted",
        "hypothesis": "The rollout percentage is zero.",
        "hypothesis_key": "hypothesis:rollout_percentage_zero",
        "invalidating_fact": "Audit logs show non-allowlisted tenants saw beta UI.",
    },
    {
        "id": "invoice_tax",
        "component": "invoice tax",
        "issue": "invoice tax rounds each line before summing.",
        "issue_key": "unresolved_issue:invoice_tax_line_rounding",
        "constraint": "Round tax only after summing taxable lines.",
        "constraint_key": "constraint:round_tax_after_sum",
        "decision": "Sum raw line taxes before rounding cents.",
        "decision_key": "decision:invoice_sum_before_round",
        "artifact": "invoice.py rounds tax after summing lines.",
        "artifact_key": "artifact:invoice_round_after_sum",
        "test": "invoice rounding tests returned OK.",
        "test_key": "tool_state_delta:invoice_tax_tests_ok",
        "fact": "line-level rounding loses one cent on fixture INV-9.",
        "fact_key": "fact:invoice_line_rounding_loses_cent",
        "hypothesis": "The fixture total is wrong.",
        "hypothesis_key": "hypothesis:invoice_fixture_wrong",
        "invalidating_fact": "The accounting oracle confirms the fixture total.",
    },
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _message(
    message_id: str, turn_id: int, source: str, content: str
) -> dict[str, Any]:
    return {
        "message_id": message_id,
        "turn_id": turn_id,
        "source": source,
        "role": "assistant" if source != "user" else "user",
        "content": content,
    }


def _evidence(
    message_id: str, source: str, turn_id: int, quote: str
) -> list[dict[str, Any]]:
    return [
        {
            "message_id": message_id,
            "source_agent": source,
            "turn_id": turn_id,
            "quote": quote,
        }
    ]


def _item(
    *,
    item_id: str,
    slot: str,
    content: str,
    merge_key: str,
    status: str,
    priority: str,
    message_id: str,
    source: str,
    turn_id: int,
    quote: str,
    confidence: float = 0.9,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "slot": slot,
        "content": content,
        "merge_key": merge_key,
        "epistemic_status": status,
        "confidence": confidence,
        "priority": priority,
        "created_at_turn": turn_id,
        "source_agent": source,
        "arguments": arguments or {},
        "evidence": _evidence(message_id, source, turn_id, quote),
    }


def _gold_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold: list[dict[str, Any]] = []
    for item in items:
        evidence = []
        for pointer in item.get("evidence", []):
            if isinstance(pointer, dict) and pointer.get("message_id"):
                evidence.append({"message_id": pointer["message_id"]})
        gold.append(
            {
                "slot": item["slot"],
                "merge_key": item["merge_key"],
                "epistemic_status": item["epistemic_status"],
                "evidence": evidence,
            }
        )
    return gold


def _structured_state(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    slot_map = {
        "fact": "facts",
        "constraint": "constraints",
        "decision": "decisions",
        "hypothesis": "hypotheses",
        "tool_state_delta": "tool_states",
        "unresolved_issue": "open_issues",
    }
    state: dict[str, list[str]] = {
        "facts": [],
        "constraints": [],
        "decisions": [],
        "hypotheses": [],
        "tool_states": [],
        "open_issues": [],
    }
    for item in items:
        bucket = slot_map.get(item["slot"], "facts")
        state[bucket].append(item["content"])
    return {key: value for key, value in state.items() if value}


def _make_extraction_sample(family: dict[str, str], variant: int) -> dict[str, Any]:
    prefix = f"gen_ex_{family['id']}_{variant + 1}"
    turn = 11 + variant
    if variant == 0:
        source = "retriever"
        messages = [
            _message("m1", turn, source, family["issue"]),
            _message("m2", turn, "planner", family["constraint"]),
        ]
        items = [
            _item(
                item_id=f"{prefix}_issue",
                slot="unresolved_issue",
                content=family["issue"],
                merge_key=family["issue_key"],
                status="observed",
                priority="critical",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote=family["issue"].split(" ", 1)[0],
                confidence=0.92,
            ),
            _item(
                item_id=f"{prefix}_constraint",
                slot="constraint",
                content=family["constraint"],
                merge_key=family["constraint_key"],
                status="active_constraint",
                priority="critical",
                message_id="m2",
                source="planner",
                turn_id=turn,
                quote="do not" if "do not" in family["constraint"] else "constraint",
                confidence=0.93,
            ),
        ]
        summary = (
            f"{family['component']} has an unresolved issue under an active constraint."
        )
        task = f"Extract repair issue and active constraint for {family['component']}."
    elif variant == 1:
        source = "coder"
        messages = [_message("m1", turn, source, family["decision"])]
        items = [
            _item(
                item_id=f"{prefix}_decision",
                slot="decision",
                content=family["decision"],
                merge_key=family["decision_key"],
                status="observed",
                priority="high",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote=family["decision"].split(" ", 1)[0],
                confidence=0.88,
            ),
            _item(
                item_id=f"{prefix}_artifact",
                slot="artifact",
                content=family["artifact"],
                merge_key=family["artifact_key"],
                status="observed",
                priority="high",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote=family["artifact"].split(" ", 1)[0],
                confidence=0.86,
            ),
        ]
        summary = f"{family['component']} implementation was updated."
        task = (
            f"Extract implementation decision and artifact for {family['component']}."
        )
    elif variant == 2:
        source = "retriever"
        messages = [
            _message("m1", turn, source, family["invalidating_fact"]),
            _message("m2", turn, source, family["hypothesis"]),
        ]
        items = [
            _item(
                item_id=f"{prefix}_fact",
                slot="fact",
                content=family["invalidating_fact"],
                merge_key=family["fact_key"],
                status="verified",
                priority="high",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote=family["invalidating_fact"].split(" ", 1)[0],
                confidence=0.91,
            ),
            _item(
                item_id=f"{prefix}_hypothesis",
                slot="hypothesis",
                content=family["hypothesis"],
                merge_key=family["hypothesis_key"],
                status="invalidated",
                priority="medium",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote="invalidates",
                confidence=0.84,
            ),
        ]
        summary = (
            f"Evidence invalidates an earlier hypothesis for {family['component']}."
        )
        task = f"Extract conflict evidence and invalidated hypothesis for {family['component']}."
    else:
        source = "tester"
        messages = [_message("m1", turn, source, family["test"])]
        items = [
            _item(
                item_id=f"{prefix}_test",
                slot="tool_state_delta",
                content=family["test"],
                merge_key=family["test_key"],
                status="verified",
                priority="critical",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote="OK" if "OK" in family["test"] else "returned",
                confidence=0.96,
            ),
            _item(
                item_id=f"{prefix}_artifact",
                slot="artifact",
                content=family["artifact"],
                merge_key=family["artifact_key"],
                status="verified",
                priority="high",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote=family["artifact"].split(" ", 1)[0],
                confidence=0.9,
            ),
        ]
        summary = f"{family['component']} verification passed after the update."
        task = f"Extract verification state and repaired artifact for {family['component']}."

    return {
        "id": prefix,
        "task": task,
        "sender": source,
        "turn_id": turn,
        "messages": messages,
        "summary": summary,
        "structured_state": _structured_state(items),
        "vacth_delta": {"items": items, "edges": []},
        "gold": {"items": _gold_items(items)},
    }


def _make_routing_sample(family: dict[str, str], variant: int) -> dict[str, Any]:
    prefix = f"gen_route_{family['id']}_{variant + 1}"
    turn = 20 + variant
    receiver = ("coder", "reviewer", "tester", "planner")[variant]
    source = ("retriever", "tester", "coder", "planner")[variant]
    short_issue = f"{family['component']} fails."
    short_constraint = "Do not change public behavior."
    short_decision = f"Patch {family['component']} only."
    short_artifact = f"{family['component']} patch is ready."
    short_test = "Regression tests returned OK."
    distractor = (
        "Historical note about an unrelated documentation cleanup that does not "
        "affect the current receiver decision."
    )
    items = [
        _item(
            item_id=f"{prefix}_noise",
            slot="fact",
            content=distractor,
            merge_key=f"fact:{family['id']}_unrelated_docs",
            status="observed",
            priority="low",
            message_id="m0",
            source="user",
            turn_id=1,
            quote="unrelated",
            confidence=0.35,
        ),
        _item(
            item_id=f"{prefix}_constraint",
            slot="constraint",
            content=short_constraint,
            merge_key=family["constraint_key"],
            status="active_constraint",
            priority="critical",
            message_id="m1",
            source=source,
            turn_id=turn,
            quote="Do not",
            confidence=0.95,
        ),
        _item(
            item_id=f"{prefix}_issue",
            slot="unresolved_issue",
            content=short_issue,
            merge_key=family["issue_key"],
            status="observed",
            priority="critical",
            message_id="m1",
            source=source,
            turn_id=turn,
            quote="fails",
            confidence=0.93,
        ),
        _item(
            item_id=f"{prefix}_decision",
            slot="decision",
            content=short_decision,
            merge_key=family["decision_key"],
            status="observed",
            priority="high",
            message_id="m2",
            source=source,
            turn_id=turn,
            quote="Patch",
            confidence=0.88,
        ),
        _item(
            item_id=f"{prefix}_artifact",
            slot="artifact",
            content=short_artifact,
            merge_key=family["artifact_key"],
            status="verified" if receiver in {"tester", "reviewer"} else "observed",
            priority="high",
            message_id="m2",
            source=source,
            turn_id=turn,
            quote="patch",
            confidence=0.88,
        ),
        _item(
            item_id=f"{prefix}_test",
            slot="tool_state_delta",
            content=short_test,
            merge_key=family["test_key"],
            status="verified",
            priority="critical",
            message_id="m3",
            source="tester",
            turn_id=turn,
            quote="OK",
            confidence=0.96,
        ),
    ]
    if receiver == "coder":
        selected = [f"{prefix}_issue", f"{prefix}_constraint", f"{prefix}_decision"]
    elif receiver == "reviewer":
        selected = [f"{prefix}_test", f"{prefix}_constraint", f"{prefix}_artifact"]
    elif receiver == "tester":
        selected = [f"{prefix}_artifact", f"{prefix}_constraint", f"{prefix}_test"]
    else:
        selected = [f"{prefix}_constraint", f"{prefix}_issue", f"{prefix}_decision"]
    ordered_items = items if variant % 2 == 0 else items[1:] + items[:1]
    return {
        "id": prefix,
        "task": f"Route receiver-specific capsule for {family['component']}.",
        "receiver": receiver,
        "sender": source,
        "turn_id": turn,
        "token_budget": 24,
        "summary": f"Route only the most relevant {family['component']} state to {receiver}.",
        "messages": [
            _message("m1", turn, source, family["issue"]),
            _message("m2", turn, source, family["decision"]),
            _message("m3", turn, "tester", family["test"]),
        ],
        "items": ordered_items,
        "edges": [],
        "summary_selected_item_ids": [],
        "gold": {"selected_item_ids": selected, "preferred_order": selected},
    }


def _make_aggregation_sample(family: dict[str, str], variant: int) -> dict[str, Any]:
    prefix = f"gen_agg_{family['id']}_{variant + 1}"
    if variant == 0:
        old_id = f"{prefix}_constraint_old"
        new_id = f"{prefix}_constraint_dup"
        item_old = _item(
            item_id=old_id,
            slot="constraint",
            content=family["constraint"],
            merge_key=family["constraint_key"],
            status="active_constraint",
            priority="critical",
            message_id="m1",
            source="user",
            turn_id=1,
            quote="constraint",
            confidence=0.9,
        )
        item_new = {**item_old, "item_id": new_id, "confidence": 0.94}
        task = f"Deduplicate repeated constraint for {family['component']}."
        deltas = [
            {"turn_id": 1, "sender": "user", "items": [item_old]},
            {"turn_id": 2, "sender": "planner", "items": [item_new]},
        ]
        gold = {
            "active_merge_keys": [family["constraint_key"]],
            "superseded_item_ids": [],
            "edges": [{"source": new_id, "relation": "duplicates", "target": old_id}],
        }
    elif variant == 1:
        old_id = f"{prefix}_test_old"
        new_id = f"{prefix}_test_new"
        item_old = _item(
            item_id=old_id,
            slot="tool_state_delta",
            content=f"{family['component']} regression tests failed before the patch.",
            merge_key=family["test_key"],
            status="observed",
            priority="high",
            message_id="m1",
            source="tester",
            turn_id=1,
            quote="failed",
            confidence=0.86,
        )
        item_new = _item(
            item_id=new_id,
            slot="tool_state_delta",
            content=family["test"],
            merge_key=family["test_key"],
            status="verified",
            priority="critical",
            message_id="m2",
            source="tester",
            turn_id=3,
            quote="OK",
            confidence=0.96,
        )
        task = f"Supersede failing tool state after {family['component']} passes."
        deltas = [
            {"turn_id": 1, "sender": "tester", "items": [item_old]},
            {"turn_id": 3, "sender": "tester", "items": [item_new]},
        ]
        gold = {
            "active_merge_keys": [family["test_key"]],
            "superseded_item_ids": [old_id],
            "edges": [{"source": new_id, "relation": "supersedes", "target": old_id}],
        }
    elif variant == 2:
        old_id = f"{prefix}_hyp_old"
        new_id = f"{prefix}_hyp_invalidated"
        item_old = _item(
            item_id=old_id,
            slot="hypothesis",
            content=family["hypothesis"],
            merge_key=family["hypothesis_key"],
            status="hypothesis",
            priority="medium",
            message_id="m1",
            source="retriever",
            turn_id=1,
            quote="hypothesis",
            confidence=0.62,
        )
        item_new = _item(
            item_id=new_id,
            slot="hypothesis",
            content=family["invalidating_fact"],
            merge_key=family["hypothesis_key"],
            status="invalidated",
            priority="high",
            message_id="m2",
            source="retriever",
            turn_id=2,
            quote="invalidates",
            confidence=0.9,
        )
        task = f"Invalidate stale hypothesis for {family['component']}."
        deltas = [
            {"turn_id": 1, "sender": "retriever", "items": [item_old]},
            {"turn_id": 2, "sender": "retriever", "items": [item_new]},
        ]
        gold = {
            "active_merge_keys": [],
            "superseded_item_ids": [old_id, new_id],
            "edges": [{"source": new_id, "relation": "contradicts", "target": old_id}],
        }
    else:
        old_id = f"{prefix}_issue"
        new_id = f"{prefix}_artifact"
        item_old = _item(
            item_id=old_id,
            slot="unresolved_issue",
            content=family["issue"],
            merge_key=family["issue_key"],
            status="observed",
            priority="critical",
            message_id="m1",
            source="retriever",
            turn_id=1,
            quote="issue",
            confidence=0.91,
        )
        item_new = _item(
            item_id=new_id,
            slot="artifact",
            content=family["artifact"],
            merge_key=family["artifact_key"],
            status="verified",
            priority="high",
            message_id="m2",
            source="coder",
            turn_id=3,
            quote="now",
            confidence=0.91,
            arguments={"resolved_issue_merge_key": family["issue_key"]},
        )
        task = f"Mark {family['component']} issue resolved by verified artifact."
        deltas = [
            {"turn_id": 1, "sender": "retriever", "items": [item_old]},
            {"turn_id": 3, "sender": "coder", "items": [item_new]},
        ]
        gold = {
            "active_merge_keys": [family["artifact_key"]],
            "superseded_item_ids": [old_id],
            "edges": [{"source": new_id, "relation": "supersedes", "target": old_id}],
        }
    messages = [
        _message("m1", 1, "retriever", family["issue"]),
        _message("m2", 3, "coder", family["artifact"]),
    ]
    return {
        "id": prefix,
        "task": task,
        "summary": task,
        "messages": messages,
        "deltas": deltas,
        "gold": gold,
    }


BAD_DELTA_PATTERNS = (
    '{"itemz": [{"slot": "constraint", "content": "missing items key"}]}',
    '{"items": {"slot": "fact", "content": "items is not a list"}, "edges": []}',
    '{"items": [{"slot": "decision", "content": "truncated"}',
    '[{"slot": "unresolved_issue", "content": "array root"}]',
    "plain text instead of JSON",
)


def _make_reask_sample(family: dict[str, str], variant: int) -> dict[str, Any]:
    prefix = f"gen_reask_{family['id']}_{variant + 1}"
    turn = 30 + variant
    source = ("planner", "tester", "retriever", "coder")[variant]
    if variant == 0:
        messages = [_message("m1", turn, source, family["constraint"])]
        items = [
            _item(
                item_id=f"{prefix}_constraint",
                slot="constraint",
                content=family["constraint"],
                merge_key=family["constraint_key"],
                status="active_constraint",
                priority="critical",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote="constraint",
                confidence=0.94,
            )
        ]
    elif variant == 1:
        messages = [_message("m1", turn, source, family["test"])]
        items = [
            _item(
                item_id=f"{prefix}_test",
                slot="tool_state_delta",
                content=family["test"],
                merge_key=family["test_key"],
                status="verified",
                priority="critical",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote="OK",
                confidence=0.96,
            )
        ]
    elif variant == 2:
        messages = [_message("m1", turn, source, family["invalidating_fact"])]
        items = [
            _item(
                item_id=f"{prefix}_hypothesis",
                slot="hypothesis",
                content=family["hypothesis"],
                merge_key=family["hypothesis_key"],
                status="invalidated",
                priority="high",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote="invalidated",
                confidence=0.86,
            )
        ]
    else:
        messages = [_message("m1", turn, source, family["artifact"])]
        items = [
            _item(
                item_id=f"{prefix}_artifact",
                slot="artifact",
                content=family["artifact"],
                merge_key=family["artifact_key"],
                status="verified",
                priority="high",
                message_id="m1",
                source=source,
                turn_id=turn,
                quote="artifact",
                confidence=0.9,
            )
        ]
    return {
        "id": prefix,
        "task": f"Repair malformed THC JSON for {family['component']}.",
        "sender": source,
        "turn_id": turn,
        "reask_target": "vacth_extractor_json",
        "bad_delta_text": BAD_DELTA_PATTERNS[
            (variant + len(family["id"])) % len(BAD_DELTA_PATTERNS)
        ],
        "messages": messages,
        "summary": f"Repair typed handoff delta for {family['component']}.",
        "structured_state": _structured_state(items),
        "repair_delta": {"items": items, "edges": []},
        "gold": {
            "reask_needed": True,
            "reask_target": "vacth_extractor_json",
            "items": _gold_items(items),
        },
    }


def _generated_records(mechanism_type: str) -> list[dict[str, Any]]:
    makers = {
        "extraction": _make_extraction_sample,
        "routing": _make_routing_sample,
        "aggregation": _make_aggregation_sample,
        "reask": _make_reask_sample,
    }
    maker = makers[mechanism_type]
    records: list[dict[str, Any]] = []
    for family in FAMILIES:
        for variant in range(4):
            records.append(maker(family, variant))
    return records


def _validate(records: list[dict[str, Any]], mechanism_type: str) -> None:
    ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{mechanism_type}:{index} missing string id")
        if sample_id in ids:
            raise ValueError(f"{mechanism_type}:{index} duplicate id {sample_id}")
        ids.add(sample_id)
        if "gold" not in record or not isinstance(record["gold"], dict):
            raise ValueError(f"{mechanism_type}:{sample_id} missing gold object")
        if mechanism_type in {"extraction", "reask"}:
            gold_items = record["gold"].get("items", [])
            if not isinstance(gold_items, list) or not gold_items:
                raise ValueError(f"{mechanism_type}:{sample_id} missing gold items")
        if mechanism_type == "routing":
            selected = record["gold"].get("selected_item_ids", [])
            if not isinstance(selected, list) or not selected:
                raise ValueError(f"routing:{sample_id} missing selected_item_ids")
        if mechanism_type == "aggregation":
            active = record["gold"].get("active_merge_keys", [])
            edges = record["gold"].get("edges", [])
            if not isinstance(active, list) or not isinstance(edges, list):
                raise ValueError(f"aggregation:{sample_id} has invalid gold shape")


def scale_data(target: int) -> None:
    if target < len(FAMILIES) * 4:
        raise ValueError(f"target must be at least {len(FAMILIES) * 4}")
    for mechanism_type in MECHANISM_TYPES:
        path = DATA_DIR / f"{mechanism_type}.jsonl"
        existing = _read_jsonl(path)
        hand_authored = [
            record
            for record in existing
            if not str(record.get("id", "")).startswith("gen_")
        ]
        if len(hand_authored) > target:
            raise ValueError(
                f"{path} has {len(hand_authored)} non-generated records, "
                f"which exceeds target {target}"
            )
        generated = _generated_records(mechanism_type)
        needed = target - len(hand_authored)
        records = hand_authored + generated[:needed]
        _validate(records, mechanism_type)
        _write_jsonl(path, records)
        print(
            f"{path.relative_to(BENCHMARK_DIR)}: "
            f"{len(hand_authored)} hand-authored + {needed} generated = {len(records)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand E8 mechanism JSONL data with deterministic generated samples."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_SAMPLES_PER_TYPE,
        help="target records per mechanism JSONL",
    )
    args = parser.parse_args()
    scale_data(args.target)


if __name__ == "__main__":
    main()
