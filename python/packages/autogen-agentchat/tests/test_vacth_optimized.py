from __future__ import annotations

import json

from autogen_agentchat.vacth import (
    VACTHRuntime,
    build_extraction_prompt,
    heuristic_items_from_messages,
    render_capsule,
)
from autogen_agentchat.vacth.schema import (
    EvidencePointer,
    VACTHCapsule,
    VACTHEdge,
    VACTHStateItem,
    estimate_tokens,
)


def _item(
    item_id: str,
    slot: str,
    content: str,
    *,
    status: str = "observed",
    priority: str = "medium",
    source_agent: str = "retriever",
    turn: int = 1,
) -> VACTHStateItem:
    return VACTHStateItem(
        item_id=item_id,
        slot=slot,  # type: ignore[arg-type]
        content=content,
        merge_key=f"{slot}:{item_id}",
        epistemic_status=status,  # type: ignore[arg-type]
        priority=priority,  # type: ignore[arg-type]
        source_agent=source_agent,
        evidence=[
            EvidencePointer(
                message_id=f"m-{item_id}",
                source_agent=source_agent,
                turn_id=turn,
                quote=f"evidence quote for {item_id} should stay out of compact render",
            )
        ],
        created_at_turn=turn,
    )


def test_compact_capsule_keeps_key_items_without_evidence_quotes() -> None:
    capsule = VACTHCapsule(
        capsule_id="capsule_7_reviewer",
        sender="tester",
        receiver="reviewer",
        task_id="repair",
        turn_id=7,
        selected_items=[
            _item(
                "c1",
                "constraint",
                "Do not modify tests; preserve the public add(a, b) API.",
                status="active_constraint",
                priority="critical",
            ),
            _item(
                "t1",
                "tool_state_delta",
                "run_tests passed returncode=0 for the current diff fingerprint.",
                status="verified",
                priority="high",
                source_agent="tester",
            ),
            _item(
                "a1",
                "artifact",
                "calculator.py now returns a + b for positive and mixed-sign inputs.",
                priority="high",
                source_agent="coder",
            ),
        ],
        selected_edges=[VACTHEdge(source="a1", relation="supports", target="t1")],
        budget={"used_tokens": 36, "max_tokens": 140},
    )

    rendered = render_capsule(capsule, compact=True, max_item_chars=90)

    assert "VACTH compact capsule" in rendered
    assert "Active Constraints:" in rendered
    assert "Tool State:" in rendered
    assert "Artifacts:" in rendered
    assert "c1 src=retriever status=active_constraint priority=critical" in rendered
    assert "t1 src=tester status=verified priority=high" in rendered
    assert "a1 src=coder status=observed priority=high" in rendered
    assert "evidence quote" not in rendered
    assert estimate_tokens(rendered) <= 140


def test_role_specific_capsule_budget_can_override_default_budget() -> None:
    runtime = VACTHRuntime(task_id="repair", token_budget=1000)
    runtime.update(
        [
            _item("short-critical", "constraint", "Keep API stable", priority="critical"),
            _item("short-test", "tool_state_delta", "Tests passed", status="verified", priority="high"),
            _item(
                "long-note",
                "fact",
                " ".join(["low value background"] * 40),
                priority="low",
            ),
        ]
    )

    capsule, routing = runtime.build_capsule(
        receiver="reviewer",
        sender="tester",
        turn_id=3,
        token_budget=8,
    )

    assert capsule.budget["max_tokens"] == 8
    assert capsule.budget["used_tokens"] <= 8
    assert routing["budget"] == 8
    assert {item.item_id for item in capsule.selected_items} == {"short-critical", "short-test"}


def test_extraction_prompt_limits_existing_items_and_items_per_turn() -> None:
    existing_items = [
        {
            "item_id": f"v{index}",
            "slot": "fact",
            "content": f"fact {index}",
            "merge_key": f"fact:{index}",
            "epistemic_status": "observed",
            "priority": "medium",
            "source_agent": "retriever",
            "created_at_turn": index,
            "evidence": [{"quote": "large evidence should not be copied"}],
        }
        for index in range(20)
    ]

    prompt = build_extraction_prompt(
        task="Fix the bug.",
        sender="coder",
        source_messages=[
            {
                "message_id": "m1",
                "turn_id": 2,
                "source": "coder",
                "type": "TextMessage",
                "content": "done",
            }
        ],
        existing_items=existing_items,
        max_existing_items=2,
        max_items_per_turn=6,
    )

    assert "Emit at most 6 high-value items" in prompt
    assert '"item_id": "v18"' in prompt
    assert '"item_id": "v19"' in prompt
    assert '"item_id": "v17"' not in prompt
    assert "large evidence should not be copied" not in prompt


def test_hybrid_heuristic_extracts_structured_tool_events() -> None:
    event = {
        "message_id": "tool-event-1",
        "turn_id": 4,
        "source": "tester",
        "role": "event",
        "type": "ToolCallExecutionEvent",
        "payload": {
            "content": [
                {
                    "name": "run_tests",
                    "content": json.dumps(
                        {
                            "returncode": 1,
                            "passed": False,
                            "stderr": "FAILED test_calc.py::test_add AssertionError: -1 != 5",
                        }
                    ),
                    "is_error": False,
                },
                {
                    "name": "replace_text",
                    "content": json.dumps(
                        {
                            "path": "calculator.py",
                            "changed": True,
                            "diff": "-    return a - b\n+    return a + b\n",
                        }
                    ),
                    "is_error": False,
                },
                {
                    "name": "git_diff",
                    "content": "-    return a - b\n+    return a + b\n",
                    "is_error": False,
                },
            ]
        },
    }

    items = heuristic_items_from_messages(
        sender="tester",
        source_messages=[event],
        turn_id=5,
        next_item_index=10,
    )

    assert [item.item_id for item in items] == ["v10", "v11", "v12"]
    assert all(item.slot == "tool_state_delta" for item in items)
    assert items[0].normalized_predicate == "tool.run_tests"
    assert items[0].epistemic_status == "observed"
    assert "run_tests failed returncode=1" in items[0].content
    assert items[1].arguments["path"] == "calculator.py"
    assert "replace_text changed=True path=calculator.py" in items[1].content
    assert items[2].normalized_predicate == "tool.git_diff"
    assert "diff_excerpt" in items[2].arguments
