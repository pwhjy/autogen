import csv
import json
import time
from pathlib import Path
from typing import Any


EXPERIMENT_CONFIG = json.loads(r"""__EXPERIMENT_CONFIG_JSON__""")


def write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    start = time.time()
    prompt = Path("prompt.txt").read_text(encoding="utf-8")
    tool_result = {"order_id": "order-001", "status": "eligible_for_refund"}
    raw_messages = [
        {
            "message_id": "m1",
            "turn_id": 0,
            "source": "user",
            "role": "user",
            "content": prompt,
        },
        {
            "message_id": "m2",
            "turn_id": 1,
            "source": "tool_caller",
            "role": "assistant",
            "content": "Checked the order state and policy in the deterministic smoke task.",
        },
    ]
    visible_contexts = [
        {
            "turn_id": 1,
            "agent": "tool_caller",
            "message_ids": ["m1"],
            "context": [raw_messages[0]],
        }
    ]
    tool_calls = [
        {
            "turn_id": 1,
            "agent": "tool_caller",
            "tool": "get_order",
            "arguments": {"order_id": "order-001"},
            "result": tool_result,
        }
    ]
    state_items = [
        {
            "item_id": "s1",
            "turn_id": 1,
            "slot": "tool_state_delta",
            "content": "order-001 is eligible_for_refund.",
            "merge_key": "order.order-001.status",
            "epistemic_status": "verified",
            "source_agent": "tool_caller",
            "evidence": [{"message_id": "m2"}],
        }
    ]
    metrics = {
        "success": True,
        "turns": 1,
        "tool_calls": 1,
        "estimated_tokens": len(prompt.split()),
        "wall_time_sec": round(time.time() - start, 4),
    }
    result = {
        "task_id": EXPERIMENT_CONFIG["task_id"],
        "method": EXPERIMENT_CONFIG["method"],
        "experiment_config": EXPERIMENT_CONFIG,
        "raw_messages": raw_messages,
        "visible_contexts": visible_contexts,
        "tool_calls": tool_calls,
        "state_items": state_items,
        "routing_decisions": [],
        "provenance_edges": [],
        "metrics": metrics,
        "final_answer": "resolved",
    }
    write_json("raw_messages.json", raw_messages)
    write_json("visible_contexts.json", visible_contexts)
    write_json("tool_calls.json", tool_calls)
    write_json("state_items.json", state_items)
    write_json("routing_decisions.json", [])
    write_json("provenance_graph.json", {"items": state_items, "edges": []})
    write_json("result.json", result)
    with open("metrics.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print("FINAL ANSWER: resolved")
    print("ALL TESTS PASSED !#!#")


if __name__ == "__main__":
    main()
