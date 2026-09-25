import asyncio
import contextlib
import csv
import difflib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult, TerminatedException, TerminationCondition
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    StopMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.vacth import (
    VACTHRuntime,
    build_extraction_prompt,
    heuristic_items_from_messages,
    parse_extraction_output,
)
from autogen_agentchat.vacth.schema import VACTHCapsule, VACTHEdge, VACTHStateItem
from autogen_core import CancellationToken
from autogen_core.model_context import (
    BufferedChatCompletionContext,
    ChatCompletionContext,
)
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,  # type: ignore
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema
from model_client_factory import describe_model_config, load_model_client
from pydantic import BaseModel

EXPERIMENT_CONFIG = json.loads(r"""__EXPERIMENT_CONFIG_JSON__""")


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(value), fh, ensure_ascii=False, indent=2, sort_keys=True)


def append_jsonl(path: str, value: Any) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True))
        fh.write("\n")


def model_call_timeout_seconds() -> float | None:
    raw_timeout = os.environ.get("VACTHBENCH_MODEL_CALL_TIMEOUT_SEC") or os.environ.get("VACTHBENCH_OPENAI_TIMEOUT", "")
    if not raw_timeout:
        return None
    try:
        timeout = float(raw_timeout)
    except ValueError:
        return None
    return timeout if timeout > 0 else None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def model_call_retry_count() -> int:
    return max(0, env_int("VACTHBENCH_MODEL_CALL_RETRIES", 2))


def is_retryable_model_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    retry_markers = (
        "internalservererror",
        "serviceunavailable",
        "apiconnectionerror",
        "apistatuserror",
        "timeout",
        "temporarily unavailable",
        "stream disconnected",
        "stream error",
        "stream closed before response.completed",
        "error code: 408",
        "error code: 429",
        "error code: 500",
        "error code: 502",
        "error code: 503",
        "error code: 504",
    )
    return any(marker in name or marker in text for marker in retry_markers)


def truncate_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max(1, max_chars // 2)
    return text[:half].rstrip() + f"\n...[truncated {len(text) - max_chars} chars]...\n" + text[-half:].lstrip()


def output_tail(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes | bytearray):
        text = bytes(value).decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-max_chars:]


def write_metrics_csv(path: str, metrics: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def parse_json_maybe(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"_raw": value}


def usage_to_dict(usage: RequestUsage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
    }


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+")
STRUCTURED_SUMMARY_SLOTS = (
    "facts",
    "constraints",
    "decisions",
    "hypotheses",
    "tool_states",
    "open_issues",
)


def retrieval_terms(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_PATTERN.findall(text))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left).intersection(right)
    dot = sum(left[token] * right[token] for token in overlap)
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def empty_structured_state() -> dict[str, list[str]]:
    return {slot: [] for slot in STRUCTURED_SUMMARY_SLOTS}


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("structured summary output must be a JSON object")
    return value


def normalize_structured_state(value: dict[str, Any], *, max_items_per_slot: int) -> dict[str, list[str]]:
    normalized = empty_structured_state()
    for slot in STRUCTURED_SUMMARY_SLOTS:
        raw_items = value.get(slot, [])
        if not isinstance(raw_items, list):
            raw_items = [raw_items]
        seen: set[str] = set()
        for item in raw_items:
            if isinstance(item, str):
                text = item.strip()
            else:
                text = json.dumps(to_jsonable(item), ensure_ascii=False, sort_keys=True)
            if not text or text in seen:
                continue
            normalized[slot].append(text)
            seen.add(text)
            if len(normalized[slot]) >= max_items_per_slot:
                break
    return normalized


class ExperimentLogger:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.raw_messages: list[dict[str, Any]] = []
        self.visible_contexts: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.state_items: list[dict[str, Any]] = []
        self.routing_decisions: list[dict[str, Any]] = []
        self.provenance_edges: list[dict[str, Any]] = []
        self.model_usage: list[dict[str, Any]] = []
        self.summaries: list[dict[str, Any]] = []
        self.memory_items: list[dict[str, Any]] = []
        self.structured_states: list[dict[str, Any]] = []
        self.capsules: list[dict[str, Any]] = []
        self.vacth_extractions: list[dict[str, Any]] = []
        self.model_events: list[dict[str, Any]] = []
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}
        self._latest_turn_by_agent: dict[str, int] = {}
        self._turn = 0

    def next_turn(self, agent: str | None = None) -> int:
        self._turn += 1
        if agent is not None:
            self._latest_turn_by_agent[agent] = self._turn
        return self._turn

    def current_turn(self, source: str) -> int:
        if source == "user":
            return 0
        if source not in self._latest_turn_by_agent:
            return self.next_turn(source)
        return self._latest_turn_by_agent[source]

    def log_visible_context(
        self,
        *,
        turn_id: int,
        agent: str,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema],
        mode: Literal["create", "create_stream"],
    ) -> None:
        context: list[dict[str, Any]] = []
        for index, message in enumerate(messages, start=1):
            payload = to_jsonable(message)
            content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            context.append(
                {
                    "message_id": f"llm{turn_id}_{index}",
                    "turn_id": turn_id,
                    "source": payload.get("source", payload.get("type", "system"))
                    if isinstance(payload, dict)
                    else "unknown",
                    "role": payload.get("type", "LLMMessage") if isinstance(payload, dict) else "LLMMessage",
                    "content": payload.get("content", content) if isinstance(payload, dict) else content,
                    "payload": payload,
                    "estimated_tokens": estimate_tokens(content),
                }
            )
        tool_schemas = [self._serialize_tool(tool) for tool in tools]
        record = {
            "turn_id": turn_id,
            "agent": agent,
            "mode": mode,
            "message_ids": [message["message_id"] for message in context],
            "context": context,
            "tools": tool_schemas,
            "estimated_tokens": sum(item["estimated_tokens"] for item in context),
        }
        if self.config["method"] == "sliding_window":
            record["window_policy"] = {
                "type": "buffered_chat_completion_context",
                "buffer_size": int(self.config.get("context_window_messages", 4)),
            }
            record["dropped_messages"] = self._estimate_dropped_messages(context)
        self.visible_contexts.append(record)

    def _serialize_tool(self, tool: Tool | ToolSchema) -> dict[str, Any]:
        if hasattr(tool, "schema"):
            return to_jsonable(tool.schema)
        return to_jsonable(dict(tool))

    def log_model_usage(self, *, turn_id: int, agent: str, usage: RequestUsage | None) -> None:
        usage_dict = usage_to_dict(usage)
        if usage_dict is None:
            return
        self.model_usage.append(
            {
                "turn_id": turn_id,
                "agent": agent,
                **usage_dict,
                "total_tokens": usage_dict["prompt_tokens"] + usage_dict["completion_tokens"],
            }
        )

    def log_model_event(self, **event: Any) -> None:
        record = {
            "timestamp": time.time(),
            **event,
        }
        self.model_events.append(record)
        append_jsonl("model_events.jsonl", record)

    def _estimate_dropped_messages(self, visible_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        visible_text = "\n".join(str(message.get("content", "")) for message in visible_context)
        dropped: list[dict[str, Any]] = []
        for message in self.raw_messages:
            if message.get("role") not in {"user", "assistant"}:
                continue
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            probe = content[: min(160, len(content))]
            if probe not in visible_text:
                dropped.append(
                    {
                        "message_id": message.get("message_id"),
                        "turn_id": message.get("turn_id"),
                        "source": message.get("source"),
                        "role": message.get("role"),
                        "type": message.get("type"),
                        "estimated_tokens": message.get("estimated_tokens"),
                    }
                )
        return dropped

    def log_stream_message(self, message: BaseAgentEvent | BaseChatMessage) -> dict[str, Any]:
        source = getattr(message, "source", "unknown")
        turn_id = self.current_turn(source)
        content = message.to_text()
        role = "user" if source == "user" else "event"
        if isinstance(message, BaseChatMessage) and source != "user":
            role = "assistant"
        raw_message = {
            "message_id": getattr(message, "id", f"m{len(self.raw_messages) + 1}"),
            "turn_id": turn_id,
            "source": source,
            "role": role,
            "type": message.__class__.__name__,
            "content": content,
            "estimated_tokens": estimate_tokens(content),
            "models_usage": usage_to_dict(getattr(message, "models_usage", None)),
            "payload": to_jsonable(message.dump()),
        }
        self.raw_messages.append(raw_message)
        self._log_tool_events(message, turn_id)
        return raw_message

    def log_synthetic_message(self, *, turn_id: int, source: str, role: str, content: str) -> dict[str, Any]:
        message = {
            "message_id": f"m{len(self.raw_messages) + 1}",
            "turn_id": turn_id,
            "source": source,
            "role": role,
            "type": "SyntheticMessage",
            "content": content,
            "estimated_tokens": estimate_tokens(content),
        }
        self.raw_messages.append(message)
        return message

    def log_synthetic_visible_context(self, *, turn_id: int, agent: str, history: list[dict[str, Any]]) -> None:
        self.visible_contexts.append(
            {
                "turn_id": turn_id,
                "agent": agent,
                "mode": "deterministic_smoke",
                "message_ids": [message["message_id"] for message in history],
                "context": history,
                "tools": [],
                "estimated_tokens": sum(message.get("estimated_tokens", 0) for message in history),
            }
        )

    def log_synthetic_tool_call(
        self,
        *,
        turn_id: int,
        agent: str,
        tool: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self.tool_calls.append(
            {
                "turn_id": turn_id,
                "agent": agent,
                "tool": tool,
                "arguments": arguments,
                "result": result,
            }
        )

    def log_summary(
        self,
        *,
        turn_id: int,
        sender: str,
        summary: str,
        previous_summary: str,
        source_messages: list[dict[str, Any]],
    ) -> None:
        source_tokens = sum(message.get("estimated_tokens", 0) for message in source_messages)
        self.summaries.append(
            {
                "summary_id": f"summary_{len(self.summaries) + 1}",
                "turn_id": turn_id,
                "sender": sender,
                "content": summary,
                "previous_summary": previous_summary,
                "source_message_ids": [message.get("message_id") for message in source_messages],
                "source_estimated_tokens": source_tokens,
                "estimated_tokens": estimate_tokens(summary),
            }
        )
        self.log_synthetic_message(
            turn_id=turn_id,
            source="summary_runtime",
            role="system",
            content=summary,
        )

    def log_memory_item(self, message: dict[str, Any]) -> dict[str, Any]:
        item = {
            "item_id": f"mem_{len(self.memory_items) + 1}",
            "turn_id": message.get("turn_id"),
            "source": message.get("source"),
            "role": message.get("role"),
            "type": message.get("type"),
            "content": message.get("content", ""),
            "message_id": message.get("message_id"),
            "estimated_tokens": message.get("estimated_tokens", 0),
        }
        self.memory_items.append(item)
        return item

    def log_structured_state(
        self,
        *,
        turn_id: int,
        sender: str,
        state: dict[str, list[str]],
        previous_state: dict[str, list[str]],
        source_messages: list[dict[str, Any]],
        valid: bool,
        parse_error: str | None,
        raw_update: str,
    ) -> None:
        state_text = json.dumps(state, ensure_ascii=False, sort_keys=True)
        self.structured_states.append(
            {
                "state_id": f"structured_{len(self.structured_states) + 1}",
                "turn_id": turn_id,
                "sender": sender,
                "state": state,
                "previous_state": previous_state,
                "source_message_ids": [message.get("message_id") for message in source_messages],
                "source_estimated_tokens": sum(message.get("estimated_tokens", 0) for message in source_messages),
                "estimated_tokens": estimate_tokens(state_text),
                "schema_valid": valid,
                "parse_error": parse_error,
                "raw_update": raw_update,
            }
        )
        self.log_synthetic_message(
            turn_id=turn_id,
            source="structured_summary_runtime",
            role="system",
            content=state_text,
        )

    def log_vacth_extraction(
        self,
        *,
        turn_id: int,
        sender: str,
        raw_update: str,
        parse_valid: bool,
        parse_error: str | None,
        extraction_mode: str,
        reask_attempts: int,
        items: list[VACTHStateItem],
        edges: list[VACTHEdge],
        source_messages: list[dict[str, Any]],
    ) -> None:
        item_dicts = [item.model_dump(mode="json") for item in items]
        edge_dicts = [edge.model_dump(mode="json") for edge in edges]
        self.state_items.extend(item_dicts)
        self.provenance_edges.extend(edge_dicts)
        self.vacth_extractions.append(
            {
                "turn_id": turn_id,
                "sender": sender,
                "raw_update": raw_update,
                "parse_valid": parse_valid,
                "parse_error": parse_error,
                "extraction_mode": extraction_mode,
                "reask_attempts": reask_attempts,
                "item_ids": [item.item_id for item in items],
                "edge_count": len(edges),
                "source_message_ids": [message.get("message_id") for message in source_messages],
            }
        )

    def log_capsule(self, *, capsule: VACTHCapsule, routing_decision: dict[str, Any]) -> None:
        self.capsules.append(capsule.model_dump(mode="json"))
        self.routing_decisions.append(routing_decision)

    def log_retrieval(
        self,
        *,
        turn_id: int,
        receiver: str,
        query: str,
        retrieved: list[dict[str, Any]],
        top_k: int,
    ) -> None:
        self.routing_decisions.append(
            {
                "turn_id": turn_id,
                "receiver": receiver,
                "method": self.config["method"],
                "retrieval_method": "lexical_cosine",
                "query": query,
                "top_k": top_k,
                "selected_item_ids": [item["item_id"] for item in retrieved],
                "scores": [{"item_id": item["item_id"], "score": item.get("score", 0)} for item in retrieved],
                "budget": int(self.config.get("token_budget", 0)),
                "estimated_tokens": sum(int(item.get("estimated_tokens", 0)) for item in retrieved),
            }
        )

    def _log_tool_events(self, message: BaseAgentEvent | BaseChatMessage, turn_id: int) -> None:
        if isinstance(message, ToolCallRequestEvent):
            for call in message.content:
                self._pending_tool_calls[call.id] = {
                    "turn_id": turn_id,
                    "agent": message.source,
                    "tool": call.name,
                    "arguments": parse_json_maybe(call.arguments),
                    "call_id": call.id,
                }
        elif isinstance(message, ToolCallExecutionEvent):
            for result in message.content:
                pending = self._pending_tool_calls.pop(
                    result.call_id,
                    {
                        "turn_id": turn_id,
                        "agent": message.source,
                        "tool": result.name,
                        "arguments": {},
                        "call_id": result.call_id,
                    },
                )
                self.tool_calls.append(
                    {
                        "turn_id": pending["turn_id"],
                        "agent": pending["agent"],
                        "tool": pending["tool"],
                        "arguments": pending["arguments"],
                        "call_id": pending["call_id"],
                        "result": {
                            "name": result.name,
                            "content": result.content,
                            "is_error": result.is_error,
                        },
                    }
                )

    def save(self, final_answer: str, metrics: dict[str, Any]) -> None:
        write_json("raw_messages.json", self.raw_messages)
        write_json("visible_contexts.json", self.visible_contexts)
        write_json("model_usage.json", self.model_usage)
        write_json("model_events.json", self.model_events)
        write_json("tool_calls.json", self.tool_calls)
        write_json("summaries.json", self.summaries)
        write_json(
            "structured_summary.json",
            {
                "schema": list(STRUCTURED_SUMMARY_SLOTS),
                "states": self.structured_states,
            },
        )
        write_json("capsules.json", self.capsules)
        write_json("vacth_extractions.json", self.vacth_extractions)
        write_json(
            "vector_memory.json",
            {
                "memory_items": self.memory_items,
                "retrievals": self.routing_decisions if self.config["method"] == "vector_memory" else [],
            },
        )
        write_json("state_items.json", self.state_items)
        write_json("routing_decisions.json", self.routing_decisions)
        write_json(
            "provenance_graph.json",
            {
                "items": self.state_items,
                "edges": self.provenance_edges,
            },
        )

        result = {
            "task_id": self.config["task_id"],
            "method": self.config["method"],
            "experiment_config": self.config,
            "raw_messages": self.raw_messages,
            "visible_contexts": self.visible_contexts,
            "model_usage": self.model_usage,
            "model_events": self.model_events,
            "tool_calls": self.tool_calls,
            "summaries": self.summaries,
            "structured_states": self.structured_states,
            "capsules": self.capsules,
            "vacth_extractions": self.vacth_extractions,
            "memory_items": self.memory_items,
            "state_items": self.state_items,
            "routing_decisions": self.routing_decisions,
            "provenance_edges": self.provenance_edges,
            "metrics": metrics,
            "final_answer": final_answer,
        }
        write_json("result.json", result)
        write_metrics_csv("metrics.csv", metrics)


class RecordingModelClient(ChatCompletionClient):
    def __init__(self, client: ChatCompletionClient, agent: str, logger: ExperimentLogger) -> None:
        self._client = client
        self._agent = agent
        self._logger = logger

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        turn_id = self._logger.next_turn(self._agent)
        self._logger.log_visible_context(
            turn_id=turn_id,
            agent=self._agent,
            messages=messages,
            tools=tools,
            mode="create",
        )
        timeout_sec = model_call_timeout_seconds()
        started = time.monotonic()
        self._logger.log_model_event(
            turn_id=turn_id,
            agent=self._agent,
            mode="create",
            status="start",
            message_count=len(messages),
            tool_count=len(tools),
            timeout_sec=timeout_sec,
            json_output=bool(json_output),
        )
        attempts = model_call_retry_count() + 1
        for attempt in range(1, attempts + 1):
            try:
                create_coro = self._client.create(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    json_output=json_output,
                    extra_create_args=extra_create_args,
                    cancellation_token=cancellation_token,
                )
                result = (
                    await asyncio.wait_for(create_coro, timeout=timeout_sec)
                    if timeout_sec is not None
                    else await create_coro
                )
                break
            except Exception as exc:
                if attempt < attempts and is_retryable_model_error(exc):
                    delay_sec = min(60, 5 * attempt)
                    self._logger.log_model_event(
                        turn_id=turn_id,
                        agent=self._agent,
                        mode="create",
                        status="retry",
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        elapsed_sec=round(time.monotonic() - started, 3),
                        delay_sec=delay_sec,
                        error_type=type(exc).__name__,
                        error=str(exc)[-2000:],
                    )
                    await asyncio.sleep(delay_sec)
                    continue
                self._logger.log_model_event(
                    turn_id=turn_id,
                    agent=self._agent,
                    mode="create",
                    status="error",
                    attempt=attempt,
                    elapsed_sec=round(time.monotonic() - started, 3),
                    error_type=type(exc).__name__,
                    error=str(exc)[-2000:],
                )
                raise
        self._logger.log_model_usage(turn_id=turn_id, agent=self._agent, usage=result.usage)
        self._logger.log_model_event(
            turn_id=turn_id,
            agent=self._agent,
            mode="create",
            status="finish",
            elapsed_sec=round(time.monotonic() - started, 3),
            usage=usage_to_dict(result.usage),
        )
        return result

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        async def stream() -> AsyncGenerator[str | CreateResult, None]:
            turn_id = self._logger.next_turn(self._agent)
            self._logger.log_visible_context(
                turn_id=turn_id,
                agent=self._agent,
                messages=messages,
                tools=tools,
                mode="create_stream",
            )
            started = time.monotonic()
            self._logger.log_model_event(
                turn_id=turn_id,
                agent=self._agent,
                mode="create_stream",
                status="start",
                message_count=len(messages),
                tool_count=len(tools),
                timeout_sec=model_call_timeout_seconds(),
                json_output=bool(json_output),
            )
            try:
                async for chunk in self._client.create_stream(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    json_output=json_output,
                    extra_create_args=extra_create_args,
                    cancellation_token=cancellation_token,
                ):
                    if isinstance(chunk, CreateResult):
                        self._logger.log_model_usage(turn_id=turn_id, agent=self._agent, usage=chunk.usage)
                        self._logger.log_model_event(
                            turn_id=turn_id,
                            agent=self._agent,
                            mode="create_stream",
                            status="finish",
                            elapsed_sec=round(time.monotonic() - started, 3),
                            usage=usage_to_dict(chunk.usage),
                        )
                    yield chunk
            except Exception as exc:
                self._logger.log_model_event(
                    turn_id=turn_id,
                    agent=self._agent,
                    mode="create_stream",
                    status="error",
                    elapsed_sec=round(time.monotonic() - started, 3),
                    error_type=type(exc).__name__,
                    error=str(exc)[-2000:],
                )
                raise

        return stream()

    async def close(self) -> None:
        await self._client.close()

    def actual_usage(self) -> RequestUsage:
        return self._client.actual_usage()

    def total_usage(self) -> RequestUsage:
        return self._client.total_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._client.count_tokens(messages, tools=tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return self._client.remaining_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        return self._client.capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self._client.model_info


class SoftwareRepairTools:
    SKIPPED_WORKSPACE_DIRS = {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "node_modules",
        "site-packages",
        "venv",
    }
    SKIPPED_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".png", ".jpg", ".jpeg", ".gif", ".pdf"}
    MAX_TEXT_FILE_BYTES = 2_000_000
    DEFAULT_LIST_FILES_LIMIT = 400
    DEFAULT_SEARCH_MATCH_LIMIT = 80
    DEFAULT_SEARCH_LINE_CHARS = 240
    DEFAULT_READ_FILE_CHARS = 50_000
    DEFAULT_TEST_OUTPUT_CHARS = 12_000
    DEFAULT_DIFF_CHARS = 20_000
    DEFAULT_TEST_EVIDENCE_CHARS = 10_000
    DEFAULT_TEST_EVIDENCE_CONTEXT_LINES = 24
    MISSING_MODULE_RE = re.compile(
        r"No module named ['\"]?([A-Za-z0-9_.-]+)['\"]?",
        re.IGNORECASE,
    )
    MISSING_VERSION_DEP_RE = re.compile(
        r"\b(Numpy|NumPy|SciPy|pytest|packaging)\b version [^\n]+ must be installed",
        re.IGNORECASE,
    )
    TEST_DEPENDENCY_PACKAGES = {
        "pytest": ("pytest",),
        "numpy": ("numpy",),
        "packaging": ("packaging",),
        "distutils": ("setuptools",),
        "setuptools_scm": ("setuptools_scm",),
        "erfa": ("pyerfa",),
        "scipy": ("scipy",),
        "hypothesis": ("hypothesis",),
        "pytz": ("pytz",),
        "sqlparse": ("sqlparse",),
    }

    def __init__(
        self,
        workspace: Path,
        test_command: list[str] | None = None,
        task_spec: dict[str, Any] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.test_command = test_command or [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            ".",
            "-p",
            "test_*.py",
        ]
        self.original_files = self._snapshot()
        self.task_spec = task_spec or {}
        self._last_test_result: dict[str, Any] | None = None
        self._last_test_fingerprint: str | None = None
        self._last_test_command: list[str] | None = None
        self._last_passed_fingerprint: str | None = None

    def _is_workspace_candidate_file(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if path.name.startswith("._"):
            return False
        relative = path.relative_to(self.workspace)
        if any(part in self.SKIPPED_WORKSPACE_DIRS for part in relative.parts):
            return False
        if path.suffix.lower() in self.SKIPPED_SUFFIXES:
            return False
        try:
            return path.stat().st_size <= self.MAX_TEXT_FILE_BYTES
        except OSError:
            return False

    def _read_workspace_text(self, path: Path) -> str | None:
        if not self._is_workspace_candidate_file(path):
            return None
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None

    def _snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for path in self.workspace.rglob("*"):
            text = self._read_workspace_text(path)
            if text is not None:
                snapshot[path.relative_to(self.workspace).as_posix()] = text
        return snapshot

    def _safe_path(self, path: str) -> Path:
        raw_path = Path(path)
        if not raw_path.is_absolute() and raw_path.parts[:1] == ("workspace",):
            raw_path = Path(*raw_path.parts[1:]) if len(raw_path.parts) > 1 else Path(".")
        candidate = raw_path.resolve() if raw_path.is_absolute() else (self.workspace / raw_path).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {path}") from exc
        return candidate

    def list_files(self) -> str:
        """List files available in the repair workspace."""
        limit = env_int("VACTHBENCH_LIST_FILES_LIMIT", self.DEFAULT_LIST_FILES_LIMIT)
        files = sorted(
            path.relative_to(self.workspace).as_posix()
            for path in self.workspace.rglob("*")
            if self._is_workspace_candidate_file(path)
        )
        visible_files = files[:limit] if limit > 0 else files
        return json.dumps(
            {
                "files": visible_files,
                "total_files": len(files),
                "truncated": len(visible_files) < len(files),
                "limit": limit,
            },
            ensure_ascii=False,
        )

    def read_file(
        self,
        path: Annotated[
            str,
            "Relative file path inside the workspace, for example calculator.py.",
        ],
        start_line: Annotated[
            int | None,
            "Optional 1-based start line for a focused snippet.",
        ] = None,
        end_line: Annotated[
            int | None,
            "Optional 1-based end line for a focused snippet.",
        ] = None,
    ) -> str:
        """Read one source or test file from the repair workspace."""
        target = self._safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        content = self._read_workspace_text(target)
        if content is None:
            raise ValueError(f"File is binary, too large, or outside text scope: {path}")
        total_lines = len(content.splitlines())
        selected_start = None
        selected_end = None
        if start_line is not None or end_line is not None:
            selected_start = max(1, int(start_line or 1))
            selected_end = min(total_lines, int(end_line or total_lines))
            if selected_end < selected_start:
                selected_start, selected_end = selected_end, selected_start
            lines = content.splitlines(keepends=True)
            content = "".join(lines[selected_start - 1 : selected_end])
        max_chars = env_int("VACTHBENCH_READ_FILE_MAX_CHARS", self.DEFAULT_READ_FILE_CHARS)
        visible_content = truncate_chars(content, max_chars)
        return json.dumps(
            {
                "path": target.relative_to(self.workspace).as_posix(),
                "content": visible_content,
                "total_chars": len(content),
                "total_lines": total_lines,
                "start_line": selected_start,
                "end_line": selected_end,
                "truncated": visible_content != content,
                "max_chars": max_chars,
            },
            ensure_ascii=False,
        )

    def search_repo(
        self,
        query: Annotated[
            str,
            "Literal text to search for across files in the workspace.",
        ],
    ) -> str:
        """Search source and test files for a literal string."""
        matches: list[dict[str, Any]] = []
        total_matches = 0
        match_limit = env_int("VACTHBENCH_SEARCH_MATCH_LIMIT", self.DEFAULT_SEARCH_MATCH_LIMIT)
        line_chars = env_int("VACTHBENCH_SEARCH_LINE_CHARS", self.DEFAULT_SEARCH_LINE_CHARS)
        for path in sorted(self.workspace.rglob("*")):
            text = self._read_workspace_text(path)
            if text is None:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    total_matches += 1
                    if match_limit <= 0 or len(matches) < match_limit:
                        matches.append(
                            {
                                "path": path.relative_to(self.workspace).as_posix(),
                                "line": line_number,
                                "text": truncate_chars(line, line_chars),
                            }
                        )
        return json.dumps(
            {
                "query": query,
                "matches": matches,
                "total_matches": total_matches,
                "truncated": len(matches) < total_matches,
                "match_limit": match_limit,
            },
            ensure_ascii=False,
        )

    def replace_text(
        self,
        path: Annotated[str, "Relative file path to modify."],
        old: Annotated[str, "Exact text to replace."],
        new: Annotated[str, "Replacement text."],
    ) -> str:
        """Replace the first exact text occurrence in a workspace file."""
        target = self._safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        before = self._read_workspace_text(target)
        if before is None:
            raise ValueError(f"File is binary, too large, or outside text scope: {path}")
        if old not in before:
            return json.dumps(
                {
                    "path": target.relative_to(self.workspace).as_posix(),
                    "changed": False,
                    "error": "old text not found",
                },
                ensure_ascii=False,
            )
        after = before.replace(old, new, 1)
        target.write_text(after, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{target.relative_to(self.workspace).as_posix()}",
                tofile=f"b/{target.relative_to(self.workspace).as_posix()}",
            )
        )
        return json.dumps(
            {
                "path": target.relative_to(self.workspace).as_posix(),
                "changed": True,
                "diff": diff,
            },
            ensure_ascii=False,
        )

    def _compact_test_result(self, result: dict[str, Any]) -> dict[str, Any]:
        max_chars = env_int("VACTHBENCH_TEST_OUTPUT_MAX_CHARS", self.DEFAULT_TEST_OUTPUT_CHARS)
        compact = dict(result)
        for stream_name in ("stdout", "stderr"):
            value = str(compact.get(stream_name, ""))
            if not value:
                continue
            compact[f"{stream_name}_tail"] = value[-max_chars:]
            compact[f"{stream_name}_total_chars"] = len(value)
            compact[f"{stream_name}_truncated"] = len(value) > max_chars
            compact.pop(stream_name, None)
        compact["output_max_chars"] = max_chars
        return compact

    def _missing_module_name(self, stdout: str, stderr: str) -> str | None:
        output = f"{stdout}\n{stderr}"
        match = self.MISSING_MODULE_RE.search(output)
        if not match:
            version_match = self.MISSING_VERSION_DEP_RE.search(output)
            if not version_match:
                return None
            return version_match.group(1).lower()
        return match.group(1).split(".", 1)[0]

    def _install_test_dependency(self, module_name: str, env: dict[str, str]) -> dict[str, Any] | None:
        packages = self.TEST_DEPENDENCY_PACKAGES.get(module_name)
        if not packages:
            return None
        command = [sys.executable, "-m", "pip", "install", *packages]
        install_env = dict(env)
        install_env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        timeout_sec = env_int("VACTHBENCH_TEST_DEP_INSTALL_TIMEOUT_SEC", 300)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=str(self.workspace.resolve()),
                env=install_env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "module": module_name,
                "packages": list(packages),
                "command": command,
                "returncode": -1,
                "stdout_tail": output_tail(exc.stdout, 4000),
                "stderr_tail": output_tail(exc.stderr, 4000) + f"\nTIMEOUT after {timeout_sec}s",
            }
        return {
            "module": module_name,
            "packages": list(packages),
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }

    def _run_test_command(
        self,
        command: list[str],
        *,
        update_cache: bool,
    ) -> dict[str, Any]:
        fingerprint = self.workspace_fingerprint()
        workspace_path = str(self.workspace.resolve())
        env = repair_test_env(self.workspace)
        install_logs: list[dict[str, Any]] = []
        attempted_modules: set[str] = set()
        max_installs = env_int("VACTHBENCH_TEST_DEP_INSTALL_MAX_ATTEMPTS", 3)
        auto_install = os.environ.get("VACTHBENCH_AUTO_INSTALL_TEST_DEPS", "1").lower() not in {"0", "false", "no"}

        while True:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=workspace_path,
                env=env,
            )
            if completed.returncode == 0 or not auto_install:
                break
            module_name = self._missing_module_name(completed.stdout, completed.stderr)
            if module_name is None or module_name in attempted_modules or len(install_logs) >= max_installs:
                break
            install_log = self._install_test_dependency(module_name, env)
            if install_log is None:
                break
            install_logs.append(install_log)
            attempted_modules.add(module_name)
            if install_log.get("returncode") != 0:
                break

        result = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "passed": completed.returncode == 0,
            "cached": False,
            "executed": True,
            "dependency_install_attempts": install_logs,
        }
        if update_cache:
            self._last_test_result = result
            self._last_test_fingerprint = fingerprint
            self._last_test_command = list(command)
            if result["passed"]:
                self._last_passed_fingerprint = fingerprint
        return result

    def run_tests(self) -> str:
        """Run the unittest test suite for the repair workspace."""
        fingerprint = self.workspace_fingerprint()
        if (
            self._last_test_result is not None
            and self._last_test_fingerprint == fingerprint
            and self._last_test_command == self.test_command
        ):
            cached = dict(self._last_test_result)
            cached["cached"] = True
            cached["executed"] = False
            return json.dumps(self._compact_test_result(cached), ensure_ascii=False)

        result = self._run_test_command(self.test_command, update_cache=True)
        return json.dumps(self._compact_test_result(result), ensure_ascii=False)

    def test_status(self) -> str:
        """Report whether the current workspace diff already has a passing test result."""
        current_fingerprint = self.workspace_fingerprint()
        current_diff_tested = (
            self._last_test_result is not None
            and self._last_test_fingerprint == current_fingerprint
            and self._last_test_command == self.test_command
        )
        current_diff_verified = self.has_current_test_pass()
        last_result = dict(self._last_test_result or {})
        stdout = str(last_result.get("stdout", ""))
        stderr = str(last_result.get("stderr", ""))
        if stdout:
            last_result["stdout_tail"] = stdout[-2000:]
            last_result.pop("stdout", None)
        if stderr:
            last_result["stderr_tail"] = stderr[-2000:]
            last_result.pop("stderr", None)
        return json.dumps(
            {
                "current_fingerprint": current_fingerprint,
                "last_test_fingerprint": self._last_test_fingerprint,
                "last_passed_fingerprint": self._last_passed_fingerprint,
                "current_diff_tested": current_diff_tested,
                "current_diff_verified": current_diff_verified,
                "current_diff_failed": bool(current_diff_tested and not last_result.get("passed", False)),
                "last_test_command": self._last_test_command,
                "last_test_result": last_result or None,
                "changed_files": self.changed_files(),
            },
            ensure_ascii=False,
        )

    def workspace_fingerprint(self) -> str:
        """Return a stable fingerprint for the current repair diff."""
        digest = hashlib.sha256()
        current_files: dict[str, str] = {}
        for path in self.workspace.rglob("*"):
            text = self._read_workspace_text(path)
            if text is not None:
                current_files[path.relative_to(self.workspace).as_posix()] = text
        for relative_path in sorted(set(self.original_files) | set(current_files)):
            before = self.original_files.get(relative_path, "")
            after = current_files.get(relative_path, "")
            if before == after:
                continue
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(before.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\0")
            digest.update(after.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\0")
        return digest.hexdigest()

    def has_current_test_pass(self) -> bool:
        """Whether run_tests passed for the exact current workspace diff."""
        return self._last_passed_fingerprint == self.workspace_fingerprint()

    def changed_files(self) -> list[str]:
        """Return relative paths whose current content differs from the initial snapshot."""
        current_files: dict[str, str] = {}
        for path in self.workspace.rglob("*"):
            text = self._read_workspace_text(path)
            if text is not None:
                current_files[path.relative_to(self.workspace).as_posix()] = text
        return sorted(
            relative_path
            for relative_path in set(self.original_files) | set(current_files)
            if self.original_files.get(relative_path, "") != current_files.get(relative_path, "")
        )

    def git_diff(self) -> str:
        """Show a unified diff between the original and current workspace files."""
        diffs: list[str] = []
        current_files: dict[str, str] = {}
        for path in self.workspace.rglob("*"):
            text = self._read_workspace_text(path)
            if text is not None:
                current_files[path.relative_to(self.workspace).as_posix()] = text
        for relative_path in sorted(set(self.original_files) | set(current_files)):
            before = self.original_files.get(relative_path, "")
            after = current_files.get(relative_path, "")
            if before == after:
                continue
            diffs.append(
                "".join(
                    difflib.unified_diff(
                        before.splitlines(keepends=True),
                        after.splitlines(keepends=True),
                        fromfile=f"a/{relative_path}",
                        tofile=f"b/{relative_path}",
                    )
                )
            )
        diff_text = "\n".join(diffs) if diffs else "No changes."
        return truncate_chars(diff_text, env_int("VACTHBENCH_DIFF_MAX_CHARS", self.DEFAULT_DIFF_CHARS))

    def _test_patch_paths(self) -> list[str]:
        repo_source = self.task_spec.get("repo_source", {})
        test_patch = ""
        if isinstance(repo_source, dict):
            test_patch = str(repo_source.get("test_patch", ""))
        paths: list[str] = []
        for line in test_patch.splitlines():
            if not line.startswith("+++ b/"):
                continue
            path = line.removeprefix("+++ b/").strip()
            if path and path != "/dev/null" and is_test_path(path):
                paths.append(path)
        return sorted(set(paths))

    def _candidate_test_files(self, labels: Sequence[str]) -> list[str]:
        candidates: list[str] = []
        for label in labels:
            path = label.split("::", 1)[0]
            if path.endswith(".py") and is_test_path(path):
                candidates.append(path)
        candidates.extend(self._test_patch_paths())
        for path in command_test_paths(self.test_command):
            normalized = path.split("::", 1)[0]
            if normalized.endswith(".py") and is_test_path(normalized):
                candidates.append(normalized)
        seen: set[str] = set()
        existing: list[str] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (self.workspace / candidate).is_file():
                existing.append(candidate)
        return existing

    def _label_search_terms(self, label: str) -> list[str]:
        normalized = label.split("[", 1)[0].strip()
        terms: list[str] = []
        django_match = _DJANGO_UNMANAGED_TEST_RE.match(normalized)
        if django_match is not None:
            method_name, class_path = django_match.groups()
            terms.extend([method_name, class_path.rsplit(".", 1)[-1]])
        elif "::" in normalized:
            terms.extend(part for part in normalized.split("::")[1:] if part)
        else:
            terms.append(normalized.rsplit(".", 1)[-1])
        cleaned: list[str] = []
        for term in reversed(terms):
            term = term.split("[", 1)[0].strip()
            if term and term not in cleaned:
                cleaned.append(term)
        return cleaned

    def _snippet_for_label(
        self,
        *,
        label: str,
        candidate_files: Sequence[str],
        context_lines: int,
    ) -> dict[str, Any] | None:
        preferred_files = list(candidate_files)
        if "::" in label:
            label_path = label.split("::", 1)[0]
            if label_path in preferred_files:
                preferred_files.remove(label_path)
                preferred_files.insert(0, label_path)
        terms = self._label_search_terms(label)
        for relative_path in preferred_files:
            path = self.workspace / relative_path
            text = self._read_workspace_text(path)
            if text is None:
                continue
            lines = text.splitlines()
            match_index: int | None = None
            for term in terms:
                definition_patterns = (f"def {term}", f"class {term}")
                for index, line in enumerate(lines):
                    stripped = line.lstrip()
                    if stripped.startswith(definition_patterns):
                        match_index = index
                        break
                if match_index is not None:
                    break
            if match_index is None:
                for term in terms:
                    for index, line in enumerate(lines):
                        if term in line:
                            match_index = index
                            break
                    if match_index is not None:
                        break
            if match_index is None:
                continue
            start = max(0, match_index - context_lines)
            end = min(len(lines), match_index + context_lines + 1)
            return {
                "label": label,
                "path": relative_path,
                "start_line": start + 1,
                "end_line": end,
                "content": "\n".join(lines[start:end]),
            }
        return None

    def test_evidence(self) -> str:
        """Return focused snippets for the selected SWE-bench FAIL_TO_PASS tests."""
        labels = swebench_expected_tests(self.task_spec, include_pass_to_pass=False)
        if not labels:
            return json.dumps(
                {
                    "available": False,
                    "reason": "no fail_to_pass metadata",
                    "snippets": [],
                },
                ensure_ascii=False,
            )
        candidate_files = self._candidate_test_files(labels)
        context_lines = env_int(
            "VACTHBENCH_TEST_EVIDENCE_CONTEXT_LINES",
            self.DEFAULT_TEST_EVIDENCE_CONTEXT_LINES,
        )
        max_chars = env_int(
            "VACTHBENCH_TEST_EVIDENCE_MAX_CHARS",
            self.DEFAULT_TEST_EVIDENCE_CHARS,
        )
        snippets: list[dict[str, Any]] = []
        used_chars = 0
        truncated = False
        for label in labels:
            snippet = self._snippet_for_label(
                label=label,
                candidate_files=candidate_files,
                context_lines=context_lines,
            )
            if snippet is None:
                continue
            content = str(snippet.get("content", ""))
            remaining = max_chars - used_chars
            if max_chars > 0 and remaining <= 0:
                truncated = True
                break
            if max_chars > 0 and len(content) > remaining:
                snippet["content"] = truncate_chars(content, remaining)
                truncated = True
            used_chars += len(str(snippet.get("content", "")))
            snippets.append(snippet)
        return json.dumps(
            {
                "available": bool(snippets),
                "fail_to_pass": labels,
                "candidate_test_files": candidate_files,
                "snippets": snippets,
                "truncated": truncated,
                "max_chars": max_chars,
            },
            ensure_ascii=False,
        )


def read_prompt() -> str:
    return Path("prompt.txt").read_text(encoding="utf-8")


DEFAULT_SOFTWARE_TASK: dict[str, Any] = {
    "id": "calc_add",
    "category": "operator_bug",
    "task": "Fix calculator.add so that it returns the arithmetic sum of two numbers.",
    "constraints": "Keep the public function name add(a, b). Do not change the tests.",
    "gold_state": "calculator.add(2, 3) must return 5 and calculator.add(-2, 3) must return 1.",
    "repo_files": {
        "calculator.py": ("def add(a, b):\n    # Return the sum of a and b.\n    return a - b\n"),
        "test_calculator.py": (
            "import unittest\n\n"
            "from calculator import add\n\n\n"
            "class CalculatorTests(unittest.TestCase):\n"
            "    def test_positive_addition(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n\n"
            "    def test_mixed_sign_addition(self):\n"
            "        self.assertEqual(add(-2, 3), 1)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
    },
    "expected_changed_files": ["calculator.py"],
    "reviewer_success_criteria": (
        "If the unittest suite passes, calculator.add performs arithmetic "
        "addition, and no test files were modified, the task is resolved."
    ),
}


def software_task_spec(config: dict[str, Any]) -> dict[str, Any]:
    task = config.get("software_task")
    if isinstance(task, dict) and isinstance(task.get("repo_files"), dict):
        return task
    return DEFAULT_SOFTWARE_TASK


def external_command_env(task_spec: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    repo_source = task_spec.get("repo_source", {})
    if isinstance(repo_source, dict) and repo_source.get("type") == "bugsinpy":
        root = Path(str(repo_source["root"])).expanduser().resolve()
        shim_dir = Path(".vacth_external_bin").resolve()
        shim_dir.mkdir(exist_ok=True)
        dos2unix = shim_dir / "dos2unix"
        if not dos2unix.exists():
            dos2unix.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            dos2unix.chmod(0o755)
        env["PATH"] = f"{shim_dir}:{root / 'framework' / 'bin'}:{env.get('PATH', '')}"
    return env


def repair_test_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    workspace_path = str(workspace.resolve())
    pythonpath_parts = [workspace_path]
    lib_path = workspace / "lib"
    if lib_path.is_dir():
        pythonpath_parts.insert(0, str(lib_path.resolve()))
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    if (workspace / "astropy").is_dir():
        env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.dev0")
        env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTROPY", "0.0.dev0")
    if (workspace / "src" / "_pytest").is_dir() or (workspace / "_pytest").is_dir():
        env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "8.0.0")
        env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST", "8.0.0")
    return env


def format_command_part(part: str, task_spec: dict[str, Any], workspace: Path) -> str:
    repo_source = task_spec.get("repo_source", {})
    bugsinpy_bin = ""
    if isinstance(repo_source, dict) and repo_source.get("type") == "bugsinpy":
        bugsinpy_bin = str(Path(str(repo_source["root"])).expanduser().resolve() / "framework" / "bin")
    replacements = {
        "{python}": sys.executable,
        "{workspace}": str(workspace.resolve()),
        "{bugsinpy_bin}": bugsinpy_bin,
    }
    for placeholder, value in replacements.items():
        part = part.replace(placeholder, value)
    return part


_DJANGO_UNMANAGED_TEST_RE = re.compile(r"^(\w+)\s+\(([\w.]+)\)$")


def django_test_label(label: str) -> str:
    """Convert SWE-bench's Django unittest label into runtests.py syntax."""
    match = _DJANGO_UNMANAGED_TEST_RE.match(label.strip())
    if match is None:
        return label.strip()
    method_name, class_path = match.groups()
    return f"{class_path}.{method_name}"


def is_swebench_task(task_spec: dict[str, Any]) -> bool:
    return str(task_spec.get("category", "")).startswith("swebench")


def swebench_repo(task_spec: dict[str, Any]) -> str:
    metadata = task_spec.get("benchmark_metadata", {})
    if isinstance(metadata, dict) and metadata.get("repo"):
        return str(metadata["repo"])
    repo_source = task_spec.get("repo_source", {})
    if isinstance(repo_source, dict) and repo_source.get("repo"):
        return str(repo_source["repo"])
    return ""


def swebench_expected_tests(task_spec: dict[str, Any], *, include_pass_to_pass: bool) -> list[str]:
    metadata = task_spec.get("benchmark_metadata", {})
    if not isinstance(metadata, dict):
        return []
    labels = [str(item) for item in metadata.get("fail_to_pass", [])]
    if include_pass_to_pass:
        labels.extend(str(item) for item in metadata.get("pass_to_pass", []))
    seen: set[str] = set()
    unique: list[str] = []
    for label in labels:
        if label and label not in seen:
            unique.append(label)
            seen.add(label)
    return unique


def formatted_command(
    raw_command: str | list[Any] | None, task_spec: dict[str, Any], workspace: Path
) -> list[str] | None:
    if raw_command is None:
        return None
    if isinstance(raw_command, str):
        return shlex.split(format_command_part(raw_command, task_spec, workspace))
    if isinstance(raw_command, list):
        return [format_command_part(str(part), task_spec, workspace) for part in raw_command]
    raise TypeError("software_task test command must be a string or list")


def ensure_pytest_verbose(command: list[str]) -> list[str]:
    if "-m" not in command:
        return command
    module_index = command.index("-m")
    if len(command) <= module_index + 1 or command[module_index + 1] != "pytest":
        return command
    if any(part in {"-v", "-vv", "--verbose"} for part in command):
        return command
    return command[: module_index + 2] + ["-vv"] + command[module_index + 2 :]


def command_test_paths(command: list[str] | None) -> list[str]:
    if command is None:
        return []
    return [
        part
        for part in command
        if (
            part.endswith(".py")
            or ".py::" in part
            or ("/" in part and not part.startswith("-") and not part.startswith("{"))
        )
    ]


def swebench_pytest_command(
    *,
    labels: list[str],
    fallback_command: list[str] | None,
) -> list[str] | None:
    base = [sys.executable, "-m", "pytest", "-vv"]
    if not labels:
        return ensure_pytest_verbose(fallback_command) if fallback_command else None
    if all(".py" in label or "/" in label for label in labels):
        return base + labels
    paths = command_test_paths(fallback_command)
    if paths:
        names = [label.split("[", 1)[0] for label in labels]
        return base + paths + ["-k", " or ".join(names)]
    return base + labels


def resolve_swebench_command(
    task_spec: dict[str, Any],
    workspace: Path,
    *,
    include_pass_to_pass: bool,
    fallback_command: list[str] | None,
) -> list[str] | None:
    labels = swebench_expected_tests(task_spec, include_pass_to_pass=include_pass_to_pass)
    if swebench_repo(task_spec) == "django/django" and labels:
        return [
            sys.executable,
            "tests/runtests.py",
            "--verbosity",
            "2",
            *[django_test_label(label) for label in labels],
        ]
    return swebench_pytest_command(labels=labels, fallback_command=fallback_command)


def resolve_test_command(task_spec: dict[str, Any], workspace: Path) -> list[str] | None:
    fallback = formatted_command(task_spec.get("test_command"), task_spec, workspace)
    if is_swebench_task(task_spec):
        resolved = resolve_swebench_command(
            task_spec,
            workspace,
            include_pass_to_pass=False,
            fallback_command=fallback,
        )
        return resolved or fallback
    return fallback


def resolve_full_test_command(task_spec: dict[str, Any], workspace: Path) -> list[str] | None:
    fallback = formatted_command(task_spec.get("full_test_command"), task_spec, workspace)
    if is_swebench_task(task_spec):
        resolved = resolve_swebench_command(
            task_spec,
            workspace,
            include_pass_to_pass=True,
            fallback_command=fallback,
        )
        return resolved or fallback
    return fallback


def run_setup_commands(*, task_spec: dict[str, Any], workspace: Path, setup_log: list[dict[str, Any]]) -> None:
    for raw_command in task_spec.get("setup_commands", []):
        if isinstance(raw_command, str):
            command = format_command_part(raw_command, task_spec, workspace)
            shell = True
            rendered_command: str | list[str] = command
        elif isinstance(raw_command, list):
            rendered_command = [format_command_part(str(part), task_spec, workspace) for part in raw_command]
            shell = False
        else:
            raise TypeError("software_task.setup_commands must contain strings or lists")
        completed = subprocess.run(
            rendered_command,
            check=False,
            capture_output=True,
            text=True,
            shell=shell,
            cwd=str(workspace),
            env=external_command_env(task_spec),
        )
        setup_log.append(
            {
                "command": rendered_command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Setup command failed with code {completed.returncode}: {rendered_command}")


def venv_environment(env_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = env_dir / ("Scripts" if (env_dir / "Scripts").is_dir() else "bin")
    env["VIRTUAL_ENV"] = str(env_dir)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    return env


def run_logged_command(
    *,
    command: str | list[str],
    cwd: Path,
    setup_log: list[dict[str, Any]],
    env: dict[str, str] | None = None,
    shell: bool = False,
    required: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        shell=shell,
    )
    setup_log.append(
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    if required and completed.returncode != 0:
        raise RuntimeError(f"Setup command failed with code {completed.returncode}: {command}")
    return completed


def run_full_test_suite(
    repair_tools: "SoftwareRepairTools",
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    """Run the FULL test suite (F2P+P2P) for final SWE-bench evaluation."""
    full_command = resolve_full_test_command(task_spec, repair_tools.workspace)
    if full_command is None:
        return json.loads(repair_tools.run_tests())

    return repair_tools._run_test_command(
        full_command,
        update_cache=False,
    )


def run_native_bugsinpy_compile(checkout_root: Path, setup_log: list[dict[str, Any]]) -> None:
    env_dir = checkout_root / "env"
    if env_dir.exists():
        shutil.rmtree(env_dir)
    run_logged_command(
        command=[sys.executable, "-m", "venv", str(env_dir)],
        cwd=checkout_root,
        setup_log=setup_log,
    )
    env_python = env_dir / "bin" / "python"
    if not env_python.exists():
        env_python = env_dir / "Scripts" / "python.exe"
    active_env = venv_environment(env_dir)
    run_logged_command(
        command=[
            str(env_python),
            "-m",
            "pip",
            "install",
            "-U",
            "pip",
            "setuptools",
            "wheel",
        ],
        cwd=checkout_root,
        setup_log=setup_log,
        env=active_env,
    )
    requirements = checkout_root / "bugsinpy_requirements.txt"
    if requirements.is_file() and requirements.read_text(encoding="utf-8").strip():
        run_logged_command(
            command=[str(env_python), "-m", "pip", "install", "-r", str(requirements)],
            cwd=checkout_root,
            setup_log=setup_log,
            env=active_env,
            required=False,
        )
    if (checkout_root / "setup.py").is_file() or (checkout_root / "pyproject.toml").is_file():
        install_tests = run_logged_command(
            command=[str(env_python), "-m", "pip", "install", "-e", ".[tests]"],
            cwd=checkout_root,
            setup_log=setup_log,
            env=active_env,
            required=False,
        )
        if install_tests.returncode != 0:
            run_logged_command(
                command=[str(env_python), "-m", "pip", "install", "-e", "."],
                cwd=checkout_root,
                setup_log=setup_log,
                env=active_env,
                required=False,
            )
    setup_script = checkout_root / "bugsinpy_setup.sh"
    if setup_script.is_file():
        for line in setup_script.read_text(encoding="utf-8", errors="replace").splitlines():
            command = line.strip()
            if not command or command.startswith("#"):
                continue
            run_logged_command(
                command=command,
                cwd=checkout_root,
                setup_log=setup_log,
                env=active_env,
                shell=True,
                required=False,
            )
    (checkout_root / "bugsinpy_compile_flag").write_text("1\n", encoding="utf-8")


def write_repo_file(repo: Path, relative_path: str, content: str) -> None:
    target = (repo / relative_path).resolve()
    try:
        target.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {relative_path}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def setup_bugsinpy_repo(repo: Path, task_spec: dict[str, Any]) -> Path:
    repo_source = task_spec["repo_source"]
    root = Path(str(repo_source["root"])).expanduser().resolve()
    project = str(repo_source["project"])
    bug_id = str(repo_source["bug_id"])
    version = str(repo_source.get("version", 0))
    checkout = root / "framework" / "bin" / "bugsinpy-checkout"
    completed = subprocess.run(
        [
            str(checkout),
            "-p",
            project,
            "-v",
            version,
            "-i",
            bug_id,
            "-w",
            str(repo.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=external_command_env(task_spec),
    )
    setup_log = [
        {
            "command": [str(checkout), "-p", project, "-v", version, "-i", bug_id],
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    ]
    if completed.returncode != 0:
        write_json("external_setup_log.json", setup_log)
        raise RuntimeError(f"BugsInPy checkout failed for {project}#{bug_id}")
    checkout_root = repo / project
    if not (checkout_root / "bugsinpy_bug.info").is_file():
        checkout_root = repo
    if bool(repo_source.get("compile", False)):
        compile_mode = str(repo_source.get("compile_mode", "native_venv"))
        if compile_mode == "native_venv":
            run_native_bugsinpy_compile(checkout_root, setup_log)
        else:
            compile_cmd = [
                str(root / "framework" / "bin" / "bugsinpy-compile"),
                "-w",
                str(checkout_root.resolve()),
            ]
            completed = subprocess.run(
                compile_cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=str(checkout_root),
                env=external_command_env(task_spec),
            )
            setup_log.append(
                {
                    "command": compile_cmd,
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                }
            )
            if completed.returncode != 0:
                write_json("external_setup_log.json", setup_log)
                raise RuntimeError(f"BugsInPy compile failed for {project}#{bug_id}")
    run_setup_commands(task_spec=task_spec, workspace=checkout_root, setup_log=setup_log)
    write_json("external_setup_log.json", setup_log)
    return checkout_root


def _log(setup_log: list[dict[str, Any]], step: str, detail: dict[str, Any]) -> None:
    setup_log.append({"step": step, **detail})


def _remove_macos_sidecars(root: Path) -> dict[str, int]:
    stats = {"files": 0, "dirs": 0, "errors": 0}
    if not root.exists():
        return stats
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir()) if current.is_dir() else []
        except OSError:
            stats["errors"] += 1
            continue
        for child in children:
            if child.name.startswith("._") or child.name == "__MACOSX":
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                        stats["dirs"] += 1
                    else:
                        child.unlink()
                        stats["files"] += 1
                except OSError:
                    stats["errors"] += 1
                continue
            if child.is_dir() and not child.is_symlink():
                stack.append(child)
    return stats


def _swebench_docker_image_name(repo: str, instance_id: str) -> str:
    org, repo_name = repo.split("/", 1)
    issue_number = instance_id.rsplit("-", 1)[-1]
    return f"sweb.eval.x86_64.{org}__{repo_name}-{issue_number}"


def _legacy_swebench_docker_image_name(repo: str, instance_id: str) -> str:
    org, repo_name = repo.split("/", 1)
    issue_number = instance_id.rsplit("-", 1)[-1]
    return f"sweb.eval.x86_64.{org}_1776_{repo_name}-{issue_number}"


def _append_docker_ref(candidates: list[str], image_ref: str) -> None:
    image_ref = image_ref.strip()
    if not image_ref:
        return
    refs = [image_ref]
    if ":" not in image_ref.rsplit("/", 1)[-1]:
        refs.append(f"{image_ref}:latest")
    for ref in refs:
        if ref not in candidates:
            candidates.append(ref)


def _swebench_docker_image_refs(*, repo_name: str, instance_id: str, docker_image: str) -> list[str]:
    candidates: list[str] = []
    if docker_image:
        _append_docker_ref(candidates, docker_image)
        if "/" not in docker_image:
            _append_docker_ref(candidates, f"swebench/{docker_image}")
    if instance_id:
        official = _swebench_docker_image_name(repo_name, instance_id)
        legacy = _legacy_swebench_docker_image_name(repo_name, instance_id)
        for image in (official, legacy):
            _append_docker_ref(candidates, image)
            _append_docker_ref(candidates, f"swebench/{image}")
    return candidates


def _docker_image_exists(image_ref: str) -> tuple[bool, subprocess.CompletedProcess[str]]:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image_ref],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        completed = subprocess.CompletedProcess(
            args=["docker", "image", "inspect", image_ref],
            returncode=127,
            stdout="",
            stderr="docker executable not found",
        )
    return completed.returncode == 0, completed


def _docker_image_id_from_listing(image_ref: str) -> str:
    try:
        completed = subprocess.run(
            ["docker", "image", "ls", "--format", "{{.Repository}}\t{{.Tag}}\t{{.ID}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    wanted = image_ref.strip()
    wanted_no_tag = wanted
    wanted_tag = "latest"
    if ":" in wanted.rsplit("/", 1)[-1]:
        wanted_no_tag, wanted_tag = wanted.rsplit(":", 1)
    for line in completed.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        repo_name, tag, image_id = parts
        if tag == "<none>":
            candidates = {repo_name}
        else:
            candidates = {f"{repo_name}:{tag}"}
            if tag == "latest":
                candidates.add(repo_name)
        if wanted in candidates or (wanted_no_tag == repo_name and wanted_tag == tag):
            return image_id
    return ""


def _setup_swebench_from_docker(
    repo: Path,
    *,
    repo_name: str,
    instance_id: str,
    docker_image: str,
    setup_log: list[dict[str, Any]],
) -> bool:
    image_refs = _swebench_docker_image_refs(
        repo_name=repo_name,
        instance_id=instance_id,
        docker_image=docker_image,
    )
    pull_timeout_sec = int(os.environ.get("VACTHBENCH_DOCKER_PULL_TIMEOUT_SEC", "900"))
    setup_timeout_sec = int(os.environ.get("VACTHBENCH_DOCKER_SETUP_TIMEOUT_SEC", "300"))
    image_ref = ""
    for candidate in image_refs:
        exists, completed = _docker_image_exists(candidate)
        resolved_ref = candidate
        if not exists:
            image_id = _docker_image_id_from_listing(candidate)
            if image_id:
                exists = True
                resolved_ref = image_id
                completed = subprocess.CompletedProcess(
                    args=["docker", "image", "ls", candidate],
                    returncode=0,
                    stdout=image_id,
                    stderr="resolved from docker image ls",
                )
        _log(
            setup_log,
            "docker_image_inspect",
            {
                "image": candidate,
                "resolved_image": resolved_ref if exists else "",
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            },
        )
        if exists:
            image_ref = resolved_ref
            break

    pull_enabled = os.environ.get("VACTHBENCH_DOCKER_PULL", "0").lower() in {"1", "true", "yes", "on"}
    if not image_ref and pull_enabled:
        pull_candidates = [candidate for candidate in image_refs if candidate.startswith("swebench/")]
        if not pull_candidates and image_refs:
            pull_candidates = [image_refs[0]]
        for pull_ref in pull_candidates:
            try:
                completed = subprocess.run(
                    ["docker", "pull", pull_ref],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=pull_timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                _log(
                    setup_log,
                    "docker_pull",
                    {
                        "image": pull_ref,
                        "returncode": -1,
                        "stdout_tail": output_tail(exc.stdout, 2000),
                        "stderr_tail": output_tail(exc.stderr, 2000),
                        "timeout_sec": pull_timeout_sec,
                    },
                )
                continue
            _log(
                setup_log,
                "docker_pull",
                {
                    "image": pull_ref,
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-2000:],
                },
            )
            if completed.returncode == 0:
                image_ref = pull_ref
                break

    if not image_ref:
        _log(
            setup_log,
            "docker_image_missing",
            {
                "candidates": image_refs,
                "pull_enabled": pull_enabled,
            },
        )
        return False

    try:
        completed = subprocess.run(
            ["docker", "create", image_ref],
            check=False,
            capture_output=True,
            text=True,
            timeout=setup_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        _log(
            setup_log,
            "docker_create",
            {
                "returncode": -1,
                "stdout_tail": output_tail(exc.stdout, 2000),
                "stderr_tail": output_tail(exc.stderr, 2000),
                "timeout_sec": setup_timeout_sec,
            },
        )
        return False
    except FileNotFoundError:
        _log(setup_log, "docker_unavailable", {"image": image_ref})
        return False
    if completed.returncode != 0:
        _log(
            setup_log,
            "docker_create",
            {
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            },
        )
        return False
    container_id = completed.stdout.strip()

    # Copy testbed contents into the repo directory
    repo.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            ["docker", "cp", f"{container_id}:/testbed/.", str(repo.resolve())],
            check=False,
            capture_output=True,
            text=True,
            timeout=setup_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            args=["docker", "cp", f"{container_id}:/testbed/.", str(repo.resolve())],
            returncode=-1,
            stdout=output_tail(exc.stdout, 2000),
            stderr=output_tail(exc.stderr, 2000) + f"\nTIMEOUT after {setup_timeout_sec}s",
        )
    _log(
        setup_log,
        "docker_cp",
        {
            "image": image_ref,
            "container_id": container_id[:12],
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
        },
    )

    # Cleanup container
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run(
            ["docker", "rm", container_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=setup_timeout_sec,
        )

    if completed.returncode == 0:
        _finalize_swebench_workspace(repo, setup_log, source="docker")

    return completed.returncode == 0


def _write_docker_compat_shim(repo: Path, setup_log: list[dict[str, Any]]) -> None:
    sitecustomize = repo / "sitecustomize.py"
    compat_code = (
        "import collections\n"
        "import collections.abc\n"
        "for _name in ('MutableMapping', 'Mapping', 'Sequence', "
        "'Iterable', 'Container', 'Sized', 'Callable', 'Set'):\n"
        "    if not hasattr(collections, _name):\n"
        "        setattr(collections, _name, getattr(collections.abc, _name))\n"
    )
    sitecustomize.write_text(compat_code, encoding="utf-8")
    _log(setup_log, "write_sitecustomize", {"path": str(sitecustomize)})


def _finalize_swebench_workspace(repo: Path, setup_log: list[dict[str, Any]], *, source: str) -> None:
    _log(setup_log, f"{source}_macos_sidecar_cleanup", _remove_macos_sidecars(repo))
    if (repo / ".git").is_dir():
        config_filemode = subprocess.run(
            ["git", "-C", str(repo.resolve()), "config", "core.fileMode", "false"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        _log(
            setup_log,
            f"{source}_git_config_filemode",
            {
                "returncode": config_filemode.returncode,
                "stderr_tail": config_filemode.stderr[-2000:],
            },
        )
    _write_docker_compat_shim(repo, setup_log)


def _results_root_from_cwd() -> Path | None:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if candidate.name == "Results":
            return candidate
    return None


def _setup_swebench_from_local_cache(
    repo: Path,
    *,
    instance_id: str,
    base_commit: str,
    setup_log: list[dict[str, Any]],
) -> bool:
    use_cache = os.environ.get("VACTHBENCH_SWEBENCH_USE_LOCAL_CACHE", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if not use_cache or not instance_id:
        return False
    results_root = _results_root_from_cwd()
    if results_root is None:
        return False
    patterns = [
        f"*/*{instance_id}*/0/workspace",
        f"*/*/*{instance_id}*/0/workspace",
    ]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in patterns:
        for candidate in results_root.glob(pattern):
            resolved = candidate.resolve()
            if resolved == repo.resolve() or resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(candidate)
    _log(
        setup_log,
        "local_cache_candidates",
        {"instance_id": instance_id, "count": len(candidates)},
    )
    for candidate in candidates:
        if not (candidate / ".git").is_dir():
            continue
        _log(
            setup_log,
            "local_cache_candidate_sidecar_cleanup",
            {"candidate": str(candidate), **_remove_macos_sidecars(candidate / ".git" / "objects" / "pack")},
        )
        rev_parse = subprocess.run(
            ["git", "-C", str(candidate.resolve()), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        _log(
            setup_log,
            "local_cache_rev_parse",
            {
                "candidate": str(candidate),
                "returncode": rev_parse.returncode,
                "stdout_tail": rev_parse.stdout[-2000:],
                "stderr_tail": rev_parse.stderr[-2000:],
            },
        )
        if rev_parse.returncode != 0 or rev_parse.stdout.strip() != base_commit:
            continue
        shutil.rmtree(repo, ignore_errors=True)
        clone = subprocess.run(
            [
                "git",
                "clone",
                "--no-hardlinks",
                str(candidate.resolve()),
                str(repo.resolve()),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        _log(
            setup_log,
            "local_cache_clone",
            {
                "candidate": str(candidate),
                "returncode": clone.returncode,
                "stdout_tail": clone.stdout[-2000:],
                "stderr_tail": clone.stderr[-2000:],
            },
        )
        if clone.returncode != 0:
            continue
        checkout = subprocess.run(
            ["git", "-C", str(repo.resolve()), "checkout", base_commit],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        _log(
            setup_log,
            "local_cache_checkout",
            {
                "returncode": checkout.returncode,
                "stdout_tail": checkout.stdout[-2000:],
                "stderr_tail": checkout.stderr[-2000:],
            },
        )
        if checkout.returncode == 0:
            _finalize_swebench_workspace(repo, setup_log, source="local_cache")
            return True
    return False


def setup_swebench_repo(repo: Path, task_spec: dict[str, Any]) -> Path:
    repo_source = task_spec["repo_source"]
    github_repo = str(repo_source["repo"])
    base_commit = str(repo_source["base_commit"])
    instance_id = str(repo_source.get("instance_id", ""))
    docker_image = str(repo_source.get("docker_image") or "")
    if not docker_image and instance_id:
        docker_image = _swebench_docker_image_name(github_repo, instance_id)
    git_timeout_sec = int(
        repo_source.get(
            "git_timeout_sec",
            os.environ.get("VACTHBENCH_SWEBENCH_GIT_TIMEOUT_SEC", "600"),
        )
    )
    remote_url = os.environ.get(
        "VACTHBENCH_SWEBENCH_REMOTE_URL",
        f"https://github.com/{github_repo}.git",
    )
    setup_log: list[dict[str, Any]] = []
    prefer_git = os.environ.get("VACTHBENCH_SWEBENCH_PREFER_GIT", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if prefer_git:
        _log(
            setup_log,
            "git_preferred",
            {"status": "using_git_checkout_without_docker_fallback"},
        )
    used_docker = False
    used_local_cache = _setup_swebench_from_local_cache(
        repo,
        instance_id=instance_id,
        base_commit=base_commit,
        setup_log=setup_log,
    )
    if docker_image and not prefer_git and not used_local_cache:
        _log(
            setup_log,
            "docker_preferred",
            {"docker_image": docker_image, "status": "trying_docker_first"},
        )
        used_docker = _setup_swebench_from_docker(
            repo,
            repo_name=github_repo,
            instance_id=instance_id,
            docker_image=docker_image,
            setup_log=setup_log,
        )
        if not used_docker:
            _log(setup_log, "docker_failed", {"status": "falling_back_to_git"})
            shutil.rmtree(repo, ignore_errors=True)
            repo.mkdir(exist_ok=True)

    git_failed = False
    if not used_docker and not used_local_cache:
        commands = [
            ["git", "init", str(repo.resolve())],
            ["git", "-C", str(repo.resolve()), "remote", "add", "origin", remote_url],
            [
                "git",
                "-C",
                str(repo.resolve()),
                "fetch",
                "--filter=blob:none",
                "--no-tags",
                "--depth",
                "1",
                "origin",
                base_commit,
            ],
            ["git", "-C", str(repo.resolve()), "checkout", "FETCH_HEAD"],
        ]
        for command in commands:
            attempts = int(os.environ.get("VACTHBENCH_GIT_FETCH_RETRIES", "3")) if "fetch" in command else 1
            completed: subprocess.CompletedProcess[str] | None = None
            for attempt in range(1, max(1, attempts) + 1):
                try:
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=git_timeout_sec,
                    )
                except subprocess.TimeoutExpired as exc:
                    setup_log.append(
                        {
                            "command": command,
                            "attempt": attempt,
                            "returncode": -1,
                            "stdout_tail": output_tail(exc.stdout, 4000),
                            "stderr_tail": output_tail(exc.stderr, 4000) + f"\nTIMEOUT after {git_timeout_sec}s",
                        }
                    )
                    completed = None
                else:
                    setup_log.append(
                        {
                            "command": command,
                            "attempt": attempt,
                            "returncode": completed.returncode,
                            "stdout_tail": completed.stdout[-4000:],
                            "stderr_tail": completed.stderr[-4000:],
                        }
                    )
                    if completed.returncode == 0:
                        break
                if attempt < attempts:
                    time.sleep(min(10, 2 * attempt))
            if completed is None or completed.returncode != 0:
                git_failed = True
                break

    if not git_failed and not used_docker and not used_local_cache:
        _finalize_swebench_workspace(repo, setup_log, source="git")

    if git_failed and docker_image and not used_docker and not prefer_git:
        _log(
            setup_log,
            "git_fetch_failed",
            {
                "docker_image": docker_image,
                "status": "falling_back_to_docker",
            },
        )
        git_failed = not _setup_swebench_from_docker(
            repo,
            repo_name=github_repo,
            instance_id=instance_id,
            docker_image=docker_image,
            setup_log=setup_log,
        )

    if git_failed:
        write_json("external_setup_log.json", setup_log)
        raise RuntimeError(f"SWE-bench checkout failed for {github_repo}")
    test_patch = repo_source.get("test_patch", "")
    if test_patch and bool(repo_source.get("apply_test_patch", True)):
        patch_file = Path("swebench_test_patch.diff").resolve()
        patch_file.write_text(str(test_patch), encoding="utf-8")
        command = ["git", "-C", str(repo.resolve()), "apply", str(patch_file)]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        setup_log.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"SWE-bench test_patch apply failed for {github_repo}")
    try:
        run_setup_commands(task_spec=task_spec, workspace=repo, setup_log=setup_log)
    except Exception as exc:
        _log(
            setup_log,
            "setup_commands_failed",
            {
                "error": str(exc)[-2000:],
                "tolerated": False,
            },
        )
        write_json("external_setup_log.json", setup_log)
        raise
    write_json("external_setup_log.json", setup_log)
    return repo


def setup_repo(config: dict[str, Any] | None = None) -> Path:
    task_spec = software_task_spec(config or {})
    repo = Path("workspace")
    if repo.exists():
        shutil.rmtree(repo, ignore_errors=True)
    repo.mkdir(exist_ok=True)
    repo_source = task_spec.get("repo_source", {})
    if isinstance(repo_source, dict) and repo_source.get("type") == "bugsinpy":
        return setup_bugsinpy_repo(repo, task_spec)
    if isinstance(repo_source, dict) and repo_source.get("type") == "swebench":
        return setup_swebench_repo(repo, task_spec)
    for relative_path, content in sorted(task_spec["repo_files"].items()):
        write_repo_file(repo, str(relative_path), str(content))
    return repo


def make_repair_tools(config: dict[str, Any], repo: Path) -> SoftwareRepairTools:
    task_spec = software_task_spec(config)
    return SoftwareRepairTools(
        repo,
        test_command=resolve_test_command(task_spec, repo),
        task_spec=task_spec,
    )


def initialize_repair_workspace(
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path, SoftwareRepairTools]:
    task_spec = software_task_spec(config)
    config["software_task_id"] = str(task_spec.get("id", config["task_id"]))
    config["software_task_category"] = str(task_spec.get("category", "unspecified"))
    repo = setup_repo(config)
    return task_spec, repo, make_repair_tools(config, repo)


def is_test_path(relative_path: str) -> bool:
    path = Path(relative_path)
    return path.name.startswith("test_") or "tests" in path.parts


# ---- SWE-bench style evaluation utilities -----------------------------------

_PYTEST_RESULT_RE = re.compile(
    r"^\s*(.+?::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b",
    re.MULTILINE,
)
_UNITTEST_RESULT_RE = re.compile(
    r"^\s*(\w+)\s+\(([\w.]+)\)\s+\.\.\.\s+"
    r"(ok|FAIL|ERROR|skipped|expected failure|unexpected success)\b",
    re.MULTILINE | re.IGNORECASE,
)
_INFRA_ERROR_RE = re.compile(
    r"(ModuleNotFoundError|ImportError|ImproperlyConfigured|No module named|"
    r"file or directory not found|not found:|settings are not configured)",
    re.IGNORECASE,
)


def parse_pytest_log(stdout: str) -> dict[str, str]:
    """Parse pytest output into a mapping of test_id -> status (pass/fail/skip/error)."""
    results: dict[str, str] = {}
    for match in _PYTEST_RESULT_RE.finditer(stdout):
        test_id, status = match.group(1), match.group(2)
        results[test_id] = status.lower()
    return results


def parse_unittest_log(output: str) -> dict[str, str]:
    """Parse unittest/Django runtests output into SWE-bench-compatible ids."""
    results: dict[str, str] = {}
    for match in _UNITTEST_RESULT_RE.finditer(output):
        method_name, class_path, raw_status = match.groups()
        status = raw_status.lower()
        normalized = "passed" if status == "ok" else "failed"
        if status == "error":
            normalized = "error"
        elif status == "skipped":
            normalized = "skipped"
        original_label = f"{method_name} ({class_path})"
        dotted_label = f"{class_path}.{method_name}"
        results[original_label] = normalized
        results[dotted_label] = normalized
    return results


def parse_test_log(output: str) -> dict[str, str]:
    results = parse_pytest_log(output)
    results.update(parse_unittest_log(output))
    return results


def candidate_test_ids(test_id: str) -> set[str]:
    candidates = {test_id}
    django_label = django_test_label(test_id)
    candidates.add(django_label)
    if "::" in test_id:
        path, _, suffix = test_id.partition("::")
        candidates.add(suffix)
        candidates.add(suffix.replace("::", "."))
        candidates.add(Path(path).stem + "::" + suffix)
    return {candidate for candidate in candidates if candidate}


def test_status(test_results: dict[str, str], expected_id: str) -> str | None:
    for candidate in candidate_test_ids(expected_id):
        if candidate in test_results:
            return test_results[candidate]
    return None


def classify_swebench_outcome(
    f2p_pass: int,
    f2p_total: int,
    p2p_pass: int,
    p2p_total: int,
) -> str:
    """Classify a patch result into one of 6 SWE-bench outcome categories.

    See Table 22 in the SWE-bench paper:
      - resolved:            all F2P pass, all P2P pass
      - breaking_resolved:   all F2P pass, some P2P fail
      - partially_resolved:  some F2P pass, all P2P pass
      - work_in_progress:    some F2P pass, some P2P fail
      - no_op:               no F2P pass, all P2P pass
      - regression:          no F2P pass, some P2P fail
    """
    if f2p_total == 0:
        return "resolved" if p2p_pass == p2p_total else "error"

    all_f2p = f2p_pass == f2p_total
    some_f2p = f2p_pass > 0
    none_f2p = f2p_pass == 0

    all_p2p = p2p_pass == p2p_total
    some_p2p = p2p_pass > 0

    if all_f2p and all_p2p:
        return "resolved"
    if all_f2p and not all_p2p:
        return "breaking_resolved"
    if some_f2p and all_p2p:
        return "partially_resolved"
    if some_f2p and some_p2p:
        return "work_in_progress"
    if none_f2p and all_p2p:
        return "no_op"
    if none_f2p and not all_p2p:
        return "regression"
    return "unknown"


def _evaluate_swebench_style(
    test_output: dict[str, Any],
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate test output against SWE-bench FAIL_TO_PASS / PASS_TO_PASS ground truth."""
    benchmark_meta = task_spec.get("benchmark_metadata", {})
    expected_f2p = set(str(t) for t in benchmark_meta.get("fail_to_pass", []))
    expected_p2p = set(str(t) for t in benchmark_meta.get("pass_to_pass", []))

    if not expected_f2p and not expected_p2p:
        passed = bool(test_output.get("passed"))
        return {
            "resolved": passed,
            "f2p_pass": 0,
            "f2p_total": 0,
            "p2p_pass": 0,
            "p2p_total": 0,
            "outcome": "resolved" if passed else "unresolved",
            "missing_f2p": [],
            "missing_p2p": [],
        }

    output = f"{test_output.get('stdout', '')}\n{test_output.get('stderr', '')}"
    test_results = parse_test_log(output)

    if not test_results:
        if bool(test_output.get("passed")):
            f2p_total = len(expected_f2p)
            p2p_total = len(expected_p2p)
            return {
                "resolved": True,
                "f2p_pass": f2p_total,
                "f2p_total": f2p_total,
                "p2p_pass": p2p_total,
                "p2p_total": p2p_total,
                "outcome": "resolved",
                "missing_f2p": [],
                "missing_p2p": [],
            }
        if _INFRA_ERROR_RE.search(output):
            return {
                "resolved": False,
                "f2p_pass": 0,
                "f2p_total": len(expected_f2p),
                "p2p_pass": 0,
                "p2p_total": len(expected_p2p),
                "outcome": "error",
                "missing_f2p": sorted(expected_f2p),
                "missing_p2p": sorted(expected_p2p),
            }

    f2p_total = len(expected_f2p)
    f2p_pass = sum(1 for t in expected_f2p if test_status(test_results, t) == "passed")
    missing_f2p = sorted(t for t in expected_f2p if test_status(test_results, t) is None)

    p2p_total = len(expected_p2p)
    p2p_pass = sum(1 for t in expected_p2p if test_status(test_results, t) == "passed")
    missing_p2p = sorted(t for t in expected_p2p if test_status(test_results, t) is None)

    if bool(test_output.get("passed")) and (missing_f2p or missing_p2p):
        f2p_pass = f2p_total
        p2p_pass = p2p_total
        missing_f2p = []
        missing_p2p = []

    outcome = classify_swebench_outcome(f2p_pass, f2p_total, p2p_pass, p2p_total)

    return {
        "resolved": outcome == "resolved",
        "f2p_pass": f2p_pass,
        "f2p_total": f2p_total,
        "p2p_pass": p2p_pass,
        "p2p_total": p2p_total,
        "outcome": outcome,
        "missing_f2p": missing_f2p,
        "missing_p2p": missing_p2p,
    }


def evaluate_repair_result(
    *,
    logger: ExperimentLogger,
    repair_tools: SoftwareRepairTools,
    task_spec: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    # Run the FULL test suite (F2P+P2P) for SWE-bench tasks;
    # fall back to the agent's test_command for non-SWE-bench tasks.
    full_test_output = run_full_test_suite(repair_tools, task_spec)

    changed_files = repair_tools.changed_files()
    changed_test_files = [path for path in changed_files if is_test_path(path)]
    expected_changed_files = [str(path) for path in task_spec.get("expected_changed_files", [])]
    missing_expected_files = sorted(set(expected_changed_files) - set(changed_files))
    final_decision = final_answer_from_messages(logger.raw_messages).lower()
    reviewer_resolved = final_decision.startswith("resolved")
    forbid_test_changes = bool(task_spec.get("forbid_test_changes", True))
    require_expected_changes = bool(expected_changed_files and task_spec.get("require_expected_changes", True))
    constraint_violations = 0
    if forbid_test_changes:
        constraint_violations += len(changed_test_files)
    if require_expected_changes:
        constraint_violations += len(missing_expected_files)

    # SWE-bench style evaluation: parse individual test results
    is_swebench = str(task_spec.get("category", "")).startswith("swebench")
    swebench_eval = _evaluate_swebench_style(full_test_output, task_spec)

    # Determine success
    if is_swebench:
        success = swebench_eval["resolved"]
    else:
        success = bool(full_test_output.get("passed")) and reviewer_resolved and constraint_violations == 0

    verification = full_test_output
    verification.update(
        {
            "changed_files": changed_files,
            "changed_file_count": len(changed_files),
            "changed_test_files": changed_test_files,
            "changed_test_file_count": len(changed_test_files),
            "expected_changed_files": expected_changed_files,
            "expected_changed_file_count": len(expected_changed_files),
            "missing_expected_files": missing_expected_files,
            "missing_expected_file_count": len(missing_expected_files),
            "reviewer_final_decision": final_decision,
            "reviewer_resolved": reviewer_resolved,
            "constraint_violations": constraint_violations,
            "invalid_patch_count": 0 if full_test_output.get("passed") else 1,
            # SWE-bench metrics
            "swebench_resolved": swebench_eval["resolved"],
            "swebench_outcome": swebench_eval["outcome"],
            "swebench_f2p_pass": swebench_eval["f2p_pass"],
            "swebench_f2p_total": swebench_eval["f2p_total"],
            "swebench_p2p_pass": swebench_eval["p2p_pass"],
            "swebench_p2p_total": swebench_eval["p2p_total"],
            "swebench_missing_f2p": swebench_eval["missing_f2p"],
            "swebench_missing_p2p": swebench_eval["missing_p2p"],
        }
    )
    return verification, success


def final_answer_from_messages(raw_messages: list[dict[str, Any]]) -> str:
    for message in reversed(raw_messages):
        if message.get("source") != "reviewer":
            continue
        content = str(message.get("content", ""))
        if "FINAL_ANSWER:" in content:
            return content.split("FINAL_ANSWER:", 1)[1].strip().splitlines()[0]
    return "unresolved"


def metric_summary(
    *,
    logger: ExperimentLogger,
    start: float,
    success: bool,
    verification: dict[str, Any],
) -> dict[str, Any]:
    total_model_tokens = sum(item["total_tokens"] for item in logger.model_usage)
    agent_model_tokens = sum(
        item["total_tokens"]
        for item in logger.model_usage
        if item.get("agent") in {"planner", "retriever", "coder", "tester", "reviewer"}
    )
    extractor_model_tokens = sum(
        item["total_tokens"] for item in logger.model_usage if item.get("agent") == "vacth_extractor"
    )
    aux_model_tokens = total_model_tokens - agent_model_tokens
    raw_message_model_tokens = sum(
        (message.get("models_usage") or {}).get("prompt_tokens", 0)
        + (message.get("models_usage") or {}).get("completion_tokens", 0)
        for message in logger.raw_messages
    )
    fallback_tokens = sum(message.get("estimated_tokens", 0) for message in logger.raw_messages)
    summary_tokens = sum(summary.get("estimated_tokens", 0) for summary in logger.summaries)
    summary_source_tokens = sum(summary.get("source_estimated_tokens", 0) for summary in logger.summaries)
    summary_compression_ratio = round(summary_tokens / summary_source_tokens, 4) if summary_source_tokens else 0
    structured_tokens = sum(state.get("estimated_tokens", 0) for state in logger.structured_states)
    structured_source_tokens = sum(state.get("source_estimated_tokens", 0) for state in logger.structured_states)
    structured_state_item_count = 0
    if logger.structured_states:
        latest_state = logger.structured_states[-1].get("state", {})
        if isinstance(latest_state, dict):
            structured_state_item_count = sum(len(items) for items in latest_state.values() if isinstance(items, list))
    structured_parse_error_count = len(
        [state for state in logger.structured_states if not state.get("schema_valid", False)]
    )
    dropped_message_count = sum(len(context.get("dropped_messages", [])) for context in logger.visible_contexts)
    max_visible_context_messages = max(
        (len(context.get("context", [])) for context in logger.visible_contexts),
        default=0,
    )
    retrieval_records = [
        decision for decision in logger.routing_decisions if decision.get("retrieval_method") == "lexical_cosine"
    ]
    retrieval_scores = [score["score"] for decision in retrieval_records for score in decision.get("scores", [])]
    retrieved_tokens = sum(int(decision.get("estimated_tokens", 0)) for decision in retrieval_records)
    cve_records = [
        decision for decision in logger.routing_decisions if str(decision.get("routing_method", "")).startswith("cve")
    ]
    capsule_tokens = [int(capsule.get("budget", {}).get("used_tokens", 0)) for capsule in logger.capsules]
    selected_capsule_items = sum(len(capsule.get("selected_items", [])) for capsule in logger.capsules)
    vacth_parse_error_count = len(
        [extraction for extraction in logger.vacth_extractions if not extraction.get("parse_valid", False)]
    )
    vacth_heuristic_extraction_count = len(
        [
            extraction
            for extraction in logger.vacth_extractions
            if "heuristic" in str(extraction.get("extraction_mode", ""))
            or "deterministic_tool_events" in str(extraction.get("raw_update", ""))
        ]
    )
    vacth_reask_count = sum(int(extraction.get("reask_attempts", 0)) for extraction in logger.vacth_extractions)
    return {
        "success": success,
        "software_task_id": logger.config.get("software_task_id", ""),
        "software_task_category": logger.config.get("software_task_category", ""),
        "turns": len([message for message in logger.raw_messages if message.get("role") == "assistant"]),
        "tool_calls": len(logger.tool_calls),
        "estimated_tokens": total_model_tokens or fallback_tokens,
        "corrected_total_tokens": total_model_tokens or fallback_tokens,
        "agent_model_tokens": agent_model_tokens,
        "aux_model_tokens": aux_model_tokens,
        "extractor_model_tokens": extractor_model_tokens,
        "raw_message_model_tokens": raw_message_model_tokens,
        "wall_time_sec": round(time.time() - start, 4),
        "constraint_violations": int(verification.get("constraint_violations", 0)),
        "invalid_patch_count": int(verification.get("invalid_patch_count", 0)),
        "changed_file_count": int(verification.get("changed_file_count", 0)),
        "changed_test_file_count": int(verification.get("changed_test_file_count", 0)),
        "expected_changed_file_count": int(verification.get("expected_changed_file_count", 0)),
        "missing_expected_file_count": int(verification.get("missing_expected_file_count", 0)),
        "reviewer_resolved": bool(verification.get("reviewer_resolved", False)),
        "raw_messages": len(logger.raw_messages),
        "visible_contexts": len(logger.visible_contexts),
        "state_items": len(logger.state_items),
        "routing_decisions": len(logger.routing_decisions),
        "provenance_edges": len(logger.provenance_edges),
        "verification_returncode": verification.get("returncode", -1),
        "summary_count": len(logger.summaries),
        "summary_tokens": summary_tokens,
        "summary_source_tokens": summary_source_tokens,
        "summary_compression_ratio": summary_compression_ratio,
        "structured_update_count": len(logger.structured_states),
        "structured_state_item_count": structured_state_item_count,
        "structured_tokens": structured_tokens,
        "structured_source_tokens": structured_source_tokens,
        "structured_parse_error_count": structured_parse_error_count,
        "structured_schema_valid": structured_parse_error_count == 0,
        "context_window_messages": config_value_int(logger.config, "context_window_messages"),
        "dropped_message_count": dropped_message_count,
        "max_visible_context_messages": max_visible_context_messages,
        "memory_item_count": len(logger.memory_items),
        "retrieval_count": len(retrieval_records),
        "retrieval_top_k": config_value_int(logger.config, "memory_top_k"),
        "retrieved_tokens": retrieved_tokens,
        "avg_retrieval_score": round(sum(retrieval_scores) / len(retrieval_scores), 4) if retrieval_scores else 0,
        "vacth_extraction_count": len(logger.vacth_extractions),
        "vacth_parse_error_count": vacth_parse_error_count,
        "vacth_heuristic_extraction_count": vacth_heuristic_extraction_count,
        "vacth_reask_count": vacth_reask_count,
        "vacth_capsule_count": len(logger.capsules),
        "vacth_selected_item_count": selected_capsule_items,
        "vacth_cve_routing_count": len(cve_records),
        "vacth_avg_capsule_tokens": round(sum(capsule_tokens) / len(capsule_tokens), 4) if capsule_tokens else 0,
        "vacth_enable_thc": bool(logger.config.get("enable_thc", False)),
        "vacth_enable_cve": bool(logger.config.get("enable_cve", False)),
        "vacth_enable_paa": bool(logger.config.get("enable_paa", False)),
        "vacth_enable_provenance": bool(logger.config.get("enable_provenance", False)),
        "vacth_enable_reask": bool(logger.config.get("enable_reask", False)),
        "vacth_role_specific_routing": bool(logger.config.get("role_specific_routing", False)),
        "vacth_skipped_extraction_agent_count": len(logger.config.get("vacth_skip_extraction_agents", []))
        if isinstance(logger.config.get("vacth_skip_extraction_agents", []), list)
        else 0,
        "vacth_compact_downstream_task": bool(logger.config.get("vacth_compact_downstream_task", False)),
        "vacth_compact_capsule": bool(logger.config.get("vacth_compact_capsule", False)),
        "vacth_hybrid_extraction": bool(logger.config.get("vacth_hybrid_extraction", False)),
        "vacth_adaptive_repair_enabled": bool(logger.config.get("vacth_adaptive_repair_enabled", False)),
        "vacth_adaptive_repair_attempts": int(logger.config.get("vacth_adaptive_repair_attempts", 0)),
        "vacth_adaptive_repair_max_attempts": int(logger.config.get("vacth_adaptive_repair_max_attempts", 0)),
        "vacth_adaptive_repair_progress_gate": bool(logger.config.get("vacth_adaptive_repair_progress_gate", False)),
        "vacth_adaptive_previous_failure_score": int(logger.config.get("vacth_adaptive_previous_failure_score", -1)),
        "vacth_adaptive_current_failure_score": int(logger.config.get("vacth_adaptive_current_failure_score", -1)),
        "vacth_source_patch_guard_attempts": int(logger.config.get("vacth_source_patch_guard_attempts", 0)),
        "vacth_source_patch_guard_max_attempts": int(
            logger.config.get("vacth_source_patch_guard_max_attempts", 0)
        ),
        "vacth_source_patch_guard_exhausted": bool(
            logger.config.get("vacth_source_patch_guard_exhausted", False)
        ),
        # SWE-bench evaluation metrics
        "swebench_resolved": bool(verification.get("swebench_resolved", False)),
        "swebench_outcome": str(verification.get("swebench_outcome", "")),
        "swebench_f2p_pass": int(verification.get("swebench_f2p_pass", 0)),
        "swebench_f2p_total": int(verification.get("swebench_f2p_total", 0)),
        "swebench_p2p_pass": int(verification.get("swebench_p2p_pass", 0)),
        "swebench_p2p_total": int(verification.get("swebench_p2p_total", 0)),
    }


def config_value_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def config_value_str_set(config: dict[str, Any], key: str, default: set[str] | None = None) -> set[str]:
    value = config.get(key)
    if value is None:
        return set(default or set())
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return {str(item).strip() for item in value if str(item).strip()}
    return set(default or set())


def config_value_int_map(config: dict[str, Any], key: str, default: dict[str, int] | None = None) -> dict[str, int]:
    value = config.get(key)
    if value is None:
        return dict(default or {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return dict(default or {})
    if not isinstance(value, Mapping):
        return dict(default or {})
    parsed: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        try:
            parsed[str(raw_key)] = int(raw_value)
        except (TypeError, ValueError):
            continue
    return parsed


def config_value_bool(config: dict[str, Any], key: str, default: bool = True) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def local_validation_early_stop_enabled(config: dict[str, Any]) -> bool:
    return config_value_bool(config, "enable_local_validation_early_stop", True)


def adaptive_repair_enabled(config: dict[str, Any]) -> bool:
    return config_value_bool(
        config,
        "vacth_adaptive_repair_enabled",
        config.get("method") == "vacth_optimized",
    )


def adaptive_repair_max_attempts(config: dict[str, Any]) -> int:
    value = config.get("vacth_adaptive_repair_max_attempts")
    try:
        if value is None:
            raise TypeError
        return max(0, int(value))
    except (TypeError, ValueError):
        return 2 if config.get("method") == "vacth_optimized" else 0


def adaptive_repair_progress_gate_enabled(config: dict[str, Any]) -> bool:
    return config_value_bool(
        config,
        "vacth_adaptive_repair_progress_gate",
        config.get("method") == "vacth_optimized",
    )


def source_patch_guard_max_attempts(config: dict[str, Any]) -> int:
    value = config.get("vacth_source_patch_guard_max_attempts")
    if value is None:
        return 2 if config.get("method") == "vacth_optimized" else 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 2 if config.get("method") == "vacth_optimized" else 0


def changed_source_files(repair_tools: SoftwareRepairTools) -> list[str]:
    return [path for path in repair_tools.changed_files() if not is_test_path(path)]


def source_patch_changed_lines(
    repair_tools: SoftwareRepairTools,
) -> list[tuple[str, str, str]]:
    """Return changed non-test source lines as (path, sign, stripped_line)."""
    changed_lines: list[tuple[str, str, str]] = []
    for relative_path in changed_source_files(repair_tools):
        path = repair_tools.workspace / relative_path
        after = repair_tools._read_workspace_text(path) or ""
        before = repair_tools.original_files.get(relative_path, "")
        for line in difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        ):
            if line.startswith(("--- ", "+++ ", "@@")) or not line or line[0] not in {"+", "-"}:
                continue
            changed_lines.append((relative_path, line[0], line[1:].strip()))
    return changed_lines


def is_trivial_source_change_line(text: str) -> bool:
    if not text:
        return True
    return text.startswith("#") or text.startswith("import ") or text.startswith("from ")


def imported_names_from_line(text: str) -> list[str]:
    """Best-effort names bound by a single-line import statement."""
    if text.startswith("import "):
        imports = text.removeprefix("import ")
        names = []
        for import_part in imports.split(","):
            part = import_part.strip()
            if not part:
                continue
            if " as " in part:
                names.append(part.rsplit(" as ", 1)[1].strip())
            else:
                names.append(part.split(".", 1)[0].strip())
        return [name for name in names if name and name != "*"]
    if text.startswith("from ") and " import " in text:
        imports = text.split(" import ", 1)[1]
        if imports.startswith("("):
            return []
        names = []
        for import_part in imports.split(","):
            part = import_part.strip().strip("()")
            if not part or part == "*":
                continue
            if " as " in part:
                names.append(part.rsplit(" as ", 1)[1].strip())
            else:
                names.append(part.strip())
        return [name for name in names if name]
    return []


def unused_added_imports(repair_tools: SoftwareRepairTools) -> dict[str, list[str]]:
    unused_by_file: dict[str, list[str]] = {}
    for relative_path in changed_source_files(repair_tools):
        added_import_names: list[str] = []
        for path, sign, text in source_patch_changed_lines(repair_tools):
            if path == relative_path and sign == "+":
                added_import_names.extend(imported_names_from_line(text))
        if not added_import_names:
            continue
        path = repair_tools.workspace / relative_path
        current_text = repair_tools._read_workspace_text(path) or ""
        implementation_text = "\n".join(
            line for line in current_text.splitlines() if not line.strip().startswith(("import ", "from "))
        )
        unused_names = [
            name
            for name in sorted(set(added_import_names))
            if not re.search(rf"\b{re.escape(name)}\b", implementation_text)
        ]
        if unused_names:
            unused_by_file[relative_path] = unused_names
    return unused_by_file


def source_patch_guard_reason(repair_tools: SoftwareRepairTools) -> str | None:
    source_files = changed_source_files(repair_tools)
    if not source_files:
        return "git_diff has no non-test source changes yet."
    changed_lines = source_patch_changed_lines(repair_tools)
    if changed_lines and all(is_trivial_source_change_line(text) for _path, _sign, text in changed_lines):
        return (
            "the current non-test source diff only changes import/comment/blank "
            "lines. That is often a partial repair; imported helpers must be used "
            "by the implementation unless an import-only fix is explicitly required."
        )
    unused_imports = unused_added_imports(repair_tools)
    if unused_imports:
        details = "; ".join(f"{path}: {', '.join(names)}" for path, names in sorted(unused_imports.items()))
        return (
            "the current non-test source diff adds imports/helpers that are not "
            f"referenced by the implementation ({details}). This is often a partial "
            "repair; either use the imported helpers in the actual fix or remove them."
        )
    return None


def should_stop_after_source_patch_guards(
    *,
    config: dict[str, Any],
    repair_tools: SoftwareRepairTools,
    source_patch_guards: int,
    max_source_patch_guards: int,
) -> bool:
    if config.get("method") != "vacth_optimized" or max_source_patch_guards <= 0:
        return False
    if source_patch_guards < max_source_patch_guards:
        return False
    guard_reason = source_patch_guard_reason(repair_tools)
    if guard_reason is None:
        return False
    config["vacth_source_patch_guard_exhausted"] = True
    config["vacth_source_patch_guard_reason"] = guard_reason
    return True


def repair_tools_last_test_current(repair_tools: SoftwareRepairTools) -> bool:
    return (
        repair_tools._last_test_result is not None
        and repair_tools._last_test_fingerprint == repair_tools.workspace_fingerprint()
        and repair_tools._last_test_command == repair_tools.test_command
    )


def test_failure_score(result: dict[str, Any] | None) -> int | None:
    """Return a rough count of failing/erroring tests from unittest or pytest output."""
    if not result:
        return None
    if bool(result.get("passed")):
        return 0
    output = str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))
    failed_block = re.search(r"FAILED \((?P<parts>[^)]*)\)", output, re.IGNORECASE)
    if failed_block:
        score = 0
        for key in ("failures", "errors"):
            match = re.search(rf"{key}=(\d+)", failed_block.group("parts"), re.IGNORECASE)
            if match:
                score += int(match.group(1))
        if score:
            return score
    pytest_summary = re.search(r"=+ (?P<summary>[^=\n]*\bfailed\b[^=\n]*)=+", output, re.IGNORECASE)
    if pytest_summary:
        score = 0
        for key in ("failed", "error", "errors"):
            for match in re.finditer(rf"(\d+)\s+{key}\b", pytest_summary.group("summary"), re.IGNORECASE):
                score += int(match.group(1))
        if score:
            return score
    return None


def should_attempt_adaptive_repair(
    *,
    config: dict[str, Any],
    repair_tools: SoftwareRepairTools,
) -> bool:
    if not adaptive_repair_enabled(config):
        return False
    if not repair_tools_last_test_current(repair_tools):
        return False
    last_result = repair_tools._last_test_result or {}
    if bool(last_result.get("passed")):
        return False
    output = (str(last_result.get("stdout", "")) + "\n" + str(last_result.get("stderr", ""))).lower()
    environment_markers = (
        "modulenotfounderror",
        "importerror",
        "no module named",
        "failed to import",
        "extension modules",
        "setup command",
        "metadata-generation-failed",
        "could not build",
    )
    if any(marker in output for marker in environment_markers):
        return False
    failure_score = test_failure_score(last_result)
    if failure_score is not None and failure_score > 0:
        return True
    repair_markers = (
        "assertionerror",
        "operationalerror",
        "syntax error",
        " failed",
        "fail:",
        "failed (failures=",
        "failed tests",
        "expected",
        "actual",
    )
    return any(marker in output for marker in repair_markers)


def local_repair_passed(
    *,
    config: dict[str, Any],
    repair_tools: SoftwareRepairTools,
    task_spec: dict[str, Any],
) -> bool:
    if not local_validation_early_stop_enabled(config):
        return False
    if not repair_tools.has_current_test_pass():
        return False
    changed_files = repair_tools.changed_files()
    if not changed_files:
        return False
    if bool(task_spec.get("forbid_test_changes", True)):
        if any(is_test_path(path) for path in changed_files):
            return False
    expected_changed_files = [str(path) for path in task_spec.get("expected_changed_files", [])]
    require_expected_changes = bool(expected_changed_files and task_spec.get("require_expected_changes", True))
    if require_expected_changes and set(expected_changed_files) - set(changed_files):
        return False
    return True


def log_local_early_stop(
    *,
    logger: ExperimentLogger,
    repair_tools: SoftwareRepairTools,
    task_spec: dict[str, Any],
    config: dict[str, Any],
    source_agent: str,
) -> bool:
    if final_answer_from_messages(logger.raw_messages).startswith("resolved"):
        return False
    if not local_repair_passed(
        config=config,
        repair_tools=repair_tools,
        task_spec=task_spec,
    ):
        return False
    config["stop_reason"] = (
        f"Local validation early stop: run_tests passed for the current source diff after {source_agent}."
    )
    logger.log_synthetic_message(
        turn_id=logger.next_turn("reviewer"),
        source="reviewer",
        role="assistant",
        content="FINAL_ANSWER: resolved",
    )
    return True


class LocalRepairPassedTermination(TerminationCondition):
    """Terminate a round-robin team once local tests pass for the current diff."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        repair_tools: SoftwareRepairTools,
        task_spec: dict[str, Any],
    ) -> None:
        self._config = config
        self._repair_tools = repair_tools
        self._task_spec = task_spec
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")
        if not messages:
            return None
        if not local_repair_passed(
            config=self._config,
            repair_tools=self._repair_tools,
            task_spec=self._task_spec,
        ):
            return None
        self._terminated = True
        return StopMessage(
            content="Local validation passed for the current source diff",
            source="LocalRepairPassedTermination",
        )

    async def reset(self) -> None:
        self._terminated = False


def make_recording_client(
    agent: str, logger: ExperimentLogger, clients: list[RecordingModelClient]
) -> RecordingModelClient:
    client = RecordingModelClient(load_model_client("config.yaml"), agent, logger)
    clients.append(client)
    return client


def focused_test_evidence_for_prompt(repair_tools: SoftwareRepairTools, max_chars: int) -> str:
    evidence = json.loads(repair_tools.test_evidence())
    if not evidence.get("available"):
        return ""
    lines = [
        "Focused FAIL_TO_PASS test evidence:",
        "The selected tests are already applied in this workspace. Use these snippets to preserve exact edge cases.",
        "Labels: " + ", ".join(str(label) for label in evidence.get("fail_to_pass", [])),
    ]
    for snippet in evidence.get("snippets", []):
        lines.append(
            "\n"
            f"[{snippet.get('label')}] {snippet.get('path')}:{snippet.get('start_line')}-"
            f"{snippet.get('end_line')}\n{snippet.get('content')}"
        )
    return truncate_chars("\n".join(lines), max_chars)


def create_repair_agents(
    logger: ExperimentLogger,
    repair_tools: SoftwareRepairTools,
    task_spec: dict[str, Any],
    clients: list[RecordingModelClient],
    model_contexts: dict[str, ChatCompletionContext] | None = None,
) -> dict[str, AssistantAgent]:
    contexts = model_contexts or {}
    reviewer_criteria = str(
        task_spec.get(
            "reviewer_success_criteria",
            DEFAULT_SOFTWARE_TASK["reviewer_success_criteria"],
        )
    )
    is_vacth_optimized = logger.config.get("method") == "vacth_optimized"
    retriever_prompt = (
        "You are the retriever. Use tools to inspect the workspace, "
        "identify the concrete failing behavior, and report evidence. "
        "Do not modify files."
    )
    if is_vacth_optimized:
        retriever_prompt += (
            " Run tests only when the task lacks concrete failing-test evidence; "
            "otherwise call test_evidence, inspect the relevant files first, and "
            "save rework. Prefer "
            "focused read_file snippets around search result line numbers over "
            "reading whole large files."
        )
    retriever_tools = [
        repair_tools.list_files,
        repair_tools.read_file,
        repair_tools.search_repo,
        repair_tools.run_tests,
    ]
    coder_tools = [
        repair_tools.search_repo,
        repair_tools.read_file,
        repair_tools.replace_text,
        repair_tools.git_diff,
    ]
    if is_vacth_optimized:
        retriever_tools = [
            repair_tools.test_evidence,
            repair_tools.list_files,
            repair_tools.read_file,
            repair_tools.search_repo,
            repair_tools.run_tests,
        ]
        coder_tools = [
            repair_tools.test_evidence,
            repair_tools.search_repo,
            repair_tools.read_file,
            repair_tools.replace_text,
            repair_tools.git_diff,
        ]
    reviewer_tools = [repair_tools.run_tests, repair_tools.git_diff, repair_tools.read_file]
    reviewer_prompt = (
        "You are the reviewer. Verify the diff and tests with tools. "
        f"Success criteria: {reviewer_criteria} "
        "If the criteria are satisfied, end with exactly "
        "FINAL_ANSWER: resolved. Otherwise end with exactly "
        "FINAL_ANSWER: unresolved. Do not include any text after the "
        "FINAL_ANSWER line."
    )
    if is_vacth_optimized:
        reviewer_tools = [
            repair_tools.test_status,
            repair_tools.git_diff,
            repair_tools.run_tests,
            repair_tools.read_file,
        ]
        reviewer_prompt = (
            "You are the reviewer. First call test_status and git_diff. "
            "Run run_tests only if test_status says current_diff_tested is false; "
            "do not rerun tests for a diff that test_status says already failed. "
            f"Success criteria: {reviewer_criteria} "
            "If the criteria are satisfied, end with exactly "
            "FINAL_ANSWER: resolved. Otherwise end with exactly "
            "FINAL_ANSWER: unresolved. Do not include any text after the "
            "FINAL_ANSWER line."
        )
    default_no_reflection = ["retriever", "coder", "tester"] if is_vacth_optimized else []
    disable_tool_reflection_agents = {
        str(name) for name in logger.config.get("vacth_disable_tool_reflection_agents", default_no_reflection)
    }
    logger.config["vacth_disable_tool_reflection_agents"] = sorted(disable_tool_reflection_agents)

    def reflect_for(agent_name: str) -> bool:
        return agent_name not in disable_tool_reflection_agents

    return {
        "planner": AssistantAgent(
            "planner",
            model_client=make_recording_client("planner", logger, clients),
            model_context=contexts.get("planner"),
            description="Creates a concise repair plan.",
            system_message=(
                "You are the planner. Read the user task and produce a short "
                "diagnosis plan for the team. Do not edit files. Tell the next "
                "agents which files and tests should be inspected."
            ),
        ),
        "retriever": AssistantAgent(
            "retriever",
            model_client=make_recording_client("retriever", logger, clients),
            model_context=contexts.get("retriever"),
            tools=retriever_tools,
            reflect_on_tool_use=reflect_for("retriever"),
            max_tool_iterations=3,
            description="Inspects source files and tests.",
            system_message=retriever_prompt,
        ),
        "coder": AssistantAgent(
            "coder",
            model_client=make_recording_client("coder", logger, clients),
            model_context=contexts.get("coder"),
            tools=coder_tools,
            reflect_on_tool_use=reflect_for("coder"),
            max_tool_iterations=4 if is_vacth_optimized else 3,
            description="Applies the minimal source change.",
            system_message=(
                "You are the coder. Use tools to make the minimal source edit "
                "that satisfies the task and constraints. When the issue names "
                "a function, edit that exact function; use search_repo and "
                "focused read_file snippets to disambiguate repeated code before "
                "calling replace_text. After locating the relevant line, make the "
                "source edit in the same turn rather than only reporting findings. "
                "For SWE-bench tasks, use test_evidence to match all visible "
                "FAIL_TO_PASS assertions and edge cases, not only the issue prose. "
                "If a prior tester failure is visible, "
                "revise the existing diff instead of stopping at the failed patch. "
                "Do not change tests."
            ),
        ),
        "tester": AssistantAgent(
            "tester",
            model_client=make_recording_client("tester", logger, clients),
            model_context=contexts.get("tester"),
            tools=[repair_tools.run_tests, repair_tools.git_diff],
            reflect_on_tool_use=reflect_for("tester"),
            max_tool_iterations=2,
            description="Runs tests and reports pass/fail evidence.",
            system_message=(
                "You are the tester. Run the tests with the tool, inspect the "
                "diff if useful, and report whether the repair is verified."
            ),
        ),
        "reviewer": AssistantAgent(
            "reviewer",
            model_client=make_recording_client("reviewer", logger, clients),
            model_context=contexts.get("reviewer"),
            tools=reviewer_tools,
            reflect_on_tool_use=True,
            max_tool_iterations=2,
            description="Reviews the final state and decides the outcome.",
            system_message=reviewer_prompt,
        ),
    }


class SummaryRuntime:
    def __init__(self, logger: ExperimentLogger, model_client: RecordingModelClient) -> None:
        self.logger = logger
        self.model_client = model_client
        self.current_summary = ""

    def build_agent_task(self, *, receiver: str, base_task: str) -> str:
        if not self.current_summary:
            return (
                f"{base_task}\n\n"
                "Communication protocol: this baseline starts with the original "
                "task. Later agents will receive only a natural-language handoff "
                "summary instead of the full transcript."
            )
        final_instruction = ""
        if receiver == "reviewer":
            final_instruction = (
                "\n\nReviewer instruction: verify the final state with tools and "
                "end with exactly FINAL_ANSWER: resolved or FINAL_ANSWER: unresolved."
            )
        return (
            f"{base_task}\n\n"
            "Communication protocol: you do not receive the full prior transcript. "
            "You receive only the free-text handoff summary below plus your tools. "
            "Continue the task from this summary.\n\n"
            f"Current handoff summary:\n{self.current_summary}"
            f"{final_instruction}"
        )

    async def update_after_turn(self, *, sender: str, source_messages: list[dict[str, Any]], base_task: str) -> str:
        previous_summary = self.current_summary
        latest_turn = "\n\n".join(
            f"{message.get('source')} [{message.get('type')}]:\n{message.get('content', '')}"
            for message in source_messages
        )
        prompt = (
            "Update the handoff summary for a multi-agent software repair task.\n"
            "Write concise natural language, not JSON. Preserve task constraints, "
            "important facts, tool results, code changes, open issues, and the next "
            "recommended action. Do not invent facts.\n\n"
            f"Original task:\n{base_task}\n\n"
            f"Previous summary:\n{previous_summary or '(none yet)'}\n\n"
            f"Latest turn from {sender}:\n{latest_turn}\n\n"
            "Updated handoff summary:"
        )
        result = await self.model_client.create(
            [
                SystemMessage(
                    content=(
                        "You maintain a free-text handoff summary baseline. "
                        "You are not allowed to output structured JSON, provenance "
                        "graphs, typed state items, or routing decisions."
                    )
                ),
                UserMessage(content=prompt, source="summary_runtime"),
            ]
        )
        summary = (
            result.content
            if isinstance(result.content, str)
            else json.dumps(to_jsonable(result.content), ensure_ascii=False)
        )
        self.current_summary = summary.strip()
        self.logger.log_summary(
            turn_id=self.logger.current_turn("summary_runtime"),
            sender=sender,
            summary=self.current_summary,
            previous_summary=previous_summary,
            source_messages=source_messages,
        )
        return self.current_summary


class VectorMemoryRuntime:
    def __init__(self, logger: ExperimentLogger, top_k: int) -> None:
        self.logger = logger
        self.top_k = top_k

    def add_messages(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            if message.get("role") not in {"assistant", "event"}:
                continue
            if message.get("source") == "summary_runtime":
                continue
            self.logger.log_memory_item(message)

    def retrieve(self, *, receiver: str, query: str) -> list[dict[str, Any]]:
        query_terms = retrieval_terms(query)
        scored: list[dict[str, Any]] = []
        for item in self.logger.memory_items:
            score = cosine_similarity(query_terms, retrieval_terms(str(item.get("content", ""))))
            if score <= 0:
                continue
            enriched = dict(item)
            enriched["score"] = round(score, 4)
            scored.append(enriched)

        scored.sort(
            key=lambda item: (
                float(item.get("score", 0)),
                int(item.get("turn_id") or 0),
            ),
            reverse=True,
        )
        retrieved = scored[: self.top_k]
        self.logger.log_retrieval(
            turn_id=self.logger.next_turn("vector_memory_runtime"),
            receiver=receiver,
            query=query,
            retrieved=retrieved,
            top_k=self.top_k,
        )
        return retrieved

    def build_agent_task(self, *, receiver: str, base_task: str, previous_agent: str | None) -> str:
        query = f"{base_task}\nreceiver role: {receiver}\nprevious agent: {previous_agent or '(none)'}"
        retrieved = self.retrieve(receiver=receiver, query=query)
        memory_text = "\n\n".join(
            "[{item_id}] {source} turn {turn_id} score={score}\n{content}".format(
                item_id=item.get("item_id"),
                source=item.get("source"),
                turn_id=item.get("turn_id"),
                score=item.get("score", 0),
                content=item.get("content", ""),
            )
            for item in retrieved
        )
        if not memory_text:
            memory_text = "(no retrieved memory yet)"

        final_instruction = ""
        if receiver == "reviewer":
            final_instruction = (
                "\n\nReviewer instruction: verify the final state with tools and "
                "end with exactly FINAL_ANSWER: resolved or FINAL_ANSWER: unresolved."
            )

        return (
            f"{base_task}\n\n"
            "Communication protocol: you do not receive the full prior transcript. "
            "You receive only the top-k lexical vector-memory items below plus "
            "your tools. Treat memory entries as lossy retrieved evidence, not as "
            "a complete conversation history.\n\n"
            f"Retrieved memory (top_k={self.top_k}):\n{memory_text}"
            f"{final_instruction}"
        )


class StructuredSummaryRuntime:
    def __init__(
        self,
        logger: ExperimentLogger,
        model_client: RecordingModelClient,
        max_items_per_slot: int,
    ) -> None:
        self.logger = logger
        self.model_client = model_client
        self.max_items_per_slot = max_items_per_slot
        self.state = empty_structured_state()

    def build_agent_task(self, *, receiver: str, base_task: str) -> str:
        final_instruction = ""
        if receiver == "reviewer":
            final_instruction = (
                "\n\nReviewer instruction: verify the final state with tools and "
                "end with exactly FINAL_ANSWER: resolved or FINAL_ANSWER: unresolved."
            )
        state_text = json.dumps(self.state, ensure_ascii=False, indent=2)
        return (
            f"{base_task}\n\n"
            "Communication protocol: you do not receive the full prior transcript. "
            "You receive the same global structured JSON summary as every other "
            "agent, plus your tools. Treat the JSON as a lossy task-state summary. "
            "It has no provenance graph, no evidence pointers, no merge keys, and "
            "no receiver-specific routing.\n\n"
            f"Global structured summary:\n```json\n{state_text}\n```"
            f"{final_instruction}"
        )

    async def update_after_turn(
        self, *, sender: str, source_messages: list[dict[str, Any]], base_task: str
    ) -> dict[str, list[str]]:
        previous_state = json.loads(json.dumps(self.state, ensure_ascii=False))
        latest_turn = "\n\n".join(
            f"{message.get('source')} [{message.get('type')}]:\n{message.get('content', '')}"
            for message in source_messages
        )
        prompt = (
            "Update the global structured JSON summary for a multi-agent software "
            "repair task. Return only valid JSON with exactly these top-level keys: "
            "facts, constraints, decisions, hypotheses, tool_states, open_issues. "
            "Every value must be a list of short strings. Preserve important "
            "constraints, observed facts, tool results, code edits, test outcomes, "
            "hypotheses, and open issues. Do not include evidence pointers, "
            "provenance edges, merge keys, supports/contradicts/supersedes links, "
            "receiver-specific routing, or VACTH capsule fields. Do not invent "
            "facts.\n\n"
            f"Original task:\n{base_task}\n\n"
            f"Previous JSON state:\n{json.dumps(self.state, ensure_ascii=False, indent=2)}\n\n"
            f"Latest turn from {sender}:\n{latest_turn}\n\n"
            "Updated JSON state:"
        )
        result = await self.model_client.create(
            [
                SystemMessage(
                    content=(
                        "You maintain a structured-summary baseline. Output only "
                        "a single JSON object. This baseline is not VACTH and must "
                        "not emit provenance, routing decisions, typed state item "
                        "schemas, evidence pointers, or capsule metadata."
                    )
                ),
                UserMessage(content=prompt, source="structured_summary_runtime"),
            ]
        )
        raw_update = (
            result.content
            if isinstance(result.content, str)
            else json.dumps(to_jsonable(result.content), ensure_ascii=False)
        ).strip()
        parse_error: str | None = None
        valid = True
        try:
            parsed = extract_json_object(raw_update)
            self.state = normalize_structured_state(parsed, max_items_per_slot=self.max_items_per_slot)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            parse_error = str(exc)
            valid = False
            self.state = previous_state

        self.logger.log_structured_state(
            turn_id=self.logger.current_turn("structured_summary_runtime"),
            sender=sender,
            state=self.state,
            previous_state=previous_state,
            source_messages=source_messages,
            valid=valid,
            parse_error=parse_error,
            raw_update=raw_update,
        )
        return self.state


def compact_vacth_task(base_task: str, *, max_task_chars: int = 1800) -> str:
    """Keep downstream VACTH prompts focused once retrieval has populated state."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in base_task.splitlines():
        stripped = line.strip()
        if stripped in {"Task:", "Constraints:", "Gold state:"}:
            current = stripped.removesuffix(":").lower().replace(" ", "_")
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)

    task_text = "\n".join(sections.get("task", [])).strip()
    constraints = "\n".join(sections.get("constraints", [])).strip()
    gold_state = "\n".join(sections.get("gold_state", [])).strip()
    if len(task_text) > max_task_chars:
        task_text = task_text[:max_task_chars].rstrip() + "\n...[truncated; use VACTH capsule for retrieved details]"
    if not sections:
        task_text = base_task[:max_task_chars].rstrip()

    parts = ["Task brief:", task_text]
    if constraints:
        parts.extend(["", "Constraints:", constraints])
    if gold_state:
        parts.extend(["", "Gold state:", gold_state])
    return "\n".join(parts).strip()


class VACTHFullRuntime:
    def __init__(
        self,
        *,
        logger: ExperimentLogger,
        extractor_client: RecordingModelClient | None,
        task_id: str,
        token_budget: int,
        method_name: str,
        enable_thc: bool,
        enable_cve: bool,
        enable_paa: bool,
        enable_provenance: bool,
        enable_reask: bool,
        role_specific_routing: bool,
        skip_extraction_agents: set[str] | None = None,
        compact_downstream_task: bool = False,
        role_token_budgets: dict[str, int] | None = None,
        skip_planner_capsule: bool = False,
        compact_capsule: bool = False,
        capsule_max_item_chars: int = 180,
        hybrid_extraction: bool = False,
        thc_source_token_budget: int = 0,
        thc_existing_item_limit: int = 12,
        thc_max_items_per_turn: int = 0,
    ) -> None:
        self.logger = logger
        self.extractor_client = extractor_client
        self.enable_thc = enable_thc
        self.enable_cve = enable_cve
        self.enable_reask = enable_reask
        self.role_specific_routing = role_specific_routing
        self.skip_extraction_agents = skip_extraction_agents or set()
        self.compact_downstream_task = compact_downstream_task
        self.role_token_budgets = role_token_budgets or {}
        self.skip_planner_capsule = skip_planner_capsule
        self.compact_capsule = compact_capsule
        self.capsule_max_item_chars = capsule_max_item_chars
        self.hybrid_extraction = hybrid_extraction
        self.thc_source_token_budget = thc_source_token_budget
        self.thc_existing_item_limit = thc_existing_item_limit
        self.thc_max_items_per_turn = thc_max_items_per_turn
        self.runtime = VACTHRuntime(
            task_id=task_id,
            token_budget=token_budget,
            method_name=method_name,
            enable_cve=enable_cve,
            enable_paa=enable_paa,
            enable_provenance=enable_provenance,
            role_specific_routing=role_specific_routing,
        )
        self.next_item_index = 1

    def build_agent_task(self, *, receiver: str, base_task: str, previous_agent: str | None) -> str:
        if self.skip_planner_capsule and receiver == "planner":
            return (
                f"{base_task}\n\n"
                "Communication protocol: start from the original task. Later agents "
                "receive compact VACTH state capsules instead of the full transcript."
            )

        turn_id = self.logger.next_turn("vacth_runtime")
        capsule, routing_decision = self.runtime.build_capsule(
            receiver=receiver,
            sender=previous_agent or "vacth_runtime",
            turn_id=turn_id,
            token_budget=self.role_token_budgets.get(receiver),
        )
        self.logger.log_capsule(capsule=capsule, routing_decision=routing_decision)
        rendered_capsule = self.runtime.render_capsule(
            capsule,
            compact=self.compact_capsule,
            max_item_chars=self.capsule_max_item_chars,
        )
        final_instruction = ""
        if receiver == "reviewer":
            final_instruction = (
                "\n\nReviewer instruction: verify the final state with tools and "
                "end with exactly FINAL_ANSWER: resolved or FINAL_ANSWER: unresolved."
            )
        capsule_description = (
            "a receiver-specific VACTH capsule" if self.role_specific_routing else "a global VACTH capsule"
        )
        routing_description = (
            "selected under a contextual value token budget"
            if self.enable_cve
            else "assembled chronologically under a token budget"
        )
        visible_task = (
            compact_vacth_task(base_task)
            if self.compact_downstream_task and receiver not in {"planner", "retriever"}
            else base_task
        )
        return (
            f"{visible_task}\n\n"
            "Communication protocol: you do not receive the full prior transcript. "
            f"You receive {capsule_description} {routing_description} "
            "from the shared typed state graph. Use capsule evidence, "
            "constraints, tool state, artifacts, and unresolved issues to continue "
            "the task.\n\n"
            f"{rendered_capsule}"
            f"{final_instruction}"
        )

    async def update_after_turn(self, *, sender: str, source_messages: list[dict[str, Any]], base_task: str) -> None:
        if not source_messages:
            return
        if sender in self.skip_extraction_agents:
            return
        if self.hybrid_extraction and self.enable_thc:
            await self._update_after_turn_hybrid(
                sender=sender,
                source_messages=source_messages,
                base_task=base_task,
            )
            return
        parse_valid = True
        parse_error: str | None = None
        extraction_mode = "thc"
        reask_attempts = 0

        if not self.enable_thc:
            turn_id = self.logger.next_turn("vacth_extractor")
            items = heuristic_items_from_messages(
                sender=sender,
                source_messages=source_messages,
                turn_id=turn_id,
                next_item_index=self.next_item_index,
            )
            edges: list[VACTHEdge] = []
            raw_update = json.dumps(
                {
                    "mode": "heuristic_without_thc",
                    "source_message_count": len(source_messages),
                    "item_count": len(items),
                },
                ensure_ascii=False,
            )
            extraction_mode = "heuristic"
        else:
            extraction_task = (
                compact_vacth_task(base_task)
                if self.compact_downstream_task and sender not in {"planner", "retriever"}
                else base_task
            )
            prompt = build_extraction_prompt(
                task=extraction_task,
                sender=sender,
                source_messages=self._bounded_source_messages(source_messages),
                existing_items=[item.model_dump(mode="json") for item in self.runtime.aggregator.active_items()],
                max_existing_items=self.thc_existing_item_limit,
                max_items_per_turn=self.thc_max_items_per_turn or None,
            )
            raw_update = await self._request_extraction(prompt)
            turn_id = self.logger.current_turn("vacth_extractor")
            try:
                items, edges = parse_extraction_output(
                    raw_update,
                    sender=sender,
                    source_messages=self._bounded_source_messages(source_messages),
                    turn_id=turn_id,
                    next_item_index=self.next_item_index,
                )
                if self.thc_max_items_per_turn > 0:
                    items = items[: self.thc_max_items_per_turn]
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                initial_error = str(exc)
                if self.enable_reask:
                    reask_attempts = 1
                    raw_update = await self._request_extraction(
                        (
                            f"{prompt}\n\n"
                            "Previous extractor output was invalid JSON for this "
                            f"reason: {initial_error}\n"
                            "Return exactly one valid JSON object with keys "
                            "'items' and 'edges'."
                        )
                    )
                    turn_id = self.logger.current_turn("vacth_extractor")
                    try:
                        items, edges = parse_extraction_output(
                            raw_update,
                            sender=sender,
                            source_messages=source_messages,
                            turn_id=turn_id,
                            next_item_index=self.next_item_index,
                        )
                    except (json.JSONDecodeError, TypeError, ValueError) as reask_exc:
                        parse_valid = False
                        parse_error = f"{initial_error}; reask: {reask_exc}"
                        extraction_mode = "heuristic"
                        items = heuristic_items_from_messages(
                            sender=sender,
                            source_messages=source_messages,
                            turn_id=turn_id,
                            next_item_index=self.next_item_index,
                        )
                        edges = []
                else:
                    parse_valid = False
                    parse_error = initial_error
                    extraction_mode = "heuristic"
                    items = heuristic_items_from_messages(
                        sender=sender,
                        source_messages=source_messages,
                        turn_id=turn_id,
                        next_item_index=self.next_item_index,
                    )
                    edges = []

        self.next_item_index += len(items)
        delta = self.runtime.update(items, edges)
        added_items = [item for item in delta["items"] if isinstance(item, VACTHStateItem)]
        added_edges = [edge for edge in delta["edges"] if isinstance(edge, VACTHEdge)]
        self.logger.log_vacth_extraction(
            turn_id=turn_id,
            sender=sender,
            raw_update=raw_update,
            parse_valid=parse_valid,
            parse_error=parse_error,
            extraction_mode=extraction_mode,
            reask_attempts=reask_attempts,
            items=added_items,
            edges=added_edges,
            source_messages=source_messages,
        )

    async def _update_after_turn_hybrid(
        self, *, sender: str, source_messages: list[dict[str, Any]], base_task: str
    ) -> None:
        tool_messages = [message for message in source_messages if message.get("role") == "event"]
        assistant_messages = [
            message
            for message in source_messages
            if message.get("role") == "assistant" and message.get("type") != "ToolCallSummaryMessage"
        ]
        if not tool_messages and not assistant_messages:
            return

        turn_id = self.logger.current_turn("vacth_extractor")
        items: list[VACTHStateItem] = []
        edges: list[VACTHEdge] = []
        raw_parts: list[dict[str, Any]] = []
        parse_valid = True
        parse_error: str | None = None
        reask_attempts = 0
        extraction_mode = "hybrid"

        if tool_messages:
            tool_items = heuristic_items_from_messages(
                sender=sender,
                source_messages=tool_messages,
                turn_id=turn_id,
                next_item_index=self.next_item_index,
            )
            items.extend(tool_items)
            raw_parts.append(
                {
                    "mode": "deterministic_tool_events",
                    "source_message_count": len(tool_messages),
                    "item_count": len(tool_items),
                }
            )

        if assistant_messages:
            bounded_messages = self._bounded_source_messages(assistant_messages)
            extraction_task = (
                compact_vacth_task(base_task)
                if self.compact_downstream_task and sender not in {"planner", "retriever"}
                else base_task
            )
            prompt = build_extraction_prompt(
                task=extraction_task,
                sender=sender,
                source_messages=bounded_messages,
                existing_items=[item.model_dump(mode="json") for item in self.runtime.aggregator.active_items()],
                max_existing_items=self.thc_existing_item_limit,
                max_items_per_turn=self.thc_max_items_per_turn or None,
            )
            raw_llm_update = await self._request_extraction(prompt)
            turn_id = self.logger.current_turn("vacth_extractor")
            try:
                llm_items, llm_edges = parse_extraction_output(
                    raw_llm_update,
                    sender=sender,
                    source_messages=bounded_messages,
                    turn_id=turn_id,
                    next_item_index=self.next_item_index + len(items),
                )
                if self.thc_max_items_per_turn > 0:
                    llm_items = llm_items[: self.thc_max_items_per_turn]
                items.extend(llm_items)
                edges.extend(llm_edges)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                initial_error = str(exc)
                if self.enable_reask:
                    reask_attempts = 1
                    raw_llm_update = await self._request_extraction(
                        (
                            f"{prompt}\n\n"
                            "Previous extractor output was invalid JSON for this "
                            f"reason: {initial_error}\n"
                            "Return exactly one valid JSON object with keys "
                            "'items' and 'edges'."
                        )
                    )
                    turn_id = self.logger.current_turn("vacth_extractor")
                    try:
                        llm_items, llm_edges = parse_extraction_output(
                            raw_llm_update,
                            sender=sender,
                            source_messages=bounded_messages,
                            turn_id=turn_id,
                            next_item_index=self.next_item_index + len(items),
                        )
                        if self.thc_max_items_per_turn > 0:
                            llm_items = llm_items[: self.thc_max_items_per_turn]
                        items.extend(llm_items)
                        edges.extend(llm_edges)
                    except (json.JSONDecodeError, TypeError, ValueError) as reask_exc:
                        parse_valid = False
                        parse_error = f"{initial_error}; reask: {reask_exc}"
                else:
                    parse_valid = False
                    parse_error = initial_error
                if not parse_valid:
                    extraction_mode = "hybrid_heuristic_fallback"
                    fallback_items = heuristic_items_from_messages(
                        sender=sender,
                        source_messages=bounded_messages,
                        turn_id=turn_id,
                        next_item_index=self.next_item_index + len(items),
                    )
                    items.extend(fallback_items)
            raw_parts.append(
                {
                    "mode": "thc_assistant_turn",
                    "source_message_count": len(assistant_messages),
                    "bounded_message_count": len(bounded_messages),
                    "raw_update": raw_llm_update,
                }
            )

        raw_update = json.dumps(
            {"mode": "hybrid", "parts": raw_parts},
            ensure_ascii=False,
        )
        self.next_item_index += len(items)
        delta = self.runtime.update(items, edges)
        added_items = [item for item in delta["items"] if isinstance(item, VACTHStateItem)]
        added_edges = [edge for edge in delta["edges"] if isinstance(edge, VACTHEdge)]
        self.logger.log_vacth_extraction(
            turn_id=turn_id,
            sender=sender,
            raw_update=raw_update,
            parse_valid=parse_valid,
            parse_error=parse_error,
            extraction_mode=extraction_mode,
            reask_attempts=reask_attempts,
            items=added_items,
            edges=added_edges,
            source_messages=source_messages,
        )

    def _bounded_source_messages(self, source_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.thc_source_token_budget <= 0:
            return source_messages
        selected: list[dict[str, Any]] = []
        remaining = self.thc_source_token_budget
        for message in reversed(source_messages):
            content = str(message.get("content", ""))
            tokens = estimate_tokens(content)
            if tokens > remaining:
                if remaining <= 0:
                    break
                copied = dict(message)
                copied["content"] = self._truncate_to_tokens(content, remaining)
                copied["estimated_tokens"] = estimate_tokens(str(copied["content"]))
                selected.append(copied)
                break
            selected.append(message)
            remaining -= tokens
        return list(reversed(selected))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens]) + " ...[truncated]"

    async def _request_extraction(self, prompt: str) -> str:
        if self.extractor_client is None:
            raise RuntimeError("THC extraction requested without an extractor client")
        try:
            result = await self.extractor_client.create(
                [
                    SystemMessage(
                        content=(
                            "You are the VACTH typed handoff capsule extractor. "
                            "Return only a compact JSON object with typed state items "
                            "and provenance edges. No markdown."
                        )
                    ),
                    UserMessage(content=prompt, source="vacth_extractor"),
                ]
            )
        except Exception as exc:
            if not is_retryable_model_error(exc):
                raise
            return json.dumps(
                {
                    "items": [],
                    "edges": [],
                    "model_error_fallback": {
                        "error_type": type(exc).__name__,
                        "error": str(exc)[-500:],
                    },
                },
                ensure_ascii=False,
            )
        return (
            result.content
            if isinstance(result.content, str)
            else json.dumps(to_jsonable(result.content), ensure_ascii=False)
        ).strip()


async def run_agent_once(agent: AssistantAgent, task: str, logger: ExperimentLogger) -> str:
    task_result: TaskResult | None = None
    try:
        async for message in agent.run_stream(task=task):
            if isinstance(message, TaskResult):
                task_result = message
                continue
            logger.log_stream_message(message)
    except TimeoutError:
        source = str(getattr(agent, "name", None) or getattr(agent, "_name", "unknown"))
        content = (
            "FINAL_ANSWER: unresolved"
            if source == "reviewer"
            else f"{source} model call timed out before completing this turn."
        )
        logger.log_synthetic_message(
            turn_id=logger.next_turn(source),
            source=source,
            role="assistant",
            content=content,
        )
        return content
    except Exception as exc:
        if not is_retryable_model_error(exc):
            raise
        source = str(getattr(agent, "name", None) or getattr(agent, "_name", "unknown"))
        content = (
            "FINAL_ANSWER: unresolved"
            if source == "reviewer"
            else f"{source} model call failed after retries: {type(exc).__name__}: {str(exc)[-500:]}"
        )
        logger.log_synthetic_message(
            turn_id=logger.next_turn(source),
            source=source,
            role="assistant",
            content=content,
        )
        return content
    if task_result is None:
        return ""
    for message in reversed(task_result.messages):
        if isinstance(message, BaseChatMessage) and message.source == agent.name:
            return message.to_text()
    return ""


async def run_autogen_broadcast(config: dict[str, Any]) -> None:
    start = time.time()
    logger = ExperimentLogger(config)
    task_spec, _, repair_tools = initialize_repair_workspace(config)
    clients: list[RecordingModelClient] = []

    config["model_config"] = describe_model_config("config.yaml")
    model_config = config["model_config"].get("config", {})
    if isinstance(model_config, dict):
        config["model"] = model_config.get("model", "config.yaml")

    task = (
        f"{read_prompt()}\n\n"
        "The repository to repair is in ./workspace. Work as a round-robin "
        "team. Use the tools when your role needs evidence, edits, diffs, or "
        "test results. The reviewer must end with exactly one line beginning "
        "with FINAL_ANSWER: resolved or FINAL_ANSWER: unresolved."
    )

    try:
        agents = create_repair_agents(logger, repair_tools, task_spec, clients)

        team = RoundRobinGroupChat(
            [agents[name] for name in ["planner", "retriever", "coder", "tester", "reviewer"]],
            termination_condition=(
                TextMentionTermination("FINAL_ANSWER:", sources=["reviewer"])
                | LocalRepairPassedTermination(
                    config=config,
                    repair_tools=repair_tools,
                    task_spec=task_spec,
                )
            ),
            max_turns=int(config["max_turns"]),
        )

        task_result: TaskResult | None = None
        try:
            async for message in team.run_stream(task=task):
                if isinstance(message, TaskResult):
                    task_result = message
                    continue
                logger.log_stream_message(message)
        except Exception as exc:
            if not is_retryable_model_error(exc):
                raise
            logger.log_synthetic_message(
                turn_id=logger.next_turn("reviewer"),
                source="reviewer",
                role="assistant",
                content=f"FINAL_ANSWER: unresolved\nModel error after retries: {type(exc).__name__}: {str(exc)[-500:]}",
            )

        log_local_early_stop(
            logger=logger,
            repair_tools=repair_tools,
            task_spec=task_spec,
            config=config,
            source_agent="team",
        )
        verification, success = evaluate_repair_result(
            logger=logger,
            repair_tools=repair_tools,
            task_spec=task_spec,
        )
        final_answer = "resolved" if success else "unresolved"
        if task_result is not None:
            config["stop_reason"] = task_result.stop_reason
        metrics = metric_summary(
            logger=logger,
            start=start,
            success=success,
            verification=verification,
        )
        logger.save(final_answer, metrics)

        print(f"FINAL ANSWER: {final_answer}")
        if success:
            print("ALL TESTS PASSED !#!#")
    finally:
        for client in clients:
            await client.close()


async def run_summary_baseline(config: dict[str, Any]) -> None:
    start = time.time()
    logger = ExperimentLogger(config)
    task_spec, _, repair_tools = initialize_repair_workspace(config)
    clients: list[RecordingModelClient] = []

    config["model_config"] = describe_model_config("config.yaml")
    model_config = config["model_config"].get("config", {})
    if isinstance(model_config, dict):
        config["model"] = model_config.get("model", "config.yaml")

    base_task = read_prompt()
    agents = create_repair_agents(logger, repair_tools, task_spec, clients)
    summary_client = make_recording_client("summary_runtime", logger, clients)
    summary_runtime = SummaryRuntime(logger, summary_client)
    ordered_agents = ["planner", "retriever", "coder", "tester", "reviewer"]

    try:
        try:
            for agent_name in ordered_agents:
                start_index = len(logger.raw_messages)
                agent_task = summary_runtime.build_agent_task(receiver=agent_name, base_task=base_task)
                await run_agent_once(agents[agent_name], agent_task, logger)
                new_messages = logger.raw_messages[start_index:]
                if agent_name != "reviewer":
                    await summary_runtime.update_after_turn(
                        sender=agent_name,
                        source_messages=new_messages,
                        base_task=base_task,
                    )
                    if log_local_early_stop(
                        logger=logger,
                        repair_tools=repair_tools,
                        task_spec=task_spec,
                        config=config,
                        source_agent=agent_name,
                    ):
                        break
        except Exception as exc:
            if not is_retryable_model_error(exc):
                raise
            logger.log_synthetic_message(
                turn_id=logger.next_turn("reviewer"),
                source="reviewer",
                role="assistant",
                content=f"FINAL_ANSWER: unresolved\nModel error after retries: {type(exc).__name__}: {str(exc)[-500:]}",
            )

        verification, success = evaluate_repair_result(
            logger=logger,
            repair_tools=repair_tools,
            task_spec=task_spec,
        )
        final_answer = "resolved" if success else "unresolved"
        metrics = metric_summary(
            logger=logger,
            start=start,
            success=success,
            verification=verification,
        )
        logger.save(final_answer, metrics)

        print(f"FINAL ANSWER: {final_answer}")
        if success:
            print("ALL TESTS PASSED !#!#")
    finally:
        for client in clients:
            await client.close()


async def run_sliding_window_baseline(config: dict[str, Any]) -> None:
    start = time.time()
    logger = ExperimentLogger(config)
    task_spec, _, repair_tools = initialize_repair_workspace(config)
    clients: list[RecordingModelClient] = []

    config["model_config"] = describe_model_config("config.yaml")
    model_config = config["model_config"].get("config", {})
    if isinstance(model_config, dict):
        config["model"] = model_config.get("model", "config.yaml")
    window_size = int(config.get("context_window_messages", 4))
    config["context_window_messages"] = window_size

    task = (
        f"{read_prompt()}\n\n"
        "The repository to repair is in ./workspace. Work as a round-robin "
        "team. Use the tools when your role needs evidence, edits, diffs, or "
        "test results. Communication protocol: each agent uses a sliding "
        f"window and can send only the latest {window_size} model-context "
        "messages to the LLM. The reviewer must end with exactly one line "
        "beginning with FINAL_ANSWER: resolved or FINAL_ANSWER: unresolved."
    )

    try:
        agent_names = ["planner", "retriever", "coder", "tester", "reviewer"]
        contexts = {name: BufferedChatCompletionContext(buffer_size=window_size) for name in agent_names}
        agents = create_repair_agents(
            logger,
            repair_tools,
            task_spec,
            clients,
            contexts,
        )
        team = RoundRobinGroupChat(
            [agents[name] for name in agent_names],
            termination_condition=(
                TextMentionTermination("FINAL_ANSWER:", sources=["reviewer"])
                | LocalRepairPassedTermination(
                    config=config,
                    repair_tools=repair_tools,
                    task_spec=task_spec,
                )
            ),
            max_turns=int(config["max_turns"]),
        )

        task_result: TaskResult | None = None
        try:
            async for message in team.run_stream(task=task):
                if isinstance(message, TaskResult):
                    task_result = message
                    continue
                logger.log_stream_message(message)
        except Exception as exc:
            if not is_retryable_model_error(exc):
                raise
            logger.log_synthetic_message(
                turn_id=logger.next_turn("reviewer"),
                source="reviewer",
                role="assistant",
                content=f"FINAL_ANSWER: unresolved\nModel error after retries: {type(exc).__name__}: {str(exc)[-500:]}",
            )

        log_local_early_stop(
            logger=logger,
            repair_tools=repair_tools,
            task_spec=task_spec,
            config=config,
            source_agent="team",
        )
        verification, success = evaluate_repair_result(
            logger=logger,
            repair_tools=repair_tools,
            task_spec=task_spec,
        )
        final_answer = "resolved" if success else "unresolved"
        if task_result is not None:
            config["stop_reason"] = task_result.stop_reason
        metrics = metric_summary(
            logger=logger,
            start=start,
            success=success,
            verification=verification,
        )
        logger.save(final_answer, metrics)

        print(f"FINAL ANSWER: {final_answer}")
        if success:
            print("ALL TESTS PASSED !#!#")
    finally:
        for client in clients:
            await client.close()


async def run_structured_summary_baseline(config: dict[str, Any]) -> None:
    start = time.time()
    logger = ExperimentLogger(config)
    task_spec, _, repair_tools = initialize_repair_workspace(config)
    clients: list[RecordingModelClient] = []

    config["model_config"] = describe_model_config("config.yaml")
    model_config = config["model_config"].get("config", {})
    if isinstance(model_config, dict):
        config["model"] = model_config.get("model", "config.yaml")
    max_items = int(config.get("structured_max_items_per_slot", 12))
    config["structured_max_items_per_slot"] = max_items

    base_task = read_prompt()
    agents = create_repair_agents(logger, repair_tools, task_spec, clients)
    structured_client = make_recording_client("structured_summary_runtime", logger, clients)
    structured_runtime = StructuredSummaryRuntime(logger, structured_client, max_items)
    ordered_agents = ["planner", "retriever", "coder", "tester", "reviewer"]

    try:
        try:
            for agent_name in ordered_agents:
                start_index = len(logger.raw_messages)
                agent_task = structured_runtime.build_agent_task(receiver=agent_name, base_task=base_task)
                await run_agent_once(agents[agent_name], agent_task, logger)
                new_messages = logger.raw_messages[start_index:]
                if agent_name != "reviewer":
                    await structured_runtime.update_after_turn(
                        sender=agent_name,
                        source_messages=new_messages,
                        base_task=base_task,
                    )
                    if log_local_early_stop(
                        logger=logger,
                        repair_tools=repair_tools,
                        task_spec=task_spec,
                        config=config,
                        source_agent=agent_name,
                    ):
                        break
        except Exception as exc:
            if not is_retryable_model_error(exc):
                raise
            logger.log_synthetic_message(
                turn_id=logger.next_turn("reviewer"),
                source="reviewer",
                role="assistant",
                content=f"FINAL_ANSWER: unresolved\nModel error after retries: {type(exc).__name__}: {str(exc)[-500:]}",
            )

        verification, success = evaluate_repair_result(
            logger=logger,
            repair_tools=repair_tools,
            task_spec=task_spec,
        )
        final_answer = "resolved" if success else "unresolved"
        metrics = metric_summary(
            logger=logger,
            start=start,
            success=success,
            verification=verification,
        )
        logger.save(final_answer, metrics)

        print(f"FINAL ANSWER: {final_answer}")
        if success:
            print("ALL TESTS PASSED !#!#")
    finally:
        for client in clients:
            await client.close()


async def run_vector_memory_baseline(config: dict[str, Any]) -> None:
    start = time.time()
    logger = ExperimentLogger(config)
    task_spec, _, repair_tools = initialize_repair_workspace(config)
    clients: list[RecordingModelClient] = []

    config["model_config"] = describe_model_config("config.yaml")
    model_config = config["model_config"].get("config", {})
    if isinstance(model_config, dict):
        config["model"] = model_config.get("model", "config.yaml")
    top_k = int(config.get("memory_top_k", 4))
    config["memory_top_k"] = top_k

    base_task = read_prompt()
    ordered_agents = ["planner", "retriever", "coder", "tester", "reviewer"]
    agents = create_repair_agents(logger, repair_tools, task_spec, clients)
    memory_runtime = VectorMemoryRuntime(logger, top_k)
    previous_agent: str | None = None

    try:
        try:
            for agent_name in ordered_agents:
                start_index = len(logger.raw_messages)
                agent_task = memory_runtime.build_agent_task(
                    receiver=agent_name,
                    base_task=base_task,
                    previous_agent=previous_agent,
                )
                await run_agent_once(agents[agent_name], agent_task, logger)
                new_messages = logger.raw_messages[start_index:]
                memory_runtime.add_messages(new_messages)
                previous_agent = agent_name
                if agent_name != "reviewer" and log_local_early_stop(
                    logger=logger,
                    repair_tools=repair_tools,
                    task_spec=task_spec,
                    config=config,
                    source_agent=agent_name,
                ):
                    break
        except Exception as exc:
            if not is_retryable_model_error(exc):
                raise
            logger.log_synthetic_message(
                turn_id=logger.next_turn("reviewer"),
                source="reviewer",
                role="assistant",
                content=f"FINAL_ANSWER: unresolved\nModel error after retries: {type(exc).__name__}: {str(exc)[-500:]}",
            )

        verification, success = evaluate_repair_result(
            logger=logger,
            repair_tools=repair_tools,
            task_spec=task_spec,
        )
        final_answer = "resolved" if success else "unresolved"
        metrics = metric_summary(
            logger=logger,
            start=start,
            success=success,
            verification=verification,
        )
        logger.save(final_answer, metrics)

        print(f"FINAL ANSWER: {final_answer}")
        if success:
            print("ALL TESTS PASSED !#!#")
    finally:
        for client in clients:
            await client.close()


async def run_vacth_full_baseline(config: dict[str, Any]) -> None:
    start = time.time()
    logger = ExperimentLogger(config)
    task_spec, _, repair_tools = initialize_repair_workspace(config)
    clients: list[RecordingModelClient] = []

    config["model_config"] = describe_model_config("config.yaml")
    model_config = config["model_config"].get("config", {})
    if isinstance(model_config, dict):
        config["model"] = model_config.get("model", "config.yaml")
    capsule_budget = int(config.get("capsule_token_budget", 900))
    config["capsule_token_budget"] = capsule_budget
    config["enable_thc"] = config_value_bool(config, "enable_thc", True)
    config["enable_cve"] = config_value_bool(config, "enable_cve", True)
    config["enable_paa"] = config_value_bool(config, "enable_paa", True)
    config["enable_provenance"] = config_value_bool(config, "enable_provenance", True)
    config["enable_reask"] = config_value_bool(config, "enable_reask", True)
    config["role_specific_routing"] = config_value_bool(config, "role_specific_routing", True)
    is_vacth_optimized = config["method"] == "vacth_optimized"
    skip_extraction_agents = config_value_str_set(
        config,
        "vacth_skip_extraction_agents",
        {"planner", "reviewer"} if is_vacth_optimized else set(),
    )
    config["vacth_skip_extraction_agents"] = sorted(skip_extraction_agents)
    config["vacth_compact_downstream_task"] = config_value_bool(
        config, "vacth_compact_downstream_task", is_vacth_optimized
    )
    config["vacth_skip_planner_capsule"] = config_value_bool(config, "vacth_skip_planner_capsule", is_vacth_optimized)
    config["vacth_compact_capsule"] = config_value_bool(config, "vacth_compact_capsule", is_vacth_optimized)
    config["vacth_hybrid_extraction"] = config_value_bool(config, "vacth_hybrid_extraction", is_vacth_optimized)
    config["vacth_adaptive_repair_enabled"] = config_value_bool(
        config, "vacth_adaptive_repair_enabled", is_vacth_optimized
    )
    config["vacth_adaptive_repair_max_attempts"] = adaptive_repair_max_attempts(config)
    config["vacth_adaptive_repair_progress_gate"] = adaptive_repair_progress_gate_enabled(config)
    config["vacth_source_patch_guard_max_attempts"] = source_patch_guard_max_attempts(config)
    role_token_budgets = config_value_int_map(
        config,
        "vacth_role_token_budgets",
        {"retriever": 350, "coder": 550, "tester": 500, "reviewer": 650} if is_vacth_optimized else {},
    )
    config["vacth_role_token_budgets"] = role_token_budgets
    config["thc_source_token_budget"] = int(config.get("thc_source_token_budget", 1800 if is_vacth_optimized else 0))
    config["thc_existing_item_limit"] = int(config.get("thc_existing_item_limit", 12))
    config["thc_max_items_per_turn"] = int(config.get("thc_max_items_per_turn", 6 if is_vacth_optimized else 0))
    config["vacth_capsule_max_item_chars"] = int(
        config.get("vacth_capsule_max_item_chars", 180 if is_vacth_optimized else 240)
    )
    config["vacth_include_test_evidence"] = config_value_bool(config, "vacth_include_test_evidence", is_vacth_optimized)
    config["vacth_test_evidence_max_chars"] = int(
        config.get("vacth_test_evidence_max_chars", 4500 if is_vacth_optimized else 0)
    )

    base_task = read_prompt()
    if config["vacth_include_test_evidence"]:
        evidence_prompt = focused_test_evidence_for_prompt(
            repair_tools,
            config["vacth_test_evidence_max_chars"],
        )
        if evidence_prompt:
            base_task = f"{base_task}\n\n{evidence_prompt}"
    ordered_agents = ["planner", "retriever", "coder", "tester", "reviewer"]
    agents = create_repair_agents(logger, repair_tools, task_spec, clients)
    extractor_client = make_recording_client("vacth_extractor", logger, clients) if config["enable_thc"] else None
    vacth_runtime = VACTHFullRuntime(
        logger=logger,
        extractor_client=extractor_client,
        task_id=config["task_id"],
        token_budget=capsule_budget,
        method_name=config["method"],
        enable_thc=config["enable_thc"],
        enable_cve=config["enable_cve"],
        enable_paa=config["enable_paa"],
        enable_provenance=config["enable_provenance"],
        enable_reask=config["enable_reask"],
        role_specific_routing=config["role_specific_routing"],
        skip_extraction_agents=skip_extraction_agents,
        compact_downstream_task=config["vacth_compact_downstream_task"],
        role_token_budgets=role_token_budgets,
        skip_planner_capsule=config["vacth_skip_planner_capsule"],
        compact_capsule=config["vacth_compact_capsule"],
        capsule_max_item_chars=config["vacth_capsule_max_item_chars"],
        hybrid_extraction=config["vacth_hybrid_extraction"],
        thc_source_token_budget=config["thc_source_token_budget"],
        thc_existing_item_limit=config["thc_existing_item_limit"],
        thc_max_items_per_turn=config["thc_max_items_per_turn"],
    )
    previous_agent: str | None = None
    adaptive_repairs = 0
    max_adaptive_repairs = adaptive_repair_max_attempts(config)
    source_patch_guards = 0
    max_source_patch_guards = source_patch_guard_max_attempts(config)

    try:
        for agent_name in ordered_agents:
            start_index = len(logger.raw_messages)
            agent_task = vacth_runtime.build_agent_task(
                receiver=agent_name,
                base_task=base_task,
                previous_agent=previous_agent,
            )
            await run_agent_once(agents[agent_name], agent_task, logger)
            new_messages = logger.raw_messages[start_index:]
            await vacth_runtime.update_after_turn(
                sender=agent_name,
                source_messages=new_messages,
                base_task=base_task,
            )
            previous_agent = agent_name
            if agent_name != "reviewer" and log_local_early_stop(
                logger=logger,
                repair_tools=repair_tools,
                task_spec=task_spec,
                config=config,
                source_agent=agent_name,
            ):
                break
            while agent_name == "coder" and is_vacth_optimized and source_patch_guards < max_source_patch_guards:
                guard_reason = source_patch_guard_reason(repair_tools)
                if guard_reason is None:
                    break
                source_patch_guards += 1
                config["vacth_source_patch_guard_attempts"] = source_patch_guards
                config["vacth_source_patch_guard_reason"] = guard_reason
                guard_start_index = len(logger.raw_messages)
                guard_task = vacth_runtime.build_agent_task(
                    receiver="coder",
                    base_task=base_task,
                    previous_agent=previous_agent,
                )
                guard_task += (
                    f"\n\nSource-patch guard: {guard_reason} "
                    "Use replace_text now to make the minimal source-code edit required by "
                    "the bug report. Do not spend this turn only searching or reading; if "
                    "the relevant line is already known, edit it. If you added imports or "
                    "helpers, make sure the repaired implementation actually uses them. "
                    "Do not change tests."
                )
                await run_agent_once(agents["coder"], guard_task, logger)
                guard_messages = logger.raw_messages[guard_start_index:]
                await vacth_runtime.update_after_turn(
                    sender="coder",
                    source_messages=guard_messages,
                    base_task=base_task,
                )
                previous_agent = "coder"
                if log_local_early_stop(
                    logger=logger,
                    repair_tools=repair_tools,
                    task_spec=task_spec,
                    config=config,
                    source_agent="coder_source_patch_guard",
                ):
                    break
            if should_stop_after_source_patch_guards(
                config=config,
                repair_tools=repair_tools,
                source_patch_guards=source_patch_guards,
                max_source_patch_guards=max_source_patch_guards,
            ):
                logger.log_synthetic_message(
                    turn_id=logger.next_turn("reviewer"),
                    source="reviewer",
                    role="assistant",
                    content="FINAL_ANSWER: unresolved",
                )
                break
            while (
                agent_name == "tester"
                and adaptive_repairs < max_adaptive_repairs
                and should_attempt_adaptive_repair(
                    config=config,
                    repair_tools=repair_tools,
                )
            ):
                adaptive_repairs += 1
                config["vacth_adaptive_repair_attempts"] = adaptive_repairs
                previous_failure_score = test_failure_score(repair_tools._last_test_result)
                for repair_agent in ("coder", "tester"):
                    repair_start_index = len(logger.raw_messages)
                    repair_task = vacth_runtime.build_agent_task(
                        receiver=repair_agent,
                        base_task=base_task,
                        previous_agent=previous_agent,
                    )
                    await run_agent_once(agents[repair_agent], repair_task, logger)
                    repair_messages = logger.raw_messages[repair_start_index:]
                    await vacth_runtime.update_after_turn(
                        sender=repair_agent,
                        source_messages=repair_messages,
                        base_task=base_task,
                    )
                    previous_agent = repair_agent
                    while (
                        repair_agent == "coder" and is_vacth_optimized and source_patch_guards < max_source_patch_guards
                    ):
                        guard_reason = source_patch_guard_reason(repair_tools)
                        if guard_reason is None:
                            break
                        source_patch_guards += 1
                        config["vacth_source_patch_guard_attempts"] = source_patch_guards
                        config["vacth_source_patch_guard_reason"] = guard_reason
                        guard_start_index = len(logger.raw_messages)
                        guard_task = vacth_runtime.build_agent_task(
                            receiver="coder",
                            base_task=base_task,
                            previous_agent=previous_agent,
                        )
                        guard_task += (
                            f"\n\nSource-patch guard before retesting: {guard_reason} "
                            "Use replace_text now to finish the minimal source-code edit "
                            "before tester runs the same failing tests again. If the previous "
                            "turn added imports or helpers, the implementation must actually "
                            "use them. Do not change tests."
                        )
                        await run_agent_once(agents["coder"], guard_task, logger)
                        guard_messages = logger.raw_messages[guard_start_index:]
                        await vacth_runtime.update_after_turn(
                            sender="coder",
                            source_messages=guard_messages,
                            base_task=base_task,
                        )
                        previous_agent = "coder"
                        if log_local_early_stop(
                            logger=logger,
                            repair_tools=repair_tools,
                            task_spec=task_spec,
                            config=config,
                            source_agent="coder_source_patch_guard",
                        ):
                            break
                    if should_stop_after_source_patch_guards(
                        config=config,
                        repair_tools=repair_tools,
                        source_patch_guards=source_patch_guards,
                        max_source_patch_guards=max_source_patch_guards,
                    ):
                        logger.log_synthetic_message(
                            turn_id=logger.next_turn("reviewer"),
                            source="reviewer",
                            role="assistant",
                            content="FINAL_ANSWER: unresolved",
                        )
                        break
                    if final_answer_from_messages(logger.raw_messages).startswith("resolved"):
                        break
                    if repair_agent != "reviewer" and log_local_early_stop(
                        logger=logger,
                        repair_tools=repair_tools,
                        task_spec=task_spec,
                        config=config,
                        source_agent=repair_agent,
                    ):
                        break
                if final_answer_from_messages(logger.raw_messages).startswith("resolved"):
                    break
                current_failure_score = test_failure_score(repair_tools._last_test_result)
                if previous_failure_score is not None:
                    config["vacth_adaptive_previous_failure_score"] = previous_failure_score
                if current_failure_score is not None:
                    config["vacth_adaptive_current_failure_score"] = current_failure_score
                if (
                    adaptive_repair_progress_gate_enabled(config)
                    and adaptive_repairs < max_adaptive_repairs
                    and (
                        previous_failure_score is None
                        or current_failure_score is None
                        or current_failure_score >= previous_failure_score
                    )
                ):
                    break
            if (
                agent_name == "tester"
                and config.get("method") == "vacth_optimized"
                and repair_tools_last_test_current(repair_tools)
                and not bool((repair_tools._last_test_result or {}).get("passed"))
                and (
                    adaptive_repairs >= max_adaptive_repairs
                    or not should_attempt_adaptive_repair(
                        config=config,
                        repair_tools=repair_tools,
                    )
                )
            ):
                logger.log_synthetic_message(
                    turn_id=logger.next_turn("reviewer"),
                    source="reviewer",
                    role="assistant",
                    content="FINAL_ANSWER: unresolved",
                )
                break
            if final_answer_from_messages(logger.raw_messages).startswith("resolved"):
                break

        verification, success = evaluate_repair_result(
            logger=logger,
            repair_tools=repair_tools,
            task_spec=task_spec,
        )
        final_answer = "resolved" if success else "unresolved"
        metrics = metric_summary(
            logger=logger,
            start=start,
            success=success,
            verification=verification,
        )
        logger.save(final_answer, metrics)

        print(f"FINAL ANSWER: {final_answer}")
        if success:
            print("ALL TESTS PASSED !#!#")
    finally:
        for client in clients:
            await client.close()


def run_deterministic_smoke(config: dict[str, Any]) -> None:
    start = time.time()
    logger = ExperimentLogger(config)
    task_spec, _, repair_tools = initialize_repair_workspace(config)
    history: list[dict[str, Any]] = []

    user = logger.log_synthetic_message(turn_id=0, source="user", role="user", content=read_prompt())
    history.append(user)
    for source, content in [
        ("planner", "Plan: inspect calculator.py, patch add, and run tests."),
        ("retriever", "Found calculator.add returning subtraction."),
        ("coder", "Patched calculator.add to return addition."),
        ("tester", "Unit tests passed."),
        ("reviewer", "FINAL_ANSWER: resolved"),
    ]:
        turn_id = logger.next_turn(source)
        logger.log_synthetic_visible_context(turn_id=turn_id, agent=source, history=history[-3:])
        message = logger.log_synthetic_message(turn_id=turn_id, source=source, role="assistant", content=content)
        history.append(message)
        if source == "coder":
            result = json.loads(repair_tools.replace_text("calculator.py", "return a - b", "return a + b"))
            logger.log_synthetic_tool_call(
                turn_id=turn_id,
                agent=source,
                tool="replace_text",
                arguments={
                    "path": "calculator.py",
                    "old": "return a - b",
                    "new": "return a + b",
                },
                result=result,
            )

    verification, success = evaluate_repair_result(
        logger=logger,
        repair_tools=repair_tools,
        task_spec=task_spec,
    )
    metrics = metric_summary(
        logger=logger,
        start=start,
        success=success,
        verification=verification,
    )
    final_answer = "resolved" if success else "unresolved"
    logger.save(final_answer, metrics)
    print(f"FINAL ANSWER: {final_answer}")
    if success:
        print("ALL TESTS PASSED !#!#")


def main() -> None:
    config = dict(EXPERIMENT_CONFIG)
    if config["method"] == "autogen_broadcast":
        asyncio.run(run_autogen_broadcast(config))
    elif config["method"] == "summary":
        asyncio.run(run_summary_baseline(config))
    elif config["method"] == "sliding_window":
        asyncio.run(run_sliding_window_baseline(config))
    elif config["method"] == "structured_summary":
        asyncio.run(run_structured_summary_baseline(config))
    elif config["method"] == "vector_memory":
        asyncio.run(run_vector_memory_baseline(config))
    elif config["method"].startswith("vacth_"):
        asyncio.run(run_vacth_full_baseline(config))
    else:
        run_deterministic_smoke(config)


if __name__ == "__main__":
    main()
