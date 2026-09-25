import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DASHBOARD_DIR = BENCHMARK_DIR / "Dashboard"
RESULTS_DIR = BENCHMARK_DIR / "Results"

SIDE_FILES = [
    "result.json",
    "raw_messages.json",
    "visible_contexts.json",
    "tool_calls.json",
    "summaries.json",
    "structured_summary.json",
    "capsules.json",
    "vacth_extractions.json",
    "vector_memory.json",
    "state_items.json",
    "routing_decisions.json",
    "provenance_graph.json",
    "mechanism_gold.json",
    "metrics.csv",
    "console_log.txt",
    "prompt.txt",
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_run_dir(run_id: str) -> Path:
    run_dir = (RESULTS_DIR / run_id).resolve()
    try:
        run_dir.relative_to(RESULTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Invalid run id: {run_id}") from exc
    if not (run_dir / "result.json").is_file():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return run_dir


def _safe_text_file(run_dir: Path, relative_path: str) -> Path:
    target = (run_dir / relative_path).resolve()
    try:
        target.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"Invalid file path: {relative_path}") from exc
    if not target.is_file():
        raise FileNotFoundError(relative_path)
    return target


def _run_id_from_result(result_path: Path) -> str:
    return result_path.parent.relative_to(RESULTS_DIR).as_posix()


def _run_summary(result_path: Path) -> dict[str, Any]:
    run_id = _run_id_from_result(result_path)
    result = _read_json(result_path)
    metrics = result.get("metrics", {})
    return {
        "id": run_id,
        "task_id": result.get("task_id", ""),
        "method": result.get("method", ""),
        "final_answer": result.get("final_answer", ""),
        "success": metrics.get("success"),
        "turns": metrics.get("turns"),
        "tool_calls": metrics.get("tool_calls"),
        "estimated_tokens": metrics.get("estimated_tokens"),
        "wall_time_sec": metrics.get("wall_time_sec"),
        "raw_messages": len(result.get("raw_messages", [])),
        "visible_contexts": len(result.get("visible_contexts", [])),
        "result_path": result_path.relative_to(BENCHMARK_DIR).as_posix(),
        "modified": result_path.stat().st_mtime,
    }


def _list_runs() -> dict[str, Any]:
    result_files = (
        sorted(RESULTS_DIR.rglob("result.json")) if RESULTS_DIR.exists() else []
    )
    runs = []
    for result_path in result_files:
        if "__pycache__" in result_path.parts:
            continue
        try:
            runs.append(_run_summary(result_path))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    runs.sort(key=lambda item: item["modified"], reverse=True)
    return {"runs": runs, "results_dir": RESULTS_DIR.as_posix()}


def _list_run_files(run_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for name in SIDE_FILES:
        path = run_dir / name
        if path.is_file():
            files.append(
                {
                    "path": name,
                    "kind": "log"
                    if name.endswith((".json", ".csv", ".txt"))
                    else "file",
                    "size": path.stat().st_size,
                }
            )
    workspace = run_dir / "workspace"
    if workspace.is_dir():
        for path in sorted(workspace.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(run_dir).as_posix(),
                        "kind": "workspace",
                        "size": path.stat().st_size,
                    }
                )
    return files


def _run_detail(run_id: str) -> dict[str, Any]:
    run_dir = _safe_run_dir(run_id)
    result = _read_json(run_dir / "result.json")
    return {
        "id": run_id,
        "result": result,
        "files": _list_run_files(run_dir),
        "run_dir": run_dir.as_posix(),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "VACTHBenchDashboard/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        try:
            self._send_static(parsed.path, head_only=True)
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc), head_only=True)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/runs":
                self._send_json(_list_runs())
                return
            if parsed.path == "/api/run":
                query = parse_qs(parsed.query)
                self._send_json(_run_detail(query.get("id", [""])[0]))
                return
            if parsed.path == "/api/text":
                query = parse_qs(parsed.query)
                run_dir = _safe_run_dir(query.get("id", [""])[0])
                path = _safe_text_file(run_dir, query.get("path", [""])[0])
                self._send_text(path.read_text(encoding="utf-8", errors="replace"))
                return
            self._send_static(parsed.path)
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _send_json(self, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, value: str) -> None:
        payload = value.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(
        self, status: HTTPStatus, message: str, *, head_only: bool = False
    ) -> None:
        payload = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if head_only:
            return
        self.wfile.write(payload)

    def _send_static(self, path: str, *, head_only: bool = False) -> None:
        relative = "index.html" if path in {"", "/"} else unquote(path.lstrip("/"))
        target = (DASHBOARD_DIR / relative).resolve()
        try:
            target.relative_to(DASHBOARD_DIR.resolve())
        except ValueError as exc:
            raise FileNotFoundError(relative) from exc
        if not target.is_file():
            raise FileNotFoundError(relative)
        content_type = (
            mimetypes.guess_type(target.as_posix())[0] or "application/octet-stream"
        )
        payload = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if head_only:
            return
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the VACTHBench dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.chdir(BENCHMARK_DIR)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.quiet = args.quiet  # type: ignore[attr-defined]
    print(f"VACTHBench dashboard: http://{args.host}:{args.port}")
    print(f"Reading results from: {RESULTS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
