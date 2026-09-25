# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Repository Status

AutoGen is **in maintenance mode** — no new features. New users should start with [Microsoft Agent Framework](https://github.com/microsoft/agent-framework). The Python packages are at version 0.7.5 and are community-managed going forward.

## Environment Setup

This is a polyglot repo with **Python** (primary) and **.NET** implementations. The Python packages use `uv` workspaces; the .NET side uses `dotnet`.

```bash
# Python — run from python/
cd python
uv sync --all-extras       # creates .venv, installs all packages in editable mode
source .venv/bin/activate

# .NET — run from dotnet/
cd dotnet
export PATH="$HOME/.dotnet:$PATH"
dotnet restore
dotnet build --configuration Release
```

**Always use `uv`**, never pip/conda. The workspace members are in `python/packages/*` and defined in `python/pyproject.toml` under `[tool.uv.workspace]`.

## Key Commands (Python)

All commands require an activated virtual environment and are run from `python/`. They use `poe` (poethepoet) tasks defined in `python/pyproject.toml` and inherited per-package via `python/shared_tasks.toml`.

| Command | What it does | Timing |
|---------|-------------|--------|
| `poe format` | Ruff formatting (line-length 120) | <1s |
| `poe lint` | Ruff linting (E, F, W, B, Q, I, ASYNC, T20) | <1s |
| `poe test` | Run all tests across packages with coverage | varies |
| `poe mypy` | Strict mypy type checking | ~6 min |
| `poe pyright` | Strict pyright type checking | ~40s |
| `poe check` | Runs format + lint + pyright + mypy + test + docs + samples | ~7+ min |
| `poe docs-build` | Build Sphinx docs from `docs/src/` | ~1 min |
| `poe docs-serve` | Auto-rebuild docs on port 8000 | ongoing |

**Run tasks in a single package:**
```bash
poe --directory ./packages/autogen-core test      # run tests for one package
poe --directory ./packages/autogen-agentchat test
poe --directory ./packages/autogen-ext test
```

Tests run with `pytest -n auto` for parallel execution. The `autogen-ext` package serializes with `-n 1` because Playwright is not parallel-safe.

## Architecture: Three-Layer Python Package Design

The Python side is a tiered architecture across three main packages:

### 1. `autogen-core` — Foundation layer
- **Agent runtime** (`AgentRuntime` protocol): message passing between agents via `send_message` (1:1) and `publish_message` (1:many on topics). There's a single-threaded reference implementation (`SingleThreadedAgentRuntime`) and a gRPC-based distributed runtime in `autogen-ext`.
- **Agent base** (`Agent` protocol): an agent is identified by `AgentId` (type + key), handles messages, and lives in a runtime. Agents are registered with subscriptions to topics.
- **Component model** (`ComponentBase[ConfigModel]`): every configurable thing (agent, model client, tool, memory, workbench) is a typed component with a Pydantic config. This enables declarative composition and schema generation.
- **Model context**: buffer-based chat completion context implementations (`BufferedChatCompletionContext`, `TokenLimitedChatCompletionContext`, etc.)
- **Memory**: simple in-memory implementations (`ListMemory`, `BaseMemory`)
- **Telemetry**: OpenTelemetry-based tracing with gen AI conventions

### 2. `autogen-agentchat` — Conversation layer
Built on `autogen-core` but does NOT use the message-passing runtime directly at runtime. Instead, agents communicate synchronously within a team.

- **`ChatAgent`** (abstract): the core abstraction. Key operations:
  - `on_messages(messages) -> Response` — handle incoming messages, return a `Response` with a `chat_message`
  - `on_messages_stream(messages) -> AsyncGenerator` — streaming variant
  - `on_reset`, `on_pause`, `on_resume` — lifecycle
  - `save_state` / `load_state` — persistence
- **Agents**: `AssistantAgent` (LLM-backed), `CodeExecutorAgent`, `SocietyOfMindAgent` (nested team as agent), `UserProxyAgent`, `MessageFilterAgent`
- **`Team`** (abstract): orchestrates multiple `ChatAgent`s. Has `run(task)` / `run_stream(task)` via `TaskRunner`.
- **Teams**: `RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`, `MagneticOneGroupChat` — all in `teams/_group_chat/`
- **Handoff** / **Termination**: control flow between agents and when a team run ends
- **Messages**: `BaseChatMessage` types include `TextMessage`, `MultiModalMessage`, `ToolCallMessage`, `ToolCallResultMessage`, `HandoffMessage`, etc.
- **Tools**: `Tool` protocol wrapping functions for agent use. Workbench (`Workbench`) provides dynamic tool listing.

### 3. `autogen-ext` — Ecosystem integrations
Extensible implementations behind `autogen-core` interfaces. All dependencies are optional (opt-in via extras like `[openai]`, `[docker]`, `[grpc]`).

- **Models**: `OpenAIChatCompletionClient`, `AnthropicChatCompletionClient`, Ollama, llama.cpp, Semantic Kernel, Azure, Gemini, `ReplayChatCompletionClient` (for testing)
- **Runtimes**: gRPC-based distributed agent runtime
- **Tools**: MCP server integration, LangChain tools, GraphRAG, HTTP tools, Azure tools
- **Code executors**: Docker-based, Jupyter-based, local command-line
- **Memory**: ChromaDB, Mem0, Redis, DiskCache integrations
- **Agents**: Web surfer, file surfer, video surfer (specialized multimodal agents)
- **Auth**: token-based authentication providers

### Supporting packages
- **`autogen-studio`**: web-based IDE (FastAPI + React frontend) for building and running agent workflows
- **`agbench`**: benchmarking suite for agent performance
- **`autogen-test-utils`**: shared test utilities used across package test suites
- **`component-schema-gen`**: generates JSON Schema from component configs

## Documentation

Docstrings follow Google style with Sphinx RST. When adding public API, include `Args`, `Returns`, `Raises` sections plus executable code examples using `.. code-block:: python`. Doc code blocks are validated by `poe docs-check-examples` using pyright.

## Testing

- Use `pytest` with `pytest-asyncio` (session-scoped event loop).
- For model clients, use `autogen_ext.models.replay.ReplayChatCompletionClient` to simulate LLM responses without API calls.
- Tests requiring external APIs should be skipped when credentials aren't available.
- gRPC tests are marked with `@pytest.mark.grpc` and run separately.

## .NET Side

The `dotnet/` directory contains the .NET implementation — a separate ecosystem with its own solution and projects. It requires both .NET 8.0 runtime and .NET 9.0 SDK. Key entry points: `dotnet restore`, `dotnet build --configuration Release`, `dotnet test --filter "Category=UnitV2"`.

## Git Conventions

- All `autogen-*` Python packages are versioned together (current: 0.7.5)
- Minor version bump (0.X.0) for breaking changes; patch (0.0.X) for features/fixes
