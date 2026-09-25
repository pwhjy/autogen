const state = {
  runs: [],
  filteredRuns: [],
  selectedRunId: "",
  selectedRun: null,
  activeTab: "messages",
};

const nodes = {
  resultsDir: document.querySelector("#results-dir"),
  runList: document.querySelector("#run-list"),
  runSearch: document.querySelector("#run-search"),
  methodFilter: document.querySelector("#method-filter"),
  refreshButton: document.querySelector("#refresh-button"),
  runMethod: document.querySelector("#run-method"),
  runTitle: document.querySelector("#run-title"),
  runPath: document.querySelector("#run-path"),
  statusPill: document.querySelector("#status-pill"),
  metricStrip: document.querySelector("#metric-strip"),
  messagesPanel: document.querySelector("#panel-messages"),
  contextsPanel: document.querySelector("#panel-contexts"),
  toolsPanel: document.querySelector("#panel-tools"),
  filesPanel: document.querySelector("#panel-files"),
  jsonView: document.querySelector("#json-view"),
  fileList: document.querySelector("#file-list"),
  fileContent: document.querySelector("#file-content"),
};

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return new Intl.NumberFormat().format(value);
  return String(value);
}

function classForSuccess(success) {
  if (success === true) return "ok";
  if (success === false) return "bad";
  return "neutral";
}

function labelForSuccess(success) {
  if (success === true) return "Success";
  if (success === false) return "Failed";
  return "Unknown";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pretty(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value ?? null, null, 2);
}

function compactJson(value) {
  return JSON.stringify(value ?? null, null, 2);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

async function loadRuns() {
  nodes.runList.innerHTML = `<div class="empty-state">Loading runs...</div>`;
  const payload = await fetchJson("/api/runs");
  state.runs = payload.runs || [];
  nodes.resultsDir.textContent = payload.results_dir || "Results";
  renderMethodFilter();
  applyRunFilters();
  if (state.runs.length > 0) {
    const selectedExists = state.runs.some((run) => run.id === state.selectedRunId);
    await selectRun(selectedExists ? state.selectedRunId : state.runs[0].id);
  } else {
    renderEmptyRun();
  }
}

function renderMethodFilter() {
  const current = nodes.methodFilter.value;
  const methods = [...new Set(state.runs.map((run) => run.method).filter(Boolean))].sort();
  nodes.methodFilter.innerHTML = `<option value="">All methods</option>`;
  for (const method of methods) {
    const option = document.createElement("option");
    option.value = method;
    option.textContent = method;
    nodes.methodFilter.append(option);
  }
  nodes.methodFilter.value = methods.includes(current) ? current : "";
}

function applyRunFilters() {
  const search = nodes.runSearch.value.trim().toLowerCase();
  const method = nodes.methodFilter.value;
  state.filteredRuns = state.runs.filter((run) => {
    const haystack = `${run.id} ${run.task_id} ${run.method} ${run.final_answer}`.toLowerCase();
    const matchesSearch = !search || haystack.includes(search);
    const matchesMethod = !method || run.method === method;
    return matchesSearch && matchesMethod;
  });
  renderRunList();
}

function renderRunList() {
  if (state.filteredRuns.length === 0) {
    nodes.runList.innerHTML = `<div class="empty-state">No matching runs.</div>`;
    return;
  }
  nodes.runList.innerHTML = state.filteredRuns
    .map((run) => {
      const active = run.id === state.selectedRunId ? " active" : "";
      const statusClass = classForSuccess(run.success);
      return `
        <button class="run-item${active}" type="button" data-run-id="${escapeHtml(run.id)}">
          <div class="run-item-title">${escapeHtml(run.task_id || run.id)}</div>
          <div class="run-item-path">${escapeHtml(run.id)}</div>
          <div class="run-item-meta">
            <span class="badge neutral">${escapeHtml(run.method || "method")}</span>
            <span class="badge ${statusClass}">${labelForSuccess(run.success)}</span>
            <span class="badge warn">${formatNumber(run.tool_calls)} tools</span>
          </div>
        </button>
      `;
    })
    .join("");
}

async function selectRun(runId) {
  state.selectedRunId = runId;
  renderRunList();
  const payload = await fetchJson(`/api/run?id=${encodeURIComponent(runId)}`);
  state.selectedRun = payload;
  renderRunDetail();
}

function renderEmptyRun() {
  state.selectedRun = null;
  nodes.runMethod.textContent = "No runs";
  nodes.runTitle.textContent = "No result.json files found";
  nodes.runPath.textContent = "Run an agbench task first, then reload this dashboard.";
  nodes.statusPill.className = "status-pill neutral";
  nodes.statusPill.textContent = "Idle";
  nodes.metricStrip.innerHTML = "";
  nodes.messagesPanel.innerHTML = `<div class="empty-state">No messages.</div>`;
  nodes.contextsPanel.innerHTML = `<div class="empty-state">No visible contexts.</div>`;
  nodes.toolsPanel.innerHTML = `<div class="empty-state">No tool calls.</div>`;
  nodes.fileList.innerHTML = "";
  nodes.fileContent.textContent = "No files.";
  nodes.jsonView.textContent = "";
}

function renderRunDetail() {
  const detail = state.selectedRun;
  const result = detail.result;
  const metrics = result.metrics || {};
  const statusClass = classForSuccess(metrics.success);

  nodes.runMethod.textContent = result.method || "method";
  nodes.runTitle.textContent = result.task_id || detail.id;
  nodes.runPath.textContent = detail.run_dir || detail.id;
  nodes.statusPill.className = `status-pill ${statusClass}`;
  nodes.statusPill.textContent = `${labelForSuccess(metrics.success)}: ${result.final_answer || "-"}`;

  renderMetrics(metrics);
  renderMessages(result.raw_messages || []);
  renderContexts(result.visible_contexts || []);
  renderTools(result.tool_calls || []);
  renderFiles(detail.files || []);
  nodes.jsonView.textContent = compactJson(result);
}

function renderMetrics(metrics) {
  const items = [
    ["Turns", metrics.turns],
    ["Tool calls", metrics.tool_calls],
    ["Visible contexts", metrics.visible_contexts],
    ["Raw messages", metrics.raw_messages],
    ["Tokens", metrics.estimated_tokens],
    ["Wall time", metrics.wall_time_sec ? `${metrics.wall_time_sec}s` : "-"],
  ];
  if (metrics.summary_count) {
    items.push(["Summaries", metrics.summary_count]);
    items.push(["Summary ratio", metrics.summary_compression_ratio]);
  }
  if (metrics.structured_update_count) {
    items.push(["JSON states", metrics.structured_update_count]);
    items.push(["State items", metrics.structured_state_item_count]);
    items.push(["Schema", metrics.structured_schema_valid ? "valid" : "invalid"]);
  }
  if (metrics.context_window_messages) {
    items.push(["Window", metrics.context_window_messages]);
    items.push(["Dropped", metrics.dropped_message_count]);
  }
  if (metrics.retrieval_count) {
    items.push(["Memory", metrics.memory_item_count]);
    items.push(["Retrievals", metrics.retrieval_count]);
    items.push(["Avg score", metrics.avg_retrieval_score]);
  }
  if (metrics.vacth_capsule_count) {
    items.push(["Capsules", metrics.vacth_capsule_count]);
    items.push(["VACTH items", metrics.state_items]);
    items.push(["CVE routes", metrics.vacth_cve_routing_count]);
    items.push(["Capsule tokens", metrics.vacth_avg_capsule_tokens]);
    items.push(["Heuristic THC", metrics.vacth_heuristic_extraction_count]);
    items.push(["Reasks", metrics.vacth_reask_count]);
    items.push(["THC", metrics.vacth_enable_thc ? "on" : "off"]);
    items.push(["CVE", metrics.vacth_enable_cve ? "on" : "off"]);
    items.push(["PAA", metrics.vacth_enable_paa ? "on" : "off"]);
    items.push(["Provenance", metrics.vacth_enable_provenance ? "on" : "off"]);
    items.push(["Role routing", metrics.vacth_role_specific_routing ? "on" : "off"]);
  }
  if (metrics.mechanism_task_type) {
    items.push(["Mechanism", metrics.mechanism_task_type]);
    items.push(["Sample", metrics.mechanism_sample_id]);
    items.push(["Mechanism score", metrics.mechanism_overall_score]);
    items.push(["Slot F1", metrics.mechanism_slot_f1]);
    items.push(["Evidence F1", metrics.mechanism_evidence_f1]);
    items.push(["Route recall", metrics.mechanism_routing_recall]);
    items.push(["Route NDCG", metrics.mechanism_routing_ndcg]);
    items.push(["Active F1", metrics.mechanism_active_item_f1]);
    items.push(["Edge F1", metrics.mechanism_edge_f1]);
    items.push(["Reask target", metrics.mechanism_reask_target_accuracy]);
  }
  nodes.metricStrip.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="metric">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${formatNumber(value)}</div>
        </div>
      `,
    )
    .join("");
}

function renderMessages(messages) {
  if (messages.length === 0) {
    nodes.messagesPanel.innerHTML = `<div class="empty-state">No messages.</div>`;
    return;
  }
  nodes.messagesPanel.innerHTML = `
    <div class="timeline">
      ${messages
        .map((message) => {
          const roleClass = message.role === "assistant" ? "ok" : message.role === "event" ? "warn" : "neutral";
          return `
            <article class="message-row">
              <header class="row-head">
                <div class="row-title">
                  <strong>${escapeHtml(message.source || "unknown")}</strong>
                  <span class="badge ${roleClass}">${escapeHtml(message.role || "message")}</span>
                  <span class="badge neutral">${escapeHtml(message.type || "")}</span>
                </div>
                <span class="muted">turn ${formatNumber(message.turn_id)}</span>
              </header>
              <div class="row-content">${escapeHtml(message.content || "")}</div>
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderContexts(contexts) {
  if (contexts.length === 0) {
    nodes.contextsPanel.innerHTML = `<div class="empty-state">No visible contexts.</div>`;
    return;
  }
  nodes.contextsPanel.innerHTML = `
    <div class="context-list">
      ${contexts
        .map((context) => {
          const messages = context.context || [];
          const tools = context.tools || [];
          return `
            <article class="context-row">
              <header class="row-head">
                <div class="row-title">
                  <strong>${escapeHtml(context.agent || "agent")}</strong>
                  <span class="badge neutral">${escapeHtml(context.mode || "create")}</span>
                  <span class="badge warn">${formatNumber(messages.length)} messages</span>
                  <span class="badge neutral">${formatNumber(tools.length)} tools</span>
                </div>
                <span class="muted">turn ${formatNumber(context.turn_id)}</span>
              </header>
              <div class="context-messages">
                ${messages
                  .map(
                    (message) => `
                      <div class="context-message">
                        <div class="row-title">
                          <strong>${escapeHtml(message.source || "")}</strong>
                          <span class="badge neutral">${escapeHtml(message.role || "")}</span>
                        </div>
                        <pre class="json-fragment">${escapeHtml(pretty(message.content))}</pre>
                      </div>
                    `,
                  )
                  .join("")}
              </div>
              ${tools.length ? `<pre class="json-fragment">${escapeHtml(compactJson(tools))}</pre>` : ""}
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderTools(toolCalls) {
  if (toolCalls.length === 0) {
    nodes.toolsPanel.innerHTML = `<div class="empty-state">No tool calls.</div>`;
    return;
  }
  nodes.toolsPanel.innerHTML = `
    <div class="tool-list">
      ${toolCalls
        .map(
          (call, index) => `
            <article class="tool-row">
              <header class="row-head">
                <div class="row-title">
                  <strong>${index + 1}. ${escapeHtml(call.tool || "tool")}</strong>
                  <span class="badge neutral">${escapeHtml(call.agent || "agent")}</span>
                  <span class="badge ${call.result?.is_error ? "bad" : "ok"}">${call.result?.is_error ? "Error" : "OK"}</span>
                </div>
                <span class="muted">turn ${formatNumber(call.turn_id)}</span>
              </header>
              <pre class="json-fragment">${escapeHtml(
                compactJson({
                  arguments: call.arguments || {},
                  result: call.result || {},
                  call_id: call.call_id || "",
                }),
              )}</pre>
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderFiles(files) {
  if (files.length === 0) {
    nodes.fileList.innerHTML = `<div class="empty-state">No files.</div>`;
    nodes.fileContent.textContent = "No files.";
    return;
  }
  nodes.fileList.innerHTML = files
    .map(
      (file) => `
        <button class="file-item" type="button" data-path="${escapeHtml(file.path)}">
          ${escapeHtml(file.path)}
        </button>
      `,
    )
    .join("");
  loadFile(files[0].path);
}

async function loadFile(path) {
  if (!state.selectedRunId) return;
  for (const button of nodes.fileList.querySelectorAll(".file-item")) {
    button.classList.toggle("active", button.dataset.path === path);
  }
  nodes.fileContent.textContent = "Loading file...";
  const response = await fetch(`/api/text?id=${encodeURIComponent(state.selectedRunId)}&path=${encodeURIComponent(path)}`);
  if (!response.ok) {
    nodes.fileContent.textContent = await response.text();
    return;
  }
  nodes.fileContent.textContent = await response.text();
}

function setActiveTab(tabName) {
  state.activeTab = tabName;
  for (const button of document.querySelectorAll(".tab")) {
    button.classList.toggle("active", button.dataset.tab === tabName);
  }
  for (const panel of document.querySelectorAll(".panel")) {
    panel.classList.toggle("active", panel.id === `panel-${tabName}`);
  }
}

nodes.refreshButton.addEventListener("click", () => {
  loadRuns().catch((error) => {
    nodes.runList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  });
});

nodes.runSearch.addEventListener("input", applyRunFilters);
nodes.methodFilter.addEventListener("change", applyRunFilters);

nodes.runList.addEventListener("click", (event) => {
  const button = event.target.closest(".run-item");
  if (!button) return;
  selectRun(button.dataset.runId).catch((error) => {
    nodes.messagesPanel.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  });
});

nodes.fileList.addEventListener("click", (event) => {
  const button = event.target.closest(".file-item");
  if (!button) return;
  loadFile(button.dataset.path);
});

document.querySelector(".tabs").addEventListener("click", (event) => {
  const button = event.target.closest(".tab");
  if (!button) return;
  setActiveTab(button.dataset.tab);
});

loadRuns().catch((error) => {
  nodes.runList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
});
