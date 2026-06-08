const COLORS = ["#b8f271", "#62c9e8", "#ffad66", "#b89cff", "#54d6ae", "#f87575"];
const state = {
  catalog: null,
  session: null,
  frame: 0,
  playing: false,
  timer: null,
  runId: 0,
};

const elements = {
  form: document.querySelector("#simulation-form"),
  runButton: document.querySelector("#run-button"),
  formError: document.querySelector("#form-error"),
  policyList: document.querySelector("#policy-list"),
  policyCount: document.querySelector("#policy-count"),
  workload: document.querySelector("#workload"),
  workloadDescription: document.querySelector("#workload-description"),
  requestCount: document.querySelector("#request-count"),
  capacityBlocks: document.querySelector("#capacity-blocks"),
  capacityTokens: document.querySelector("#capacity-tokens"),
  blockSizeTokens: document.querySelector("#block-size-tokens"),
  seed: document.querySelector("#seed"),
  sourceLabel: document.querySelector("#source-label"),
  sourceDescription: document.querySelector("#source-description"),
  sessionTitle: document.querySelector("#session-title"),
  sessionDescription: document.querySelector("#session-description"),
  runStatus: document.querySelector("#run-status"),
  summaryGrid: document.querySelector("#summary-grid"),
  requestHeading: document.querySelector("#request-heading"),
  requestFacts: document.querySelector("#request-facts"),
  timeline: document.querySelector("#timeline"),
  chart: document.querySelector("#chart"),
  chartLegend: document.querySelector("#chart-legend"),
  chartMetric: document.querySelector("#chart-metric"),
  cacheComparison: document.querySelector("#cache-comparison"),
  eventTable: document.querySelector("#event-table"),
  playButton: document.querySelector("#play-button"),
  resetButton: document.querySelector("#reset-button"),
  stepBackButton: document.querySelector("#step-back-button"),
  stepForwardButton: document.querySelector("#step-forward-button"),
  speed: document.querySelector("#speed"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function percent(value) {
  return `${(100 * value).toFixed(1)}%`;
}

function number(value, digits = 0) {
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function policyColor(index) {
  return COLORS[index % COLORS.length];
}

function selectedPolicies() {
  return [...elements.policyList.querySelectorAll("input:checked")].map((input) => input.value);
}

function updatePolicyCount() {
  const selected = selectedPolicies();
  elements.policyCount.textContent = `${selected.length} / 6`;
  for (const input of elements.policyList.querySelectorAll("input:not(:checked)")) {
    input.disabled = selected.length >= 6;
  }
}

function updateCapacityTokens() {
  const capacity = Number(elements.capacityBlocks.value);
  const blockSize = Number(elements.blockSizeTokens.value);
  elements.capacityTokens.textContent = number(capacity * blockSize);
}

function populateCatalog(catalog) {
  state.catalog = catalog;
  elements.sourceLabel.textContent = catalog.source.label;
  elements.sourceDescription.textContent = catalog.source.description;
  elements.workload.innerHTML = catalog.workloads
    .map((workload) => `<option value="${escapeHtml(workload.id)}">${escapeHtml(workload.label)}</option>`)
    .join("");
  elements.policyList.innerHTML = catalog.policies
    .map((policy, index) => `
      <label class="policy-option ${policy.promoted ? "promoted" : ""}">
        <input type="checkbox" value="${escapeHtml(policy.id)}" ${policy.default_selected ? "checked" : ""}>
        <span class="policy-copy">
          <span class="policy-title-row">
            <strong>${escapeHtml(policy.label)}</strong>
            ${policy.promoted ? '<b class="policy-badge">Promoted</b>' : ""}
          </span>
          <small class="policy-status">${escapeHtml(policy.status)}</small>
          <small class="policy-description">${escapeHtml(policy.description)}</small>
        </span>
        <i style="background:${policyColor(index)}"></i>
      </label>
    `)
    .join("");

  const defaults = catalog.defaults;
  elements.workload.value = defaults.workload;
  elements.requestCount.value = defaults.request_count;
  elements.capacityBlocks.value = defaults.capacity_blocks;
  elements.blockSizeTokens.value = defaults.block_size_tokens;
  elements.seed.value = defaults.seed;
  for (const [name, range] of Object.entries(catalog.limits)) {
    if (name === "max_policies") continue;
    const input = document.querySelector(`#${name.replaceAll("_", "-")}`);
    input.min = range[0];
    input.max = range[1];
  }
  updateWorkloadDescription();
  updateCapacityTokens();
  updatePolicyCount();
}

function updateWorkloadDescription() {
  if (!state.catalog) return;
  const workload = state.catalog.workloads.find((item) => item.id === elements.workload.value);
  elements.workloadDescription.textContent = workload?.description || "";
  if (!state.session || state.session.config.workload !== elements.workload.value) {
    elements.sessionTitle.textContent = workload?.label || "Unknown workload";
    elements.sessionDescription.textContent = workload?.description || "";
  }
}

function workloadLabel(workloadId) {
  return state.catalog?.workloads.find((workload) => workload.id === workloadId)?.label || workloadId;
}

function workloadDescription(workloadId) {
  return state.catalog?.workloads.find((workload) => workload.id === workloadId)?.description || "";
}

function humanizeIdentifier(value) {
  return value.replaceAll("_", " ");
}

function observedTrafficDescription(events) {
  const phaseCounts = new Map();
  for (const event of events) {
    phaseCounts.set(event.request_type, (phaseCounts.get(event.request_type) || 0) + 1);
  }
  const phases = [...phaseCounts.entries()]
    .sort((left, right) => right[1] - left[1])
    .map(([phase, count]) => `${humanizeIdentifier(phase)} × ${count}`)
    .join(", ");
  return phases ? `Observed request phases: ${phases}.` : "";
}

function setStatus(kind, title, detail) {
  elements.runStatus.className = `run-status ${kind}`;
  elements.runStatus.querySelector("strong").textContent = title;
  elements.runStatus.querySelector("small").textContent = detail;
}

function markConfigurationPending() {
  if (state.session) setStatus("", "Pending", "Run to apply changes");
}

async function runSimulation() {
  const runId = ++state.runId;
  stopPlayback();
  elements.formError.textContent = "";
  const policies = selectedPolicies();
  if (!policies.length) {
    elements.formError.textContent = "Select at least one policy.";
    elements.runButton.disabled = false;
    elements.runButton.querySelector("span").textContent = "Run comparison";
    setStatus("", "Pending", "Select at least one policy");
    return;
  }
  const payload = {
    policies,
    workload: elements.workload.value,
    request_count: Number(elements.requestCount.value),
    capacity_blocks: Number(elements.capacityBlocks.value),
    block_size_tokens: Number(elements.blockSizeTokens.value),
    seed: Number(elements.seed.value),
  };
  elements.runButton.disabled = true;
  elements.runButton.querySelector("span").textContent = "Simulating...";
  setStatus("running", "Running", `${policies.length} policies`);
  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Simulation failed");
    if (runId !== state.runId) return;
    state.session = result;
    state.frame = 0;
    elements.sessionTitle.textContent = workloadLabel(result.config.workload);
    elements.sessionDescription.textContent = [
      workloadDescription(result.config.workload),
      observedTrafficDescription(result.policies[0]?.events || []),
    ].filter(Boolean).join(" ");
    setStatus("complete", "Complete", `${result.config.request_count} requests replayable`);
    renderAll();
  } catch (error) {
    if (runId !== state.runId) return;
    elements.formError.textContent = error.message;
    setStatus("", "Error", "Review configuration");
  } finally {
    if (runId === state.runId) {
      elements.runButton.disabled = false;
      elements.runButton.querySelector("span").textContent = "Run comparison";
    }
  }
}

function renderAll() {
  if (!state.session) return;
  renderSummary();
  renderRequest();
  renderTimeline();
  renderChart();
  renderCaches();
  renderEventTable();
}

function renderSummary() {
  const ranked = [...state.session.policies].sort(
    (left, right) => right.summary.token_hit_rate - left.summary.token_hit_rate
  );
  elements.summaryGrid.innerHTML = ranked
    .map((policy, rank) => {
      const index = state.session.policies.findIndex((item) => item.id === policy.id);
      return `
        <article class="summary-card ${policy.promoted ? "promoted" : ""}" style="--policy-color:${policyColor(index)}">
          <span class="rank">#${rank + 1}</span>
          <p class="eyebrow">${escapeHtml(policy.status)}</p>
          <h3>${escapeHtml(policy.label)}</h3>
          ${policy.promoted ? `
            <p class="benchmark-score">
              Discovery selection <strong>${number(policy.benchmark_selection_score, 3)}</strong>
              · ${escapeHtml(policy.benchmark_context)}
            </p>
          ` : ""}
          <div class="primary-stat">
            <strong>${percent(policy.summary.token_hit_rate)}</strong>
            <span>token hit rate</span>
          </div>
          <div class="micro-stats">
            <div><span>P95 latency</span><strong>${number(policy.summary.p95_latency_proxy, 1)}</strong></div>
            <div><span>Evictions</span><strong>${number(policy.summary.eviction_count)}</strong></div>
            <div><span>Admissions</span><strong>${number(policy.summary.admission_count)}</strong></div>
          </div>
        </article>
      `;
    })
    .join("");
}

function currentReferenceEvent() {
  return state.session?.policies[0]?.events[state.frame];
}

function renderRequest() {
  const event = currentReferenceEvent();
  if (!event) return;
  elements.requestHeading.textContent = `Request ${state.frame + 1} of ${state.session.config.request_count}`;
  elements.requestFacts.innerHTML = `
    <span>Traffic <strong>${escapeHtml(workloadLabel(state.session.config.workload))}</strong></span>
    <span>Request phase <strong>${escapeHtml(event.request_type)}</strong></span>
    <span>Tenant <strong>${event.tenant_id}</strong></span>
    <span>Priority <strong>${event.priority}</strong></span>
    <span>Prompt <strong>${event.prompt_tokens} tokens / ${event.prompt_blocks} blocks</strong></span>
    <span>Logical time <strong>${event.now}</strong></span>
  `;
}

function renderTimeline() {
  const events = state.session.policies[0].events;
  elements.timeline.innerHTML = events
    .map((event, index) => {
      const rate = event.hit_tokens / Math.max(1, event.prompt_tokens);
      const outcome = rate >= 0.75 ? "hit" : rate > 0 ? "partial" : "";
      const position = index === state.frame ? "current" : index < state.frame ? "past" : "";
      const priority = event.priority > 0 ? "priority" : "";
      return `<button class="${outcome} ${position} ${priority}" data-frame="${index}" title="Request ${index + 1}: ${percent(rate)} hit"></button>`;
    })
    .join("");
  const current = elements.timeline.querySelector(".current");
  current?.scrollIntoView({ block: "nearest", inline: "center" });
}

function metricValue(event, metric) {
  if (metric === "occupancy") return event.resident_blocks / Math.max(1, event.capacity_blocks);
  return event[metric];
}

function metricLabel(metric, value) {
  if (metric === "cumulative_token_hit_rate" || metric === "occupancy") return percent(value);
  return number(value, metric === "latency_proxy" ? 1 : 0);
}

function linePath(events, metric, maxValue, endIndex) {
  const width = 900;
  const height = 220;
  const insetX = 44;
  const insetY = 16;
  const plotWidth = width - insetX - 15;
  const plotHeight = height - insetY - 28;
  return events.slice(0, endIndex + 1).map((event, index) => {
    const x = insetX + (index / Math.max(1, events.length - 1)) * plotWidth;
    const y = insetY + plotHeight - (metricValue(event, metric) / Math.max(maxValue, 0.0001)) * plotHeight;
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
}

function renderChart() {
  const metric = elements.chartMetric.value;
  const allValues = state.session.policies.flatMap((policy) =>
    policy.events.map((event) => metricValue(event, metric))
  );
  const maxValue = ["cumulative_token_hit_rate", "occupancy"].includes(metric)
    ? 1
    : Math.max(...allValues, 1) * 1.08;
  const frameX = 44 + (state.frame / Math.max(1, state.session.config.request_count - 1)) * 841;
  const guides = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const y = 16 + (1 - ratio) * 176;
    return `
      <line x1="44" x2="885" y1="${y}" y2="${y}" stroke="rgba(204,244,224,.08)" />
      <text x="37" y="${y + 3}" text-anchor="end" fill="#597267" font-size="8">${metricLabel(metric, maxValue * ratio)}</text>
    `;
  }).join("");
  const lines = state.session.policies.map((policy, index) => `
    <path d="${linePath(policy.events, metric, maxValue, policy.events.length - 1)}"
      fill="none" stroke="${policyColor(index)}" stroke-width="1" opacity=".16" />
    <path d="${linePath(policy.events, metric, maxValue, state.frame)}"
      fill="none" stroke="${policyColor(index)}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />
  `).join("");
  elements.chart.innerHTML = `
    <svg viewBox="0 0 900 220" role="img" aria-label="Policy metric trajectories">
      ${guides}
      <line x1="${frameX}" x2="${frameX}" y1="12" y2="195" stroke="rgba(238,248,242,.36)" stroke-dasharray="2 4" />
      ${lines}
      <text x="44" y="214" fill="#597267" font-size="8">request 1</text>
      <text x="885" y="214" text-anchor="end" fill="#597267" font-size="8">request ${state.session.config.request_count}</text>
    </svg>
  `;
  elements.chartLegend.innerHTML = state.session.policies
    .map((policy, index) => `
      <span style="--policy-color:${policyColor(index)}">
        <i></i>${escapeHtml(policy.label)}
        <strong>${metricLabel(metric, metricValue(policy.events[state.frame], metric))}</strong>
      </span>
    `)
    .join("");
}

function renderCaches() {
  elements.cacheComparison.innerHTML = state.session.policies
    .map((policy, index) => {
      const event = policy.events[state.frame];
      const blocks = event.cache.map((block) => {
        const classes = [
          "cache-block",
          block.in_request ? "path" : "",
          block.hit_this_request ? "hit" : "",
          block.active_ref_count > 0 ? "active" : "",
          block.is_leaf ? "leaf" : "",
        ].join(" ");
        const title = [
          `depth ${block.depth}`,
          `${block.token_count} tokens`,
          `${block.hit_count} hits`,
          `${block.descendant_count} descendants`,
          `tenant ${block.tenant_id}`,
          block.active_ref_count > 0 ? `${block.active_ref_count} active refs` : "not pinned",
        ].join(" · ");
        return `<div class="${classes}" title="${escapeHtml(title)}">d${block.depth}<br>${block.block_id.slice(-4)}</div>`;
      }).join("");
      return `
        <article class="policy-cache" style="--policy-color:${policyColor(index)}">
          <header>
            <h4>${escapeHtml(policy.label)}</h4>
            <span>${event.resident_blocks} / ${event.capacity_blocks} blocks</span>
          </header>
          <div class="cache-grid">${blocks || '<span class="neutral">Cache empty</span>'}</div>
        </article>
      `;
    })
    .join("");
}

function renderEventTable() {
  const rows = state.session.policies.map((policy, index) => {
    const event = policy.events[state.frame];
    const hitRate = event.hit_tokens / Math.max(1, event.prompt_tokens);
    return `
      <tr style="--policy-color:${policyColor(index)}">
        <td><i class="policy-dot"></i>${escapeHtml(policy.label)}</td>
        <td class="${hitRate > 0 ? "positive" : "neutral"}">${percent(hitRate)}</td>
        <td>${event.matched_blocks} / ${event.prompt_blocks}</td>
        <td>${event.admissions}</td>
        <td class="${event.evictions ? "negative" : "neutral"}">${event.evictions}</td>
        <td>${event.bypassed_tokens}</td>
        <td>${event.resident_blocks}</td>
        <td>${number(event.latency_proxy, 1)}</td>
      </tr>
    `;
  }).join("");
  elements.eventTable.innerHTML = `
    <table>
      <thead><tr>
        <th>Policy</th><th>Token hit</th><th>Matched blocks</th><th>Admit</th>
        <th>Evict</th><th>Bypass tokens</th><th>Resident</th><th>Latency</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function setFrame(frame) {
  if (!state.session) return;
  state.frame = Math.max(0, Math.min(frame, state.session.config.request_count - 1));
  renderRequest();
  renderTimeline();
  renderChart();
  renderCaches();
  renderEventTable();
}

function stopPlayback() {
  state.playing = false;
  clearInterval(state.timer);
  state.timer = null;
  elements.playButton.textContent = "Play";
}

function togglePlayback() {
  if (!state.session) return;
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (state.frame >= state.session.config.request_count - 1) setFrame(0);
  state.playing = true;
  elements.playButton.textContent = "Pause";
  const delays = [900, 600, 350, 180, 80];
  state.timer = setInterval(() => {
    if (state.frame >= state.session.config.request_count - 1) {
      stopPlayback();
      return;
    }
    setFrame(state.frame + 1);
  }, delays[Number(elements.speed.value) - 1]);
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  runSimulation();
});
elements.policyList.addEventListener("change", () => {
  updatePolicyCount();
  markConfigurationPending();
});
elements.workload.addEventListener("change", () => {
  updateWorkloadDescription();
  runSimulation();
});
for (const input of [
  elements.requestCount,
  elements.capacityBlocks,
  elements.blockSizeTokens,
  elements.seed,
]) {
  input.addEventListener("input", () => {
    updateCapacityTokens();
    markConfigurationPending();
  });
}
elements.timeline.addEventListener("click", (event) => {
  const button = event.target.closest("[data-frame]");
  if (button) {
    stopPlayback();
    setFrame(Number(button.dataset.frame));
  }
});
elements.chartMetric.addEventListener("change", renderChart);
elements.playButton.addEventListener("click", togglePlayback);
elements.resetButton.addEventListener("click", () => {
  stopPlayback();
  setFrame(0);
});
elements.stepBackButton.addEventListener("click", () => {
  stopPlayback();
  setFrame(state.frame - 1);
});
elements.stepForwardButton.addEventListener("click", () => {
  stopPlayback();
  setFrame(state.frame + 1);
});
elements.speed.addEventListener("change", () => {
  if (state.playing) {
    stopPlayback();
    togglePlayback();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "r" && !event.metaKey && !event.ctrlKey && !event.altKey) {
    const tag = document.activeElement?.tagName;
    if (!["INPUT", "SELECT"].includes(tag)) runSimulation();
  }
  if (event.key === " " && !["INPUT", "SELECT"].includes(document.activeElement?.tagName)) {
    event.preventDefault();
    togglePlayback();
  }
  if (event.key === "ArrowRight") setFrame(state.frame + 1);
  if (event.key === "ArrowLeft") setFrame(state.frame - 1);
});

fetch("/api/catalog")
  .then((response) => response.json())
  .then((catalog) => {
    populateCatalog(catalog);
    runSimulation();
  })
  .catch((error) => {
    elements.formError.textContent = `Could not load lab catalog: ${error.message}`;
    setStatus("", "Offline", "Server unavailable");
  });
