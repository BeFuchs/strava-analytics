"use strict";

// Frontend logic: session handling, upload flow, global date filter, and
// rendering of tiles, charts and tables against the JSON API.

const state = {
  sessionId: sessionStorage.getItem("sessionId") || null,
  dateRange: { min: null, max: null }, // available range from the upload
  activePick: null,                    // active quick-pick chip, if any
  tables: {},                          // per-table rows + sort, for re-sorting
  selectedCluster: "",                 // "" = all climbs (flat list)
};

const el = (id) => document.getElementById(id);
const show = (node) => node.classList.remove("hidden");
const hide = (node) => node.classList.add("hidden");

function showGlobalError(message) {
  const banner = el("global-error");
  banner.textContent = message;
  show(banner);
}
const clearGlobalError = () => hide(el("global-error"));

// ---------- Upload ----------

function setupUpload() {
  const dropzone = el("dropzone");
  const input = el("file-input");

  el("pick-files").addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files.length) uploadFiles(input.files);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
  });
}

async function uploadFiles(fileList) {
  clearGlobalError();
  const files = Array.from(fileList);
  const status = el("upload-status");
  el("upload-summary").textContent = "";
  el("progress-fill").style.width = "0";
  document.querySelector("#upload-status .spinner").classList.remove("hidden");
  el("upload-message").textContent =
    files.length === 1
      ? "Verarbeite Datei — das kann bei großen Exporten dauern …"
      : `Verarbeite ${files.length} Dateien — das kann bei großen Exporten dauern …`;
  show(status);

  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const headers = {};
  if (state.sessionId) headers["X-Session-Id"] = state.sessionId;

  try {
    const response = await fetch("/api/upload", { method: "POST", body: form, headers });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Upload fehlgeschlagen.");
    onUploadComplete(body);
  } catch (err) {
    hide(status);
    showGlobalError(`Upload fehlgeschlagen: ${err.message}`);
  }
}

function onUploadComplete(body) {
  state.sessionId = body.session_id;
  sessionStorage.setItem("sessionId", state.sessionId);
  state.dateRange = body.date_range;
  sessionStorage.setItem("dateRange", JSON.stringify(state.dateRange));

  el("progress-fill").style.width = "100%";
  document.querySelector("#upload-status .spinner").classList.add("hidden");
  el("upload-message").textContent =
    `${body.rides_processed} Fahrt(en) verarbeitet` +
    (body.rides_skipped ? `, ${body.rides_skipped} übersprungen` : "") + ".";
  renderSkipSummary(body.skip_reasons);

  if (body.rides_processed === 0 && !state.dateRange.min) {
    showGlobalError("Keine Radfahrten in den hochgeladenen Dateien gefunden.");
    return;
  }
  enterDashboard();
}

function renderSkipSummary(skipReasons) {
  const summary = el("upload-summary");
  if (!skipReasons || !skipReasons.length) {
    summary.textContent = "";
    return;
  }
  const items = skipReasons
    .map((s) => `<li>${escapeHtml(s.file)} — ${escapeHtml(s.reason)}</li>`)
    .join("");
  summary.innerHTML =
    `<details class="skip-summary"><summary>${skipReasons.length} Datei(en) übersprungen</summary>` +
    `<ul>${items}</ul></details>`;
}

// ---------- View switching ----------

function enterDashboard() {
  hide(el("view-upload"));
  show(el("view-dashboard"));
  show(el("clear-data"));
  initFilterDefaults();
  refreshDashboard();
}

function restoreSession() {
  const storedRange = sessionStorage.getItem("dateRange");
  if (!state.sessionId || !storedRange) return false;
  // Enter the dashboard; a stale (server-restarted) session surfaces on the
  // first data request, which 404s and drops back to the upload view.
  state.dateRange = JSON.parse(storedRange);
  enterDashboard();
  return true;
}

function backToUpload() {
  state.sessionId = null;
  sessionStorage.removeItem("sessionId");
  sessionStorage.removeItem("dateRange");
  hide(el("view-dashboard"));
  hide(el("clear-data"));
  hide(el("upload-status"));
  show(el("view-upload"));
  showGlobalError("Die Sitzung ist abgelaufen — bitte Dateien erneut hochladen.");
}

// ---------- Filter ----------

function setupFilter() {
  el("date-from").addEventListener("change", () => onDateEdited());
  el("date-to").addEventListener("change", () => onDateEdited());
  el("quick-picks").querySelectorAll(".chip").forEach((chip) =>
    chip.addEventListener("click", () => applyQuickPick(chip.dataset.range, chip))
  );
}

function initFilterDefaults() {
  const { min, max } = state.dateRange;
  for (const id of ["date-from", "date-to"]) {
    el(id).min = min;
    el(id).max = max;
  }
  el("date-from").value = min;
  el("date-to").value = max;
  setActivePick(el("quick-picks").querySelector('[data-range="all"]'));
}

function onDateEdited() {
  setActivePick(null); // manual edit clears the quick-pick highlight
  if (validateRange()) refreshDashboard();
}

function setActivePick(chip) {
  el("quick-picks").querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
  if (chip) chip.classList.add("active");
}

function applyQuickPick(range, chip) {
  const { min, max } = state.dateRange;
  let from = min;
  if (range === "30") from = shiftDays(max, -29);
  else if (range === "90") from = shiftDays(max, -89);
  else if (range === "ytd") from = `${max.slice(0, 4)}-01-01`;
  // "all" keeps min.
  if (from < min) from = min;

  el("date-from").value = from;
  el("date-to").value = max;
  setActivePick(chip);
  if (validateRange()) refreshDashboard();
}

function validateRange() {
  const from = el("date-from").value;
  const to = el("date-to").value;
  const invalid = from && to && from > to;
  el("date-from").classList.toggle("invalid", invalid);
  el("date-to").classList.toggle("invalid", invalid);
  el("date-error").classList.toggle("hidden", !invalid);
  return !invalid;
}

function activeParams() {
  const params = new URLSearchParams();
  if (el("date-from").value) params.set("date_from", el("date-from").value);
  if (el("date-to").value) params.set("date_to", el("date-to").value);
  return params;
}

// ---------- Dashboard refresh ----------

async function refreshDashboard() {
  const params = activeParams();
  updateRangeCaptions();
  setLoading();

  await Promise.all([
    load("/api/summary", params, renderTiles, "tiles"),
    loadChart("/api/pmc", params, "chart-pmc", (b) => b.figure),
    loadChart("/api/power-curve", params, "chart-power-curve", (b) => b.figure),
    loadChart("/api/durability", params, "chart-durability", (b) => b.figure),
    loadChart("/api/zones", params, "chart-power-zones", (b) => b.power),
    loadChart("/api/zones", params, "chart-hr-zones", (b) => b.hr),
    load("/api/rides", params, renderRides, "table-rides"),
    load("/api/climbs", params, renderClimbs, "table-climbs"),
    load("/api/climbs/clusters", params, renderClusterOptions, "table-climbs"),
  ]);
  // Runs last: the cluster list decides whether the current pick still exists.
  applyClimbSelection();
}

function updateRangeCaptions() {
  const from = el("date-from").value;
  const to = el("date-to").value;
  const label = `${fmtDate(from)} – ${fmtDate(to)}`;
  el("range-caption").innerHTML = `<span id="range-count"></span><br>${label}`;
  el("header-range").textContent = label;
}

function setLoading() {
  el("tiles").innerHTML = skeleton(90);
  ["chart-pmc", "chart-power-curve", "chart-durability", "chart-power-zones", "chart-hr-zones"]
    .forEach((id) => (el(id).innerHTML = skeleton(id === "chart-pmc" ? 340 : 200)));
  el("table-rides").innerHTML = skeleton(200);
  el("table-climbs").innerHTML = skeleton(200);
}

const skeleton = (h) => `<div class="card-skeleton" style="height:${h}px"></div>`;

async function fetchJSON(path, params) {
  const url = params && params.toString() ? `${path}?${params}` : path;
  const response = await fetch(url, { headers: { "X-Session-Id": state.sessionId } });
  const body = await response.json().catch(() => ({}));
  if (response.status === 404 && body.error === "session not found") {
    const expired = new Error("session expired");
    expired.sessionExpired = true;
    throw expired;
  }
  if (!response.ok) throw new Error(body.detail || "Anfrage fehlgeschlagen.");
  return body;
}

async function load(path, params, render, containerId) {
  try {
    render(await fetchJSON(path, params));
  } catch (err) {
    if (err.sessionExpired) return backToUpload();
    el(containerId).innerHTML = `<div class="card-error">Konnte nicht geladen werden.</div>`;
  }
}

async function loadChart(path, params, containerId, pick) {
  try {
    const figure = pick(await fetchJSON(path, params));
    renderChart(containerId, figure);
  } catch (err) {
    if (err.sessionExpired) return backToUpload();
    el(containerId).innerHTML = `<div class="card-error">Diagramm konnte nicht geladen werden.</div>`;
  }
}

// ---------- Renderers ----------

const EMPTY =
  '<div class="card-empty">Keine Fahrten im gewählten Zeitraum. ' +
  "Erweitere den Zeitraum oder wähle „Alles“.</div>";

function renderChart(containerId, figure) {
  const node = el(containerId);
  if (!figure) {
    node.innerHTML = EMPTY;
    return;
  }
  if (typeof Plotly === "undefined") {
    // The chart library did not load — say so instead of blaming the data.
    node.innerHTML =
      '<div class="card-error">Diagramm-Bibliothek nicht geladen. ' +
      "Seite neu laden (Cmd+Shift+R).</div>";
    return;
  }
  node.innerHTML = "";
  Plotly.react(node, figure.data, figure.layout, { displayModeBar: false, responsive: true });
}

function renderTiles(summary) {
  const count = el("range-count");
  if (count) count.innerHTML = `<strong>${summary.n_rides} Fahrten</strong>`;
  // No rides in range -> nothing to export.
  el("export-csv").disabled = summary.n_rides === 0;
  const tiles = [
    ["Fahrten", fmtInt(summary.n_rides), ""],
    ["Distanz", fmtInt(summary.distance_km), "km"],
    ["Höhenmeter", fmtInt(summary.elevation_gain_m), "m"],
    ["Zeit", fmtHoursMin(summary.moving_time_s), "h"],
    ["Gesamt-TSS", fmtInt(summary.total_tss), ""],
    ["Ø CTL", fmtOrDash(summary.avg_ctl, 0), ""],
    ["Gesch. FTP", fmtOrDash(summary.ftp_estimate_watts, 0), "W"],
  ];
  el("tiles").innerHTML = tiles
    .map(
      ([label, value, unit]) =>
        `<div class="tile"><div class="label">${label}</div>` +
        `<div class="value">${value}${unit ? `<span class="unit"> ${unit}</span>` : ""}</div></div>`
    )
    .join("");
}

const RIDE_COLUMNS = [
  { key: "date", label: "Datum", type: "date" },
  { key: "source", label: "Datei", type: "text" },
  { key: "distance_km", label: "km", type: "num", digits: 1 },
  { key: "moving_time_s", label: "Zeit", type: "duration" },
  { key: "elevation_gain_m", label: "Höhe (m)", type: "num", digits: 0 },
  { key: "np_watts", label: "NP", type: "num", digits: 0 },
  { key: "intensity_factor", label: "IF", type: "num", digits: 2 },
  { key: "tss", label: "TSS", type: "tss" },
  { key: "avg_hr", label: "Ø HF", type: "num", digits: 0 },
];

const CLIMB_COLUMNS = [
  { key: "date", label: "Datum", type: "date" },
  { key: "length_km", label: "Länge (km)", type: "num", digits: 1 },
  { key: "elevation_gain_m", label: "Höhe (m)", type: "num", digits: 0 },
  { key: "avg_gradient_pct", label: "Ø %", type: "num", digits: 1 },
  { key: "max_gradient_pct", label: "Max %", type: "num", digits: 1 },
  { key: "duration_s", label: "Zeit", type: "duration" },
  { key: "vam_m_per_h", label: "VAM", type: "num", digits: 0 },
  { key: "avg_power_watts", label: "Ø W", type: "num", digits: 0 },
  { key: "watts_per_kg", label: "W/kg", type: "num", digits: 1 },
  { key: "kj_before_climb", label: "kJ vorher", type: "num", digits: 0 },
];

function renderRides(body) {
  el("rides-subtitle").textContent = `${body.n_rides} Fahrten, neueste zuerst`;
  renderTable("table-rides", RIDE_COLUMNS, body.rides, "Keine Fahrten im gewählten Zeitraum.");
}

function renderClimbs(body) {
  el("climbs-subtitle").textContent = `${body.n_climbs} Anstiege, neueste zuerst`;
  renderTable("table-climbs", CLIMB_COLUMNS, body.climbs, "Keine Anstiege im gewählten Zeitraum.");
}

// ---------- Climb cluster picker ----------

function clusterLabel(cluster) {
  const parts = [
    cluster.name || cluster.location_label,
    `${fmtNum(cluster.length_km, 1)} km`,
    `${fmtNum(cluster.avg_gradient_pct, 1)} %`,
    cluster.ascent_count === 1 ? "1 Befahrung" : `${cluster.ascent_count} Befahrungen`,
  ];
  return parts.join(" · ");
}

function renderClusterOptions(body) {
  const select = el("climb-select");
  const clusters = body.clusters || [];
  // Repeatedly ridden climbs first; one-offs are grouped at the end.
  const repeated = clusters.filter((c) => c.ascent_count > 1);
  const singles = clusters.filter((c) => c.ascent_count === 1);

  const option = (c) => `<option value="${c.cluster_id}">${escapeHtml(clusterLabel(c))}</option>`;
  let html = '<option value="">Alle Anstiege</option>';
  html += repeated.map(option).join("");
  if (singles.length) {
    html +=
      '<optgroup label="Einmalig gefahren">' + singles.map(option).join("") + "</optgroup>";
  }
  select.innerHTML = html;

  // Keep the current pick when the date filter reshuffles the cluster list.
  const stillThere = clusters.some((c) => c.cluster_id === state.selectedCluster);
  state.selectedCluster = stillThere ? state.selectedCluster : "";
  select.value = state.selectedCluster;
}

function onClusterSelected() {
  state.selectedCluster = el("climb-select").value;
  applyClimbSelection();
}

/** Show either the flat climbs table or the detail view for one cluster. */
function applyClimbSelection() {
  const detail = el("climb-detail");
  const flat = el("table-climbs");
  if (!state.selectedCluster) {
    hide(detail);
    show(flat);
    return;
  }
  hide(flat);
  show(detail);
  el("climb-detail-head").innerHTML = "";
  el("chart-climb-trend").innerHTML = "";
  el("table-ascents").innerHTML = skeleton(200);
  load(
    `/api/climbs/clusters/${encodeURIComponent(state.selectedCluster)}`,
    activeParams(),
    renderClimbDetail,
    "table-ascents"
  );
}

const ASCENT_COLUMNS = [
  { key: "date", label: "Datum", type: "date" },
  { key: "duration_s", label: "Zeit", type: "duration" },
  { key: "vam_m_per_h", label: "VAM", type: "num", digits: 0 },
  { key: "avg_power_watts", label: "Ø W", type: "num", digits: 0 },
  { key: "watts_per_kg", label: "W/kg", type: "num", digits: 1 },
  { key: "avg_hr", label: "Ø HF", type: "num", digits: 0 },
  { key: "pacing_quarters", label: "Pacing (Viertel)", type: "pacing" },
];

function renderClimbDetail(cluster) {
  state.detailCluster = cluster;
  renderDetailHead(cluster);

  renderChart("chart-climb-trend", cluster.trend_figure);
  if (!cluster.trend_figure) hide(el("chart-climb-trend"));
  else show(el("chart-climb-trend"));

  // Mark the fastest ascent so the personal best is readable, not just tinted.
  const best = Math.min(...cluster.ascents.map((a) => a.duration_s));
  const rows = cluster.ascents.map((a) => ({ ...a, _best: a.duration_s === best }));
  renderTable("table-ascents", ASCENT_COLUMNS, rows, "Keine Befahrungen im Zeitraum.");
}

function detailFacts(cluster) {
  return (
    `${fmtNum(cluster.length_km, 1)} km · ` +
    `${fmtNum(cluster.avg_gradient_pct, 1)} % · ` +
    `${fmtNum(cluster.elevation_gain_m, 0)} Hm · ` +
    `${cluster.ascent_count} Befahrungen · Bestzeit ${fmtHMS(cluster.best_time_s)}`
  );
}

function renderDetailHead(cluster) {
  const title = cluster.name || cluster.location_label;
  el("climb-detail-head").innerHTML =
    `<div class="title"><span id="climb-title">${escapeHtml(title)}</span>` +
    `<button class="btn-rename" id="btn-rename" title="Umbenennen" aria-label="Anstieg umbenennen">✎</button></div>` +
    `<div class="facts">${detailFacts(cluster)}</div>`;
  el("btn-rename").addEventListener("click", startRename);
}

function startRename() {
  const cluster = state.detailCluster;
  const titleRow = document.querySelector("#climb-detail-head .title");
  titleRow.innerHTML =
    `<div class="rename-box">` +
    `<input id="rename-input" maxlength="60" value="${escapeHtml(cluster.name || "")}" ` +
    `placeholder="${escapeHtml(cluster.location_label)}">` +
    `<button class="btn-primary" id="rename-save">Speichern</button>` +
    `<span class="rename-hint">Namen gelten nur für diese Sitzung. Leer = Koordinaten.</span>` +
    `</div>`;
  const input = el("rename-input");
  input.focus();
  input.select();
  el("rename-save").addEventListener("click", saveRename);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") saveRename();
    if (e.key === "Escape") renderDetailHead(state.detailCluster);
  });
}

async function saveRename() {
  const cluster = state.detailCluster;
  const name = el("rename-input").value.trim();
  try {
    const response = await fetch(
      `/api/climbs/clusters/${encodeURIComponent(cluster.cluster_id)}/name`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-Session-Id": state.sessionId },
        body: JSON.stringify({ name }),
      }
    );
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Umbenennen fehlgeschlagen.");
    cluster.name = body.name;
    renderDetailHead(cluster);
    refreshClusterOptions(); // reflect the new name in the dropdown
  } catch (err) {
    if (err.sessionExpired) return backToUpload();
    renderDetailHead(cluster);
    showGlobalError(err.message);
  }
}

async function refreshClusterOptions() {
  await load("/api/climbs/clusters", activeParams(), renderClusterOptions, "table-climbs");
}

function renderTable(containerId, columns, rows, emptyText) {
  if (!rows.length) {
    el(containerId).innerHTML = `<div class="card-empty">${emptyText}</div>`;
    return;
  }
  // Reload resets sorting to the server order (date descending).
  state.tables[containerId] = { columns, rows, sort: null };
  drawTable(containerId);
}

function drawTable(containerId) {
  const table = state.tables[containerId];
  const { columns, rows, sort } = table;

  const sorted = sort ? sortRows(rows, sort) : rows;
  const head = columns
    .map((col, i) => {
      const arrow = sort && sort.index === i ? (sort.dir > 0 ? " ▲" : " ▼") : "";
      return `<th class="sortable" data-index="${i}">${col.label}${arrow}</th>`;
    })
    .join("");
  const body = sorted
    .map((row) => {
      const cells = columns.map((col, i) => {
        let content = fmtCell(row[col.key], col);
        if (i === 0 && row._best) content += '<span class="badge-best">Bestzeit</span>';
        return `<td>${content}</td>`;
      });
      return `<tr${row._best ? ' class="is-best"' : ""}>${cells.join("")}</tr>`;
    })
    .join("");

  el(containerId).innerHTML =
    `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  el(containerId)
    .querySelectorAll("th.sortable")
    .forEach((th) => th.addEventListener("click", () => toggleSort(containerId, +th.dataset.index)));
}

function toggleSort(containerId, index) {
  const table = state.tables[containerId];
  const dir = table.sort && table.sort.index === index ? -table.sort.dir : 1;
  table.sort = { index, dir, key: table.columns[index].key };
  drawTable(containerId);
}

function sortRows(rows, sort) {
  return [...rows].sort((a, b) => {
    const x = a[sort.key];
    const y = b[sort.key];
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    if (x < y) return -sort.dir;
    if (x > y) return sort.dir;
    return 0;
  });
}

// ---------- CSV export ----------

async function exportCsv() {
  const button = el("export-csv");
  if (button.disabled) return;
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Erzeuge …";
  clearGlobalError();

  try {
    const url = `/api/export/csv?${activeParams()}`;
    const response = await fetch(url, { headers: { "X-Session-Id": state.sessionId } });
    if (response.status === 404) {
      const body = await response.json().catch(() => ({}));
      if (body.error === "session not found") return backToUpload();
      throw new Error(body.detail || "Nichts zu exportieren.");
    }
    if (!response.ok) throw new Error("Export fehlgeschlagen.");
    triggerDownload(await response.blob(), filenameFromResponse(response));
  } catch (err) {
    showGlobalError(err.message);
  } finally {
    button.textContent = original;
    button.disabled = false;
  }
}

function filenameFromResponse(response) {
  const disposition = response.headers.get("content-disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  return match ? match[1] : "ride-analytics.zip";
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// ---------- Clear session ----------

async function clearData() {
  if (!state.sessionId) return;
  try {
    await fetch("/api/session", {
      method: "DELETE",
      headers: { "X-Session-Id": state.sessionId },
    });
  } catch (err) {
    // Local-only app: drop client state even if the call fails.
  }
  state.sessionId = null;
  sessionStorage.removeItem("sessionId");
  location.reload();
}

// ---------- Formatting ----------

function fmtCell(value, col) {
  if (col.type === "date") return fmtDate(value);
  if (col.type === "text") return escapeHtml(value ?? "");
  if (col.type === "duration") return fmtHMS(value);
  if (col.type === "pacing") {
    if (!value) return "–";
    return value.map((q) => (q === null ? "–" : fmtNum(q, 0))).join(" / ");
  }
  if (col.type === "tss") {
    return value === null || value === undefined ? "–" : fmtNum(value, 0);
  }
  return fmtOrDash(value, col.digits);
}

function fmtDate(iso) {
  if (!iso) return "–";
  const [y, m, d] = iso.split("-");
  return `${d}.${m}.${y}`;
}

function fmtNum(value, digits) {
  return Number(value).toLocaleString("de-DE", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

const fmtInt = (value) => fmtNum(value ?? 0, 0);

function fmtOrDash(value, digits) {
  return value === null || value === undefined ? "–" : fmtNum(value, digits);
}

function fmtHMS(seconds) {
  if (seconds === null || seconds === undefined) return "–";
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function fmtHoursMin(seconds) {
  const total = Math.round(seconds || 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  return `${h}:${String(m).padStart(2, "0")}`;
}

function shiftDays(iso, delta) {
  const date = new Date(`${iso}T00:00:00`);
  date.setDate(date.getDate() + delta);
  return date.toISOString().slice(0, 10);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---------- Boot ----------

function boot() {
  setupUpload();
  setupFilter();
  el("climb-select").addEventListener("change", onClusterSelected);
  el("export-csv").addEventListener("click", exportCsv);
  el("clear-data").addEventListener("click", clearData);
  restoreSession();
}

document.addEventListener("DOMContentLoaded", boot);
