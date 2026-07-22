"use strict";

// Frontend scaffolding: session handling, upload flow, view/state switching.
// Filter wiring and data rendering are added in the data-binding step.

const state = {
  sessionId: sessionStorage.getItem("sessionId") || null,
  dateRange: { min: null, max: null }, // available range from the upload
};

const el = (id) => document.getElementById(id);

function show(node) { node.classList.remove("hidden"); }
function hide(node) { node.classList.add("hidden"); }

function showGlobalError(message) {
  const banner = el("global-error");
  banner.textContent = message;
  show(banner);
}

function clearGlobalError() {
  hide(el("global-error"));
}

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
  const summary = el("upload-summary");
  summary.textContent = "";
  el("progress-fill").style.width = "0";
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
    if (!response.ok) {
      throw new Error(body.detail || "Upload fehlgeschlagen.");
    }
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

function initFilterDefaults() {
  const { min, max } = state.dateRange;
  for (const id of ["date-from", "date-to"]) {
    el(id).min = min;
    el(id).max = max;
  }
  el("date-from").value = min;
  el("date-to").value = max;
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
    // Local-only app: even if the call fails, drop the client state.
  }
  state.sessionId = null;
  sessionStorage.removeItem("sessionId");
  location.reload();
}

// ---------- Data binding (added in the next step) ----------

function refreshDashboard() {
  // Filled in by the data-binding step: fetch summary/pmc/curves/zones/tables
  // for the active date range and render tiles, charts and tables.
}

// ---------- Utils ----------

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---------- Boot ----------

function boot() {
  setupUpload();
  el("clear-data").addEventListener("click", clearData);
  // A stored session id may be stale (server restarted) — the first data
  // request will 404 and send the user back to the upload view.
}

document.addEventListener("DOMContentLoaded", boot);
