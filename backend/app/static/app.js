const $ = (id) => document.getElementById(id);

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

let sessionId = null;
let session = null;
let sketchAssetId = null;
let currentRev = null;
let constraintsHash = null;
let revisionsCache = [];

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = JSON.parse(text); } catch { data = text; }
  if (!res.ok) {
    const detail = (data && data.detail) ? data.detail : data;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail, null, 2));
  }
  return data;
}

function requireSession() {
  if (!sessionId) throw new Error("Create or load a session first.");
}

function fmtNum(x) {
  if (x === null || x === undefined || x === "") return "";
  const n = Number(x);
  if (!Number.isFinite(n)) return String(x);
  // Compact formatting
  const abs = Math.abs(n);
  if (abs >= 100) return n.toFixed(1);
  if (abs >= 10) return n.toFixed(2);
  return n.toFixed(3);
}

function badge(status, label) {
  const cls = status === "ok" ? "ok" : (status === "warn" ? "warn" : "fail");
  const txt = label || (status === "ok" ? "OK" : (status === "warn" ? "WARN" : "FAIL"));
  return `<span class="badge ${cls}">${escapeHtml(txt)}</span>`;
}

function updateUi() {
  $("sessionId").textContent = sessionId || "(none)";
  $("sketchAssetId").textContent = sketchAssetId || "(none)";
  $("currentRev").textContent = currentRev || "(none)";
  $("constraintsHash").textContent = constraintsHash || "(none)";

  $("btnUploadSketch").disabled = !sessionId;
  $("btnCreateFromText").disabled = !sessionId;
  $("btnCreateFromSpec").disabled = !sessionId;
  $("btnApplyChange").disabled = !sessionId || !currentRev;
  $("btnLoadConstraints").disabled = !sessionId;
  $("btnSaveConstraints").disabled = !sessionId;

  // Approve can be further gated by constraint status in showRevision
  $("btnApprove").disabled = !sessionId || !currentRev;

  // Finalize requires an approved revision
  const canFinalize = !!(sessionId && session && session.approved_revision);
  $("btnFinalize").disabled = !canFinalize;
}

function clearCurrentRevisionView() {
  currentRev = null;
  $("drawingWrap").innerHTML = "";
  $("aiNotes").textContent = "(none)";
  $("clarifyWrap").innerHTML = "";
  $("constraintsSummary").innerHTML = "<p class=\"muted small\">(none)</p>";
  $("linkStl").href = "#";
  $("linkDims").href = "#";
  $("report").textContent = "";
}

// -----------------------
// Constraints UI
// -----------------------

function _readVec3(ids) {
  const vals = ids.map((id) => String($(id).value || "").trim());
  const allBlank = vals.every((v) => !v);
  if (allBlank) return null;
  const anyBlank = vals.some((v) => !v);
  if (anyBlank) throw new Error("Please fill all 3 fields or leave all blank.");
  const nums = vals.map((v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) throw new Error(`Invalid number: ${v}`);
    return n;
  });
  return nums;
}

function _readFloat(id, { allowBlank = true, defaultValue = null } = {}) {
  const raw = String($(id).value || "").trim();
  if (!raw) {
    if (!allowBlank) throw new Error(`Missing value for ${id}`);
    return defaultValue;
  }
  const n = Number(raw);
  if (!Number.isFinite(n)) throw new Error(`Invalid number: ${raw}`);
  return n;
}

function fillConstraintsForm(c) {
  // c is a plain object from API
  $("c_units").value = c.preferred_units || "mm";

  // optional vec3
  const fit = c.must_fit_within || null;
  $("c_fit_w").value = fit ? fmtNum(fit[0]) : "";
  $("c_fit_d").value = fit ? fmtNum(fit[1]) : "";
  $("c_fit_h").value = fit ? fmtNum(fit[2]) : "";

  const tgt = c.target_overall_size || null;
  $("c_target_w").value = tgt ? fmtNum(tgt[0]) : "";
  $("c_target_d").value = tgt ? fmtNum(tgt[1]) : "";
  $("c_target_h").value = tgt ? fmtNum(tgt[2]) : "";

  $("c_tol").value = (c.overall_size_tolerance !== undefined && c.overall_size_tolerance !== null)
    ? fmtNum(c.overall_size_tolerance)
    : "";

  $("c_min_wall").value = (c.min_wall_thickness !== undefined && c.min_wall_thickness !== null)
    ? fmtNum(c.min_wall_thickness)
    : "";

  $("c_min_feat").value = (c.min_feature_size !== undefined && c.min_feature_size !== null)
    ? fmtNum(c.min_feature_size)
    : "";

  $("c_hole_offset").value = (c.hole_diameter_offset !== undefined && c.hole_diameter_offset !== null)
    ? fmtNum(c.hole_diameter_offset)
    : "";

  $("c_printable").checked = !!c.must_be_printable;
  $("c_overhang").value = (c.max_overhang_angle_deg !== undefined && c.max_overhang_angle_deg !== null)
    ? fmtNum(c.max_overhang_angle_deg)
    : "";

  $("c_notes").value = c.notes || "";
}

function readConstraintsPatchFromForm() {
  const preferred_units = $("c_units").value;
  const must_fit_within = _readVec3(["c_fit_w", "c_fit_d", "c_fit_h"]);
  const target_overall_size = _readVec3(["c_target_w", "c_target_d", "c_target_h"]);

  const overall_size_tolerance = _readFloat("c_tol", { allowBlank: true, defaultValue: null });
  const min_wall_thickness = _readFloat("c_min_wall", { allowBlank: true, defaultValue: null });
  const min_feature_size = _readFloat("c_min_feat", { allowBlank: true, defaultValue: null });

  // Blank offset defaults to 0
  const hole_diameter_offset = _readFloat("c_hole_offset", { allowBlank: true, defaultValue: 0.0 });

  const must_be_printable = !!$("c_printable").checked;
  const max_overhang_angle_deg = _readFloat("c_overhang", { allowBlank: true, defaultValue: 45.0 });

  const notesRaw = String($("c_notes").value || "").trim();
  const notes = notesRaw ? notesRaw : null;

  return {
    preferred_units,
    must_fit_within,
    target_overall_size,
    overall_size_tolerance,
    min_wall_thickness,
    min_feature_size,
    hole_diameter_offset,
    must_be_printable,
    max_overhang_angle_deg,
    notes,
  };
}

async function loadConstraints() {
  requireSession();
  const data = await api(`/sessions/${sessionId}/constraints`);
  constraintsHash = data.constraints_hash;
  fillConstraintsForm(data.constraints || {});
  $("constraintsDebug").textContent = JSON.stringify(data, null, 2);
  updateUi();
}

async function saveConstraints() {
  requireSession();
  const patch = readConstraintsPatchFromForm();
  const data = await api(`/sessions/${sessionId}/constraints`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  // data.session is the new session record
  session = data.session;
  constraintsHash = data.constraints_hash;
  $("constraintsDebug").textContent = JSON.stringify(data, null, 2);
  await refreshRevisionsList();
  if (currentRev) {
    const rev = await api(`/sessions/${sessionId}/revisions/${currentRev}`);
    await showRevision(rev);
  }
  updateUi();
}

// -----------------------
// Session + revisions
// -----------------------

async function createSession() {
  const name = String($("sessionName").value || "").trim() || null;
  const payload = { name };
  const data = await api("/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  session = data;
  sessionId = data.id;
  sketchAssetId = null;
  currentRev = null;

  $("sessionsList").textContent = JSON.stringify(data, null, 2);
  clearCurrentRevisionView();
  await loadConstraints();
  await refreshRevisionsList();
  updateUi();
}

async function listSessions() {
  const data = await api("/sessions");
  $("sessionsList").textContent = JSON.stringify(data, null, 2);
}

async function loadSessionById() {
  const id = String($("sessionLoadId").value || "").trim();
  if (!id) throw new Error("Enter a session id to load.");
  const s = await api(`/sessions/${id}`);
  session = s;
  sessionId = s.id;
  sketchAssetId = null;
  currentRev = s.current_revision || null;

  await loadConstraints();
  await refreshRevisionsList();

  if (currentRev) {
    const rev = await api(`/sessions/${sessionId}/revisions/${currentRev}`);
    await showRevision(rev);
  } else {
    clearCurrentRevisionView();
  }

  $("sessionsList").textContent = JSON.stringify(s, null, 2);
  updateUi();
}

async function refreshRevisionsList() {
  if (!sessionId) {
    $("revisionsWrap").innerHTML = "";
    return;
  }
  const revs = await api(`/sessions/${sessionId}/revisions`);
  revisionsCache = Array.isArray(revs) ? revs : [];

  if (!revisionsCache.length) {
    $("revisionsWrap").innerHTML = `<p class="muted small">(no revisions yet)</p>`;
    return;
  }

  let html = `<table><thead><tr>`
    + `<th>#</th><th>Stage</th><th>Source</th><th>Constraints</th><th>Created</th>`
    + `</tr></thead><tbody>`;

  for (const r of revisionsCache.slice().sort((a, b) => a.rev - b.rev)) {
    const cErrs = (r.validation && r.validation.constraint_errors) ? r.validation.constraint_errors : [];
    const cWarns = (r.validation && r.validation.constraint_warnings) ? r.validation.constraint_warnings : [];
    const revHash = r.metadata ? r.metadata.constraints_hash : null;
    const stale = !!(constraintsHash && revHash && constraintsHash !== revHash);

    let cStatus = badge("ok");
    if (cErrs && cErrs.length) cStatus = badge("fail");
    else if (stale || (cWarns && cWarns.length)) cStatus = badge("warn");

    html += `<tr>`
      + `<td><a href="#" data-rev="${r.rev}">${r.rev}</a></td>`
      + `<td>${escapeHtml(r.stage || "")}</td>`
      + `<td>${escapeHtml(r.source || "")}</td>`
      + `<td>${cStatus}${stale ? ` <span class="muted small">stale</span>` : ""}</td>`
      + `<td class="muted small">${escapeHtml(r.created_at || "")}</td>`
      + `</tr>`;
  }

  html += `</tbody></table>`;
  $("revisionsWrap").innerHTML = html;

  // Wire clicks
  $("revisionsWrap").querySelectorAll("a[data-rev]").forEach((a) => {
    a.onclick = (ev) => {
      ev.preventDefault();
      const n = Number(a.getAttribute("data-rev"));
      if (!Number.isFinite(n)) return;
      api(`/sessions/${sessionId}/revisions/${n}`).then(showRevision).catch((e) => alert(e.message));
    };
  });
}

async function uploadSketch() {
  requireSession();
  const f = $("sketchFile").files[0];
  if (!f) throw new Error("Pick a file first.");
  const form = new FormData();
  form.append("file", f);
  const data = await api(`/sessions/${sessionId}/uploads/sketch`, { method: "POST", body: form });
  sketchAssetId = data.asset_id;
  $("sketchPreviewWrap").innerHTML = `<img src="${data.url}" alt="sketch preview" />`;
  $("report").textContent = JSON.stringify(data, null, 2);
  updateUi();
}

async function createRevisionFromText() {
  requireSession();
  const description = String($("desc").value || "").trim();
  if (!description) throw new Error("Enter a description.");
  const csv_path = String($("csvPath").value || "user_api.csv").trim() || "user_api.csv";
  const model = String($("modelName").value || "gpt-4.1").trim() || "gpt-4.1";

  const payload = {
    mode: "from_text",
    description,
    note: "concept",
    csv_path,
    model,
    sketch_asset_id: sketchAssetId,
  };

  const data = await api(`/sessions/${sessionId}/revisions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  // Session pointers likely updated; update locally
  if (session) {
    session.current_revision = data.rev;
    session.status = "concept";
    session.approved_revision = null;
    session.final_revision = null;
  }

  await showRevision(data);
  await refreshRevisionsList();
}

async function createRevisionFromSpec() {
  requireSession();
  const raw = String($("specJson").value || "").trim();
  if (!raw) throw new Error("Paste PartSpec JSON.");
  let spec = null;
  try { spec = JSON.parse(raw); } catch { throw new Error("Spec JSON is not valid JSON."); }

  const payload = { mode: "from_spec", spec, note: "concept" };
  const data = await api(`/sessions/${sessionId}/revisions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (session) {
    session.current_revision = data.rev;
    session.status = "concept";
    session.approved_revision = null;
    session.final_revision = null;
  }

  await showRevision(data);
  await refreshRevisionsList();
}

function renderAiNotes(meta) {
  const lines = [];
  if (meta && meta.plan_summary) {
    lines.push("PLAN:\n" + String(meta.plan_summary));
  }
  if (meta && Array.isArray(meta.assumptions) && meta.assumptions.length) {
    lines.push("ASSUMPTIONS:\n" + meta.assumptions.map((s) => "- " + s).join("\n"));
  }
  if (meta && Array.isArray(meta.questions_for_user) && meta.questions_for_user.length) {
    lines.push("QUESTIONS FOR YOU:\n" + meta.questions_for_user.map((s) => "- " + s).join("\n"));
  }
  $("aiNotes").textContent = lines.length ? lines.join("\n\n") : "(none)";
}

function renderConstraintsSummary(rev) {
  const v = (rev && rev.validation) ? rev.validation : {};
  const report = (v && v.constraints_report) ? v.constraints_report : {};

  const cErrs = Array.isArray(v.constraint_errors) ? v.constraint_errors : [];
  const cWarns = Array.isArray(v.constraint_warnings) ? v.constraint_warnings : [];

  const revHash = (rev && rev.metadata) ? rev.metadata.constraints_hash : null;
  const stale = !!(constraintsHash && revHash && constraintsHash !== revHash);

  const parts = [];

  // Overall status
  let overallStatus = "ok";
  if (cErrs.length) overallStatus = "fail";
  else if (stale || cWarns.length) overallStatus = "warn";

  parts.push(`<div class="line">${badge(overallStatus, "Constraints")}`
    + (stale ? ` <span class="muted small">Revision was generated under a different constraints hash.</span>` : "")
    + `</div>`);

  // Units
  if (report.units) {
    const ok = !!report.units.ok;
    parts.push(`<div class="line">${badge(ok ? "ok" : "fail", "Units")}`
      + `<span class="kv">preferred=${escapeHtml(report.units.preferred)} actual=${escapeHtml(report.units.actual)}</span></div>`);
  }

  // Fit-within
  if (report.must_fit_within) {
    const ok = !!report.must_fit_within.ok;
    const limit = report.must_fit_within.limit;
    const margins = report.must_fit_within.margins;
    parts.push(`<div class="line">${badge(ok ? "ok" : "fail", "Fit")}`
      + `<span class="kv">limit=${escapeHtml(JSON.stringify(limit))} margins=${escapeHtml(JSON.stringify(margins))}</span></div>`);
  }

  // Target size
  if (report.target_overall_size) {
    const ok = !!report.target_overall_size.ok;
    const target = report.target_overall_size.target;
    const tol = report.target_overall_size.tolerance;
    const deltas = report.target_overall_size.deltas;
    parts.push(`<div class="line">${badge(ok ? "ok" : "fail", "Target")}`
      + `<span class="kv">target=${escapeHtml(JSON.stringify(target))} ±${escapeHtml(String(tol))} deltas=${escapeHtml(JSON.stringify(deltas))}</span></div>`);
  }

  // Min wall
  if (report.min_wall_thickness) {
    const mw = report.min_wall_thickness;
    if (!mw.supported) {
      parts.push(`<div class="line">${badge("warn", "Min wall")}`
        + `<span class="muted small">unsupported: ${escapeHtml(mw.reason || "")}</span></div>`);
    } else {
      const ok = !!mw.ok;
      parts.push(`<div class="line">${badge(ok ? "ok" : "fail", "Min wall")}`
        + `<span class="kv">required=${escapeHtml(String(mw.required))} actual=${escapeHtml(String(mw.min_wall_actual))}</span></div>`);
    }
  }

  // Min feature
  if (report.min_feature_size) {
    const mfs = report.min_feature_size;
    const ok = !!mfs.ok;
    parts.push(`<div class="line">${badge(ok ? "ok" : "fail", "Min feature")}`
      + `<span class="kv">required=${escapeHtml(String(mfs.required))} actual_min=${escapeHtml(String(mfs.actual_min))}</span></div>`);
  }

  // Export effects
  if (report.export_effects && report.export_effects.hole_diameter_offset !== undefined) {
    parts.push(`<div class="line">${badge("warn", "Export")}`
      + `<span class="kv">hole_diameter_offset=${escapeHtml(String(report.export_effects.hole_diameter_offset))}</span></div>`);
  }

  // Printability (from mesh report)
  const pr = rev?.artifacts?.mesh_report?.printability;
  if (pr) {
    const frac = pr.support_fraction_of_total_area;
    let status = "ok";
    if (frac > 0.25) status = "warn";
    if (frac > 0.45) status = "fail";
    parts.push(`<div class="line">${badge(status, "Printability")}`
      + `<span class="kv">support_fraction_total=${escapeHtml(fmtNum(frac))} (angle=${escapeHtml(String(pr.max_overhang_angle_deg))}°)</span></div>`);
  } else if (report.must_be_printable && report.must_be_printable.enabled) {
    parts.push(`<div class="line">${badge("warn", "Printability")}`
      + `<span class="muted small">enabled but no mesh printability report found (regenerate artifacts).</span></div>`);
  }

  // Errors & warnings list
  if (cErrs.length) {
    parts.push(`<div class="line">${badge("fail", "Constraint errors")}</div>`);
    parts.push(`<ul class="muted small">${cErrs.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>`);
  }
  if (cWarns.length) {
    parts.push(`<div class="line">${badge("warn", "Constraint warnings")}</div>`);
    parts.push(`<ul class="muted small">${cWarns.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>`);
  }

  $("constraintsSummary").innerHTML = parts.join("\n") || `<p class="muted small">(none)</p>`;

  // Gate approve button
  let reason = null;
  if (stale) reason = "Constraints changed since this revision was generated.";
  if (cErrs.length) reason = (reason ? reason + " " : "") + "Constraint errors must be resolved before approval.";

  $("btnApprove").disabled = !sessionId || !currentRev || !!reason;
  $("btnApprove").title = reason || "";
}

function renderClarifications(meta) {
  const wrap = $("clarifyWrap");
  const qs = (meta && Array.isArray(meta.questions_for_user)) ? meta.questions_for_user : [];

  if (!qs.length) {
    wrap.innerHTML = `<p class="muted small">(none)</p>`;
    return;
  }

  let html = `<p class="muted small">Answer these questions to improve accuracy, then generate a new revision.</p>`;
  html += `<div class="qa">`;
  qs.forEach((q, i) => {
    html += `
      <div class="qa-item">
        <div class="qa-q"><strong>Q${i + 1}:</strong> ${escapeHtml(q)}</div>
        <input class="qa-a" id="clarifyAnswer_${i}" placeholder="Your answer (e.g. 12mm, through-hole, centered, etc.)" />
      </div>
    `;
  });
  html += `</div>`;
  html += `<div class="row"><button id="btnSubmitClarify" class="success">Apply answers (new revision)</button></div>`;
  html += `<p class="muted small">Requires OpenAI + user_api.csv (BYOK).</p>`;

  wrap.innerHTML = html;
  $("btnSubmitClarify").onclick = () => submitClarifications(qs).catch((e) => alert(e.message));
}

async function submitClarifications(questions) {
  requireSession();
  if (!currentRev) throw new Error("No current revision.");

  const answers = [];
  for (let i = 0; i < questions.length; i++) {
    const inp = $(`clarifyAnswer_${i}`);
    const val = inp ? String(inp.value || "").trim() : "";
    if (!val) continue;
    answers.push({ question: questions[i], answer: val });
  }
  if (!answers.length) throw new Error("Please answer at least one question before submitting.");

  const csv_path = String($("csvPath").value || "user_api.csv").trim() || "user_api.csv";
  const model = String($("modelName").value || "gpt-4.1").trim() || "gpt-4.1";

  const payload = { answers, note: "clarify", csv_path, model };
  const data = await api(`/sessions/${sessionId}/revisions/${currentRev}/clarify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (session) {
    session.current_revision = data.rev;
    session.status = "concept";
    session.approved_revision = null;
    session.final_revision = null;
  }

  await showRevision(data);
  await refreshRevisionsList();
}

async function showRevision(rev) {
  currentRev = rev.rev;
  $("report").textContent = JSON.stringify(rev, null, 2);

  const meta = (rev && rev.spec && rev.spec.metadata) ? rev.spec.metadata : {};
  renderAiNotes(meta);
  renderClarifications(meta);
  renderConstraintsSummary(rev);

  // Drawing
  const drawingUrl = rev?.artifacts?.drawing_url;
  if (drawingUrl) {
    try {
      const svgText = await fetch(drawingUrl).then((r) => r.text());
      $("drawingWrap").innerHTML = svgText;
    } catch {
      $("drawingWrap").innerHTML = `<p class="muted small">Failed to load drawing.</p>`;
    }
  } else {
    $("drawingWrap").innerHTML = "";
  }

  // Links
  $("linkStl").href = rev?.artifacts?.stl_url || "#";
  $("linkDims").href = rev?.artifacts?.dimensions_url || "#";

  updateUi();
}

async function approveRevision() {
  requireSession();
  if (!currentRev) throw new Error("No current revision.");
  const data = await api(`/sessions/${sessionId}/revisions/${currentRev}/approve`, { method: "POST" });
  session = data;
  $("report").textContent = JSON.stringify(data, null, 2);
  updateUi();
}

async function finalizeSession() {
  requireSession();
  const factor = Number($("pitchFactor").value || "0.5");
  if (!(factor > 0 && factor < 1)) throw new Error("pitch factor must be between 0 and 1 (e.g. 0.5)."
  );

  const payload = { target_pitch_factor: factor, note: "final" };
  const data = await api(`/sessions/${sessionId}/finalize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  // Finalize updates session status; fetch session for UI state
  session = await api(`/sessions/${sessionId}`);

  await showRevision(data);
  await refreshRevisionsList();
  updateUi();
}

async function applyChange() {
  requireSession();
  if (!currentRev) throw new Error("No current revision.");
  const change_request = String($("changeRequest").value || "").trim();
  if (!change_request) throw new Error("Enter a change request.");

  const csv_path = String($("csvPath").value || "user_api.csv").trim() || "user_api.csv";
  const model = String($("modelName").value || "gpt-4.1").trim() || "gpt-4.1";
  const payload = { change_request, note: "change", csv_path, model };

  const data = await api(`/sessions/${sessionId}/revisions/${currentRev}/change`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (session) {
    session.current_revision = data.rev;
    session.status = "concept";
    session.approved_revision = null;
    session.final_revision = null;
  }

  await showRevision(data);
  await refreshRevisionsList();
}

function wire() {
  $("btnCreateSession").onclick = () => createSession().catch((e) => alert(e.message));
  $("btnListSessions").onclick = () => listSessions().catch((e) => alert(e.message));
  $("btnLoadSession").onclick = () => loadSessionById().catch((e) => alert(e.message));

  $("btnLoadConstraints").onclick = () => loadConstraints().catch((e) => alert(e.message));
  $("btnSaveConstraints").onclick = () => saveConstraints().catch((e) => alert(e.message));

  $("btnUploadSketch").onclick = () => uploadSketch().catch((e) => alert(e.message));
  $("btnCreateFromText").onclick = () => createRevisionFromText().catch((e) => alert(e.message));
  $("btnCreateFromSpec").onclick = () => createRevisionFromSpec().catch((e) => alert(e.message));
  $("btnApprove").onclick = () => approveRevision().catch((e) => alert(e.message));
  $("btnFinalize").onclick = () => finalizeSession().catch((e) => alert(e.message));
  $("btnApplyChange").onclick = () => applyChange().catch((e) => alert(e.message));

  updateUi();
}

wire();
