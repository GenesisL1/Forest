/*
MIT License

Copyright (c) 2026 Decentralized Science Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


import { setupNav } from "./ui_nav.js";
import { setupDebugDock } from "./debug_dock.js";
import {makeLogger, loadSystem, mustAddr, nowTs, clamp, ethToWei, weiToEth, packNftFeatures, unpackNftFeatures, sigmoid, taskLabel} from "./common.js";
import { getReadProvider, getSignerProvider, getWalletState } from "./eth.js";
import { parseCSV, toNumericMatrix, inferLabelValues, toBinaryMatrix, toMulticlassMatrix, toMultilabelMatrix } from "./csv_parse.js";
import { estimateModelBytes, estimateModelBytesV2 } from "./train_gbdt.js";
import { decodeModel, predictQ, predictClassQ, predictMultiQ, parseGl1fPackage, attachGl1xFooter } from "./local_infer.js";
import { ABI_STORE, ABI_REGISTRY } from "./abis.js";

const ethers = globalThis.ethers;

const SIZE_LIMIT = 15_000_000;
const CHUNK_SIZE = 24000;
const DEFAULT_SCALE_Q = 1_000_000;
const INT32_SAFE = 2_147_480_000; 
const SEARCH_PAGE_SIZE = 25;
const PY_UI_SAMPLE_ROWS = 2048;


// ===== Python (local trainer) bridge =====
// Browser cannot directly launch Python. In "Python engine" mode we call a localhost server:
//   python3 local_trainer_server.py --port 8787
// The server caches the dataset (upload once) and trains via train_gl1f.py, returning .gl1f bytes.
//
// IMPORTANT: This file is an ES module. We must not touch DOM elements before they exist.
// We therefore bind the bridge to elements inside init() after all getElementById calls.

let trainEngineSel = null;
let pythonApiUrl = null;
let pyDatasetPill = null;
let pyUploadBtn = null;

let pyDatasetId = null;
let pyDatasetName = null;
let pyUploading = false;
let pyTrainAbort = null;

function bindPythonBridgeUI() {
  trainEngineSel = document.getElementById("trainEngineSel");
  pythonApiUrl = document.getElementById("pythonApiUrl");
  pyDatasetPill = document.getElementById("pyDatasetPill");
  pyUploadBtn = document.getElementById("pyUploadBtn");

  // Default pill
  try { _setPyDatasetPill("Dataset: not cached"); } catch {}

  if (trainEngineSel) {
    trainEngineSel.addEventListener("change", () => {
      if (_isPythonEngine()) {
        // keep current pill
      } else {
        // leaving python mode: don't assume cache is valid for current CSV
        pyDatasetId = null;
        pyDatasetName = null;
        try { _setPyDatasetPill("Dataset: not cached"); } catch {}
      }
    });
  }
}

function _isPythonEngine() {
  const v = String(trainEngineSel?.value || "browser");
  return v === "python" || v === "cpp";
}

function _localEngineName() {
  const v = String(trainEngineSel?.value || "python");
  return v === "cpp" ? "C++" : "Python";
}

function _pythonBase() {
  const raw = String(pythonApiUrl?.value || "http://127.0.0.1:8787").trim();
  return raw.replace(/\/+$/, "");
}

function _setPyDatasetPill(text, cls = "") {
  if (!pyDatasetPill) return;
  pyDatasetPill.textContent = text;
  pyDatasetPill.classList.remove("ok", "warn", "bad");
  if (cls) pyDatasetPill.classList.add(cls);
}

async function _pyPing() {
  const base = _pythonBase();
  const r = await fetch(`${base}/api/ping`, { method: "GET" });
  if (!r.ok) throw new Error(`Local trainer API ping failed (${r.status})`);
  const j = await r.json();
  // If server reports C++ availability, reflect it in the UI option.
  if (j && typeof j.supportsCpp === "boolean") {
    const optCpp = trainEngineSel?.querySelector('option[value="cpp"]');
    if (optCpp) {
      optCpp.disabled = !j.supportsCpp;
      if (!j.supportsCpp && trainEngineSel.value === "cpp") {
        trainEngineSel.value = "python";
      }
    }
  }
  return j;
}

async function _pyUploadDatasetFile(file) {
  if (!file) throw new Error("Select a CSV file first (Dataset tab)");
  if (pyUploading) throw new Error("Upload already in progress");
  pyUploading = true;
  try {
    await _pyPing();
    const base = _pythonBase();
    _setPyDatasetPill("Dataset: uploading…", "warn");
    if (pyUploadBtn) pyUploadBtn.disabled = true;

    const url = `${base}/api/upload?filename=${encodeURIComponent(file.name || "dataset.csv")}`;
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file
    });
    if (!resp.ok) {
      const t = await resp.text().catch(() => "");
      throw new Error(`Upload failed (${resp.status}): ${t || resp.statusText}`);
    }
    const out = await resp.json();
    if (!out || !out.ok || !out.datasetId) throw new Error("Upload failed: bad response");
    pyDatasetId = String(out.datasetId);
    pyDatasetName = String(out.filename || file.name || "dataset.csv");
    _setPyDatasetPill(`Dataset cached: ${pyDatasetName}`, "ok");
    return pyDatasetId;
  } finally {
    pyUploading = false;
    if (pyUploadBtn) pyUploadBtn.disabled = false;
  }
}

function _b64ToU8(b64) {
  const bin = atob(String(b64 || ""));
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i) & 255;
  return u8;
}

// Top-level helper: used by the Python-engine bridge (which runs outside init()).
// NOTE: There is also an identical helper inside init() for legacy code paths;
// keeping this one prevents ReferenceError when Python-engine is selected.
function _maybeDeepCopy(x) {
  if (x == null) return null;
  try { return JSON.parse(JSON.stringify(x)); } catch { return x; }
}

  function _sliceParsedRowsForUI(p, maxRows = PY_UI_SAMPLE_ROWS) {
    if (!p || !Array.isArray(p.rows)) return p;
    if (p.rows.length <= maxRows) return p;
    return { ...p, rows: p.rows.slice(0, maxRows) };
  }

async function runTrainRoundPython({ params, round, totalRounds, labelPrefix, parsed, selectedFeatures, selectedTask, selectedLabel, selectedLabelCols, selectedNegLabel, selectedPosLabel, selectedMultiLabels, curve, trainPill, trainBar, setDockState, nowTs, log }) {
  const base = _pythonBase();
  await _pyPing();

  if (!pyDatasetId) {
    const f = document.getElementById("csvFile")?.files?.[0];
    await _pyUploadDatasetFile(f);
  }
  if (!pyDatasetId) throw new Error("Dataset not cached (upload failed)");
  const localCurve = { steps: [], train: [], val: [], test: [], bestVal: [] };
  // NOTE: Do not clear global Plotly curves here. The caller decides when to clear.
  // During heuristic search we keep the previous curve visible until this round completes.

  const labelTxt = labelPrefix || ((totalRounds > 1) ? `Search ${round}/${totalRounds}` : "Training");
  trainPill.textContent = `${labelTxt}…`;
  setDockState("training");

  const ac = new AbortController();
  pyTrainAbort = ac;

  const overall = (totalRounds > 1) ? ((round - 1) / totalRounds) : 0;
  trainBar.style.width = `${Math.max(0, Math.min(100, Math.floor(overall * 100)))}%`;

  const headers = parsed?.headers || [];
  const featureCols = (selectedFeatures || []).map((i) => headers[i] || `col${i}`);
  const task = params.task || selectedTask;

  let labelCol = null;
  let labelCols = null;
  let negLabel = null;
  let posLabel = null;
  let classLabels = null;

  if (task === "multilabel_classification") {
    const cols = Array.isArray(selectedLabelCols) ? selectedLabelCols.slice() : [];
    labelCols = cols.map((i, k) => headers[i] || `label${k}`);
  } else {
    const idx = Number(selectedLabel);
    labelCol = headers[idx] || `label`;
  }

  if (task === "binary_classification") {
    negLabel = String(selectedNegLabel || "");
    posLabel = String(selectedPosLabel || "");
  } else if (task === "multiclass_classification") {
    classLabels = Array.isArray(selectedMultiLabels) ? selectedMultiLabels.map(String) : [];
  }

  const req = {
    task,
    engine: String(trainEngineSel?.value || "python"),
    datasetId: pyDatasetId,
    featureCols,
    labelCol,
    labelCols,
    negLabel,
    posLabel,
    classLabels,
    params: { ...params, scaleQ: "auto" }
  };

  const resp = await fetch(`${base}/api/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal: ac.signal
  }).catch((e) => {
    if (ac.signal.aborted) throw new Error("Stopped");
    throw e;
  });

  if (!resp.ok) {
    const t = await resp.text().catch(() => "");
    throw new Error(`Local training failed (${resp.status}): ${t || resp.statusText}`);
  }
  const out = await resp.json();
  if (!out || !out.ok) throw new Error(out?.error || "Local training failed");

  const meta = out.meta || {};
  const c = out.curve || {};
  localCurve.steps = Array.isArray(c.steps) ? c.steps.slice() : [];
  localCurve.train = Array.isArray(c.train) ? c.train.slice() : [];
  localCurve.val = Array.isArray(c.val) ? c.val.slice() : [];
  localCurve.test = Array.isArray(c.test) ? c.test.slice() : [];
  localCurve.bestVal = Array.isArray(c.bestVal) ? c.bestVal.slice() : [];

  curve.steps = localCurve.steps;
  curve.train = localCurve.train;
  curve.val = localCurve.val;
  curve.test = localCurve.test;
  curve.bestVal = localCurve.bestVal;

  const overallDone = (totalRounds > 1) ? (round / totalRounds) : 1;
  trainBar.style.width = `${Math.max(0, Math.min(100, Math.floor(overallDone * 100)))}%`;
  trainPill.textContent = `${labelTxt} done`;

  pyTrainAbort = null;

  const bytes = _b64ToU8(out.modelBytesB64 || "");
  return { bytes, meta, curve: localCurve, params: _maybeDeepCopy(params) };
}

async function _pyStop() {
  try {
    const base = _pythonBase();
    await fetch(`${base}/api/stop`, { method: "POST" }).catch(() => {});
  } catch {}
}

function chooseScaleQ(task, maxAbsX, maxAbsY) {
  // Keep scaleQ high for precision, but clamp so quantized int32 values won't overflow.
  let limX = INT32_SAFE;
  if (Number.isFinite(maxAbsX) && maxAbsX > 0) limX = Math.floor(INT32_SAFE / maxAbsX);

  let limY = INT32_SAFE;
  if (task === "regression") {
    if (Number.isFinite(maxAbsY) && maxAbsY > 0) limY = Math.floor(INT32_SAFE / maxAbsY);
  }

  let scaleQ = Math.min(DEFAULT_SCALE_Q, limX, limY);
  if (!Number.isFinite(scaleQ) || scaleQ < 1) scaleQ = 1;
  return Math.floor(scaleQ);
}

function setKV(el, entries) {
  if (!el) return;
  el.innerHTML = "";
  for (const [k, v] of entries) {
    const kk = document.createElement("div"); kk.className = "k"; kk.textContent = k;
    const vv = document.createElement("div"); vv.className = "v";
    if (v && typeof v === "object") {
      vv.textContent = String(v.text ?? "");
      if (v.className) vv.classList.add(...String(v.className).split(/\s+/).filter(Boolean));
    } else {
      vv.textContent = String(v);
    }
    el.appendChild(kk); el.appendChild(vv);
  }
}

function titleWordHashes(title) {
  const words = String(title || "")
    .toLowerCase()
    .split(/[\s,]+/)
    .map(w => w.trim())
    .filter(w => w.length >= 2);
  const uniq = Array.from(new Set(words));
  return uniq.map(w => ethers.keccak256(ethers.toUtf8Bytes(w)));
}

async function validateIcon128(file) {
  if (!file) throw new Error("No file selected");
  if (file.type !== "image/png") throw new Error("Icon must be a PNG");
  const buf = await file.arrayBuffer();
  const u8 = new Uint8Array(buf);

  const sig = [0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a];
  for (let i=0;i<sig.length;i++) if (u8[i] !== sig[i]) throw new Error("Invalid PNG signature");

  const blob = new Blob([u8], {type:"image/png"});
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    await new Promise((res, rej) => { img.onload=res; img.onerror=()=>rej(new Error("Failed to decode PNG")); img.src=url; });
    if (img.width !== 128 || img.height !== 128) throw new Error(`Icon must be 128×128 (got ${img.width}×${img.height})`);
  } finally {
    URL.revokeObjectURL(url);
  }
  return u8;
}

function estimateBytesForTask(trees, depth, task, nClasses) {
  const t = Math.max(0, trees | 0);
  const d = Math.max(1, depth | 0);
  if (task === "multiclass_classification" || task === "multilabel_classification") {
    const K = Math.max(2, nClasses | 0);
    return estimateModelBytesV2(t, d, K);
  }
  return estimateModelBytes(t, d);
}

function clampForSize(trees, depth, task = "regression", nClasses = 2) {
  const minT = (task === "multiclass_classification" || task === "multilabel_classification") ? 1 : 10;
  let t = Math.max(minT, Math.floor(trees));
  let d = Math.max(1, Math.floor(depth));
  if (task === "multiclass_classification" || task === "multilabel_classification") {
    const K = Math.max(2, nClasses | 0);
    const maxTPC = Math.max(1, Math.floor(65535 / K)); // registry stores nTrees as uint16
    if (t > maxTPC) t = maxTPC;
  }
  let est = estimateBytesForTask(t, d, task, nClasses);
  while (est > SIZE_LIMIT && t > minT) {
    t = Math.max(minT, t - 25);
    est = estimateBytesForTask(t, d, task, nClasses);
  }
  while (est > SIZE_LIMIT && d > 2) {
    d = Math.max(2, d - 1);
    est = estimateBytesForTask(t, d, task, nClasses);
  }
  return { trees: t, depth: d, estBytes: est };
}

// Parse a piecewise learning-rate schedule from user text.
// Supported formats (one per line, commas also accepted):
//   1-100 0.1
//   101-200:0.05
//   201-250=0.001
//   42 0.02            (single tree)
// Ranges are 1-indexed and inclusive.
function parsePiecewiseLrSchedule(text) {
  const raw = String(text || "").trim();
  if (!raw) return [];

  // Allow comma-separated rules.
  const parts = raw.replace(/,/g, "\n").split(/\n+/g);
  const segs = [];

  for (let line of parts) {
    line = String(line || "").trim();
    if (!line) continue;
    // Normalize separators
    line = line.replace(/[:=]/g, " ");
    line = line.replace(/\s+/g, " ");

    // Examples:
    //  1-100 0.1
    //  101-200 0.05
    //  42 0.02
    const m = line.match(/^\s*(\d+)(?:\s*[-–—]\s*(\d+))?\s+([+\-]?(?:\d+\.?\d*|\d*\.?\d+)(?:e[+\-]?\d+)?)\s*$/i);
    if (!m) {
      throw new Error(`Invalid LR schedule line: "${line}" (expected: start-end lr)`);
    }
    const start = parseInt(m[1], 10);
    const end = m[2] ? parseInt(m[2], 10) : start;
    const lr = Number(m[3]);
    if (!Number.isFinite(start) || start < 1) throw new Error(`Invalid schedule start: ${m[1]}`);
    if (!Number.isFinite(end) || end < start) throw new Error(`Invalid schedule end: ${m[2] || m[1]}`);
    if (!Number.isFinite(lr) || lr <= 0) throw new Error(`Invalid schedule LR: ${m[3]}`);
    segs.push({ start, end, lr });
  }

  // Sort by start, then end.
  segs.sort((a, b) => (a.start - b.start) || (a.end - b.end));

  // Detect overlaps.
  for (let i = 1; i < segs.length; i++) {
    const prev = segs[i - 1];
    const cur = segs[i];
    if (cur.start <= prev.end) {
      throw new Error(`LR schedule ranges overlap: ${prev.start}-${prev.end} and ${cur.start}-${cur.end}`);
    }
  }
  return segs;
}

function packFeaturesI32LE(vals) {
  const out = new Uint8Array(vals.length * 4);
  const dv = new DataView(out.buffer);
  for (let i=0;i<vals.length;i++) dv.setInt32(i*4, vals[i], true);
  return out;
}

function setupCreateTabs({
  tablistId = "createTabs",
  storageKey = "create.activeTab",
  defaultTab = "dataset",
  onChange = null,
} = {}) {
  const tablist = document.getElementById(tablistId);
  if (!tablist) return null;

  const tabs = Array.from(tablist.querySelectorAll('[role="tab"][data-tab]'));
  const panels = Array.from(document.querySelectorAll('[role="tabpanel"][data-tab-panel]'));
  if (!tabs.length || !panels.length) return null;

  const valid = new Set(tabs.map(t => String(t.dataset.tab || "").trim()).filter(Boolean));
  let activeTab = null;

  function tabFromHash() {
    const h = String(location.hash || "").replace(/^#/, "").trim();
    return valid.has(h) ? h : null;
  }

  function activate(nextId, { focus = false, updateHash = true, save = true } = {}) {
    const tabId = valid.has(String(nextId)) ? String(nextId) : (tabs[0]?.dataset?.tab || defaultTab);
    activeTab = tabId;

    for (const t of tabs) {
      const on = String(t.dataset.tab) === tabId;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
    }
    for (const p of panels) {
      const on = String(p.dataset.tabPanel) === tabId;
      p.classList.toggle("active", on);
      p.hidden = !on;
    }

    if (save) {
      try { localStorage.setItem(storageKey, tabId); } catch {}
    }
    if (updateHash) {
      const newHash = `#${tabId}`;
      if (location.hash !== newHash) {
        history.replaceState(null, "", newHash);
      }
    }
    if (focus) {
      const t = tabs.find(x => String(x.dataset.tab) === tabId);
      try { t?.focus(); } catch {}
    }

    try { onChange?.(tabId); } catch {}
  }

  // Click / keyboard support
  for (const t of tabs) {
    t.addEventListener("click", (e) => {
      e.preventDefault();
      activate(String(t.dataset.tab), { focus: false });
    });

    t.addEventListener("keydown", (e) => {
      const idx = tabs.indexOf(t);
      if (e.key === "ArrowRight" || e.key === "ArrowLeft" || e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const dir = (e.key === "ArrowRight" || e.key === "ArrowDown") ? 1 : -1;
        const next = (idx + dir + tabs.length) % tabs.length;
        activate(String(tabs[next].dataset.tab), { focus: true });
      } else if (e.key === "Home") {
        e.preventDefault();
        activate(String(tabs[0].dataset.tab), { focus: true });
      } else if (e.key === "End") {
        e.preventDefault();
        activate(String(tabs[tabs.length - 1].dataset.tab), { focus: true });
      }
    });
  }

  // Jump buttons ("Next" / "Back") inside panels
  const jumpers = Array.from(document.querySelectorAll('[data-tab-jump]'));
  for (const btn of jumpers) {
    btn.addEventListener("click", (e) => {
      const id = btn.getAttribute("data-tab-jump");
      if (!id) return;
      e.preventDefault();
      activate(id, { focus: true });
      try { tablist.scrollIntoView({ behavior: "smooth", block: "start" }); } catch {}
    });
  }

  // Deep link / back-forward support
  window.addEventListener("hashchange", () => {
    const id = tabFromHash();
    if (id) activate(id, { updateHash: false, save: true });
  });

  // Initial tab selection
  let initial = tabFromHash();
  if (!initial) {
    try {
      const v = localStorage.getItem(storageKey);
      if (valid.has(String(v))) initial = String(v);
    } catch {}
  }
  if (!initial) initial = defaultTab;

  activate(initial, { focus: false, updateHash: true, save: false });
  return { activate, getActive: () => activeTab };
}

document.addEventListener("DOMContentLoaded", async () => {
  // Route logs into the floating debug dock (instead of the old bottom-of-page log panel).
  setupNav({ active: "create", logElId: "debugLines" });

  const dbg = setupDebugDock({ state: "idle" });
  const log = dbg.log;
  const setDockState = dbg.setDockState;
  const setDockConn = dbg.setDockConn;

  const dlog = makeLogger(document.getElementById("deployLog"));

  // Dataset elements
  const csvFile = document.getElementById("csvFile");
  const taskSel = document.getElementById("taskSel");
  const labelCol = document.getElementById("labelCol");
  const singleLabelRow = document.getElementById("singleLabelRow");
  // Multilabel UI
  const multiLabelBox = document.getElementById("multiLabelBox");
  const multiLabelSel = document.getElementById("multiLabelSel");
  const multiLabelAllBtn = document.getElementById("multiLabelAllBtn");
  const multiLabelClearBtn = document.getElementById("multiLabelClearBtn");
  const multiLabelNote = document.getElementById("multiLabelNote");
  const classBox = document.getElementById("classBox");
  const negClassSel = document.getElementById("negClassSel");
  const posClassSel = document.getElementById("posClassSel");
  const swapClassesBtn = document.getElementById("swapClassesBtn");
  const classNote = document.getElementById("classNote");
  // Multiclass UI
  const multiClassBox = document.getElementById("multiClassBox");
  const multiClassSel = document.getElementById("multiClassSel");
  const multiClassUpBtn = document.getElementById("multiClassUpBtn");
  const multiClassDownBtn = document.getElementById("multiClassDownBtn");
  const multiClassAllBtn = document.getElementById("multiClassAllBtn");
  const multiClassClearBtn = document.getElementById("multiClassClearBtn");
  const multiClassNote = document.getElementById("multiClassNote");
  const featureCols = document.getElementById("featureCols");
  const dsKV = document.getElementById("dsKV");
  const dsNotes = document.getElementById("dsNotes");

  // Dataset 3D distribution (Data Galaxy)
  const ds3dDetails = document.getElementById("ds3dDetails");
  const ds3dHint = document.getElementById("ds3dHint");
  const ds3dMode = document.getElementById("ds3dMode");
  const ds3dSample = document.getElementById("ds3dSample");
  const ds3dColor = document.getElementById("ds3dColor");
  const ds3dFeatureBox = document.getElementById("ds3dFeatureBox");
  const ds3dX = document.getElementById("ds3dX");
  const ds3dY = document.getElementById("ds3dY");
  const ds3dZ = document.getElementById("ds3dZ");
  const ds3dPlot = document.getElementById("ds3dPlot");
  const ds3dNote = document.getElementById("ds3dNote");

  // Class imbalance (dataset-level)
  const imbalanceDetails = document.getElementById("imbalanceDetails");
  const imbSummaryHint = document.getElementById("imbSummaryHint");
  const imbMode = document.getElementById("imbMode");
  const imbCap = document.getElementById("imbCap");
  const imbNormalize = document.getElementById("imbNormalize");
  const imbStratify = document.getElementById("imbStratify");
  const imbNote = document.getElementById("imbNote");
  const imbRows = document.getElementById("imbRows");

  // Feature importance (post-train)
  const featImpBox = document.getElementById("featImpBox");
  const featImpTable = document.getElementById("featImpTable");
  const featImpNote = document.getElementById("featImpNote");

  // Training elements
  const treesRange = document.getElementById("treesRange");
  const treesNum = document.getElementById("treesNum");
  const depthRange = document.getElementById("depthRange");
  const depthNum = document.getElementById("depthNum");
  const lrRange = document.getElementById("lrRange");
  const lrNum = document.getElementById("lrNum");
  // Learning rate schedule
  const lrSchedMode = document.getElementById("lrSchedMode");
  const lrPlateauBox = document.getElementById("lrPlateauBox");
  const lrPlateauN = document.getElementById("lrPlateauN");
  const lrPlateauPct = document.getElementById("lrPlateauPct");
  const lrPlateauMin = document.getElementById("lrPlateauMin");
  const lrPiecewiseBox = document.getElementById("lrPiecewiseBox");
  const lrScheduleText = document.getElementById("lrScheduleText");
  const lrScheduleExampleBtn = document.getElementById("lrScheduleExampleBtn");
  const lrScheduleClearBtn = document.getElementById("lrScheduleClearBtn");
  const minLeafRange = document.getElementById("minLeafRange");
  const minLeafNum = document.getElementById("minLeafNum");
  const binsRange = document.getElementById("binsRange");
  const binsNum = document.getElementById("binsNum");
  const binningMode = document.getElementById("binningMode");
  const seedNum = document.getElementById("seedNum");

  // Dataset split controls (train / val / test)
  const trainSplitRange = document.getElementById("trainSplitRange");
  const trainSplitNum = document.getElementById("trainSplitNum");
  const valSplitRange = document.getElementById("valSplitRange");
  const valSplitNum = document.getElementById("valSplitNum");
  const testSplitPill = document.getElementById("testSplitPill");
  const splitCounts = document.getElementById("splitCounts");
  const earlyStopOn = document.getElementById("earlyStopOn");
  const patienceRange = document.getElementById("patienceRange");
  const patienceNum = document.getElementById("patienceNum");
  const refitOn = document.getElementById("refitOn");

  const sizeKV = document.getElementById("sizeKV");
  const sizeNote = document.getElementById("sizeNote");
  const metricsKV = document.getElementById("metricsKV");
  const trainBtn = document.getElementById("trainBtn");
  const stopBtn = document.getElementById("stopBtn");

  // Bind Python training engine UI (safe: after DOM element lookups)
  try { bindPythonBridgeUI(); } catch {}

  // Heuristic search (optional)
  const searchDetails = document.getElementById("searchDetails");
  const heuristicSearchOn = document.getElementById("heuristicSearchOn");
  const heuristicSearchRounds = document.getElementById("heuristicSearchRounds");
  const heuristicSearchClearBtn = document.getElementById("heuristicSearchClearBtn");
  const searchTable = document.getElementById("searchTable");
  // Search table pagination (25 rows/page)
  const searchPager = document.getElementById("searchPager");
  const searchPrevBtn = document.getElementById("searchPrevBtn");
  const searchNextBtn = document.getElementById("searchNextBtn");
  const searchPageInfo = document.getElementById("searchPageInfo");
  // Stop is only meaningful while training is running.
  stopBtn.disabled = true;
  const trainPill = document.getElementById("trainPill");
  const trainBar = document.getElementById("trainBar");
  const curvePlot = document.getElementById("curvePlot");
  const previewCurvePlot = document.getElementById("previewCurvePlot");
  // Training curve points. We also include `bestVal`, which is monotonic
  // (best validation metric seen so far) so users can clearly see the plateau
  // even if the raw validation curve jitters.
  const curve = { steps: [], train: [], val: [], test: [], bestVal: [] };
  let _curveRev = 0; // Plotly datarevision counter (needed when arrays are mutated in-place)


  const _curvePlotCfg = { responsive: true, displayModeBar: false };

  function _curveLayout(compact = false) {
    const empty = (curve.steps.length < 2);
    // Plotly's default autorange uses "nice" tick rounding + padding, which can
    // make curves look almost flat/linear (especially when loss improves only a
    // little). The previous canvas chart used the exact min/max range, which
    // visually emphasized the improvement and the plateau. We reproduce that
    // tighter scaling here so the curve shape matches prior behavior.
    const layout = {
      margin: compact ? { l: 52, r: 12, t: 18, b: 44 } : { l: 52, r: 12, t: 18, b: 44 },
      xaxis: { title: "round", automargin: true, zeroline: false },
      yaxis: { title: "metric", automargin: true, zeroline: false },
      legend: { orientation: "h", x: 0, y: 1.18 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };

    // Plotly.react only notices array changes if either you pass new array
    // references, or you bump layout.datarevision. We mutate arrays in-place
    // (push) for performance, so we must bump datarevision to keep the chart
    // updating beyond the first couple points.
    layout.datarevision = _curveRev;

    // Tight y-range (like the old canvas chart) — computed with a loop (no
    // spread/Math.min(...arr)) to avoid browser argument limits on large runs.
    layout.yaxis.rangemode = "normal";
    let ymin = Infinity;
    let ymax = -Infinity;
    let count = 0;
    const scan = (arr) => {
      if (!Array.isArray(arr)) return;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (!Number.isFinite(v)) continue;
        count++;
        if (v < ymin) ymin = v;
        if (v > ymax) ymax = v;
      }
    };
    scan(curve.train);
    scan(curve.val);
    scan(curve.test);
    scan(curve.bestVal);

    if (count >= 2 && Number.isFinite(ymin) && Number.isFinite(ymax)) {
      if (ymin === ymax) { ymin = ymin - 1; ymax = ymax + 1; }
      // A tiny padding avoids clipping the line at the border, while keeping the
      // "full drop" look.
      const pad = (ymax - ymin) * 0.01;
      layout.yaxis.autorange = false;
      layout.yaxis.range = [ymin - pad, ymax + pad];
    } else {
      layout.yaxis.autorange = true;
    }
    if (empty) {
      layout.annotations = [{
        text: "Train to see curve…",
        xref: "paper", yref: "paper",
        x: 0, y: 1,
        xanchor: "left", yanchor: "top",
        showarrow: false,
        font: { size: 14, color: "#64748b" }
      }];
    }
    return layout;
  }

  function _curveTraces() {
    const x = curve.steps;
    const traces = [];
    const add = (name, y, color = null, dash = null) => {
      if (!Array.isArray(y) || !y.length) return;
      // Disable Plotly line simplification; otherwise it may collapse many near-
      // collinear points into an almost-straight line, hiding the plateau.
      const line = { simplify: false, width: 2 };
      if (color) line.color = color;
      if (dash) line.dash = dash;
      traces.push({ x, y, type: "scatter", mode: "lines", name, line });
    };
    // Match the previous canvas chart colors for continuity.
    add("train", curve.train, "#2563eb");
    add("val", curve.val, "#0f172a");
    add("best val", curve.bestVal, "#334155", "dot");
    add("test", curve.test, "#22c55e");
    if (!traces.length) traces.push({ x: [], y: [], type: "scatter", mode: "lines", name: "train", line: { simplify: false, width: 2, color: "#2563eb" } });
    return traces;
  }

  function drawCurves() {
    if (!globalThis.Plotly) return;
    const data = _curveTraces();
    try {
      if (curvePlot) globalThis.Plotly.react(curvePlot, data, _curveLayout(false), _curvePlotCfg);
    } catch {}
    try {
      if (previewCurvePlot) globalThis.Plotly.react(previewCurvePlot, data, _curveLayout(true), _curvePlotCfg);
    } catch {}
  }

  function resetCurve() {
    curve.steps = []; curve.train = []; curve.val = []; curve.test = []; curve.bestVal = [];
    _curveRev++;
    drawCurves();
  }

  // Create page tabs (Dataset / Training / Local preview / Mint)
  const tabsApi = setupCreateTabs({
    onChange: (tabId) => {
      // Training curve is Plotly-based; redraw and resize once the tab is visible.
      if (tabId === "training" || tabId === "preview") {
        requestAnimationFrame(() => {
          try { drawCurves(); } catch {}
          try {
            if (globalThis.Plotly) {
              if (curvePlot) globalThis.Plotly.Plots.resize(curvePlot);
              if (previewCurvePlot) globalThis.Plotly.Plots.resize(previewCurvePlot);
            }
          } catch {}
        });
      }

      // Plotly needs a resize after becoming visible.
      if (tabId === "dataset") {
        requestAnimationFrame(() => {
          try {
            if (globalThis.Plotly && ds3dPlot) {
              globalThis.Plotly.Plots.resize(ds3dPlot);
            }
          } catch {}
          try { ds3dRefreshControls(); ds3dScheduleRender("tab"); } catch {}
        });
      }
    },
  });

  // Keep Plotly charts sized correctly on window resize
  window.addEventListener("resize", () => {
    try {
      const active = tabsApi?.getActive?.();
      if ((active === "training" || active === "preview") && globalThis.Plotly) {
        if (curvePlot) globalThis.Plotly.Plots.resize(curvePlot);
        if (previewCurvePlot) globalThis.Plotly.Plots.resize(previewCurvePlot);
      }
    } catch {}

    try {
      if (tabsApi?.getActive?.() === "dataset" && globalThis.Plotly && ds3dPlot) {
        globalThis.Plotly.Plots.resize(ds3dPlot);
      }
    } catch {}
  });


  // Preview elements
  const previewFeatHint = document.getElementById("previewFeatHint");
  const previewFeatGrid = document.getElementById("previewFeatGrid");
  const previewBtn = document.getElementById("previewBtn");
  const rowIndex = document.getElementById("rowIndex");
  const loadRowBtn = document.getElementById("loadRowBtn");
  const compareRowBtn = document.getElementById("compareRowBtn");
  const previewKV = document.getElementById("previewKV");
  const previewMetricsKV = document.getElementById("previewMetricsKV");
  const previewParamsKV = document.getElementById("previewParamsKV");
  const previewParamsNote = document.getElementById("previewParamsNote");

  // Local preview: save/load .gl1f package
  const exportGl1fBtn = document.getElementById("exportGl1fBtn");
  const importGl1fFile = document.getElementById("importGl1fFile");
  const gl1fPill = document.getElementById("gl1fPill");

  // Deploy elements
  const metaName = document.getElementById("metaName");
  const metaDesc = document.getElementById("metaDesc");
  const iconFile = document.getElementById("iconFile");

const ownerKeyAddr = document.getElementById("ownerKeyAddr");
const ownerKeyPriv = document.getElementById("ownerKeyPriv");
const genOwnerKeyBtn = document.getElementById("genOwnerKeyBtn");
const copyOwnerKeyBtn = document.getElementById("copyOwnerKeyBtn");
const downloadOwnerKeyBtn = document.getElementById("downloadOwnerKeyBtn");
const ownerKeySaved = document.getElementById("ownerKeySaved");

let ownerKeyWallet = null;
function genOwnerKey() {
  ownerKeyWallet = ethers.Wallet.createRandom();
  if (ownerKeyAddr) ownerKeyAddr.value = ownerKeyWallet.address;
  if (ownerKeyPriv) ownerKeyPriv.value = ownerKeyWallet.privateKey;
  if (ownerKeySaved) ownerKeySaved.checked = false;
}

if (genOwnerKeyBtn) genOwnerKeyBtn.addEventListener("click", () => {
  genOwnerKey();
  log("Generated a new owner API key. Save the private key now.");
  try { updateDeployState(); } catch {}
});


async function _copyTextCompat(text) {
  // Clipboard API is only available in secure contexts (https, localhost). 0.0.0.0 is NOT secure.
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function' && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (_) {}

  // Fallback: execCommand('copy')
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    ta.style.top = '0';
    document.body.appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, ta.value.length);
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return !!ok;
  } catch (_) {
    return false;
  }
}

function _downloadTextFile(filename, text) {
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function _downloadBytesFile(filename, bytesU8) {
  const u8 = (bytesU8 instanceof Uint8Array) ? bytesU8 : new Uint8Array(bytesU8);
  const blob = new Blob([u8], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function _bytesToBase64(bytesU8) {
  const u8 = (bytesU8 instanceof Uint8Array) ? bytesU8 : new Uint8Array(bytesU8);
  let bin = "";
  const CH = 0x8000;
  for (let i = 0; i < u8.length; i += CH) {
    bin += String.fromCharCode(...u8.subarray(i, i + CH));
  }
  return btoa(bin);
}

function _base64ToBytes(b64) {
  const bin = atob(String(b64 || ""));
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

function _safeSlug(s) {
  return String(s || "model")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9\-_]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 64) || "model";
}


if (copyOwnerKeyBtn) copyOwnerKeyBtn.addEventListener('click', async () => {
  const t = (ownerKeyPriv?.value || '').trim();
  if (!t) return;
  const ok = await _copyTextCompat(t);
  if (ok) {
    log('Copied private key to clipboard.');
  } else {
    log('Clipboard copy unavailable on this URL — use Download or copy manually.');
  }
});

if (downloadOwnerKeyBtn) downloadOwnerKeyBtn.addEventListener('click', () => {
  if (!ownerKeyWallet) genOwnerKey();
  const addr = ownerKeyWallet?.address || (ownerKeyAddr?.value || '');
  const pk = ownerKeyWallet?.privateKey || (ownerKeyPriv?.value || '');
  const ts = new Date().toISOString().replace(/[:.]/g,'-');
  const name = `genesisl1-forest-owner-key-${ts}.txt`;
  const body = `GenesisL1 Forest — Owner API Key

Address: ${addr}
PrivateKey: ${pk}

KEEP THIS SECRET. Anyone with the private key can use owner-level API inference for this model.`;
  _downloadTextFile(name, body);
  log('Downloaded owner API key file.');
});

// Auto-generate on first load

if (ownerKeyAddr && !ownerKeyAddr.value) {
  genOwnerKey();
}

  const pricingMode = document.getElementById("pricingMode");
  const pricingFee = document.getElementById("pricingFee");
  const pricingRecipient = document.getElementById("pricingRecipient");
  const licenseLine = document.getElementById("licenseLine");
  const tosLine = document.getElementById("tosLine");
  const agreeTos = document.getElementById("agreeTos");
  const agreeLicense = document.getElementById("agreeLicense");
  const deployEstKV = document.getElementById("deployEstKV");
  const deployEstNote = document.getElementById("deployEstNote");
  const deployBtn = document.getElementById("deployBtn");
  const deployPill = document.getElementById("deployPill");

  // State
  let parsed = null;
  let selectedTask = "regression"; // regression | binary_classification | multiclass_classification | multilabel_classification
  let selectedLabel = null;
  let selectedLabelCols = []; // label column indices for multilabel
  let selectedFeatures = [];
  let labelValuesInfo = null; // inferred label values (for classification)
  let selectedNegLabel = null;
  let selectedPosLabel = null;
  let selectedMultiLabels = []; // ordered list of selected class labels for multiclass
  // Class imbalance handling (manual weights stored client-side)
  const _imbManual = {
    binary: { w0: 1, w1: 1 },
    multiclass: Object.create(null), // label -> weight
    multilabel: Object.create(null),  // colIdx -> posWeight
  };

  let datasetNumeric = null;
  let trained = null; // {bytes, decoded, modelId, meta}

  // Local preview save/load (.gl1f) state
  let _gl1fLoadedName = null;
  let _gl1fLoadedHasFooter = false;
  let _gl1fLoadedPkg = null;


  // Keep the exact train parameters we used so we can compute post-train
  // feature importance (including test-split permutation importance) consistently.
  let lastTrainInfo = null; // {task, seed, splitTrain, splitVal, nRows, nFeatures, nClasses}

  // Feature-importance computation is async (permutation importance can take time).
  // Use a nonce to cancel stale computations when the dataset/model changes.
  let _featImpNonce = 0;
  let _featImpLast = null; // cached last rendered importance rows
  let worker = null;
  let isTraining = false;
  // Heuristic search state
  let isSearching = false;
  let searchAbort = false;
  let searchHistory = []; // [{round,status,params,meta,curve,error}]
  let searchPage = 1; // 1-indexed page for heuristic search table
  let _activeTrainReject = null;
  let iconBytes = null;

  let activeTosVersion = 0;
  let activeLicenseId = 0;

  // Deploy estimate state (must be initialized before updateSize()/updateDeployState() can run).
  let _deployEstTimer = null;
  let _deployEstNonce = 0;

  // Dataset 3D distribution render state
  let _ds3dTimer = null;
  let _ds3dNonce = 0;
  let _ds3dLastKey = "";
  let _ds3dLastSample = null; // cached sample to avoid rescanning on pure UI-only changes


  // ---- Local preview .gl1f save/load ----

  function updateGl1fUI() {
    try {
      if (exportGl1fBtn) exportGl1fBtn.disabled = !(trained?.bytes?.length);
    } catch {}
    try {
      if (!gl1fPill) return;
      if (_gl1fLoadedName) {
        gl1fPill.textContent = `Loaded: ${_gl1fLoadedName}${_gl1fLoadedHasFooter ? "" : " (no meta)"}`;
      } else if (trained?.bytes?.length) {
        gl1fPill.textContent = "Model file: in memory";
      } else {
        gl1fPill.textContent = "Model file: none";
      }
    } catch {}
  }

  function _buildFeaturesPackedForCurrentModel(task) {
    const t = String(task || selectedTask || "regression");
    const featureNames = datasetNumeric?.featureNames || [];
    let labelName = datasetNumeric?.labelName || "";
    let labels = null;
    let labelNames = null;

    if (t === "binary_classification") {
      const c = datasetNumeric?.classes || {};
      const a0 = (c && (c[0] != null)) ? String(c[0]) : String(selectedNegLabel || "0");
      const a1 = (c && (c[1] != null)) ? String(c[1]) : String(selectedPosLabel || "1");
      labels = [a0, a1];
      if (!labelName) labelName = "(binary)";
    } else if (t === "multiclass_classification") {
      const arr = Array.isArray(datasetNumeric?.classes) ? datasetNumeric.classes : (Array.isArray(selectedMultiLabels) ? selectedMultiLabels : []);
      labels = arr && arr.length ? arr.map((x)=>String(x)) : null;
      if (!labelName) labelName = "(multiclass)";
    } else if (t === "multilabel_classification") {
      labelNames = Array.isArray(datasetNumeric?.labelNames) ? datasetNumeric.labelNames.map((x)=>String(x)) : null;
      labels = ["0","1"];
      labelName = "(multilabel)";
    }

    return packNftFeatures({ task: t, featureNames, labelName, labels, labelNames });
  }

  function _buildGl1fExportPackage() {
    if (!trained?.bytes?.length) throw new Error("Train or load a model first");

    const task = String(trained?.meta?.task || selectedTask || "regression");
    const title = (metaName?.value || "").trim();
    const description = (metaDesc?.value || "").trim();

    const featuresPacked = _buildFeaturesPackedForCurrentModel(task);
    const words = title ? titleWordHashes(title) : [];

    let mode = 0;
    let feeWeiStr = "0";
    try {
      mode = Number(pricingMode?.value || 0);
      if (mode === 0) feeWeiStr = "0";
      else feeWeiStr = String(ethToWei(String(pricingFee?.value || "0")));
    } catch {}

    const recipient = String(pricingRecipient?.value || "").trim();
    const ownerKey = String(ownerKeyAddr?.value || "").trim();

    const pkg = {
      kind: "GL1F_PACKAGE",
      v: 1,
      createdAt: new Date().toISOString(),
      chainId: 29,
      chunkSize: CHUNK_SIZE,
      model: {
        gl1fVersion: Number(trained?.decoded?.version || 1),
        nFeatures: Number(trained?.decoded?.nFeatures || 0),
        depth: Number(trained?.decoded?.depth || 0),
        scaleQ: Number(trained?.decoded?.scaleQ || 0),
        bytes: Number(trained?.bytes?.length || 0),
      },
      nft: {
        title,
        description,
        iconPngB64: iconBytes?.length ? _bytesToBase64(iconBytes) : null,
        featuresPacked,
        titleWordHashes: words,
      },
      registry: {
        pricingMode: mode,
        feeWei: feeWeiStr,
        recipient,
        ownerKey,
        tosVersionAccepted: Number(activeTosVersion || 0),
        licenseIdAccepted: Number(activeLicenseId || 0),
      },
      local: {
        trainMeta: trained?.meta || null,
        trainParams: trained?.params || null,
        curve: {
          steps: Array.isArray(curve.steps) ? curve.steps.slice() : [],
          train: Array.isArray(curve.train) ? curve.train.slice() : [],
          val: Array.isArray(curve.val) ? curve.val.slice() : [],
          test: Array.isArray(curve.test) ? curve.test.slice() : [],
          bestVal: Array.isArray(curve.bestVal) ? curve.bestVal.slice() : [],
        },
      }
    };

    // IMPORTANT: attach footer, but keep trained.bytes core-only (deploy stays identical)
    const outBytes = attachGl1xFooter(trained.bytes, pkg);
    return { pkg, outBytes };
  }

  async function _loadGl1fFile(file) {
    const u8 = new Uint8Array(await file.arrayBuffer());
    const { modelBytes, pkg, hasFooter } = parseGl1fPackage(u8);

    // Parse featuresPacked if present (this is what makes mint+preview work without CSV)
    let fpMeta = null;
    let fpFeatures = null;
    try {
      const fpRaw = (pkg && pkg.nft && typeof pkg.nft.featuresPacked === "string") ? pkg.nft.featuresPacked : "";
      if (fpRaw) {
        const u = unpackNftFeatures(fpRaw);
        fpMeta = u.meta;
        fpFeatures = u.features;
      }
    } catch {}

    const decoded = decodeModel(modelBytes);

    // Decide task
    let task = (pkg && pkg.local && pkg.local.trainMeta && typeof pkg.local.trainMeta.task === "string") ? pkg.local.trainMeta.task : null;
    if (!task && fpMeta && typeof fpMeta.task === "string") task = fpMeta.task;
    if (!task) task = (decoded.version === 2) ? "multiclass_classification" : (selectedTask || "regression");
    task = String(task || "regression");

    // Feature names
    let featureNames = [];
    if (Array.isArray(fpFeatures) && fpFeatures.length) featureNames = fpFeatures.map(String);

    if (!featureNames.length) {
      featureNames = Array.from({ length: Number(decoded.nFeatures || 0) }, (_, i) => `f${i}`);
    } else if (featureNames.length !== Number(decoded.nFeatures || 0)) {
      const n = Number(decoded.nFeatures || 0);
      featureNames = featureNames.slice(0, n);
      while (featureNames.length < n) featureNames.push(`f${featureNames.length}`);
    }

    // Labels/classes
    let labelName = (fpMeta && typeof fpMeta.labelName === "string") ? fpMeta.labelName : "label";
    let classes = null;
    let labelNames = null;

    if (task === "binary_classification") {
      const labs = (fpMeta && Array.isArray(fpMeta.labels) && fpMeta.labels.length >= 2) ? fpMeta.labels : ["0","1"];
      classes = { 0: String(labs[0]), 1: String(labs[1]) };
      selectedNegLabel = String(labs[0]);
      selectedPosLabel = String(labs[1]);
    } else if (task === "multiclass_classification") {
      const labs = (fpMeta && Array.isArray(fpMeta.labels) && fpMeta.labels.length >= 2) ? fpMeta.labels : Array.from({ length: Number(decoded.nClasses || 2) }, (_, i) => String(i));
      classes = labs.map(String);
      selectedMultiLabels = classes.slice();
    } else if (task === "multilabel_classification") {
      labelNames = (fpMeta && Array.isArray(fpMeta.labelNames) && fpMeta.labelNames.length >= 1) ? fpMeta.labelNames.map(String) : [];
      const n = Number(decoded.nClasses || labelNames.length || 2);
      labelNames = labelNames.slice(0, n);
      while (labelNames.length < n) labelNames.push(`label${labelNames.length}`);
      labelName = "(multilabel)";
      classes = { 0: "0", 1: "1" };
    }

    // Minimal dataset for preview+mint (no raw rows)
    datasetNumeric = {
      featureNames,
      X: [],
      y: [],
      droppedRows: 0,
      labelName,
      labelNames,
      classes,
    };

    selectedTask = task;
    selectedFeatures = Array.from({ length: featureNames.length }, (_, i) => i);
    labelValuesInfo = null;

    // Restore curve (if present), else blank
    const c = (pkg && pkg.local && pkg.local.curve && typeof pkg.local.curve === "object") ? pkg.local.curve : null;
    if (c && Array.isArray(c.steps)) {
      curve.steps = c.steps.slice();
      curve.train = Array.isArray(c.train) ? c.train.slice() : [];
      curve.val = Array.isArray(c.val) ? c.val.slice() : [];
      curve.test = Array.isArray(c.test) ? c.test.slice() : [];
      curve.bestVal = Array.isArray(c.bestVal) ? c.bestVal.slice() : [];
      _curveRev++;
      drawCurves();
    } else {
      resetCurve();
    }

    // Restore mint fields (best-effort)
    try {
      if (pkg && pkg.nft) {
        if (metaName && typeof pkg.nft.title === "string") metaName.value = pkg.nft.title;
        if (metaDesc && typeof pkg.nft.description === "string") metaDesc.value = pkg.nft.description;
        if (pkg.nft.iconPngB64) {
          try { iconBytes = _base64ToBytes(pkg.nft.iconPngB64); } catch {}
        }
      }
    } catch {}

    try {
      if (pkg && pkg.registry) {
        if (pricingMode && pkg.registry.pricingMode != null) pricingMode.value = String(pkg.registry.pricingMode);
        if (pricingRecipient && typeof pkg.registry.recipient === "string") pricingRecipient.value = pkg.registry.recipient;
        if (pricingFee && pkg.registry.feeWei != null) {
          try { pricingFee.value = weiToEth(BigInt(String(pkg.registry.feeWei || "0"))); } catch {}
        }
        if (ownerKeyAddr && typeof pkg.registry.ownerKey === "string" && pkg.registry.ownerKey.trim()) ownerKeyAddr.value = pkg.registry.ownerKey.trim();
        if (ownerKeySaved) ownerKeySaved.checked = false;
      }
    } catch {}

    const meta = (pkg && pkg.local && pkg.local.trainMeta && typeof pkg.local.trainMeta === "object") ? pkg.local.trainMeta : { task };
    const params = (pkg && pkg.local && pkg.local.trainParams && typeof pkg.local.trainParams === "object") ? pkg.local.trainParams : null;

    // IMPORTANT: load core bytes only → deploy is identical to trained
    applyTrainedModel({ bytes: modelBytes, meta, params });

    try { applyParamsToTrainingUI(params); } catch {}
    try { updateTaskUI(); } catch {}
    try { updateSize(); } catch {}
    try { renderPreviewInputs(); } catch {}
    try { updateDeployState(); } catch {}

    _gl1fLoadedName = file.name;
    _gl1fLoadedHasFooter = !!hasFooter && !!pkg;
    _gl1fLoadedPkg = pkg;
    updateGl1fUI();

    log(`[${nowTs()}] Loaded .gl1f: ${file.name} (${u8.length.toLocaleString()} bytes; model ${modelBytes.length.toLocaleString()} bytes; task=${task})`);
  }

  if (exportGl1fBtn) exportGl1fBtn.addEventListener("click", () => {
    try {
      const { outBytes } = _buildGl1fExportPackage();
      const title = (metaName?.value || "").trim();
      const ts = new Date().toISOString().replace(/[:.]/g, "-");
      const fname = `forest-${_safeSlug(title)}-${ts}.gl1f`;
      _downloadBytesFile(fname, outBytes);
      log(`[${nowTs()}] Exported .gl1f package: ${fname} (${outBytes.length.toLocaleString()} bytes)`);
    } catch (e) {
      log(`[${nowTs()}] [error] ${e.message || e}`);
    }
  });

  if (importGl1fFile) importGl1fFile.addEventListener("change", async () => {
    const f = importGl1fFile.files?.[0];
    if (!f) return;
    try {
      await _loadGl1fFile(f);
    } catch (e) {
      log(`[${nowTs()}] [error] Failed to load .gl1f: ${e.message || e}`);
    } finally {
      try { importGl1fFile.value = ""; } catch {}
    }
  });


  function syncRange(rangeEl, numEl) {
    const fromRange = () => { numEl.value = rangeEl.value; updateSize(); };
    const fromNum = () => { rangeEl.value = numEl.value; updateSize(); };
    rangeEl.addEventListener("input", fromRange);
    numEl.addEventListener("input", fromNum);
    fromRange();
  }
  syncRange(treesRange, treesNum);
  syncRange(depthRange, depthNum);
  lrRange.addEventListener("input", () => { lrNum.value = lrRange.value; });
  lrNum.addEventListener("input", () => { lrRange.value = lrNum.value; });

  // Learning-rate schedule UI
  function updateLrScheduleUI() {
    const mode = String(lrSchedMode?.value || "none");
    if (lrPlateauBox) lrPlateauBox.style.display = (mode === "plateau") ? "" : "none";
    if (lrPiecewiseBox) lrPiecewiseBox.style.display = (mode === "piecewise") ? "" : "none";
  }
  if (lrSchedMode) lrSchedMode.addEventListener("change", updateLrScheduleUI);
  updateLrScheduleUI();

  if (lrScheduleExampleBtn) lrScheduleExampleBtn.addEventListener("click", () => {
    if (!lrScheduleText) return;
    lrScheduleText.value = "1-100 0.1\n101-200 0.05\n201-250 0.001";
    if (lrSchedMode) lrSchedMode.value = "piecewise";
    updateLrScheduleUI();
  });
  if (lrScheduleClearBtn) lrScheduleClearBtn.addEventListener("click", () => {
    if (lrScheduleText) lrScheduleText.value = "";
  });

  minLeafRange.addEventListener("input", () => { minLeafNum.value = minLeafRange.value; });
  minLeafNum.addEventListener("input", () => { minLeafRange.value = minLeafNum.value; });
  if (binsRange && binsNum) {
    binsRange.addEventListener("input", () => { binsNum.value = binsRange.value; });
    binsNum.addEventListener("input", () => { binsRange.value = binsNum.value; });
  }
  patienceRange.addEventListener("input", () => { patienceNum.value = patienceRange.value; });
  patienceNum.addEventListener("input", () => { patienceRange.value = patienceNum.value; });

  // Heuristic search UI
  function updateHeuristicSearchUI() {
    const on = !!heuristicSearchOn?.checked;
    if (heuristicSearchRounds) heuristicSearchRounds.disabled = !on;
    try { if (on && searchDetails) searchDetails.open = true; } catch {}
  }
  if (heuristicSearchOn) heuristicSearchOn.addEventListener("change", updateHeuristicSearchUI);
  updateHeuristicSearchUI();

  if (heuristicSearchClearBtn) heuristicSearchClearBtn.addEventListener("click", () => {
    searchHistory = [];
    searchPage = 1;
    renderSearchTable();
  });

  // Heuristic search table pagination (25 per page)
  if (searchPrevBtn) searchPrevBtn.addEventListener("click", () => {
    const n = searchHistory?.length || 0;
    const pages = Math.max(1, Math.ceil(n / SEARCH_PAGE_SIZE));
    searchPage = Math.max(1, Math.min(pages, (searchPage | 0) - 1));
    renderSearchTable();
  });
  if (searchNextBtn) searchNextBtn.addEventListener("click", () => {
    const n = searchHistory?.length || 0;
    const pages = Math.max(1, Math.ceil(n / SEARCH_PAGE_SIZE));
    searchPage = Math.max(1, Math.min(pages, (searchPage | 0) + 1));
    renderSearchTable();
  });

  // Render empty table on load.
  renderSearchTable();


  function clampInt(x, lo, hi) {
    if (!Number.isFinite(x)) return lo;
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  function updateSplitUICounts() {
    if (!splitCounts) return;
    const n = datasetNumeric?.X?.length || parsed?.rows?.length || 0;
    if (!n) { splitCounts.textContent = ""; return; }
    const trainPct = clampInt(parseInt(trainSplitNum?.value || "70", 10), 50, 90);
    const valPct = clampInt(parseInt(valSplitNum?.value || "20", 10), 5, 40);
    const testPct = Math.max(0, 100 - trainPct - valPct);
    const nTrain = Math.max(1, Math.floor((n * trainPct) / 100));
    const nVal = Math.max(1, Math.floor((n * valPct) / 100));
    const nTest = Math.max(1, n - nTrain - nVal);
    splitCounts.textContent = `Rows: train ${nTrain.toLocaleString()} · val ${nVal.toLocaleString()} · test ${nTest.toLocaleString()}`;
  }

  // -----------------------------
  // Feature importance (post-train)
  // -----------------------------
  const FI_INT32_MAX = 2147483647;

  function _fiClampI32(x) {
    // Clamp to int32 range. (JS bitwise ops already clamp, but we avoid UB on large floats.)
    if (x > 2147483647) return 2147483647;
    if (x < -2147483648) return -2147483648;
    return x | 0;
  }

  function _fiXorshift32(seed) {
    let x = (seed | 0) || 123456789;
    return () => {
      x ^= x << 13;
      x ^= x >>> 17;
      x ^= x << 5;
      return x >>> 0;
    };
  }

  function _fiShuffledIndices(n, seed) {
    const rng = _fiXorshift32(seed);
    const idx = new Uint32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = rng() % (i + 1);
      const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
    return idx;
  }

  function _fiSplitIdx(idx, fracTrain = 0.7, fracVal = 0.2) {
    const n = idx.length;
    let nTrain = Math.floor(n * fracTrain);
    let nVal = Math.floor(n * fracVal);
    if (nTrain < 1) nTrain = 1;
    if (nVal < 1) nVal = 1;
    if (nTrain + nVal >= n) nVal = Math.max(1, n - nTrain - 1);
    const nTest = Math.max(1, n - nTrain - nVal);
    const train = idx.slice(0, nTrain);
    const val = idx.slice(nTrain, nTrain + nVal);
    const test = idx.slice(nTrain + nVal, nTrain + nVal + nTest);
    return { train, val, test };
  }

  function _fiYield() {
    // Yield to the UI thread so long computations don't freeze the page.
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  function _fiSetVisible(visible) {
    if (!featImpBox) return;
    featImpBox.style.display = visible ? "" : "none";
  }

  function _fiClear(note = "Train a model to see feature importance.") {
    // Also cancels any in-flight permutation importance computation.
    _featImpNonce += 1;
    if (featImpTable) featImpTable.innerHTML = "";
    if (featImpNote) featImpNote.textContent = note;
    _fiSetVisible(false);
    _featImpLast = null;
  }

  function _fiClassFromDelta(delta) {
    if (!Number.isFinite(delta)) return { cls: "fiZero", dot: "neu", label: "—" };
    if (delta > 0) return { cls: "fiPos", dot: "pos", label: "useful" };
    if (delta < 0) return { cls: "fiNeg", dot: "neg", label: "harm/noise" };
    return { cls: "fiZero", dot: "neu", label: "neutral" };
  }

  function _fiFormatSigned(x, digits = 6) {
    if (!Number.isFinite(x)) return "—";
    const s = x >= 0 ? "+" : "";
    return s + x.toFixed(digits);
  }

  function escapeHtml(str) {
    // Minimal HTML escaping for safe table rendering.
    const s = String(str ?? "");
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // --------------------------------------
  // Heuristic search: history table helpers
  // --------------------------------------

  function _isClassTask(task) {
    return (task === "binary_classification" || task === "multiclass_classification" || task === "multilabel_classification");
  }

  function _fmtFloat(x, digits = 6) {
    if (!Number.isFinite(x)) return "—";
    return Number(x).toFixed(digits);
  }

  function _fmtPct(x, digits = 2) {
    if (!Number.isFinite(x)) return "—";
    return (Number(x) * 100).toFixed(digits) + "%";
  }

  function _lrScheduleSummary(sched) {
    if (!sched || typeof sched !== "object") return "none";
    const mode = String(sched.mode || "none");
    if (mode === "plateau") {
      const n = Number.isFinite(sched.patience) ? (sched.patience | 0) : "?";
      const pct = Number.isFinite(sched.dropPct) ? (sched.dropPct | 0) : "?";
      const minLR = Number.isFinite(sched.minLR) ? Number(sched.minLR) : 0;
      return `plateau(n=${n},-${pct}%,min=${minLR})`;
    }
    if (mode === "piecewise") {
      const segs = Array.isArray(sched.segments) ? sched.segments.length : 0;
      return `piecewise(segs=${segs})`;
    }
    return mode || "none";
  }

  function _bestSearchIndex() {
    let bestIdx = -1;
    let best = Infinity;
    for (let i = 0; i < (searchHistory?.length || 0); i++) {
      const e = searchHistory[i];
      if (!e || e.status !== "done") continue;
      const v = e?.meta?.bestValMetric;
      if (!Number.isFinite(v)) continue;
      if (v < best) { best = v; bestIdx = i; }
    }
    return bestIdx;
  }

  function renderSearchTable() {
    if (!searchTable) return;
    const task = selectedTask;
    const isClass = _isClassTask(task);
    const isMulti = (task === "multiclass_classification" || task === "multilabel_classification");

    // Compact, responsive layout (no horizontal scrolling).
    const cols = ["Round", "Status", "Metrics", "Params", "Run"];
    const head = `<thead><tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>`;

    const nAll = (searchHistory?.length || 0);
    if (!nAll) {
      searchTable.innerHTML = head + `<tbody><tr><td colspan="${cols.length}" class="fiSmall">No search runs yet. Enable heuristic search and click Train.</td></tr></tbody>`;
      if (searchPager) searchPager.style.display = "none";
      return;
    }

    // Pagination (fixed page size).
    const totalPages = Math.max(1, Math.ceil(nAll / SEARCH_PAGE_SIZE));
    if (!Number.isFinite(searchPage) || searchPage < 1) searchPage = 1;
    if (searchPage > totalPages) searchPage = totalPages;

    const start = (searchPage - 1) * SEARCH_PAGE_SIZE;
    const end = Math.min(nAll, start + SEARCH_PAGE_SIZE);

    if (searchPager) searchPager.style.display = "";
    if (searchPrevBtn) searchPrevBtn.disabled = (searchPage <= 1);
    if (searchNextBtn) searchNextBtn.disabled = (searchPage >= totalPages);
    if (searchPageInfo) {
      const a = (nAll ? (start + 1) : 0);
      const b = end;
      searchPageInfo.textContent = `Page ${searchPage} / ${totalPages} · ${a}-${b} / ${nAll}`;
    }

    const bestIdx = _bestSearchIndex();

    const slice = searchHistory.slice(start, end);

    const bodyRows = slice.map((e, j) => {
      const i = start + j;
      const isBest = (i === bestIdx);
      const trCls = (e.status === "running") ? "searchRunning" : (e.status === "error") ? "searchError" : isBest ? "searchBest" : "";

      const val = e?.meta?.bestValMetric;
      const tr = e?.meta?.bestTrainMetric;
      const te = e?.meta?.bestTestMetric;
      const valAcc = e?.meta?.bestValAcc;
      const teAcc = e?.meta?.bestTestAcc;
      const bestIter = e?.meta?.bestIter;
      const usedTrees = e?.meta?.usedTrees;
      const totalTrees = e?.meta?.totalTrees;
      const scaleQ = e?.meta?.scaleQ;
      const k = e?.meta?.nClasses;

      const p = e?.params || {};

      const splitTxt = (Number.isFinite(p.splitTrain) && Number.isFinite(p.splitVal))
        ? `${Math.round(p.splitTrain * 100)}/${Math.round(p.splitVal * 100)}/${Math.max(0, 100 - Math.round(p.splitTrain * 100) - Math.round(p.splitVal * 100))}`
        : "—";

      const status = String(e.status || "—");
      const bestBadge = (isBest && e.status === "done") ? ` <span class="fiPos">best</span>` : "";
      const statusHtml = `<div class="stStack"><div class="stLine"><span class="stV">${escapeHtml(status)}</span>${bestBadge}</div>${e?.error ? `<div class="stMuted">${escapeHtml(String(e.error)).slice(0, 140)}</div>` : ""}</div>`;

      const valLabel = isClass ? "Val LogLoss" : "Val MSE";
      const trLabel = isClass ? "Train LogLoss" : "Train MSE";
      const teLabel = isClass ? "Test LogLoss" : "Test MSE";

      const metricsBits = [
        `<div class="stLine"><span class="stK">${escapeHtml(valLabel)}</span> <span class="stV">${escapeHtml(_fmtFloat(val, 6))}</span></div>`,
        `<div class="stLine"><span class="stK">${escapeHtml(teLabel)}</span> <span class="stV">${escapeHtml(_fmtFloat(te, 6))}</span></div>`,
      ];
      if (Number.isFinite(tr)) {
        metricsBits.push(`<div class="stLine"><span class="stK">${escapeHtml(trLabel)}</span> <span class="stV">${escapeHtml(_fmtFloat(tr, 6))}</span></div>`);
      }
      if (isClass) {
        metricsBits.push(`<div class="stLine"><span class="stK">Val Acc</span> <span class="stV">${escapeHtml(_fmtPct(valAcc, 2))}</span></div>`);
        metricsBits.push(`<div class="stLine"><span class="stK">Test Acc</span> <span class="stV">${escapeHtml(_fmtPct(teAcc, 2))}</span></div>`);
      }
      metricsBits.push(`<div class="stMuted">Best iter: ${escapeHtml(Number.isFinite(bestIter) ? String(bestIter) : "—")}</div>`);
      const metricsHtml = `<div class="stStack">${metricsBits.join("")}</div>`;

      const treesTxt = Number.isFinite(p.trees)
        ? String(p.trees)
        : (Number.isFinite(e?.meta?.maxTrees) ? String(e.meta.maxTrees) : "—");

      const paramsBits = [];
      if (isMulti) {
        const perTxt = (task === "multiclass_classification") ? "trees/class" : "trees/label";
        paramsBits.push(`<div class="stLine"><span class="stK">${escapeHtml(perTxt)}</span> <span class="stV">${escapeHtml(treesTxt)}</span></div>`);
        paramsBits.push(`<div class="stLine"><span class="stK">Total trees</span> <span class="stV">${escapeHtml(Number.isFinite(totalTrees) ? String(totalTrees) : "—")}</span></div>`);
      } else {
        paramsBits.push(`<div class="stLine"><span class="stK">Trees</span> <span class="stV">${escapeHtml(treesTxt)}</span></div>`);
      }
      paramsBits.push(`<div class="stLine"><span class="stK">Depth</span> <span class="stV">${escapeHtml(Number.isFinite(p.depth) ? String(p.depth) : "—")}</span></div>`);
      paramsBits.push(`<div class="stLine"><span class="stK">LR</span> <span class="stV">${escapeHtml(Number.isFinite(p.lr) ? String(p.lr) : "—")}</span></div>`);
      paramsBits.push(`<div class="stLine"><span class="stK">Min leaf</span> <span class="stV">${escapeHtml(Number.isFinite(p.minLeaf) ? String(p.minLeaf) : "—")}</span></div>`);
      paramsBits.push(`<div class="stLine"><span class="stK">Bins</span> <span class="stV">${escapeHtml(Number.isFinite(p.bins) ? String(p.bins) : (Number.isFinite(e?.meta?.bins) ? String(e.meta.bins) : "—"))}</span></div>`);
      paramsBits.push(`<div class="stLine"><span class="stK">Binning</span> <span class="stV">${escapeHtml(String(p.binning || e?.meta?.binning || "linear"))}</span></div>`);
      paramsBits.push(`<div class="stMuted">LR schedule: ${escapeHtml(_lrScheduleSummary(p.lrSchedule))}</div>`);
      const paramsHtml = `<div class="stStack">${paramsBits.join("")}</div>`;

      const runBits = [
        `<div class="stLine"><span class="stK">Split</span> <span class="stV">${escapeHtml(splitTxt)}</span></div>`,
        `<div class="stLine"><span class="stK">Seed</span> <span class="stV">${escapeHtml(Number.isFinite(p.seed) ? String(p.seed) : "—")}</span></div>`,
        `<div class="stLine"><span class="stK">Early stop</span> <span class="stV">${escapeHtml(p.earlyStop ? "on" : "off")}</span></div>`,
        `<div class="stLine"><span class="stK">Patience</span> <span class="stV">${escapeHtml(Number.isFinite(p.patience) ? String(p.patience) : "—")}</span></div>`,
        `<div class="stLine"><span class="stK">ScaleQ</span> <span class="stV">${escapeHtml(Number.isFinite(scaleQ) ? String(scaleQ) : "—")}</span></div>`,
        `<div class="stLine"><span class="stK">K</span> <span class="stV">${escapeHtml(Number.isFinite(p.nClasses) ? String(p.nClasses) : (Number.isFinite(k) ? String(k) : "—"))}</span></div>`,
        `<div class="stLine"><span class="stK">Used</span> <span class="stV">${escapeHtml(Number.isFinite(usedTrees) ? String(usedTrees) : "—")}</span></div>`,
      ];
      const runHtml = `<div class="stStack">${runBits.join("")}</div>`;

      const statusTitle = e?.error ? ` title="${escapeHtml(String(e.error)).slice(0, 300)}"` : "";

      return (
        `<tr class="${trCls}">`
        + `<td data-label="Round">${escapeHtml(String(e.round ?? (i + 1)))}</td>`
        + `<td data-label="Status"${statusTitle}>${statusHtml}</td>`
        + `<td data-label="Metrics">${metricsHtml}</td>`
        + `<td data-label="Params">${paramsHtml}</td>`
        + `<td data-label="Run">${runHtml}</td>`
        + `</tr>`
      );
    }).join("");

    searchTable.innerHTML = head + `<tbody>${bodyRows}</tbody>`;
  }

  function _fiComputeSplitCounts(model) {
    if (!model || !model.dv) return { counts: new Uint32Array(0), total: 0 };
    const nFeat = model.nFeatures | 0;
    const counts = new Uint32Array(Math.max(0, nFeat));
    const { dv, nTrees, internal, perTree, treesOff } = model;
    let total = 0;
    for (let t = 0; t < (nTrees | 0); t++) {
      const base = treesOff + t * perTree;
      for (let i = 0; i < internal; i++) {
        const nodeOff = base + i * 8;
        const thr = dv.getInt32(nodeOff + 2, true);
        if (thr === FI_INT32_MAX) continue; // forced node (no real split)
        const f = dv.getUint16(nodeOff, true);
        if (f < counts.length) {
          counts[f] += 1;
          total += 1;
        }
      }
    }
    return { counts, total };
  }

  function _fiQuantizeEvalFeaturesFloat(Xrows, nFeatures, scaleQ) {
    // Pack/quantize row-major features into Int32Array [r*nFeatures+f].
    const out = new Int32Array(Xrows.length * nFeatures);
    for (let r = 0; r < Xrows.length; r++) {
      const row = Xrows[r];
      const base = r * nFeatures;
      for (let f = 0; f < nFeatures; f++) {
        const x = Number(row[f]);
        const q = Math.round(x * scaleQ);
        out[base + f] = _fiClampI32(q);
      }
    }
    return out;
  }

  function _fiPredictV1_Q_fromFeatQ(model, featQRow) {
    // Scalar model prediction in Q-units, but expects the feature vector already quantized.
    const { dv, nTrees, depth, baseQ, treesOff, internal, perTree } = model;
    let acc = Number(baseQ || 0);
    for (let t = 0; t < (nTrees | 0); t++) {
      const base = treesOff + t * perTree;
      let idx = 0;
      for (let level = 0; level < (depth | 0); level++) {
        const nodeOff = base + idx * 8;
        const f = dv.getUint16(nodeOff, true);
        const thr = dv.getInt32(nodeOff + 2, true);
        const xq = featQRow[f];
        idx = (xq > thr) ? (2 * idx + 2) : (2 * idx + 1);
      }
      const leafIndex = idx - internal;
      const leafOff = base + internal * 8 + leafIndex * 4;
      acc += dv.getInt32(leafOff, true);
    }
    return acc;
  }

  function _fiPredictV2_logitsQ_fromFeatQ(model, featQRow, outLogitsQ) {
    // Vector-output model prediction in Q-units (logits), expects pre-quantized features.
    // IMPORTANT: On-chain inference uses int256 accumulators; avoid int32 wrapping in the UI.
    const { dv, nClasses, treesPerClass, depth, treesOff, internal, perTree, baseLogitsQ } = model;
    for (let k = 0; k < (nClasses | 0); k++) {
      let acc = Number(baseLogitsQ?.[k] ?? 0);
      const classTreeBase = treesOff + (k * (treesPerClass | 0)) * perTree;
      for (let t = 0; t < (treesPerClass | 0); t++) {
        const base = classTreeBase + t * perTree;
        let idx = 0;
        for (let level = 0; level < (depth | 0); level++) {
          const nodeOff = base + idx * 8;
          const f = dv.getUint16(nodeOff, true);
          const thr = dv.getInt32(nodeOff + 2, true);
          const xq = featQRow[f];
          idx = (xq > thr) ? (2 * idx + 2) : (2 * idx + 1);
        }
        const leafIndex = idx - internal;
        const leafOff = base + internal * 8 + leafIndex * 4;
        acc += dv.getInt32(leafOff, true);
      }
      outLogitsQ[k] = acc;
    }
  }

  function _fiEvalMetricsOnFeatQ({ model, task, featQMat, yEval, yFlatEval, nRowsEval }) {
    const scaleQ = Number(model.scaleQ || 1);

    // Classification
    const isBinary = (task === "binary_classification");
    const isMulti = (task === "multiclass_classification");
    const isMultiLabel = (task === "multilabel_classification");

    if (isBinary) {
      let loss = 0;
      let correct = 0;
      const EPS = 1e-12;
      for (let r = 0; r < nRowsEval; r++) {
        const row = featQMat.subarray(r * model.nFeatures, (r + 1) * model.nFeatures);
        const logit = _fiPredictV1_Q_fromFeatQ(model, row) / scaleQ;
        let p = sigmoid(logit);
        if (p < EPS) p = EPS;
        else if (p > 1 - EPS) p = 1 - EPS;
        const y = (yEval[r] >= 0.5) ? 1 : 0;
        loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
        const pred = (p >= 0.5) ? 1 : 0;
        if (pred === y) correct += 1;
      }
      return { loss: loss / Math.max(1, nRowsEval), acc: correct / Math.max(1, nRowsEval) };
    }

    if (isMulti) {
      const K = model.nClasses | 0;
      const logitsQ = new Float64Array(K);
      let loss = 0;
      let correct = 0;
      const EPS = 1e-12;
      for (let r = 0; r < nRowsEval; r++) {
        const row = featQMat.subarray(r * model.nFeatures, (r + 1) * model.nFeatures);
        _fiPredictV2_logitsQ_fromFeatQ(model, row, logitsQ);

        // softmax(logits)
        let maxZ = -Infinity;
        for (let k = 0; k < K; k++) {
          const z = logitsQ[k] / scaleQ;
          if (z > maxZ) maxZ = z;
        }
        let sum = 0;
        let bestK = 0;
        let bestZ = logitsQ[0] / scaleQ;
        for (let k = 0; k < K; k++) {
          const z = logitsQ[k] / scaleQ;
          const e = Math.exp(z - maxZ);
          sum += e;
          if (z > bestZ) { bestZ = z; bestK = k; }
        }

        const y = yEval[r] | 0;
        const zy = logitsQ[y] / scaleQ;
        let py = Math.exp(zy - maxZ) / (sum || 1);
        if (py < EPS) py = EPS;
        else if (py > 1 - EPS) py = 1 - EPS;
        loss += -Math.log(py);
        if (bestK === y) correct += 1;
      }
      return { loss: loss / Math.max(1, nRowsEval), acc: correct / Math.max(1, nRowsEval) };
    }

    if (isMultiLabel) {
      const K = model.nClasses | 0;
      const logitsQ = new Float64Array(K);
      const EPS = 1e-12;
      let loss = 0;
      let correct = 0;
      const denom = Math.max(1, nRowsEval) * Math.max(1, K);

      for (let r = 0; r < nRowsEval; r++) {
        const row = featQMat.subarray(r * model.nFeatures, (r + 1) * model.nFeatures);
        _fiPredictV2_logitsQ_fromFeatQ(model, row, logitsQ);
        const yBase = r * K;
        for (let k = 0; k < K; k++) {
          const logit = logitsQ[k] / scaleQ;
          let p = sigmoid(logit);
          if (p < EPS) p = EPS;
          else if (p > 1 - EPS) p = 1 - EPS;
          const y = (yFlatEval[yBase + k] >= 0.5) ? 1 : 0;
          loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
          const pred = (p >= 0.5) ? 1 : 0;
          if (pred === y) correct += 1;
        }
      }

      return { loss: loss / denom, acc: correct / denom };
    }

    // Regression
    let mseSum = 0;
    for (let r = 0; r < nRowsEval; r++) {
      const row = featQMat.subarray(r * model.nFeatures, (r + 1) * model.nFeatures);
      const pred = _fiPredictV1_Q_fromFeatQ(model, row) / scaleQ;
      const diff = (Number(yEval[r]) - pred);
      mseSum += diff * diff;
    }
    return { loss: mseSum / Math.max(1, nRowsEval), acc: NaN };
  }

  function _fiRenderTable({ task, featureNames, rows, baseline, note }) {
    if (!featImpTable) return;

    const isClass = (task === "binary_classification" || task === "multiclass_classification" || task === "multilabel_classification");

    const cols = [
      "Feature",
      "Split uses",
      "Split %",
      isClass ? "Perm ΔLogLoss" : "Perm ΔMSE",
      ...(isClass ? ["Perm ΔAcc"] : []),
      "Tag"
    ];

    const head = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;

    const bodyRows = rows.map((r) => {
      const deltaLoss = r.permDeltaLoss;
      const tag = _fiClassFromDelta(deltaLoss);
      const splitPct = (Number.isFinite(r.splitPct) ? `${(r.splitPct * 100).toFixed(1)}%` : "—");

      const deltaLossTxt = (Number.isFinite(deltaLoss)
        ? _fiFormatSigned(deltaLoss, 6)
        : "—");

      const deltaAccTxt = (isClass && Number.isFinite(r.permDeltaAcc))
        ? _fiFormatSigned(r.permDeltaAcc * 100, 3) + "%"
        : (isClass ? "—" : "");

      return (
        `<tr>`
        + `<td class="fiName">${escapeHtml(String(r.name || ""))}</td>`
        + `<td>${Number.isFinite(r.splitCount) ? String(r.splitCount) : "—"}</td>`
        + `<td>${splitPct}</td>`
        + `<td class="${tag.cls}">${escapeHtml(deltaLossTxt)}</td>`
        + (isClass ? `<td class="${Number.isFinite(r.permDeltaAcc) ? (r.permDeltaAcc >= 0 ? "fiPos" : "fiNeg") : "fiZero"}">${escapeHtml(deltaAccTxt)}</td>` : "")
        + `<td class="fiSmall"><span class="fiTag"><span class="fiDot ${tag.dot}"></span>${tag.label}</span></td>`
        + `</tr>`
      );
    }).join("");

    featImpTable.innerHTML = head + `<tbody>${bodyRows}</tbody>`;
    if (featImpNote) {
      const baseBits = [];
      if (baseline && Number.isFinite(baseline.loss)) {
        baseBits.push(isClass ? `Baseline test LogLoss: ${baseline.loss.toFixed(6)}` : `Baseline test MSE: ${baseline.loss.toFixed(6)}`);
      }
      if (isClass && baseline && Number.isFinite(baseline.acc)) {
        baseBits.push(`Baseline test Acc: ${(baseline.acc * 100).toFixed(2)}%`);
      }
      const extra = baseBits.length ? ` · ${baseBits.join(" · ")}` : "";
      featImpNote.textContent = (note || "Permutation importance is computed by shuffling one feature at a time on the test split.") + extra;
    }
    _fiSetVisible(true);
  }

  async function _fiComputeAndRender() {
    const nonce = ++_featImpNonce;
    _featImpLast = null;

    // If we don't have a trained model, keep the box hidden.
    try { renderPreviewBestModelBlocks(); } catch {}

    if (!trained?.decoded || !datasetNumeric?.featureNames?.length) {
      _fiClear("Train a model to see feature importance.");
      return;
    }

    if (!featImpBox || !featImpTable || !featImpNote) {
      // UI elements not present; nothing to do.
      return;
    }

    const model = trained.decoded;
    const task = (trained?.meta?.task || lastTrainInfo?.task || selectedTask);
    const featureNames = Array.isArray(datasetNumeric.featureNames) ? datasetNumeric.featureNames : [];

    // Show the box immediately so the user sees progress.
    _fiSetVisible(true);
    featImpNote.textContent = "Computing feature importance…";
    featImpTable.innerHTML = "";

    // Split usage counts are fast and always available.
    const { counts: splitCounts, total: splitTotal } = _fiComputeSplitCounts(model);
    const rows = featureNames.map((name, i) => ({
      idx: i,
      name,
      splitCount: splitCounts[i] || 0,
      splitPct: (splitTotal > 0 ? (splitCounts[i] || 0) / splitTotal : 0),
      permDeltaLoss: NaN,
      permDeltaAcc: NaN,
    }));

    // Render immediately (perm columns will show "—" until computed).
    const rowsSorted0 = rows.slice().sort((a, b) => (b.splitCount - a.splitCount));
    _fiRenderTable({ task, featureNames, rows: rowsSorted0, baseline: null, note: "Split-use counts come from the trained trees (forced/no-split nodes are excluded)." });

    // --- Permutation importance (test split) ---
    // We try to compute permutation importance for *all* selected features, but adaptively
    // shrink the evaluation sample so it stays responsive.
    await _fiYield();
    if (nonce !== _featImpNonce) return;

    const nAllRows = datasetNumeric.X.length;
    const nFeat = featureNames.length;
    const seed = Number.isFinite(lastTrainInfo?.seed) ? (lastTrainInfo.seed | 0) : (parseInt(seedNum?.value || "42", 10) | 0);
    const splitTrain = (typeof lastTrainInfo?.splitTrain === "number") ? lastTrainInfo.splitTrain : (parseInt(trainSplitNum?.value || "70", 10) / 100);
    const splitVal = (typeof lastTrainInfo?.splitVal === "number") ? lastTrainInfo.splitVal : (parseInt(valSplitNum?.value || "20", 10) / 100);

    // Recreate the same deterministic split as the worker.
    let testIdx = null;
    try {
      const idx = _fiShuffledIndices(nAllRows, seed);
      const sp = _fiSplitIdx(idx, splitTrain, splitVal);
      testIdx = sp.test;
    } catch {
      // Fallback: use all rows
      testIdx = null;
    }

    // Build eval index list.
    let evalIdx = testIdx ? Array.from(testIdx) : Array.from({ length: nAllRows }, (_, i) => i);

    // Adaptive evaluation size.
    const depth = model.depth | 0;
    const totalTrees = model.nTrees | 0;
    const budget = 12_000_000; // rough node-evaluation budget to keep UI responsive
    const perRowCost = Math.max(1, totalTrees * Math.max(1, depth));
    let maxRows = Math.min(1024, evalIdx.length);
    let evalRows = Math.max(16, Math.min(maxRows, Math.floor(budget / (Math.max(1, nFeat) * perRowCost))));
    if (evalRows > maxRows) evalRows = maxRows;
    if (evalRows < maxRows) {
      // evalIdx is already randomly ordered via shuffle; taking the prefix is fine.
      evalIdx = evalIdx.slice(0, evalRows);
    }
    evalRows = evalIdx.length;

    // Pack feature rows and labels.
    const Xrows = new Array(evalRows);
    for (let i = 0; i < evalRows; i++) Xrows[i] = datasetNumeric.X[evalIdx[i]];

    const featQMat = _fiQuantizeEvalFeaturesFloat(Xrows, nFeat, model.scaleQ);

    // Labels
    const yEval = new Float32Array(evalRows);
    let yFlatEval = null;
    if (task === "multilabel_classification") {
      const K = model.nClasses | 0;
      yFlatEval = new Float32Array(evalRows * K);
      const src = datasetNumeric.yFlat instanceof Float32Array ? datasetNumeric.yFlat : null;
      if (!src) {
        // Should not happen, but keep safe.
        for (let r = 0; r < evalRows; r++) yEval[r] = 0;
      } else {
        for (let r = 0; r < evalRows; r++) {
          const rr = evalIdx[r];
          const srcBase = rr * K;
          const dstBase = r * K;
          for (let k = 0; k < K; k++) yFlatEval[dstBase + k] = src[srcBase + k];
        }
      }
    } else {
      for (let r = 0; r < evalRows; r++) yEval[r] = datasetNumeric.y[evalIdx[r]];
    }

    // Baseline metrics
    featImpNote.textContent = `Computing permutation importance on ${evalRows.toLocaleString()} test rows… (baseline)`;
    await _fiYield();
    if (nonce !== _featImpNonce) return;
    const baseline = _fiEvalMetricsOnFeatQ({ model, task, featQMat, yEval, yFlatEval, nRowsEval: evalRows });

    // Per-feature permutation
    const rngBase = _fiXorshift32((seed ^ 0x9e3779b9) | 0);
    const colOrig = new Int32Array(evalRows);
    const colPerm = new Int32Array(evalRows);

    for (let f = 0; f < nFeat; f++) {
      if (nonce !== _featImpNonce) return;

      // Extract column
      for (let r = 0; r < evalRows; r++) {
        const v = featQMat[r * nFeat + f];
        colOrig[r] = v;
        colPerm[r] = v;
      }

      // Shuffle (Fisher-Yates)
      for (let i = evalRows - 1; i > 0; i--) {
        const j = (rngBase() + (f * 1013)) % (i + 1);
        const tmp = colPerm[i]; colPerm[i] = colPerm[j]; colPerm[j] = tmp;
      }

      // Apply permuted column
      for (let r = 0; r < evalRows; r++) featQMat[r * nFeat + f] = colPerm[r];

      const met = _fiEvalMetricsOnFeatQ({ model, task, featQMat, yEval, yFlatEval, nRowsEval: evalRows });

      // Restore original
      for (let r = 0; r < evalRows; r++) featQMat[r * nFeat + f] = colOrig[r];

      const deltaLoss = (Number.isFinite(met.loss) && Number.isFinite(baseline.loss)) ? (met.loss - baseline.loss) : NaN;
      const deltaAcc = (Number.isFinite(met.acc) && Number.isFinite(baseline.acc)) ? (baseline.acc - met.acc) : NaN;

      rows[f].permDeltaLoss = deltaLoss;
      rows[f].permDeltaAcc = deltaAcc;

      if (f % 8 === 0 || f === nFeat - 1) {
        featImpNote.textContent = `Computing permutation importance: ${Math.min(f + 1, nFeat)}/${nFeat} features…`;
        await _fiYield();
      }
    }

    if (nonce !== _featImpNonce) return;

    // Final render: sort by permutation importance (largest positive Δloss first).
    const rowsSorted = rows.slice().sort((a, b) => {
      const da = Number.isFinite(a.permDeltaLoss) ? a.permDeltaLoss : -Infinity;
      const db = Number.isFinite(b.permDeltaLoss) ? b.permDeltaLoss : -Infinity;
      if (db !== da) return db - da;
      return (b.splitCount - a.splitCount);
    });

    const isClass = (task === "binary_classification" || task === "multiclass_classification" || task === "multilabel_classification");
    const note = `Permutation importance: Δ${isClass ? "LogLoss" : "MSE"} when shuffling one feature at a time on the test split (sampled). Negative values can indicate a harmful/noisy feature.`;
    _fiRenderTable({ task, featureNames, rows: rowsSorted, baseline, note });
    _featImpLast = { task, rows: rowsSorted, baseline };
  }

  function syncSplit(source) {
    if (!trainSplitRange || !trainSplitNum || !valSplitRange || !valSplitNum || !testSplitPill) return;

    let trainPct = clampInt(parseInt(trainSplitRange.value, 10), 50, 90);
    let valPct = clampInt(parseInt(valSplitRange.value, 10), 5, 40);

    if (source === "trainNum") trainPct = clampInt(parseInt(trainSplitNum.value, 10), 50, 90);
    if (source === "trainRange") trainPct = clampInt(parseInt(trainSplitRange.value, 10), 50, 90);
    if (source === "valNum") valPct = clampInt(parseInt(valSplitNum.value, 10), 5, 40);
    if (source === "valRange") valPct = clampInt(parseInt(valSplitRange.value, 10), 5, 40);

    // Enforce a non-trivial test set (>=5%)
    if (trainPct + valPct > 95) {
      valPct = Math.max(5, 95 - trainPct);
    }
    const testPct = 100 - trainPct - valPct;

    trainSplitRange.value = String(trainPct);
    trainSplitNum.value = String(trainPct);
    valSplitRange.value = String(valPct);
    valSplitNum.value = String(valPct);
    testSplitPill.textContent = `Test ${testPct}%`;

    updateSplitUICounts();
  }

  // Wire split controls
  if (trainSplitRange && trainSplitNum && valSplitRange && valSplitNum) {
    trainSplitRange.addEventListener("input", () => syncSplit("trainRange"));
    trainSplitNum.addEventListener("input", () => syncSplit("trainNum"));
    valSplitRange.addEventListener("input", () => syncSplit("valRange"));
    valSplitNum.addEventListener("input", () => syncSplit("valNum"));
    // Initial
    syncSplit("init");
  }

  function updateSize() {
    const trees0 = parseInt(treesNum.value, 10);
    const depth0 = parseInt(depthNum.value, 10);
    const nClasses = (selectedTask === "multiclass_classification")
      ? Math.max(2, (selectedMultiLabels?.length || labelValuesInfo?.values?.length || 2))
      : (selectedTask === "multilabel_classification")
        ? Math.max(2, (selectedLabelCols?.length || 2))
        : 2;
    const cl = clampForSize(trees0, depth0, selectedTask, nClasses);
    if (cl.trees !== trees0) { treesNum.value = String(cl.trees); treesRange.value = String(cl.trees); }
    if (cl.depth !== depth0) { depthNum.value = String(cl.depth); depthRange.value = String(cl.depth); }

    const est = cl.estBytes;
    const treeKey = (selectedTask === "multiclass_classification")
      ? "Trees/class"
      : (selectedTask === "multilabel_classification")
        ? "Trees/label"
        : "Trees";
    const entries = [
      ["Features", String(selectedFeatures.length || 0)],
      [treeKey, String(cl.trees)],
      ["Depth", String(cl.depth)],
      ["Est. bytes", `${est.toLocaleString()}`],
      ["Limit", `${SIZE_LIMIT.toLocaleString()}`]
    ];
    if (selectedTask === "multiclass_classification") {
      entries.splice(2, 0, ["Classes", String(nClasses)], ["Total trees", String(cl.trees * nClasses)]);
    } else if (selectedTask === "multilabel_classification") {
      entries.splice(2, 0, ["Labels", String(nClasses)], ["Total trees", String(cl.trees * nClasses)]);
    }
    setKV(sizeKV, entries);
    sizeNote.textContent = est > SIZE_LIMIT ? "Too large. Reduce trees/depth." : "OK for on-chain.";
    updateDeployState();
  }

  function renderPreviewInputs() {
    previewFeatGrid.innerHTML = "";
    previewBtn.disabled = true;
    loadRowBtn.disabled = true;
    compareRowBtn.disabled = true;
    if (selectedTask === "binary_classification") {
      setKV(previewKV, [["Logit (Q)", "—"], ["Probability", "—"], ["Predicted", "—"]]);
    } else if (selectedTask === "multiclass_classification") {
      setKV(previewKV, [["Best class", "—"], ["Best logit (Q)", "—"], ["Top probabilities", "—"]]);
    } else if (selectedTask === "multilabel_classification") {
      setKV(previewKV, [["Predicted positives", "—"], ["Top probabilities", "—"], ["Micro accuracy vs GT", "—"]]);
    } else {
      setKV(previewKV, [["Local score (Q)", "—"], ["Value", "—"]]);
    }

    try { renderPreviewBestModelBlocks(); } catch {}

    if (!trained?.decoded || !datasetNumeric?.featureNames?.length) {
      previewFeatHint.textContent = "Train a model to enable.";
      return;
    }

    previewFeatHint.textContent = "Enter feature values (or load a dataset row) and run local prediction.";
    datasetNumeric.featureNames.forEach((nm, i) => {
      const name = (nm && String(nm).trim()) ? String(nm).trim() : `f${i}`;
      const cell = document.createElement("div");
      cell.className = "featCell";

      const title = document.createElement("div");
      title.className = "featName";
      title.textContent = name;

      const inp = document.createElement("input");
      inp.type = "number";
      inp.step = "any";
      inp.value = "";
      inp.placeholder = name;
      inp.title = name;
      inp.dataset.idx = String(i);

      cell.appendChild(title);
      cell.appendChild(inp);
      previewFeatGrid.appendChild(cell);
    });

    // enable controls
    previewBtn.disabled = false;
    if (datasetNumeric?.X?.length) {
      rowIndex.max = String(Math.max(0, datasetNumeric.X.length - 1));
      loadRowBtn.disabled = false;
      compareRowBtn.disabled = false;
    }
  }

  function clampRowIndex(idx) {
    if (!datasetNumeric?.X?.length) return 0;
    const max = datasetNumeric.X.length - 1;
    const n = Number(idx);
    if (!Number.isFinite(n)) return 0;
    return Math.max(0, Math.min(max, Math.floor(n)));
  }

  function loadRowIntoInputs(idxRaw) {
    if (!datasetNumeric?.X?.length) throw new Error("No dataset loaded");
    const idx = clampRowIndex(idxRaw);
    const row = datasetNumeric.X[idx];
    const inputs = Array.from(previewFeatGrid.querySelectorAll("input"));
    if (inputs.length !== row.length) {
      throw new Error(`Row has ${row.length} features but UI expects ${inputs.length}. Re-train / re-build features.`);
    }
    inputs.forEach((inp, i) => { inp.value = String(row[i]); });
    if (rowIndex) rowIndex.value = String(idx);
    log(`[${nowTs()}] Loaded dataset row ${idx}`);
    return idx;
  }

  loadRowBtn.addEventListener("click", () => {
    try {
      if (!trained?.decoded) throw new Error("Train a model first");
      const idx = loadRowIntoInputs(rowIndex?.value ?? 0);
      const y = datasetNumeric?.y?.[idx];
      if (Number.isFinite(y)) {
        if (selectedTask === "binary_classification" && datasetNumeric?.classes) {
          const cls = y >= 0.5 ? 1 : 0;
          log(`[${nowTs()}] Row ${idx} label=${y} (${datasetNumeric.classes[cls]})`);
        } else if (selectedTask === "multiclass_classification" && Array.isArray(datasetNumeric?.classes)) {
          const cls = (y | 0);
          const name = datasetNumeric.classes[cls] ?? String(cls);
          log(`[${nowTs()}] Row ${idx} label=${y} (${name})`);
        } else {
          log(`[${nowTs()}] Row ${idx} label=${y}`);
        }
      } else if (selectedTask === "multilabel_classification" && Array.isArray(y) && Array.isArray(datasetNumeric?.labelNames)) {
        const names = datasetNumeric.labelNames;
        const pos = [];
        for (let k=0;k<y.length;k++) {
          if (y[k] >= 0.5) pos.push(names[k] ?? `label${k}`);
        }
        const shown = pos.slice(0, 12).join(", ") + (pos.length > 12 ? ", …" : "");
        log(`[${nowTs()}] Row ${idx} positives=${pos.length}${pos.length ? ` [${shown}]` : ""}`);
      }
    } catch (e) {
      log(`[${nowTs()}] [error] ${e.message || e}`);
    }
  });

  compareRowBtn.addEventListener("click", () => {
    try {
      if (!trained?.decoded) throw new Error("No model");
      if (!datasetNumeric?.X?.length || !datasetNumeric?.y?.length) throw new Error("No dataset loaded");
      const idx = loadRowIntoInputs(rowIndex?.value ?? 0);
      const inputs = Array.from(previewFeatGrid.querySelectorAll("input"));
      const vals = inputs.map((x) => Number(x.value));
      if (selectedTask === "multiclass_classification" && Array.isArray(datasetNumeric?.classes)) {
        const res = predictClassQ(trained.decoded, vals);
        const logits = res.logitsQ;
        const scl = Number(trained.decoded.scaleQ || 1);
        // softmax for display
        let maxZ = -Infinity;
        for (let k = 0; k < logits.length; k++) {
          const z = logits[k] / scl;
          if (z > maxZ) maxZ = z;
        }
        const probs = new Array(logits.length);
        let sum = 0;
        for (let k = 0; k < logits.length; k++) {
          const e = Math.exp((logits[k] / scl) - maxZ);
          probs[k] = e;
          sum += e;
        }
        const inv = 1 / (sum || 1);
        for (let k = 0; k < probs.length; k++) probs[k] *= inv;

        const predK = res.classIndex;
        const trueK = Number(datasetNumeric.y[idx]) | 0;
        const ok = predK === trueK;

        // Top-3 pretty string
        const top = probs
          .map((p, k) => ({ k, p }))
          .sort((a, b) => b.p - a.p)
          .slice(0, Math.min(3, probs.length))
          .map(({ k, p }) => `${datasetNumeric.classes[k] ?? k}: ${(p * 100).toFixed(2)}%`)
          .join(" · ");

        setKV(previewKV, [
          ["Best class", `${predK} (${datasetNumeric.classes[predK] ?? predK})`],
          ["Best logit (Q)", String(res.bestLogitQ)],
          ["Label", `${trueK} (${datasetNumeric.classes[trueK] ?? trueK})`],
          ["Correct", { text: ok ? "YES" : "NO", className: ok ? "good" : "bad" }],
          ["Top probabilities", top || "—"],
        ]);
        log(`[${nowTs()}] Compare row ${idx}: pred=${predK} label=${trueK} ok=${ok}`);
        return;
      }

      if (selectedTask === "multilabel_classification" && Array.isArray(datasetNumeric?.labelNames) && Array.isArray(datasetNumeric?.y?.[idx])) {
        const logitsQ = predictMultiQ(trained.decoded, vals);
        const scl = Number(trained.decoded.scaleQ || 1);
        const probs = new Array(logitsQ.length);
        for (let k = 0; k < logitsQ.length; k++) {
          const z = logitsQ[k] / scl;
          probs[k] = 1 / (1 + Math.exp(-z));
        }
        const names = datasetNumeric.labelNames;
        const yRow = datasetNumeric.y[idx];
        const n = Math.min(probs.length, yRow.length);
        let correct = 0;
        const predPos = [];
        const truePos = [];
        for (let k = 0; k < n; k++) {
          const pred01 = probs[k] >= 0.5 ? 1 : 0;
          const y01 = yRow[k] >= 0.5 ? 1 : 0;
          if (pred01 === y01) correct++;
          if (pred01) predPos.push(names[k] ?? `label${k}`);
          if (y01) truePos.push(names[k] ?? `label${k}`);
        }
        const microAcc = correct / (n || 1);
        const predShown = predPos.slice(0, 12).join(", ") + (predPos.length > 12 ? ", …" : "");
        const top = probs
          .map((p, k) => ({ k, p }))
          .sort((a, b) => b.p - a.p)
          .slice(0, Math.min(5, probs.length))
          .map(({ k, p }) => `${names[k] ?? k}: ${(p * 100).toFixed(2)}%`)
          .join(" · ");

        setKV(previewKV, [
          ["Predicted positives", predPos.length ? predShown : "(none)"],
          ["Top probabilities", top || "—"],
          ["Micro accuracy vs GT", `${(microAcc * 100).toFixed(2)}% (gt+${truePos.length})`],
        ]);
        log(`[${nowTs()}] Compare row ${idx}: microAcc=${(microAcc * 100).toFixed(2)}% predPos=${predPos.length} truePos=${truePos.length}`);
        return;
      }

      const scoreQ = predictQ(trained.decoded, vals);
      if (selectedTask === "binary_classification" && datasetNumeric?.classes) {
        const logit = scoreQ / trained.decoded.scaleQ;
        const prob = sigmoid(logit);
        const pred01 = prob >= 0.5 ? 1 : 0;
        const true01 = Number(datasetNumeric.y[idx]) >= 0.5 ? 1 : 0;
        const ok = pred01 === true01;
        setKV(previewKV, [
          ["Logit (Q)", String(scoreQ)],
          ["Probability", prob.toFixed(6)],
          ["Predicted", `${pred01} (${datasetNumeric.classes[pred01]})`],
          ["Label", `${true01} (${datasetNumeric.classes[true01]})`],
          ["Correct", { text: ok ? "YES" : "NO", className: ok ? "good" : "bad" }],
        ]);
        log(`[${nowTs()}] Compare row ${idx}: prob=${prob} pred=${pred01} label=${true01}`);
      } else {
        const pred = scoreQ / trained.decoded.scaleQ;
        const label = Number(datasetNumeric.y[idx]);
        const absErr = Math.abs(pred - label);
        const relErr = (label !== 0) ? (absErr / Math.abs(label) * 100) : NaN;
        let cls = "bad";
        if (!Number.isFinite(relErr)) cls = "warn";
        else if (relErr <= 5) cls = "good";
        else if (relErr <= 15) cls = "warn";
        setKV(previewKV, [
          ["Local score (Q)", String(scoreQ)],
          ["Predicted", String(pred)],
          ["Label", String(label)],
          ["Abs error", { text: String(absErr), className: cls }],
          ["Rel error", { text: Number.isFinite(relErr) ? `${relErr.toFixed(3)}%` : "—", className: cls }],
        ]);
        log(`[${nowTs()}] Compare row ${idx}: pred=${pred} label=${label} absErr=${absErr} relErr=${relErr}`);
      }
    } catch (e) {
      log(`[${nowTs()}] [error] ${e.message || e}`);
    }
  });

  previewBtn.addEventListener("click", () => {
    try {
      if (!trained?.decoded) throw new Error("No model");
      const inputs = Array.from(previewFeatGrid.querySelectorAll("input"));
      const vals = inputs.map((x) => Number(x.value));
      if (selectedTask === "multiclass_classification" && Array.isArray(datasetNumeric?.classes)) {
        const res = predictClassQ(trained.decoded, vals);
        const logits = res.logitsQ;
        const scl = Number(trained.decoded.scaleQ || 1);
        let maxZ = -Infinity;
        for (let k = 0; k < logits.length; k++) {
          const z = logits[k] / scl;
          if (z > maxZ) maxZ = z;
        }
        const probs = new Array(logits.length);
        let sum = 0;
        for (let k = 0; k < logits.length; k++) {
          const e = Math.exp((logits[k] / scl) - maxZ);
          probs[k] = e;
          sum += e;
        }
        const inv = 1 / (sum || 1);
        for (let k = 0; k < probs.length; k++) probs[k] *= inv;

        const predK = res.classIndex;
        const top = probs
          .map((p, k) => ({ k, p }))
          .sort((a, b) => b.p - a.p)
          .slice(0, Math.min(3, probs.length))
          .map(({ k, p }) => `${datasetNumeric.classes[k] ?? k}: ${(p * 100).toFixed(2)}%`)
          .join(" · ");

        setKV(previewKV, [
          ["Best class", `${predK} (${datasetNumeric.classes[predK] ?? predK})`],
          ["Best logit (Q)", String(res.bestLogitQ)],
          ["Top probabilities", top || "—"],
        ]);
        log(`[${nowTs()}] Local predict: class=${predK}`);
        return;
      }

      if (selectedTask === "multilabel_classification" && Array.isArray(datasetNumeric?.labelNames)) {
        const logitsQ = predictMultiQ(trained.decoded, vals);
        const scl = Number(trained.decoded.scaleQ || 1);
        const probs = new Array(logitsQ.length);
        for (let k = 0; k < logitsQ.length; k++) {
          const z = logitsQ[k] / scl;
          probs[k] = 1 / (1 + Math.exp(-z));
        }
        const names = datasetNumeric.labelNames;
        const predPos = [];
        for (let k = 0; k < probs.length; k++) {
          if (probs[k] >= 0.5) predPos.push(names[k] ?? `label${k}`);
        }
        const predShown = predPos.slice(0, 12).join(", ") + (predPos.length > 12 ? ", …" : "");
        const top = probs
          .map((p, k) => ({ k, p }))
          .sort((a, b) => b.p - a.p)
          .slice(0, Math.min(5, probs.length))
          .map(({ k, p }) => `${names[k] ?? k}: ${(p * 100).toFixed(2)}%`)
          .join(" · ");

        setKV(previewKV, [
          ["Predicted positives", predPos.length ? predShown : "(none)"],
          ["Top probabilities", top || "—"],
          ["Micro accuracy vs GT", "—"],
        ]);
        log(`[${nowTs()}] Local predict: positives=${predPos.length}`);
        return;
      }

      const scoreQ = predictQ(trained.decoded, vals);
      if (selectedTask === "binary_classification" && datasetNumeric?.classes) {
        const logit = scoreQ / trained.decoded.scaleQ;
        const prob = sigmoid(logit);
        const pred01 = prob >= 0.5 ? 1 : 0;
        setKV(previewKV, [
          ["Logit (Q)", String(scoreQ)],
          ["Probability", prob.toFixed(6)],
          ["Predicted", `${pred01} (${datasetNumeric.classes[pred01]})`],
        ]);
        log(`[${nowTs()}] Local predict: logit=${logit} prob=${prob}`);
      } else {
        const value = scoreQ / trained.decoded.scaleQ;
        setKV(previewKV, [["Local score (Q)", String(scoreQ)], ["Value", String(value)]]);
        log(`[${nowTs()}] Local predict: scoreQ=${scoreQ} value=${value}`);
      }
    } catch (e) {
      log(`[${nowTs()}] [error] ${e.message || e}`);
    }
  });

  // ===== Dataset / task selection UI =====

  // Default task from UI.
  selectedTask = String(taskSel?.value || "regression");

  function updateDatasetSummary() {
    if (!parsed) {
      dsNotes.textContent = "Upload a CSV to begin.";
      setKV(dsKV, []);
      if (classBox) classBox.style.display = "none";
      if (multiClassBox) multiClassBox.style.display = "none";
      try { ds3dRefreshControls(); } catch {}
      try { ds3dScheduleRender("no-dataset"); } catch {}
      return;
    }

    const taskTxt = taskLabel(selectedTask);

    // Task-specific label UI
    let note = "";
    const entries = [
      ["Task", taskTxt],
      ["Columns", String(parsed.headers.length)],
      ["Rows (raw)", String(parsed.rows.length)],
      ["Selected features", String(selectedFeatures.length)]
    ];

    if (selectedTask === "multilabel_classification") {
      const cols = Array.isArray(selectedLabelCols) ? selectedLabelCols : [];
      const names = cols.map((i) => parsed.headers[i] || `col${i}`);
      note = `Dataset ready. Task=${taskTxt} · Labels=${cols.length} columns · Selected features=${selectedFeatures.length}`;
      if (cols.length) {
        const few = names.slice(0, 5).join(", ");
        note += ` · [${few}${cols.length > 5 ? ", …" : ""}]`;
      }
      if (cols.length < 2) note += " · (select 2+ label columns)";
      entries.push(["Labels", String(cols.length)]);
      if (names.length) {
        entries.push(["Label columns", names.slice(0, 8).join(", ") + (names.length > 8 ? ", …" : "")]);
      }
    } else {
      const labelName = parsed.headers[selectedLabel] || `col${selectedLabel}`;
      note = `Dataset ready. Task=${taskTxt} · Label="${labelName}" · Selected features=${selectedFeatures.length}`;
      entries.splice(3, 0, ["Label", labelName]);

      if (selectedTask === "binary_classification" && selectedNegLabel && selectedPosLabel) {
        note += ` · Classes: 0="${selectedNegLabel}" 1="${selectedPosLabel}"`;
        if (labelValuesInfo?.values?.length > 2) note += " · (other labels will be dropped)";
      } else if (selectedTask === "multiclass_classification" && Array.isArray(selectedMultiLabels) && selectedMultiLabels.length >= 2) {
        note += ` · Classes: K=${selectedMultiLabels.length}`;
        const few = selectedMultiLabels.slice(0, 5).map((x) => String(x)).join(", ");
        note += ` · [${few}${selectedMultiLabels.length > 5 ? ", …" : ""}]`;
        if (labelValuesInfo?.values?.length && labelValuesInfo.values.length > selectedMultiLabels.length) note += " · (other labels will be dropped)";
      }

      if (selectedTask === "binary_classification" && selectedNegLabel && selectedPosLabel) {
        entries.push(["Class mapping", `0=${selectedNegLabel}, 1=${selectedPosLabel}`]);
      } else if (selectedTask === "multiclass_classification" && Array.isArray(selectedMultiLabels) && selectedMultiLabels.length >= 2) {
        entries.push(["Classes", `${selectedMultiLabels.length}`]);
        entries.push(["Class labels", selectedMultiLabels.slice(0, 8).join(", ") + (selectedMultiLabels.length > 8 ? ", …" : "")]);
      }
    }

    dsNotes.textContent = note;

    setKV(dsKV, entries);
    try { updateImbalanceUI(); } catch {}

    // Keep the 3D distribution controls in sync with the current dataset / selection.
    try { ds3dRefreshControls(); } catch {}
    try { ds3dScheduleRender("dataset-summary"); } catch {}
  }



  // ===== Dataset 3D Distribution ("Data Galaxy") =====

  function ds3dPlotlyReady() {
    const P = globalThis.Plotly;
    return !!(P && typeof P.react === "function" && ds3dPlot);
  }

  function ds3dSetNote(txt) {
    if (ds3dNote) ds3dNote.textContent = String(txt || "");
  }

  function ds3dPurgePlot() {
    if (!ds3dPlot) return;
    try {
      if (globalThis.Plotly && typeof globalThis.Plotly.purge === "function") {
        globalThis.Plotly.purge(ds3dPlot);
      }
    } catch {}
    try { ds3dPlot.innerHTML = ""; } catch {}
  }

  function ds3dRefreshControls() {
    if (!ds3dDetails) return;

    // Plotly is loaded from a CDN; in offline / blocked contexts, it may be unavailable.
    if (!globalThis.Plotly) {
      ds3dSetNote("Plotly.js failed to load. Check your network / adblocker.");
      return;
    }

    const mode = String(ds3dMode?.value || "pca");

    // Mode hint
    if (ds3dHint) ds3dHint.textContent = (mode === "scatter3") ? "3-feat" : "PCA";

    // Show/hide feature pickers
    if (ds3dFeatureBox) ds3dFeatureBox.style.display = (mode === "scatter3") ? "" : "none";

    // Sample input clamp
    if (ds3dSample) {
      const v = parseInt(ds3dSample.value || "2000", 10);
      const cl = clampInt(v, 500, 10000);
      if (String(cl) !== String(ds3dSample.value)) ds3dSample.value = String(cl);
    }

    // Feature selectors are based on *selected* feature columns.
    const featCols = Array.isArray(selectedFeatures) ? selectedFeatures.slice() : [];
    const featOptions = [];
    if (parsed && parsed.headers && featCols.length) {
      for (const colIdx of featCols) {
        const idx = colIdx | 0;
        const name = parsed.headers[idx] || `col${idx}`;
        featOptions.push({ value: String(idx), text: name });
      }
    }

    function _fillSelect(sel, fallbackIdx) {
      if (!sel) return;
      const prev = String(sel.value || "");
      sel.innerHTML = "";
      for (const o of featOptions) {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.text;
        sel.appendChild(opt);
      }
      // Restore previous choice if still available; else choose a sensible default.
      const still = featOptions.some(o => o.value === prev);
      if (still) sel.value = prev;
      else if (featOptions.length) {
        const pick = featOptions[Math.min(fallbackIdx, featOptions.length - 1)].value;
        sel.value = pick;
      }
      // Disable if not enough features.
      sel.disabled = (featOptions.length < 1);
    }

    _fillSelect(ds3dX, 0);
    _fillSelect(ds3dY, 1);
    _fillSelect(ds3dZ, 2);

    // Color-by options depend on the task.
    if (ds3dColor) {
      const prev = String(ds3dColor.value || "");
      ds3dColor.innerHTML = "";

      if (!parsed) {
        const opt = document.createElement("option");
        opt.value = "none";
        opt.textContent = "(upload CSV)";
        opt.disabled = true;
        ds3dColor.appendChild(opt);
        ds3dColor.value = "none";
      } else if (selectedTask === "multilabel_classification") {
        const cols = Array.isArray(selectedLabelCols) ? selectedLabelCols.slice() : [];
        if (!cols.length) {
          const opt = document.createElement("option");
          opt.value = "none";
          opt.textContent = "(select label columns)";
          opt.disabled = true;
          ds3dColor.appendChild(opt);
          ds3dColor.value = "none";
        } else {
          for (let k = 0; k < cols.length; k++) {
            const colIdx = cols[k] | 0;
            const nm = parsed.headers[colIdx] || `label${k}`;
            const opt = document.createElement("option");
            opt.value = `ml:${colIdx}`;
            opt.textContent = nm;
            ds3dColor.appendChild(opt);
          }
          // Restore if still available
          const still = Array.from(ds3dColor.options).some(o => o.value === prev);
          ds3dColor.value = still ? prev : String(ds3dColor.options[0]?.value || "none");
        }
      } else if (selectedTask === "regression") {
        const opt = document.createElement("option");
        opt.value = "target";
        opt.textContent = "Target";
        ds3dColor.appendChild(opt);
        ds3dColor.value = "target";
      } else {
        const labelName = parsed.headers[selectedLabel] || `col${selectedLabel}`;
        const opt = document.createElement("option");
        opt.value = "label";
        opt.textContent = `Label (${labelName})`;
        ds3dColor.appendChild(opt);
        ds3dColor.value = "label";
      }
    }
  }

  function ds3dScheduleRender(reason = "") {
    if (!ds3dDetails) return;
    // Only render when the Dataset tab is visible (keeps other tabs snappy).
    try {
      if (tabsApi?.getActive?.() !== "dataset") return;
    } catch {}

    if (_ds3dTimer) {
      try { clearTimeout(_ds3dTimer); } catch {}
      _ds3dTimer = null;
    }
    const token = ++_ds3dNonce;
    _ds3dTimer = setTimeout(() => {
      void ds3dRender(token, reason);
    }, 180);
  }

  function _ds3dIsStrictNumber(str) {
    const s = String(str ?? "").trim();
    if (!s) return false;
    return /^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$/.test(s);
  }

  function _ds3dCanonLabel(raw) {
    const s = String(raw ?? "").trim();
    if (!s) return "";
    if (_ds3dIsStrictNumber(s)) {
      const n = Number(s);
      if (Number.isFinite(n)) return String(n);
    }
    return s;
  }

  function _ds3dParseBinary01(raw) {
    if (raw === null || raw === undefined) return null;
    const s = String(raw).trim();
    if (!s) return null;
    const k = s.toLowerCase();
    if (k === "0" || k === "0.0") return 0;
    if (k === "1" || k === "1.0") return 1;
    if (k === "true" || k === "t" || k === "yes" || k === "y") return 1;
    if (k === "false" || k === "f" || k === "no" || k === "n") return 0;
    if (_ds3dIsStrictNumber(k)) {
      const n = Number(k);
      if (n === 0) return 0;
      if (n === 1) return 1;
    }
    return null;
  }

  function _ds3dMakeKey({ mode, sampleN, featIdx }) {
    const seed = clampInt(parseInt(seedNum?.value || "42", 10), 1, 2147483647);
    const featKey = Array.isArray(featIdx) ? featIdx.join(",") : "";

    if (!parsed) return `none|${mode}|k=${sampleN}`;

    if (selectedTask === "multilabel_classification") {
      const cols = Array.isArray(selectedLabelCols) ? selectedLabelCols.slice() : [];
      return `ml|${mode}|k=${sampleN}|seed=${seed}|labels=${cols.join(",")}|feat=${featKey}`;
    }

    const labelIdx = Number(selectedLabel);
    if (selectedTask === "binary_classification") {
      return `bin|${mode}|k=${sampleN}|seed=${seed}|label=${labelIdx}|neg=${String(selectedNegLabel)}|pos=${String(selectedPosLabel)}|feat=${featKey}`;
    }
    if (selectedTask === "multiclass_classification") {
      const labs = Array.isArray(selectedMultiLabels) ? selectedMultiLabels.join("|") : "";
      return `mc|${mode}|k=${sampleN}|seed=${seed}|label=${labelIdx}|labs=${labs}|feat=${featKey}`;
    }
    // regression
    return `reg|${mode}|k=${sampleN}|seed=${seed}|label=${labelIdx}|feat=${featKey}`;
  }

  function _ds3dYield() {
    return new Promise((res) => setTimeout(res, 0));
  }

  async function _ds3dBuildSample({ token, mode, sampleN, featIdx }) {
    if (!parsed) throw new Error("Upload a CSV first");
    const rows = parsed.rows || [];
    const headers = parsed.headers || [];

    const featureIndices = Array.from(featIdx || []).map((x) => Number(x)).filter((x) => Number.isFinite(x)).map((x) => x | 0);
    const d = featureIndices.length;
    if (d < 1) throw new Error("Select at least 1 feature column");
    if (mode === "scatter3" && d !== 3) throw new Error("3-feature scatter requires exactly 3 feature columns");

    const kMax = clampInt(sampleN | 0, 500, 10000);
    const seed = clampInt(parseInt(seedNum?.value || "42", 10), 1, 2147483647);
    const rng = _xorshift32(seed ^ 0xC0FFEE);

    // Reservoir storage (only keep what we need for plotting)
    const X = new Float32Array(kMax * d);
    const rowIdx = new Int32Array(kMax);

    let yReg = null;
    let yCls = null;
    let yML = null;
    let labelCols = null;
    let labelNames = null;
    let classLabels = null;
    let classesBin = null;

    if (selectedTask === "regression") {
      yReg = new Float32Array(kMax);
    } else if (selectedTask === "binary_classification") {
      yCls = new Int32Array(kMax);
      const neg = _ds3dCanonLabel(selectedNegLabel);
      const pos = _ds3dCanonLabel(selectedPosLabel);
      if (!neg || !pos || neg === pos) throw new Error("Select negative/positive classes");
      classesBin = { 0: neg, 1: pos };
    } else if (selectedTask === "multiclass_classification") {
      yCls = new Int32Array(kMax);
      const labsRaw = Array.isArray(selectedMultiLabels) ? selectedMultiLabels.map(String) : [];
      if (labsRaw.length < 2) throw new Error("Select 2+ classes");
      classLabels = labsRaw.map(_ds3dCanonLabel);
    } else if (selectedTask === "multilabel_classification") {
      labelCols = Array.isArray(selectedLabelCols) ? selectedLabelCols.slice() : [];
      if (!labelCols.length) throw new Error("Select label columns for multilabel");
      labelNames = labelCols.map((i, k) => headers[i | 0] || `label${k}`);
      yML = new Uint8Array(kMax * labelCols.length);
    }

    // For multiclass mapping
    let classMap = null;
    if (selectedTask === "multiclass_classification") {
      classMap = new Map();
      for (let i = 0; i < classLabels.length; i++) {
        const v = String(classLabels[i] || "");
        if (!v) continue;
        if (!classMap.has(v)) classMap.set(v, i);
      }
      if (classMap.size < 2) throw new Error("Need at least 2 distinct classes");
    }

    const labelIndex = Number(selectedLabel);
    const tmpFeat = new Float32Array(d);
    const tmpLab = (selectedTask === "multilabel_classification" && labelCols) ? new Uint8Array(labelCols.length) : null;

    let filled = 0;
    let valid = 0;

    for (let r = 0; r < rows.length; r++) {
      if (token !== _ds3dNonce) return null;
      const row = rows[r];

      // Parse label(s) first.
      let yValReg = 0;
      let yValCls = 0;

      if (selectedTask === "regression") {
        const yy = parseFloat(row[labelIndex]);
        if (!Number.isFinite(yy)) continue;
        yValReg = yy;
      } else if (selectedTask === "binary_classification") {
        const lab = _ds3dCanonLabel(row[labelIndex]);
        if (!lab) continue;
        if (lab === classesBin[0]) yValCls = 0;
        else if (lab === classesBin[1]) yValCls = 1;
        else continue;
      } else if (selectedTask === "multiclass_classification") {
        const lab = _ds3dCanonLabel(row[labelIndex]);
        if (!lab) continue;
        const cls = classMap.get(lab);
        if (cls === undefined) continue;
        yValCls = cls | 0;
      } else if (selectedTask === "multilabel_classification") {
        let ok = true;
        for (let k = 0; k < labelCols.length; k++) {
          const b = _ds3dParseBinary01(row[labelCols[k]]);
          if (b === null) { ok = false; break; }
          tmpLab[k] = b;
        }
        if (!ok) continue;
      }

      // Parse features
      let okFeat = true;
      for (let j = 0; j < d; j++) {
        const v = parseFloat(row[featureIndices[j]]);
        if (!Number.isFinite(v)) { okFeat = false; break; }
        tmpFeat[j] = v;
      }
      if (!okFeat) continue;

      const rowValidIdx = valid;
      valid += 1;

      let pos = -1;
      if (filled < kMax) {
        pos = filled;
        filled += 1;
      } else {
        // Reservoir replacement
        const j = Math.floor(_rand01(rng) * valid);
        if (j < 0 || j >= kMax) {
          // not selected
          if ((r % 6000) === 0) await _ds3dYield();
          continue;
        }
        pos = j;
      }

      const base = pos * d;
      for (let j = 0; j < d; j++) X[base + j] = tmpFeat[j];
      rowIdx[pos] = rowValidIdx;

      if (yReg) yReg[pos] = yValReg;
      if (yCls) yCls[pos] = yValCls;
      if (yML) {
        const L = labelCols.length;
        const baseY = pos * L;
        for (let k = 0; k < L; k++) yML[baseY + k] = tmpLab[k];
      }

      if ((r % 6000) === 0) await _ds3dYield();
    }

    const n = filled;
    const featureNames = featureIndices.map((i) => headers[i] || `col${i}`);

    return {
      n,
      valid,
      d,
      featureIndices,
      featureNames,
      X: X.subarray(0, n * d),
      rowIdx: rowIdx.subarray(0, n),
      yReg: yReg ? yReg.subarray(0, n) : null,
      yCls: yCls ? yCls.subarray(0, n) : null,
      yML,
      labelCols,
      labelNames,
      classLabels,
      classesBin,
    };
  }

  async function _ds3dComputeCoords({ token, mode, sample }) {
    const n = sample.n;
    const d = sample.d;
    const X = sample.X;

    if (mode === "scatter3") {
      const x = new Float32Array(n);
      const y = new Float32Array(n);
      const z = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const base = i * 3;
        x[i] = X[base + 0];
        y[i] = X[base + 1];
        z[i] = X[base + 2];
      }
      return { x, y, z, axis: sample.featureNames.slice(0, 3) };
    }

    // PCA(3) embedding
    if (n < 2) {
      const x = new Float32Array(n);
      const y = new Float32Array(n);
      const z = new Float32Array(n);
      return { x, y, z, axis: ["PC1", "PC2", "PC3"] };
    }

    // Standardize X in-place to reduce scale issues.
    ds3dSetNote(`Computing PCA(3)… (n=${n.toLocaleString()}, d=${d.toLocaleString()})`);
    const mean = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      const base = i * d;
      for (let j = 0; j < d; j++) mean[j] += X[base + j];
      if ((i % 2000) === 0) {
        if (token !== _ds3dNonce) return null;
        await _ds3dYield();
      }
    }
    for (let j = 0; j < d; j++) mean[j] /= n;

    const varr = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      const base = i * d;
      for (let j = 0; j < d; j++) {
        const dv = X[base + j] - mean[j];
        varr[j] += dv * dv;
      }
      if ((i % 2000) === 0) {
        if (token !== _ds3dNonce) return null;
        await _ds3dYield();
      }
    }
    const denom = Math.max(1, n - 1);
    const std = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      const s = Math.sqrt(varr[j] / denom);
      std[j] = (s > 0 && Number.isFinite(s)) ? s : 1;
    }
    for (let i = 0; i < n; i++) {
      const base = i * d;
      for (let j = 0; j < d; j++) X[base + j] = (X[base + j] - mean[j]) / std[j];
      if ((i % 2000) === 0) {
        if (token !== _ds3dNonce) return null;
        await _ds3dYield();
      }
    }

    // Power-iteration on covariance: C v = X^T (X v) / (n-1)
    const nIter = (d > 256) ? 14 : 20;
    const rng = _xorshift32((clampInt(parseInt(seedNum?.value || "42", 10), 1, 2147483647) ^ 0x9E3779B9) | 0);

    function dotVV(a, b) {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += a[i] * b[i];
      return s;
    }
    function normV(a) {
      return Math.sqrt(Math.max(0, dotVV(a, a)));
    }
    function normalizeV(a) {
      const nrm = normV(a);
      const inv = nrm > 0 ? (1 / nrm) : 1;
      for (let i = 0; i < a.length; i++) a[i] *= inv;
      return a;
    }

    function covMul(v) {
      const out = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        const base = i * d;
        let dp = 0;
        for (let j = 0; j < d; j++) dp += X[base + j] * v[j];
        for (let j = 0; j < d; j++) out[j] += X[base + j] * dp;
      }
      const inv = 1 / denom;
      for (let j = 0; j < d; j++) out[j] *= inv;
      return out;
    }

    async function powerIter(orth) {
      const v = new Float64Array(d);
      for (let j = 0; j < d; j++) v[j] = (_rand01(rng) * 2 - 1);
      normalizeV(v);

      for (let it = 0; it < nIter; it++) {
        if (token !== _ds3dNonce) return null;
        const w = covMul(v);

        if (Array.isArray(orth)) {
          for (const u of orth) {
            const proj = dotVV(w, u);
            for (let j = 0; j < d; j++) w[j] -= proj * u[j];
          }
        }

        normalizeV(w);
        for (let j = 0; j < d; j++) v[j] = w[j];

        await _ds3dYield();
      }
      return v;
    }

    const v1 = (d >= 1) ? await powerIter([]) : null;
    const v2 = (d >= 2) ? await powerIter([v1]) : null;
    const v3 = (d >= 3) ? await powerIter([v1, v2]) : null;
    if (token !== _ds3dNonce) return null;

    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const z = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const base = i * d;
      let a = 0, b = 0, c = 0;
      for (let j = 0; j < d; j++) {
        const vv = X[base + j];
        a += vv * (v1 ? v1[j] : 0);
        b += vv * (v2 ? v2[j] : 0);
        c += vv * (v3 ? v3[j] : 0);
      }
      x[i] = a;
      y[i] = b;
      z[i] = c;
      if ((i % 4000) === 0) {
        if (token !== _ds3dNonce) return null;
        await _ds3dYield();
      }
    }

    return { x, y, z, axis: ["PC1", "PC2", "PC3"] };
  }

  // Data Galaxy colors (default: blue gradient)
  const DS3D_BLUE_SCALE = [
    [0.0, "#dbeafe"],
    [0.5, "#60a5fa"],
    [1.0, "#1d4ed8"],
  ];

  function _ds3dBlueShade(t) {
    const clamp01 = (v) => Math.max(0, Math.min(1, v));
    const tt = clamp01(Number(t));
    // Linear interpolation between light and dark blue.
    const c0 = [219, 234, 254]; // #dbeafe
    const c1 = [29, 78, 216];   // #1d4ed8
    const r = Math.round(c0[0] + (c1[0] - c0[0]) * tt);
    const g = Math.round(c0[1] + (c1[1] - c0[1]) * tt);
    const b = Math.round(c0[2] + (c1[2] - c0[2]) * tt);
    return `rgb(${r},${g},${b})`;
  }

  function _ds3dTraceFromPoints({ name, xs, ys, zs, customdata, size = 3, opacity = 0.85, showLegend = true, color = null, colorscale = null, showScale = false, colorbarTitle = "" }) {
    const marker = { size, opacity };
    if (color != null) {
      marker.color = color;
      if (colorscale) marker.colorscale = colorscale;
      if (showScale) marker.colorbar = { title: String(colorbarTitle || "") };
    }
    return {
      type: "scatter3d",
      mode: "markers",
      name: String(name || ""),
      x: xs,
      y: ys,
      z: zs,
      customdata,
      hovertemplate: "row %{customdata[0]}<br>label %{customdata[1]}<extra></extra>",
      marker,
      showlegend: !!showLegend,
    };
  }

  async function ds3dRender(token, reason = "") {
    if (!ds3dDetails) return;
    try {
      if (tabsApi?.getActive?.() !== "dataset") return;
    } catch {}

    // Keep control lists fresh.
    ds3dRefreshControls();

    if (!globalThis.Plotly) {
      ds3dSetNote("Plotly.js failed to load. Check your network / adblocker.");
      return;
    }

    if (!parsed) {
      ds3dPurgePlot();
      ds3dSetNote("Upload a CSV and select features to render.");
      return;
    }

    const mode = String(ds3dMode?.value || "pca");
    const sampleN = clampInt(parseInt(ds3dSample?.value || "2000", 10), 500, 10000);

    // Feature columns for this view
    let featIdx = [];
    if (mode === "scatter3") {
      const a = Number(ds3dX?.value);
      const b = Number(ds3dY?.value);
      const c = Number(ds3dZ?.value);
      if (![a, b, c].every((x) => Number.isFinite(x))) {
        ds3dSetNote("Pick 3 feature columns to render.");
        return;
      }
      featIdx = [a | 0, b | 0, c | 0];
      if (featIdx.length !== 3) {
        ds3dSetNote("Pick 3 feature columns to render.");
        return;
      }
    } else {
      featIdx = Array.isArray(selectedFeatures) ? selectedFeatures.slice() : [];
      if (!featIdx.length) {
        ds3dSetNote("Select at least 1 feature column.");
        ds3dPurgePlot();
        return;
      }
    }

    // Basic task validation (for correct filtering)
    if (selectedTask === "multilabel_classification") {
      const cols = Array.isArray(selectedLabelCols) ? selectedLabelCols : [];
      if (!cols.length) {
        ds3dSetNote("Select label columns to color-by (multilabel).");
        ds3dPurgePlot();
        return;
      }
    } else {
      if (!Number.isFinite(Number(selectedLabel))) {
        ds3dSetNote("Select a label column.");
        ds3dPurgePlot();
        return;
      }
      if (selectedTask === "binary_classification") {
        if (!selectedNegLabel || !selectedPosLabel) {
          ds3dSetNote("Select negative/positive classes.");
          ds3dPurgePlot();
          return;
        }
      }
      if (selectedTask === "multiclass_classification") {
        if (!Array.isArray(selectedMultiLabels) || selectedMultiLabels.length < 2) {
          ds3dSetNote("Select 2+ classes.");
          ds3dPurgePlot();
          return;
        }
      }
    }

    const key = _ds3dMakeKey({ mode, sampleN, featIdx });
    const colorKey = String(ds3dColor?.value || "");

    // Reuse cached sample/coords when possible (fast path for color changes).
    let cached = null;
    if (_ds3dLastSample && _ds3dLastSample.key === key) {
      cached = _ds3dLastSample;
    }

    if (!cached) {
      ds3dSetNote(`Sampling & filtering… (k=${sampleN.toLocaleString()})`);
      const sample = await _ds3dBuildSample({ token, mode, sampleN, featIdx });
      if (!sample) return;
      if (token !== _ds3dNonce) return;

      if (!sample.n) {
        ds3dPurgePlot();
        ds3dSetNote("No usable rows after filtering (check label/class selections and missing values).");
        return;
      }

      const coords = await _ds3dComputeCoords({ token, mode, sample });
      if (!coords) return;
      if (token !== _ds3dNonce) return;

      cached = {
        key,
        mode,
        n: sample.n,
        valid: sample.valid,
        rowIdx: sample.rowIdx,
        coords,
        task: selectedTask,
        yReg: sample.yReg,
        yCls: sample.yCls,
        yML: sample.yML,
        labelCols: sample.labelCols,
        labelNames: sample.labelNames,
        classLabels: sample.classLabels,
        classesBin: sample.classesBin,
        axisFeatureNames: sample.featureNames,
      };

      _ds3dLastSample = cached;
      _ds3dLastKey = key;
    }

    // Build traces from cached sample.
    const x = cached.coords.x;
    const y = cached.coords.y;
    const z = cached.coords.z;
    const rowIdx = cached.rowIdx;

    const data = [];
    const N = cached.n;

    // Point size tuning for performance
    const pointSize = (N >= 8000) ? 2 : (N >= 3000 ? 3 : 4);
    const opacity = (N >= 8000) ? 0.65 : 0.80;

    if (cached.task === "regression") {
      const yTarget = cached.yReg;
      const xs = new Array(N);
      const ys = new Array(N);
      const zs = new Array(N);
      const colors = new Array(N);
      const custom = new Array(N);
      for (let i = 0; i < N; i++) {
        xs[i] = x[i];
        ys[i] = y[i];
        zs[i] = z[i];
        const t = Number(yTarget?.[i]);
        colors[i] = t;
        custom[i] = [rowIdx[i], Number.isFinite(t) ? (Math.round(t * 1e6) / 1e6) : "NaN"];
      }
      data.push(_ds3dTraceFromPoints({
        name: "Target",
        xs,
        ys,
        zs,
        customdata: custom,
        size: pointSize,
        opacity,
        showLegend: false,
        color: colors,
        colorscale: DS3D_BLUE_SCALE,
        showScale: true,
        colorbarTitle: "Target",
      }));
    } else if (cached.task === "binary_classification") {
      const yCls = cached.yCls;
      const clsNames = cached.classesBin || { 0: "0", 1: "1" };
      for (const cls of [0, 1]) {
        const shade = _ds3dBlueShade(cls === 0 ? 0.20 : 0.85);
        const xs = [];
        const ys = [];
        const zs = [];
        const custom = [];
        for (let i = 0; i < N; i++) {
          if ((yCls[i] | 0) !== cls) continue;
          xs.push(x[i]);
          ys.push(y[i]);
          zs.push(z[i]);
          custom.push([rowIdx[i], clsNames[cls] ?? String(cls)]);
        }
        data.push(_ds3dTraceFromPoints({
          name: `${cls} (${clsNames[cls] ?? cls})`,
          xs,
          ys,
          zs,
          customdata: custom,
          size: pointSize,
          opacity,
          showLegend: true,
          color: shade,
        }));
      }
    } else if (cached.task === "multiclass_classification") {
      const yCls = cached.yCls;
      const labs = Array.isArray(cached.classLabels) ? cached.classLabels : [];
      const K = labs.length;

      // For very large K, Plotly traces can become heavy. Keep the legend readable.
      for (let cls = 0; cls < K; cls++) {
        const name = labs[cls] ?? String(cls);
        const shade = _ds3dBlueShade(K <= 1 ? 0.5 : (0.15 + 0.75 * (cls / (K - 1))));
        const xs = [];
        const ys = [];
        const zs = [];
        const custom = [];
        for (let i = 0; i < N; i++) {
          if ((yCls[i] | 0) !== cls) continue;
          xs.push(x[i]);
          ys.push(y[i]);
          zs.push(z[i]);
          custom.push([rowIdx[i], name]);
        }
        if (!xs.length) continue;
        data.push(_ds3dTraceFromPoints({
          name: `${cls}: ${name}`,
          xs,
          ys,
          zs,
          customdata: custom,
          size: pointSize,
          opacity,
          showLegend: true,
          color: shade,
        }));
      }
    } else if (cached.task === "multilabel_classification") {
      const yML = cached.yML;
      const cols = Array.isArray(cached.labelCols) ? cached.labelCols : [];
      const names = Array.isArray(cached.labelNames) ? cached.labelNames : [];
      if (!(yML instanceof Uint8Array) || !cols.length) {
        ds3dSetNote("Multilabel data unavailable for plotting.");
        return;
      }
      let colIdx = null;
      if (colorKey.startsWith("ml:")) {
        colIdx = Number(colorKey.slice(3));
      }
      if (!Number.isFinite(colIdx)) colIdx = cols[0];
      const k = cols.findIndex((c) => (c | 0) === (colIdx | 0));
      const kk = (k >= 0) ? k : 0;
      const labelName = names[kk] || `label${kk}`;
      const L = cols.length;

      // Two traces: 0 and 1
      for (const val of [0, 1]) {
        const shade = _ds3dBlueShade(val === 0 ? 0.20 : 0.85);
        const xs = [];
        const ys = [];
        const zs = [];
        const custom = [];
        for (let i = 0; i < N; i++) {
          const b = yML[i * L + kk] | 0;
          if (b !== val) continue;
          xs.push(x[i]);
          ys.push(y[i]);
          zs.push(z[i]);
          custom.push([rowIdx[i], `${labelName}=${val}`]);
        }
        data.push(_ds3dTraceFromPoints({
          name: `${labelName} = ${val}`,
          xs,
          ys,
          zs,
          customdata: custom,
          size: pointSize,
          opacity,
          showLegend: true,
          color: shade,
        }));
      }
    }

    // Axis titles
    const axis = cached.coords.axis || ["X", "Y", "Z"];
    const layout = {
      margin: { l: 0, r: 0, b: 0, t: 26 },
      scene: {
        xaxis: { title: String(axis[0] || "X") },
        yaxis: { title: String(axis[1] || "Y") },
        zaxis: { title: String(axis[2] || "Z") },
      },
      showlegend: (cached.task !== "regression"),
    };

    const config = {
      responsive: true,
      displaylogo: false,
    };

    if (!ds3dPlotlyReady()) {
      ds3dSetNote("Plotly is not ready.");
      return;
    }

    try {
      await globalThis.Plotly.react(ds3dPlot, data, layout, config);
    } catch (e) {
      ds3dSetNote(`Plot render failed: ${e?.message || e}`);
      return;
    }

    // Click → set row index for local preview.
    if (!ds3dPlot._ds3dClickBound && typeof ds3dPlot.on === "function") {
      ds3dPlot._ds3dClickBound = true;
      ds3dPlot.on("plotly_click", (ev) => {
        try {
          const p = ev?.points?.[0];
          const cd = p?.customdata;
          const idx = Array.isArray(cd) ? cd[0] : cd;
          if (!Number.isFinite(Number(idx))) return;
          if (rowIndex) rowIndex.value = String(Number(idx) | 0);

          // If a model is trained, also load feature inputs immediately.
          if (trained?.decoded && datasetNumeric?.X?.length) {
            try { loadRowIntoInputs(Number(idx) | 0); } catch {}
          }
          log(`[${nowTs()}] Data Galaxy click → rowIndex=${Number(idx) | 0}`);
        } catch {}
      });
    }

    ds3dSetNote(`Showing ${N.toLocaleString()} point${N === 1 ? "" : "s"} (sampled from ${cached.valid.toLocaleString()} usable rows). Click a point to set the dataset row index.`);

    // Resize in case the details panel was just opened.
    try {
      requestAnimationFrame(() => {
        try { globalThis.Plotly.Plots.resize(ds3dPlot); } catch {}
      });
    } catch {}
  }

  // Wire 3D controls
  if (ds3dMode) ds3dMode.addEventListener("change", () => {
    ds3dRefreshControls();
    ds3dScheduleRender("mode");
  });
  if (ds3dSample) {
    ds3dSample.addEventListener("input", () => ds3dScheduleRender("sample"));
    ds3dSample.addEventListener("change", () => ds3dScheduleRender("sample"));
  }
  if (ds3dColor) ds3dColor.addEventListener("change", () => ds3dScheduleRender("color"));
  if (ds3dX) ds3dX.addEventListener("change", () => ds3dScheduleRender("x"));
  if (ds3dY) ds3dY.addEventListener("change", () => ds3dScheduleRender("y"));
  if (ds3dZ) ds3dZ.addEventListener("change", () => ds3dScheduleRender("z"));


  // ===== Class imbalance UI =====
  function _parseBinary01(raw) {
    if (raw == null) return null;
    const s = String(raw).trim().toLowerCase();
    if (!s) return null;
    if (s === "1" || s === "true" || s === "t" || s === "yes" || s === "y") return 1;
    if (s === "0" || s === "false" || s === "f" || s === "no" || s === "n") return 0;
    return null;
  }

  function _imbClampCap(x) {
    const v = parseFloat(String(x ?? ""));
    if (!Number.isFinite(v)) return 20;
    return Math.max(1, Math.min(1000, v));
  }

  function _imbWeightInputHTML({ key, value, disabled }) {
    const v = Number.isFinite(value) ? String(Math.round(value * 1e6) / 1e6) : "1";
    const dis = disabled ? "disabled" : "";
    return `<input class="imbInput" type="number" min="0.000001" step="0.01" value="${v}" data-imb-key="${key}" ${dis} />`;
  }

  function _imbRowHTML({ name, meta, key, weight, disabled }) {
    const inp = _imbWeightInputHTML({ key, value: weight, disabled });
    return `
      <div class="imbRow">
        <div class="imbLeft">
          <div class="imbName">${escapeHtml(name)}</div>
          <div class="imbMeta">${escapeHtml(meta || "")}</div>
        </div>
        ${inp}
      </div>
    `;
  }

  function _imbSetVisible(show) {
    if (!imbalanceDetails) return;
    imbalanceDetails.style.display = show ? "block" : "none";
  }

  function _imbSetSummaryHint(t) {
    if (imbSummaryHint) imbSummaryHint.textContent = t || "Optional";
  }

  function _imbSetNote(t) {
    if (imbNote) imbNote.textContent = t || "";
  }

  function _imbClearRows() {
    if (imbRows) imbRows.innerHTML = "";
  }

  function _imbResetDisableState() {
    if (imbNormalize) imbNormalize.disabled = false;
    if (imbStratify) imbStratify.disabled = false;
  }

  function _imbRenderRows(rows, { kind, editable }) {
    if (!imbRows) return;
    imbRows.innerHTML = rows.map(r => _imbRowHTML({ ...r, disabled: !editable })).join("");

    // Update manual state on input.
    for (const inp of Array.from(imbRows.querySelectorAll('input[data-imb-key]'))) {
      inp.addEventListener('input', () => {
        const key = String(inp.getAttribute('data-imb-key') || "");
        const v = parseFloat(inp.value);
        const val = Number.isFinite(v) ? Math.max(0.000001, v) : 1;

        if (kind === 'binary') {
          if (key === 'bin0') _imbManual.binary.w0 = val;
          if (key === 'bin1') _imbManual.binary.w1 = val;
        } else if (kind === 'multiclass') {
          _imbManual.multiclass[key] = val;
        } else if (kind === 'multilabel') {
          // key is column index string
          _imbManual.multilabel[key] = val;
        }

        // Any change invalidates trained output.
        trained = null;
        updateDeployState();
      });
    }
  }

  function updateImbalanceUI() {
    if (!imbalanceDetails) return;

    const isClassTask = (selectedTask === "binary_classification" || selectedTask === "multiclass_classification" || selectedTask === "multilabel_classification");
    if (!parsed || !isClassTask) {
      _imbSetVisible(false);
      return;
    }

    _imbSetVisible(true);
    _imbResetDisableState();

    const mode = String(imbMode?.value || "none");
    const cap = _imbClampCap(imbCap?.value);
    const normalize = !!imbNormalize?.checked;
    let stratify = !!imbStratify?.checked;

    // Stratified split is only for binary/multiclass.
    if (selectedTask === "multilabel_classification") {
      if (imbStratify) {
        imbStratify.checked = false;
        imbStratify.disabled = true;
      }
      stratify = false;
    }

    // We'll show approximate counts based on label columns only.
    // Final training counts may differ after dropping rows with non-numeric feature values.
    const rows = [];

    if (selectedTask === "binary_classification") {
      _imbSetSummaryHint("Binary · optional");

      if (!labelValuesInfo || !selectedNegLabel || !selectedPosLabel) {
        _imbSetNote("Select a label column and choose Negative/Positive classes to enable imbalance handling.");
        _imbClearRows();
        return;
      }

      const c0 = Number(labelValuesInfo.counts.get(selectedNegLabel) || 0);
      const c1 = Number(labelValuesInfo.counts.get(selectedPosLabel) || 0);
      const N = c0 + c1;

      let w0 = 1, w1 = 1;
      if (mode === "auto" && N > 0) {
        w0 = c0 > 0 ? (N / (2 * c0)) : 1;
        w1 = c1 > 0 ? (N / (2 * c1)) : 1;
      } else if (mode === "manual") {
        w0 = Number(_imbManual.binary.w0 || 1);
        w1 = Number(_imbManual.binary.w1 || 1);
      }

      // cap
      w0 = Math.min(cap, Math.max(0.000001, w0));
      w1 = Math.min(cap, Math.max(0.000001, w1));

      // normalize avg weight to ~1 (approx based on selected-class counts)
      if (normalize && N > 0) {
        const avg = (w0 * c0 + w1 * c1) / N;
        if (Number.isFinite(avg) && avg > 0) {
          w0 /= avg;
          w1 /= avg;
        }
      }

      const ratio = (c0 > 0 && c1 > 0) ? (c0 / c1) : NaN;
      _imbSetSummaryHint(`Binary · neg=${c0} pos=${c1}${Number.isFinite(ratio) ? ` (${(ratio).toFixed(2)}:1)` : ''}`);
      _imbSetNote("Auto/manual weights are applied in the loss (training-only). Rows with other label values are dropped.");

      rows.push({ kind: "binary", key: "bin0", name: `0 = ${selectedNegLabel}`, meta: `count ${c0}`, weight: w0 });
      rows.push({ kind: "binary", key: "bin1", name: `1 = ${selectedPosLabel}`, meta: `count ${c1}`, weight: w1 });
      _imbRenderRows(rows, { kind: "binary", editable: (mode === "manual") });
      return;
    }

    if (selectedTask === "multiclass_classification") {
      _imbSetSummaryHint("Multiclass · optional");

      if (!labelValuesInfo || !Array.isArray(selectedMultiLabels) || selectedMultiLabels.length < 2) {
        _imbSetNote("Select at least 2 classes to enable multiclass imbalance handling.");
        _imbClearRows();
        return;
      }

      const labels = selectedMultiLabels.map(String);
      const counts = labels.map((lab) => Number(labelValuesInfo.counts.get(lab) || 0));
      const N = counts.reduce((a, b) => a + b, 0);
      const K = labels.length;

      let weights = new Array(K).fill(1);
      if (mode === "auto" && N > 0) {
        weights = counts.map((c) => (c > 0 ? (N / (K * c)) : 1));
      } else if (mode === "manual") {
        weights = labels.map((lab) => {
          const v = parseFloat(String(_imbManual.multiclass[lab] ?? ""));
          return Number.isFinite(v) ? Math.max(0.000001, v) : 1;
        });
      }

      weights = weights.map((w) => Math.min(cap, Math.max(0.000001, w)));

      if (normalize && N > 0) {
        let avg = 0;
        for (let i = 0; i < K; i++) avg += weights[i] * counts[i];
        avg /= N;
        if (Number.isFinite(avg) && avg > 0) weights = weights.map((w) => w / avg);
      }

      _imbSetSummaryHint(`Multiclass · K=${K} · stratify=${stratify ? 'on' : 'off'}`);
      _imbSetNote("Auto/manual class weights are applied in the loss (training-only). Rows with other labels are dropped.");

      for (let i = 0; i < K; i++) {
        const lab = labels[i];
        rows.push({ kind: "multiclass", key: lab, name: `Class ${i}: ${lab}`, meta: `count ${counts[i]}`, weight: weights[i] });
        // seed default manual weight for smoother UX
        if (!(_imbManual.multiclass[lab] > 0)) _imbManual.multiclass[lab] = weights[i];
      }

      _imbRenderRows(rows, { kind: "multiclass", editable: (mode === "manual") });
      return;
    }

    // Multilabel classification
    if (selectedTask === "multilabel_classification") {
      const cols = Array.isArray(selectedLabelCols) ? selectedLabelCols.map((x) => Number(x)) : [];
      if (!cols.length) {
        _imbSetSummaryHint("Multilabel · optional");
        _imbSetNote("Select label columns to configure imbalance handling.");
        _imbClearRows();
        return;
      }

      // Compute counts on rows where ALL selected labels are valid.
      const L = cols.length;
      const pos = new Array(L).fill(0);
      const neg = new Array(L).fill(0);
      let validRows = 0;

      for (const row of parsed.rows) {
        let ok = true;
        const vals = new Array(L);
        for (let j = 0; j < L; j++) {
          const v = _parseBinary01(row[cols[j]]);
          if (v === null) { ok = false; break; }
          vals[j] = v;
        }
        if (!ok) continue;
        validRows += 1;
        for (let j = 0; j < L; j++) {
          if (vals[j] === 1) pos[j] += 1;
          else neg[j] += 1;
        }
      }

      const mode2 = String(imbMode?.value || "none");
      const showMode = (mode2 === "auto" || mode2 === "manual");

      _imbSetSummaryHint(`Multilabel · labels=${L} · validRows=${validRows}`);
      _imbSetNote("Per-label positive weighting (training-only). Rows must have valid 0/1 (or true/false) for all selected label columns.");

      for (let j = 0; j < L; j++) {
        const colIdx = cols[j];
        const name = parsed.headers[colIdx] || `col${colIdx}`;
        const pCount = pos[j];
        const nCount = neg[j];
        const denom = pCount + nCount;

        let wPos = 1;
        if (mode2 === "auto" && denom > 0) {
          wPos = (pCount > 0) ? (nCount / pCount) : 1;
        } else if (mode2 === "manual") {
          const key = String(colIdx);
          const v = parseFloat(String(_imbManual.multilabel[key] ?? ""));
          wPos = Number.isFinite(v) ? Math.max(0.000001, v) : 1;
        }

        wPos = Math.min(cap, Math.max(0.000001, wPos));

        rows.push({ kind: "multilabel", key: String(colIdx), name: name, meta: `pos ${pCount} · neg ${nCount}`, weight: showMode ? wPos : 1 });

        if (!(_imbManual.multilabel[String(colIdx)] > 0)) _imbManual.multilabel[String(colIdx)] = wPos;
      }

      _imbRenderRows(rows, { kind: "multilabel", editable: (mode2 === "manual") });
      return;
    }
  }

  function readImbalanceConfigForParams({ nClasses }) {
    // Always pass an object (worker can ignore if mode=none)
    const isClassTask = (selectedTask === "binary_classification" || selectedTask === "multiclass_classification" || selectedTask === "multilabel_classification");
    if (!isClassTask) return { mode: "none", cap: 20, normalize: true, stratify: false };

    const mode = String(imbMode?.value || "none");
    const cap = _imbClampCap(imbCap?.value);
    const normalize = !!imbNormalize?.checked;
    const stratify = (selectedTask === "binary_classification" || selectedTask === "multiclass_classification") ? (!!imbStratify?.checked) : false;

    const out = { mode, cap, normalize, stratify };

    if (mode === "manual") {
      if (selectedTask === "binary_classification") {
        out.w0 = Number(_imbManual.binary.w0 || 1);
        out.w1 = Number(_imbManual.binary.w1 || 1);
      } else if (selectedTask === "multiclass_classification") {
        const labs = Array.isArray(selectedMultiLabels) ? selectedMultiLabels.map(String) : [];
        out.classWeights = labs.map((lab) => {
          const v = parseFloat(String(_imbManual.multiclass[lab] ?? ""));
          return Number.isFinite(v) ? Math.max(0.000001, v) : 1;
        });
      } else if (selectedTask === "multilabel_classification") {
        const cols = Array.isArray(selectedLabelCols) ? selectedLabelCols.map((x) => Number(x)) : [];
        out.posWeights = cols.map((colIdx) => {
          const v = parseFloat(String(_imbManual.multilabel[String(colIdx)] ?? ""));
          return Number.isFinite(v) ? Math.max(0.000001, v) : 1;
        });
      }
    }

    return out;
  }


  // Wire imbalance UI controls
  function _onImbChange() {
    // Any change invalidates trained output.
    trained = null;
    datasetNumeric = null;
    try { if (imbalanceDetails && String(imbMode?.value || "none") !== "none") imbalanceDetails.open = true; } catch {}
    try { updateImbalanceUI(); } catch {}
    updateDeployState();
  }
  if (imbMode) imbMode.addEventListener("change", _onImbChange);
  if (imbCap) {
    imbCap.addEventListener("input", _onImbChange);
    imbCap.addEventListener("change", _onImbChange);
  }
  if (imbNormalize) imbNormalize.addEventListener("change", _onImbChange);
  if (imbStratify) imbStratify.addEventListener("change", _onImbChange);

  function defaultBinaryMapping(values, allNumeric) {
    const vals = Array.from(values || []);
    if (vals.length < 2) return { neg: null, pos: null };
    if (allNumeric) {
      return { neg: vals[0], pos: vals[vals.length - 1] };
    }
    // Heuristics: prefer common "positive" tokens if present.
    const POS = new Set(["1","true","yes","y","t","pos","positive","spam","fraud"]);
    const NEG = new Set(["0","false","no","n","f","neg","negative","ham"]);
    let pos = null;
    let neg = null;
    for (const v of vals) {
      const k = String(v).trim().toLowerCase();
      if (!pos && POS.has(k)) pos = v;
      if (!neg && NEG.has(k)) neg = v;
    }
    if (pos && !neg) neg = vals.find(x => x !== pos) || null;
    if (neg && !pos) pos = vals.find(x => x !== neg) || null;
    if (!pos || !neg) {
      // Fallback: first two sorted values.
      neg = vals[0];
      pos = vals.find(x => x !== neg) || vals[1];
    }
    return { neg, pos };
  }

  function refreshClassUI({ reset = false } = {}) {
    // Hide both class UIs by default.
    if (classBox) classBox.style.display = "none";
    if (multiClassBox) multiClassBox.style.display = "none";
    if (classNote) classNote.textContent = "—";
    if (multiClassNote) multiClassNote.textContent = "—";

    labelValuesInfo = null;

    // Not a classification task (or no dataset).
    if (!parsed || (selectedTask !== "binary_classification" && selectedTask !== "multiclass_classification")) {
      selectedNegLabel = null;
      selectedPosLabel = null;
      selectedMultiLabels = [];
      return;
    }

    const labelIdx = Number(labelCol?.value || selectedLabel || 0);
    labelValuesInfo = inferLabelValues(parsed, labelIdx);
    const values = labelValuesInfo.values || [];

    // =========================
    // Binary classification UI
    // =========================
    if (selectedTask === "binary_classification") {
      if (!classBox || !negClassSel || !posClassSel || !classNote) return;

      // Rebuild options
      negClassSel.innerHTML = "";
      posClassSel.innerHTML = "";
      for (const v of values) {
        const opt0 = document.createElement("option");
        opt0.value = String(v);
        opt0.textContent = String(v);
        negClassSel.appendChild(opt0);

        const opt1 = document.createElement("option");
        opt1.value = String(v);
        opt1.textContent = String(v);
        posClassSel.appendChild(opt1);
      }

      if (values.length < 2) {
        selectedNegLabel = null;
        selectedPosLabel = null;
        classBox.style.display = "block";
        classNote.textContent = "Need at least 2 distinct label values for binary classification.";
        return;
      }

      const def = defaultBinaryMapping(values, labelValuesInfo.allNumeric);
      if (reset || !selectedNegLabel || !values.includes(selectedNegLabel)) selectedNegLabel = def.neg;
      if (reset || !selectedPosLabel || !values.includes(selectedPosLabel)) selectedPosLabel = def.pos;

      if (selectedNegLabel === selectedPosLabel) {
        // Force difference
        selectedPosLabel = values.find((v) => v !== selectedNegLabel) || def.pos;
      }

      // Apply values
      if (selectedNegLabel) negClassSel.value = String(selectedNegLabel);
      if (selectedPosLabel) posClassSel.value = String(selectedPosLabel);

      // Build a compact summary
      const parts = values.slice(0, 6).map((v) => {
        const c = labelValuesInfo.counts.get(v) || 0;
        return `${v}(${c})`;
      });
      let txt = `${values.length} unique labels: ${parts.join(", ")}`;
      if (values.length > 6) txt += "…";
      if (values.length > 2) txt += " · Rows with other labels will be dropped.";
      if (selectedNegLabel && selectedPosLabel) txt += ` · Mapping: 0="${selectedNegLabel}", 1="${selectedPosLabel}"`;
      classNote.textContent = txt;
      classBox.style.display = "block";
      return;
    }

    // =========================
    // Multiclass classification UI
    // =========================
    if (selectedTask === "multiclass_classification") {
      if (!multiClassBox || !multiClassSel || !multiClassNote) return;

      multiClassSel.innerHTML = "";
      for (const v of values) {
        const c = labelValuesInfo.counts.get(v) || 0;
        const opt = document.createElement("option");
        opt.value = String(v);
        opt.textContent = `${v} (${c})`;
        multiClassSel.appendChild(opt);
      }

      if (values.length < 2) {
        selectedMultiLabels = [];
        multiClassBox.style.display = "block";
        multiClassNote.textContent = "Need at least 2 distinct label values for multiclass classification.";
        return;
      }

      // Default: select all labels (in inferred order). Preserve existing selections when possible.
      const existing = new Set((selectedMultiLabels || []).map((x) => String(x)));
      if (reset || existing.size === 0) {
        selectedMultiLabels = values.map((v) => String(v));
      } else {
        // Keep only still-present labels, in the new inferred order.
        selectedMultiLabels = values.filter((v) => existing.has(String(v))).map((v) => String(v));
      }

      // Apply selection
      const selSet = new Set(selectedMultiLabels);
      for (const opt of Array.from(multiClassSel.options)) {
        opt.selected = selSet.has(opt.value);
      }

      // Note text
      const shown = values.slice(0, 6).map((v) => {
        const c = labelValuesInfo.counts.get(v) || 0;
        return `${v}(${c})`;
      }).join(", ");
      let txt = `${values.length} unique labels: ${shown}`;
      if (values.length > 6) txt += "…";
      txt += ` · Selected ${selectedMultiLabels.length}`;
      if (values.length > selectedMultiLabels.length) txt += " · Rows with other labels will be dropped.";
      multiClassNote.textContent = txt;
      multiClassBox.style.display = "block";
    }
  }

  function getSelectedLabelColsFromUI() {
    if (!multiLabelSel) return [];
    return Array.from(multiLabelSel.selectedOptions)
      .map((o) => Number(o.value))
      .filter((i) => Number.isFinite(i))
      .map((i) => i | 0);
  }

  function updateMultiLabelNoteUI() {
    if (!multiLabelNote) return;
    if (!parsed) { multiLabelNote.textContent = "—"; return; }
    if (!multiLabelSel) { multiLabelNote.textContent = "—"; return; }
    const cols = getSelectedLabelColsFromUI();
    selectedLabelCols = cols;
    if (!cols.length) {
      multiLabelNote.textContent = "Select 2+ label columns.";
      return;
    }
    const names = cols.map((i) => parsed.headers[i] || `col${i}`);
    const shown = names.slice(0, 6).join(", ");
    let txt = `Selected ${cols.length}: ${shown}`;
    if (cols.length > 6) txt += "…";
    if (cols.length < 2) txt += " · Need at least 2 for multilabel training.";
    multiLabelNote.textContent = txt;
  }

  function labelExclusionSet() {
    const set = new Set();
    if (!parsed) return set;
    if (selectedTask === "multilabel_classification") {
      for (const i of (selectedLabelCols || [])) set.add(i | 0);
    } else {
      const labelIdx = Number(labelCol?.value ?? selectedLabel ?? 0);
      if (Number.isFinite(labelIdx)) set.add(labelIdx | 0);
    }
    return set;
  }

  function applyFeatureExclusions() {
    if (!parsed || !featureCols) return;
    const exclude = labelExclusionSet();
    const inputs = Array.from(featureCols.querySelectorAll("input[type=checkbox][data-colidx]"));
    for (const chk of inputs) {
      const colIdx = Number(chk.dataset.colidx);
      const isExcluded = exclude.has(colIdx | 0);
      chk.disabled = isExcluded;
      if (isExcluded) chk.checked = false;
    }
  }

  function rebuildFeatureColsUI({ preserveSelection = true } = {}) {
    if (!parsed || !featureCols) return;
    const prevSel = new Set(preserveSelection ? (selectedFeatures || []).map((i) => String(i)) : []);
    featureCols.innerHTML = "";

    parsed.headers.forEach((h, idx) => {
      const wrap = document.createElement("label");
      wrap.style.textTransform = "none";
      wrap.style.letterSpacing = "0";
      wrap.style.marginBottom = "0";

      const row = document.createElement("div");
      row.className = "row";
      row.style.justifyContent = "flex-start";
      row.style.gap = "12px";
      row.style.background = "var(--bg-input)";
      row.style.border = "1px solid var(--border-light)";
      row.style.borderRadius = "10px";
      row.style.padding = "10px 12px";

      const nm = document.createElement("div");
      nm.style.fontWeight = "700";
      nm.textContent = h || `col${idx}`;

      const chk = document.createElement("input");
      chk.type = "checkbox";
      chk.style.flex = "0 0 auto";
      chk.style.marginTop = "2px";
      chk.dataset.colidx = String(idx);
      chk.checked = (prevSel.size === 0) ? true : prevSel.has(String(idx));

      chk.addEventListener("change", () => {
        collectSelectedUI();
        trained = null;
        renderPreviewInputs();
        updateSize();
        updateDeployState();
      });

      row.appendChild(chk);
      row.appendChild(nm);
      wrap.appendChild(row);
      featureCols.appendChild(wrap);
    });

    applyFeatureExclusions();
    collectSelectedUI({ resetClasses: true });
  }

  function collectSelectedUI({ resetClasses = false } = {}) {
    if (!parsed) return;

    // Sync label selections from UI.
    selectedLabel = Number(labelCol?.value ?? selectedLabel ?? 0);
    if (selectedTask === "multilabel_classification") {
      selectedLabelCols = getSelectedLabelColsFromUI();
      updateMultiLabelNoteUI();
    }

    applyFeatureExclusions();
    const exclude = labelExclusionSet();

    const inputs = Array.from(featureCols.querySelectorAll("input[type=checkbox][data-colidx]"));
    const next = [];
    for (const chk of inputs) {
      if (!chk.checked) continue;
      const colIdx = Number(chk.dataset.colidx);
      if (exclude.has(colIdx | 0)) continue;
      next.push(colIdx | 0);
    }
    selectedFeatures = next;

    refreshClassUI({ reset: resetClasses });
    updateDatasetSummary();
    updateSplitUICounts();
    updateSize();
  }

  function updateTaskUI() {
    const isMultilabel = (selectedTask === "multilabel_classification");
    if (singleLabelRow) singleLabelRow.style.display = isMultilabel ? "none" : "block";
    if (multiLabelBox) multiLabelBox.style.display = isMultilabel ? "block" : "none";

    // Keep feature checkboxes aligned with label selection.
    if (isMultilabel) updateMultiLabelNoteUI();
    applyFeatureExclusions();
  }

  // Wire up task/label/class controls
  if (taskSel) {
    taskSel.addEventListener("change", () => {
      selectedTask = String(taskSel.value || "regression");
      trained = null;
      datasetNumeric = null;
      updateTaskUI();
      collectSelectedUI({ resetClasses: true });
      renderPreviewInputs();
      updateDeployState();
    });
  }

  if (labelCol) {
    // Replace any older inline listeners by assigning onchange.
    labelCol.onchange = () => {
      if (!parsed) return;
      rebuildFeatureColsUI();
      trained = null;
      datasetNumeric = null;
      renderPreviewInputs();
      updateSize();
      updateDeployState();
    };
  }

  // Multilabel column selection
  if (multiLabelSel) {
    multiLabelSel.addEventListener("change", () => {
      if (!parsed) return;
      selectedLabelCols = getSelectedLabelColsFromUI();
      updateMultiLabelNoteUI();
      trained = null;
      datasetNumeric = null;
      applyFeatureExclusions();
      collectSelectedUI();
      renderPreviewInputs();
      updateDeployState();
    });
  }
  if (multiLabelAllBtn) {
    multiLabelAllBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (!multiLabelSel) return;
      trained = null;
      datasetNumeric = null;
      for (const opt of Array.from(multiLabelSel.options)) opt.selected = true;
      selectedLabelCols = getSelectedLabelColsFromUI();
      updateMultiLabelNoteUI();
      applyFeatureExclusions();
      collectSelectedUI();
      renderPreviewInputs();
      updateDeployState();
    });
  }
  if (multiLabelClearBtn) {
    multiLabelClearBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (!multiLabelSel) return;
      trained = null;
      datasetNumeric = null;
      for (const opt of Array.from(multiLabelSel.options)) opt.selected = false;
      selectedLabelCols = [];
      updateMultiLabelNoteUI();
      applyFeatureExclusions();
      collectSelectedUI();
      renderPreviewInputs();
      updateDeployState();
    });
  }

  if (negClassSel) {
    negClassSel.addEventListener("change", () => {
      selectedNegLabel = String(negClassSel.value || "");
      // Prevent equal selections
      if (selectedNegLabel && selectedNegLabel === selectedPosLabel) {
        const vals = labelValuesInfo?.values || [];
        selectedPosLabel = vals.find(v => v !== selectedNegLabel) || selectedPosLabel;
        if (posClassSel) posClassSel.value = String(selectedPosLabel || "");
      }
      trained = null;
      datasetNumeric = null;
      updateDatasetSummary();
      renderPreviewInputs();
      updateDeployState();
    });
  }

  if (posClassSel) {
    posClassSel.addEventListener("change", () => {
      selectedPosLabel = String(posClassSel.value || "");
      if (selectedPosLabel && selectedPosLabel === selectedNegLabel) {
        const vals = labelValuesInfo?.values || [];
        selectedNegLabel = vals.find(v => v !== selectedPosLabel) || selectedNegLabel;
        if (negClassSel) negClassSel.value = String(selectedNegLabel || "");
      }
      trained = null;
      datasetNumeric = null;
      updateDatasetSummary();
      renderPreviewInputs();
      updateDeployState();
    });
  }

  if (swapClassesBtn) {
    swapClassesBtn.addEventListener("click", (e) => {
      e.preventDefault();
      const a = selectedNegLabel;
      selectedNegLabel = selectedPosLabel;
      selectedPosLabel = a;
      if (negClassSel) negClassSel.value = String(selectedNegLabel || "");
      if (posClassSel) posClassSel.value = String(selectedPosLabel || "");
      trained = null;
      datasetNumeric = null;
      updateDatasetSummary();
      renderPreviewInputs();
      updateDeployState();
    });
  }

  // Multiclass controls
  function updateMultiSelectionFromUI() {
    if (!multiClassSel) return;
    selectedMultiLabels = Array.from(multiClassSel.options)
      .filter((o) => o.selected)
      .map((o) => String(o.value));
    trained = null;
    datasetNumeric = null;
    updateDatasetSummary();
    renderPreviewInputs();
    updateSize();
    updateDeployState();
  }

  function moveMultiSelected(dir) {
    if (!multiClassSel) return;
    const sel = multiClassSel;
    if (dir < 0) {
      for (let i = 1; i < sel.options.length; i++) {
        const o = sel.options[i];
        const prev = sel.options[i - 1];
        if (o.selected && !prev.selected) {
          sel.insertBefore(o, prev);
        }
      }
    } else {
      for (let i = sel.options.length - 2; i >= 0; i--) {
        const o = sel.options[i];
        const next = sel.options[i + 1];
        if (o.selected && !next.selected) {
          sel.insertBefore(next, o);
        }
      }
    }
    updateMultiSelectionFromUI();
  }

  if (multiClassSel) {
    multiClassSel.addEventListener("change", () => {
      updateMultiSelectionFromUI();
    });
  }

  if (multiClassAllBtn) {
    multiClassAllBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (!multiClassSel) return;
      for (const o of Array.from(multiClassSel.options)) o.selected = true;
      updateMultiSelectionFromUI();
    });
  }
  if (multiClassClearBtn) {
    multiClassClearBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (!multiClassSel) return;
      for (const o of Array.from(multiClassSel.options)) o.selected = false;
      updateMultiSelectionFromUI();
    });
  }
  if (multiClassUpBtn) {
    multiClassUpBtn.addEventListener("click", (e) => {
      e.preventDefault();
      moveMultiSelected(-1);
    });
  }
  if (multiClassDownBtn) {
    multiClassDownBtn.addEventListener("click", (e) => {
      e.preventDefault();
      moveMultiSelected(1);
    });
  }

  csvFile.addEventListener("change", async () => {
    const f = csvFile.files?.[0];
    if (!f) return;
    log(`[${nowTs()}] Parsing CSV: ${f.name} (${f.size} bytes)`);
    // In Python engine mode, avoid loading the full CSV in the browser (keeps huge datasets workable).
    // We only parse a small prefix to get headers + a small sample for UI selectors/plots.
    const prefixBytes = _isPythonEngine() ? Math.min(f.size, 2 * 1024 * 1024) : f.size;
    const text = await f.slice(0, prefixBytes).text();
    parsed = parseCSV(text);

    // In Python engine mode, upload the full dataset once to the localhost cache (async).
    if (_isPythonEngine()) {
      try {
        _setPyDatasetPill("Dataset: uploading…", "warn");
        void _pyUploadDatasetFile(f).catch((e) => {
          _setPyDatasetPill("Dataset: not cached", "bad");
          log(`[${nowTs()}] [warn] Local upload failed (${_localEngineName()}): ${e?.message || e}`);
        });
      } catch (e) {
        log(`[${nowTs()}] [warn] Local upload skipped (${_localEngineName()}): ${e?.message || e}`);
      }
    } else {
      // Browser engine: clear python cache marker (optional)
      pyDatasetId = null;
      pyDatasetName = null;
      _setPyDatasetPill("Dataset: not cached");
    }


    labelCol.innerHTML = "";
    if (multiLabelSel) multiLabelSel.innerHTML = "";
    parsed.headers.forEach((h, idx) => {
      const name = h || `col${idx}`;
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = name;
      labelCol.appendChild(opt);

      if (multiLabelSel) {
        const opt2 = document.createElement("option");
        opt2.value = String(idx);
        opt2.textContent = name;
        multiLabelSel.appendChild(opt2);
      }
    });

    selectedLabel = parsed.headers.length - 1;
    labelCol.value = String(selectedLabel);

    // Reset multilabel selection on new dataset.
    selectedLabelCols = [];
    if (multiLabelSel) {
      for (const opt of Array.from(multiLabelSel.options)) opt.selected = false;
    }
    updateMultiLabelNoteUI();

    trained = null;
    datasetNumeric = null;
    rebuildFeatureColsUI({ preserveSelection: false });
    updateTaskUI();
    renderPreviewInputs();
    updateSize();
    updateDeployState();
  });

  // Training worker
  function stopTraining() {
    // Stop is used both for single training and multi-round heuristic search.
    searchAbort = true;
    isSearching = false;

    // If the current training round is awaited via a Promise, reject it so
    // the outer control flow can unwind cleanly.
    if (typeof _activeTrainReject === "function") {
      try { _activeTrainReject(new Error("Training stopped by user")); } catch {}
      _activeTrainReject = null;
    }

    if (worker) {
      try { worker.terminate(); } catch {}
      worker = null;
    }

    // Python engine stop (abort fetch + ask server to terminate subprocess)
    if (pyTrainAbort) {
      try { pyTrainAbort.abort(); } catch {}
      pyTrainAbort = null;
      try { void _pyStop(); } catch {}
    }


    isTraining = false;
    trainPill.textContent = "Stopped";
    trainBtn.disabled = false;
    stopBtn.disabled = true;
    setDockState("idle");
    log(`[${nowTs()}] Training stopped by user.`);
    updateDeployState();
  }
  stopBtn.addEventListener("click", stopTraining);

  if (pyUploadBtn) {
    pyUploadBtn.addEventListener("click", async () => {
      try {
        const f = csvFile?.files?.[0];
        await _pyUploadDatasetFile(f);
      } catch (e) {
        _setPyDatasetPill("Dataset: not cached", "bad");
        log(`[${nowTs()}] [warn] Upload failed: ${e?.message || e}`);
      }
    });
  }


  function clampFloat(x, lo, hi) {
    if (!Number.isFinite(x)) return lo;
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  function _xorshift32(seed) {
    let x = (seed | 0) || 123456789;
    return () => {
      x ^= x << 13;
      x ^= x >>> 17;
      x ^= x << 5;
      return x >>> 0;
    };
  }

  function _rand01(rng) {
    return (rng() >>> 0) / 4294967296;
  }

  function _maybeDeepCopy(x) {
    if (x == null) return null;
    try { return JSON.parse(JSON.stringify(x)); } catch { return x; }
  }

  function buildLrScheduleFromUI() {
    const mode = String(lrSchedMode?.value || "none");
    let lrSchedule = null;
    if (mode === "plateau") {
      const n = clampInt(parseInt(lrPlateauN?.value || "25", 10), 1, 1000);
      const pct = clampInt(parseInt(lrPlateauPct?.value || "10", 10), 1, 99);
      const minLR = clampFloat(parseFloat(lrPlateauMin?.value || "0"), 0, 1);
      lrSchedule = { mode: "plateau", patience: n, dropPct: pct, minLR };
    } else if (mode === "piecewise") {
      const segs = parsePiecewiseLrSchedule(lrScheduleText?.value || "");
      if (segs.length) lrSchedule = { mode: "piecewise", segments: segs };
    }
    return lrSchedule;
  }

  function lrScheduleToText(sched) {
    if (!sched || typeof sched !== "object") return "";
    if (sched.mode !== "piecewise" || !Array.isArray(sched.segments)) return "";
    return sched.segments.map(s => `${s.start}-${s.end} ${s.lr}`).join("\n");
  }

  function applyParamsToTrainingUI(p) {
    if (!p) return;
    // Core params
    if (Number.isFinite(p.trees)) { treesNum.value = String(p.trees); treesRange.value = String(p.trees); }
    if (Number.isFinite(p.depth)) { depthNum.value = String(p.depth); depthRange.value = String(p.depth); }
    if (Number.isFinite(p.lr)) { lrNum.value = String(p.lr); lrRange.value = String(p.lr); }
    if (Number.isFinite(p.minLeaf)) { minLeafNum.value = String(p.minLeaf); minLeafRange.value = String(p.minLeaf); }
    if (Number.isFinite(p.bins) && binsNum && binsRange) { binsNum.value = String(p.bins); binsRange.value = String(p.bins); }
    if (typeof p.binning === "string" && binningMode) {
      const v = String(p.binning || "");
      if (v) binningMode.value = v;
    }
    if (Number.isFinite(p.seed)) { seedNum.value = String(p.seed); }

    if (typeof p.earlyStop === "boolean") earlyStopOn.checked = p.earlyStop;
    if (Number.isFinite(p.patience)) { patienceNum.value = String(p.patience); patienceRange.value = String(p.patience); }

    // LR schedule (ensure it is applied for each round)
    const sched = p.lrSchedule;
    const mode = (sched && typeof sched === "object") ? String(sched.mode || "none") : "none";
    if (lrSchedMode) lrSchedMode.value = mode;
    updateLrScheduleUI();

    if (mode === "plateau") {
      if (lrPlateauN && Number.isFinite(sched.patience)) lrPlateauN.value = String(sched.patience | 0);
      if (lrPlateauPct && Number.isFinite(sched.dropPct)) lrPlateauPct.value = String(sched.dropPct | 0);
      if (lrPlateauMin && Number.isFinite(sched.minLR)) lrPlateauMin.value = String(sched.minLR);
    } else if (mode === "piecewise") {
      if (lrScheduleText) lrScheduleText.value = lrScheduleToText(sched);
    }
  }

  function readBaseTrainParams({ nClasses }) {
    const trees0 = parseInt(treesNum.value, 10);
    const depth0 = parseInt(depthNum.value, 10);
    const lr = clampFloat(parseFloat(lrNum.value), parseFloat(lrNum.min || "0.001"), parseFloat(lrNum.max || "1"));
    const minLeaf = clampInt(parseInt(minLeafNum.value, 10), parseInt(minLeafNum.min || "1", 10), parseInt(minLeafNum.max || "1000", 10));
    const bins = clampInt(parseInt(binsNum?.value || "32", 10), parseInt(binsNum?.min || "8", 10), parseInt(binsNum?.max || "512", 10));
    const binning = String(binningMode?.value || "linear");
    const seed = clampInt(parseInt(seedNum.value, 10), 1, 2147483647);
    const earlyStop = !!earlyStopOn.checked;
    const patience = clampInt(parseInt(patienceNum.value, 10), parseInt(patienceNum.min || "1", 10), parseInt(patienceNum.max || "500", 10));

    const splitTrain = clampInt(parseInt(trainSplitNum?.value || "70", 10), 50, 90) / 100;
    const splitVal = clampInt(parseInt(valSplitNum?.value || "20", 10), 5, 40) / 100;

    const lrSchedule = buildLrScheduleFromUI();

    const cl = clampForSize(trees0, depth0, selectedTask, nClasses);
    // Keep UI in sync with size clamp.
    if (cl.trees !== trees0) { treesNum.value = String(cl.trees); treesRange.value = String(cl.trees); }
    if (cl.depth !== depth0) { depthNum.value = String(cl.depth); depthRange.value = String(cl.depth); }

    return {
      task: selectedTask,
      trees: cl.trees,
      depth: cl.depth,
      lr,
      lrSchedule,
      minLeaf,
      bins,
      binning,
      seed,
      earlyStop,
      patience,
      splitTrain,
      splitVal,
      nClasses,
      imbalance: readImbalanceConfigForParams({ nClasses })
    };
  }

  function generateHeuristicCandidate({ baseParams, bestParams, round, rng }) {
    const pivot = (bestParams && _rand01(rng) < 0.75) ? bestParams : baseParams;
    const p = { ...pivot };

    const treesMin = parseInt(treesNum.min || "10", 10);
    const treesMax = parseInt(treesNum.max || "5000", 10);
    const depthMin = parseInt(depthNum.min || "1", 10);
    const depthMax = parseInt(depthNum.max || "12", 10);
    const lrMin = parseFloat(lrNum.min || "0.001");
    const lrMax = parseFloat(lrNum.max || "1");
    const minLeafMin = parseInt(minLeafNum.min || "1", 10);
    const minLeafMax = parseInt(minLeafNum.max || "1000", 10);
    const patMin = parseInt(patienceNum.min || "1", 10);
    const patMax = parseInt(patienceNum.max || "500", 10);

    // --- trees ---
    // Multiplicative exploration around pivot.
    const treesFactor = Math.pow(2, (_rand01(rng) - 0.5) * 1.4); // ~[0.62..1.62]
    let trees = Math.round((Number(pivot.trees) * treesFactor) / 25) * 25;
    trees = clampInt(trees, treesMin, treesMax);

    // --- depth ---
    const dStep = Math.round((_rand01(rng) - 0.5) * 4); // [-2..2]
    let depth = clampInt((pivot.depth | 0) + dStep, depthMin, depthMax);

    // --- lr ---
    const lrFactor = Math.pow(10, (_rand01(rng) - 0.5) * 0.8); // ~[0.40..2.51]
    let lr = clampFloat(Number(pivot.lr) * lrFactor, lrMin, lrMax);
    lr = Math.round(lr * 1e6) / 1e6;

    // --- minLeaf ---
    const mlFactor = Math.pow(2, (_rand01(rng) - 0.5) * 2.0); // ~[0.5..2]
    let minLeaf = clampInt(Math.round(Number(pivot.minLeaf) * mlFactor), minLeafMin, minLeafMax);

    // --- patience ---
    let patience = pivot.patience | 0;
    if (pivot.earlyStop) {
      const patFactor = Math.pow(2, (_rand01(rng) - 0.5) * 1.6); // ~[0.57..1.74]
      patience = clampInt(Math.round((pivot.patience | 0) * patFactor / 5) * 5, patMin, patMax);
    }

    // LR schedule: always present in params (can be null), and we optionally
    // perturb plateau settings if plateau mode is active.
    let lrSchedule = _maybeDeepCopy(pivot.lrSchedule);
    if (lrSchedule && lrSchedule.mode === "plateau") {
      const n0 = lrSchedule.patience | 0;
      const pct0 = lrSchedule.dropPct | 0;
      const minLR0 = Number(lrSchedule.minLR || 0);
      const nFactor = Math.pow(2, (_rand01(rng) - 0.5) * 1.2);
      lrSchedule.patience = clampInt(Math.round(n0 * nFactor), 1, 1000);
      const pctStep = Math.round((_rand01(rng) - 0.5) * 20);
      lrSchedule.dropPct = clampInt(pct0 + pctStep, 1, 99);
      const minLRFactor = Math.pow(10, (_rand01(rng) - 0.5) * 1.2);
      lrSchedule.minLR = clampFloat(minLR0 * minLRFactor, 0, 1);
    }

    // Never vary the split fractions during search (keeps scores comparable).
    p.trees = trees;
    p.depth = depth;
    p.lr = lr;
    p.minLeaf = minLeaf;
    p.bins = baseParams.bins;
    p.binning = baseParams.binning;
    p.patience = patience;
    p.lrSchedule = lrSchedule;
    p.task = baseParams.task;
    p.seed = baseParams.seed;
    p.earlyStop = baseParams.earlyStop;
    p.splitTrain = baseParams.splitTrain;
    p.splitVal = baseParams.splitVal;
    p.nClasses = baseParams.nClasses;
    p.imbalance = baseParams.imbalance;

    const cl = clampForSize(p.trees, p.depth, p.task, p.nClasses);
    p.trees = cl.trees;
    p.depth = cl.depth;
    return p;
  }

  async function runTrainRound({ XMaster, yMaster, nRows, nFeat, scaleQ, params, round, totalRounds, label = null }) {
    if (_isPythonEngine()) {
      // Ensure UI shows the current candidate params (parity with Worker mode).
      applyParamsToTrainingUI(params);
      try { updateSize(); } catch {}

      // For Python/C++ engines the server returns the full curve only at the end (no streaming).
      // Best practice for heuristic search: keep the previous curve visible while the next round runs,
      // otherwise fast rounds can briefly flash an empty chart and look like it never updated.
      const isSearch = (totalRounds > 1);
      if (!isSearch || round === 1) {
        const localCurve = { steps: [], train: [], val: [], test: [], bestVal: [] };
        curve.steps = localCurve.steps;
        curve.train = localCurve.train;
        curve.val = localCurve.val;
        curve.test = localCurve.test;
        curve.bestVal = localCurve.bestVal;
        _curveRev++;
        drawCurves();
      }

      // Keep the feature-importance box from showing stale results during any training.
      _featImpNonce += 1;
      try { _fiClear("Training… feature importance will appear here when done."); } catch {}

      // Kill any previous worker (if user switched engines mid-session).
      try { worker?.terminate(); } catch {}
      worker = null;
      _activeTrainReject = null;

      const labelPrefix = label || ((totalRounds > 1) ? `Search ${round}/${totalRounds}` : "Training");
      const res = await runTrainRoundPython({
        params, round, totalRounds, labelPrefix,
        parsed,
        selectedFeatures,
        selectedTask,
        selectedLabel,
        selectedLabelCols,
        selectedNegLabel,
        selectedPosLabel,
        selectedMultiLabels,
        curve,
        trainPill,
        trainBar,
        setDockState,
        nowTs,
        log
      });

      // Render curve + final metrics (Python/C++ engines return them only at end).
      try {
        _curveRev++;
        drawCurves();
        if (totalRounds > 1 && typeof requestAnimationFrame === "function") {
          await new Promise((r) => requestAnimationFrame(() => r()));
        }
      } catch {}

      try {
        const m = res?.meta || {};
        const metricName = m.metricName || ((m.task || selectedTask) === "regression" ? "MSE" : "LogLoss");
        const isClass = _isClassTask(String(m.task || selectedTask));

        const entries = [
          ["Round", (totalRounds > 1) ? `${round}/${totalRounds}` : "1/1"],
          ["Metric", String(metricName)],
          ["Train", Number.isFinite(m.bestTrainMetric) ? Number(m.bestTrainMetric).toFixed(6) : "—"],
          ["Val", Number.isFinite(m.bestValMetric) ? Number(m.bestValMetric).toFixed(6) : "—"],
          ["Test", Number.isFinite(m.bestTestMetric) ? Number(m.bestTestMetric).toFixed(6) : "—"],
          ["Best iter", Number.isFinite(m.bestIter) ? String(m.bestIter) : "—"],
        ];
        if (isClass) {
          const vAcc = (m.bestValAcc != null) ? m.bestValAcc : (m.valAcc != null ? m.valAcc : m.bestValAccuracy);
          const tAcc = (m.bestTestAcc != null) ? m.bestTestAcc : (m.testAcc != null ? m.testAcc : m.bestTestAccuracy);
          entries.push(["Val Acc", Number.isFinite(vAcc) ? (Number(vAcc) * 100).toFixed(2) + "%" : "—"]);
          entries.push(["Test Acc", Number.isFinite(tAcc) ? (Number(tAcc) * 100).toFixed(2) + "%" : "—"]);
        }
        setKV(metricsKV, entries);
      } catch {}

      return res;
    }


    // Ensure UI shows the current candidate params.
    applyParamsToTrainingUI(params);
    try { updateSize(); } catch {}

    // New curve buffers for this round (so we can keep them for the search history).
    const localCurve = { steps: [], train: [], val: [], test: [], bestVal: [] };
    curve.steps = localCurve.steps;
    curve.train = localCurve.train;
    curve.val = localCurve.val;
    curve.test = localCurve.test;
    curve.bestVal = localCurve.bestVal;
    _curveRev++;
    drawCurves();

    // Kill any previous worker.
    try { worker?.terminate(); } catch {}
    worker = null;

    // Copy buffers for this round; they will be transferred to the Worker.
    const X = new Float32Array(XMaster);
    const y = new Float32Array(yMaster);

    const labelPrefix = label || ((totalRounds > 1) ? `Search ${round}/${totalRounds}` : "Training");
    trainPill.textContent = `${labelPrefix}…`;
    setDockState("training");

    // Keep the feature-importance box from showing stale results during any training.
    _featImpNonce += 1;
    try { _fiClear("Training… feature importance will appear here when done."); } catch {}

    return await new Promise((resolve, reject) => {
      _activeTrainReject = reject;

      let w = null;
      try {
        w = new Worker("src/train_worker.js", { type: "module" });
      } catch (e) {
        _activeTrainReject = null;
        throw e;
      }
      worker = w;

      w.onerror = (e) => {
        try { w.terminate(); } catch {}
        worker = null;
        _activeTrainReject = null;
        reject(new Error(e?.message || "Worker error"));
      };
      w.onmessageerror = (e) => {
        try { w.terminate(); } catch {}
        worker = null;
        _activeTrainReject = null;
        reject(new Error(e?.message || "Worker message error"));
      };

      w.onmessage = (ev) => {
        const msg = ev.data;
        if (!msg || typeof msg.type !== "string") return;

        if (msg.type === "progress") {
          const inRound = (msg.total > 0) ? (msg.done / msg.total) : 0;
          const overall = (totalRounds > 1)
            ? ((round - 1 + inRound) / totalRounds)
            : inRound;
          trainBar.style.width = `${Math.max(0, Math.min(100, Math.floor(overall * 100)))}%`;
          trainPill.textContent = `${labelPrefix}… ${msg.done}/${msg.total}`;

          const metricName = msg.metricName || msg.metric || (msg.task === "regression" ? "MSE" : "LogLoss");
          const isClass = _isClassTask(String(msg.task || selectedTask));
          const entries = [
            ["Round", (totalRounds > 1) ? `${round}/${totalRounds}` : "1/1"],
            ["Metric", String(metricName)],
            ["Train", Number.isFinite(msg.trainMetric) ? msg.trainMetric.toFixed(6) : "—"],
            ["Val", Number.isFinite(msg.valMetric) ? msg.valMetric.toFixed(6) : "—"],
            ["Test", Number.isFinite(msg.testMetric) ? msg.testMetric.toFixed(6) : "—"],
            ["Best val", Number.isFinite(msg.bestValMetric) ? msg.bestValMetric.toFixed(6) : "—"],
            ["Best iter", Number.isFinite(msg.bestIter) ? String(msg.bestIter) : "—"],
          ];
          if (isClass) {
            entries.push(["Val Acc", Number.isFinite(msg.valAcc) ? (msg.valAcc * 100).toFixed(2) + "%" : "—"]);
            entries.push(["Test Acc", Number.isFinite(msg.testAcc) ? (msg.testAcc * 100).toFixed(2) + "%" : "—"]);
          }
          if (Number.isFinite(msg.lr)) {
            entries.push(["LR (current)", String(msg.lr)]);
          }
          setKV(metricsKV, entries);

          // Curve (avoid duplicates)
          const lastT = curve.steps.length ? curve.steps[curve.steps.length - 1] : -1;
          if (msg.done !== lastT) {
            curve.steps.push(msg.done);
            curve.train.push(Number(msg.trainMetric));
            curve.val.push(Number(msg.valMetric));
            curve.test.push(Number(msg.testMetric));
            curve.bestVal.push(Number.isFinite(msg.bestValMetric) ? Number(msg.bestValMetric) : NaN);
            _curveRev++;
            drawCurves();
          }
          return;
        }

        if (msg.type === "log") {
          const prefix = (totalRounds > 1) ? `[${round}/${totalRounds}] ` : "";
          log(`[${nowTs()}] ${prefix}${msg.line || ""}`);
          return;
        }

        if (msg.type === "done") {
          // Detach from active reject before resolving.
          _activeTrainReject = null;
          try { w.terminate(); } catch {}
          worker = null;
          trainPill.textContent = label ? `${labelPrefix} done` : ((totalRounds > 1) ? `${labelPrefix} done` : "Done");
          resolve({ bytes: new Uint8Array(msg.modelBytes), meta: msg.meta, curve: localCurve, params: _maybeDeepCopy(params) });
          return;
        }

        if (msg.type === "error") {
          _activeTrainReject = null;
          try { w.terminate(); } catch {}
          worker = null;
          reject(new Error(msg.message || "Training failed"));
          return;
        }
      };

      // Post params and data.
      w.postMessage({
        type: "train",
        X: X.buffer,
        y: y.buffer,
        nRows,
        nFeatures: nFeat,
        featureNames: datasetNumeric?.featureNames || [],
        params: { ...params, scaleQ }
      }, [X.buffer, y.buffer]);
    });
  }

  function _previewTreeKey(task) {
    const t = String(task || "");
    if (t === "multiclass_classification") return "Boosting rounds (trees/class)";
    if (t === "multilabel_classification") return "Boosting rounds (trees/label)";
    return "Boosting rounds (trees)";
  }

  function _fmtSplit(train, val) {
    if (!Number.isFinite(train) || !Number.isFinite(val)) return "—";
    const tr = Math.round(Number(train) * 100);
    const va = Math.round(Number(val) * 100);
    const te = Math.max(0, 100 - tr - va);
    return `${tr}/${va}/${te}`;
  }

  function _imbSummary(imb) {
    if (!imb || typeof imb !== "object") return "none";
    const mode = String(imb.mode || "none");
    if (mode === "none") return "none";
    const parts = [mode];
    if (Number.isFinite(imb.cap)) parts.push(`cap=${Number(imb.cap)}`);
    if (typeof imb.normalize === "boolean") parts.push(imb.normalize ? "normalize" : "no-normalize");
    if (typeof imb.stratify === "boolean" && imb.stratify) parts.push("stratified");

    if (mode === "manual") {
      if (Number.isFinite(imb.w0) || Number.isFinite(imb.w1)) {
        parts.push(`w0=${Number(imb.w0 ?? 1)}`);
        parts.push(`w1=${Number(imb.w1 ?? 1)}`);
      } else if (Array.isArray(imb.classWeights)) {
        parts.push(`classWeights=${imb.classWeights.length}`);
      } else if (Array.isArray(imb.posWeights)) {
        parts.push(`posWeights=${imb.posWeights.length}`);
      }
    }

    return parts.join(", ");
  }

  function renderPreviewBestModelBlocks() {
    // Best model blocks live on the Local preview tab.
    if (!previewMetricsKV && !previewParamsKV) return;

    if (!trained?.decoded || !trained?.meta) {
      if (previewMetricsKV) setKV(previewMetricsKV, [["Status", "Train a model to see metrics."]]);
      if (previewParamsKV) setKV(previewParamsKV, [["Status", "Train a model to see hyperparameters."]]);
      try { if (previewParamsNote) previewParamsNote.style.display = "none"; } catch {}
      return;
    }

    // --- Metrics (same as Training tab) ---
    const meta = trained.meta;
    const bytes = trained.bytes;
    const modelId = trained.modelId;

    const task = meta?.task || selectedTask;
    const metricName = meta?.metricName || (task === "regression" ? "MSE" : "LogLoss");
    const isClass = _isClassTask(task);

    const mEntries = [];
    mEntries.push(["Task", taskLabel(task)]);
    mEntries.push(["Metric", metricName]);
    if (Number.isFinite(meta?.bestTrainMetric)) mEntries.push(["Best train", meta.bestTrainMetric.toFixed(6)]);
    if (Number.isFinite(meta?.bestValMetric)) mEntries.push(["Best val", meta.bestValMetric.toFixed(6)]);
    if (Number.isFinite(meta?.bestTestMetric)) mEntries.push(["Best test", meta.bestTestMetric.toFixed(6)]);
    if (isClass) {
      if (Number.isFinite(meta?.bestTrainAcc)) mEntries.push(["Best train Acc", (meta.bestTrainAcc * 100).toFixed(2) + "%"]);
      if (Number.isFinite(meta?.bestValAcc)) mEntries.push(["Best val Acc", (meta.bestValAcc * 100).toFixed(2) + "%"]);
      if (Number.isFinite(meta?.bestTestAcc)) mEntries.push(["Best test Acc", (meta.bestTestAcc * 100).toFixed(2) + "%"]);
    }
    if (Number.isFinite(meta?.usedTrees)) mEntries.push(["Used trees", String(meta.usedTrees)]);
    if (Number.isFinite(meta?.totalTrees)) mEntries.push(["Total trees", String(meta.totalTrees)]);
    if (Number.isFinite(meta?.depth)) mEntries.push(["Depth", String(meta.depth)]);
    if (Number.isFinite(meta?.bins)) mEntries.push(["Bins", String(meta.bins)]);
    if (meta?.binning) mEntries.push(["Binning", String(meta.binning)]);
    mEntries.push(["Model bytes", bytes.length.toLocaleString()]);
    mEntries.push(["Model ID", modelId]);

    if (previewMetricsKV) setKV(previewMetricsKV, mEntries);

    // --- Hyperparameters used ---
    const p = trained.params || {};
    const hEntries = [];

    // Core
    const treeKey = _previewTreeKey(task);
    if (task === "multiclass_classification") {
      if (Number.isFinite(meta?.nClasses)) hEntries.push(["Classes", String(meta.nClasses)]);
    } else if (task === "multilabel_classification") {
      if (Number.isFinite(meta?.nClasses)) hEntries.push(["Labels", String(meta.nClasses)]);
    }
    if (Number.isFinite(p.trees)) hEntries.push([treeKey, String(p.trees)]);
    else if (Number.isFinite(meta?.maxTrees)) hEntries.push([treeKey, String(meta.maxTrees)]);

    if (Number.isFinite(p.depth)) hEntries.push(["Depth", String(p.depth)]);
    if (Number.isFinite(p.lr)) hEntries.push(["Learning rate", String(p.lr)]);
    hEntries.push(["LR schedule", _lrScheduleSummary(p.lrSchedule)]);

    if (Number.isFinite(p.minLeaf)) hEntries.push(["Min leaf", String(p.minLeaf)]);
    if (Number.isFinite(p.bins)) hEntries.push(["Bins", String(p.bins)]);
    if (p.binning) hEntries.push(["Binning", String(p.binning)]);

    // Train control
    if (typeof p.earlyStop === "boolean") hEntries.push(["Early stopping", p.earlyStop ? "on" : "off"]);
    if (Number.isFinite(p.patience)) hEntries.push(["Patience (rounds)", String(p.patience)]);

    if (Number.isFinite(p.seed)) hEntries.push(["Seed", String(p.seed)]);
    if (Number.isFinite(p.splitTrain) && Number.isFinite(p.splitVal)) {
      hEntries.push(["Split (Train/Val/Test)", _fmtSplit(p.splitTrain, p.splitVal)]);
    }

    hEntries.push(["Refit on Train+Val", p.refitTrainVal ? "yes" : "no"]);

    // Class imbalance
    if (_isClassTask(task)) {
      hEntries.push(["Imbalance", _imbSummary(p.imbalance)]);
    }

    // Quantization (useful for on-chain)
    if (Number.isFinite(meta?.scaleQ)) hEntries.push(["scaleQ", String(meta.scaleQ)]);

    if (previewParamsKV) setKV(previewParamsKV, hEntries.length ? hEntries : [["Hyperparameters", "—"]]);

    // Cosmetic: hide note if no params were captured.
    try {
      if (previewParamsNote) previewParamsNote.style.display = trained.params ? "" : "none";
    } catch {}
  }

  function applyTrainedModel({ bytes, meta, params }) {
    const modelId = ethers.keccak256(bytes);
    const decoded = decodeModel(bytes);
    trained = { bytes, modelId, decoded, meta, params: _maybeDeepCopy(params) };

    const task = meta?.task || selectedTask;
    const metricName = meta?.metricName || (task === "regression" ? "MSE" : "LogLoss");
    const isClass = _isClassTask(task);

    const entries = [];
    entries.push(["Task", taskLabel(task)]);
    entries.push(["Metric", metricName]);
    if (Number.isFinite(meta?.bestTrainMetric)) entries.push(["Best train", meta.bestTrainMetric.toFixed(6)]);
    if (Number.isFinite(meta?.bestValMetric)) entries.push(["Best val", meta.bestValMetric.toFixed(6)]);
    if (Number.isFinite(meta?.bestTestMetric)) entries.push(["Best test", meta.bestTestMetric.toFixed(6)]);
    if (isClass) {
      if (Number.isFinite(meta?.bestTrainAcc)) entries.push(["Best train Acc", (meta.bestTrainAcc * 100).toFixed(2) + "%"]);
      if (Number.isFinite(meta?.bestValAcc)) entries.push(["Best val Acc", (meta.bestValAcc * 100).toFixed(2) + "%"]);
      if (Number.isFinite(meta?.bestTestAcc)) entries.push(["Best test Acc", (meta.bestTestAcc * 100).toFixed(2) + "%"]);
    }
    if (Number.isFinite(meta?.usedTrees)) entries.push(["Used trees", String(meta.usedTrees)]);
    if (Number.isFinite(meta?.totalTrees)) entries.push(["Total trees", String(meta.totalTrees)]);
    if (Number.isFinite(meta?.depth)) entries.push(["Depth", String(meta.depth)]);
    if (Number.isFinite(meta?.bins)) entries.push(["Bins", String(meta.bins)]);
    if (meta?.binning) entries.push(["Binning", String(meta.binning)]);
    entries.push(["Model bytes", bytes.length.toLocaleString()]);
    entries.push(["Model ID", modelId]);
    setKV(metricsKV, entries);

    // Keep Local preview's "Best model" blocks in sync immediately.
    try { renderPreviewBestModelBlocks(); } catch {}
  }

  trainBtn.addEventListener("click", async () => {
    try {
      if (isTraining) {
        log(`[${nowTs()}] Training already running. Press Stop to abort.`);
        return;
      }
      if (!parsed) throw new Error("Upload a CSV first");
      if (!selectedFeatures.length) throw new Error("Select at least 1 feature");

      // Validate label selection
      if (selectedTask === "multilabel_classification") {
        selectedLabelCols = getSelectedLabelColsFromUI();
        if (!Array.isArray(selectedLabelCols) || selectedLabelCols.length < 2) {
          throw new Error("Select at least 2 label columns for multilabel classification");
        }
      } else {
        if (selectedLabel === null) throw new Error("Select a label");
      }


      // ===== Python engine: train via localhost server (dataset cached) =====
      if (_isPythonEngine()) {

        // Build an in-browser SAMPLE dataset for Preview/Compare/plots, while training uses the full cached dataset.
        // We parse only a small CSV prefix (see csvFile.onchange) so this stays fast even for huge datasets.
        const headers = parsed.headers || [];
        const parsedUI = _sliceParsedRowsForUI(parsed, PY_UI_SAMPLE_ROWS);

        let dsUI = null;
        try {
          if (selectedTask === "binary_classification") {
            dsUI = toBinaryMatrix(parsedUI, {
              labelIndex: selectedLabel,
              featureIndices: selectedFeatures,
              negLabel: selectedNegLabel,
              posLabel: selectedPosLabel
            });
          } else if (selectedTask === "multiclass_classification") {
            dsUI = toMulticlassMatrix(parsedUI, {
              labelIndex: selectedLabel,
              featureIndices: selectedFeatures,
              classLabels: selectedMultiLabels
            });
          } else if (selectedTask === "multilabel_classification") {
            dsUI = toMultilabelMatrix(parsedUI, {
              labelIndices: selectedLabelCols,
              featureIndices: selectedFeatures
            });
          } else {
            dsUI = toNumericMatrix(parsedUI, { labelIndex: selectedLabel, featureIndices: selectedFeatures });
          }
        } catch (e) {
          dsUI = null;
          log(`[${nowTs()}] [warn] Could not build in-browser sample dataset: ${e?.message || e}`);
        }

        // Always keep feature/label names consistent with the selector UI, even if sample parsing failed.
        const featNames = (selectedFeatures || []).map((i) => headers[i] || `col${i}`);

        if (!dsUI) {
          dsUI = {
            featureNames: featNames,
            X: [],
            y: [],
            droppedRows: 0,
            labelName: (selectedTask === "multilabel_classification") ? "" : (headers[selectedLabel] || "label"),
            labelNames: (selectedTask === "multilabel_classification")
              ? (Array.isArray(selectedLabelCols) ? selectedLabelCols.map((i,k)=>headers[i]||`label${k}`) : [])
              : null,
            classes: (selectedTask === "binary_classification")
              ? { 0: String(selectedNegLabel||"0"), 1: String(selectedPosLabel||"1") }
              : (selectedTask === "multiclass_classification" ? (Array.isArray(selectedMultiLabels) ? selectedMultiLabels.slice() : []) : null)
          };
        }

        // Ensure names/order match exactly what will be used for training.
        dsUI.featureNames = featNames.slice();
        dsUI._sample = true;
        dsUI._sampleLimit = PY_UI_SAMPLE_ROWS;
        datasetNumeric = dsUI;

        // Dataset summary: show sample rows (preview) + note training uses full cached file.
        const dsEntries = [
          ["Task", taskLabel(selectedTask)],
          ["Rows (sample)", datasetNumeric?.X?.length ? String(datasetNumeric.X.length) : "—"],
          ["Dropped rows (sample)", Number.isFinite(datasetNumeric?.droppedRows) ? String(datasetNumeric.droppedRows) : "—"],
          ["Features", String(featNames.length)],
          ["Engine", `${_localEngineName()} (training uses cached full file)`],
        ];
        if (selectedTask === "binary_classification" && datasetNumeric?.classes) {
          dsEntries.push(["Class mapping", `0=${datasetNumeric.classes[0]}, 1=${datasetNumeric.classes[1]}`]);
          dsEntries.push(["Dropped other labels (sample)", String(datasetNumeric.droppedOtherLabel || 0)]);
        } else if (selectedTask === "multiclass_classification") {
          const nC = Array.isArray(selectedMultiLabels) ? selectedMultiLabels.length : (Array.isArray(datasetNumeric?.classes) ? datasetNumeric.classes.length : 0);
          dsEntries.push(["Classes", String(Math.max(2, nC || 0))]);
          dsEntries.push(["Dropped other labels (sample)", String(datasetNumeric.droppedOtherLabel || 0)]);
        } else if (selectedTask === "multilabel_classification") {
          const nL = Array.isArray(selectedLabelCols) ? selectedLabelCols.length : (Array.isArray(datasetNumeric?.labelNames) ? datasetNumeric.labelNames.length : 0);
          dsEntries.push(["Labels", String(Math.max(2, nL || 0))]);
          const miss = Number(datasetNumeric.droppedLabelMissing || 0);
          const bad = Number(datasetNumeric.droppedLabelInvalid || 0);
          const badF = Number(datasetNumeric.droppedBadFeature || 0);
          if (miss) dsEntries.push(["Dropped missing labels (sample)", String(miss)]);
          if (bad) dsEntries.push(["Dropped invalid labels (sample)", String(bad)]);
          if (badF) dsEntries.push(["Dropped bad features (sample)", String(badF)]);
        } else {
          dsEntries.push(["Label", headers[selectedLabel] || "label"]);
        }
        setKV(dsKV, dsEntries);


        // Determine nClasses for size clamp.
        let nClasses = 1;
        if (selectedTask === "binary_classification") nClasses = 2;
        else if (selectedTask === "multiclass_classification") nClasses = Math.max(2, Array.isArray(selectedMultiLabels) ? selectedMultiLabels.length : 2);
        else if (selectedTask === "multilabel_classification") nClasses = Math.max(2, Array.isArray(selectedLabelCols) ? selectedLabelCols.length : 2);

        // Read base params (UI) + enforce size constraints.
        const baseParams = readBaseTrainParams({ nClasses });
        const doSearch = !!heuristicSearchOn?.checked;
        const doRefit = !!refitOn?.checked;

        const maxRounds = doSearch
          ? clampInt(parseInt(heuristicSearchRounds?.value || "10", 10), 1, 1000)
          : 1;

        // Clear previous trained model while running (prevents deploying stale output).
        trained = null;
        renderPreviewInputs();
        updateDeployState();

        isTraining = true;
        isSearching = doSearch;
        searchAbort = false;
        trainBtn.disabled = true;
        stopBtn.disabled = false;
        trainBar.style.width = "0%";
        trainPill.textContent = doSearch ? `Search 1/${maxRounds}…` : "Training…";
        setDockState("training");

        lastTrainInfo = {
          task: selectedTask,
          seed: baseParams.seed,
          splitTrain: baseParams.splitTrain,
          splitVal: baseParams.splitVal,
          nRows: NaN,
          nFeatures: featNames.length,
          nClasses
        };

        // Search loop (unchanged logic: only backend differs)
        if (doSearch) {
          searchHistory = [];
          searchPage = 1;
          renderSearchTable();

          const rng = _xorshift32(baseParams.seed ^ 0x9e3779b9);
          let best = null;
          let bestScore = Infinity;
          let bestRound = 0;

          for (let round = 1; round <= maxRounds; round++) {
            if (searchAbort) break;

            const params = (round === 1)
              ? { ...baseParams }
              : generateHeuristicCandidate({ baseParams, bestParams: best?.params || null, round, rng });

            const entry = { round, status: "running", params: _maybeDeepCopy(params), meta: null, curve: null, error: null };
            // Keep the view on the latest page while the search grows (but don't yank the user
            // off an older page if they've intentionally paged back).
            const pagesBefore = Math.max(1, Math.ceil((searchHistory?.length || 0) / SEARCH_PAGE_SIZE));
            searchHistory.push(entry);
            const pagesAfter = Math.max(1, Math.ceil((searchHistory?.length || 0) / SEARCH_PAGE_SIZE));
            if ((searchPage | 0) >= pagesBefore) searchPage = pagesAfter;
            renderSearchTable();

            try {
              const res = await runTrainRound({ XMaster: null, yMaster: null, nRows: 0, nFeat: featNames.length, scaleQ: "auto", params, round, totalRounds: maxRounds });
              entry.status = "done";
              entry.meta = res.meta;
              entry.curve = res.curve;

              const score = res?.meta?.bestValMetric;
              if (Number.isFinite(score) && score < bestScore) {
                bestScore = score;
                bestRound = round;
                best = { ...res, params };
              }
            } catch (err) {
              if (searchAbort) {
                entry.status = "stopped";
                entry.error = "Stopped";
                renderSearchTable();
                break;
              }
              entry.status = "error";
              entry.error = err?.message || String(err);
            }
            renderSearchTable();
          }

          if (!best || !best.bytes || !best.meta) throw new Error(searchAbort ? "Training stopped" : "No successful search runs");

          curve.steps = best.curve.steps;
          curve.train = best.curve.train;
          curve.val = best.curve.val;
          curve.test = best.curve.test;
          curve.bestVal = best.curve.bestVal || [];
          drawCurves();

          applyParamsToTrainingUI(best.params);
          try { updateSize(); } catch {}

          // Optional refit: python backend supports refitTrainVal param (same as worker).
          let finalRes = best;
          if (doRefit && !searchAbort) {
            const usedTrees = Number(best?.meta?.usedTrees ?? best?.params?.trees ?? baseParams.trees);
            const refitParams = { ...best.params, trees: usedTrees, earlyStop: false, refitTrainVal: true };
            log(`[${nowTs()}] Refit enabled: training on Train+Val for ${usedTrees} trees (size unchanged).`);
            const refitRes = await runTrainRound({ XMaster: null, yMaster: null, nRows: 0, nFeat: featNames.length, scaleQ: "auto", params: refitParams, round: 1, totalRounds: 1, label: "Refit" });
            finalRes = { ...refitRes, params: refitParams };
            applyParamsToTrainingUI(refitParams);
            try { updateSize(); } catch {}
          }

          applyTrainedModel(finalRes);

          trainBar.style.width = "100%";
          trainPill.textContent = searchAbort ? `Stopped (best round ${bestRound}/${maxRounds})` : `Done (best round ${bestRound}/${maxRounds}${(finalRes !== best) ? " + refit" : ""})`;
          if (searchAbort) log(`[${nowTs()}] Search stopped by user. Best round=${bestRound}/${maxRounds} bestVal=${bestScore.toFixed(6)}`);
          else log(`[${nowTs()}] Search complete. Best round=${bestRound}/${maxRounds} bestVal=${bestScore.toFixed(6)}${(finalRes !== best) ? " (refit applied)" : ""}`);

          renderPreviewInputs();
          updateDeployState();

        } else {
          const res = await runTrainRound({ XMaster: null, yMaster: null, nRows: 0, nFeat: featNames.length, scaleQ: "auto", params: baseParams, round: 1, totalRounds: 1 });

          let finalRes = res;
          if (doRefit) {
            const usedTrees = Number(res?.meta?.usedTrees ?? baseParams.trees);
            const refitParams = { ...baseParams, trees: usedTrees, earlyStop: false, refitTrainVal: true };
            log(`[${nowTs()}] Refit enabled: training on Train+Val for ${usedTrees} trees (size unchanged).`);
            const refitRes = await runTrainRound({ XMaster: null, yMaster: null, nRows: 0, nFeat: featNames.length, scaleQ: "auto", params: refitParams, round: 1, totalRounds: 1, label: "Refit" });
            finalRes = { ...refitRes, params: refitParams };
            applyParamsToTrainingUI(refitParams);
            try { updateSize(); } catch {}
          }

          applyTrainedModel(finalRes);
          trainBar.style.width = "100%";
          trainPill.textContent = (finalRes !== res) ? "Done (refit)" : "Done";
          log(`[${nowTs()}] Model ready (${_localEngineName()}). task=${selectedTask} modelId=${trained.modelId} bytes=${trained.bytes.length} features=${trained.decoded.nFeatures} trees=${trained.decoded.nTrees}` + ((finalRes !== res) ? " (refit applied)" : ""));

          renderPreviewInputs();
          updateDeployState();
        }

        isTraining = false;
        isSearching = false;
        stopBtn.disabled = true;
        trainBtn.disabled = false;
        setDockState("idle");
        return;
      }

      // Build numeric dataset
      if (selectedTask === "binary_classification") {
        if (!selectedNegLabel || !selectedPosLabel) {
          throw new Error("Select the two classes (negative/positive) for binary classification");
        }
        datasetNumeric = toBinaryMatrix(parsed, {
          labelIndex: selectedLabel,
          featureIndices: selectedFeatures,
          negLabel: selectedNegLabel,
          posLabel: selectedPosLabel
        });
      } else if (selectedTask === "multiclass_classification") {
        if (!Array.isArray(selectedMultiLabels) || selectedMultiLabels.length < 2) {
          throw new Error("Select at least 2 classes for multiclass classification");
        }
        datasetNumeric = toMulticlassMatrix(parsed, {
          labelIndex: selectedLabel,
          featureIndices: selectedFeatures,
          classLabels: selectedMultiLabels
        });
      } else if (selectedTask === "multilabel_classification") {
        if (!Array.isArray(selectedLabelCols) || selectedLabelCols.length < 2) {
          throw new Error("Select at least 2 label columns for multilabel classification");
        }
        datasetNumeric = toMultilabelMatrix(parsed, {
          labelIndices: selectedLabelCols,
          featureIndices: selectedFeatures
        });
      } else {
        datasetNumeric = toNumericMatrix(parsed, { labelIndex: selectedLabel, featureIndices: selectedFeatures });
      }

      if (datasetNumeric.X.length < 30) throw new Error("Too few usable rows after filtering (need at least 30)");

      // Dataset summary
      const dsEntries = [
        ["Task", taskLabel(selectedTask)],
        ["Rows (usable)", String(datasetNumeric.X.length)],
        ["Dropped rows", String(datasetNumeric.droppedRows)],
      ];
      if (selectedTask === "binary_classification") {
        dsEntries.push(["Dropped other labels", String(datasetNumeric.droppedOtherLabel || 0)]);
        dsEntries.push(["Class mapping", `0=${datasetNumeric.classes[0]}, 1=${datasetNumeric.classes[1]}`]);
      } else if (selectedTask === "multiclass_classification") {
        dsEntries.push(["Dropped other labels", String(datasetNumeric.droppedOtherLabel || 0)]);
        dsEntries.push(["Classes", String(Array.isArray(datasetNumeric.classes) ? datasetNumeric.classes.length : 0)]);
        if (Array.isArray(datasetNumeric.classes) && datasetNumeric.classes.length) {
          const shown = datasetNumeric.classes.slice(0, 8).join(", ") + (datasetNumeric.classes.length > 8 ? ", …" : "");
          dsEntries.push(["Class labels", shown]);
        }
      } else if (selectedTask === "multilabel_classification") {
        dsEntries.push(["Labels", String(Array.isArray(datasetNumeric.labelNames) ? datasetNumeric.labelNames.length : 0)]);
        if (Array.isArray(datasetNumeric.labelNames) && datasetNumeric.labelNames.length) {
          const shown = datasetNumeric.labelNames.slice(0, 8).join(", ") + (datasetNumeric.labelNames.length > 8 ? ", …" : "");
          dsEntries.push(["Label columns", shown]);
        }
        const miss = Number(datasetNumeric.droppedLabelMissing || 0);
        const bad = Number(datasetNumeric.droppedLabelInvalid || 0);
        const badF = Number(datasetNumeric.droppedBadFeature || 0);
        if (miss) dsEntries.push(["Dropped missing labels", String(miss)]);
        if (bad) dsEntries.push(["Dropped invalid labels", String(bad)]);
        if (badF) dsEntries.push(["Dropped bad features", String(badF)]);
      }
      if (selectedTask !== "multilabel_classification") dsEntries.push(["Label", datasetNumeric.labelName]);
      dsEntries.push(["Features", String(datasetNumeric.featureNames.length)]);
      setKV(dsKV, dsEntries);

      const nRows = datasetNumeric.X.length;
      const nFeat = datasetNumeric.featureNames.length;

      // Determine class/label count (used for size clamp + worker params).
      let nClasses = 1;
      if (selectedTask === "binary_classification") {
        nClasses = 2;
      } else if (selectedTask === "multiclass_classification") {
        nClasses = Math.max(2, Array.isArray(datasetNumeric.classes) ? datasetNumeric.classes.length : 2);
      } else if (selectedTask === "multilabel_classification") {
        nClasses = Math.max(2, Array.isArray(datasetNumeric.labelNames) ? datasetNumeric.labelNames.length : 2);
      }

      // Quantization scale selection uses observed magnitudes.
      const XMaster = new Float32Array(nRows * nFeat);
      let maxAbsX = 0;
      for (let i = 0; i < nRows; i++) {
        const row = datasetNumeric.X[i];
        for (let j = 0; j < nFeat; j++) {
          const v = row[j];
          XMaster[i * nFeat + j] = v;
          const av = Math.abs(v);
          if (Number.isFinite(av) && av > maxAbsX) maxAbsX = av;
        }
      }

      let maxAbsY = 0;
      let yMaster = null;
      if (selectedTask === "multilabel_classification") {
        if (!(datasetNumeric.yFlat instanceof Float32Array)) {
          throw new Error("Internal: multilabel dataset is missing yFlat buffer");
        }
        yMaster = datasetNumeric.yFlat;
        for (let i = 0; i < yMaster.length; i++) {
          const av = Math.abs(yMaster[i]);
          if (Number.isFinite(av) && av > maxAbsY) maxAbsY = av;
        }
      } else {
        const y = new Float32Array(nRows);
        for (let i = 0; i < nRows; i++) {
          const v = datasetNumeric.y[i];
          y[i] = v;
          const av = Math.abs(v);
          if (Number.isFinite(av) && av > maxAbsY) maxAbsY = av;
        }
        yMaster = y;
      }

      const scaleQ = chooseScaleQ(selectedTask, maxAbsX, maxAbsY);

      // Read base params (from UI) + enforce size constraints.
      const baseParams = readBaseTrainParams({ nClasses });

      const doSearch = !!heuristicSearchOn?.checked;
      const doRefit = !!refitOn?.checked;

      const maxRounds = doSearch
        ? clampInt(parseInt(heuristicSearchRounds?.value || "10", 10), 1, 1000)
        : 1;

      // Clear previous trained model while running (prevents deploying stale output).
      trained = null;
      renderPreviewInputs();
      updateDeployState();

      // Training state
      isTraining = true;
      isSearching = doSearch;
      searchAbort = false;
      trainBtn.disabled = true;
      stopBtn.disabled = false;
      trainBar.style.width = "0%";
      trainPill.textContent = doSearch ? `Search 1/${maxRounds}…` : "Training…";
      setDockState("training");

      // Save training params for post-train feature importance.
      lastTrainInfo = {
        task: selectedTask,
        seed: baseParams.seed,
        splitTrain: baseParams.splitTrain,
        splitVal: baseParams.splitVal,
        nRows,
        nFeatures: nFeat,
        nClasses
      };

      // Cancel any in-flight importance computation and clear previous results.
      _featImpNonce += 1;
      try { _fiClear("Training… feature importance will appear here when done."); } catch {}

      if (doSearch) {
        // Reset search history table.
        searchHistory = [];
        searchPage = 1;
        renderSearchTable();

        const rng = _xorshift32(baseParams.seed ^ 0x9e3779b9);
        let best = null;
        let bestScore = Infinity;
        let bestRound = 0;

        for (let round = 1; round <= maxRounds; round++) {
          if (searchAbort) break;

          const params = (round === 1)
            ? { ...baseParams }
            : generateHeuristicCandidate({ baseParams, bestParams: best?.params || null, round, rng });

          // Create table row (params are shown even while running).
          const entry = {
            round,
            status: "running",
            params: _maybeDeepCopy(params),
            meta: null,
            curve: null,
            error: null,
          };
          // Keep the view on the latest page while the search grows (but don't yank the user
          // off an older page if they've intentionally paged back).
          const pagesBefore = Math.max(1, Math.ceil((searchHistory?.length || 0) / SEARCH_PAGE_SIZE));
          searchHistory.push(entry);
          const pagesAfter = Math.max(1, Math.ceil((searchHistory?.length || 0) / SEARCH_PAGE_SIZE));
          if ((searchPage | 0) >= pagesBefore) searchPage = pagesAfter;
          renderSearchTable();

          try {
            const res = await runTrainRound({ XMaster, yMaster, nRows, nFeat, scaleQ, params, round, totalRounds: maxRounds });
            entry.status = "done";
            entry.meta = res.meta;
            entry.curve = res.curve;

            const score = res?.meta?.bestValMetric;
            if (Number.isFinite(score) && score < bestScore) {
              bestScore = score;
              bestRound = round;
              best = { ...res, params };
            }
          } catch (err) {
            if (searchAbort) {
              entry.status = "stopped";
              entry.error = "Stopped";
              renderSearchTable();
              break;
            }
            entry.status = "error";
            entry.error = err?.message || String(err);
          }

          renderSearchTable();
        }

        if (!best || !best.bytes || !best.meta) {
          throw new Error(searchAbort ? "Training stopped" : "No successful search runs");
        }

        // Show best run visually + make it the trained model.
        curve.steps = best.curve.steps;
        curve.train = best.curve.train;
        curve.val = best.curve.val;
        curve.test = best.curve.test;
        curve.bestVal = best.curve.bestVal || [];
        drawCurves();

        applyParamsToTrainingUI(best.params);
        try { updateSize(); } catch {}

        lastTrainInfo = {
          task: selectedTask,
          seed: best.params.seed,
          splitTrain: best.params.splitTrain,
          splitVal: best.params.splitVal,
          nRows,
          nFeatures: nFeat,
          nClasses
        };

        // Optional final refit: retrain on Train+Val for the selected tree count (keeps on-chain size the same).
        let finalRes = best;
        if (doRefit && !searchAbort) {
          const usedTrees = Number(best?.meta?.usedTrees ?? best?.params?.trees ?? baseParams.trees);
          const refitParams = { ...best.params, trees: usedTrees, earlyStop: false, refitTrainVal: true };
          log(`[${nowTs()}] Refit enabled: training on Train+Val for ${usedTrees} trees (size unchanged).`);
          const refitRes = await runTrainRound({ XMaster, yMaster, nRows, nFeat, scaleQ, params: refitParams, round: 1, totalRounds: 1, label: "Refit" });
          finalRes = { ...refitRes, params: refitParams };
          applyParamsToTrainingUI(refitParams);
          try { updateSize(); } catch {}
        }

        const refitApplied = (finalRes !== best);

        applyTrainedModel(finalRes);

        trainBar.style.width = "100%";
        if (searchAbort) {
          trainPill.textContent = `Stopped (best round ${bestRound}/${maxRounds})`;
          log(`[${nowTs()}] Search stopped by user. Best round=${bestRound}/${maxRounds} bestVal=${bestScore.toFixed(6)}`);
        } else {
          trainPill.textContent = refitApplied ? `Done (best round ${bestRound}/${maxRounds} + refit)` : `Done (best round ${bestRound}/${maxRounds})`;
          log(`[${nowTs()}] Search complete. Best round=${bestRound}/${maxRounds} bestVal=${bestScore.toFixed(6)}${refitApplied ? " (refit applied)" : ""}`);
        }

        renderPreviewInputs();
        updateDeployState();
        try { void _fiComputeAndRender(); } catch {}

      } else {
        const res = await runTrainRound({ XMaster, yMaster, nRows, nFeat, scaleQ, params: baseParams, round: 1, totalRounds: 1 });

        lastTrainInfo = {
          task: selectedTask,
          seed: baseParams.seed,
          splitTrain: baseParams.splitTrain,
          splitVal: baseParams.splitVal,
          nRows,
          nFeatures: nFeat,
          nClasses
        };

        // Optional final refit: retrain on Train+Val for the selected tree count (keeps on-chain size the same).
        let finalRes = res;
        if (doRefit) {
          const usedTrees = Number(res?.meta?.usedTrees ?? baseParams.trees);
          const refitParams = { ...baseParams, trees: usedTrees, earlyStop: false, refitTrainVal: true };
          log(`[${nowTs()}] Refit enabled: training on Train+Val for ${usedTrees} trees (size unchanged).`);
          const refitRes = await runTrainRound({ XMaster, yMaster, nRows, nFeat, scaleQ, params: refitParams, round: 1, totalRounds: 1, label: "Refit" });
          finalRes = { ...refitRes, params: refitParams };
          applyParamsToTrainingUI(refitParams);
          try { updateSize(); } catch {}
        }

        applyTrainedModel(finalRes);

        trainBar.style.width = "100%";
        trainPill.textContent = (finalRes !== res) ? "Done (refit)" : "Done";
        log(`[${nowTs()}] Model ready. task=${selectedTask} modelId=${trained.modelId} bytes=${trained.bytes.length} features=${trained.decoded.nFeatures} trees=${trained.decoded.nTrees}` + ((finalRes !== res) ? " (refit applied)" : ""));

        renderPreviewInputs();
        updateDeployState();
        try { void _fiComputeAndRender(); } catch {}
      }

      isTraining = false;
      isSearching = false;
      stopBtn.disabled = true;
      trainBtn.disabled = false;
      setDockState("idle");

    } catch (e) {
      // IMPORTANT: ensure UI resets even if Worker creation fails.
      const stopped = searchAbort || /stopp/i.test(String(e?.message || ""));
      trainPill.textContent = stopped ? "Stopped" : "Idle";
      try { worker?.terminate(); } catch {}
      worker = null;
      isTraining = false;
      isSearching = false;
      trainBtn.disabled = false;
      stopBtn.disabled = true;
      try { trainBar.style.width = "0%"; } catch {}

      const msg = e?.message || String(e);
      if (stopped) {
        log(`[${nowTs()}] Training stopped.`);
      } else {
        log(`[${nowTs()}] [error] ${msg}`);
      }
      updateDeployState();
    }
  });

  // Legal refresh
  async function refreshLegal() {
    const sys = loadSystem();
    if (!sys.registry || !sys.rpc) return;
    try {
      const rp = getReadProvider(sys.rpc);
      const registry = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, rp);
      const licId = Number(await registry.activeLicenseId({ gasLimit: 2_000_000_000 }));
      const lic = await registry.getLicense(BigInt(licId), { gasLimit: 2_000_000_000 });
      activeLicenseId = licId;
      activeTosVersion = Number(await registry.tosVersion({ gasLimit: 2_000_000_000 }));
      const tosHash = await registry.tosHash({ gasLimit: 2_000_000_000 });

      licenseLine.textContent = `Active license: #${licId} ${lic[0]} ${lic[1]}`;
      tosLine.textContent = `Active Terms: version #${activeTosVersion} (hash ${String(tosHash).slice(0,10)}…)`;
    } catch (e) {
      log(`[${nowTs()}] [warn] Failed to load legal: ${e.message || e}`);
    }
  }
  await refreshLegal();

  iconFile.addEventListener("change", async () => {
    const f = iconFile.files?.[0];
    if (!f) return;
    try {
      iconBytes = await validateIcon128(f);
      dlog(`[${nowTs()}] Icon OK: ${f.name} (${iconBytes.length} bytes)`);
    } catch (e) {
      iconBytes = null;
      dlog(`[${nowTs()}] [error] Icon invalid: ${e.message || e}`);
    }
    updateDeployState();
  });

  // Deploy estimate (transactions + on-chain deploy value)

  function scheduleDeployEstimate() {
    if (!deployEstKV) return;
    if (_deployEstTimer) clearTimeout(_deployEstTimer);
    _deployEstTimer = setTimeout(() => {
      refreshDeployEstimate().catch(() => {});
    }, 250);
  }

  async function refreshDeployEstimate() {
    if (!deployEstKV) return;
    const sys = loadSystem();
    const total = trained?.bytes?.length || 0;

    if (!sys.rpc || !sys.registry || !sys.store || !total) {
      setKV(deployEstKV, [
        ["Transactions", "—"],
        ["Required deploy value", "—"],
      ]);
      if (deployEstNote) {
        deployEstNote.textContent = "Train a model and apply System to see deploy estimate (transactions + on-chain deploy value).";
      }
      return;
    }

    const chunkSize = CHUNK_SIZE;
    const numChunks = Math.ceil(total / chunkSize);
    const totalTxs = numChunks + 2; // N chunk writes + pointer-table write + register

    setKV(deployEstKV, [
      ["Model bytes", String(total)],
      ["Chunk size", `${chunkSize} bytes (fixed)`],
      ["Chunks", String(numChunks)],
      ["Transactions", `${totalTxs} (Store writes: ${numChunks + 1}, Register: 1)`],
      ["Required deploy value", "Loading…"],
    ]);
    if (deployEstNote) deployEstNote.textContent = "Loading chain fee info…";

    const nonce = ++_deployEstNonce;
    try {
      const rp = getReadProvider(sys.rpc);
      const regRead = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, rp);

      let deployFeeWei = 0n;
      let sizeFeeWeiPerByte = 0n;
      let requiredFeeWei = 0n;
      try { deployFeeWei = BigInt(await regRead.deployFeeWei({ gasLimit: 2_000_000_000 })); } catch {}
      try { sizeFeeWeiPerByte = BigInt(await regRead.sizeFeeWeiPerByte({ gasLimit: 2_000_000_000 })); } catch {}
      try {
        requiredFeeWei = BigInt(await regRead.requiredDeployFeeWei(total, { gasLimit: 2_000_000_000 }));
      } catch {
        requiredFeeWei = deployFeeWei + (sizeFeeWeiPerByte * BigInt(total));
      }

      if (nonce !== _deployEstNonce) return;

      setKV(deployEstKV, [
        ["Model bytes", String(total)],
        ["Chunk size", `${chunkSize} bytes (fixed)`],
        ["Chunks", String(numChunks)],
        ["Transactions", `${totalTxs} (Store writes: ${numChunks + 1}, Register: 1)`],
        ["Deploy fee (base)", `${weiToEth(deployFeeWei)} L1`],
        ["Size fee", `${weiToEth(sizeFeeWeiPerByte)} L1 / byte`],
        ["Required deploy value", `${weiToEth(requiredFeeWei)} L1`],
      ]);

      if (deployEstNote) {
        deployEstNote.textContent = "Gas for each transaction is paid in addition to the on-chain deploy value. Your wallet will show gas before you confirm.";
      }
    } catch (e) {
      if (nonce !== _deployEstNonce) return;
      if (deployEstNote) deployEstNote.textContent = "Could not load fee data from chain (check RPC endpoint + Registry address).";
      log(`[${nowTs()}] [warn] Deploy estimate: ${e.message || e}`);
    }
  }

  function updateDeployState() {
    const sys = loadSystem();
    const w = getWalletState();

    const haveSys = !!(sys.rpc && sys.store && sys.registry);
    const walletOk = !!w.address;
    const chainOk = String(w.chainId || "") === "29";
    const haveModel = !!trained?.bytes?.length;
    const haveMeta = metaName.value.trim().length >= 3 && metaDesc.value.trim().length >= 8;
    const haveIcon = !!iconBytes?.length;
    const agreed = agreeTos.checked && agreeLicense.checked;
    const haveOwnerKey = !!(ownerKeyAddr?.value && ownerKeyAddr.value.trim().length > 0);
    const ownerKeyConfirmed = !!ownerKeySaved?.checked;

    const trainingNow = !!isTraining;

    const ok = haveSys && walletOk && chainOk && haveModel && haveMeta && haveIcon && agreed && haveOwnerKey && ownerKeyConfirmed && !trainingNow;
    deployBtn.disabled = !ok;

    const reasons = [];
    if (trainingNow) reasons.push("wait for training");
    if (!haveSys) reasons.push("system config");
    if (!walletOk) reasons.push("connect wallet");
    if (walletOk && !chainOk) reasons.push("switch to chainId=29");
    if (!haveModel) reasons.push("train model");
    if (!haveOwnerKey) reasons.push("generate owner key");
    if (haveOwnerKey && !ownerKeyConfirmed) reasons.push("confirm owner key saved");
    if (!haveMeta) reasons.push("name+description");
    if (!haveIcon) reasons.push("128×128 icon");
    if (!agreed) reasons.push("agree Terms+License");

    deployPill.textContent = ok ? "Ready to deploy" : `Need: ${reasons.join(", ")}`;

    // Keep the feature-importance box in sync with whether a trained model exists.
    // (Avoids showing stale scores after changing features/labels.)
    try {
      if (!trained?.decoded) _fiClear("Train a model to see feature importance.");
    } catch {}

    // Keep Local preview summary blocks in sync with trained model presence.
    try { renderPreviewBestModelBlocks(); } catch {}

    scheduleDeployEstimate();

    try { updateGl1fUI(); } catch {}
  }

  [metaName, metaDesc, agreeTos, agreeLicense, ownerKeySaved, pricingMode, pricingFee, pricingRecipient].forEach((el) => {
    el.addEventListener("input", updateDeployState);
    el.addEventListener("change", updateDeployState);
  });

  // Deploy flow
  deployBtn.addEventListener("click", async () => {
    try {
      const sys = loadSystem();
      if (!sys.rpc || !sys.store || !sys.registry) throw new Error("Missing system config");
      const w = getWalletState();
      if (!w.address) throw new Error("Connect wallet");
      if (String(w.chainId) !== "29") throw new Error("Switch to chainId=29");

      if (!trained?.bytes?.length) throw new Error("Train a model first");
      if (!datasetNumeric?.featureNames?.length) throw new Error("Feature labels missing");
      if (!iconBytes?.length) throw new Error("Upload icon");
      if (!ownerKeyAddr?.value) throw new Error("Generate owner API key");
      if (!ownerKeySaved?.checked) throw new Error("Confirm you saved the owner API key private key");
      if (!(agreeTos.checked && agreeLicense.checked)) throw new Error("Agree to Terms and License");

      const title = metaName.value.trim();
      const desc = metaDesc.value.trim();
      if (title.length < 3 || desc.length < 8) throw new Error("Provide name + description");
      const words = titleWordHashes(title);
      if (!words.length) throw new Error("Title should include at least one word (2+ chars)");

      const mode = Number(pricingMode.value);
      const feeEth = clamp(pricingFee.value || "0", 0.001, 1);
      let feeWei = 0n;
      if (mode === 0) feeWei = 0n;
      else feeWei = ethToWei(String(feeEth));

      const { signer } = await getSignerProvider();
      const signerAddr = await signer.getAddress();

      const store = new ethers.Contract(mustAddr(sys.store), ABI_STORE, signer);
      const registry = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, signer);

      // Model bytes
      const bytes = trained.bytes;
      const total = bytes.length;

      // Read chain settings via the dedicated RPC (more reliable than wallet provider for eth_call).
      const rprov = getReadProvider(sys.rpc);
      const regRead = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, rprov);

      let deployFeeWei = 0n;
      let sizeFeeWeiPerByte = 0n;
      let requiredFeeWei = 0n;
      let licId = 0;
      let tosVer = 0;
      try { deployFeeWei = BigInt(await regRead.deployFeeWei()); } catch {}
      try { sizeFeeWeiPerByte = BigInt(await regRead.sizeFeeWeiPerByte()); } catch {}
      try {
        requiredFeeWei = BigInt(await regRead.requiredDeployFeeWei(total));
      } catch {
        requiredFeeWei = deployFeeWei + (sizeFeeWeiPerByte * BigInt(total));
      }
      try { licId = Number(await regRead.activeLicenseId()); } catch {}
      try { tosVer = Number(await regRead.tosVersion()); } catch {}

      dlog(`[${nowTs()}] Deploy fee (base): ${weiToEth(deployFeeWei)} L1`);
      dlog(`[${nowTs()}] Size fee: ${weiToEth(sizeFeeWeiPerByte)} L1 per byte`);
      dlog(`[${nowTs()}] Required deploy value: ${weiToEth(requiredFeeWei)} L1`);
      dlog(`[${nowTs()}] Active licenseId=${licId} tosVersion=${tosVer}`);

      let recipient = signerAddr;
      if (pricingRecipient.value.trim()) recipient = mustAddr(pricingRecipient.value.trim());

      // chunking (fixed)
      const chunkSize = CHUNK_SIZE;
      const numChunks = Math.ceil(total / chunkSize);

      dlog(`[${nowTs()}] Chunking: total=${total} chunkSize=${chunkSize} (fixed) chunks=${numChunks}`);

      const iface = new ethers.Interface(ABI_STORE);
      const ptrs = [];

      for (let i=0;i<numChunks;i++){
        const start = i*chunkSize;
        const end = Math.min(total, start+chunkSize);
        const chunk = bytes.slice(start,end);
        dlog(`[${nowTs()}] Chunk ${i+1}/${numChunks}: store.write(${chunk.length} bytes)`);
        const tx = await store.write(chunk, { gasLimit: 30_000_000 });
        dlog(`  tx.hash ${tx.hash}`);
        const rcpt = await tx.wait();
        dlog(`  mined status=${rcpt.status} gasUsed=${rcpt.gasUsed?.toString?.()||"?"}`);
        if (rcpt.status !== 1) throw new Error("chunk write reverted");

        let ptr = null;
        for (const lg of rcpt.logs) {
          try {
            const pl = iface.parseLog(lg);
            if (pl?.name === "ChunkWritten") { ptr = pl.args.pointer; break; }
          } catch {}
        }
        if (!ptr) throw new Error("ChunkWritten not found");
        ptrs.push(ptr);
        dlog(`  chunk pointer: ${ptr}`);
      }

      // pointer table chunk: 32 bytes each pointer
      const table = new Uint8Array(32*numChunks);
      for (let i=0;i<numChunks;i++){
        const addr = ethers.getAddress(ptrs[i]);
        const ab = ethers.getBytes(addr);
        table.set(ab, i*32 + 12);
      }
      dlog(`[${nowTs()}] Writing pointer-table: ${table.length} bytes`);
      const ttx = await store.write(table, { gasLimit: 30_000_000 });
      dlog(`  table tx.hash ${ttx.hash}`);
      const trc = await ttx.wait();
      dlog(`  table mined status=${trc.status} gasUsed=${trc.gasUsed?.toString?.()||"?"}`);
      if (trc.status !== 1) throw new Error("table write reverted");

      let tablePtr = null;
      for (const lg of trc.logs) {
        try {
          const pl = iface.parseLog(lg);
          if (pl?.name === "ChunkWritten") { tablePtr = pl.args.pointer; break; }
        } catch {}
      }
      if (!tablePtr) throw new Error("table ChunkWritten not found");
      dlog(`[${nowTs()}] Pointer-table pointer: ${tablePtr}`);

      // Register
      const modelId = trained.modelId;
      let labelsForNft = null;
      let labelNamesForNft = null;
      if (selectedTask === "binary_classification" && datasetNumeric?.classes) {
        // Class labels for binary classification.
        labelsForNft = [datasetNumeric.classes[0], datasetNumeric.classes[1]];
      } else if (selectedTask === "multiclass_classification" && Array.isArray(datasetNumeric?.classes)) {
        // Class labels for multiclass classification.
        labelsForNft = datasetNumeric.classes;
      } else if (selectedTask === "multilabel_classification" && Array.isArray(datasetNumeric?.labelNames)) {
        // Multilabel:
        // - `labelNames` are the output label names (one per selected label column)
        // - `labels` are the binary class labels (defaults to 0/1)
        labelNamesForNft = datasetNumeric.labelNames;
        labelsForNft = ["0", "1"];
      }
      const featuresPacked = packNftFeatures({
        task: selectedTask,
        // Regression/binary/multiclass store the single label column name. Multilabel stores labelNames instead.
        labelName: (selectedTask === "multilabel_classification") ? "(multilabel)" : datasetNumeric.labelName,
        labels: labelsForNft,
        labelNames: labelNamesForNft,
        featureNames: datasetNumeric.featureNames
      });

      const depth = trained.decoded.depth;
      const nTrees = trained.decoded.nTrees;
      const nFeatures = trained.decoded.nFeatures;
      const baseQ = trained.decoded.baseQ;
      const scaleQ = trained.decoded.scaleQ;

      dlog(`[${nowTs()}] Registering model…`);

      const regTx = await registry.registerModel(
        modelId,
        tablePtr,
        chunkSize,
        numChunks,
        total,
        nFeatures,
        nTrees,
        depth,
        baseQ,
        scaleQ,
        title,
        desc,
        iconBytes,
        featuresPacked,
        words,
        mode,
        feeWei,
        recipient,
        tosVer,
        licId,
        ownerKeyAddr.value,
        { value: requiredFeeWei, gasLimit: 35_000_000 }
      );
      dlog(`  reg tx.hash ${regTx.hash}`);
      const rrc = await regTx.wait();
      dlog(`  reg mined status=${rrc.status} gasUsed=${rrc.gasUsed?.toString?.()||"?"}`);
      if (rrc.status !== 1) throw new Error("register reverted");

      dlog(`[${nowTs()}] ✅ Model NFT deployed.`);
      dlog(`Model ID: ${modelId}`);
      updateDeployState();
    } catch (e) {
      dlog(`[${nowTs()}] [error] ${e.message || e}`);
    }
  });

  window.addEventListener("genesis_wallet_changed", () => {
    const w = getWalletState();
    if (w?.address) log(`[${nowTs()}] Wallet: ${w.address} chainId=${w.chainId}`);
    updateDeployState();
  });

  updateTaskUI();
  updateSize();
  updateDeployState();
  try { updateGl1fUI(); } catch {}
});