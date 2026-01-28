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

// Robust CSV parser with quoted fields + delimiter auto-detection.
// API expected by create_page.js:
//  - parseCSV(text) -> { headers: string[], rows: string[][], delimiter: string }
//  - toNumericMatrix
//  - inferLabelValues
//  - toBinaryMatrix
//  - toMulticlassMatrix
//  - toMultilabelMatrix
//
// Supports delimiters: comma, semicolon, tab, pipe.

const _DELIM_CANDS = [",", ";", "\t", "|"];

// Split a single CSV line with quotes, for delimiter detection (no newlines).
function _splitCsvLine(line, delim) {
  const out = [];
  let cur = "";
  let inQ = false;

  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    const n = line[i + 1];

    if (inQ) {
      if (c === '"' && n === '"') { cur += '"'; i++; continue; }
      if (c === '"') { inQ = false; continue; }
      cur += c;
      continue;
    }

    if (c === '"') { inQ = true; continue; }
    if (c === "\r") continue;

    if (c === delim) { out.push(cur); cur = ""; continue; }
    cur += c;
  }
  out.push(cur);
  return out;
}

export function detectCSVDelimiter(text) {
  const s = String(text || "");
  const lines = s
    .split("\n")
    .map((x) => x.replace(/\r/g, ""))
    .filter((x) => x.trim().length);

  const sample = lines.slice(0, 20);
  if (!sample.length) return ",";

  let best = ",";
  let bestMode = 1;
  let bestFreq = 0;
  let bestPenalty = Number.POSITIVE_INFINITY;

  for (const d of _DELIM_CANDS) {
    const freq = new Map();
    let penalty = 0;

    for (const ln of sample) {
      const row = _splitCsvLine(ln, d);
      const n = row.length;
      freq.set(n, (freq.get(n) || 0) + 1);

      // Penalty: how many other delimiter chars appear inside cells.
      for (const cell of row) {
        for (const od of _DELIM_CANDS) {
          if (od === d) continue;
          for (let i = 0; i < cell.length; i++) if (cell[i] === od) penalty++;
        }
      }
    }

    // Find mode column count
    let modeN = 1, modeF = 0;
    for (const [k, v] of freq.entries()) {
      if (v > modeF || (v === modeF && k > modeN)) { modeN = k; modeF = v; }
    }
    if (modeN < 2) continue;

    // Prefer: higher mode frequency; tie-break: higher modeN; tie-break: lower penalty
    if (
      modeF > bestFreq ||
      (modeF === bestFreq && modeN > bestMode) ||
      (modeF === bestFreq && modeN === bestMode && penalty < bestPenalty)
    ) {
      best = d;
      bestFreq = modeF;
      bestMode = modeN;
      bestPenalty = penalty;
    }
  }

  return best;
}

export function parseCSV(text) {
  const s = String(text || "");
  const delimiter = detectCSVDelimiter(s);

  const rows = [];
  let row = [];
  let cur = "";
  let inQ = false;

  const pushCell = () => { row.push(cur); cur = ""; };
  const pushRow = () => { rows.push(row); row = []; };

  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    const n = s[i + 1];

    if (inQ) {
      if (c === '"' && n === '"') { cur += '"'; i++; continue; }
      if (c === '"') { inQ = false; continue; }
      cur += c;
      continue;
    }

    if (c === '"') { inQ = true; continue; }
    if (c === delimiter) { pushCell(); continue; }
    if (c === "\r") continue;

    if (c === "\n") { pushCell(); pushRow(); continue; }
    cur += c;
  }

  pushCell();
  if (row.length) pushRow();

  const headers = (rows[0] || []).map((h) => String(h || "").trim());
  if (headers.length && headers[0].startsWith("\uFEFF")) headers[0] = headers[0].replace(/^\uFEFF+/, "");

  const data = rows
    .slice(1)
    .filter((r) => r.length && r.some((x) => String(x || "").trim().length));

  // Normalize row length to header length
  const norm = data.map((r) => {
    const out = new Array(headers.length).fill("");
    for (let i = 0; i < headers.length; i++) out[i] = (r[i] ?? "").toString().trim();
    return out;
  });

  return { headers, rows: norm, delimiter };
}

export function toNumericMatrix(parsed, { labelIndex, featureIndices }) {
  const headers = parsed.headers;
  const rows = parsed.rows;

  const featureNames = featureIndices.map((i) => headers[i] ?? `f${i}`);
  const labelName = headers[labelIndex] ?? "label";

  const X = [];
  const y = [];
  let droppedRows = 0;

  for (const r of rows) {
    const yy = parseFloat(r[labelIndex]);
    if (!Number.isFinite(yy)) { droppedRows++; continue; }

    const xx = new Array(featureIndices.length);
    let ok = true;
    for (let j = 0; j < featureIndices.length; j++) {
      const v = parseFloat(r[featureIndices[j]]);
      if (!Number.isFinite(v)) { ok = false; break; }
      xx[j] = v;
    }
    if (!ok) { droppedRows++; continue; }

    X.push(xx);
    y.push(yy);
  }

  return { X, y, featureNames, labelName, droppedRows };
}

// ---- label canonicalization helpers ----
function _isStrictNumber(str) {
  const s = String(str ?? "").trim();
  if (!s) return false;
  return /^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$/.test(s);
}

function _canonLabel(raw) {
  const s = String(raw ?? "").trim();
  if (!s) return "";
  if (_isStrictNumber(s)) {
    const n = Number(s);
    if (Number.isFinite(n)) return String(n);
  }
  return s;
}

export function inferLabelValues(parsed, labelIndex) {
  const rows = parsed?.rows || [];
  const counts = new Map();
  let allNumeric = true;

  for (const r of rows) {
    const raw = r?.[labelIndex];
    const v = _canonLabel(raw);
    if (!v) continue;
    counts.set(v, (counts.get(v) || 0) + 1);
    if (!_isStrictNumber(v)) allNumeric = false;
  }

  let values = Array.from(counts.keys());
  if (allNumeric) {
    values = values
      .map((s) => ({ s, n: Number(s) }))
      .filter((o) => Number.isFinite(o.n))
      .sort((a, b) => a.n - b.n)
      .map((o) => o.s);
  } else {
    values.sort((a, b) => a.localeCompare(b));
  }

  return { values, counts, allNumeric };
}

export function toBinaryMatrix(parsed, { labelIndex, featureIndices, negLabel, posLabel }) {
  const headers = parsed.headers;
  const rows = parsed.rows;

  const featureNames = featureIndices.map((i) => headers[i] ?? `f${i}`);
  const labelName = headers[labelIndex] ?? "label";

  const neg = _canonLabel(negLabel);
  const pos = _canonLabel(posLabel);
  if (!neg || !pos) throw new Error("Select both negative and positive classes");
  if (neg === pos) throw new Error("Negative and positive classes must differ");

  const X = [];
  const y = [];
  let droppedRows = 0;
  let droppedOtherLabel = 0;

  for (const r of rows) {
    const labRaw = _canonLabel(r[labelIndex]);
    if (!labRaw) { droppedRows++; continue; }

    let yy = null;
    if (labRaw === neg) yy = 0;
    else if (labRaw === pos) yy = 1;
    else { droppedRows++; droppedOtherLabel++; continue; }

    const xx = new Array(featureIndices.length);
    let ok = true;
    for (let j = 0; j < featureIndices.length; j++) {
      const v = parseFloat(r[featureIndices[j]]);
      if (!Number.isFinite(v)) { ok = false; break; }
      xx[j] = v;
    }
    if (!ok) { droppedRows++; continue; }

    X.push(xx);
    y.push(yy);
  }

  return {
    X,
    y,
    featureNames,
    labelName,
    droppedRows,
    droppedOtherLabel,
    classes: { 0: neg, 1: pos }
  };
}

export function toMulticlassMatrix(parsed, { labelIndex, featureIndices, classLabels }) {
  const headers = parsed.headers;
  const rows = parsed.rows;

  const featureNames = featureIndices.map((i) => headers[i] ?? `f${i}`);
  const labelName = headers[labelIndex] ?? "label";

  const labelsRaw = Array.isArray(classLabels) ? classLabels : [];
  if (labelsRaw.length < 2) throw new Error("Select at least 2 classes");

  const classes = labelsRaw.map((x) => _canonLabel(x));
  const map = new Map();
  for (let i = 0; i < classes.length; i++) {
    const v = classes[i];
    if (!v) continue;
    if (!map.has(v)) map.set(v, i);
  }
  if (map.size < 2) throw new Error("Need at least 2 distinct classes");

  const X = [];
  const y = [];
  let droppedRows = 0;
  let droppedOtherLabel = 0;

  for (const r of rows) {
    const labRaw = _canonLabel(r[labelIndex]);
    if (!labRaw) { droppedRows++; continue; }
    const cls = map.get(labRaw);
    if (cls === undefined) { droppedRows++; droppedOtherLabel++; continue; }

    const xx = new Array(featureIndices.length);
    let ok = true;
    for (let j = 0; j < featureIndices.length; j++) {
      const v = parseFloat(r[featureIndices[j]]);
      if (!Number.isFinite(v)) { ok = false; break; }
      xx[j] = v;
    }
    if (!ok) { droppedRows++; continue; }

    X.push(xx);
    y.push(cls);
  }

  return {
    X,
    y,
    featureNames,
    labelName,
    droppedRows,
    droppedOtherLabel,
    classes
  };
}

// ---- Multilabel helpers ----

// Parse a binary 0/1 label from a CSV cell.
// Supports: 0/1, true/false, yes/no, t/f, y/n (case-insensitive)
function _parseBinary01(raw) {
  if (raw === null || raw === undefined) return null;
  const s = String(raw).trim();
  if (!s) return null;
  const k = s.toLowerCase();
  if (k === "0" || k === "0.0") return 0;
  if (k === "1" || k === "1.0") return 1;
  if (k === "true" || k === "t" || k === "yes" || k === "y") return 1;
  if (k === "false" || k === "f" || k === "no" || k === "n") return 0;
  if (_isStrictNumber(k)) {
    const n = Number(k);
    if (n === 0) return 0;
    if (n === 1) return 1;
  }
  return null;
}

// Convert to numeric X and multilabel y (one binary label per selected column).
// Drops rows where any selected label cell is missing/invalid or any selected feature is non-numeric.
export function toMultilabelMatrix(parsed, { labelIndices, featureIndices }) {
  if (!parsed || !Array.isArray(parsed.headers) || !Array.isArray(parsed.rows)) throw new Error("Bad parsed CSV");
  const nCols = parsed.headers.length;

  const labIdx = Array.from(labelIndices || [])
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x));

  if (!labIdx.length) throw new Error("Need at least 1 label column");

  // Dedup while preserving order
  const seenL = new Set();
  const labelCols = [];
  for (const i of labIdx) {
    const ii = i | 0;
    if (ii < 0 || ii >= nCols) continue;
    if (seenL.has(ii)) continue;
    seenL.add(ii);
    labelCols.push(ii);
  }
  if (!labelCols.length) throw new Error("No valid label columns selected");

  const featIdx = Array.from(featureIndices || [])
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x));

  const seenF = new Set();
  const featCols = [];
  for (const i of featIdx) {
    const ii = i | 0;
    if (ii < 0 || ii >= nCols) continue;
    if (seenF.has(ii)) continue;
    if (seenL.has(ii)) continue; // disallow overlap with labels
    seenF.add(ii);
    featCols.push(ii);
  }

  const labelNames = labelCols.map((i) => parsed.headers[i] || `label${i}`);
  const featureNames = featCols.map((i) => parsed.headers[i] || `f${i}`);
  const nLabels = labelCols.length;

  const X = [];
  const y = [];

  let droppedRows = 0;
  let droppedLabelMissing = 0;
  let droppedLabelInvalid = 0;
  let droppedBadFeature = 0;

  for (const row of parsed.rows) {
    const yRow = new Array(nLabels);
    let bad = false;

    // labels
    for (let k = 0; k < nLabels; k++) {
      const v = row[labelCols[k]];
      const s = (v === null || v === undefined) ? "" : String(v).trim();
      if (!s) { droppedLabelMissing++; bad = true; break; }
      const b = _parseBinary01(s);
      if (b === null) { droppedLabelInvalid++; bad = true; break; }
      yRow[k] = b;
    }
    if (bad) { droppedRows++; continue; }

    // features
    const xRow = new Array(featCols.length);
    for (let j = 0; j < featCols.length; j++) {
      const raw = row[featCols[j]];
      const num = parseFloat(raw);
      if (!Number.isFinite(num)) { droppedBadFeature++; bad = true; break; }
      xRow[j] = num;
    }
    if (bad) { droppedRows++; continue; }

    X.push(xRow);
    y.push(yRow);
  }

  const yFlat = new Float32Array(y.length * nLabels);
  for (let r = 0; r < y.length; r++) {
    const base = r * nLabels;
    const yr = y[r];
    for (let k = 0; k < nLabels; k++) yFlat[base + k] = yr[k];
  }

  return {
    X,
    y,
    yFlat,
    labelNames,
    featureNames,
    droppedRows,
    droppedLabelMissing,
    droppedLabelInvalid,
    droppedBadFeature,
  };
}

