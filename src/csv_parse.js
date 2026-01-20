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

// Simple CSV parser with quoted fields.
// Returns: { headers: string[], rows: string[][] }

export function parseCSV(text) {
  const s = String(text || "");
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
    if (c === ",") { pushCell(); continue; }
    if (c === "\r") continue;

    if (c === "\n") { pushCell(); pushRow(); continue; }

    cur += c;
  }

  pushCell();
  if (row.length) pushRow();

  const headers = (rows[0] || []).map((h) => String(h || "").trim());
  const data = rows.slice(1).filter((r) => r.length && r.some((x) => String(x || "").trim().length));

  const norm = data.map((r) => {
    const out = new Array(headers.length).fill("");
    for (let i = 0; i < headers.length; i++) out[i] = (r[i] ?? "").toString().trim();
    return out;
  });

  return { headers, rows: norm };
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

// ---- Binary classification helpers ----

function _isStrictNumber(str) {
  const s = String(str ?? "").trim();
  if (!s) return false;
  // Covers: 1, -1, 1.23, .5, 1e-3, -2E6
  return /^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$/.test(s);
}

function _canonLabel(raw) {
  const s = String(raw ?? "").trim();
  if (!s) return "";
  if (_isStrictNumber(s)) {
    // Canonical numeric string ("01" -> "1", "1.0" -> "1")
    const n = Number(s);
    if (Number.isFinite(n)) return String(n);
  }
  return s;
}

// Infer unique label values (canonicalized). Useful for binary classification.
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

// Convert to numeric X and binary y (0/1). Rows with:
// - missing/non-numeric features
// - missing labels
// - labels not in the chosen {negLabel,posLabel}
// are dropped.
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

// Convert to numeric X and multiclass y (0..K-1) according to the provided classLabels.
// Rows with:
// - missing/non-numeric features
// - missing labels
// - labels not in classLabels
// are dropped.
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

// Parse a binary 0/1 label from a CSV cell.
// Supports: 0/1, true/false, yes/no, t/f, y/n (case-insensitive)
// Returns: 0 | 1 | null (invalid/missing)
function _parseBinary01(raw) {
  if (raw === null || raw === undefined) return null;
  const s = String(raw).trim();
  if (!s) return null;
  const k = s.toLowerCase();
  if (k === "0" || k === "0.0") return 0;
  if (k === "1" || k === "1.0") return 1;
  if (k === "true" || k === "t" || k === "yes" || k === "y") return 1;
  if (k === "false" || k === "f" || k === "no" || k === "n") return 0;
  // Fallback: strict numeric parse for "0"/"1" variants.
  if (_isStrictNumber(k)) {
    const n = Number(k);
    if (n === 0) return 0;
    if (n === 1) return 1;
  }
  return null;
}

// Convert to numeric X and multilabel y (one binary label per selected column).
// Drops rows where any selected label cell is missing/invalid or any selected feature is non-numeric.
// Returns y both as a matrix (array-of-arrays) and a packed Float32Array row-major.
export function toMultilabelMatrix(parsed, { labelIndices, featureIndices }) {
  if (!parsed || !Array.isArray(parsed.headers) || !Array.isArray(parsed.rows)) throw new Error("Bad parsed CSV");
  const nCols = parsed.headers.length;

  const labIdx = Array.from(labelIndices || []).map((x) => Number(x)).filter((x) => Number.isFinite(x));
  if (!labIdx.length) throw new Error("Need at least 1 label column");
  // Dedup while preserving order.
  const seen = new Set();
  const labelCols = [];
  for (const i of labIdx) {
    const ii = i | 0;
    if (ii < 0 || ii >= nCols) continue;
    if (seen.has(ii)) continue;
    seen.add(ii);
    labelCols.push(ii);
  }
  if (!labelCols.length) throw new Error("No valid label columns selected");

  const featIdx = Array.from(featureIndices || []).map((x) => Number(x)).filter((x) => Number.isFinite(x));
  const featCols = [];
  const seenF = new Set();
  for (const i of featIdx) {
    const ii = i | 0;
    if (ii < 0 || ii >= nCols) continue;
    if (seenF.has(ii)) continue;
    // Features cannot overlap labels.
    if (seen.has(ii)) continue;
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
    // Parse labels
    const yRow = new Array(nLabels);
    let bad = false;
    for (let k = 0; k < nLabels; k++) {
      const v = row[labelCols[k]];
      const s = (v === null || v === undefined) ? "" : String(v).trim();
      if (!s) {
        droppedLabelMissing += 1;
        bad = true;
        break;
      }
      const b = _parseBinary01(s);
      if (b === null) {
        droppedLabelInvalid += 1;
        bad = true;
        break;
      }
      yRow[k] = b;
    }
    if (bad) {
      droppedRows += 1;
      continue;
    }

    // Parse features (numeric)
    const xRow = new Array(featCols.length);
    for (let j = 0; j < featCols.length; j++) {
      const raw = row[featCols[j]];
      const num = parseFloat(raw);
      if (!Number.isFinite(num)) {
        droppedBadFeature += 1;
        bad = true;
        break;
      }
      xRow[j] = num;
    }
    if (bad) {
      droppedRows += 1;
      continue;
    }

    X.push(xRow);
    y.push(yRow);
  }

  // Pack yFlat (row-major)
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
