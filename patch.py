#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parent

def die(msg: str) -> None:
    print("ERROR:", msg, file=sys.stderr)
    raise SystemExit(1)

def read(p: Path) -> str:
    if not p.exists():
        die(f"Missing file: {p}")
    return p.read_text(encoding="utf-8")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def ensure(cond: bool, msg: str) -> None:
    if not cond:
        die(msg)

def patch_cpp_train_gl1f_cpp() -> None:
    p = ROOT / "cpp" / "train_gl1f_cpp.cpp"
    s = read(p)
    orig = s

    # ---- 1) Add helpers (headers_preview + autodetect_delimiter_from_lines)
    if "autodetect_delimiter_from_lines" not in s:
        marker = "static inline bool is_int_string"
        i = s.find(marker)
        ensure(i != -1, "C++ patch: insertion marker not found (is_int_string)")

        helpers = r"""
static std::string headers_preview(const std::vector<std::string>& headers, int max_cols = 40) {
  std::string out;
  int n = (int)headers.size();
  int m = std::min(n, max_cols);
  for (int i = 0; i < m; i++) {
    if (i) out += ", ";
    out += headers[(size_t)i];
  }
  if (n > m) {
    out += ", ... (";
    out += std::to_string(n);
    out += " cols)";
  }
  return out;
}

static char autodetect_delimiter_from_lines(const std::vector<std::string>& lines, char fallback) {
  const char candidates[] = {',', ';', '\t', '|'};
  if (lines.empty()) return fallback;

  std::vector<std::string_view> fields;
  std::vector<std::string> quoted;

  int best_mode = 1;
  int best_freq = 0;
  long long best_penalty = (1LL<<62);
  char best = fallback;

  for (char d : candidates) {
    std::unordered_map<int,int> freq;
    freq.reserve(lines.size() * 2);
    long long penalty = 0;

    for (const auto& line : lines) {
      if (line.empty()) continue;
      parse_csv_line(line, d, fields, quoted);
      int n = (int)fields.size();
      freq[n] += 1;

      for (auto sv : fields) {
        for (char od : candidates) {
          if (od == d) continue;
          for (char c : sv) if (c == od) penalty++;
        }
      }
    }

    int mode_n = 1, mode_f = 0;
    for (const auto& kv : freq) {
      if (kv.second > mode_f || (kv.second == mode_f && kv.first > mode_n)) {
        mode_n = kv.first;
        mode_f = kv.second;
      }
    }
    if (mode_n < 2) continue;

    if (mode_f > best_freq ||
        (mode_f == best_freq && mode_n > best_mode) ||
        (mode_f == best_freq && mode_n == best_mode && penalty < best_penalty)) {
      best_freq = mode_f;
      best_mode = mode_n;
      best_penalty = penalty;
      best = d;
    }
  }
  return best;
}
"""
        s = s[:i] + helpers + "\n\n" + s[i:]

    # ---- 2) Update load_from_csv signature: add bool auto_delimiter
    s = s.replace(
        "  char delimiter,\n  bool has_header,\n",
        "  char delimiter,\n  bool auto_delimiter,\n  bool has_header,\n"
    )

    # ---- 3) Insert auto-detection block near top of load_from_csv (rewind to start; no buffering needed)
    needle = (
        "  Dataset ds;\n"
        "  std::string line;\n"
        "  std::vector<std::string_view> fields;\n"
        "  std::vector<std::string> quoted;\n\n"
        "  // Read header / establish column count\n"
    )
    if needle in s and "Auto-detected delimiter" not in s:
        block = (
            "  Dataset ds;\n"
            "  std::string line;\n"
            "  std::vector<std::string_view> fields;\n"
            "  std::vector<std::string> quoted;\n\n"
            "  if (auto_delimiter) {\n"
            "    std::vector<std::string> sample;\n"
            "    sample.reserve(32);\n"
            "    for (int i = 0; i < 20; ) {\n"
            "      if (!std::getline(f, line)) break;\n"
            "      if (line.empty()) continue;\n"
            "      sample.push_back(line);\n"
            "      i++;\n"
            "    }\n"
            "    delimiter = autodetect_delimiter_from_lines(sample, delimiter);\n"
            "    f.clear();\n"
            "    f.seekg(0, std::ios::beg);\n"
            "    if (delimiter == '\\t') std::cerr << \"INFO: Auto-detected delimiter '\\\\t'\\n\";\n"
            "    else std::cerr << \"INFO: Auto-detected delimiter '\" << delimiter << \"'\\n\";\n"
            "  }\n\n"
            "  // Read header / establish column count\n"
        )
        s = s.replace(needle, block)

    # ---- 4) Strip UTF-8 BOM from first header cell
    hdr_needle = "    for (auto sv : fields) ds.headers.push_back(trim_copy(sv));\n    n_cols = (int)ds.headers.size();\n"
    if hdr_needle in s and "Strip UTF-8 BOM" not in s:
        hdr_block = (
            "    for (auto sv : fields) ds.headers.push_back(trim_copy(sv));\n"
            "    // Strip UTF-8 BOM if present on first header\n"
            "    if (!ds.headers.empty() && ds.headers[0].size() >= 3 &&\n"
            "        (unsigned char)ds.headers[0][0] == 0xEF &&\n"
            "        (unsigned char)ds.headers[0][1] == 0xBB &&\n"
            "        (unsigned char)ds.headers[0][2] == 0xBF) {\n"
            "      ds.headers[0].erase(0, 3);\n"
            "    }\n"
            "    n_cols = (int)ds.headers.size();\n"
        )
        s = s.replace(hdr_needle, hdr_block)

    # ---- 5) Improve errors for missing label/feature columns
    s = s.replace(
        "      if (idx < 0) throw std::runtime_error(\"Bad label col: \" + spec);",
        "      if (idx < 0) throw std::runtime_error(\"Bad label col: \" + spec + \" (available: \" + headers_preview(ds.headers) + \")\");"
    )
    s = s.replace(
        "    if (single_label_idx < 0) throw std::runtime_error(\"Bad label col: \" + label_col);",
        "    if (single_label_idx < 0) throw std::runtime_error(\"Bad label col: \" + label_col + \" (available: \" + headers_preview(ds.headers) + \")\");"
    )
    s = s.replace(
        "      if (idx < 0) throw std::runtime_error(\"Bad feature col: \" + spec);",
        "      if (idx < 0) throw std::runtime_error(\"Bad feature col: \" + spec + \" (available: \" + headers_preview(ds.headers) + \")\");"
    )

    # ---- 6) CLI args: default delimiter auto, allow --delimiter auto/tab/semicolon/pipe etc.
    s = s.replace(
        "  char delimiter = ',';\n  bool no_header = false;\n",
        "  char delimiter = ',';\n  bool delimiter_set = false;\n  bool delimiter_auto = true;\n  bool no_header = false;\n"
    )

    # usage text line
    s = s.replace(
        "  --delimiter <char>         CSV delimiter (default ',')\n",
        "  --delimiter <auto|char>    CSV delimiter (default: auto; supports comma/semicolon/tab/pipe)\n"
    )

    # parse_args delimiter handling
    old_delim = "    else if (k == \"--delimiter\") { std::string v = need(k); a.delimiter = v.empty() ? ',' : v[0]; }\n"
    if old_delim in s:
        new_delim = (
            "    else if (k == \"--delimiter\") {\n"
            "      std::string v = need(k);\n"
            "      a.delimiter_set = true;\n"
            "      std::string t = lower_copy(trim_view(v));\n"
            "      if (t == \"auto\") {\n"
            "        a.delimiter_auto = true;\n"
            "      } else if (t == \"\\\\t\" || t == \"tab\" || t == \"tsv\") {\n"
            "        a.delimiter = '\\t';\n"
            "        a.delimiter_auto = false;\n"
            "      } else if (t == \"semicolon\" || t == \";\") {\n"
            "        a.delimiter = ';';\n"
            "        a.delimiter_auto = false;\n"
            "      } else if (t == \"comma\" || t == \",\") {\n"
            "        a.delimiter = ',';\n"
            "        a.delimiter_auto = false;\n"
            "      } else if (t == \"pipe\" || t == \"|\") {\n"
            "        a.delimiter = '|';\n"
            "        a.delimiter_auto = false;\n"
            "      } else if (!v.empty()) {\n"
            "        a.delimiter = v[0];\n"
            "        a.delimiter_auto = false;\n"
            "      }\n"
            "    }\n"
        )
        s = s.replace(old_delim, new_delim)

    # call site: add args.delimiter_auto
    s = s.replace(
        "      args.delimiter,\n      !args.no_header,\n",
        "      args.delimiter,\n      args.delimiter_auto,\n      !args.no_header,\n"
    )

    # ensure we actually changed something meaningful
    ensure(s != orig, "C++ patch: nothing changed (unexpected)")
    write(p, s)

def patch_python_train_gl1f_py() -> None:
    p = ROOT / "train_gl1f.py"
    s = read(p)
    orig = s

    # 1) argparse default delimiter -> auto + help
    s = re.sub(
        r"ap\.add_argument\(\"--delimiter\", default=.*?\)\n",
        "ap.add_argument(\"--delimiter\", default=\"auto\", help=\"CSV delimiter: auto, comma, semicolon, tab, pipe, or a single character\")\n",
        s
    )

    # 2) Insert delimiter helper functions once (near CSV helpers section)
    if "_autodetect_delimiter_from_lines" not in s:
        insert_at = s.find("# -------------------------\n# Data loading\n# -------------------------\n")
        ensure(insert_at != -1, "Python patch: could not find Data loading section marker")

        helpers = r"""
def _normalize_delimiter_arg(delim: str) -> str:
    d = str(delim or "").strip().lower()
    if not d or d == "auto":
        return "auto"
    if d in (",", "comma"):
        return ","
    if d in (";", "semicolon"):
        return ";"
    if d in ("\\t", "tab", "tsv"):
        return "\t"
    if d in ("|", "pipe"):
        return "|"
    # single-char fallback
    return d[0]

def _autodetect_delimiter_from_lines(lines: List[str], fallback: str = ",") -> str:
    candidates = [",", ";", "\t", "|"]
    best = fallback
    best_mode = 1
    best_freq = 0
    best_penalty = 10**18

    for d in candidates:
        freq: Dict[int, int] = {}
        penalty = 0
        for line in lines:
            if not line:
                continue
            try:
                row = next(csv.reader([line], delimiter=d))
            except Exception:
                continue
            n = len(row)
            freq[n] = freq.get(n, 0) + 1
            for cell in row:
                for od in candidates:
                    if od == d:
                        continue
                    penalty += str(cell).count(od)

        mode_n, mode_f = 1, 0
        for k, v in freq.items():
            if v > mode_f or (v == mode_f and k > mode_n):
                mode_n, mode_f = k, v
        if mode_n < 2:
            continue

        if (mode_f > best_freq) or (mode_f == best_freq and mode_n > best_mode) or (mode_f == best_freq and mode_n == best_mode and penalty < best_penalty):
            best_freq = mode_f
            best_mode = mode_n
            best_penalty = penalty
            best = d
    return best

def _autodetect_csv_delimiter(path: str, limit_lines: int = 20, fallback: str = ",") -> str:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for _ in range(limit_lines):
            line = f.readline()
            if not line:
                break
            line = line.strip("\r\n")
            if not line:
                continue
            lines.append(line)
    if not lines:
        return fallback
    return _autodetect_delimiter_from_lines(lines, fallback=fallback)
"""
        s = s[:insert_at] + helpers + "\n\n" + s[insert_at:]

    # 3) In load_from_csv: allow delimiter='auto' and strip BOM on header[0]
    #    Replace the function signature default delimiter="," if present
    s = s.replace("delimiter: str = \",\",", "delimiter: str = \"auto\",")

    # ensure inside load_from_csv we normalize+auto-detect before reading first row
    if "delim_norm = _normalize_delimiter_arg" not in s:
        pattern = r"def load_from_csv\(\n(.*?\n)\) -> Tuple\[np\.ndarray, np\.ndarray, Dict\[str, Any\]\]:\n"
        m = re.search(pattern, s, flags=re.DOTALL)
        ensure(m is not None, "Python patch: load_from_csv signature not found")
        start = m.end()

        # After docstring comment `t = normalize_task(task)` exists; insert after it
        anchor = "    t = normalize_task(task)\n\n"
        idx = s.find(anchor, start)
        ensure(idx != -1, "Python patch: load_from_csv anchor not found")
        inject = (
            "    delim_norm = _normalize_delimiter_arg(delimiter)\n"
            "    if delim_norm == \"auto\":\n"
            "        delim_norm = _autodetect_csv_delimiter(path)\n"
            "    delimiter = delim_norm\n\n"
        )
        s = s[:idx + len(anchor)] + inject + s[idx + len(anchor):]

    # BOM strip after headers read
    if "lstrip(\"\\ufeff\")" not in s:
        s = s.replace(
            "    if has_header:\n        headers = [str(x or \"\").strip() for x in first]\n",
            "    if has_header:\n        headers = [str(x or \"\").strip() for x in first]\n        if headers and headers[0].startswith(\"\\ufeff\"):\n            headers[0] = headers[0].lstrip(\"\\ufeff\")\n"
        )

    ensure(s != orig, "Python patch: nothing changed (unexpected)")
    write(p, s)

def patch_local_trainer_server_py() -> None:
    p = ROOT / "local_trainer_server.py"
    s = read(p)
    orig = s

    if "_autodetect_delimiter_from_lines" not in s:
        insert = r"""
def _autodetect_delimiter_from_lines(lines: list[str], fallback: str = ",") -> str:
    candidates = [",", ";", "\t", "|"]
    best = fallback
    best_mode = 1
    best_freq = 0
    best_penalty = 10**18

    for d in candidates:
        freq: dict[int, int] = {}
        penalty = 0
        try:
            reader = csv.reader(lines, delimiter=d, quotechar='"', escapechar="\\")
            parsed = list(reader)
        except Exception:
            parsed = []

        for row in parsed:
            n = len(row)
            freq[n] = freq.get(n, 0) + 1
            for cell in row:
                for od in candidates:
                    if od == d:
                        continue
                    penalty += str(cell).count(od)

        mode_n, mode_f = 1, 0
        for k, v in freq.items():
            if v > mode_f or (v == mode_f and k > mode_n):
                mode_n, mode_f = k, v
        if mode_n < 2:
            continue

        if (mode_f > best_freq) or (mode_f == best_freq and mode_n > best_mode) or (mode_f == best_freq and mode_n == best_mode and penalty < best_penalty):
            best_freq = mode_f
            best_mode = mode_n
            best_penalty = penalty
            best = d

    return best
"""
        # insert after _csv_columns_from_file docstring block area (near top utilities)
        anchor = "def _csv_columns_from_file"
        idx = s.find(anchor)
        ensure(idx != -1, "Server patch: could not find _csv_columns_from_file")
        s = s[:idx] + insert + "\n\n" + s[idx:]

    # Update _csv_columns_from_file to auto-detect delimiter for header parsing
    # Replace delimiter=',' line with autodetected delimiter usage
    if "Auto-detect delimiter for header" not in s:
        s = s.replace(
            "        reader = csv.reader(sio, delimiter=\",\", quotechar='\"', escapechar=\"\\\\\")\n",
            "        # Auto-detect delimiter for header\n        header_lines = [text.strip(\"\\r\\n\")]\n        delim = _autodetect_delimiter_from_lines(header_lines, fallback=\",\")\n        reader = csv.reader(sio, delimiter=delim, quotechar='\"', escapechar=\"\\\\\")\n"
        )

    # Strip BOM on first column name (optional nice-to-have)
    if "Strip UTF-8 BOM" not in s:
        s = s.replace(
            "        for row in reader:\n            return row\n",
            "        for row in reader:\n            if row and isinstance(row[0], str) and row[0].startswith(\"\\ufeff\"):\n                row[0] = row[0].lstrip(\"\\ufeff\")\n            return row\n"
        )

    ensure(s != orig, "Server patch: nothing changed (unexpected)")
    write(p, s)

def patch_js_csv_parse() -> None:
    p = ROOT / "src" / "csv_parse.js"
    s = read(p)
    orig = s

    if "detectCSVDelimiter" in s:
        # already patched
        return

    # Replace the file with a robust auto-delim version that preserves API.
    # (keeps parseCSV(text) signature)
    new = r"""/*
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

// Simple CSV parser with quoted fields + delimiter auto-detection.
// Returns: { headers: string[], rows: string[][] }

const _DELIM_CANDS = [",", ";", "\t", "|"];

function splitCsvLine(line, delim) {
  const fields = [];
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

    if (c === delim) { fields.push(cur); cur = ""; continue; }
    fields.push ? null : null;
    cur += c;
  }
  fields.push(cur);
  return fields;
}

export function detectCSVDelimiter(text) {
  const s = String(text || "");
  const lines = s.split("\n").map((x) => x.replace(/\r/g, "")).filter((x) => x.trim().length);
  const sample = lines.slice(0, 20);

  let best = ",";
  let bestMode = 1;
  let bestFreq = 0;
  let bestPenalty = Number.POSITIVE_INFINITY;

  for (const d of _DELIM_CANDS) {
    const freq = new Map();
    let penalty = 0;

    for (const ln of sample) {
      const row = splitCsvLine(ln, d);
      const n = row.length;
      freq.set(n, (freq.get(n) || 0) + 1);
      for (const cell of row) {
        for (const od of _DELIM_CANDS) {
          if (od === d) continue;
          let k = 0;
          for (let i = 0; i < cell.length; i++) if (cell[i] === od) k++;
          penalty += k;
        }
      }
    }

    let modeN = 1, modeF = 0;
    for (const [k, v] of freq.entries()) {
      if (v > modeF || (v === modeF && k > modeN)) { modeN = k; modeF = v; }
    }
    if (modeN < 2) continue;

    if (modeF > bestFreq || (modeF === bestFreq && modeN > bestMode) || (modeF === bestFreq && modeN === bestMode && penalty < bestPenalty)) {
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
  const delim = detectCSVDelimiter(s);

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
    if (c === delim) { pushCell(); continue; }
    if (c === "\r") continue;

    if (c === "\n") { pushCell(); pushRow(); continue; }
    cur += c;
  }

  pushCell();
  if (row.length) pushRow();

  const headers = (rows[0] || []).map((h) => String(h || "").trim());
  if (headers.length && headers[0].startsWith("\uFEFF")) headers[0] = headers[0].replace(/^\uFEFF+/, "");

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
    classes: Object.fromEntries(Array.from(map.entries()).map(([k, v]) => [v, k]))
  };
}
"""
    write(p, new)
    ensure(new != orig, "JS patch: nothing changed (unexpected)")

def main() -> int:
    patch_cpp_train_gl1f_cpp()
    patch_python_train_gl1f_py()
    patch_local_trainer_server_py()
    patch_js_csv_parse()
    print("OK: patched CSV auto-delimiter support across C++/Python/server/JS.")
    print("Next: rebuild C++ trainer: ./build_cpp_trainer.sh && chmod +x train_gl1f_cpp")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

