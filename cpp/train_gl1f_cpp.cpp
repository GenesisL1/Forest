
// train_gl1f_cpp.cpp
// C++17 local trainer for Forest (GL1F/GL1X).
// Intended to be API-compatible with train_gl1f.py for use with local_trainer_server.py.
//
// Build (example):
//   g++ -O3 -std=c++17 -o train_gl1f_cpp cpp/train_gl1f_cpp.cpp
//
// Notes:
// - This is a straightforward port of the Python trainer logic (fixed-depth, complete binary trees).
// - The output model format matches src/local_infer.js (GL1F v1/v2) and optional GL1X footer for metadata/curve.

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <clocale>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <functional>

static constexpr int32_t INT32_MAX_V = 2147483647;
static constexpr int32_t INT32_MIN_V = (-2147483647 - 1);
static constexpr int32_t INT32_SAFE = 2147480000; // match UI
static constexpr int DEFAULT_SCALE_Q = 1000000;

static inline std::string now_iso_utc() {
  using namespace std::chrono;
  auto now = system_clock::now();
  std::time_t t = system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  // 2026-01-24T12:34:56Z
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec);
  return std::string(buf);
}

// Match JavaScript Math.round(x) for finite values: floor(x + 0.5)
static inline int64_t js_round_double(double x) {
  return (int64_t)std::floor(x + 0.5);
}

static inline int32_t clamp_i32(int64_t x) {
  if (x > (int64_t)INT32_MAX_V) return INT32_MAX_V;
  if (x < (int64_t)INT32_MIN_V) return INT32_MIN_V;
  return (int32_t)x;
}

static inline int32_t quantize_to_i32(double x, int64_t scaleQ) {
  // q = clamp_i32(floor(x*scaleQ + 0.5))
  double qd = std::floor(x * (double)scaleQ + 0.5);
  if (!std::isfinite(qd)) return 0;
  return clamp_i32((int64_t)qd);
}

static inline double sigmoid(double z) {
  if (!std::isfinite(z)) return 0.5;
  if (z >= 0.0) {
    double ez = std::exp(-z);
    return 1.0 / (1.0 + ez);
  }
  double ez = std::exp(z);
  return ez / (1.0 + ez);
}

static inline std::string_view ltrim_view(std::string_view s) {
  size_t i = 0;
  while (i < s.size() && std::isspace((unsigned char)s[i])) i++;
  return s.substr(i);
}
static inline std::string_view rtrim_view(std::string_view s) {
  size_t n = s.size();
  while (n > 0 && std::isspace((unsigned char)s[n - 1])) n--;
  return s.substr(0, n);
}
static inline std::string_view trim_view(std::string_view s) {
  return rtrim_view(ltrim_view(s));
}
static inline std::string trim_copy(std::string_view s) {
  auto t = trim_view(s);
  return std::string(t);
}

static inline bool is_strict_number(std::string_view raw) {
  // JS regex: /^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$/
  std::string_view s = trim_view(raw);
  if (s.empty()) return false;
  size_t i = 0;
  if (s[i] == '+' || s[i] == '-') i++;
  bool int_digits = false;
  while (i < s.size() && std::isdigit((unsigned char)s[i])) {
    int_digits = true;
    i++;
  }
  bool frac_digits = false;
  if (i < s.size() && s[i] == '.') {
    i++;
    while (i < s.size() && std::isdigit((unsigned char)s[i])) {
      frac_digits = true;
      i++;
    }
  }
  if (!int_digits && !frac_digits) return false;
  if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
    i++;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
    bool exp_digits = false;
    while (i < s.size() && std::isdigit((unsigned char)s[i])) {
      exp_digits = true;
      i++;
    }
    if (!exp_digits) return false;
  }
  return i == s.size();
}

static inline std::optional<double> parse_double(std::string_view s) {
  s = trim_view(s);
  if (s.empty()) return std::nullopt;

  // NOTE: std::from_chars for floating-point is not implemented reliably across
  // all standard library versions. Use strtod/strtof for portability.
  char stack[128];
  std::string heap;
  const char* cstr = nullptr;
  if (s.size() < sizeof(stack)) {
    std::memcpy(stack, s.data(), s.size());
    stack[s.size()] = '\0';
    cstr = stack;
  } else {
    heap.assign(s.begin(), s.end());
    cstr = heap.c_str();
  }

  errno = 0;
  char* end = nullptr;
  double x = std::strtod(cstr, &end);
  if (end == cstr || (end && *end != '\0')) return std::nullopt;
  if (errno == ERANGE) return std::nullopt;
  if (!std::isfinite(x)) return std::nullopt;
  return x;
}

static inline std::optional<float> parse_float(std::string_view s) {
  s = trim_view(s);
  if (s.empty()) return std::nullopt;

  char stack[128];
  std::string heap;
  const char* cstr = nullptr;
  if (s.size() < sizeof(stack)) {
    std::memcpy(stack, s.data(), s.size());
    stack[s.size()] = '\0';
    cstr = stack;
  } else {
    heap.assign(s.begin(), s.end());
    cstr = heap.c_str();
  }

  errno = 0;
  char* end = nullptr;
  float x = std::strtof(cstr, &end);
  if (end == cstr || (end && *end != '\0')) return std::nullopt;
  if (errno == ERANGE) return std::nullopt;
  if (!std::isfinite((double)x)) return std::nullopt;
  return x;
}

static inline std::optional<int> parse_int(std::string_view s) {
  s = trim_view(s);
  if (s.empty()) return std::nullopt;
  int x = 0;
  auto res = std::from_chars(s.data(), s.data() + s.size(), x);
  if (res.ec != std::errc() || res.ptr != s.data() + s.size()) return std::nullopt;
  return x;
}

static inline std::string lower_copy(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) out.push_back((char)std::tolower((unsigned char)c));
  return out;
}

static inline std::optional<int> parse_binary01(std::string_view raw) {
  std::string s = lower_copy(trim_view(raw));
  if (s.empty()) return std::nullopt;
  if (s == "0" || s == "0.0") return 0;
  if (s == "1" || s == "1.0") return 1;
  if (s == "true" || s == "t" || s == "yes" || s == "y") return 1;
  if (s == "false" || s == "f" || s == "no" || s == "n") return 0;
  if (is_strict_number(s)) {
    auto d = parse_double(s);
    if (d && *d == 0.0) return 0;
    if (d && *d == 1.0) return 1;
  }
  return std::nullopt;
}

static inline std::vector<std::string> split_list(std::string_view s, char delim = ',') {
  std::vector<std::string> out;
  std::string_view t = trim_view(s);
  if (t.empty()) return out;
  // If passed like "[1, 2, 3]" from Python str(list), strip brackets.
  if (t.size() >= 2 && ((t.front() == '[' && t.back() == ']') || (t.front() == '(' && t.back() == ')'))) {
    t = trim_view(t.substr(1, t.size() - 2));
  }
  size_t start = 0;
  while (start <= t.size()) {
    size_t pos = t.find(delim, start);
    if (pos == std::string_view::npos) pos = t.size();
    std::string_view part = trim_view(t.substr(start, pos - start));
    // Strip optional quotes around list items
    if (part.size() >= 2 && ((part.front() == '"' && part.back() == '"') || (part.front() == '\'' && part.back() == '\''))) {
      part = part.substr(1, part.size() - 2);
      part = trim_view(part);
    }
    if (!part.empty()) out.emplace_back(part);
    if (pos == t.size()) break;
    start = pos + 1;
  }
  return out;
}

// --------------- CSV parsing ---------------
// We parse CSV lines into string_views (unquoted) when possible; quoted fields are unescaped into a per-line storage vector.
static void parse_csv_line(std::string_view line, char delim, std::vector<std::string_view>& fields, std::vector<std::string>& quoted_storage) {
  fields.clear();
  quoted_storage.clear();

  // Only append a trailing empty field if the line actually ends with the delimiter (e.g. "a,b,").
  bool ends_with_delim = false;
  {
    size_t end = line.size();
    while (end > 0 && (line[end - 1] == '\r' || line[end - 1] == '\n')) end--;
    if (end > 0 && line[end - 1] == delim) ends_with_delim = true;
    line = line.substr(0, end);
  }

  size_t i = 0;
  const size_t n = line.size();
  while (i < n) {
    if (line[i] == '"') {
      // quoted field
      i++; // skip opening quote
      std::string val;
      while (i < n) {
        char c = line[i];
        if (c == '"') {
          if (i + 1 < n && line[i + 1] == '"') { // escaped quote
            val.push_back('"');
            i += 2;
            continue;
          }
          i++; // closing quote
          break;
        }
        val.push_back(c);
        i++;
      }
      // Skip until delimiter or end (consume any whitespace between quote and delimiter)
      while (i < n && line[i] != delim) i++;
      if (i < n && line[i] == delim) i++;

      quoted_storage.emplace_back(std::move(val));
      fields.emplace_back(quoted_storage.back());
      continue;
    }

    // unquoted field: take slice until delimiter or end
    size_t start = i;
    while (i < n && line[i] != delim) i++;
    size_t end = i;
    if (i < n && line[i] == delim) i++;
    fields.emplace_back(line.substr(start, end - start));
  }

  if (ends_with_delim) fields.emplace_back(std::string_view{});
}


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


static inline bool is_int_string(std::string_view s) {
  s = trim_view(s);
  if (s.empty()) return false;
  size_t i = 0;
  if (s[i] == '+' || s[i] == '-') i++;
  if (i >= s.size()) return false;
  for (; i < s.size(); i++) {
    if (!std::isdigit((unsigned char)s[i])) return false;
  }
  return true;
}

static int find_col_index(const std::vector<std::string>& headers, const std::string& spec) {
  // If spec is int-like, use as index directly.
  std::string t = trim_copy(spec);
  if (t.empty()) return -1;

  if (is_int_string(t)) {
    auto iv = parse_int(t);
    if (!iv) return -1;
    int idx = *iv;
    if (idx < 0) idx = (int)headers.size() + idx; // allow negative indexing like Python? (best-effort)
    if (idx < 0 || idx >= (int)headers.size()) return -1;
    return idx;
  }

  // Exact match
  for (int i = 0; i < (int)headers.size(); i++) {
    if (headers[i] == t) return i;
  }
  // Case-insensitive match
  std::string tl = lower_copy(t);
  for (int i = 0; i < (int)headers.size(); i++) {
    if (lower_copy(headers[i]) == tl) return i;
  }
  return -1;
}

struct LabelLookup {
  std::unordered_map<uint64_t, int> num_bits_to_int;
  std::unordered_map<std::string, int> str_to_int;
  bool has_numeric = false;
  bool has_string = false;

  static uint64_t dbl_bits(double x) {
    uint64_t u = 0;
    static_assert(sizeof(double) == sizeof(uint64_t), "double size");
    std::memcpy(&u, &x, sizeof(double));
    return u;
  }

  void add_label(const std::string& lab, int idx) {
    std::string t = trim_copy(lab);
    if (t.empty()) return;
    if (is_strict_number(t)) {
      auto d = parse_double(t);
      if (d) {
        num_bits_to_int[dbl_bits(*d)] = idx;
        has_numeric = true;
        return;
      }
    }
    str_to_int[t] = idx;
    has_string = true;
  }

  std::optional<int> map_value(std::string_view raw) const {
    std::string_view s = trim_view(raw);
    if (s.empty()) return std::nullopt;

    if (has_numeric && is_strict_number(s)) {
      auto d = parse_double(s);
      if (d) {
        auto it = num_bits_to_int.find(dbl_bits(*d));
        if (it != num_bits_to_int.end()) return it->second;
      }
    }
    if (has_string) {
      std::string key = trim_copy(s);
      auto it = str_to_int.find(key);
      if (it != str_to_int.end()) return it->second;
    }
    return std::nullopt;
  }
};

struct Dataset {
  int n_rows = 0;
  int n_features = 0;
  std::vector<float> X; // row-major: [n_rows*n_features]
  // Single label (regression/binary/multiclass): y holds float values (regression y, or class index as float).
  std::vector<float> y;
  // Multilabel: y_flat holds [n_rows*n_labels] floats (0/1)
  std::vector<float> y_flat;
  int n_labels = 0;

  std::vector<std::string> headers;
  std::vector<std::string> feature_names;
  std::string label_name;
  std::vector<std::string> label_names;
  std::vector<std::string> classes; // for binary/multiclass: class labels; for multilabel: ["0","1"]

  // Stats
  int dropped_rows = 0;
  int dropped_other_label = 0;
  int dropped_bad_feature = 0;
  int dropped_bad_label = 0;
  int dropped_label_missing = 0;
  int dropped_label_invalid = 0;
};

static void infer_label_values_csv(
  const std::string& path,
  int label_index,
  char delimiter,
  bool auto_delimiter,
  bool has_header,
  int limit_rows,
  std::vector<std::string>& out_values,
  bool& out_all_numeric
) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Failed to open CSV for label inference: " + path);

  std::string line;
  std::vector<std::string_view> fields;
  std::vector<std::string> quoted;
  if (has_header) {
    if (!std::getline(f, line)) {
      out_values.clear();
      out_all_numeric = true;
      return;
    }
  }

  std::unordered_map<std::string, int> counts;
  out_all_numeric = true;
  int row_i = 0;
  while (std::getline(f, line)) {
    if (limit_rows > 0 && row_i >= limit_rows) break;
    row_i++;
    if (line.empty()) continue;
    parse_csv_line(line, delimiter, fields, quoted);
    if (label_index < 0 || label_index >= (int)fields.size()) continue;
    std::string_view raw = fields[label_index];
    raw = trim_view(raw);
    if (raw.empty()) continue;

    std::string v = std::string(raw);
    counts[v] += 1;
    if (!is_strict_number(raw)) out_all_numeric = false;
  }

  out_values.clear();
  out_values.reserve(counts.size());
  for (const auto& kv : counts) out_values.push_back(kv.first);

  if (out_all_numeric) {
    struct Pair { double n; std::string s; };
    std::vector<Pair> tmp;
    tmp.reserve(out_values.size());
    for (const auto& s : out_values) {
      auto d = parse_double(s);
      if (d) tmp.push_back({*d, s});
    }
    std::sort(tmp.begin(), tmp.end(), [](const Pair& a, const Pair& b){ return a.n < b.n; });
    out_values.clear();
    for (auto& p : tmp) out_values.push_back(p.s);
  } else {
    std::sort(out_values.begin(), out_values.end(), [](const std::string& a, const std::string& b){
      return a < b; // simple byte-wise; UI uses localeCompare, but good enough
    });
  }
}

// Load CSV with header parsing and task-specific label parsing.
// NOTE: This is built to match the Python/JS trainer behavior used by the UI.
static Dataset load_from_csv(
  const std::string& path,
  const std::string& task,
  const std::string& label_col,
  const std::vector<std::string>& label_cols,
  const std::vector<std::string>& feature_cols,
  char delimiter,
  bool auto_delimiter,
  bool has_header,
  int limit_rows,
  const std::string& neg_label,
  const std::string& pos_label,
  const std::vector<std::string>& class_labels
) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Failed to open CSV: " + path);

  Dataset ds;
  std::string line;
  std::vector<std::string_view> fields;
  std::vector<std::string> quoted;

  if (auto_delimiter) {
    std::vector<std::string> sample;
    sample.reserve(32);
    for (int i = 0; i < 20; ) {
      if (!std::getline(f, line)) break;
      if (line.empty()) continue;
      sample.push_back(line);
      i++;
    }
    delimiter = autodetect_delimiter_from_lines(sample, delimiter);
    f.clear();
    f.seekg(0, std::ios::beg);
    if (delimiter == '\t') std::cerr << "INFO: Auto-detected delimiter '\\t'\n";
    else std::cerr << "INFO: Auto-detected delimiter '" << delimiter << "'\n";
  }

  // Read header / establish column count
  int n_cols = 0;
  if (has_header) {
    if (!std::getline(f, line)) throw std::runtime_error("CSV is empty");
    parse_csv_line(line, delimiter, fields, quoted);
    ds.headers.clear();
    ds.headers.reserve(fields.size());
    for (auto sv : fields) ds.headers.push_back(trim_copy(sv));
    // Strip UTF-8 BOM if present on first header
    if (!ds.headers.empty() && ds.headers[0].size() >= 3 &&
        (unsigned char)ds.headers[0][0] == 0xEF &&
        (unsigned char)ds.headers[0][1] == 0xBB &&
        (unsigned char)ds.headers[0][2] == 0xBF) {
      ds.headers[0].erase(0, 3);
    }
    n_cols = (int)ds.headers.size();
  } else {
    // Peek first data line to count columns, then rewind
    std::streampos pos0 = f.tellg();
    if (!std::getline(f, line)) throw std::runtime_error("CSV is empty");
    parse_csv_line(line, delimiter, fields, quoted);
    n_cols = (int)fields.size();
    ds.headers.resize(n_cols);
    for (int i = 0; i < n_cols; i++) ds.headers[i] = "c" + std::to_string(i);
    f.clear();
    f.seekg(pos0);
  }

  auto is_multilabel = (task == "multilabel_classification");

  std::vector<int> label_idx;
  int single_label_idx = -1;
  if (is_multilabel) {
    if (label_cols.size() < 2) throw std::runtime_error("labelCols required for multilabel_classification");
    std::vector<char> seen((size_t)n_cols, 0);
    for (const auto& spec : label_cols) {
      int idx = find_col_index(ds.headers, spec);
      if (idx < 0) throw std::runtime_error("Bad label col: " + spec + " (available: " + headers_preview(ds.headers) + ")");
      if (!seen[(size_t)idx]) {
        seen[(size_t)idx] = 1;
        label_idx.push_back(idx);
      }
    }
    ds.n_labels = (int)label_idx.size();
    ds.label_names.reserve(ds.n_labels);
    for (int idx : label_idx) ds.label_names.push_back(ds.headers[(size_t)idx]);
  } else {
    if (label_col.empty()) throw std::runtime_error("labelCol required for non-multilabel tasks");
    single_label_idx = find_col_index(ds.headers, label_col);
    if (single_label_idx < 0) throw std::runtime_error("Bad label col: " + label_col + " (available: " + headers_preview(ds.headers) + ")");
    ds.label_name = ds.headers[(size_t)single_label_idx];
  }

  // Feature indices
  std::vector<int> feat_idx;
  if (feature_cols.empty()) {
    // all columns except label(s)
    std::vector<char> is_label((size_t)n_cols, 0);
    if (is_multilabel) {
      for (int idx : label_idx) if (0 <= idx && idx < n_cols) is_label[(size_t)idx] = 1;
    } else {
      is_label[(size_t)single_label_idx] = 1;
    }
    for (int j = 0; j < n_cols; j++) {
      if (!is_label[(size_t)j]) feat_idx.push_back(j);
    }
  } else {
    std::vector<char> is_label((size_t)n_cols, 0);
    if (is_multilabel) {
      for (int idx : label_idx) if (0 <= idx && idx < n_cols) is_label[(size_t)idx] = 1;
    } else {
      is_label[(size_t)single_label_idx] = 1;
    }

    std::vector<char> seen((size_t)n_cols, 0);
    for (const auto& spec : feature_cols) {
      int idx = find_col_index(ds.headers, spec);
      if (idx < 0) throw std::runtime_error("Bad feature col: " + spec + " (available: " + headers_preview(ds.headers) + ")");
      if (is_label[(size_t)idx]) continue;
      if (!seen[(size_t)idx]) {
        seen[(size_t)idx] = 1;
        feat_idx.push_back(idx);
      }
    }
  }
  if (feat_idx.empty()) throw std::runtime_error("Need at least 1 feature column");

  ds.feature_names.reserve(feat_idx.size());
  for (int idx : feat_idx) ds.feature_names.push_back(ds.headers[(size_t)idx]);

  // Prepare label mapping for classification tasks
  LabelLookup lab_map;
  std::vector<std::string> classes;
  if (task == "binary_classification") {
    if (!neg_label.empty() || !pos_label.empty()) {
      std::string neg = trim_copy(neg_label);
      std::string pos = trim_copy(pos_label);
      if (neg.empty() || pos.empty()) throw std::runtime_error("Binary: provide both --neg-label and --pos-label");
      // Ensure distinct (numeric compare if possible)
      bool same = (neg == pos);
      if (!same && is_strict_number(neg) && is_strict_number(pos)) {
        auto dn = parse_double(neg);
        auto dp = parse_double(pos);
        if (dn && dp && *dn == *dp) same = true;
      }
      if (same) throw std::runtime_error("Binary: neg and pos labels must differ");
      classes = {neg, pos};
    } else {
      bool all_numeric = true;
      infer_label_values_csv(path, single_label_idx, delimiter, auto_delimiter, has_header, limit_rows, classes, all_numeric);
      if (classes.size() < 2) throw std::runtime_error("Binary: need at least 2 label values");
      if (classes.size() > 2) throw std::runtime_error("Binary: more than 2 labels; provide --neg-label/--pos-label");
    }
    // Build map: classes[0]->0, classes[1]->1
    lab_map.add_label(classes[0], 0);
    lab_map.add_label(classes[1], 1);
    ds.classes = classes;
  } else if (task == "multiclass_classification") {
    if (!class_labels.empty()) {
      for (auto& s : class_labels) {
        auto t = trim_copy(s);
        if (!t.empty()) classes.push_back(t);
      }
      if (classes.size() < 2) throw std::runtime_error("Multiclass: need >=2 --class-labels");
    } else {
      bool all_numeric = true;
      infer_label_values_csv(path, single_label_idx, delimiter, auto_delimiter, has_header, limit_rows, classes, all_numeric);
      if (classes.size() < 2) throw std::runtime_error("Multiclass: need at least 2 classes");
    }
    // Build map
    for (int i = 0; i < (int)classes.size(); i++) lab_map.add_label(classes[i], i);
    ds.classes = classes;
  } else if (task == "multilabel_classification") {
    ds.classes = {"0", "1"};
  }

  // Parse rows
  ds.X.clear();
  ds.y.clear();
  ds.y_flat.clear();
  ds.dropped_rows = 0;
  int row_i = 0;

  // Reserve rough
  ds.X.reserve(1024 * feat_idx.size());

  while (std::getline(f, line)) {
    if (limit_rows > 0 && row_i >= limit_rows) break;
    row_i++;
    if (line.empty()) continue;
    parse_csv_line(line, delimiter, fields, quoted);

    bool ok = true;

    if (is_multilabel) {
      // Parse labels
      std::vector<float> yy;
      yy.reserve(label_idx.size());
      for (int li : label_idx) {
        if (li < 0 || li >= (int)fields.size()) { ok = false; ds.dropped_label_missing++; break; }
        std::string_view raw = trim_view(fields[li]);
        if (raw.empty()) { ok = false; ds.dropped_label_missing++; break; }
        auto b = parse_binary01(raw);
        if (!b) { ok = false; ds.dropped_label_invalid++; break; }
        yy.push_back(*b ? 1.0f : 0.0f);
      }
      if (!ok) { ds.dropped_rows++; continue; }

      // Parse features
      std::vector<float> xx;
      xx.reserve(feat_idx.size());
      for (int fi : feat_idx) {
        if (fi < 0 || fi >= (int)fields.size()) { ok = false; ds.dropped_bad_feature++; break; }
        auto v = parse_float(fields[fi]);
        if (!v) { ok = false; ds.dropped_bad_feature++; break; }
        xx.push_back(*v);
      }
      if (!ok) { ds.dropped_rows++; continue; }

      // Commit row
      ds.X.insert(ds.X.end(), xx.begin(), xx.end());
      ds.y_flat.insert(ds.y_flat.end(), yy.begin(), yy.end());
      continue;
    }

    // Single-label tasks
    float yv = 0.0f;
    if (single_label_idx < 0) { ok = false; }
    if (ok) {
      if (single_label_idx >= (int)fields.size()) { ok = false; ds.dropped_bad_label++; }
      else {
        std::string_view raw = trim_view(fields[single_label_idx]);
        if (raw.empty()) { ok = false; ds.dropped_bad_label++; }
        else if (task == "regression") {
          auto v = parse_float(raw);
          if (!v) { ok = false; ds.dropped_bad_label++; }
          else yv = *v;
        } else {
          // classification
          auto mapped = lab_map.map_value(raw);
          if (!mapped) { ok = false; ds.dropped_other_label++; }
          else yv = (float)(*mapped);
        }
      }
    }
    if (!ok) { ds.dropped_rows++; continue; }

    // Parse features
    std::vector<float> xx;
    xx.reserve(feat_idx.size());
    for (int fi : feat_idx) {
      if (fi < 0 || fi >= (int)fields.size()) { ok = false; ds.dropped_bad_feature++; break; }
      auto v = parse_float(fields[fi]);
      if (!v) { ok = false; ds.dropped_bad_feature++; break; }
      xx.push_back(*v);
    }
    if (!ok) { ds.dropped_rows++; continue; }

    ds.X.insert(ds.X.end(), xx.begin(), xx.end());
    ds.y.push_back(yv);
  }

  // finalize
  ds.n_rows = is_multilabel ? (int)(ds.y_flat.size() / (size_t)ds.n_labels) : (int)ds.y.size();
  ds.n_features = (int)feat_idx.size();
  if (ds.n_rows <= 0) throw std::runtime_error("No valid rows parsed from CSV (all rows dropped?)");
  if ((int)ds.X.size() != ds.n_rows * ds.n_features) throw std::runtime_error("Internal error: X shape mismatch");
  if (is_multilabel) {
    if ((int)ds.y_flat.size() != ds.n_rows * ds.n_labels) throw std::runtime_error("Internal error: y_flat shape mismatch");
  }
  return ds;
}

// --------------- RNG (xorshift32, JS compatible) ---------------
struct XorShift32 {
  uint32_t x;
  explicit XorShift32(uint32_t seed) {
    x = seed & 0xFFFFFFFFu;
    if (x == 0) x = 123456789u;
  }
  uint32_t next_u32() {
    uint32_t y = x;
    y ^= (y << 13);
    y ^= (y >> 17);
    y ^= (y << 5);
    x = y;
    return x;
  }
};

static std::vector<int> shuffled_indices(int n, int seed) {
  XorShift32 rng((uint32_t)seed);
  std::vector<uint32_t> idx((size_t)n);
  for (uint32_t i = 0; i < (uint32_t)n; i++) idx[i] = i;
  for (int i = n - 1; i > 0; i--) {
    uint32_t j = rng.next_u32() % (uint32_t)(i + 1);
    uint32_t tmp = idx[(size_t)i];
    idx[(size_t)i] = idx[(size_t)j];
    idx[(size_t)j] = tmp;
  }
  std::vector<int> out((size_t)n);
  for (int i = 0; i < n; i++) out[(size_t)i] = (int)idx[(size_t)i];
  return out;
}

static void split_idx(const std::vector<int>& idx, double frac_train, double frac_val,
                      std::vector<int>& train, std::vector<int>& val, std::vector<int>& test) {
  int n = (int)idx.size();
  int n_train = (int)std::floor((double)n * frac_train);
  int n_val = (int)std::floor((double)n * frac_val);
  if (n_train < 1) n_train = 1;
  if (n_val < 1) n_val = 1;
  if (n_train + n_val >= n) {
    n_val = std::max(1, n - n_train - 1);
  }
  int n_test = std::max(1, n - n_train - n_val);
  train.assign(idx.begin(), idx.begin() + n_train);
  val.assign(idx.begin() + n_train, idx.begin() + n_train + n_val);
  test.assign(idx.begin() + n_train + n_val, idx.begin() + n_train + n_val + n_test);
}

static void split_idx_stratified_by_class(
  const std::vector<int>& idx,
  const std::vector<int>& yK,
  int n_classes,
  double frac_train,
  double frac_val,
  std::vector<int>& train,
  std::vector<int>& val,
  std::vector<int>& test
) {
  std::vector<std::vector<int>> buckets((size_t)n_classes);
  for (int r : idx) {
    int k = 0;
    if (r >= 0 && r < (int)yK.size()) k = yK[(size_t)r];
    if (k < 0 || k >= n_classes) k = 0;
    buckets[(size_t)k].push_back(r);
  }

  train.clear(); val.clear(); test.clear();
  for (int k = 0; k < n_classes; k++) {
    auto& arr = buckets[(size_t)k];
    int n = (int)arr.size();
    if (n <= 0) continue;
    int n_train = (int)std::floor((double)n * frac_train);
    int n_val = (int)std::floor((double)n * frac_val);
    if (n_train + n_val >= n) {
      n_val = std::max(0, n - n_train - 1);
      if (n_train + n_val >= n) {
        n_train = std::max(0, n - n_val - 1);
      }
    }
    train.insert(train.end(), arr.begin(), arr.begin() + n_train);
    val.insert(val.end(), arr.begin() + n_train, arr.begin() + n_train + n_val);
    test.insert(test.end(), arr.begin() + n_train + n_val, arr.end());
  }

  if (train.empty() || val.empty() || test.empty()) {
    split_idx(idx, frac_train, frac_val, train, val, test);
  }
}

// --------------- Metrics ---------------
static double mse_q(const std::vector<double>& yQ, const std::vector<double>& predQ, const std::vector<int>& indices, int scaleQ) {
  if (indices.empty()) return std::numeric_limits<double>::quiet_NaN();
  double sum = 0.0;
  for (int r : indices) {
    double diff = (yQ[(size_t)r] - predQ[(size_t)r]) / (double)scaleQ;
    sum += diff * diff;
  }
  return sum / (double)indices.size();
}

static std::pair<double,double> logloss_acc_binary(
  const std::vector<uint8_t>& y01,
  const std::vector<double>& predQ,
  const std::vector<int>& indices,
  int scaleQ,
  const std::vector<float>* w_row
) {
  if (indices.empty()) return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  const double eps = 1e-12;
  double w_sum = 0.0;
  double loss_sum = 0.0;
  double correct_sum = 0.0;
  for (int r : indices) {
    double y = (y01[(size_t)r] ? 1.0 : 0.0);
    double logit = predQ[(size_t)r] / (double)scaleQ;
    double p = sigmoid(logit);
    p = std::min(std::max(p, eps), 1.0 - eps);
    double w = 1.0;
    if (w_row) {
      double ww = (double)(*w_row)[(size_t)r];
      if (ww > 0) w = ww; else w = 0.0;
    }
    w_sum += w;
    loss_sum += w * (-(y * std::log(p) + (1.0 - y) * std::log(1.0 - p)));
    double pred = (p >= 0.5) ? 1.0 : 0.0;
    correct_sum += w * ((pred == y) ? 1.0 : 0.0);
  }
  if (!(w_sum > 0)) w_sum = 1.0;
  return {loss_sum / w_sum, correct_sum / w_sum};
}

static void softmax_probs_inplace(
  const std::vector<double>& predQ_flat,
  int n_rows,
  int n_classes,
  int scaleQ,
  std::vector<float>& out_prob // size n_rows*n_classes
) {
  out_prob.assign((size_t)n_rows * (size_t)n_classes, 0.0f);
  for (int r = 0; r < n_rows; r++) {
    // find max for stability
    double max_z = -std::numeric_limits<double>::infinity();
    size_t base = (size_t)r * (size_t)n_classes;
    for (int k = 0; k < n_classes; k++) {
      double z = predQ_flat[base + (size_t)k] / (double)scaleQ;
      if (z > max_z) max_z = z;
    }
    double sum = 0.0;
    for (int k = 0; k < n_classes; k++) {
      double z = predQ_flat[base + (size_t)k] / (double)scaleQ - max_z;
      double e = std::exp(z);
      sum += e;
      out_prob[base + (size_t)k] = (float)e;
    }
    if (!(sum > 0)) sum = 1.0;
    for (int k = 0; k < n_classes; k++) {
      out_prob[base + (size_t)k] = (float)((double)out_prob[base + (size_t)k] / sum);
    }
  }
}

static std::pair<double,double> logloss_acc_multiclass(
  const std::vector<int>& yK,
  const std::vector<float>& prob_flat, // n_rows*n_classes
  const std::vector<int>& indices,
  int n_classes,
  const std::vector<float>* w_row
) {
  if (indices.empty()) return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  const double eps = 1e-12;
  double w_sum = 0.0;
  double loss_sum = 0.0;
  double correct_sum = 0.0;
  for (int r : indices) {
    int y = yK[(size_t)r];
    if (y < 0 || y >= n_classes) y = 0;
    size_t base = (size_t)r * (size_t)n_classes;

    // argmax
    int pred = 0;
    float bestp = prob_flat[base];
    for (int k = 1; k < n_classes; k++) {
      float pk = prob_flat[base + (size_t)k];
      if (pk > bestp) { bestp = pk; pred = k; }
    }
    double correct = (pred == y) ? 1.0 : 0.0;

    double py = (double)prob_flat[base + (size_t)y];
    py = std::min(std::max(py, eps), 1.0 - eps);

    double w = 1.0;
    if (w_row) {
      double ww = (double)(*w_row)[(size_t)r];
      if (ww > 0) w = ww; else w = 0.0;
    }
    w_sum += w;
    loss_sum += w * (-std::log(py));
    correct_sum += w * correct;
  }
  if (!(w_sum > 0)) w_sum = 1.0;
  return {loss_sum / w_sum, correct_sum / w_sum};
}

static std::pair<double,double> logloss_acc_multilabel(
  const std::vector<float>& y_flat,
  const std::vector<double>& predQ_flat,
  const std::vector<int>& indices,
  int n_labels,
  int scaleQ,
  const std::vector<float>* pos_w, // size n_labels
  double w_scale
) {
  if (indices.empty()) return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  const double eps = 1e-12;
  double w_sum = 0.0;
  double loss_sum = 0.0;
  double correct_sum = 0.0;

  for (int r : indices) {
    size_t base = (size_t)r * (size_t)n_labels;
    for (int k = 0; k < n_labels; k++) {
      double y = (y_flat[base + (size_t)k] >= 0.5f) ? 1.0 : 0.0;
      double z = predQ_flat[base + (size_t)k] / (double)scaleQ;
      double p = sigmoid(z);
      p = std::min(std::max(p, eps), 1.0 - eps);

      double w = 1.0;
      if (pos_w) {
        double pw = (double)(*pos_w)[(size_t)k];
        if (y >= 0.5) w = pw;
        w *= w_scale;
      }
      if (!(w > 0)) w = 0.0;
      w_sum += w;
      loss_sum += w * (-(y * std::log(p) + (1.0 - y) * std::log(1.0 - p)));
      double pred = (p >= 0.5) ? 1.0 : 0.0;
      correct_sum += w * ((pred == y) ? 1.0 : 0.0);
    }
  }
  if (!(w_sum > 0)) w_sum = 1.0;
  return {loss_sum / w_sum, correct_sum / w_sum};
}

// --------------- LR schedule ---------------
struct LRSchedule {
  std::string mode = "none"; // none|plateau|piecewise
  double lr_base = 0.05;
  int max_trees = 200;

  // plateau
  int plateau_patience = 0;
  double plateau_factor = 1.0;
  double lr_min = 0.0;
  int plateau_since = 0;
  double lr_cur = 0.05;

  // piecewise
  struct Segment { int start; int end; double lr; };
  std::vector<Segment> segments;
  double piecewise_last_lr = std::numeric_limits<double>::quiet_NaN();

  double lr_for_iter(int t) {
    if (mode == "piecewise" && !segments.empty()) {
      double lr = lr_base;
      for (auto& seg : segments) {
        if (t < seg.start) break;
        if (seg.start <= t && t <= seg.end) { lr = seg.lr; break; }
      }
      piecewise_last_lr = lr;
      return lr;
    }
    if (mode == "plateau") return lr_cur;
    return lr_base;
  }

  void after_metric(bool improved) {
    if (mode != "plateau") return;
    plateau_since = improved ? 0 : (plateau_since + 1);
    if (plateau_since >= plateau_patience) {
      lr_cur = lr_cur * plateau_factor;
      if (lr_min > 0 && lr_cur < lr_min) lr_cur = lr_min;
      if (lr_cur < 1e-12) lr_cur = 1e-12;
      plateau_since = 0;
    }
  }
};

static LRSchedule make_lr_schedule(
  const std::string& mode_raw,
  double lr_base,
  int max_trees,
  int lr_patience,
  double lr_drop_pct,
  double lr_min,
  const std::string& lr_segments
) {
  LRSchedule sched;
  sched.mode = "none";
  sched.lr_base = lr_base;
  sched.max_trees = max_trees;
  sched.lr_cur = lr_base;

  std::string mode = lower_copy(trim_view(mode_raw));
  if (mode != "none" && mode != "plateau" && mode != "piecewise") mode = "none";

  if (mode == "plateau") {
    int p = lr_patience;
    if (p < 1) {
      p = (int)std::round((double)max_trees * 0.1);
      p = std::max(5, std::min(100, p));
    }
    double drop = lr_drop_pct;
    double factor = 1.0 - (drop / 100.0);
    if (!(factor > 0.0 && factor < 1.0)) factor = 0.9;
    if (lr_min < 0) lr_min = 0.0;

    sched.mode = "plateau";
    sched.plateau_patience = p;
    sched.plateau_factor = factor;
    sched.lr_min = lr_min;
    sched.lr_cur = lr_base;
    return sched;
  }

  if (mode == "piecewise") {
    std::string segs_raw = trim_copy(lr_segments);
    if (segs_raw.empty()) return sched;
    std::vector<LRSchedule::Segment> segs;
    for (auto& part : split_list(segs_raw, ',')) {
      auto bits = split_list(part, ':');
      if (bits.size() != 3) throw std::runtime_error("Bad segment '" + part + "'. Use start:end:lr");
      int start = std::stoi(bits[0]);
      int end = std::stoi(bits[1]);
      double lr = std::stod(bits[2]);
      if (start < 1 || end < start || !(lr > 0)) throw std::runtime_error("Bad segment '" + part + "'");
      segs.push_back({start, end, lr});
    }
    std::sort(segs.begin(), segs.end(), [](auto& a, auto& b){ return (a.start < b.start) || (a.start == b.start && a.end < b.end); });
    for (size_t i = 1; i < segs.size(); i++) {
      if (segs[i].start <= segs[i - 1].end) throw std::runtime_error("LR schedule ranges overlap");
    }
    sched.mode = "piecewise";
    sched.segments = std::move(segs);
    return sched;
  }

  return sched;
}

static int choose_scale_q(const std::string& task, double max_abs_x, double max_abs_y) {
  int lim_x = DEFAULT_SCALE_Q;
  if (max_abs_x > 0) lim_x = (int)((double)INT32_SAFE / max_abs_x);
  int lim_y = DEFAULT_SCALE_Q;
  if (task == "regression" && max_abs_y > 0) lim_y = (int)((double)INT32_SAFE / max_abs_y);
  int scale = std::min(DEFAULT_SCALE_Q, std::min(lim_x, lim_y));
  if (scale < 1) scale = 1;
  if ((uint64_t)scale > 0xFFFFFFFFull) scale = (int)0xFFFFFFFFull;
  return scale;
}

static double max_abs_x(const std::vector<float>& X) {
  double m = 0.0;
  for (float v : X) {
    double a = std::fabs((double)v);
    if (a > m) m = a;
  }
  return m;
}

static double max_abs_y(const std::vector<float>& y) {
  double m = 0.0;
  for (float v : y) {
    double a = std::fabs((double)v);
    if (a > m) m = a;
  }
  return m;
}

// --------------- Tree structures and training ---------------
struct Tree {
  std::vector<uint16_t> feat; // internal nodes
  std::vector<int32_t> thr;   // internal nodes (Q units)
  std::vector<int32_t> leaf;  // leaves (Q units)
};

static std::vector<int> sample_features(int n_features, int k, XorShift32& rng) {
  if (k >= n_features) {
    std::vector<int> all((size_t)n_features);
    for (int i = 0; i < n_features; i++) all[(size_t)i] = i;
    return all;
  }
  std::vector<char> used((size_t)n_features, 0);
  std::vector<int> out;
  out.reserve((size_t)k);
  while ((int)out.size() < k) {
    int f = (int)(rng.next_u32() % (uint32_t)n_features);
    if (!used[(size_t)f]) {
      used[(size_t)f] = 1;
      out.push_back(f);
    }
  }
  return out;
}

static void compute_feat_min_max(
  const std::vector<float>& X,
  int n_rows,
  int n_features,
  const std::vector<int>& rows,
  std::vector<float>& feat_min,
  std::vector<float>& feat_max
) {
  feat_min.assign((size_t)n_features, std::numeric_limits<float>::infinity());
  feat_max.assign((size_t)n_features, -std::numeric_limits<float>::infinity());
  for (int r : rows) {
    size_t base = (size_t)r * (size_t)n_features;
    for (int f = 0; f < n_features; f++) {
      float v = X[base + (size_t)f];
      if (v < feat_min[(size_t)f]) feat_min[(size_t)f] = v;
      if (v > feat_max[(size_t)f]) feat_max[(size_t)f] = v;
    }
  }
  for (int f = 0; f < n_features; f++) {
    if (!std::isfinite((double)feat_min[(size_t)f])) feat_min[(size_t)f] = 0.0f;
    if (!std::isfinite((double)feat_max[(size_t)f])) feat_max[(size_t)f] = 0.0f;
  }
}

static std::vector<std::vector<float>> compute_quantile_thresholds(
  const std::vector<float>& X,
  int n_rows,
  int n_features,
  const std::vector<int>& train_rows,
  const std::vector<float>& feat_min,
  const std::vector<float>& feat_range,
  int bins,
  int sample_n0
) {
  int sample_n = std::min((int)train_rows.size(), std::max(256, sample_n0));
  std::vector<int> rows(train_rows.begin(), train_rows.begin() + sample_n);

  std::vector<std::vector<float>> q_thr((size_t)n_features);
  for (int f = 0; f < n_features; f++) {
    float r = feat_range[(size_t)f];
    if (!(r > 0)) {
      q_thr[(size_t)f].clear();
      continue;
    }
    std::vector<float> vals;
    vals.reserve((size_t)sample_n);
    for (int rr : rows) {
      vals.push_back(X[(size_t)rr * (size_t)n_features + (size_t)f]);
    }
    std::sort(vals.begin(), vals.end());
    int nV = (int)vals.size();
    if (nV <= 0) { q_thr[(size_t)f].clear(); continue; }

    int thr_n = std::max(1, bins - 1);
    std::vector<float> thr((size_t)thr_n, 0.0f);
    float prev = -std::numeric_limits<float>::infinity();
    for (int j = 1; j < bins; j++) {
      double q = (double)j / (double)bins;
      double pos = q * (double)(nV - 1);
      int lo = (int)std::floor(pos);
      int hi = std::min(nV - 1, lo + 1);
      double t = (double)vals[(size_t)lo];
      if (hi != lo) t += ((double)vals[(size_t)hi] - (double)vals[(size_t)lo]) * (pos - (double)lo);
      if (!std::isfinite(t)) {
        t = (double)feat_min[(size_t)f] + (double)feat_range[(size_t)f] * ((double)j / (double)bins);
      }
      float tf = (float)t;
      if (tf < prev) tf = prev;
      thr[(size_t)(j - 1)] = tf;
      prev = tf;
    }
    q_thr[(size_t)f] = std::move(thr);
  }
  return q_thr;
}

static Tree build_tree_regression(
  const std::vector<float>& X,
  int n_rows,
  int n_features,
  const std::vector<int>& train_samples,
  const std::vector<float>& residual, // float residuals (not Q)
  const std::vector<float>& feat_min,
  const std::vector<float>& feat_range,
  int depth,
  int min_leaf,
  double lr,
  int scaleQ,
  XorShift32& rng,
  int bins,
  const std::string& binning,
  const std::vector<std::vector<float>>* q_thr // optional
) {
  const int BINS = std::max(8, bins);
  const bool is_quantile = (binning == "quantile");

  int pow2 = 1 << depth;
  int internal = pow2 - 1;
  Tree tree;
  tree.feat.assign((size_t)internal, 0);
  tree.thr.assign((size_t)internal, 0);
  tree.leaf.assign((size_t)pow2, 0);

  auto compute_leaf_q = [&](const std::vector<int>& samples) -> int32_t {
    if (samples.empty()) return 0;
    double sum = 0.0;
    for (int r : samples) sum += (double)residual[(size_t)r];
    double mean = sum / (double)samples.size();
    double w = (double)lr * mean;
    return clamp_i32(js_round_double(w * (double)scaleQ));
  };

  std::function<void(int,int,const std::vector<int>&)> fill_forced = [&](int node_idx, int level, const std::vector<int>& /*samples*/) {
    int32_t leaf_q = compute_leaf_q(train_samples); // fallback if samples unused
    // NOTE: We pass the real samples in caller; use this version only when caller already computed leaf_q.
    (void)node_idx; (void)level; (void)leaf_q;
  };

  std::function<void(int,int,int32_t)> fill_forced_leaf = [&](int node_idx, int level, int32_t leaf_q) {
    if (level == depth) {
      int leaf_idx = node_idx - internal;
      if (leaf_idx >= 0 && leaf_idx < pow2) tree.leaf[(size_t)leaf_idx] = leaf_q;
      return;
    }
    tree.feat[(size_t)node_idx] = 0;
    tree.thr[(size_t)node_idx] = INT32_MAX_V;
    fill_forced_leaf(node_idx * 2 + 1, level + 1, leaf_q);
    fill_forced_leaf(node_idx * 2 + 2, level + 1, leaf_q);
  };

  std::function<void(int,int,const std::vector<int>&)> node_split = [&](int node_idx, int level, const std::vector<int>& samples) {
    if (samples.empty()) {
      fill_forced_leaf(node_idx, level, 0);
      return;
    }
    if (level >= depth) {
      int leaf_idx = node_idx - internal;
      tree.leaf[(size_t)leaf_idx] = compute_leaf_q(samples);
      return;
    }
    if ((int)samples.size() < 2 * min_leaf) {
      fill_forced_leaf(node_idx, level, compute_leaf_q(samples));
      return;
    }

    int k = (int)std::round(std::sqrt((double)n_features));
    if (k < 1) k = 1;
    auto cand = sample_features(n_features, k, rng);

    double best_sse = std::numeric_limits<double>::infinity();
    int best_f = -1;
    int32_t best_thr_q = 0;

    std::vector<int> cnt((size_t)BINS);
    std::vector<double> sum_r((size_t)BINS);
    std::vector<double> sum2_r((size_t)BINS);

    for (int f : cand) {
      float r = feat_range[(size_t)f];
      if (!(r > 0)) continue;

      // reset
      std::fill(cnt.begin(), cnt.end(), 0);
      std::fill(sum_r.begin(), sum_r.end(), 0.0);
      std::fill(sum2_r.begin(), sum2_r.end(), 0.0);

      const std::vector<float>* thr_arr = nullptr;
      if (is_quantile) {
        if (!q_thr) continue;
        const auto& v = (*q_thr)[(size_t)f];
        if ((int)v.size() != (BINS - 1)) continue;
        thr_arr = &v;
      }

      // build hist
      for (int rr : samples) {
        float x = X[(size_t)rr * (size_t)n_features + (size_t)f];
        int b = 0;
        if (is_quantile) {
          auto it = std::lower_bound(thr_arr->begin(), thr_arr->end(), x);
          b = (int)(it - thr_arr->begin());
        } else {
          double bf = std::floor(((double)x - (double)feat_min[(size_t)f]) / (double)r * (double)BINS);
          if (!std::isfinite(bf)) bf = 0.0;
          b = (int)bf;
          if (b < 0) b = 0;
          if (b > BINS - 1) b = BINS - 1;
        }
        double rv = (double)residual[(size_t)rr];
        cnt[(size_t)b] += 1;
        sum_r[(size_t)b] += rv;
        sum2_r[(size_t)b] += rv * rv;
      }

      int total_count = 0;
      double total_sum = 0.0;
      double total_sum2 = 0.0;
      for (int b = 0; b < BINS; b++) {
        total_count += cnt[(size_t)b];
        total_sum += sum_r[(size_t)b];
        total_sum2 += sum2_r[(size_t)b];
      }
      if (total_count < 2 * min_leaf) continue;

      // scan splits
      int left_cnt = 0;
      double left_sum = 0.0;
      double left_sum2 = 0.0;
      for (int b = 0; b < BINS - 1; b++) {
        left_cnt += cnt[(size_t)b];
        left_sum += sum_r[(size_t)b];
        left_sum2 += sum2_r[(size_t)b];
        int right_cnt = total_count - left_cnt;
        if (left_cnt < min_leaf || right_cnt < min_leaf) continue;
        double right_sum = total_sum - left_sum;
        double right_sum2 = total_sum2 - left_sum2;
        double left_sse = left_sum2 - (left_sum * left_sum) / (double)left_cnt;
        double right_sse = right_sum2 - (right_sum * right_sum) / (double)right_cnt;
        double sse = left_sse + right_sse;
        if (sse < best_sse) {
          best_sse = sse;
          best_f = f;
          double thr_f = 0.0;
          if (is_quantile) thr_f = (double)(*thr_arr)[(size_t)b];
          else thr_f = (double)feat_min[(size_t)f] + (double)r * ((double)(b + 1) / (double)BINS);
          best_thr_q = clamp_i32(js_round_double(thr_f * (double)scaleQ));
        }
      }
    }

    if (best_f < 0) {
      fill_forced_leaf(node_idx, level, compute_leaf_q(samples));
      return;
    }

    // partition
    std::vector<int> left;
    std::vector<int> right;
    left.reserve(samples.size());
    right.reserve(samples.size());
    for (int rr : samples) {
      float x = X[(size_t)rr * (size_t)n_features + (size_t)best_f];
      int32_t xq = quantize_to_i32((double)x, (int64_t)scaleQ);
      if (xq > best_thr_q) right.push_back(rr);
      else left.push_back(rr);
    }

    if ((int)left.size() < min_leaf || (int)right.size() < min_leaf) {
      fill_forced_leaf(node_idx, level, compute_leaf_q(samples));
      return;
    }

    tree.feat[(size_t)node_idx] = (uint16_t)best_f;
    tree.thr[(size_t)node_idx] = best_thr_q;
    node_split(node_idx * 2 + 1, level + 1, left);
    node_split(node_idx * 2 + 2, level + 1, right);
  };

  node_split(0, 0, train_samples);
  return tree;
}

static Tree build_tree_binary(
  const std::vector<float>& X,
  int n_rows,
  int n_features,
  const std::vector<int>& train_samples,
  const std::vector<float>& grad,
  const std::vector<float>& hess,
  const std::vector<float>& feat_min,
  const std::vector<float>& feat_range,
  int depth,
  int min_leaf,
  double lr,
  int scaleQ,
  XorShift32& rng,
  int bins,
  const std::string& binning,
  const std::vector<std::vector<float>>* q_thr
) {
  const int BINS = std::max(8, bins);
  const bool is_quantile = (binning == "quantile");
  const double LAMBDA = 1.0;

  int pow2 = 1 << depth;
  int internal = pow2 - 1;
  Tree tree;
  tree.feat.assign((size_t)internal, 0);
  tree.thr.assign((size_t)internal, 0);
  tree.leaf.assign((size_t)pow2, 0);

  auto compute_leaf_q = [&](const std::vector<int>& samples) -> int32_t {
    if (samples.empty()) return 0;
    double G = 0.0, H = 0.0;
    for (int r : samples) {
      G += (double)grad[(size_t)r];
      H += (double)hess[(size_t)r];
    }
    double w = (double)lr * (G / (H + LAMBDA));
    return clamp_i32(js_round_double(w * (double)scaleQ));
  };

  std::function<void(int,int,int32_t)> fill_forced_leaf = [&](int node_idx, int level, int32_t leaf_q) {
    if (level == depth) {
      int leaf_idx = node_idx - internal;
      if (leaf_idx >= 0 && leaf_idx < pow2) tree.leaf[(size_t)leaf_idx] = leaf_q;
      return;
    }
    tree.feat[(size_t)node_idx] = 0;
    tree.thr[(size_t)node_idx] = INT32_MAX_V;
    fill_forced_leaf(node_idx * 2 + 1, level + 1, leaf_q);
    fill_forced_leaf(node_idx * 2 + 2, level + 1, leaf_q);
  };

  std::function<void(int,int,const std::vector<int>&)> node_split = [&](int node_idx, int level, const std::vector<int>& samples) {
    if (samples.empty()) {
      fill_forced_leaf(node_idx, level, 0);
      return;
    }
    if (level >= depth) {
      int leaf_idx = node_idx - internal;
      tree.leaf[(size_t)leaf_idx] = compute_leaf_q(samples);
      return;
    }
    if ((int)samples.size() < 2 * min_leaf) {
      fill_forced_leaf(node_idx, level, compute_leaf_q(samples));
      return;
    }

    int k = (int)std::round(std::sqrt((double)n_features));
    if (k < 1) k = 1;
    auto cand = sample_features(n_features, k, rng);

    double best_gain = 0.0;
    int best_f = -1;
    int32_t best_thr_q = 0;

    std::vector<int> cnt((size_t)BINS);
    std::vector<double> sum_g((size_t)BINS);
    std::vector<double> sum_h((size_t)BINS);

    for (int f : cand) {
      float r = feat_range[(size_t)f];
      if (!(r > 0)) continue;

      std::fill(cnt.begin(), cnt.end(), 0);
      std::fill(sum_g.begin(), sum_g.end(), 0.0);
      std::fill(sum_h.begin(), sum_h.end(), 0.0);

      const std::vector<float>* thr_arr = nullptr;
      if (is_quantile) {
        if (!q_thr) continue;
        const auto& v = (*q_thr)[(size_t)f];
        if ((int)v.size() != (BINS - 1)) continue;
        thr_arr = &v;
      }

      for (int rr : samples) {
        float x = X[(size_t)rr * (size_t)n_features + (size_t)f];
        int b = 0;
        if (is_quantile) {
          auto it = std::lower_bound(thr_arr->begin(), thr_arr->end(), x);
          b = (int)(it - thr_arr->begin());
        } else {
          double bf = std::floor(((double)x - (double)feat_min[(size_t)f]) / (double)r * (double)BINS);
          if (!std::isfinite(bf)) bf = 0.0;
          b = (int)bf;
          if (b < 0) b = 0;
          if (b > BINS - 1) b = BINS - 1;
        }
        cnt[(size_t)b] += 1;
        sum_g[(size_t)b] += (double)grad[(size_t)rr];
        sum_h[(size_t)b] += (double)hess[(size_t)rr];
      }

      int total_count = 0;
      double total_g = 0.0, total_h = 0.0;
      for (int b = 0; b < BINS; b++) {
        total_count += cnt[(size_t)b];
        total_g += sum_g[(size_t)b];
        total_h += sum_h[(size_t)b];
      }
      if (total_count < 2 * min_leaf) continue;

      double parent_score = (total_g * total_g) / (total_h + LAMBDA);

      int left_cnt = 0;
      double left_g = 0.0, left_h = 0.0;
      for (int b = 0; b < BINS - 1; b++) {
        left_cnt += cnt[(size_t)b];
        left_g += sum_g[(size_t)b];
        left_h += sum_h[(size_t)b];
        int right_cnt = total_count - left_cnt;
        if (left_cnt < min_leaf || right_cnt < min_leaf) continue;
        double right_g = total_g - left_g;
        double right_h = total_h - left_h;
        double gain = (left_g * left_g) / (left_h + LAMBDA) + (right_g * right_g) / (right_h + LAMBDA) - parent_score;
        if (gain > best_gain) {
          best_gain = gain;
          best_f = f;
          double thr_f = 0.0;
          if (is_quantile) thr_f = (double)(*thr_arr)[(size_t)b];
          else thr_f = (double)feat_min[(size_t)f] + (double)r * ((double)(b + 1) / (double)BINS);
          best_thr_q = clamp_i32(js_round_double(thr_f * (double)scaleQ));
        }
      }
    }

    if (best_f < 0) {
      fill_forced_leaf(node_idx, level, compute_leaf_q(samples));
      return;
    }

    std::vector<int> left;
    std::vector<int> right;
    left.reserve(samples.size());
    right.reserve(samples.size());
    for (int rr : samples) {
      float x = X[(size_t)rr * (size_t)n_features + (size_t)best_f];
      int32_t xq = quantize_to_i32((double)x, (int64_t)scaleQ);
      if (xq > best_thr_q) right.push_back(rr);
      else left.push_back(rr);
    }
    if ((int)left.size() < min_leaf || (int)right.size() < min_leaf) {
      fill_forced_leaf(node_idx, level, compute_leaf_q(samples));
      return;
    }

    tree.feat[(size_t)node_idx] = (uint16_t)best_f;
    tree.thr[(size_t)node_idx] = best_thr_q;
    node_split(node_idx * 2 + 1, level + 1, left);
    node_split(node_idx * 2 + 2, level + 1, right);
  };

  node_split(0, 0, train_samples);
  return tree;
}

static inline int tree_traverse_leaf_index(const Tree& tree, const std::vector<float>& X, int n_features, int row, int depth, int scaleQ) {
  int node = 0;
  int pow2 = 1 << depth;
  int internal = pow2 - 1;
  for (int d = 0; d < depth; d++) {
    uint16_t f = tree.feat[(size_t)node];
    int32_t thr = tree.thr[(size_t)node];
    float x = X[(size_t)row * (size_t)n_features + (size_t)f];
    int32_t xq = quantize_to_i32((double)x, (int64_t)scaleQ);
    node = (xq > thr) ? (node * 2 + 2) : (node * 2 + 1);
  }
  int leaf_idx = node - internal;
  if (leaf_idx < 0) leaf_idx = 0;
  if (leaf_idx >= pow2) leaf_idx = pow2 - 1;
  return leaf_idx;
}

static void apply_tree_scalar(const Tree& tree, const std::vector<float>& X, int n_features,
                             const std::vector<int>& idx, std::vector<double>& predQ, int depth, int scaleQ) {
  if (idx.empty()) return;
  for (int r : idx) {
    int leaf_idx = tree_traverse_leaf_index(tree, X, n_features, r, depth, scaleQ);
    predQ[(size_t)r] += (double)tree.leaf[(size_t)leaf_idx];
  }
}

static void apply_tree_vector(const Tree& tree, const std::vector<float>& X, int n_features,
                             const std::vector<int>& idx, std::vector<double>& predQ_flat, int depth, int scaleQ, int n_out, int k) {
  if (idx.empty()) return;
  for (int r : idx) {
    int leaf_idx = tree_traverse_leaf_index(tree, X, n_features, r, depth, scaleQ);
    size_t pos = (size_t)r * (size_t)n_out + (size_t)k;
    predQ_flat[pos] += (double)tree.leaf[(size_t)leaf_idx];
  }
}

// --------------- Model serialization (GL1F v1/v2) ---------------
// Helpers to write little-endian values into a std::vector<uint8_t>
static inline void write_u8(std::vector<uint8_t>& buf, uint8_t v) { buf.push_back(v); }
static inline void write_u16(std::vector<uint8_t>& buf, uint16_t v) {
  buf.push_back((uint8_t)(v & 0xFF));
  buf.push_back((uint8_t)((v >> 8) & 0xFF));
}
static inline void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
  buf.push_back((uint8_t)(v & 0xFF));
  buf.push_back((uint8_t)((v >> 8) & 0xFF));
  buf.push_back((uint8_t)((v >> 16) & 0xFF));
  buf.push_back((uint8_t)((v >> 24) & 0xFF));
}
static inline void write_i32(std::vector<uint8_t>& buf, int32_t v) {
  write_u32(buf, (uint32_t)v);
}

static std::vector<uint8_t> serialize_model_v1(int n_features, int depth, const std::vector<Tree>& trees, int32_t baseQ, uint32_t scaleQ) {
  int pow2 = 1 << depth;
  int internal = pow2 - 1;
  size_t per_tree = (size_t)internal * 8 + (size_t)pow2 * 4;
  size_t total = 24 + (size_t)trees.size() * per_tree;
  std::vector<uint8_t> out;
  out.reserve(total);

  // header
  out.push_back('G'); out.push_back('L'); out.push_back('1'); out.push_back('F');
  write_u8(out, 1); // version
  write_u8(out, 0);
  write_u16(out, (uint16_t)n_features);
  write_u16(out, (uint16_t)depth);
  write_u32(out, (uint32_t)trees.size());
  write_i32(out, baseQ);
  write_u32(out, scaleQ);
  write_u16(out, 0); // reserved

  for (const auto& t : trees) {
    // internal nodes
    for (int i = 0; i < internal; i++) {
      uint16_t f = (i < (int)t.feat.size()) ? t.feat[(size_t)i] : 0;
      int32_t thr = (i < (int)t.thr.size()) ? t.thr[(size_t)i] : 0;
      write_u16(out, f);
      write_i32(out, thr);
      write_u16(out, 0); // flags/reserved
    }
    // leaves
    for (int i = 0; i < pow2; i++) {
      int32_t leaf = (i < (int)t.leaf.size()) ? t.leaf[(size_t)i] : 0;
      write_i32(out, leaf);
    }
  }
  return out;
}

static std::vector<uint8_t> serialize_model_v2(
  int n_features,
  int depth,
  int n_classes,
  int trees_per_class,
  const std::vector<int32_t>& base_logits_q,
  uint32_t scaleQ,
  const std::vector<std::vector<Tree>>& trees_by_class
) {
  int pow2 = 1 << depth;
  int internal = pow2 - 1;
  size_t per_tree = (size_t)internal * 8 + (size_t)pow2 * 4;
  size_t header = 24 + (size_t)n_classes * 4;
  size_t total = header + (size_t)n_classes * (size_t)trees_per_class * per_tree;
  std::vector<uint8_t> out;
  out.reserve(total);

  out.push_back('G'); out.push_back('L'); out.push_back('1'); out.push_back('F');
  write_u8(out, 2);
  write_u8(out, 0);
  write_u16(out, (uint16_t)n_features);
  write_u16(out, (uint16_t)depth);
  write_u32(out, (uint32_t)trees_per_class);
  write_i32(out, 0); // reserved
  write_u32(out, scaleQ);
  write_u16(out, (uint16_t)n_classes);

  // base logits
  for (int k = 0; k < n_classes; k++) {
    int32_t v = (k < (int)base_logits_q.size()) ? base_logits_q[(size_t)k] : 0;
    write_i32(out, v);
  }

  for (int k = 0; k < n_classes; k++) {
    const auto& arr = trees_by_class[(size_t)k];
    for (int t = 0; t < trees_per_class; t++) {
      const Tree& tr = arr[(size_t)t];
      for (int i = 0; i < internal; i++) {
        uint16_t f = (i < (int)tr.feat.size()) ? tr.feat[(size_t)i] : 0;
        int32_t thr = (i < (int)tr.thr.size()) ? tr.thr[(size_t)i] : 0;
        write_u16(out, f);
        write_i32(out, thr);
        write_u16(out, 0);
      }
      for (int i = 0; i < pow2; i++) {
        int32_t leaf = (i < (int)tr.leaf.size()) ? tr.leaf[(size_t)i] : 0;
        write_i32(out, leaf);
      }
    }
  }
  return out;
}

// --------------- Minimal JSON builder ---------------
static inline void json_escape_append(std::string& out, std::string_view s) {
  for (char c : s) {
    switch (c) {
      case '\"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
}
static inline std::string json_str(std::string_view s) {
  std::string out;
  out.push_back('\"');
  json_escape_append(out, s);
  out.push_back('\"');
  return out;
}
static inline std::string json_num(double x) {
  if (!std::isfinite(x)) return "null";
  std::array<char, 64> buf;
  // Use a reasonable precision.
  auto [p, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), x, std::chars_format::general, 17);
  if (ec != std::errc()) return "null";
  return std::string(buf.data(), p);
}
static inline std::string json_int(int64_t x) {
  return std::to_string(x);
}
static inline std::string json_bool(bool b) {
  return b ? "true" : "false";
}

static std::string json_array_int(const std::vector<int>& arr) {
  std::string out = "[";
  for (size_t i = 0; i < arr.size(); i++) {
    if (i) out += ",";
    out += json_int(arr[i]);
  }
  out += "]";
  return out;
}
static std::string json_array_f64(const std::vector<double>& arr) {
  std::string out = "[";
  for (size_t i = 0; i < arr.size(); i++) {
    if (i) out += ",";
    out += json_num(arr[i]);
  }
  out += "]";
  return out;
}

static std::string json_array_f64_from_f32(const std::vector<float>& arr) {
  std::string out = "[";
  for (size_t i = 0; i < arr.size(); i++) {
    if (i) out += ",";
    out += json_num((double)arr[i]);
  }
  out += "]";
  return out;
}

static std::string json_array_str(const std::vector<std::string>& arr) {
  std::string out = "[";
  for (size_t i = 0; i < arr.size(); i++) {
    if (i) out += ",";
    out += json_str(arr[i]);
  }
  out += "]";
  return out;
}

// Pack features metadata exactly like src/common.js packNftFeatures.
static std::string pack_nft_features(
  const std::string& task,
  const std::vector<std::string>& feature_names,
  const std::string& label_name,
  const std::vector<std::string>& labels,
  const std::vector<std::string>& label_names // multilabel outputs
) {
  std::string t = task;
  if (t == "binary") t = "binary_classification";
  if (t == "multiclass") t = "multiclass_classification";
  if (t == "multilabel") t = "multilabel_classification";
  if (t != "regression" && t != "binary_classification" && t != "multiclass_classification" && t != "multilabel_classification") t = "regression";

  // meta object to JSON (minimal)
  std::string meta = "{\"v\":1,\"task\":" + json_str(t);
  if (!label_name.empty()) meta += ",\"labelName\":" + json_str(label_name);

  if (t == "multilabel_classification") {
    if (!label_names.empty()) meta += ",\"labelNames\":" + json_array_str(label_names);
    if (labels.size() >= 2) meta += ",\"labels\":[" + json_str(labels[0]) + "," + json_str(labels[1]) + "]";
    else meta += ",\"labels\":[\"0\",\"1\"]";
  } else if ((t == "binary_classification" || t == "multiclass_classification") && labels.size() >= 2) {
    meta += ",\"labels\":" + json_array_str(labels);
  }
  meta += "}";

  std::string out;
  out.reserve(64 + feature_names.size() * 16);
  out += "#meta=";
  out += meta;
  for (auto& f : feature_names) {
    std::string s = trim_copy(f);
    if (!s.empty()) {
      out.push_back('\n');
      out += s;
    }
  }
  return out;
}

static std::vector<uint8_t> append_gl1x_footer(const std::vector<uint8_t>& model_bytes, const std::string& json_payload) {
  std::vector<uint8_t> out = model_bytes;
  // footer header: "GL1X" + version u8 + 3 reserved + u32 payloadLen
  out.push_back('G'); out.push_back('L'); out.push_back('1'); out.push_back('X');
  out.push_back(1);
  out.push_back(0); out.push_back(0); out.push_back(0);
  uint32_t len = (uint32_t)json_payload.size();
  write_u32(out, len);
  out.insert(out.end(), json_payload.begin(), json_payload.end());
  return out;
}

// --------------- Training entrypoints ---------------
struct TrainOut {
  std::vector<uint8_t> model_bytes; // GL1F only (no footer)
  std::string meta_json; // JSON object string
  std::string curve_json; // JSON object string
  std::string features_packed; // string
  int gl1f_version = 1;
};

static TrainOut train_regression_cpp(const Dataset& ds, const LRSchedule& lr_sched_in,
                                    int depth, int max_trees, int min_leaf, int seed, bool early_stop, int patience,
                                    int scaleQ, int bins, const std::string& binning, double split_train, double split_val, bool refit_train_val, int quantile_samples) {
  int n_rows = ds.n_rows;
  int n_features = ds.n_features;

  // yQ
  std::vector<double> yQ((size_t)n_rows);
  for (int i = 0; i < n_rows; i++) {
    yQ[(size_t)i] = (double)clamp_i32(js_round_double((double)ds.y[(size_t)i] * (double)scaleQ));
  }

  auto idx = shuffled_indices(n_rows, seed);
  std::vector<int> train, val, test;
  split_idx(idx, split_train, split_val, train, val, test);

  std::vector<int> train_fit = train;
  if (refit_train_val) {
    train_fit.insert(train_fit.end(), val.begin(), val.end());
  }
  std::vector<int>& train_idx = train_fit;

  std::vector<float> feat_min, feat_max;
  compute_feat_min_max(ds.X, n_rows, n_features, train_fit, feat_min, feat_max);
  std::vector<float> feat_range((size_t)n_features, 0.0f);
  for (int f = 0; f < n_features; f++) {
    float r = feat_max[(size_t)f] - feat_min[(size_t)f];
    feat_range[(size_t)f] = (r > 0) ? r : 0.0f;
  }

  std::optional<std::vector<std::vector<float>>> q_thr;
  if (binning == "quantile") {
    q_thr = compute_quantile_thresholds(ds.X, n_rows, n_features, train_fit, feat_min, feat_range, bins, quantile_samples);
  }

  // baseQ = round(mean yQ[train])
  double sumy = 0.0;
  for (int r : train_idx) sumy += yQ[(size_t)r];
  int32_t baseQ = clamp_i32(js_round_double(sumy / (double)train_idx.size()));

  std::vector<double> predQ((size_t)n_rows, (double)baseQ);
  std::vector<float> residual((size_t)n_rows, 0.0f);

  XorShift32 rng((uint32_t)(seed ^ 0x9E3779B9));

  std::vector<Tree> trees;
  trees.reserve((size_t)max_trees);

  std::vector<int> curve_steps;
  std::vector<double> curve_train, curve_val, curve_test;

  double best_val = std::numeric_limits<double>::infinity();
  double best_train = std::numeric_limits<double>::infinity();
  double best_test = std::numeric_limits<double>::infinity();
  int best_iter = 0;
  int since_best = 0;

  LRSchedule lr_sched = lr_sched_in;

  for (int t = 1; t <= max_trees; t++) {
    double lr_used = lr_sched.lr_for_iter(t);

    // residual on train_idx
    for (int r : train_idx) {
      residual[(size_t)r] = (float)((yQ[(size_t)r] - predQ[(size_t)r]) / (double)scaleQ);
    }

    Tree tree = build_tree_regression(ds.X, n_rows, n_features, train_idx, residual, feat_min, feat_range,
                                      depth, min_leaf, lr_used, scaleQ, rng, bins, binning,
                                      q_thr ? &(*q_thr) : nullptr);
    trees.push_back(tree);

    apply_tree_scalar(tree, ds.X, n_features, train_idx, predQ, depth, scaleQ);
    apply_tree_scalar(tree, ds.X, n_features, val, predQ, depth, scaleQ);
    apply_tree_scalar(tree, ds.X, n_features, test, predQ, depth, scaleQ);

    double train_mse = mse_q(yQ, predQ, train_idx, scaleQ);
    double val_mse = mse_q(yQ, predQ, val, scaleQ);
    double test_mse = mse_q(yQ, predQ, test, scaleQ);

    curve_steps.push_back(t);
    curve_train.push_back(train_mse);
    curve_val.push_back(val_mse);
    curve_test.push_back(test_mse);

    bool improved = false;
    if (val_mse + 1e-12 < best_val) {
      best_val = val_mse;
      best_train = train_mse;
      best_test = test_mse;
      best_iter = t;
      since_best = 0;
      improved = true;
    } else {
      since_best += 1;
    }

    lr_sched.after_metric(improved);

    if (early_stop && since_best >= patience) break;
  }

  int used_trees = early_stop ? std::max(1, best_iter) : std::max(1, (int)trees.size());
  std::vector<Tree> final_trees(trees.begin(), trees.begin() + used_trees);

  auto model_bytes = serialize_model_v1(n_features, depth, final_trees, baseQ, (uint32_t)scaleQ);

  // meta + curve JSON (match python keys)
  std::string meta = "{";
  meta += "\"task\":\"regression\"";
  meta += ",\"metricName\":\"MSE\"";
  meta += ",\"nFeatures\":" + json_int(n_features);
  meta += ",\"depth\":" + json_int(depth);
  meta += ",\"maxTrees\":" + json_int(max_trees);
  meta += ",\"usedTrees\":" + json_int(used_trees);
  meta += ",\"baseQ\":" + json_int(baseQ);
  meta += ",\"scaleQ\":" + json_int(scaleQ);
  meta += ",\"bins\":" + json_int(bins);
  meta += ",\"binning\":" + json_str(binning);
  meta += ",\"bestIter\":" + json_int(early_stop ? best_iter : used_trees);
  meta += ",\"bestTrainMetric\":" + json_num(best_train);
  meta += ",\"bestValMetric\":" + json_num(best_val);
  meta += ",\"bestTestMetric\":" + json_num(best_test);
  meta += ",\"bestTrainMSE\":" + json_num(best_train);
  meta += ",\"bestValMSE\":" + json_num(best_val);
  meta += ",\"bestTestMSE\":" + json_num(best_test);
  meta += ",\"earlyStop\":" + json_bool(early_stop);
  meta += "}";

  std::string curve = "{";
  curve += "\"steps\":" + json_array_int(curve_steps);
  curve += ",\"train\":" + json_array_f64(curve_train);
  curve += ",\"val\":" + json_array_f64(curve_val);
  curve += ",\"test\":" + json_array_f64(curve_test);
  curve += "}";

  TrainOut out;
  out.model_bytes = std::move(model_bytes);
  out.meta_json = std::move(meta);
  out.curve_json = std::move(curve);
  out.features_packed = pack_nft_features("regression", ds.feature_names, ds.label_name, {}, {});
  out.gl1f_version = 1;
  return out;
}

static TrainOut train_binary_cpp(const Dataset& ds, const LRSchedule& lr_sched_in,
                                 int depth, int max_trees, int min_leaf, int seed, bool early_stop, int patience,
                                 int scaleQ, int bins, const std::string& binning, double split_train, double split_val, bool refit_train_val, int quantile_samples,
                                 const std::string& imb_mode, double imb_cap, bool imb_normalize, bool stratify, double w0_manual, double w1_manual) {
  int n_rows = ds.n_rows;
  int n_features = ds.n_features;

  std::vector<uint8_t> y01((size_t)n_rows, 0);
  for (int i = 0; i < n_rows; i++) {
    y01[(size_t)i] = (ds.y[(size_t)i] >= 0.5f) ? 1 : 0;
  }
  std::vector<int> y_bin((size_t)n_rows, 0);
  for (int i = 0; i < n_rows; i++) y_bin[(size_t)i] = y01[(size_t)i] ? 1 : 0;

  auto idx = shuffled_indices(n_rows, seed);
  std::vector<int> train, val, test;
  if (stratify) {
    split_idx_stratified_by_class(idx, y_bin, 2, split_train, split_val, train, val, test);
  } else {
    split_idx(idx, split_train, split_val, train, val, test);
  }

  std::vector<int> train_fit = train;
  if (refit_train_val) train_fit.insert(train_fit.end(), val.begin(), val.end());
  std::vector<int>& train_idx = train_fit;

  std::vector<float> feat_min, feat_max;
  compute_feat_min_max(ds.X, n_rows, n_features, train_fit, feat_min, feat_max);
  std::vector<float> feat_range((size_t)n_features, 0.0f);
  for (int f = 0; f < n_features; f++) {
    float r = feat_max[(size_t)f] - feat_min[(size_t)f];
    feat_range[(size_t)f] = (r > 0) ? r : 0.0f;
  }

  std::optional<std::vector<std::vector<float>>> q_thr;
  if (binning == "quantile") {
    q_thr = compute_quantile_thresholds(ds.X, n_rows, n_features, train_fit, feat_min, feat_range, bins, quantile_samples);
  }

  // Optional row weights
  std::optional<std::vector<float>> w_row_opt;
  const std::vector<float>* w_row = nullptr;
  std::string imb = lower_copy(trim_view(imb_mode));
  if (imb == "auto" || imb == "manual") {
    int c0 = 0, c1 = 0;
    for (int r : train_idx) {
      if (y_bin[(size_t)r] == 1) c1++; else c0++;
    }
    int N = c0 + c1;
    double w0 = 1.0, w1 = 1.0;
    if (imb == "manual") {
      w0 = (w0_manual > 0) ? w0_manual : 1.0;
      w1 = (w1_manual > 0) ? w1_manual : 1.0;
    } else {
      if (c0 > 0) w0 = (double)N / (2.0 * (double)c0);
      if (c1 > 0) w1 = (double)N / (2.0 * (double)c1);
    }
    if (imb_cap > 0) {
      w0 = std::min(w0, imb_cap);
      w1 = std::min(w1, imb_cap);
    }
    if (imb_normalize && N > 0) {
      double avg = (w0 * (double)c0 + w1 * (double)c1) / (double)N;
      if (avg > 0) { w0 /= avg; w1 /= avg; }
    }
    std::vector<float> w_row_v((size_t)n_rows, 1.0f);
    for (int i = 0; i < n_rows; i++) w_row_v[(size_t)i] = (y_bin[(size_t)i] == 1) ? (float)w1 : (float)w0;
    w_row_opt = std::move(w_row_v);
    w_row = &(*w_row_opt);
  }

  // base logit: log-odds of weighted positive rate
  double sum_w = 0.0, sum_w_pos = 0.0;
  for (int r : train_idx) {
    double w = w_row ? (double)(*w_row)[(size_t)r] : 1.0;
    sum_w += w;
    sum_w_pos += w * ((y_bin[(size_t)r] == 1) ? 1.0 : 0.0);
  }
  double p0 = sum_w_pos / std::max(1e-12, sum_w);
  double eps = 1e-6;
  p0 = std::min(std::max(p0, eps), 1.0 - eps);
  double base_logit = std::log(p0 / (1.0 - p0));
  int32_t baseQ = clamp_i32(js_round_double(base_logit * (double)scaleQ));

  std::vector<double> predQ((size_t)n_rows, (double)baseQ);
  std::vector<float> grad((size_t)n_rows, 0.0f);
  std::vector<float> hess((size_t)n_rows, 0.0f);

  XorShift32 rng((uint32_t)(seed ^ 0x9E3779B9));
  std::vector<Tree> trees;
  trees.reserve((size_t)max_trees);

  std::vector<int> curve_steps;
  std::vector<double> curve_train, curve_val, curve_test;
  std::vector<double> curve_train_acc, curve_val_acc, curve_test_acc;

  double best_val = std::numeric_limits<double>::infinity();
  double best_train = std::numeric_limits<double>::infinity();
  double best_test = std::numeric_limits<double>::infinity();
  double best_train_acc = 0.0, best_val_acc = 0.0, best_test_acc = 0.0;
  int best_iter = 0;
  int since_best = 0;

  LRSchedule lr_sched = lr_sched_in;

  for (int t = 1; t <= max_trees; t++) {
    double lr_used = lr_sched.lr_for_iter(t);

    // grad/hess on train_idx
    for (int r : train_idx) {
      double logit = predQ[(size_t)r] / (double)scaleQ;
      double p = sigmoid(logit);
      double y = (y_bin[(size_t)r] == 1) ? 1.0 : 0.0;
      double w = w_row ? (double)(*w_row)[(size_t)r] : 1.0;
      grad[(size_t)r] = (float)((y - p) * w);
      hess[(size_t)r] = (float)((p * (1.0 - p)) * w);
    }

    Tree tree = build_tree_binary(ds.X, n_rows, n_features, train_idx, grad, hess,
                                  feat_min, feat_range, depth, min_leaf, lr_used, scaleQ, rng, bins, binning,
                                  q_thr ? &(*q_thr) : nullptr);
    trees.push_back(tree);

    apply_tree_scalar(tree, ds.X, n_features, train_idx, predQ, depth, scaleQ);
    apply_tree_scalar(tree, ds.X, n_features, val, predQ, depth, scaleQ);
    apply_tree_scalar(tree, ds.X, n_features, test, predQ, depth, scaleQ);

    auto [train_loss, train_acc] = logloss_acc_binary(y01, predQ, train_idx, scaleQ, w_row);
    auto [val_loss, val_acc] = logloss_acc_binary(y01, predQ, val, scaleQ, w_row);
    auto [test_loss, test_acc] = logloss_acc_binary(y01, predQ, test, scaleQ, w_row);

    curve_steps.push_back(t);
    curve_train.push_back(train_loss);
    curve_val.push_back(val_loss);
    curve_test.push_back(test_loss);
    curve_train_acc.push_back(train_acc);
    curve_val_acc.push_back(val_acc);
    curve_test_acc.push_back(test_acc);

    bool improved = false;
    if (val_loss + 1e-12 < best_val) {
      best_val = val_loss;
      best_train = train_loss;
      best_test = test_loss;
      best_train_acc = train_acc;
      best_val_acc = val_acc;
      best_test_acc = test_acc;
      best_iter = t;
      since_best = 0;
      improved = true;
    } else {
      since_best += 1;
    }

    lr_sched.after_metric(improved);
    if (early_stop && since_best >= patience) break;
  }

  int used_trees = early_stop ? std::max(1, best_iter) : std::max(1, (int)trees.size());
  std::vector<Tree> final_trees(trees.begin(), trees.begin() + used_trees);

  auto model_bytes = serialize_model_v1(n_features, depth, final_trees, baseQ, (uint32_t)scaleQ);

  std::string meta = "{";
  meta += "\"task\":\"binary_classification\"";
  meta += ",\"metricName\":\"LogLoss\"";
  meta += ",\"nFeatures\":" + json_int(n_features);
  meta += ",\"depth\":" + json_int(depth);
  meta += ",\"maxTrees\":" + json_int(max_trees);
  meta += ",\"usedTrees\":" + json_int(used_trees);
  meta += ",\"baseQ\":" + json_int(baseQ);
  meta += ",\"scaleQ\":" + json_int(scaleQ);
  meta += ",\"bins\":" + json_int(bins);
  meta += ",\"binning\":" + json_str(binning);
  meta += ",\"bestIter\":" + json_int(early_stop ? best_iter : used_trees);
  meta += ",\"bestTrainMetric\":" + json_num(best_train);
  meta += ",\"bestValMetric\":" + json_num(best_val);
  meta += ",\"bestTestMetric\":" + json_num(best_test);
  meta += ",\"bestTrainLoss\":" + json_num(best_train);
  meta += ",\"bestValLoss\":" + json_num(best_val);
  meta += ",\"bestTestLoss\":" + json_num(best_test);
  meta += ",\"bestTrainAcc\":" + json_num(best_train_acc);
  meta += ",\"bestValAcc\":" + json_num(best_val_acc);
  meta += ",\"bestTestAcc\":" + json_num(best_test_acc);
  meta += ",\"earlyStop\":" + json_bool(early_stop);
  meta += "}";

  std::string curve = "{";
  curve += "\"steps\":" + json_array_int(curve_steps);
  curve += ",\"train\":" + json_array_f64(curve_train);
  curve += ",\"val\":" + json_array_f64(curve_val);
  curve += ",\"test\":" + json_array_f64(curve_test);
  curve += ",\"trainAcc\":" + json_array_f64(curve_train_acc);
  curve += ",\"valAcc\":" + json_array_f64(curve_val_acc);
  curve += ",\"testAcc\":" + json_array_f64(curve_test_acc);
  curve += "}";

  TrainOut out;
  out.model_bytes = std::move(model_bytes);
  out.meta_json = std::move(meta);
  out.curve_json = std::move(curve);
  out.features_packed = pack_nft_features("binary_classification", ds.feature_names, ds.label_name, ds.classes, {});
  out.gl1f_version = 1;
  return out;
}

static TrainOut train_multiclass_cpp(const Dataset& ds, const LRSchedule& lr_sched_in,
                                     int depth, int max_trees, int min_leaf, int seed, bool early_stop, int patience,
                                     int scaleQ, int bins, const std::string& binning, double split_train, double split_val, bool refit_train_val, int quantile_samples,
                                     const std::string& imb_mode, double imb_cap, bool imb_normalize, bool stratify, const std::vector<double>& class_weights_manual) {
  int n_rows = ds.n_rows;
  int n_features = ds.n_features;
  int n_classes = (int)ds.classes.size();
  if (n_classes < 2) {
    // fallback: infer from y
    int mx = 0;
    for (float v : ds.y) mx = std::max(mx, (int)std::round(v));
    n_classes = std::max(2, mx + 1);
  }

  std::vector<int> yK((size_t)n_rows, 0);
  for (int i = 0; i < n_rows; i++) {
    int k = (int)std::round((double)ds.y[(size_t)i]);
    if (k < 0) k = 0;
    if (k >= n_classes) k = n_classes - 1;
    yK[(size_t)i] = k;
  }

  auto idx = shuffled_indices(n_rows, seed);
  std::vector<int> train, val, test;
  if (stratify) {
    split_idx_stratified_by_class(idx, yK, n_classes, split_train, split_val, train, val, test);
  } else {
    split_idx(idx, split_train, split_val, train, val, test);
  }

  std::vector<int> train_fit = train;
  if (refit_train_val) train_fit.insert(train_fit.end(), val.begin(), val.end());
  std::vector<int>& train_idx = train_fit;

  std::vector<float> feat_min, feat_max;
  compute_feat_min_max(ds.X, n_rows, n_features, train_fit, feat_min, feat_max);
  std::vector<float> feat_range((size_t)n_features, 0.0f);
  for (int f = 0; f < n_features; f++) {
    float r = feat_max[(size_t)f] - feat_min[(size_t)f];
    feat_range[(size_t)f] = (r > 0) ? r : 0.0f;
  }

  std::optional<std::vector<std::vector<float>>> q_thr;
  if (binning == "quantile") {
    q_thr = compute_quantile_thresholds(ds.X, n_rows, n_features, train_fit, feat_min, feat_range, bins, quantile_samples);
  }

  // Optional class weighting
  std::optional<std::vector<float>> w_row_opt;
  const std::vector<float>* w_row = nullptr;
  std::string imb = lower_copy(trim_view(imb_mode));
  std::vector<double> w_class((size_t)n_classes, 1.0);
  if (imb == "auto" || imb == "manual") {
    std::vector<int64_t> counts((size_t)n_classes, 0);
    for (int r : train_idx) {
      int cls = yK[(size_t)r];
      if (cls < 0 || cls >= n_classes) cls = 0;
      counts[(size_t)cls] += 1;
    }
    int N = (int)train_idx.size();
    if (imb == "manual" && (int)class_weights_manual.size() >= n_classes) {
      for (int k = 0; k < n_classes; k++) {
        double w = class_weights_manual[(size_t)k];
        if (!(w > 0)) w = 1.0;
        w_class[(size_t)k] = w;
      }
    } else {
      for (int k = 0; k < n_classes; k++) {
        int64_t c = counts[(size_t)k];
        double w = (c > 0) ? ((double)N / ((double)n_classes * (double)c)) : 1.0;
        w_class[(size_t)k] = w;
      }
    }
    double cap = (imb_cap > 0) ? imb_cap : 20.0;
    for (int k = 0; k < n_classes; k++) w_class[(size_t)k] = std::min(w_class[(size_t)k], cap);
    if (imb_normalize && N > 0) {
      double avg = 0.0;
      for (int k = 0; k < n_classes; k++) avg += w_class[(size_t)k] * (double)counts[(size_t)k];
      avg /= (double)N;
      if (avg > 0) for (int k = 0; k < n_classes; k++) w_class[(size_t)k] /= avg;
    }
    std::vector<float> w_row_v((size_t)n_rows, 1.0f);
    for (int i = 0; i < n_rows; i++) {
      int cls = yK[(size_t)i];
      if (cls < 0 || cls >= n_classes) cls = 0;
      w_row_v[(size_t)i] = (float)w_class[(size_t)cls];
    }
    w_row_opt = std::move(w_row_v);
    w_row = &(*w_row_opt);
  }

  // Base logits: log(prior) with smoothing (match python)
  const double smooth = 1e-3;
  std::vector<double> sum_w_class((size_t)n_classes, 0.0);
  double sum_w = 0.0;
  for (int r : train_idx) {
    int cls = yK[(size_t)r];
    double w = w_row ? (double)(*w_row)[(size_t)r] : 1.0;
    sum_w += w;
    if (0 <= cls && cls < n_classes) sum_w_class[(size_t)cls] += w;
  }
  double denom = std::max(1e-9, sum_w + smooth * (double)n_classes);
  std::vector<int32_t> base_logits_q((size_t)n_classes, 0);
  for (int k = 0; k < n_classes; k++) {
    double pk = (sum_w_class[(size_t)k] + smooth) / denom;
    pk = std::max(pk, 1e-9);
    base_logits_q[(size_t)k] = clamp_i32(js_round_double(std::log(pk) * (double)scaleQ));
  }

  std::vector<double> predQ_flat((size_t)n_rows * (size_t)n_classes, 0.0);
  for (int r = 0; r < n_rows; r++) {
    size_t base = (size_t)r * (size_t)n_classes;
    for (int k = 0; k < n_classes; k++) predQ_flat[base + (size_t)k] = (double)base_logits_q[(size_t)k];
  }

  std::vector<float> prob_flat;
  softmax_probs_inplace(predQ_flat, n_rows, n_classes, scaleQ, prob_flat);

  std::vector<float> grad((size_t)n_rows, 0.0f);
  std::vector<float> hess((size_t)n_rows, 0.0f);

  XorShift32 rng((uint32_t)(seed ^ 0x9E3779B9));
  std::vector<std::vector<Tree>> trees_by_class((size_t)n_classes);
  for (int k = 0; k < n_classes; k++) trees_by_class[(size_t)k].reserve((size_t)max_trees);

  std::vector<int> curve_steps;
  std::vector<double> curve_train, curve_val, curve_test;
  std::vector<double> curve_train_acc, curve_val_acc, curve_test_acc;

  double best_val = std::numeric_limits<double>::infinity();
  double best_train = std::numeric_limits<double>::infinity();
  double best_test = std::numeric_limits<double>::infinity();
  double best_train_acc = 0.0, best_val_acc = 0.0, best_test_acc = 0.0;
  int best_iter = 0;
  int since_best = 0;

  LRSchedule lr_sched = lr_sched_in;

  for (int t = 1; t <= max_trees; t++) {
    double lr_used = lr_sched.lr_for_iter(t);

    for (int k = 0; k < n_classes; k++) {
      // grad/hess for class k on train_idx
      for (int r : train_idx) {
        size_t base = (size_t)r * (size_t)n_classes;
        double p_k = (double)prob_flat[base + (size_t)k];
        double yk = (yK[(size_t)r] == k) ? 1.0 : 0.0;
        double w = w_row ? (double)(*w_row)[(size_t)r] : 1.0;
        grad[(size_t)r] = (float)((yk - p_k) * w);
        hess[(size_t)r] = (float)((p_k * (1.0 - p_k)) * w);
      }

      Tree tree = build_tree_binary(ds.X, n_rows, n_features, train_idx, grad, hess,
                                    feat_min, feat_range, depth, min_leaf, lr_used, scaleQ, rng, bins, binning,
                                    q_thr ? &(*q_thr) : nullptr);
      trees_by_class[(size_t)k].push_back(tree);

      apply_tree_vector(tree, ds.X, n_features, train_idx, predQ_flat, depth, scaleQ, n_classes, k);
      apply_tree_vector(tree, ds.X, n_features, val, predQ_flat, depth, scaleQ, n_classes, k);
      apply_tree_vector(tree, ds.X, n_features, test, predQ_flat, depth, scaleQ, n_classes, k);
    }

    softmax_probs_inplace(predQ_flat, n_rows, n_classes, scaleQ, prob_flat);

    auto [train_loss, train_acc] = logloss_acc_multiclass(yK, prob_flat, train_idx, n_classes, w_row);
    auto [val_loss, val_acc] = logloss_acc_multiclass(yK, prob_flat, val, n_classes, w_row);
    auto [test_loss, test_acc] = logloss_acc_multiclass(yK, prob_flat, test, n_classes, w_row);

    curve_steps.push_back(t);
    curve_train.push_back(train_loss);
    curve_val.push_back(val_loss);
    curve_test.push_back(test_loss);
    curve_train_acc.push_back(train_acc);
    curve_val_acc.push_back(val_acc);
    curve_test_acc.push_back(test_acc);

    bool improved = false;
    if (val_loss + 1e-12 < best_val) {
      best_val = val_loss;
      best_train = train_loss;
      best_test = test_loss;
      best_train_acc = train_acc;
      best_val_acc = val_acc;
      best_test_acc = test_acc;
      best_iter = t;
      since_best = 0;
      improved = true;
    } else {
      since_best += 1;
    }

    lr_sched.after_metric(improved);
    if (early_stop && since_best >= patience) break;
  }

  int iters_done = trees_by_class.empty() ? 0 : (int)trees_by_class[0].size();
  int used_trees_per_class = early_stop ? std::max(1, best_iter) : std::max(1, iters_done);

  std::vector<std::vector<Tree>> final_trees_by_class((size_t)n_classes);
  for (int k = 0; k < n_classes; k++) {
    auto& arr = trees_by_class[(size_t)k];
    if ((int)arr.size() < used_trees_per_class) used_trees_per_class = (int)arr.size();
  }
  used_trees_per_class = std::max(1, used_trees_per_class);
  for (int k = 0; k < n_classes; k++) {
    auto& arr = trees_by_class[(size_t)k];
    final_trees_by_class[(size_t)k].assign(arr.begin(), arr.begin() + used_trees_per_class);
  }

  auto model_bytes = serialize_model_v2(n_features, depth, n_classes, used_trees_per_class, base_logits_q, (uint32_t)scaleQ, final_trees_by_class);

  std::string meta = "{";
  meta += "\"task\":\"multiclass_classification\"";
  meta += ",\"metricName\":\"LogLoss\"";
  meta += ",\"nFeatures\":" + json_int(n_features);
  meta += ",\"depth\":" + json_int(depth);
  meta += ",\"scaleQ\":" + json_int(scaleQ);
  meta += ",\"bins\":" + json_int(bins);
  meta += ",\"binning\":" + json_str(binning);
  meta += ",\"nClasses\":" + json_int(n_classes);
  meta += ",\"maxTrees\":" + json_int(max_trees);
  meta += ",\"usedTrees\":" + json_int(used_trees_per_class);
  meta += ",\"treesPerClass\":" + json_int(used_trees_per_class);
  meta += ",\"totalTrees\":" + json_int((int64_t)used_trees_per_class * (int64_t)n_classes);
  meta += ",\"bestIter\":" + json_int(early_stop ? best_iter : used_trees_per_class);
  meta += ",\"bestTrainMetric\":" + json_num(best_train);
  meta += ",\"bestValMetric\":" + json_num(best_val);
  meta += ",\"bestTestMetric\":" + json_num(best_test);
  meta += ",\"bestTrainLoss\":" + json_num(best_train);
  meta += ",\"bestValLoss\":" + json_num(best_val);
  meta += ",\"bestTestLoss\":" + json_num(best_test);
  meta += ",\"bestTrainAcc\":" + json_num(best_train_acc);
  meta += ",\"bestValAcc\":" + json_num(best_val_acc);
  meta += ",\"bestTestAcc\":" + json_num(best_test_acc);
  meta += ",\"earlyStop\":" + json_bool(early_stop);
  meta += "}";

  std::string curve = "{";
  curve += "\"steps\":" + json_array_int(curve_steps);
  curve += ",\"train\":" + json_array_f64(curve_train);
  curve += ",\"val\":" + json_array_f64(curve_val);
  curve += ",\"test\":" + json_array_f64(curve_test);
  curve += ",\"trainAcc\":" + json_array_f64(curve_train_acc);
  curve += ",\"valAcc\":" + json_array_f64(curve_val_acc);
  curve += ",\"testAcc\":" + json_array_f64(curve_test_acc);
  curve += "}";

  TrainOut out;
  out.model_bytes = std::move(model_bytes);
  out.meta_json = std::move(meta);
  out.curve_json = std::move(curve);
  out.features_packed = pack_nft_features("multiclass_classification", ds.feature_names, ds.label_name, ds.classes, {});
  out.gl1f_version = 2;
  return out;
}

static TrainOut train_multilabel_cpp(const Dataset& ds, const LRSchedule& lr_sched_in,
                                     int depth, int max_trees, int min_leaf, int seed, bool early_stop, int patience,
                                     int scaleQ, int bins, const std::string& binning, double split_train, double split_val, bool refit_train_val, int quantile_samples,
                                     const std::string& imb_mode, double imb_cap, bool imb_normalize, const std::vector<double>& pos_weights_manual) {
  int n_rows = ds.n_rows;
  int n_features = ds.n_features;
  int n_labels = ds.n_labels;
  if (n_labels < 2) throw std::runtime_error("Need at least 2 label columns for multilabel");

  auto idx = shuffled_indices(n_rows, seed);
  std::vector<int> train, val, test;
  split_idx(idx, split_train, split_val, train, val, test);

  std::vector<int> train_fit = train;
  if (refit_train_val) train_fit.insert(train_fit.end(), val.begin(), val.end());
  std::vector<int>& train_idx = train_fit;

  std::vector<float> feat_min, feat_max;
  compute_feat_min_max(ds.X, n_rows, n_features, train_fit, feat_min, feat_max);
  std::vector<float> feat_range((size_t)n_features, 0.0f);
  for (int f = 0; f < n_features; f++) {
    float r = feat_max[(size_t)f] - feat_min[(size_t)f];
    feat_range[(size_t)f] = (r > 0) ? r : 0.0f;
  }

  std::optional<std::vector<std::vector<float>>> q_thr;
  if (binning == "quantile") {
    q_thr = compute_quantile_thresholds(ds.X, n_rows, n_features, train_fit, feat_min, feat_range, bins, quantile_samples);
  }

  // pos counts on train_idx
  std::vector<int64_t> pos_count((size_t)n_labels, 0);
  for (int r : train_idx) {
    size_t base = (size_t)r * (size_t)n_labels;
    for (int k = 0; k < n_labels; k++) {
      if (ds.y_flat[base + (size_t)k] >= 0.5f) pos_count[(size_t)k] += 1;
    }
  }

  // imbalance per-label pos weights
  std::optional<std::vector<float>> pos_w_opt;
  const std::vector<float>* pos_w = nullptr;
  double w_scale = 1.0;
  std::string imb = lower_copy(trim_view(imb_mode));
  if (imb == "auto" || imb == "manual") {
    std::vector<float> pos_w_v((size_t)n_labels, 1.0f);
    if (imb == "manual" && (int)pos_weights_manual.size() >= n_labels) {
      for (int k = 0; k < n_labels; k++) {
        double w = pos_weights_manual[(size_t)k];
        if (!(w > 0)) w = 1.0;
        pos_w_v[(size_t)k] = (float)w;
      }
    } else {
      for (int k = 0; k < n_labels; k++) {
        int64_t pos = pos_count[(size_t)k];
        int64_t neg = (int64_t)train_idx.size() - pos;
        double w = (pos > 0) ? ((double)neg / (double)pos) : 1.0;
        pos_w_v[(size_t)k] = (float)w;
      }
    }
    double cap = (imb_cap > 0) ? imb_cap : 20.0;
    for (int k = 0; k < n_labels; k++) {
      double w = (double)pos_w_v[(size_t)k];
      if (w < 0.000001) w = 0.000001;
      if (w > cap) w = cap;
      pos_w_v[(size_t)k] = (float)w;
    }
    if (imb_normalize && !train_idx.empty()) {
      double w_sum = 0.0;
      for (int k = 0; k < n_labels; k++) {
        int64_t pos = pos_count[(size_t)k];
        int64_t neg = (int64_t)train_idx.size() - pos;
        w_sum += (double)neg + (double)pos_w_v[(size_t)k] * (double)pos;
      }
      double avg = w_sum / ((double)train_idx.size() * (double)n_labels);
      if (avg > 0) w_scale = 1.0 / avg;
    }
    pos_w_opt = std::move(pos_w_v);
    pos_w = &(*pos_w_opt);
  }

  // base logits per label
  std::vector<int32_t> base_logits_q((size_t)n_labels, 0);
  for (int k = 0; k < n_labels; k++) {
    int64_t pos = pos_count[(size_t)k];
    int64_t neg = (int64_t)train_idx.size() - pos;
    double p0 = 0.5;
    if (!train_idx.empty()) {
      if (pos_w) {
        double num = (double)(*pos_w)[(size_t)k] * (double)pos;
        double den = (double)neg + (double)(*pos_w)[(size_t)k] * (double)pos;
        p0 = num / std::max(1e-12, den);
      } else {
        p0 = (double)pos / std::max<int64_t>(1, (int64_t)train_idx.size());
      }
    }
    double eps = 1e-6;
    p0 = std::min(std::max(p0, eps), 1.0 - eps);
    double base_logit = std::log(p0 / (1.0 - p0));
    base_logits_q[(size_t)k] = clamp_i32(js_round_double(base_logit * (double)scaleQ));
  }

  std::vector<double> predQ_flat((size_t)n_rows * (size_t)n_labels, 0.0);
  for (int r = 0; r < n_rows; r++) {
    size_t base = (size_t)r * (size_t)n_labels;
    for (int k = 0; k < n_labels; k++) predQ_flat[base + (size_t)k] = (double)base_logits_q[(size_t)k];
  }

  std::vector<float> grad((size_t)n_rows, 0.0f);
  std::vector<float> hess((size_t)n_rows, 0.0f);

  XorShift32 rng((uint32_t)(seed ^ 0x9E3779B9));
  std::vector<std::vector<Tree>> trees_by_label((size_t)n_labels);
  for (int k = 0; k < n_labels; k++) trees_by_label[(size_t)k].reserve((size_t)max_trees);

  std::vector<int> curve_steps;
  std::vector<double> curve_train, curve_val, curve_test;
  std::vector<double> curve_train_acc, curve_val_acc, curve_test_acc;

  double best_val = std::numeric_limits<double>::infinity();
  double best_train = std::numeric_limits<double>::infinity();
  double best_test = std::numeric_limits<double>::infinity();
  double best_train_acc = 0.0, best_val_acc = 0.0, best_test_acc = 0.0;
  int best_iter = 0;
  int since_best = 0;

  LRSchedule lr_sched = lr_sched_in;

  for (int t = 1; t <= max_trees; t++) {
    double lr_used = lr_sched.lr_for_iter(t);

    for (int k = 0; k < n_labels; k++) {
      // grad/hess for label k
      for (int r : train_idx) {
        size_t pos = (size_t)r * (size_t)n_labels + (size_t)k;
        double logit = predQ_flat[pos] / (double)scaleQ;
        double p = sigmoid(logit);
        double yk = (ds.y_flat[pos] >= 0.5f) ? 1.0 : 0.0;
        double w = 1.0;
        if (pos_w) {
          double pw = (double)(*pos_w)[(size_t)k];
          if (yk >= 0.5) w = pw;
          w *= w_scale;
        }
        grad[(size_t)r] = (float)((yk - p) * w);
        hess[(size_t)r] = (float)((p * (1.0 - p)) * w);
      }

      Tree tree = build_tree_binary(ds.X, n_rows, n_features, train_idx, grad, hess,
                                    feat_min, feat_range, depth, min_leaf, lr_used, scaleQ, rng, bins, binning,
                                    q_thr ? &(*q_thr) : nullptr);
      trees_by_label[(size_t)k].push_back(tree);

      apply_tree_vector(tree, ds.X, n_features, train_idx, predQ_flat, depth, scaleQ, n_labels, k);
      apply_tree_vector(tree, ds.X, n_features, val, predQ_flat, depth, scaleQ, n_labels, k);
      apply_tree_vector(tree, ds.X, n_features, test, predQ_flat, depth, scaleQ, n_labels, k);
    }

    auto [train_loss, train_acc] = logloss_acc_multilabel(ds.y_flat, predQ_flat, train_idx, n_labels, scaleQ, pos_w, w_scale);
    auto [val_loss, val_acc] = logloss_acc_multilabel(ds.y_flat, predQ_flat, val, n_labels, scaleQ, pos_w, w_scale);
    auto [test_loss, test_acc] = logloss_acc_multilabel(ds.y_flat, predQ_flat, test, n_labels, scaleQ, pos_w, w_scale);

    curve_steps.push_back(t);
    curve_train.push_back(train_loss);
    curve_val.push_back(val_loss);
    curve_test.push_back(test_loss);
    curve_train_acc.push_back(train_acc);
    curve_val_acc.push_back(val_acc);
    curve_test_acc.push_back(test_acc);

    bool improved = false;
    if (val_loss + 1e-12 < best_val) {
      best_val = val_loss;
      best_train = train_loss;
      best_test = test_loss;
      best_train_acc = train_acc;
      best_val_acc = val_acc;
      best_test_acc = test_acc;
      best_iter = t;
      since_best = 0;
      improved = true;
    } else {
      since_best += 1;
    }

    lr_sched.after_metric(improved);
    if (early_stop && since_best >= patience) break;
  }

  int iters_done = trees_by_label.empty() ? 0 : (int)trees_by_label[0].size();
  int used_trees_per_label = early_stop ? std::max(1, best_iter) : std::max(1, iters_done);

  std::vector<std::vector<Tree>> final_trees_by_label((size_t)n_labels);
  for (int k = 0; k < n_labels; k++) {
    auto& arr = trees_by_label[(size_t)k];
    if ((int)arr.size() < used_trees_per_label) used_trees_per_label = (int)arr.size();
  }
  used_trees_per_label = std::max(1, used_trees_per_label);
  for (int k = 0; k < n_labels; k++) {
    auto& arr = trees_by_label[(size_t)k];
    final_trees_by_label[(size_t)k].assign(arr.begin(), arr.begin() + used_trees_per_label);
  }

  auto model_bytes = serialize_model_v2(n_features, depth, n_labels, used_trees_per_label, base_logits_q, (uint32_t)scaleQ, final_trees_by_label);

  std::string meta = "{";
  meta += "\"task\":\"multilabel_classification\"";
  meta += ",\"metricName\":\"LogLoss\"";
  meta += ",\"nFeatures\":" + json_int(n_features);
  meta += ",\"depth\":" + json_int(depth);
  meta += ",\"scaleQ\":" + json_int(scaleQ);
  meta += ",\"bins\":" + json_int(bins);
  meta += ",\"binning\":" + json_str(binning);
  meta += ",\"nClasses\":" + json_int(n_labels);
  meta += ",\"maxTrees\":" + json_int(max_trees);
  meta += ",\"usedTrees\":" + json_int(used_trees_per_label);
  meta += ",\"treesPerClass\":" + json_int(used_trees_per_label);
  meta += ",\"totalTrees\":" + json_int((int64_t)used_trees_per_label * (int64_t)n_labels);
  meta += ",\"bestIter\":" + json_int(early_stop ? best_iter : used_trees_per_label);
  meta += ",\"bestTrainMetric\":" + json_num(best_train);
  meta += ",\"bestValMetric\":" + json_num(best_val);
  meta += ",\"bestTestMetric\":" + json_num(best_test);
  meta += ",\"bestTrainLoss\":" + json_num(best_train);
  meta += ",\"bestValLoss\":" + json_num(best_val);
  meta += ",\"bestTestLoss\":" + json_num(best_test);
  meta += ",\"bestTrainAcc\":" + json_num(best_train_acc);
  meta += ",\"bestValAcc\":" + json_num(best_val_acc);
  meta += ",\"bestTestAcc\":" + json_num(best_test_acc);
  meta += ",\"earlyStop\":" + json_bool(early_stop);
  meta += "}";

  std::string curve = "{";
  curve += "\"steps\":" + json_array_int(curve_steps);
  curve += ",\"train\":" + json_array_f64(curve_train);
  curve += ",\"val\":" + json_array_f64(curve_val);
  curve += ",\"test\":" + json_array_f64(curve_test);
  curve += ",\"trainAcc\":" + json_array_f64(curve_train_acc);
  curve += ",\"valAcc\":" + json_array_f64(curve_val_acc);
  curve += ",\"testAcc\":" + json_array_f64(curve_test_acc);
  curve += "}";

  TrainOut out;
  out.model_bytes = std::move(model_bytes);
  out.meta_json = std::move(meta);
  out.curve_json = std::move(curve);
  out.features_packed = pack_nft_features("multilabel_classification", ds.feature_names, "", ds.classes, ds.label_names);
  out.gl1f_version = 2;
  return out;
}

// --------------- CLI parsing ---------------
struct Args {
  std::string task;
  std::string out_path;
  bool no_package = false;

  std::string input;
  bool npz = false;
  std::string npz_x_key = "X";
  std::string npz_y_key = "y";
  std::string npy_x;
  std::string npy_y;
  bool mmap = false;

  char delimiter = ',';
  bool delimiter_set = false;
  bool delimiter_auto = true;
  bool no_header = false;
  std::string label_col;
  std::string label_cols;
  std::string feature_cols;
  int limit_rows = 0;
  std::string neg_label;
  std::string pos_label;
  std::string class_labels;

  std::string title;
  std::string description;
  int chain_id = 29;
  int chunk_size = 24000;

  int trees = 200;
  int depth = 4;
  double lr = 0.05;
  int min_leaf = 5;
  int seed = 42;
  bool early_stop = false;
  int patience = 20;
  std::string scaleQ = "auto";
  int bins = 32;
  std::string binning = "linear";
  int quantile_samples = 50000;
  double split_train = 0.7;
  double split_val = 0.2;
  bool refit_train_val = false;

  std::string imbalance_mode = "none";
  double imbalance_cap = 20.0;
  bool imbalance_normalize = false;
  bool stratify = false;
  double w0 = 1.0;
  double w1 = 1.0;
  std::string class_weights;
  std::string pos_weights;

  std::string lr_schedule = "none";
  int lr_patience = 0;
  double lr_drop_pct = 10.0;
  double lr_min = 0.0;
  std::string lr_segments;
};

static void print_usage() {
  std::cerr << R"USAGE(Forest C++ trainer (GL1F/GL1X)
Usage:
  train_gl1f_cpp --task <regression|binary_classification|multiclass_classification|multilabel_classification> \
                --input <data.csv> --out <model.gl1f> [options]

Input:
  --delimiter <auto|char>    CSV delimiter (default: auto; supports comma/semicolon/tab/pipe)
  --no-header                Treat CSV as headerless
  --label-col <name|idx>     Label column (default: last column)
  --label-cols <list>        (multilabel) comma-separated label columns
  --feature-cols <list>      comma-separated feature column names/idx
  --limit-rows <N>           Limit number of data rows

Training:
  --trees <N> --depth <N> --lr <float> --min-leaf <N> --seed <N>
  --bins <N> --binning <linear|quantile> --quantile-samples <N>
  --split-train <f> --split-val <f> --refit-train-val
  --early-stop --patience <N>

Packaging (GL1X footer):
  --no-package               Do not append GL1X JSON footer
  --title <text> --description <text> --chain-id <id> --chunk-size <bytes>

Misc:
  -h, --help                 Show this help
)USAGE";
}

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    if (k == "--help" || k == "-h") { print_usage(); std::exit(0); }
    auto need = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
      return std::string(argv[++i]);
    };
    if (k == "--task") a.task = need(k);
    else if (k == "--out") a.out_path = need(k);
    else if (k == "--no-package") a.no_package = true;
    else if (k == "--input") a.input = need(k);
    else if (k == "--delimiter") {
      std::string v = need(k);
      a.delimiter_set = true;
      std::string t = lower_copy(trim_view(v));
      if (t == "auto") {
        a.delimiter_auto = true;
      } else if (t == "\\t" || t == "tab" || t == "tsv") {
        a.delimiter = '\t';
        a.delimiter_auto = false;
      } else if (t == "semicolon" || t == ";") {
        a.delimiter = ';';
        a.delimiter_auto = false;
      } else if (t == "comma" || t == ",") {
        a.delimiter = ',';
        a.delimiter_auto = false;
      } else if (t == "pipe" || t == "|") {
        a.delimiter = '|';
        a.delimiter_auto = false;
      } else if (!v.empty()) {
        a.delimiter = v[0];
        a.delimiter_auto = false;
      }
    }
    else if (k == "--no-header") a.no_header = true;
    else if (k == "--label-col") a.label_col = need(k);
    else if (k == "--label-cols") a.label_cols = need(k);
    else if (k == "--feature-cols") a.feature_cols = need(k);
    else if (k == "--limit-rows") a.limit_rows = std::stoi(need(k));
    else if (k == "--neg-label") a.neg_label = need(k);
    else if (k == "--pos-label") a.pos_label = need(k);
    else if (k == "--class-labels") a.class_labels = need(k);

    else if (k == "--title") a.title = need(k);
    else if (k == "--description") a.description = need(k);
    else if (k == "--desc") a.description = need(k); // server bug-compat
    else if (k == "--chain-id") a.chain_id = std::stoi(need(k));
    else if (k == "--chunk-size") a.chunk_size = std::stoi(need(k));

    else if (k == "--trees") a.trees = std::stoi(need(k));
    else if (k == "--depth") a.depth = std::stoi(need(k));
    else if (k == "--lr") a.lr = std::stod(need(k));
    else if (k == "--min-leaf") a.min_leaf = std::stoi(need(k));
    else if (k == "--seed") a.seed = std::stoi(need(k));
    else if (k == "--early-stop") a.early_stop = true;
    else if (k == "--patience") a.patience = std::stoi(need(k));
    else if (k == "--scaleQ") a.scaleQ = need(k);
    else if (k == "--bins") a.bins = std::stoi(need(k));
    else if (k == "--binning") a.binning = need(k);
    else if (k == "--quantile-samples") a.quantile_samples = std::stoi(need(k));
    else if (k == "--split-train") a.split_train = std::stod(need(k));
    else if (k == "--split-val") a.split_val = std::stod(need(k));
    else if (k == "--refit-train-val") a.refit_train_val = true;

    else if (k == "--imbalance-mode") a.imbalance_mode = need(k);
    else if (k == "--imbalance-cap") a.imbalance_cap = std::stod(need(k));
    else if (k == "--imbalance-normalize") a.imbalance_normalize = true;
    else if (k == "--stratify") a.stratify = true;
    else if (k == "--w0") a.w0 = std::stod(need(k));
    else if (k == "--w1") a.w1 = std::stod(need(k));
    else if (k == "--class-weights") a.class_weights = need(k);
    else if (k == "--pos-weights") a.pos_weights = need(k);

    else if (k == "--lr-schedule") a.lr_schedule = need(k);
    else if (k == "--lr-patience") a.lr_patience = std::stoi(need(k));
    else if (k == "--lr-drop-pct") a.lr_drop_pct = std::stod(need(k));
    else if (k == "--lr-min") a.lr_min = std::stod(need(k));
    else if (k == "--lr-segments") a.lr_segments = need(k);

    else {
      throw std::runtime_error("Unknown arg: " + k);
    }
  }
  if (a.task.empty() || a.out_path.empty()) {
    print_usage();
    throw std::runtime_error("Missing required --task/--out");
  }
  if (a.input.empty() && (a.npy_x.empty() || a.npy_y.empty())) {
    print_usage();
    throw std::runtime_error("Provide --input (CSV) (NPY/NPZ unsupported in C++ build)");
  }
  return a;
}

static std::string normalize_task(const std::string& raw) {
  std::string s = trim_copy(raw);
  if (s == "binary") s = "binary_classification";
  if (s == "multiclass") s = "multiclass_classification";
  if (s == "multilabel") s = "multilabel_classification";
  if (s != "regression" && s != "binary_classification" && s != "multiclass_classification" && s != "multilabel_classification") s = "regression";
  return s;
}

int main(int argc, char** argv) {
  try {
    std::setlocale(LC_NUMERIC, "C");
    Args args = parse_args(argc, argv);
    std::string task = normalize_task(args.task);

    if (args.npz || !args.npy_x.empty() || !args.npy_y.empty()) {
      throw std::runtime_error("NPY/NPZ input is not implemented in train_gl1f_cpp (CSV only).");
    }

    std::vector<std::string> label_cols = split_list(args.label_cols, ',');
    std::vector<std::string> feat_cols = split_list(args.feature_cols, ',');
    std::vector<std::string> cls_labels = split_list(args.class_labels, ',');

    Dataset ds = load_from_csv(
      args.input,
      task,
      args.label_col,
      label_cols,
      feat_cols,
      args.delimiter,
      args.delimiter_auto,
      !args.no_header,
      args.limit_rows,
      args.neg_label,
      args.pos_label,
      cls_labels
    );

    if (ds.n_rows < 3) throw std::runtime_error("Need at least 3 rows for train/val/test split");

    // derive n_classes
    int n_classes = 0;
    if (task == "multiclass_classification") n_classes = (int)ds.classes.size();
    else if (task == "multilabel_classification") n_classes = ds.n_labels;
    else if (task == "binary_classification") n_classes = 2;

    // scaleQ
    int scaleQ = 1;
    std::string scaleQ_raw = lower_copy(trim_view(args.scaleQ));
    if (scaleQ_raw == "auto" || scaleQ_raw == "0") {
      double maxx = max_abs_x(ds.X);
      double maxy = (task == "regression") ? max_abs_y(ds.y) : 0.0;
      scaleQ = choose_scale_q(task, maxx, maxy);
    } else {
      scaleQ = (int)std::floor(std::stod(args.scaleQ));
      if (scaleQ < 1) scaleQ = 1;
      if ((uint64_t)scaleQ > 0xFFFFFFFFull) scaleQ = (int)0xFFFFFFFFull;
    }

    // clamp bins
    int bins = std::max(8, std::min(512, args.bins));

    LRSchedule lr_sched = make_lr_schedule(args.lr_schedule, args.lr, args.trees, args.lr_patience, args.lr_drop_pct, args.lr_min, args.lr_segments);

    // Parse manual weights lists
    std::vector<double> class_weights_manual;
    if (!args.class_weights.empty()) {
      for (auto& s : split_list(args.class_weights, ',')) {
        if (s.empty()) continue;
        try { class_weights_manual.push_back(std::stod(s)); } catch (...) {}
      }
    }
    std::vector<double> pos_weights_manual;
    if (!args.pos_weights.empty()) {
      for (auto& s : split_list(args.pos_weights, ',')) {
        if (s.empty()) continue;
        try { pos_weights_manual.push_back(std::stod(s)); } catch (...) {}
      }
    }

    TrainOut out;
    if (task == "regression") {
      out = train_regression_cpp(ds, lr_sched, args.depth, args.trees, args.min_leaf, args.seed, args.early_stop, args.patience,
                                 scaleQ, bins, lower_copy(trim_view(args.binning)), args.split_train, args.split_val, args.refit_train_val, args.quantile_samples);
    } else if (task == "binary_classification") {
      out = train_binary_cpp(ds, lr_sched, args.depth, args.trees, args.min_leaf, args.seed, args.early_stop, args.patience,
                             scaleQ, bins, lower_copy(trim_view(args.binning)), args.split_train, args.split_val, args.refit_train_val, args.quantile_samples,
                             args.imbalance_mode, args.imbalance_cap, args.imbalance_normalize, args.stratify, args.w0, args.w1);
    } else if (task == "multiclass_classification") {
      out = train_multiclass_cpp(ds, lr_sched, args.depth, args.trees, args.min_leaf, args.seed, args.early_stop, args.patience,
                                 scaleQ, bins, lower_copy(trim_view(args.binning)), args.split_train, args.split_val, args.refit_train_val, args.quantile_samples,
                                 args.imbalance_mode, args.imbalance_cap, args.imbalance_normalize, args.stratify, class_weights_manual);
    } else if (task == "multilabel_classification") {
      out = train_multilabel_cpp(ds, lr_sched, args.depth, args.trees, args.min_leaf, args.seed, args.early_stop, args.patience,
                                 scaleQ, bins, lower_copy(trim_view(args.binning)), args.split_train, args.split_val, args.refit_train_val, args.quantile_samples,
                                 args.imbalance_mode, args.imbalance_cap, args.imbalance_normalize, pos_weights_manual);
    } else {
      throw std::runtime_error("Unsupported task: " + task);
    }

    std::vector<uint8_t> bytes_to_write = out.model_bytes;
    if (!args.no_package) {
      // Build a minimal package JSON with local trainMeta + curve.
      std::string createdAt = now_iso_utc();
      std::string pkg = "{";
      pkg += "\"kind\":\"GL1F_PACKAGE\"";
      pkg += ",\"v\":1";
      pkg += ",\"createdAt\":" + json_str(createdAt);
      pkg += ",\"chainId\":" + json_int(args.chain_id);
      pkg += ",\"chunkSize\":" + json_int(args.chunk_size);

      pkg += ",\"model\":{";
      pkg += "\"gl1fVersion\":" + json_int(out.gl1f_version);
      pkg += ",\"nFeatures\":" + json_int(ds.n_features);
      pkg += ",\"depth\":" + json_int(args.depth);
      if (out.gl1f_version == 1) {
        // v1 header already contains baseQ; but meta already includes it.
      } else {
        pkg += ",\"nClasses\":" + json_int(ds.n_labels > 0 ? ds.n_labels : (int)ds.classes.size());
      }
      pkg += ",\"scaleQ\":" + json_int(scaleQ);
      pkg += ",\"bytes\":" + json_int((int64_t)out.model_bytes.size());
      pkg += "}";

      // NFT meta
      pkg += ",\"nft\":{";
      pkg += "\"title\":" + json_str(args.title);
      pkg += ",\"description\":" + json_str(args.description);
      pkg += ",\"iconPngB64\":null";
      pkg += ",\"featuresPacked\":" + json_str(out.features_packed);
      pkg += ",\"titleWordHashes\":[]";
      pkg += "}";

      // local train info
      pkg += ",\"local\":{";
      pkg += "\"trainMeta\":" + out.meta_json;
      pkg += ",\"curve\":" + out.curve_json;
      pkg += "}";

      pkg += "}";

      bytes_to_write = append_gl1x_footer(out.model_bytes, pkg);
    }

    // Write file
    std::ofstream of(args.out_path, std::ios::binary);
    if (!of) throw std::runtime_error("Failed to open output: " + args.out_path);
    of.write((const char*)bytes_to_write.data(), (std::streamsize)bytes_to_write.size());
    of.close();

    std::cerr << "OK: wrote " << bytes_to_write.size() << " bytes to " << args.out_path << "\n";
    std::cerr << "Rows: " << ds.n_rows << " (dropped " << ds.dropped_rows << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 2;
  }
}
