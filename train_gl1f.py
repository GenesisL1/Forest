#!/usr/bin/env python3
"""
train_gl1f.py — Train GenesisL1 Forest models in Python and export .gl1f files.

Goals:
- Minimal dependencies: requires only numpy.
- Output format matches the UI preview loader:
  - Model bytes are "GL1F" v1 (scalar: regression/binary) or v2 (vector: multiclass/multilabel)
  - Optional "GL1X" footer contains JSON package metadata (featuresPacked, etc.)

Usage examples
--------------

1) Regression from CSV (label column name "y"):

  python3 train_gl1f.py \
    --task regression \
    --input data.csv \
    --label-col y \
    --out model.gl1f \
    --trees 400 --depth 5 --lr 0.05 --min-leaf 10 \
    --early-stop --patience 25 \
    --scaleQ auto

2) Binary classification (auto infer 2 labels):

  python3 train_gl1f.py \
    --task binary_classification \
    --input data.csv \
    --label-col outcome \
    --out model.gl1f \
    --trees 300 --depth 4 --lr 0.05 --min-leaf 20 \
    --imbalance-mode auto --imbalance-normalize

3) Multiclass classification (infer classes):

  python3 train_gl1f.py \
    --task multiclass_classification \
    --input iris.csv \
    --label-col species \
    --out iris.gl1f \
    --trees 120 --depth 3 --lr 0.1 --min-leaf 5

4) Multilabel classification:

  python3 train_gl1f.py \
    --task multilabel_classification \
    --input toxic.csv \
    --label-cols spam,toxic,nsfw \
    --out toxic.gl1f \
    --trees 120 --depth 3 --lr 0.1 --min-leaf 5

For giant datasets:
- Prefer using binary arrays (.npy/.npz) or a memmap pipeline for X/y. This script supports:
  --npz (expects keys X and y by default) or
  --npy-x / --npy-y (memory-mapped with --mmap).

Notes
-----
This trainer is a faithful port of the browser trainer (src/train_worker.js):
- Fixed-depth trees
- Histogram binning for split candidates (linear or quantile)
- Regression: squared loss
- Binary: logistic loss (Newton leaf weights)
- Multiclass: softmax cross-entropy via per-class Newton trees (v2)
- Multilabel: independent sigmoids per label (v2)

Copyright (c) 2026 Decentralized Science Labs
MIT License.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires numpy. Install with: pip install numpy") from e


# -------------------------
# Constants / helpers
# -------------------------

INT32_MAX = 2147483647
INT32_MIN = -2147483648
INT32_SAFE = 2147480000  # used by UI for auto scaleQ selection
DEFAULT_SCALE_Q = 1_000_000

PACKAGE_MAGIC = b"GL1X"
MODEL_MAGIC = b"GL1F"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def js_round(x: float) -> int:
    """Match JavaScript Math.round for finite values.

    JS: Math.round(x) == floor(x + 0.5) for all x (with -0 edge which is irrelevant for ints).
    """
    return int(math.floor(x + 0.5))


def clamp_i32(x: int) -> int:
    if x > INT32_MAX:
        return INT32_MAX
    if x < INT32_MIN:
        return INT32_MIN
    return int(x)


def quantize_to_i32(arr_f: np.ndarray, scaleQ: int) -> np.ndarray:
    """Vectorized quantization: q = clamp_i32(Math.round(x * scaleQ)). Returns int32."""
    q = np.floor(arr_f.astype(np.float64) * float(scaleQ) + 0.5)
    q = np.clip(q, INT32_MIN, INT32_MAX)
    return q.astype(np.int32, copy=False)


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid for numpy arrays."""
    z = z.astype(np.float64, copy=False)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


# -------------------------
# RNG compatible with JS xorshift32
# -------------------------

class XorShift32:
    __slots__ = ("x",)

    def __init__(self, seed: int):
        x = int(seed) & 0xFFFFFFFF
        if x == 0:
            x = 123456789
        self.x = x

    def next_u32(self) -> int:
        x = self.x
        x ^= ((x << 13) & 0xFFFFFFFF)
        x ^= ((x >> 17) & 0xFFFFFFFF)
        x ^= ((x << 5) & 0xFFFFFFFF)
        self.x = x & 0xFFFFFFFF
        return self.x


def shuffled_indices(n: int, seed: int) -> np.ndarray:
    rng = XorShift32(seed)
    idx = np.arange(n, dtype=np.uint32)
    # Fisher-Yates
    for i in range(n - 1, 0, -1):
        j = int(rng.next_u32() % (i + 1))
        tmp = idx[i]
        idx[i] = idx[j]
        idx[j] = tmp
    return idx


def split_idx(idx: np.ndarray, frac_train: float = 0.7, frac_val: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(idx.shape[0])
    n_train = int(math.floor(n * frac_train))
    n_val = int(math.floor(n * frac_val))
    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = max(1, n - n_train - n_val)
    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:n_train + n_val + n_test]
    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)


def split_idx_stratified_by_class(
    idx: np.ndarray,
    yK: np.ndarray,
    n_classes: int,
    frac_train: float = 0.7,
    frac_val: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    buckets: List[List[int]] = [[] for _ in range(int(n_classes))]
    for r_u in idx.tolist():
        r = int(r_u)
        k = int(yK[r])
        if not (0 <= k < n_classes):
            k = 0
        buckets[k].append(r)

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []

    for k in range(int(n_classes)):
        arr = buckets[k]
        n = len(arr)
        if n <= 0:
            continue

        n_train = int(math.floor(n * frac_train))
        n_val = int(math.floor(n * frac_val))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
            if n_train + n_val >= n:
                n_train = max(0, n - n_val - 1)

        train.extend(arr[:n_train])
        val.extend(arr[n_train:n_train + n_val])
        test.extend(arr[n_train + n_val:])

    if len(train) < 1 or len(val) < 1 or len(test) < 1:
        return split_idx(idx, frac_train, frac_val)

    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64), np.asarray(test, dtype=np.int64)


# -------------------------
# Metrics
# -------------------------

def mse_q(yQ: np.ndarray, predQ: np.ndarray, indices: np.ndarray, scaleQ: int) -> float:
    if indices.size == 0:
        return float("nan")
    diff = (yQ[indices] - predQ[indices]) / float(scaleQ)
    return float(np.mean(diff * diff, dtype=np.float64))


def logloss_acc_binary(
    y01: np.ndarray,
    predQ: np.ndarray,
    indices: np.ndarray,
    scaleQ: int,
    w_row: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    if indices.size == 0:
        return float("nan"), float("nan")
    eps = 1e-12
    y = (y01[indices] >= 0.5).astype(np.float64)
    logit = predQ[indices] / float(scaleQ)
    p = sigmoid_np(logit)
    p = np.clip(p, eps, 1.0 - eps)
    if w_row is None:
        w = np.ones_like(p, dtype=np.float64)
    else:
        w = w_row[indices].astype(np.float64, copy=False)
        w = np.where(w > 0, w, 0.0)
    w_sum = float(np.sum(w, dtype=np.float64)) or 1.0
    loss = float(np.sum(w * (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))), dtype=np.float64) / w_sum)
    pred = (p >= 0.5).astype(np.float64)
    acc = float(np.sum(w * (pred == y), dtype=np.float64) / w_sum)
    return loss, acc


def softmax_probs_inplace(predQ_flat: np.ndarray, n_rows: int, n_classes: int, scaleQ: int, out_prob: np.ndarray) -> None:
    """Fill out_prob (float32) with softmax probabilities computed from predQ_flat (float64, Q-units)."""
    mat = predQ_flat.reshape(n_rows, n_classes)
    # subtract max per row for stability
    max_z = np.max(mat, axis=1, keepdims=True) / float(scaleQ)
    z = (mat / float(scaleQ)) - max_z
    np.exp(z, out=out_prob)  # casts to float32 if out_prob is float32
    s = np.sum(out_prob, axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    out_prob /= s


def logloss_acc_multiclass(
    yK: np.ndarray,
    prob: np.ndarray,
    indices: np.ndarray,
    n_classes: int,
    w_row: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    if indices.size == 0:
        return float("nan"), float("nan")
    eps = 1e-12
    y = yK[indices].astype(np.int64, copy=False)
    p = prob[indices]  # (m, K) copy
    pred = np.argmax(p, axis=1).astype(np.int64)
    correct = (pred == y).astype(np.float64)
    rows = np.arange(p.shape[0], dtype=np.int64)
    py = p[rows, y]
    py = np.clip(py.astype(np.float64, copy=False), eps, 1.0 - eps)
    if w_row is None:
        w = np.ones_like(py, dtype=np.float64)
    else:
        w = w_row[indices].astype(np.float64, copy=False)
        w = np.where(w > 0, w, 0.0)
    w_sum = float(np.sum(w, dtype=np.float64)) or 1.0
    loss = float(np.sum(w * (-np.log(py)), dtype=np.float64) / w_sum)
    acc = float(np.sum(w * correct, dtype=np.float64) / w_sum)
    return loss, acc


def logloss_acc_multilabel(
    y_flat: np.ndarray,
    predQ_flat: np.ndarray,
    indices: np.ndarray,
    n_labels: int,
    scaleQ: int,
    pos_w: Optional[np.ndarray] = None,
    w_scale: float = 1.0,
) -> Tuple[float, float]:
    if indices.size == 0:
        return float("nan"), float("nan")
    eps = 1e-12
    y2 = y_flat.reshape(-1, n_labels)[indices]
    z2 = (predQ_flat.reshape(-1, n_labels)[indices] / float(scaleQ)).astype(np.float64, copy=False)
    p2 = sigmoid_np(z2)
    p2 = np.clip(p2, eps, 1.0 - eps)
    yb = (y2 >= 0.5).astype(np.float64)
    if pos_w is None:
        w = np.ones_like(p2, dtype=np.float64)
    else:
        pos_w2 = pos_w.astype(np.float64, copy=False).reshape(1, n_labels)
        w = np.where(yb >= 0.5, pos_w2, 1.0) * float(w_scale)
    w = np.where(w > 0, w, 0.0)
    w_sum = float(np.sum(w, dtype=np.float64)) or 1.0
    loss = float(np.sum(w * (-(yb * np.log(p2) + (1.0 - yb) * np.log(1.0 - p2))), dtype=np.float64) / w_sum)
    pred = (p2 >= 0.5).astype(np.float64)
    acc = float(np.sum(w * (pred == yb), dtype=np.float64) / w_sum)
    return loss, acc


# -------------------------
# Model / tree format (GL1F)
# -------------------------

@dataclass
class Tree:
    feat: np.ndarray  # uint16 (internal nodes)
    thr: np.ndarray   # int32  (internal nodes)
    leaf: np.ndarray  # int32  (leaves)


def serialize_model_v1(n_features: int, depth: int, trees: Sequence[Tree], baseQ: int, scaleQ: int) -> bytes:
    n_trees = int(len(trees))
    pow2 = 1 << int(depth)
    internal = pow2 - 1
    per_tree = internal * 8 + pow2 * 4
    total_bytes = 24 + n_trees * per_tree

    if not (0 <= n_features <= 0xFFFF):
        raise ValueError(f"nFeatures={n_features} exceeds uint16 limit")
    if not (0 <= scaleQ <= 0xFFFFFFFF):
        raise ValueError(f"scaleQ={scaleQ} exceeds uint32 limit")

    out = bytearray(total_bytes)
    out[0:4] = MODEL_MAGIC
    out[4] = 1
    out[5] = 0
    struct.pack_into("<H", out, 6, int(n_features))
    struct.pack_into("<H", out, 8, int(depth))
    struct.pack_into("<I", out, 10, int(n_trees))
    struct.pack_into("<i", out, 14, int(baseQ))
    struct.pack_into("<I", out, 18, int(scaleQ))
    out[22] = 0
    out[23] = 0

    node_dtype = np.dtype([("feat", "<u2"), ("thr", "<i4"), ("res", "<u2")])

    off = 24
    for tr in trees:
        feat = np.asarray(tr.feat, dtype=np.uint16)
        thr = np.asarray(tr.thr, dtype=np.int32)
        leaf = np.asarray(tr.leaf, dtype=np.int32)

        if feat.shape[0] != internal or thr.shape[0] != internal or leaf.shape[0] != pow2:
            raise ValueError("Tree shape mismatch for this depth")

        nodes = np.empty(internal, dtype=node_dtype)
        nodes["feat"] = feat
        nodes["thr"] = thr
        nodes["res"] = 0
        blob_nodes = nodes.tobytes()
        out[off:off + internal * 8] = blob_nodes
        off += internal * 8

        blob_leaf = leaf.astype("<i4", copy=False).tobytes()
        out[off:off + pow2 * 4] = blob_leaf
        off += pow2 * 4

    return bytes(out)


def serialize_model_v2(
    n_features: int,
    depth: int,
    n_classes: int,
    trees_per_class: int,
    base_logits_q: np.ndarray,
    scaleQ: int,
    trees_by_class: Sequence[Sequence[Tree]],
) -> bytes:
    pow2 = 1 << int(depth)
    internal = pow2 - 1
    per_tree = internal * 8 + pow2 * 4
    header_size = 24 + int(n_classes) * 4
    total_trees = int(trees_per_class) * int(n_classes)
    total_bytes = header_size + total_trees * per_tree

    if not (0 <= n_features <= 0xFFFF):
        raise ValueError(f"nFeatures={n_features} exceeds uint16 limit")
    if not (2 <= n_classes <= 0xFFFF):
        raise ValueError(f"nClasses={n_classes} invalid (must be 2..65535)")
    if not (0 <= scaleQ <= 0xFFFFFFFF):
        raise ValueError(f"scaleQ={scaleQ} exceeds uint32 limit")

    out = bytearray(total_bytes)
    out[0:4] = MODEL_MAGIC
    out[4] = 2
    out[5] = 0
    struct.pack_into("<H", out, 6, int(n_features))
    struct.pack_into("<H", out, 8, int(depth))
    struct.pack_into("<I", out, 10, int(trees_per_class))
    struct.pack_into("<i", out, 14, 0)  # reserved
    struct.pack_into("<I", out, 18, int(scaleQ))
    struct.pack_into("<H", out, 22, int(n_classes))

    off = 24
    base_logits_q = np.asarray(base_logits_q, dtype=np.int32)
    if base_logits_q.shape[0] != int(n_classes):
        raise ValueError("base_logits_q length mismatch")
    out[off:off + int(n_classes) * 4] = base_logits_q.astype("<i4", copy=False).tobytes()
    off += int(n_classes) * 4

    node_dtype = np.dtype([("feat", "<u2"), ("thr", "<i4"), ("res", "<u2")])

    # Trees are class-major
    for k in range(int(n_classes)):
        cls_trees = trees_by_class[k]
        if len(cls_trees) < int(trees_per_class):
            raise ValueError(f"class {k} has {len(cls_trees)} trees < treesPerClass={trees_per_class}")
        for t in range(int(trees_per_class)):
            tr = cls_trees[t]
            feat = np.asarray(tr.feat, dtype=np.uint16)
            thr = np.asarray(tr.thr, dtype=np.int32)
            leaf = np.asarray(tr.leaf, dtype=np.int32)

            if feat.shape[0] != internal or thr.shape[0] != internal or leaf.shape[0] != pow2:
                raise ValueError("Tree shape mismatch for this depth")

            nodes = np.empty(internal, dtype=node_dtype)
            nodes["feat"] = feat
            nodes["thr"] = thr
            nodes["res"] = 0
            out[off:off + internal * 8] = nodes.tobytes()
            off += internal * 8

            out[off:off + pow2 * 4] = leaf.astype("<i4", copy=False).tobytes()
            off += pow2 * 4

    return bytes(out)


def build_gl1x_footer(pkg_obj: Dict[str, Any]) -> bytes:
    payload = json.dumps(pkg_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header = bytearray(12)
    header[0:4] = PACKAGE_MAGIC
    header[4] = 1
    header[5] = 0
    header[6] = 0
    header[7] = 0
    struct.pack_into("<I", header, 8, len(payload))
    return bytes(header) + payload


# -------------------------
# FeaturesPacked (on-chain metadata string)
# -------------------------

_TASK_ALIASES = {
    "binary": "binary_classification",
    "classification": "binary_classification",
    "multiclass": "multiclass_classification",
    "multilabel": "multilabel_classification",
}


def normalize_task(raw: str) -> str:
    s = str(raw or "regression").strip()
    s = _TASK_ALIASES.get(s, s)
    if s not in ("regression", "binary_classification", "multiclass_classification", "multilabel_classification"):
        return "regression"
    return s


def pack_nft_features(
    task: str,
    feature_names: Sequence[str],
    label_name: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> str:
    t = normalize_task(task)
    meta: Dict[str, Any] = {"v": 1, "task": t}
    if label_name:
        meta["labelName"] = str(label_name)

    if t == "multilabel_classification":
        if label_names is not None and len(label_names) >= 1:
            meta["labelNames"] = [str(x) for x in label_names]
        elif labels is not None and len(labels) >= 1:
            # Back-compat: older callers used labels for output names.
            meta["labelNames"] = [str(x) for x in labels]

        if labels is not None and len(labels) >= 2:
            meta["labels"] = [str(labels[0]), str(labels[1])]
        else:
            meta["labels"] = ["0", "1"]
    elif t in ("binary_classification", "multiclass_classification") and labels is not None and len(labels) >= 2:
        meta["labels"] = [str(x) for x in labels]

    lines = [f"#meta={json.dumps(meta, ensure_ascii=False, separators=(',', ':'))}"]
    for f in feature_names:
        s = str(f or "").strip()
        if s:
            lines.append(s)
    return "\n".join(lines)


# -------------------------
# CSV label parsing helpers (ported from src/csv_parse.js)
# -------------------------

_IS_STRICT_NUMBER_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")


def _is_strict_number(s: str) -> bool:
    s = str(s or "").strip()
    if not s:
        return False
    return bool(_IS_STRICT_NUMBER_RE.match(s))


def _canon_label(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if _is_strict_number(s):
        try:
            n = float(s)
        except Exception:
            return s
        if math.isfinite(n):
            if abs(n - int(n)) < 1e-12:
                return str(int(n))
            # Use a reasonably short representation (JS String(Number(...)) is similar)
            out = repr(n)
            # repr can be "1.23" or "1e-03" etc; accept.
            return out
    return s


def _infer_label_values_csv(path: str, label_index: int, delimiter: str, has_header: bool, limit_rows: Optional[int]) -> Tuple[List[str], Dict[str, int], bool]:
    counts: Dict[str, int] = {}
    all_numeric = True
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            try:
                next(reader)
            except StopIteration:
                return [], {}, True
        for i, row in enumerate(reader):
            if limit_rows is not None and i >= limit_rows:
                break
            if not row:
                continue
            if label_index >= len(row):
                continue
            v = _canon_label(row[label_index])
            if not v:
                continue
            counts[v] = counts.get(v, 0) + 1
            if not _is_strict_number(v):
                all_numeric = False

    values = list(counts.keys())
    if all_numeric:
        tmp = []
        for s in values:
            try:
                n = float(s)
            except Exception:
                continue
            if math.isfinite(n):
                tmp.append((n, s))
        tmp.sort(key=lambda x: x[0])
        values = [s for _, s in tmp]
    else:
        values.sort()
    return values, counts, all_numeric


def _parse_binary01(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    k = s.lower()
    if k in ("0", "0.0"):
        return 0
    if k in ("1", "1.0"):
        return 1
    if k in ("true", "t", "yes", "y"):
        return 1
    if k in ("false", "f", "no", "n"):
        return 0
    if _is_strict_number(k):
        try:
            n = float(k)
        except Exception:
            return None
        if n == 0:
            return 0
        if n == 1:
            return 1
    return None



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


# -------------------------
# Data loading
# -------------------------

def _find_col_index(spec: Union[str, int], headers: Sequence[str]) -> int:
    if isinstance(spec, int):
        return int(spec)
    s = str(spec).strip()
    if not s:
        raise ValueError("Empty column spec")
    if s.lstrip("-").isdigit():
        return int(s)
    # Case-sensitive first, then case-insensitive match
    if s in headers:
        return headers.index(s)
    lower = [h.lower() for h in headers]
    if s.lower() in lower:
        return lower.index(s.lower())
    raise ValueError(f"Column '{s}' not found in CSV headers")


def load_from_csv(
    path: str,
    task: str,
    label_col: Optional[Union[str, int]] = None,
    label_cols: Optional[Sequence[Union[str, int]]] = None,
    feature_cols: Optional[Sequence[Union[str, int]]] = None,
    *,
    delimiter: str = "auto",
    has_header: bool = True,
    limit_rows: Optional[int] = None,
    neg_label: Optional[str] = None,
    pos_label: Optional[str] = None,
    class_labels: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load dataset from CSV and return (X, y, info). y layout depends on task."""
    t = normalize_task(task)

    delim_norm = _normalize_delimiter_arg(delimiter)
    if delim_norm == "auto":
        delim_norm = _autodetect_csv_delimiter(path)
    delimiter = delim_norm

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            first = next(reader)
        except StopIteration:
            raise ValueError("Empty CSV")

    if has_header:
        headers = [str(x or "").strip() for x in first]
        if headers and headers[0].startswith("\ufeff"):
            headers[0] = headers[0].lstrip("\ufeff")
    else:
        headers = [f"c{i}" for i in range(len(first))]

    n_cols = len(headers)

    if t == "multilabel_classification":
        if not label_cols or len(label_cols) < 2:
            raise ValueError("multilabel_classification requires --label-cols with at least 2 columns")
        label_idx = [_find_col_index(c, headers) for c in label_cols]
        # de-dup preserve order
        seen = set()
        label_idx2: List[int] = []
        for i in label_idx:
            if 0 <= i < n_cols and i not in seen:
                label_idx2.append(i)
                seen.add(i)
        if len(label_idx2) < 2:
            raise ValueError("Need at least 2 distinct label columns")
        label_idx = label_idx2
        label_names = [headers[i] or f"label{i}" for i in label_idx]
        n_labels = len(label_idx)

        if feature_cols is None:
            feat_idx = [i for i in range(n_cols) if i not in seen]
        else:
            feat_idx = [_find_col_index(c, headers) for c in feature_cols]
            feat_idx = [i for i in feat_idx if 0 <= i < n_cols and i not in seen]
        if len(feat_idx) < 1:
            raise ValueError("Need at least 1 feature column")

        feature_names = [headers[i] or f"f{i}" for i in feat_idx]

        x_list: List[List[float]] = []
        y_rows: List[List[int]] = []
        dropped = 0
        dropped_label_missing = 0
        dropped_label_invalid = 0
        dropped_bad_feature = 0

        # Iterate full file
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            if has_header:
                try:
                    next(reader)
                except StopIteration:
                    pass
            for r_i, row in enumerate(reader):
                if limit_rows is not None and r_i >= limit_rows:
                    break
                if not row:
                    continue
                # pad
                if len(row) < n_cols:
                    row = row + [""] * (n_cols - len(row))

                bad = False
                y_row: List[int] = []
                for i in label_idx:
                    v = row[i]
                    s = str(v or "").strip()
                    if not s:
                        dropped_label_missing += 1
                        bad = True
                        break
                    b = _parse_binary01(s)
                    if b is None:
                        dropped_label_invalid += 1
                        bad = True
                        break
                    y_row.append(int(b))
                if bad:
                    dropped += 1
                    continue

                x_row: List[float] = []
                for i in feat_idx:
                    try:
                        num = float(row[i])
                    except Exception:
                        dropped_bad_feature += 1
                        bad = True
                        break
                    if not math.isfinite(num):
                        dropped_bad_feature += 1
                        bad = True
                        break
                    x_row.append(float(num))
                if bad:
                    dropped += 1
                    continue

                x_list.append(x_row)
                y_rows.append(y_row)

        if not x_list:
            raise ValueError("No valid rows after parsing")

        X = np.asarray(x_list, dtype=np.float32)
        y_mat = np.asarray(y_rows, dtype=np.float32)  # 0/1
        y_flat = y_mat.reshape(-1).astype(np.float32, copy=False)

        info = {
            "headers": headers,
            "feature_names": feature_names,
            "label_names": label_names,
            "label_name": "(multilabel)",
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_labels": int(n_labels),
            "dropped_rows": int(dropped),
            "dropped_label_missing": int(dropped_label_missing),
            "dropped_label_invalid": int(dropped_label_invalid),
            "dropped_bad_feature": int(dropped_bad_feature),
            "classes": ["0", "1"],
        }
        return X, y_flat, info

    # Non-multilabel tasks: single label column
    if label_col is None:
        raise ValueError("This task requires --label-col")
    label_index = _find_col_index(label_col, headers)
    if not (0 <= label_index < n_cols):
        raise ValueError(f"label-col index out of range: {label_index}")

    if feature_cols is None:
        feat_idx = [i for i in range(n_cols) if i != label_index]
    else:
        feat_idx = [_find_col_index(c, headers) for c in feature_cols]
        feat_idx = [i for i in feat_idx if 0 <= i < n_cols and i != label_index]

    if len(feat_idx) < 1:
        raise ValueError("Need at least 1 feature column")

    feature_names = [headers[i] or f"f{i}" for i in feat_idx]
    label_name = headers[label_index] or "label"

    # Classification label mapping (for string labels)
    classes: Optional[List[str]] = None
    map_label_to_int: Optional[Dict[str, int]] = None

    if t == "binary_classification":
        if neg_label is not None and pos_label is not None:
            neg = _canon_label(neg_label)
            pos = _canon_label(pos_label)
            if not neg or not pos:
                raise ValueError("Both --neg-label and --pos-label must be non-empty")
            if neg == pos:
                raise ValueError("--neg-label and --pos-label must differ")
            classes = [neg, pos]
        else:
            values, _, _ = _infer_label_values_csv(path, label_index, delimiter, has_header, limit_rows)
            if len(values) < 2:
                raise ValueError("Need at least 2 distinct label values for binary classification")
            if len(values) > 2:
                raise ValueError(f"Binary classification found {len(values)} label values; pass --neg-label/--pos-label to choose.")
            classes = [values[0], values[1]]

        map_label_to_int = {classes[0]: 0, classes[1]: 1}

    elif t == "multiclass_classification":
        if class_labels is not None and len(class_labels) >= 2:
            classes = [_canon_label(x) for x in class_labels]
        else:
            values, _, _ = _infer_label_values_csv(path, label_index, delimiter, has_header, limit_rows)
            if len(values) < 2:
                raise ValueError("Need at least 2 distinct label values for multiclass classification")
            classes = values

        # de-dup preserve order
        seen2 = set()
        classes2: List[str] = []
        for c in classes:
            c2 = _canon_label(c)
            if c2 and c2 not in seen2:
                classes2.append(c2)
                seen2.add(c2)
        if len(classes2) < 2:
            raise ValueError("Need at least 2 distinct classes")
        classes = classes2
        map_label_to_int = {c: i for i, c in enumerate(classes)}

    # Parse rows
    x_list: List[List[float]] = []
    y_list: List[float] = []
    dropped = 0
    dropped_other_label = 0
    dropped_bad_feature = 0
    dropped_bad_label = 0

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            try:
                next(reader)
            except StopIteration:
                pass
        for r_i, row in enumerate(reader):
            if limit_rows is not None and r_i >= limit_rows:
                break
            if not row:
                continue
            if len(row) < n_cols:
                row = row + [""] * (n_cols - len(row))

            # label
            if t == "regression":
                try:
                    yy = float(row[label_index])
                except Exception:
                    dropped += 1
                    dropped_bad_label += 1
                    continue
                if not math.isfinite(yy):
                    dropped += 1
                    dropped_bad_label += 1
                    continue
                y_val = float(yy)
            else:
                lab = _canon_label(row[label_index])
                if not lab:
                    dropped += 1
                    dropped_bad_label += 1
                    continue
                if map_label_to_int is None:
                    raise RuntimeError("internal: missing label map")
                cls = map_label_to_int.get(lab, None)
                if cls is None:
                    dropped += 1
                    dropped_other_label += 1
                    continue
                y_val = float(cls)

            # features
            x_row: List[float] = []
            bad = False
            for i in feat_idx:
                try:
                    num = float(row[i])
                except Exception:
                    bad = True
                    break
                if not math.isfinite(num):
                    bad = True
                    break
                x_row.append(float(num))
            if bad:
                dropped += 1
                dropped_bad_feature += 1
                continue

            x_list.append(x_row)
            y_list.append(y_val)

    if not x_list:
        raise ValueError("No valid rows after parsing")

    X = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    info = {
        "headers": headers,
        "feature_names": feature_names,
        "label_name": label_name,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "dropped_rows": int(dropped),
        "dropped_other_label": int(dropped_other_label),
        "dropped_bad_feature": int(dropped_bad_feature),
        "dropped_bad_label": int(dropped_bad_label),
        "classes": classes,  # None for regression
    }
    return X, y, info


def load_from_npz(path: str, x_key: str = "X", y_key: str = "y", mmap: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    mode = "r" if mmap else None
    with np.load(path, allow_pickle=False, mmap_mode=mode) as z:
        if x_key not in z or y_key not in z:
            raise ValueError(f"npz must contain keys '{x_key}' and '{y_key}'")
        X = z[x_key]
        y = z[y_key]
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    info = {
        "feature_names": [f"f{i}" for i in range(int(X.shape[1]))],
        "label_name": "label",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": None,
    }
    return X, y, info


def load_from_npy(path_x: str, path_y: str, mmap: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    mode = "r" if mmap else None
    X = np.load(path_x, mmap_mode=mode, allow_pickle=False)
    y = np.load(path_y, mmap_mode=mode, allow_pickle=False)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    info = {
        "feature_names": [f"f{i}" for i in range(int(X.shape[1]))],
        "label_name": "label",
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": None,
    }
    return X, y, info


# -------------------------
# Training core (ported from src/train_worker.js)
# -------------------------

def sample_features(n_features: int, k: int, rng: XorShift32) -> List[int]:
    k = max(1, min(int(n_features), int(k)))
    used = np.zeros(int(n_features), dtype=np.uint8)
    out: List[int] = []
    while len(out) < k:
        f = int(rng.next_u32() % int(n_features))
        if used[f]:
            continue
        used[f] = 1
        out.append(f)
    return out


def _compute_feat_min_max(X: np.ndarray, rows: np.ndarray, chunk: int = 65536) -> Tuple[np.ndarray, np.ndarray]:
    n_features = int(X.shape[1])
    feat_min = np.full(n_features, np.inf, dtype=np.float32)
    feat_max = np.full(n_features, -np.inf, dtype=np.float32)

    n = int(rows.shape[0])
    for start in range(0, n, chunk):
        sl = rows[start:start + chunk]
        if sl.size == 0:
            continue
        block = X[sl]  # advanced indexing copy, bounded by chunk size
        feat_min = np.minimum(feat_min, np.min(block, axis=0))
        feat_max = np.maximum(feat_max, np.max(block, axis=0))

    return feat_min, feat_max


def _compute_quantile_thresholds(
    X: np.ndarray,
    train_rows: np.ndarray,
    feat_min: np.ndarray,
    feat_range: np.ndarray,
    bins: int,
    sample_n0: int,
) -> List[Optional[np.ndarray]]:
    n_features = int(X.shape[1])
    sample_n = min(int(train_rows.shape[0]), max(256, int(sample_n0)))
    rows = train_rows[:sample_n]
    q_thr: List[Optional[np.ndarray]] = [None] * n_features

    for f in range(n_features):
        if not (float(feat_range[f]) > 0.0):
            q_thr[f] = None
            continue
        vals = np.asarray(X[rows, f], dtype=np.float32).copy()
        vals.sort()
        nV = int(vals.shape[0])
        thr = np.empty(max(1, bins - 1), dtype=np.float32)
        prev = -float("inf")
        for j in range(1, bins):
            q = j / float(bins)
            pos = q * (nV - 1)
            lo = int(math.floor(pos))
            hi = min(nV - 1, lo + 1)
            t = float(vals[lo])
            if hi != lo:
                t = t + (float(vals[hi]) - float(vals[lo])) * (pos - lo)
            if not math.isfinite(t):
                t = float(feat_min[f]) + float(feat_range[f]) * (j / float(bins))
            if t < prev:
                t = prev
            thr[j - 1] = t
            prev = t
        q_thr[f] = thr
    return q_thr


def build_tree_regression(
    X: np.ndarray,
    train_samples: np.ndarray,
    residual: np.ndarray,
    feat_min: np.ndarray,
    feat_range: np.ndarray,
    *,
    depth: int,
    min_leaf: int,
    lr: float,
    scaleQ: int,
    rng: XorShift32,
    bins: int = 32,
    binning: str = "linear",
    q_thr: Optional[List[Optional[np.ndarray]]] = None,
) -> Tree:
    BINS = max(8, int(bins))
    is_quantile = str(binning or "").lower() == "quantile"

    pow2 = 1 << int(depth)
    internal = pow2 - 1

    feat_u16 = np.zeros(internal, dtype=np.uint16)
    thr_i32 = np.zeros(internal, dtype=np.int32)
    leaf_i32 = np.zeros(pow2, dtype=np.int32)

    def fill_forced(node_idx: int, level: int, leaf_val_q: int) -> None:
        if level == depth:
            leaf_i32[node_idx - internal] = int(leaf_val_q)
            return
        feat_u16[node_idx] = 0
        thr_i32[node_idx] = INT32_MAX
        fill_forced(node_idx * 2 + 1, level + 1, leaf_val_q)
        fill_forced(node_idx * 2 + 2, level + 1, leaf_val_q)

    def compute_leaf_q(samples: np.ndarray) -> int:
        if samples.size == 0:
            return 0
        m = float(np.mean(residual[samples], dtype=np.float64))
        v = float(lr) * m
        return clamp_i32(js_round(v * float(scaleQ)))

    def node_split(node_idx: int, level: int, samples: np.ndarray) -> None:
        if samples.size == 0:
            fill_forced(node_idx, level, 0)
            return
        if level == depth:
            leaf_i32[node_idx - internal] = int(compute_leaf_q(samples))
            return
        if samples.size < 2 * int(min_leaf):
            fill_forced(node_idx, level, compute_leaf_q(samples))
            return

        colsample = max(1, int(round(math.sqrt(int(X.shape[1])))))
        feats = sample_features(int(X.shape[1]), colsample, rng)

        best_f = -1
        best_thr_q = 0
        best_sse = float("inf")

        rr = residual[samples].astype(np.float64, copy=False)

        for f in feats:
            r = float(feat_range[f])
            if not (r > 0.0):
                continue

            if is_quantile:
                thr_arr = None if q_thr is None else q_thr[f]
                if thr_arr is None or int(thr_arr.shape[0]) != (BINS - 1):
                    continue
                x = X[samples, f].astype(np.float64, copy=False)
                b = np.searchsorted(thr_arr.astype(np.float64, copy=False), x, side="left")
                b = b.astype(np.int64, copy=False)
            else:
                min_f = float(feat_min[f])
                x = X[samples, f].astype(np.float64, copy=False)
                b = np.floor(((x - min_f) / r) * BINS).astype(np.int64)
                b = np.clip(b, 0, BINS - 1)

            cnt = np.bincount(b, minlength=BINS).astype(np.int64, copy=False)
            total_count = int(cnt.sum())
            if total_count < 2 * int(min_leaf):
                continue

            sum_r = np.bincount(b, weights=rr, minlength=BINS).astype(np.float64, copy=False)
            sum2_r = np.bincount(b, weights=rr * rr, minlength=BINS).astype(np.float64, copy=False)

            c_cnt = np.cumsum(cnt, dtype=np.int64)[:-1]
            left_cnt = c_cnt
            right_cnt = total_count - left_cnt
            mask = (left_cnt >= int(min_leaf)) & (right_cnt >= int(min_leaf))
            if not bool(np.any(mask)):
                continue

            c_sum = np.cumsum(sum_r, dtype=np.float64)[:-1]
            c_sum2 = np.cumsum(sum2_r, dtype=np.float64)[:-1]
            total_sum = float(sum_r.sum())
            total_sum2 = float(sum2_r.sum())

            left_sum = c_sum
            right_sum = total_sum - left_sum
            left_sum2 = c_sum2
            right_sum2 = total_sum2 - left_sum2

            left_cnt_safe = np.where(left_cnt > 0, left_cnt, 1)
            right_cnt_safe = np.where(right_cnt > 0, right_cnt, 1)

            left_sse = left_sum2 - (left_sum * left_sum) / left_cnt_safe
            right_sse = right_sum2 - (right_sum * right_sum) / right_cnt_safe
            sse = left_sse + right_sse

            sse_masked = np.where(mask, sse, np.inf)
            b_best = int(np.argmin(sse_masked))
            sse_best = float(sse_masked[b_best])
            if sse_best < best_sse:
                best_sse = sse_best
                best_f = int(f)
                if is_quantile:
                    thr_f = float(thr_arr[b_best])
                else:
                    thr_f = float(feat_min[f]) + r * ((b_best + 1) / float(BINS))
                best_thr_q = clamp_i32(js_round(thr_f * float(scaleQ)))

        if best_f < 0:
            fill_forced(node_idx, level, compute_leaf_q(samples))
            return

        xq = quantize_to_i32(X[samples, best_f], scaleQ)
        go_right = xq > int(best_thr_q)
        right = samples[go_right]
        left = samples[~go_right]

        if right.size < int(min_leaf) or left.size < int(min_leaf):
            fill_forced(node_idx, level, compute_leaf_q(samples))
            return

        feat_u16[node_idx] = int(best_f)
        thr_i32[node_idx] = int(best_thr_q)

        node_split(node_idx * 2 + 1, level + 1, left)
        node_split(node_idx * 2 + 2, level + 1, right)

    node_split(0, 0, np.asarray(train_samples, dtype=np.int64))
    return Tree(feat=feat_u16, thr=thr_i32, leaf=leaf_i32)


def build_tree_binary(
    X: np.ndarray,
    train_samples: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    feat_min: np.ndarray,
    feat_range: np.ndarray,
    *,
    depth: int,
    min_leaf: int,
    lr: float,
    scaleQ: int,
    rng: XorShift32,
    bins: int = 32,
    binning: str = "linear",
    q_thr: Optional[List[Optional[np.ndarray]]] = None,
) -> Tree:
    BINS = max(8, int(bins))
    is_quantile = str(binning or "").lower() == "quantile"

    pow2 = 1 << int(depth)
    internal = pow2 - 1

    feat_u16 = np.zeros(internal, dtype=np.uint16)
    thr_i32 = np.zeros(internal, dtype=np.int32)
    leaf_i32 = np.zeros(pow2, dtype=np.int32)

    LAMBDA = 1.0

    def fill_forced(node_idx: int, level: int, leaf_val_q: int) -> None:
        if level == depth:
            leaf_i32[node_idx - internal] = int(leaf_val_q)
            return
        feat_u16[node_idx] = 0
        thr_i32[node_idx] = INT32_MAX
        fill_forced(node_idx * 2 + 1, level + 1, leaf_val_q)
        fill_forced(node_idx * 2 + 2, level + 1, leaf_val_q)

    def compute_leaf_q(samples: np.ndarray) -> int:
        if samples.size == 0:
            return 0
        G = float(np.sum(grad[samples], dtype=np.float64))
        H = float(np.sum(hess[samples], dtype=np.float64))
        w = float(lr) * (G / (H + LAMBDA))
        return clamp_i32(js_round(w * float(scaleQ)))

    def node_split(node_idx: int, level: int, samples: np.ndarray) -> None:
        if samples.size == 0:
            fill_forced(node_idx, level, 0)
            return
        if level == depth:
            leaf_i32[node_idx - internal] = int(compute_leaf_q(samples))
            return
        if samples.size < 2 * int(min_leaf):
            fill_forced(node_idx, level, compute_leaf_q(samples))
            return

        colsample = max(1, int(round(math.sqrt(int(X.shape[1])))))
        feats = sample_features(int(X.shape[1]), colsample, rng)

        best_f = -1
        best_thr_q = 0
        best_gain = 0.0

        g = grad[samples].astype(np.float64, copy=False)
        h = hess[samples].astype(np.float64, copy=False)

        for f in feats:
            r = float(feat_range[f])
            if not (r > 0.0):
                continue

            if is_quantile:
                thr_arr = None if q_thr is None else q_thr[f]
                if thr_arr is None or int(thr_arr.shape[0]) != (BINS - 1):
                    continue
                x = X[samples, f].astype(np.float64, copy=False)
                b = np.searchsorted(thr_arr.astype(np.float64, copy=False), x, side="left")
                b = b.astype(np.int64, copy=False)
            else:
                min_f = float(feat_min[f])
                x = X[samples, f].astype(np.float64, copy=False)
                b = np.floor(((x - min_f) / r) * BINS).astype(np.int64)
                b = np.clip(b, 0, BINS - 1)

            cnt = np.bincount(b, minlength=BINS).astype(np.int64, copy=False)
            total_count = int(cnt.sum())
            if total_count < 2 * int(min_leaf):
                continue

            sum_g = np.bincount(b, weights=g, minlength=BINS).astype(np.float64, copy=False)
            sum_h = np.bincount(b, weights=h, minlength=BINS).astype(np.float64, copy=False)
            total_g = float(sum_g.sum())
            total_h = float(sum_h.sum())

            parent_score = (total_g * total_g) / (total_h + LAMBDA)

            c_cnt = np.cumsum(cnt, dtype=np.int64)[:-1]
            left_cnt = c_cnt
            right_cnt = total_count - left_cnt
            mask = (left_cnt >= int(min_leaf)) & (right_cnt >= int(min_leaf))
            if not bool(np.any(mask)):
                continue

            c_g = np.cumsum(sum_g, dtype=np.float64)[:-1]
            c_h = np.cumsum(sum_h, dtype=np.float64)[:-1]
            left_g = c_g
            left_h = c_h
            right_g = total_g - left_g
            right_h = total_h - left_h

            gain = (left_g * left_g) / (left_h + LAMBDA) + (right_g * right_g) / (right_h + LAMBDA) - parent_score
            gain_masked = np.where(mask, gain, -np.inf)
            b_best = int(np.argmax(gain_masked))
            gain_best = float(gain_masked[b_best])
            if gain_best > best_gain:
                best_gain = gain_best
                best_f = int(f)
                if is_quantile:
                    thr_f = float(thr_arr[b_best])
                else:
                    thr_f = float(feat_min[f]) + r * ((b_best + 1) / float(BINS))
                best_thr_q = clamp_i32(js_round(thr_f * float(scaleQ)))

        if best_f < 0:
            fill_forced(node_idx, level, compute_leaf_q(samples))
            return

        xq = quantize_to_i32(X[samples, best_f], scaleQ)
        go_right = xq > int(best_thr_q)
        right = samples[go_right]
        left = samples[~go_right]

        if right.size < int(min_leaf) or left.size < int(min_leaf):
            fill_forced(node_idx, level, compute_leaf_q(samples))
            return

        feat_u16[node_idx] = int(best_f)
        thr_i32[node_idx] = int(best_thr_q)

        node_split(node_idx * 2 + 1, level + 1, left)
        node_split(node_idx * 2 + 2, level + 1, right)

    node_split(0, 0, np.asarray(train_samples, dtype=np.int64))
    return Tree(feat=feat_u16, thr=thr_i32, leaf=leaf_i32)


def tree_predict_leaf_q(tree: Tree, X: np.ndarray, idx: np.ndarray, *, depth: int, scaleQ: int) -> np.ndarray:
    """Vectorized tree traversal. Returns int32 leaf values in Q-units for each row index in idx."""
    idx = np.asarray(idx, dtype=np.int64)
    n = int(idx.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    pow2 = 1 << int(depth)
    internal = pow2 - 1
    node = np.zeros(n, dtype=np.int32)

    for _ in range(int(depth)):
        f = tree.feat[node].astype(np.int64, copy=False)
        thr = tree.thr[node].astype(np.int32, copy=False)
        x = X[idx, f]  # elementwise gather
        xq = quantize_to_i32(x, scaleQ)
        go_right = xq > thr
        node = np.where(go_right, node * 2 + 2, node * 2 + 1).astype(np.int32)

    leaf_idx = node - internal
    return tree.leaf[leaf_idx].astype(np.int32, copy=False)


def apply_tree_scalar(tree: Tree, X: np.ndarray, idx: np.ndarray, predQ: np.ndarray, *, depth: int, scaleQ: int) -> None:
    if idx.size == 0:
        return
    leaf = tree_predict_leaf_q(tree, X, idx, depth=depth, scaleQ=scaleQ)
    predQ[idx] += leaf.astype(np.float64, copy=False)


def apply_tree_vector(tree: Tree, X: np.ndarray, idx: np.ndarray, predQ_flat: np.ndarray, *, depth: int, scaleQ: int, n_out: int, k: int) -> None:
    """Apply a v1-style tree to one output channel k in a flattened [nRows*nOut] predQ array."""
    if idx.size == 0:
        return
    leaf = tree_predict_leaf_q(tree, X, idx, depth=depth, scaleQ=scaleQ).astype(np.float64, copy=False)
    pos = idx.astype(np.int64) * int(n_out) + int(k)
    predQ_flat[pos] += leaf


# -------------------------
# LR schedule (ported)
# -------------------------

@dataclass
class LRSchedule:
    mode: str = "none"  # none|plateau|piecewise
    lr_base: float = 0.05
    max_trees: int = 200

    # plateau
    plateau_patience: int = 0
    plateau_factor: float = 1.0
    lr_min: float = 0.0
    plateau_since: int = 0
    lr_cur: float = 0.05

    # piecewise
    segments: Optional[List[Tuple[int, int, float]]] = None
    piecewise_last_lr: float = float("nan")

    def lr_for_iter(self, t: int) -> float:
        if self.mode == "piecewise" and self.segments:
            lr = self.lr_base
            for (start, end, seg_lr) in self.segments:
                if t < start:
                    break
                if start <= t <= end:
                    lr = seg_lr
                    break
            self.piecewise_last_lr = lr
            return lr
        if self.mode == "plateau":
            return self.lr_cur
        return self.lr_base

    def after_metric(self, improved: bool, t: int) -> None:
        if self.mode != "plateau":
            return
        self.plateau_since = 0 if improved else (self.plateau_since + 1)
        if self.plateau_since >= self.plateau_patience:
            self.lr_cur = self.lr_cur * self.plateau_factor
            if self.lr_min > 0 and self.lr_cur < self.lr_min:
                self.lr_cur = self.lr_min
            if self.lr_cur < 1e-12:
                self.lr_cur = 1e-12
            self.plateau_since = 0


def make_lr_schedule(args: argparse.Namespace) -> LRSchedule:
    lr_base = float(args.lr)
    sched = LRSchedule(mode="none", lr_base=lr_base, max_trees=int(args.trees), lr_cur=lr_base)

    mode = str(args.lr_schedule or "none").strip().lower()
    if mode not in ("none", "plateau", "piecewise"):
        mode = "none"

    if mode == "plateau":
        p = int(args.lr_patience or 0)
        if p < 1:
            p = max(5, min(100, int(round(int(args.trees) * 0.1))))
        drop_pct = float(args.lr_drop_pct or 10.0)
        factor = 1.0 - (drop_pct / 100.0)
        if not (0.0 < factor < 1.0):
            factor = 0.9
        lr_min = float(args.lr_min or 0.0)
        if lr_min < 0:
            lr_min = 0.0
        sched.mode = "plateau"
        sched.plateau_patience = p
        sched.plateau_factor = factor
        sched.lr_min = lr_min
        sched.lr_cur = lr_base
        return sched

    if mode == "piecewise":
        segs_raw = str(args.lr_segments or "").strip()
        if not segs_raw:
            return sched
        segs: List[Tuple[int, int, float]] = []
        for part in segs_raw.split(","):
            part = part.strip()
            if not part:
                continue
            bits = part.split(":")
            if len(bits) != 3:
                raise ValueError(f"Bad segment '{part}'. Use start:end:lr, comma-separated.")
            start = int(bits[0])
            end = int(bits[1])
            lr = float(bits[2])
            if start < 1 or end < start or not (lr > 0):
                raise ValueError(f"Bad segment '{part}'.")
            segs.append((start, end, lr))
        segs.sort(key=lambda x: (x[0], x[1]))
        for i in range(1, len(segs)):
            if segs[i][0] <= segs[i - 1][1]:
                raise ValueError("LR schedule ranges overlap")
        sched.mode = "piecewise"
        sched.segments = segs
        return sched

    return sched


# -------------------------
# Training entrypoints
# -------------------------

def choose_scale_q(task: str, max_abs_x: float, max_abs_y: float) -> int:
    t = normalize_task(task)
    lim_x = int(DEFAULT_SCALE_Q)
    if max_abs_x > 0:
        lim_x = int(INT32_SAFE / max_abs_x)
    lim_y = int(DEFAULT_SCALE_Q)
    if t == "regression" and max_abs_y > 0:
        lim_y = int(INT32_SAFE / max_abs_y)
    scale = min(DEFAULT_SCALE_Q, lim_x, lim_y)
    if scale < 1:
        scale = 1
    if scale > 0xFFFFFFFF:
        scale = 0xFFFFFFFF
    return int(scale)


def _max_abs(X: np.ndarray) -> float:
    if X.size == 0:
        return 0.0
    return float(np.max(np.abs(X.astype(np.float64, copy=False))))


def train_regression(X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    n_rows, n_features = X.shape
    depth = int(params["depth"])
    max_trees = int(params["trees"])
    min_leaf = int(params["min_leaf"])
    seed = int(params["seed"])
    early_stop = bool(params["early_stop"])
    patience = int(params["patience"])
    scaleQ = int(params["scaleQ"])
    bins = int(params["bins"])
    binning = str(params["binning"])
    split_train = float(params["split_train"])
    split_val = float(params["split_val"])
    refit_train_val = bool(params["refit_train_val"])

    lr_sched: LRSchedule = params["lr_sched"]

    idx = shuffled_indices(n_rows, seed)
    train, val, test = split_idx(idx, split_train, split_val)
    train_fit = np.concatenate([train, val]) if refit_train_val else train
    train_idx = train_fit

    feat_min, feat_max = _compute_feat_min_max(X, train_fit)
    feat_range = (feat_max - feat_min).astype(np.float32, copy=False)
    feat_range = np.where(feat_range > 0, feat_range, 0.0).astype(np.float32)

    q_thr = None
    if binning == "quantile":
        q_thr = _compute_quantile_thresholds(
            X=X,
            train_rows=train_fit,
            feat_min=feat_min,
            feat_range=feat_range,
            bins=bins,
            sample_n0=int(params["quantile_samples"]),
        )

    # Quantize y into Q-units (stored as float64 like JS)
    yQ_i32 = quantize_to_i32(y.astype(np.float32, copy=False), scaleQ)
    yQ = yQ_i32.astype(np.float64, copy=False)

    baseQ = clamp_i32(js_round(float(np.sum(yQ[train_idx], dtype=np.float64) / max(1, train_idx.size))))
    predQ = np.full(n_rows, float(baseQ), dtype=np.float64)
    residual = np.zeros(n_rows, dtype=np.float32)

    rng = XorShift32(seed ^ 0x9E3779B9)

    trees: List[Tree] = []
    curve_steps: List[int] = []
    curve_train: List[float] = []
    curve_val: List[float] = []
    curve_test: List[float] = []

    best_val = float("inf")
    best_train = float("inf")
    best_test = float("inf")
    best_iter = 0
    since_best = 0

    for t in range(1, max_trees + 1):
        # residual on train
        residual[train_idx] = ((yQ[train_idx] - predQ[train_idx]) / float(scaleQ)).astype(np.float32, copy=False)

        lr_used = float(lr_sched.lr_for_iter(t))
        tree = build_tree_regression(
            X=X,
            train_samples=train_idx,
            residual=residual,
            feat_min=feat_min,
            feat_range=feat_range,
            depth=depth,
            min_leaf=min_leaf,
            lr=lr_used,
            scaleQ=scaleQ,
            rng=rng,
            bins=bins,
            binning=binning,
            q_thr=q_thr,
        )
        trees.append(tree)

        apply_tree_scalar(tree, X, train_idx, predQ, depth=depth, scaleQ=scaleQ)
        apply_tree_scalar(tree, X, val, predQ, depth=depth, scaleQ=scaleQ)
        apply_tree_scalar(tree, X, test, predQ, depth=depth, scaleQ=scaleQ)

        train_mse = mse_q(yQ, predQ, train_idx, scaleQ)
        val_mse = mse_q(yQ, predQ, val, scaleQ)
        test_mse = mse_q(yQ, predQ, test, scaleQ)

        curve_steps.append(t)
        curve_train.append(train_mse)
        curve_val.append(val_mse)
        curve_test.append(test_mse)

        improved = False
        if val_mse + 1e-12 < best_val:
            best_val = val_mse
            best_train = train_mse
            best_test = test_mse
            best_iter = t
            since_best = 0
            improved = True
        else:
            since_best += 1

        lr_sched.after_metric(improved, t)

        if early_stop and since_best >= patience:
            break

    used_trees = max(1, best_iter) if early_stop else max(1, len(trees))
    final_trees = trees[:used_trees]
    model_bytes = serialize_model_v1(n_features, depth, final_trees, baseQ, scaleQ)

    meta = {
        "task": "regression",
        "metricName": "MSE",
        "nFeatures": int(n_features),
        "depth": int(depth),
        "maxTrees": int(max_trees),
        "usedTrees": int(used_trees),
        "baseQ": int(baseQ),
        "scaleQ": int(scaleQ),
        "bins": int(bins),
        "binning": str(binning),
        "bestIter": int(best_iter if early_stop else used_trees),
        "bestTrainMetric": float(best_train),
        "bestValMetric": float(best_val),
        "bestTestMetric": float(best_test),
        # Back-compat UI fields
        "bestTrainMSE": float(best_train),
        "bestValMSE": float(best_val),
        "bestTestMSE": float(best_test),
        "earlyStop": bool(early_stop),
    }

    curve = {"steps": curve_steps, "train": curve_train, "val": curve_val, "test": curve_test}
    return model_bytes, meta, curve


def train_binary(X: np.ndarray, y01: np.ndarray, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    n_rows, n_features = X.shape
    depth = int(params["depth"])
    max_trees = int(params["trees"])
    min_leaf = int(params["min_leaf"])
    seed = int(params["seed"])
    early_stop = bool(params["early_stop"])
    patience = int(params["patience"])
    scaleQ = int(params["scaleQ"])
    bins = int(params["bins"])
    binning = str(params["binning"])
    split_train = float(params["split_train"])
    split_val = float(params["split_val"])
    refit_train_val = bool(params["refit_train_val"])

    lr_sched: LRSchedule = params["lr_sched"]

    imbalance = params.get("imbalance", {}) or {}
    imb_mode = str(imbalance.get("mode", "none")).strip().lower()
    imb_cap = float(imbalance.get("cap", 20.0))
    imb_normalize = bool(imbalance.get("normalize", False))
    stratify = bool(imbalance.get("stratify", False))

    idx = shuffled_indices(n_rows, seed)
    if stratify:
        yK = (y01 >= 0.5).astype(np.int32)
        train, val, test = split_idx_stratified_by_class(idx, yK, 2, split_train, split_val)
    else:
        train, val, test = split_idx(idx, split_train, split_val)

    train_fit = np.concatenate([train, val]) if refit_train_val else train
    train_idx = train_fit

    feat_min, feat_max = _compute_feat_min_max(X, train_fit)
    feat_range = (feat_max - feat_min).astype(np.float32, copy=False)
    feat_range = np.where(feat_range > 0, feat_range, 0.0).astype(np.float32)

    q_thr = None
    if binning == "quantile":
        q_thr = _compute_quantile_thresholds(
            X=X,
            train_rows=train_fit,
            feat_min=feat_min,
            feat_range=feat_range,
            bins=bins,
            sample_n0=int(params["quantile_samples"]),
        )

    # Optional class weighting
    w_row = None
    w0 = 1.0
    w1 = 1.0
    if imb_mode in ("auto", "manual"):
        y_bin = (y01 >= 0.5).astype(np.int32)
        c0 = int(np.sum(y_bin[train_idx] == 0))
        c1 = int(np.sum(y_bin[train_idx] == 1))
        N = c0 + c1
        if imb_mode == "manual":
            w0 = float(imbalance.get("w0", 1.0))
            w1 = float(imbalance.get("w1", 1.0))
            if not (w0 > 0):
                w0 = 1.0
            if not (w1 > 0):
                w1 = 1.0
        else:
            if c0 > 0:
                w0 = N / (2.0 * c0)
            if c1 > 0:
                w1 = N / (2.0 * c1)
        if imb_cap > 0:
            w0 = min(w0, imb_cap)
            w1 = min(w1, imb_cap)
        if imb_normalize and N > 0:
            avg = (w0 * c0 + w1 * c1) / N
            if avg > 0:
                w0 /= avg
                w1 /= avg
        w_row = np.where(y_bin == 1, w1, w0).astype(np.float32)

    # Base score: log-odds of (weighted) positive rate
    sum_w = 0.0
    sum_w_pos = 0.0
    y_bin = (y01 >= 0.5).astype(np.int32)
    for r in train_idx.tolist():
        w = float(w_row[r]) if w_row is not None else 1.0
        sum_w += w
        sum_w_pos += w * (1.0 if y_bin[r] == 1 else 0.0)
    p0 = sum_w_pos / max(1e-12, sum_w)
    eps = 1e-6
    p0 = min(max(p0, eps), 1.0 - eps)
    base_logit = math.log(p0 / (1.0 - p0))
    baseQ = clamp_i32(js_round(base_logit * float(scaleQ)))

    predQ = np.full(n_rows, float(baseQ), dtype=np.float64)
    grad = np.zeros(n_rows, dtype=np.float32)
    hess = np.zeros(n_rows, dtype=np.float32)

    rng = XorShift32(seed ^ 0x9E3779B9)
    trees: List[Tree] = []

    curve_steps: List[int] = []
    curve_train: List[float] = []
    curve_val: List[float] = []
    curve_test: List[float] = []
    curve_train_acc: List[float] = []
    curve_val_acc: List[float] = []
    curve_test_acc: List[float] = []

    best_val = float("inf")
    best_train = float("inf")
    best_test = float("inf")
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_iter = 0
    since_best = 0

    for t in range(1, max_trees + 1):
        # Refresh grad/hess on train split
        logit = predQ[train_idx] / float(scaleQ)
        p = sigmoid_np(logit)
        w = w_row[train_idx].astype(np.float64) if w_row is not None else 1.0
        yy = y_bin[train_idx].astype(np.float64)
        g = (yy - p) * w
        h = (p * (1.0 - p)) * w
        grad[train_idx] = g.astype(np.float32)
        hess[train_idx] = h.astype(np.float32)

        lr_used = float(lr_sched.lr_for_iter(t))
        tree = build_tree_binary(
            X=X,
            train_samples=train_idx,
            grad=grad,
            hess=hess,
            feat_min=feat_min,
            feat_range=feat_range,
            depth=depth,
            min_leaf=min_leaf,
            lr=lr_used,
            scaleQ=scaleQ,
            rng=rng,
            bins=bins,
            binning=binning,
            q_thr=q_thr,
        )
        trees.append(tree)

        apply_tree_scalar(tree, X, train_idx, predQ, depth=depth, scaleQ=scaleQ)
        apply_tree_scalar(tree, X, val, predQ, depth=depth, scaleQ=scaleQ)
        apply_tree_scalar(tree, X, test, predQ, depth=depth, scaleQ=scaleQ)

        train_loss, train_acc = logloss_acc_binary(y01, predQ, train_idx, scaleQ, w_row=w_row)
        val_loss, val_acc = logloss_acc_binary(y01, predQ, val, scaleQ, w_row=w_row)
        test_loss, test_acc = logloss_acc_binary(y01, predQ, test, scaleQ, w_row=w_row)

        curve_steps.append(t)
        curve_train.append(train_loss)
        curve_val.append(val_loss)
        curve_test.append(test_loss)
        curve_train_acc.append(train_acc)
        curve_val_acc.append(val_acc)
        curve_test_acc.append(test_acc)

        improved = False
        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_test = test_loss
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_iter = t
            since_best = 0
            improved = True
        else:
            since_best += 1

        lr_sched.after_metric(improved, t)

        if early_stop and since_best >= patience:
            break

    used_trees = max(1, best_iter) if early_stop else max(1, len(trees))
    final_trees = trees[:used_trees]
    model_bytes = serialize_model_v1(n_features, depth, final_trees, baseQ, scaleQ)

    meta = {
        "task": "binary_classification",
        "metricName": "LogLoss",
        "nFeatures": int(n_features),
        "depth": int(depth),
        "maxTrees": int(max_trees),
        "usedTrees": int(used_trees),
        "baseQ": int(baseQ),
        "scaleQ": int(scaleQ),
        "bins": int(bins),
        "binning": str(binning),
        "bestIter": int(best_iter if early_stop else used_trees),
        "bestTrainMetric": float(best_train),
        "bestValMetric": float(best_val),
        "bestTestMetric": float(best_test),
        "bestTrainLoss": float(best_train),
        "bestValLoss": float(best_val),
        "bestTestLoss": float(best_test),
        "bestTrainAcc": float(best_train_acc),
        "bestValAcc": float(best_val_acc),
        "bestTestAcc": float(best_test_acc),
        "earlyStop": bool(early_stop),
    }

    curve = {
        "steps": curve_steps,
        "train": curve_train,
        "val": curve_val,
        "test": curve_test,
        "trainAcc": curve_train_acc,
        "valAcc": curve_val_acc,
        "testAcc": curve_test_acc,
    }
    return model_bytes, meta, curve


def train_multiclass(X: np.ndarray, yK_f: np.ndarray, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    n_rows, n_features = X.shape
    depth = int(params["depth"])
    max_trees = int(params["trees"])
    min_leaf = int(params["min_leaf"])
    seed = int(params["seed"])
    early_stop = bool(params["early_stop"])
    patience = int(params["patience"])
    scaleQ = int(params["scaleQ"])
    bins = int(params["bins"])
    binning = str(params["binning"])
    split_train = float(params["split_train"])
    split_val = float(params["split_val"])
    refit_train_val = bool(params["refit_train_val"])
    n_classes = int(params["n_classes"])

    lr_sched: LRSchedule = params["lr_sched"]

    imbalance = params.get("imbalance", {}) or {}
    imb_mode = str(imbalance.get("mode", "none")).strip().lower()
    imb_cap = float(imbalance.get("cap", 20.0))
    imb_normalize = bool(imbalance.get("normalize", False))
    stratify = bool(imbalance.get("stratify", False))

    yK = yK_f.astype(np.int32, copy=False)

    idx = shuffled_indices(n_rows, seed)
    if stratify:
        train, val, test = split_idx_stratified_by_class(idx, yK, n_classes, split_train, split_val)
    else:
        train, val, test = split_idx(idx, split_train, split_val)

    train_fit = np.concatenate([train, val]) if refit_train_val else train
    train_idx = train_fit

    feat_min, feat_max = _compute_feat_min_max(X, train_fit)
    feat_range = (feat_max - feat_min).astype(np.float32, copy=False)
    feat_range = np.where(feat_range > 0, feat_range, 0.0).astype(np.float32)

    q_thr = None
    if binning == "quantile":
        q_thr = _compute_quantile_thresholds(
            X=X,
            train_rows=train_fit,
            feat_min=feat_min,
            feat_range=feat_range,
            bins=bins,
            sample_n0=int(params["quantile_samples"]),
        )

    # Optional class weighting
    w_row = None
    w_class = None
    if imb_mode in ("auto", "manual"):
        counts = np.zeros(n_classes, dtype=np.int64)
        for r in train_idx.tolist():
            cls = int(yK[r])
            if 0 <= cls < n_classes:
                counts[cls] += 1
        N = int(train_idx.size)
        w_class = np.ones(n_classes, dtype=np.float64)
        if imb_mode == "manual":
            manual = imbalance.get("classWeights", None)
            if manual is None:
                manual = imbalance.get("class_weights", None)
            if isinstance(manual, (list, tuple)) and len(manual) >= n_classes:
                for k in range(n_classes):
                    w = float(manual[k])
                    if not (w > 0):
                        w = 1.0
                    w_class[k] = w
        else:
            for k in range(n_classes):
                c = int(counts[k])
                w = (N / (n_classes * c)) if c > 0 else 1.0
                w_class[k] = w
        cap = imb_cap if imb_cap > 0 else 20.0
        w_class = np.minimum(w_class, cap)
        if imb_normalize and N > 0:
            avg = float(np.sum(w_class * counts) / N)
            if avg > 0:
                w_class = w_class / avg
        w_row = w_class[np.clip(yK, 0, n_classes - 1)].astype(np.float32)

    # Base logits: log(prior) with smoothing
    smooth = 1e-3
    sum_w_class = np.zeros(n_classes, dtype=np.float64)
    sum_w = 0.0
    for r in train_idx.tolist():
        cls = int(yK[r])
        w = float(w_row[r]) if w_row is not None else 1.0
        sum_w += w
        if 0 <= cls < n_classes:
            sum_w_class[cls] += w
    denom = max(1e-9, sum_w + smooth * n_classes)
    base_logits_q = np.zeros(n_classes, dtype=np.int32)
    for k in range(n_classes):
        pk = (sum_w_class[k] + smooth) / denom
        pk = max(pk, 1e-9)
        base_logits_q[k] = clamp_i32(js_round(math.log(pk) * float(scaleQ)))

    # predQ flat
    predQ = np.empty(n_rows * n_classes, dtype=np.float64)
    for r in range(n_rows):
        base = r * n_classes
        predQ[base:base + n_classes] = base_logits_q.astype(np.float64)

    prob = np.zeros((n_rows, n_classes), dtype=np.float32)
    softmax_probs_inplace(predQ, n_rows, n_classes, scaleQ, prob)

    grad = np.zeros(n_rows, dtype=np.float32)
    hess = np.zeros(n_rows, dtype=np.float32)

    rng = XorShift32(seed ^ 0x9E3779B9)
    trees_by_class: List[List[Tree]] = [[] for _ in range(n_classes)]

    curve_steps: List[int] = []
    curve_train: List[float] = []
    curve_val: List[float] = []
    curve_test: List[float] = []
    curve_train_acc: List[float] = []
    curve_val_acc: List[float] = []
    curve_test_acc: List[float] = []

    best_val = float("inf")
    best_train = float("inf")
    best_test = float("inf")
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_iter = 0
    since_best = 0

    for t in range(1, max_trees + 1):
        lr_used = float(lr_sched.lr_for_iter(t))

        for k in range(n_classes):
            # grad/hess for this class
            p_k = prob[train_idx, k].astype(np.float64, copy=False)
            yk = (yK[train_idx] == k).astype(np.float64)
            w = w_row[train_idx].astype(np.float64) if w_row is not None else 1.0
            g = (yk - p_k) * w
            h = (p_k * (1.0 - p_k)) * w
            grad[train_idx] = g.astype(np.float32)
            hess[train_idx] = h.astype(np.float32)

            tree = build_tree_binary(
                X=X,
                train_samples=train_idx,
                grad=grad,
                hess=hess,
                feat_min=feat_min,
                feat_range=feat_range,
                depth=depth,
                min_leaf=min_leaf,
                lr=lr_used,
                scaleQ=scaleQ,
                rng=rng,
                bins=bins,
                binning=binning,
                q_thr=q_thr,
            )
            trees_by_class[k].append(tree)

            apply_tree_vector(tree, X, train_idx, predQ, depth=depth, scaleQ=scaleQ, n_out=n_classes, k=k)
            apply_tree_vector(tree, X, val, predQ, depth=depth, scaleQ=scaleQ, n_out=n_classes, k=k)
            apply_tree_vector(tree, X, test, predQ, depth=depth, scaleQ=scaleQ, n_out=n_classes, k=k)

        # Update probabilities
        softmax_probs_inplace(predQ, n_rows, n_classes, scaleQ, prob)

        train_loss, train_acc = logloss_acc_multiclass(yK, prob, train_idx, n_classes, w_row=w_row)
        val_loss, val_acc = logloss_acc_multiclass(yK, prob, val, n_classes, w_row=w_row)
        test_loss, test_acc = logloss_acc_multiclass(yK, prob, test, n_classes, w_row=w_row)

        curve_steps.append(t)
        curve_train.append(train_loss)
        curve_val.append(val_loss)
        curve_test.append(test_loss)
        curve_train_acc.append(train_acc)
        curve_val_acc.append(val_acc)
        curve_test_acc.append(test_acc)

        improved = False
        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_test = test_loss
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_iter = t
            since_best = 0
            improved = True
        else:
            since_best += 1

        lr_sched.after_metric(improved, t)

        if early_stop and since_best >= patience:
            break

    iters_done = len(trees_by_class[0]) if trees_by_class else 0
    used_trees_per_class = max(1, best_iter) if early_stop else max(1, iters_done)
    final_trees_by_class = [arr[:used_trees_per_class] for arr in trees_by_class]

    model_bytes = serialize_model_v2(
        n_features=n_features,
        depth=depth,
        n_classes=n_classes,
        trees_per_class=used_trees_per_class,
        base_logits_q=base_logits_q,
        scaleQ=scaleQ,
        trees_by_class=final_trees_by_class,
    )

    meta = {
        "task": "multiclass_classification",
        "metricName": "LogLoss",
        "nFeatures": int(n_features),
        "depth": int(depth),
        "scaleQ": int(scaleQ),
        "bins": int(bins),
        "binning": str(binning),
        "nClasses": int(n_classes),
        "maxTrees": int(max_trees),
        "usedTrees": int(used_trees_per_class),
        "treesPerClass": int(used_trees_per_class),
        "totalTrees": int(used_trees_per_class * n_classes),
        "bestIter": int(best_iter if early_stop else used_trees_per_class),
        "bestTrainMetric": float(best_train),
        "bestValMetric": float(best_val),
        "bestTestMetric": float(best_test),
        "bestTrainLoss": float(best_train),
        "bestValLoss": float(best_val),
        "bestTestLoss": float(best_test),
        "bestTrainAcc": float(best_train_acc),
        "bestValAcc": float(best_val_acc),
        "bestTestAcc": float(best_test_acc),
        "earlyStop": bool(early_stop),
    }

    curve = {
        "steps": curve_steps,
        "train": curve_train,
        "val": curve_val,
        "test": curve_test,
        "trainAcc": curve_train_acc,
        "valAcc": curve_val_acc,
        "testAcc": curve_test_acc,
    }
    return model_bytes, meta, curve


def train_multilabel(X: np.ndarray, y_flat: np.ndarray, params: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any], Dict[str, Any]]:
    n_rows, n_features = X.shape
    depth = int(params["depth"])
    max_trees = int(params["trees"])
    min_leaf = int(params["min_leaf"])
    seed = int(params["seed"])
    early_stop = bool(params["early_stop"])
    patience = int(params["patience"])
    scaleQ = int(params["scaleQ"])
    bins = int(params["bins"])
    binning = str(params["binning"])
    split_train = float(params["split_train"])
    split_val = float(params["split_val"])
    refit_train_val = bool(params["refit_train_val"])
    n_labels = int(params["n_classes"])

    lr_sched: LRSchedule = params["lr_sched"]

    imbalance = params.get("imbalance", {}) or {}
    imb_mode = str(imbalance.get("mode", "none")).strip().lower()
    imb_cap = float(imbalance.get("cap", 20.0))
    imb_normalize = bool(imbalance.get("normalize", False))
    stratify = bool(imbalance.get("stratify", False))  # unused for multilabel

    idx = shuffled_indices(n_rows, seed)
    train, val, test = split_idx(idx, split_train, split_val)
    train_fit = np.concatenate([train, val]) if refit_train_val else train
    train_idx = train_fit

    feat_min, feat_max = _compute_feat_min_max(X, train_fit)
    feat_range = (feat_max - feat_min).astype(np.float32, copy=False)
    feat_range = np.where(feat_range > 0, feat_range, 0.0).astype(np.float32)

    q_thr = None
    if binning == "quantile":
        q_thr = _compute_quantile_thresholds(
            X=X,
            train_rows=train_fit,
            feat_min=feat_min,
            feat_range=feat_range,
            bins=bins,
            sample_n0=int(params["quantile_samples"]),
        )

    y2 = y_flat.reshape(n_rows, n_labels).astype(np.float32, copy=False)

    # pos counts
    pos_count = np.zeros(n_labels, dtype=np.int64)
    for r in train_idx.tolist():
        pos_count += (y2[r] >= 0.5).astype(np.int64)

    # imbalance per-label pos weights
    pos_w = None
    w_scale = 1.0
    if imb_mode in ("auto", "manual"):
        pos_w = np.ones(n_labels, dtype=np.float32)
        if imb_mode == "manual":
            arr = imbalance.get("posWeights", None)
            if arr is None:
                arr = imbalance.get("pos_weights", None)
            if isinstance(arr, (list, tuple)) and len(arr) >= n_labels:
                for k in range(n_labels):
                    w = float(arr[k])
                    if not (w > 0):
                        w = 1.0
                    pos_w[k] = w
        else:
            for k in range(n_labels):
                pos = int(pos_count[k])
                neg = int(train_idx.size) - pos
                w = (neg / pos) if pos > 0 else 1.0
                pos_w[k] = float(w)

        cap = imb_cap if imb_cap > 0 else 20.0
        pos_w = np.minimum(pos_w, cap).astype(np.float32)

        if imb_normalize and train_idx.size > 0:
            w_sum = 0.0
            for k in range(n_labels):
                pos = int(pos_count[k])
                neg = int(train_idx.size) - pos
                w_sum += neg + float(pos_w[k]) * pos
            avg = w_sum / float(train_idx.size * n_labels)
            if avg > 0:
                w_scale = 1.0 / avg

    # base logits per label
    base_logits_q = np.zeros(n_labels, dtype=np.int32)
    for k in range(n_labels):
        pos = int(pos_count[k])
        neg = int(train_idx.size) - pos
        p0 = 0.5
        if train_idx.size > 0:
            if pos_w is not None:
                num = float(pos_w[k]) * pos
                den = neg + float(pos_w[k]) * pos
                p0 = num / max(1e-12, den)
            else:
                p0 = pos / max(1, int(train_idx.size))
        eps = 1e-6
        p0 = min(max(p0, eps), 1.0 - eps)
        base_logit = math.log(p0 / (1.0 - p0))
        base_logits_q[k] = clamp_i32(js_round(base_logit * float(scaleQ)))

    predQ = np.empty(n_rows * n_labels, dtype=np.float64)
    for r in range(n_rows):
        base = r * n_labels
        predQ[base:base + n_labels] = base_logits_q.astype(np.float64)

    grad = np.zeros(n_rows, dtype=np.float32)
    hess = np.zeros(n_rows, dtype=np.float32)

    rng = XorShift32(seed ^ 0x9E3779B9)
    trees_by_label: List[List[Tree]] = [[] for _ in range(n_labels)]

    curve_steps: List[int] = []
    curve_train: List[float] = []
    curve_val: List[float] = []
    curve_test: List[float] = []
    curve_train_acc: List[float] = []
    curve_val_acc: List[float] = []
    curve_test_acc: List[float] = []

    best_val = float("inf")
    best_train = float("inf")
    best_test = float("inf")
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_iter = 0
    since_best = 0

    for t in range(1, max_trees + 1):
        lr_used = float(lr_sched.lr_for_iter(t))

        for k in range(n_labels):
            # grad/hess for this label
            logits = predQ.reshape(n_rows, n_labels)[train_idx, k] / float(scaleQ)
            p = sigmoid_np(logits)
            yk = (y2[train_idx, k] >= 0.5).astype(np.float64)
            if pos_w is not None:
                w = np.where(yk >= 0.5, float(pos_w[k]), 1.0) * float(w_scale)
            else:
                w = 1.0
            g = (yk - p) * w
            h = (p * (1.0 - p)) * w
            grad[train_idx] = g.astype(np.float32)
            hess[train_idx] = h.astype(np.float32)

            tree = build_tree_binary(
                X=X,
                train_samples=train_idx,
                grad=grad,
                hess=hess,
                feat_min=feat_min,
                feat_range=feat_range,
                depth=depth,
                min_leaf=min_leaf,
                lr=lr_used,
                scaleQ=scaleQ,
                rng=rng,
                bins=bins,
                binning=binning,
                q_thr=q_thr,
            )
            trees_by_label[k].append(tree)

            apply_tree_vector(tree, X, train_idx, predQ, depth=depth, scaleQ=scaleQ, n_out=n_labels, k=k)
            apply_tree_vector(tree, X, val, predQ, depth=depth, scaleQ=scaleQ, n_out=n_labels, k=k)
            apply_tree_vector(tree, X, test, predQ, depth=depth, scaleQ=scaleQ, n_out=n_labels, k=k)

        train_loss, train_acc = logloss_acc_multilabel(y_flat, predQ, train_idx, n_labels, scaleQ, pos_w=pos_w, w_scale=w_scale)
        val_loss, val_acc = logloss_acc_multilabel(y_flat, predQ, val, n_labels, scaleQ, pos_w=pos_w, w_scale=w_scale)
        test_loss, test_acc = logloss_acc_multilabel(y_flat, predQ, test, n_labels, scaleQ, pos_w=pos_w, w_scale=w_scale)

        curve_steps.append(t)
        curve_train.append(train_loss)
        curve_val.append(val_loss)
        curve_test.append(test_loss)
        curve_train_acc.append(train_acc)
        curve_val_acc.append(val_acc)
        curve_test_acc.append(test_acc)

        improved = False
        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_train = train_loss
            best_test = test_loss
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_iter = t
            since_best = 0
            improved = True
        else:
            since_best += 1

        lr_sched.after_metric(improved, t)

        if early_stop and since_best >= patience:
            break

    iters_done = len(trees_by_label[0]) if trees_by_label else 0
    used_trees_per_label = max(1, best_iter) if early_stop else max(1, iters_done)
    final_trees_by_label = [arr[:used_trees_per_label] for arr in trees_by_label]

    model_bytes = serialize_model_v2(
        n_features=n_features,
        depth=depth,
        n_classes=n_labels,
        trees_per_class=used_trees_per_label,
        base_logits_q=base_logits_q,
        scaleQ=scaleQ,
        trees_by_class=final_trees_by_label,
    )

    meta = {
        "task": "multilabel_classification",
        "metricName": "LogLoss",
        "nFeatures": int(n_features),
        "depth": int(depth),
        "scaleQ": int(scaleQ),
        "bins": int(bins),
        "binning": str(binning),
        "nClasses": int(n_labels),
        "maxTrees": int(max_trees),
        "usedTrees": int(used_trees_per_label),
        "treesPerClass": int(used_trees_per_label),
        "totalTrees": int(used_trees_per_label * n_labels),
        "bestIter": int(best_iter if early_stop else used_trees_per_label),
        "bestTrainMetric": float(best_train),
        "bestValMetric": float(best_val),
        "bestTestMetric": float(best_test),
        "bestTrainLoss": float(best_train),
        "bestValLoss": float(best_val),
        "bestTestLoss": float(best_test),
        "bestTrainAcc": float(best_train_acc),
        "bestValAcc": float(best_val_acc),
        "bestTestAcc": float(best_test_acc),
        "earlyStop": bool(early_stop),
    }

    curve = {
        "steps": curve_steps,
        "train": curve_train,
        "val": curve_val,
        "test": curve_test,
        "trainAcc": curve_train_acc,
        "valAcc": curve_val_acc,
        "testAcc": curve_test_acc,
    }
    return model_bytes, meta, curve


# -------------------------
# CLI / main
# -------------------------

def parse_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s2 = str(s).strip()
    if not s2:
        return None
    return [p.strip() for p in s2.split(",") if p.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train a GL1F model in Python and export .gl1f")
    ap.add_argument("--task", required=True, choices=["regression", "binary_classification", "multiclass_classification", "multilabel_classification"])
    ap.add_argument("--out", required=True, help="Output .gl1f path")
    ap.add_argument("--no-package", action="store_true", help="Write raw model bytes only (no GL1X footer)")

    # Input formats
    ap.add_argument("--input", help="Input CSV or NPZ (if --npz), or ignored if using --npy-x/--npy-y")
    ap.add_argument("--npz", action="store_true", help="Treat --input as .npz with arrays X and y")
    ap.add_argument("--npz-x-key", default="X")
    ap.add_argument("--npz-y-key", default="y")
    ap.add_argument("--npy-x", help="Path to X.npy (float32 2D)")
    ap.add_argument("--npy-y", help="Path to y.npy (float32 or int)")
    ap.add_argument("--mmap", action="store_true", help="Use numpy mmap_mode='r' when loading .npy/.npz")

    # CSV parsing
    ap.add_argument("--delimiter", default="auto", help="CSV delimiter: auto, comma, semicolon, tab, pipe, or a single character")
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--label-col", help="Label column name/index (single-label tasks)")
    ap.add_argument("--label-cols", help="Comma-separated label columns for multilabel")
    ap.add_argument("--feature-cols", help="Comma-separated feature columns to use (optional)")
    ap.add_argument("--limit-rows", type=int, default=None, help="Debug: only read first N rows")
    ap.add_argument("--neg-label", default=None, help="Binary: explicit negative label value")
    ap.add_argument("--pos-label", default=None, help="Binary: explicit positive label value")
    ap.add_argument("--class-labels", default=None, help="Multiclass: comma-separated class labels (optional)")

    # Model metadata (optional)
    ap.add_argument("--title", default="")
    ap.add_argument("--description", default="")
    ap.add_argument("--chain-id", type=int, default=29)
    ap.add_argument("--chunk-size", type=int, default=24000)

    # Training params
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-leaf", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--scaleQ", default="auto", help="Quantization scaleQ integer, or 'auto'")
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--binning", choices=["linear", "quantile"], default="linear")
    ap.add_argument("--quantile-samples", type=int, default=50000)
    ap.add_argument("--split-train", type=float, default=0.7)
    ap.add_argument("--split-val", type=float, default=0.2)
    ap.add_argument("--refit-train-val", action="store_true")

    # Imbalance
    ap.add_argument("--imbalance-mode", choices=["none", "auto", "manual"], default="none")
    ap.add_argument("--imbalance-cap", type=float, default=20.0)
    ap.add_argument("--imbalance-normalize", action="store_true")
    ap.add_argument("--stratify", action="store_true", help="Stratify splits by class for binary/multiclass (similar to UI)")
    ap.add_argument("--w0", type=float, default=1.0, help="Binary manual: weight for class 0")
    ap.add_argument("--w1", type=float, default=1.0, help="Binary manual: weight for class 1")
    ap.add_argument("--class-weights", default=None, help="Multiclass manual: comma-separated weights (len=nClasses)")
    ap.add_argument("--pos-weights", default=None, help="Multilabel manual: comma-separated pos weights (len=nLabels)")

    # LR schedule
    ap.add_argument("--lr-schedule", choices=["none", "plateau", "piecewise"], default="none")
    ap.add_argument("--lr-patience", type=int, default=0)
    ap.add_argument("--lr-drop-pct", type=float, default=10.0)
    ap.add_argument("--lr-min", type=float, default=0.0)
    ap.add_argument("--lr-segments", default="", help="Piecewise segments: start:end:lr,start:end:lr,... (1-indexed inclusive)")

    args = ap.parse_args(argv)

    task = normalize_task(args.task)

    # Load data
    if args.npy_x and args.npy_y:
        X, y, info = load_from_npy(args.npy_x, args.npy_y, mmap=bool(args.mmap))
    else:
        if not args.input:
            raise SystemExit("Provide --input (CSV or NPZ) or --npy-x/--npy-y")
        if args.npz:
            X, y, info = load_from_npz(args.input, x_key=args.npz_x_key, y_key=args.npz_y_key, mmap=bool(args.mmap))
        else:
            label_cols = parse_list(args.label_cols)
            feat_cols = parse_list(args.feature_cols)
            cls_labels = parse_list(args.class_labels)
            X, y, info = load_from_csv(
                args.input,
                task=task,
                label_col=args.label_col,
                label_cols=label_cols,
                feature_cols=feat_cols,
                delimiter=args.delimiter,
                has_header=not args.no_header,
                limit_rows=args.limit_rows,
                neg_label=args.neg_label,
                pos_label=args.pos_label,
                class_labels=cls_labels,
            )

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n_rows, n_features = X.shape
    if n_rows < 3:
        raise ValueError("Need at least 3 rows for train/val/test split")

    # Derive task-specific settings
    n_classes = None
    if task == "multiclass_classification":
        classes = info.get("classes") or []
        n_classes = int(len(classes)) if classes else int(np.max(y) + 1)
        if n_classes < 2:
            raise ValueError("Need at least 2 classes")
    elif task == "multilabel_classification":
        n_classes = int(info.get("n_labels") or 0)
        if n_classes < 2:
            raise ValueError("Need at least 2 label columns for multilabel")
    else:
        n_classes = 2 if task == "binary_classification" else None

    # scaleQ
    scaleQ_raw = str(args.scaleQ).strip().lower()
    if scaleQ_raw == "auto" or scaleQ_raw == "0":
        max_abs_x = _max_abs(X)
        max_abs_y = _max_abs(y) if (task == "regression") else 0.0
        scaleQ = choose_scale_q(task, max_abs_x, max_abs_y)
    else:
        scaleQ = int(float(args.scaleQ))
        if scaleQ < 1:
            scaleQ = 1
        if scaleQ > 0xFFFFFFFF:
            scaleQ = 0xFFFFFFFF

    # Build params dict (similar to UI)
    lr_sched = make_lr_schedule(args)

    imbalance: Dict[str, Any] = {
        "mode": args.imbalance_mode,
        "cap": float(args.imbalance_cap),
        "normalize": bool(args.imbalance_normalize),
        "stratify": bool(args.stratify),
    }
    if task == "binary_classification" and args.imbalance_mode == "manual":
        imbalance["w0"] = float(args.w0)
        imbalance["w1"] = float(args.w1)
    if task == "multiclass_classification" and args.imbalance_mode == "manual":
        cw = parse_list(args.class_weights)
        if cw:
            imbalance["classWeights"] = [float(x) for x in cw]
    if task == "multilabel_classification" and args.imbalance_mode == "manual":
        pw = parse_list(args.pos_weights)
        if pw:
            imbalance["posWeights"] = [float(x) for x in pw]

    train_params: Dict[str, Any] = {
        "task": task,
        "trees": int(args.trees),
        "depth": int(args.depth),
        "lr": float(args.lr),
        "min_leaf": int(args.min_leaf),
        "seed": int(args.seed),
        "early_stop": bool(args.early_stop),
        "patience": int(args.patience),
        "scaleQ": int(scaleQ),
        "bins": int(max(8, min(512, int(args.bins)))),
        "binning": str(args.binning),
        "quantile_samples": int(args.quantile_samples),
        "split_train": float(args.split_train),
        "split_val": float(args.split_val),
        "refit_train_val": bool(args.refit_train_val),
        "imbalance": imbalance,
        "lr_sched": lr_sched,
    }
    if task in ("multiclass_classification", "multilabel_classification"):
        train_params["n_classes"] = int(n_classes or 2)

    # Train
    if task == "regression":
        model_bytes, meta, curve = train_regression(X, y.astype(np.float32, copy=False), train_params)
    elif task == "binary_classification":
        model_bytes, meta, curve = train_binary(X, y.astype(np.float32, copy=False), train_params)
    elif task == "multiclass_classification":
        model_bytes, meta, curve = train_multiclass(X, y.astype(np.float32, copy=False), train_params)
    elif task == "multilabel_classification":
        model_bytes, meta, curve = train_multilabel(X, y.astype(np.float32, copy=False), train_params)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Build featuresPacked for UI auto task detection
    feature_names = info.get("feature_names") or [f"f{i}" for i in range(n_features)]
    label_name = info.get("label_name") or "label"

    labels_for_meta = None
    label_names_for_meta = None
    if task == "binary_classification":
        cls = info.get("classes") or ["0", "1"]
        # cls is [neg,pos]
        labels_for_meta = [str(cls[0]), str(cls[1])]
    elif task == "multiclass_classification":
        cls = info.get("classes") or [str(i) for i in range(int(n_classes or 2))]
        labels_for_meta = [str(x) for x in cls]
    elif task == "multilabel_classification":
        label_names_for_meta = [str(x) for x in (info.get("label_names") or [f"label{i}" for i in range(int(n_classes or 2))])]
        labels_for_meta = ["0", "1"]

    features_packed = pack_nft_features(
        task=task,
        feature_names=feature_names,
        label_name=("(multilabel)" if task == "multilabel_classification" else label_name),
        labels=labels_for_meta,
        label_names=label_names_for_meta,
    )

    # Package JSON (optional footer)
    out_bytes = model_bytes
    if not args.no_package:
        pkg_obj: Dict[str, Any] = {
            "kind": "GL1F_PACKAGE",
            "v": 1,
            "createdAt": now_iso(),
            "chainId": int(args.chain_id),
            "chunkSize": int(args.chunk_size),
            "model": {
                "gl1fVersion": 2 if task in ("multiclass_classification", "multilabel_classification") else 1,
                "nFeatures": int(n_features),
                "depth": int(args.depth),
                "scaleQ": int(scaleQ),
                "bytes": int(len(model_bytes)),
            },
            "nft": {
                "title": str(args.title or "").strip(),
                "description": str(args.description or "").strip(),
                "iconPngB64": None,
                "featuresPacked": features_packed,
                "titleWordHashes": [],
            },
            "registry": {
                "pricingMode": 0,
                "feeWei": "0",
                "recipient": "",
                "ownerKey": "",
                "tosVersionAccepted": 0,
                "licenseIdAccepted": 0,
            },
            "local": {
                "trainMeta": meta,
                "trainParams": {
                    # Make JSON friendly: remove lr_sched object
                    **{k: v for k, v in train_params.items() if k != "lr_sched"},
                    "lrSchedule": {
                        "mode": lr_sched.mode,
                        "lr": lr_sched.lr_base,
                        "plateauPatience": lr_sched.plateau_patience,
                        "plateauFactor": lr_sched.plateau_factor,
                        "minLR": lr_sched.lr_min,
                        "segments": lr_sched.segments or [],
                    },
                },
                "curve": curve,
            },
        }
        footer = build_gl1x_footer(pkg_obj)
        out_bytes = model_bytes + footer

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(out_bytes)

    # Small console summary
    sys.stderr.write(f"Wrote: {out_path} ({len(out_bytes):,} bytes; model={len(model_bytes):,})\n")
    sys.stderr.write(f"Task: {task}, rows={n_rows:,}, features={n_features}, scaleQ={scaleQ}\n")
    sys.stderr.write(f"Best: {meta.get('bestValMetric')}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
