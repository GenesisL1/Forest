#!/usr/bin/env python3
"""
gl1f_search.py — web-exact heuristic hyperparameter search for GL1F.

This is a command-line version of the heuristic search implemented in the
Forest web UI's create_page.js. It intentionally mirrors the web logic:

  * round 1 trains the current/base hyperparameters exactly;
  * rounds 2..N use the same xorshift32 RNG seeded as seed ^ 0x9e3779b9;
  * each candidate pivots from best-so-far with probability 0.75, else base;
  * trees/depth/lr/minLeaf/patience perturbations use the same formulas;
  * bins/binning/split/seed/task/class-imbalance stay fixed;
  * best is selected by the chosen trainMeta metric (default: bestValMetric, min);
  * optional refit happens only after search, using Train+Val and usedTrees.

It drives the same local trainers as the web UI / local server:

  python: python train_gl1f.py --task ... --input ... --out ... <params>
  cpp   : ./train_gl1f_cpp     --task ... --input ... --out ... <params>

Example:
  python gl1f_search.py \
      --engine cpp --task regression \
      --input data/btc_vol.csv --label-col y \
      --trees 250 --depth 4 --lr 0.05 --min-leaf 10 \
      --bins 32 --binning linear --seed 42 --early-stop --patience 25 \
      --trials 10 --best-by bestValMetric --out best.gl1f --work runs/web_exact

  Or pass the exact web-style round-1 object:
  python gl1f_search.py \
      --engine cpp --task regression --input data.csv --label-col y \
      --initial-params '{"trees":250,"depth":4,"lr":0.05,"minLeaf":10,"bins":32,"binning":"linear","seed":42,"earlyStop":true,"patience":25,"splitTrain":0.7,"splitVal":0.2}'
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

GL1X_MAGIC = b"GL1X"
SIZE_LIMIT = 15_000_000
UINT32_MASK = 0xFFFFFFFF

# Keys written by train_gl1f.py / train_gl1f_cpp.cpp under
# local.trainMeta.  bestValMetric is the web UI default; accuracy keys are
# available only for classification tasks.
METRIC_ALIASES = {
    "val": "bestValMetric",
    "validation": "bestValMetric",
    "val_metric": "bestValMetric",
    "val_loss": "bestValMetric",
    "best_val": "bestValMetric",
    "best_val_metric": "bestValMetric",
    "bestvalmetric": "bestValMetric",
    "test": "bestTestMetric",
    "test_metric": "bestTestMetric",
    "test_loss": "bestTestMetric",
    "best_test": "bestTestMetric",
    "best_test_metric": "bestTestMetric",
    "besttestmetric": "bestTestMetric",
    "train": "bestTrainMetric",
    "train_metric": "bestTrainMetric",
    "train_loss": "bestTrainMetric",
    "best_train": "bestTrainMetric",
    "best_train_metric": "bestTrainMetric",
    "besttrainmetric": "bestTrainMetric",
    "val_acc": "bestValAcc",
    "val_accuracy": "bestValAcc",
    "validation_accuracy": "bestValAcc",
    "best_val_acc": "bestValAcc",
    "bestvalacc": "bestValAcc",
    "test_acc": "bestTestAcc",
    "test_accuracy": "bestTestAcc",
    "best_test_acc": "bestTestAcc",
    "besttestacc": "bestTestAcc",
    "train_acc": "bestTrainAcc",
    "train_accuracy": "bestTrainAcc",
    "best_train_acc": "bestTrainAcc",
    "besttrainacc": "bestTrainAcc",
}

MAXIMIZE_METRIC_KEYS = {
    "bestValAcc",
    "bestTestAcc",
    "bestTrainAcc",
    "valAcc",
    "testAcc",
    "trainAcc",
}


# ---------------------------------------------------------------------------
# JavaScript-compatible numeric helpers
# ---------------------------------------------------------------------------

def js_uint32(x: int) -> int:
    return int(x) & UINT32_MASK


def js_int32(x: int) -> int:
    x = js_uint32(x)
    return x - 0x100000000 if x & 0x80000000 else x


def js_round(x: float) -> int:
    """Match JavaScript Math.round for finite values used here.

    Python's round() is banker's rounding, so it does not match JS. JS behaves
    like floor(x + 0.5) for our numeric candidate-generation use cases.
    """
    return int(math.floor(float(x) + 0.5))


def clamp_int(x: Any, lo: int, hi: int) -> int:
    try:
        v = int(x)
    except Exception:
        return int(lo)
    if v < lo:
        return int(lo)
    if v > hi:
        return int(hi)
    return int(v)


def clamp_float(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(lo)
    if not math.isfinite(v):
        return float(lo)
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def parse_bool_value(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def normalize_metric_key(key: str) -> str:
    raw = str(key or "bestValMetric").strip()
    compact = raw.replace("-", "_").replace(" ", "_")
    alias = METRIC_ALIASES.get(compact.lower())
    return alias or raw


def infer_metric_direction(metric_key: str, requested: str = "auto") -> str:
    req = str(requested or "auto").strip().lower()
    if req in ("min", "max"):
        return req
    key = normalize_metric_key(metric_key)
    if key in MAXIMIZE_METRIC_KEYS or key.lower().endswith(("acc", "accuracy", "auc")):
        return "max"
    return "min"


def load_json_arg(value: Optional[str], what: str) -> Dict[str, Any]:
    """Load an inline JSON object, @file, or plain file path."""
    if not value:
        return {}
    src = str(value).strip()
    if not src:
        return {}

    text: str
    if src.startswith("@"):
        path = src[1:]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif (src.startswith("{") and src.endswith("}")) or (src.startswith("[") and src.endswith("]")):
        text = src
    elif os.path.exists(src):
        with open(src, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = src

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[fatal] Could not parse {what} JSON: {e}")

    if not isinstance(obj, dict):
        raise SystemExit(f"[fatal] {what} must be a JSON object")

    # Convenience: allow passing a leaderboard row or {best:{params:{...}}}.
    if isinstance(obj.get("params"), dict):
        obj = obj["params"]
    elif isinstance(obj.get("best"), dict) and isinstance(obj["best"].get("params"), dict):
        obj = obj["best"]["params"]
    return obj


def first_present(d: Dict[str, Any], names: List[str], default: Any = None) -> Any:
    for name in names:
        if name in d:
            return d[name]
    return default


def normalize_split_fraction(value: Any, *, default: float, lo_pct: int, hi_pct: int) -> float:
    """Accept 0.70, 70, or '70%' and return a web-style fraction."""
    if value is None:
        value = default
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        f = float(value)
    except Exception:
        f = float(default)
    pct = js_round(f * 100.0) if abs(f) <= 1.0 else js_round(f)
    return clamp_int(pct, lo_pct, hi_pct) / 100.0


def normalize_binning(value: Any) -> str:
    s = str(value or "linear").strip().lower()
    return s if s in ("linear", "quantile") else "linear"


class XorShift32:
    """The exact RNG used by create_page.js.

    JS source being mirrored:
      let x = (seed | 0) || 123456789;
      x ^= x << 13;
      x ^= x >>> 17;
      x ^= x << 5;
      return x >>> 0;
    """

    def __init__(self, seed: int):
        x = js_int32(seed)
        if x == 0:
            x = 123456789
        self.x = js_uint32(x)

    def u32(self) -> int:
        x = self.x
        x = js_uint32(x ^ js_uint32(x << 13))
        x = js_uint32(x ^ (x >> 17))
        x = js_uint32(x ^ js_uint32(x << 5))
        self.x = x
        return x

    def rand01(self) -> float:
        return self.u32() / 4294967296.0


def web_rng_seed(seed: int) -> int:
    """Match JS: baseParams.seed ^ 0x9e3779b9."""
    return js_int32(js_int32(seed) ^ js_int32(0x9E3779B9))


# ---------------------------------------------------------------------------
# Web model-size clamp, mirrored from create_page.js + train_gbdt.js
# ---------------------------------------------------------------------------

def estimate_model_bytes(n_trees: int, depth: int) -> int:
    d = max(1, int(depth))
    t = max(0, int(n_trees))
    pow2 = 1 << d
    internal = pow2 - 1
    per_tree = internal * 8 + pow2 * 4
    return 24 + t * per_tree


def estimate_model_bytes_v2(trees_per_class: int, depth: int, n_classes: int) -> int:
    d = max(1, int(depth))
    t = max(0, int(trees_per_class))
    k = max(0, int(n_classes))
    pow2 = 1 << d
    internal = pow2 - 1
    per_tree = internal * 8 + pow2 * 4
    header = 24 + max(0, k) * 4
    return header + (t * k) * per_tree


def estimate_bytes_for_task(trees: int, depth: int, task: str, n_classes: int) -> int:
    t = max(0, int(trees))
    d = max(1, int(depth))
    if task in ("multiclass_classification", "multilabel_classification"):
        k = max(2, int(n_classes))
        return estimate_model_bytes_v2(t, d, k)
    return estimate_model_bytes(t, d)


def clamp_for_size(trees: Any, depth: Any, task: str = "regression", n_classes: int = 2) -> Dict[str, int]:
    """Match create_page.js clampForSize()."""
    min_t = 1 if task in ("multiclass_classification", "multilabel_classification") else 10
    try:
        t = max(min_t, math.floor(float(trees)))
    except Exception:
        t = min_t
    try:
        d = max(1, math.floor(float(depth)))
    except Exception:
        d = 1

    if task in ("multiclass_classification", "multilabel_classification"):
        k = max(2, int(n_classes))
        max_tpc = max(1, math.floor(65535 / k))
        if t > max_tpc:
            t = max_tpc

    est = estimate_bytes_for_task(t, d, task, n_classes)
    while est > SIZE_LIMIT and t > min_t:
        t = max(min_t, t - 25)
        est = estimate_bytes_for_task(t, d, task, n_classes)
    while est > SIZE_LIMIT and d > 2:
        d = max(2, d - 1)
        est = estimate_bytes_for_task(t, d, task, n_classes)
    return {"trees": int(t), "depth": int(d), "estBytes": int(est)}


# ---------------------------------------------------------------------------
# Engine + GL1X parsing
# ---------------------------------------------------------------------------
@dataclass
class Engine:
    kind: str
    python_exe: str
    train_script: str
    cpp_bin: str

    def base_cmd(self) -> List[str]:
        if self.kind == "python":
            return [self.python_exe, self.train_script]
        return [self.cpp_bin]


def resolve_engine(args: argparse.Namespace) -> Engine:
    kind = args.engine
    if kind == "auto":
        kind = "cpp" if os.path.exists(args.cpp_bin) and os.access(args.cpp_bin, os.X_OK) else "python"

    if kind == "python" and not os.path.exists(args.train_script):
        sys.exit(f"[fatal] train script not found: {args.train_script}")
    if kind == "cpp" and not (os.path.exists(args.cpp_bin) and os.access(args.cpp_bin, os.X_OK)):
        sys.exit(
            f"[fatal] cpp trainer not found/executable: {args.cpp_bin} "
            f"(build it with build_cpp_trainer.sh)"
        )
    return Engine(kind=kind, python_exe=args.python_exe, train_script=args.train_script, cpp_bin=args.cpp_bin)


def parse_gl1x_package(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except OSError:
        return None

    pos = blob.rfind(GL1X_MAGIC)
    if pos < 0 or pos + 12 > len(blob):
        return None

    try:
        (length,) = struct.unpack_from("<I", blob, pos + 8)
        payload = blob[pos + 12: pos + 12 + length]
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None


def metric_from_package(pkg: Optional[Dict[str, Any]], key: str = "bestValMetric") -> Optional[float]:
    if not isinstance(pkg, dict):
        return None
    try:
        meta = pkg.get("local", {}).get("trainMeta", {})
        v = meta.get(key)
        return float(v) if v is not None else None
    except Exception:
        return None


def meta_from_package(pkg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(pkg, dict):
        return {}
    meta = pkg.get("local", {}).get("trainMeta", {})
    return meta if isinstance(meta, dict) else {}


def curve_from_package(pkg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(pkg, dict):
        return {}
    curve = pkg.get("local", {}).get("curve", {})
    return curve if isinstance(curve, dict) else {}


def scrape_best_from_stderr(stderr: bytes) -> Optional[float]:
    for line in reversed(stderr.decode("utf-8", "replace").splitlines()):
        if line.strip().startswith("Best:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Trial structures
# ---------------------------------------------------------------------------
@dataclass
class Trial:
    round: int
    params: Dict[str, Any]
    raw: Optional[float] = None
    score: float = math.inf
    ok: bool = False
    seconds: float = 0.0
    cmd: str = ""
    out_path: str = ""
    err: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    curve: Dict[str, Any] = field(default_factory=dict)
    refit: bool = False


def param_hash(params: Dict[str, Any], prefix: str = "") -> str:
    s = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1((prefix + s).encode("utf-8")).hexdigest()[:16]


def short_params(p: Dict[str, Any]) -> str:
    bits = []
    for k in ("trees", "depth", "lr", "minLeaf", "patience", "bins", "binning"):
        if k in p:
            label = "min-leaf" if k == "minLeaf" else k
            bits.append(f"{label}={p[k]}")
    lrs = p.get("lrSchedule")
    if isinstance(lrs, dict) and lrs.get("mode") and lrs.get("mode") != "none":
        if lrs.get("mode") == "plateau":
            bits.append(f"lrSched=plateau(n={lrs.get('patience')},drop={lrs.get('dropPct')},min={lrs.get('minLR')})")
        else:
            bits.append("lrSched=piecewise")
    return " ".join(bits)


def trial_dict(t: Trial) -> Dict[str, Any]:
    return {
        "round": t.round,
        "ok": t.ok,
        "raw": t.raw,
        "score": t.score,
        "seconds": round(t.seconds, 3),
        "params": t.params,
        "meta": t.meta,
        "cmd": t.cmd,
        "out_path": t.out_path,
        "err": t.err,
        "refit": t.refit,
    }


# ---------------------------------------------------------------------------
# Base params + web-exact candidate generator
# ---------------------------------------------------------------------------

def make_lr_schedule_from_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    mode = str(args.lr_schedule or "none").lower().strip()
    if mode == "none":
        return None
    if mode == "plateau":
        return {
            "mode": "plateau",
            "patience": clamp_int(args.lr_patience, 1, 10000),
            "dropPct": clamp_int(args.lr_drop_pct, 1, 90),
            "minLR": clamp_float(args.lr_min, 0.0, 1.0),
        }
    if mode == "piecewise":
        return {
            "mode": "piecewise",
            "segments": str(args.lr_segments or ""),
        }
    raise ValueError(f"unknown lr schedule mode: {mode}")


def make_imbalance_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    mode = str(args.imbalance_mode or "none").lower().strip()
    imb: Dict[str, Any] = {
        "mode": mode,
        "cap": clamp_float(args.imbalance_cap, 1.0, 1000.0),
        "normalize": bool(args.imbalance_normalize),
        "stratify": bool(args.stratify) if args.task in ("binary_classification", "multiclass_classification") else False,
    }
    if mode == "manual":
        if args.w0 is not None:
            imb["w0"] = float(args.w0)
        if args.w1 is not None:
            imb["w1"] = float(args.w1)
        if args.class_weights:
            imb["classWeights"] = args.class_weights
        if args.pos_weights:
            imb["posWeights"] = args.pos_weights
    return imb


def read_base_train_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Command-line equivalent of create_page.js readBaseTrainParams().

    Round 1 is this returned object, exactly like the web UI does:
      const params = (round === 1) ? { ...baseParams } : generateHeuristicCandidate(...)

    Base params can come from individual CLI flags or from --initial-params.
    --initial-params accepts the web object keys directly, for example:
      {"trees":250,"depth":4,"lr":0.05,"minLeaf":10,"bins":32,
       "binning":"linear","seed":42,"earlyStop":true,"patience":25,
       "splitTrain":0.7,"splitVal":0.2}
    """
    cli_base: Dict[str, Any] = {
        "task": args.task,
        "trees": args.trees,
        "depth": args.depth,
        "lr": args.lr,
        "lrSchedule": make_lr_schedule_from_args(args),
        "minLeaf": args.min_leaf,
        "bins": args.bins,
        "binning": args.binning,
        "seed": args.seed,
        "earlyStop": args.early_stop,
        "patience": args.patience,
        "splitTrain": args.split_train,
        "splitVal": args.split_val,
        "nClasses": args.n_classes,
        "imbalance": make_imbalance_from_args(args),
    }

    initial = load_json_arg(args.initial_params, "--initial-params") if getattr(args, "initial_params", None) else {}
    if initial:
        init_task = first_present(initial, ["task"], None)
        if init_task is not None and str(init_task) != str(args.task):
            raise SystemExit(
                f"[fatal] --initial-params task={init_task!r} does not match --task={args.task!r}. "
                "Use the same task for data loading and round-1 params."
            )

        # Accept both web camelCase keys and CLI-style keys.
        overlays = {
            "trees": first_present(initial, ["trees", "nTrees", "numTrees"], None),
            "depth": first_present(initial, ["depth", "maxDepth"], None),
            "lr": first_present(initial, ["lr", "learningRate", "learning_rate"], None),
            "lrSchedule": first_present(initial, ["lrSchedule", "lr_schedule"], None),
            "minLeaf": first_present(initial, ["minLeaf", "min_leaf", "min-leaf"], None),
            "bins": first_present(initial, ["bins", "nBins", "numBins"], None),
            "binning": first_present(initial, ["binning", "binningMode", "binning_mode"], None),
            "seed": first_present(initial, ["seed"], None),
            "earlyStop": first_present(initial, ["earlyStop", "early_stop", "early-stop"], None),
            "patience": first_present(initial, ["patience", "earlyStopPatience", "early_stop_patience"], None),
            "splitTrain": first_present(initial, ["splitTrain", "split_train", "split-train", "trainPct", "train_pct"], None),
            "splitVal": first_present(initial, ["splitVal", "split_val", "split-val", "valPct", "val_pct"], None),
            "nClasses": first_present(initial, ["nClasses", "n_classes", "classes", "nLabels", "n_labels"], None),
            "imbalance": first_present(initial, ["imbalance"], None),
        }
        for k, v in overlays.items():
            if v is not None:
                cli_base[k] = v

    # Web inputs have these min/max constraints in create.html; readBaseTrainParams
    # then applies clampForSize(), which can reduce trees/depth further.
    trees0 = clamp_int(cli_base["trees"], 10, 5000)
    depth0 = clamp_int(cli_base["depth"], 1, 12)
    lr = clamp_float(cli_base["lr"], 0.001, 1.0)
    min_leaf = clamp_int(cli_base["minLeaf"], 1, 1000)
    bins = clamp_int(cli_base["bins"], 8, 512)
    binning = normalize_binning(cli_base["binning"])
    seed = clamp_int(cli_base["seed"], 1, 2147483647)
    early_stop = parse_bool_value(cli_base["earlyStop"], default=True)
    patience = clamp_int(cli_base["patience"], 1, 500)

    split_train = normalize_split_fraction(cli_base["splitTrain"], default=0.7, lo_pct=50, hi_pct=90)
    split_val = normalize_split_fraction(cli_base["splitVal"], default=0.2, lo_pct=5, hi_pct=40)

    # Web split UI enforces a non-trivial test set (>=5%) before readBaseTrainParams().
    train_pct = int(split_train * 100)
    val_pct = int(split_val * 100)
    if train_pct + val_pct > 95:
        val_pct = max(5, 95 - train_pct)
    split_train = train_pct / 100.0
    split_val = val_pct / 100.0

    n_classes = max(2, clamp_int(cli_base["nClasses"], 2, 65535))
    cl = clamp_for_size(trees0, depth0, args.task, n_classes)

    lr_schedule = cli_base.get("lrSchedule")
    if lr_schedule in ("", "none", "null"):
        lr_schedule = None
    if lr_schedule is not None and not isinstance(lr_schedule, dict):
        raise SystemExit("[fatal] lrSchedule inside --initial-params must be null or an object")

    imbalance = cli_base.get("imbalance")
    if imbalance is None:
        imbalance = {"mode": "none"}
    if not isinstance(imbalance, dict):
        raise SystemExit("[fatal] imbalance inside --initial-params must be an object")

    return {
        "task": args.task,
        "trees": cl["trees"],
        "depth": cl["depth"],
        "lr": lr,
        "lrSchedule": copy.deepcopy(lr_schedule),
        "minLeaf": min_leaf,
        "bins": bins,
        "binning": binning,
        "seed": seed,
        "earlyStop": early_stop,
        "patience": patience,
        "splitTrain": split_train,
        "splitVal": split_val,
        "nClasses": n_classes,
        "imbalance": copy.deepcopy(imbalance),
    }

def generate_heuristic_candidate(
    *,
    base_params: Dict[str, Any],
    best_params: Optional[Dict[str, Any]],
    round_no: int,
    rng: XorShift32,
) -> Dict[str, Any]:
    """Exact port of create_page.js generateHeuristicCandidate()."""
    pivot = best_params if (best_params is not None and rng.rand01() < 0.75) else base_params
    p = copy.deepcopy(pivot)

    trees_min = 10
    trees_max = 5000
    depth_min = 1
    depth_max = 12
    lr_min = 0.001
    lr_max = 1.0
    min_leaf_min = 1
    min_leaf_max = 1000
    pat_min = 1
    pat_max = 500

    # --- trees ---
    trees_factor = math.pow(2.0, (rng.rand01() - 0.5) * 1.4)  # ~[0.62..1.62]
    trees = js_round((float(pivot["trees"]) * trees_factor) / 25.0) * 25
    trees = clamp_int(trees, trees_min, trees_max)

    # --- depth ---
    d_step = js_round((rng.rand01() - 0.5) * 4.0)  # [-2..2], JS rounding
    depth = clamp_int(int(pivot["depth"]) + d_step, depth_min, depth_max)

    # --- lr ---
    lr_factor = math.pow(10.0, (rng.rand01() - 0.5) * 0.8)  # ~[0.40..2.51]
    lr = clamp_float(float(pivot["lr"]) * lr_factor, lr_min, lr_max)
    lr = js_round(lr * 1_000_000.0) / 1_000_000.0

    # --- minLeaf ---
    ml_factor = math.pow(2.0, (rng.rand01() - 0.5) * 2.0)  # ~[0.5..2]
    min_leaf = clamp_int(js_round(float(pivot["minLeaf"]) * ml_factor), min_leaf_min, min_leaf_max)

    # --- patience ---
    patience = int(pivot.get("patience", 25))
    if bool(pivot.get("earlyStop")):
        pat_factor = math.pow(2.0, (rng.rand01() - 0.5) * 1.6)  # ~[0.57..1.74]
        patience = clamp_int(js_round((int(pivot.get("patience", 25)) * pat_factor) / 5.0) * 5, pat_min, pat_max)

    # --- LR schedule ---
    lr_schedule = copy.deepcopy(pivot.get("lrSchedule"))
    if isinstance(lr_schedule, dict) and lr_schedule.get("mode") == "plateau":
        n0 = int(lr_schedule.get("patience") or 0)
        pct0 = int(lr_schedule.get("dropPct") or 0)
        min_lr0 = float(lr_schedule.get("minLR") or 0.0)
        n_factor = math.pow(2.0, (rng.rand01() - 0.5) * 1.2)
        lr_schedule["patience"] = clamp_int(js_round(n0 * n_factor), 1, 1000)
        pct_step = js_round((rng.rand01() - 0.5) * 20.0)
        lr_schedule["dropPct"] = clamp_int(pct0 + pct_step, 1, 99)
        min_lr_factor = math.pow(10.0, (rng.rand01() - 0.5) * 1.2)
        lr_schedule["minLR"] = clamp_float(min_lr0 * min_lr_factor, 0.0, 1.0)

    # Fixed-for-comparability fields.
    p["trees"] = trees
    p["depth"] = depth
    p["lr"] = lr
    p["minLeaf"] = min_leaf
    p["bins"] = base_params["bins"]
    p["binning"] = base_params["binning"]
    p["patience"] = patience
    p["lrSchedule"] = lr_schedule
    p["task"] = base_params["task"]
    p["seed"] = base_params["seed"]
    p["earlyStop"] = base_params["earlyStop"]
    p["splitTrain"] = base_params["splitTrain"]
    p["splitVal"] = base_params["splitVal"]
    p["nClasses"] = base_params["nClasses"]
    p["imbalance"] = copy.deepcopy(base_params.get("imbalance", {"mode": "none"}))

    cl = clamp_for_size(p["trees"], p["depth"], p["task"], p["nClasses"])
    p["trees"] = cl["trees"]
    p["depth"] = cl["depth"]
    return p


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self, engine: Engine, args: argparse.Namespace):
        self.engine = engine
        self.args = args
        self.work = args.work
        os.makedirs(self.work, exist_ok=True)

    def data_flags(self) -> List[str]:
        a = self.args
        flags = ["--task", a.task, "--input", a.input]
        if a.npz:
            flags.append("--npz")
            if a.npz_x_key:
                flags += ["--npz-x-key", a.npz_x_key]
            if a.npz_y_key:
                flags += ["--npz-y-key", a.npz_y_key]
        if a.npy_x:
            flags += ["--npy-x", a.npy_x]
        if a.npy_y:
            flags += ["--npy-y", a.npy_y]
        if a.mmap:
            flags.append("--mmap")

        if a.label_col is not None:
            flags += ["--label-col", a.label_col]
        if a.label_cols:
            flags += ["--label-cols", a.label_cols]
        if a.feature_cols:
            flags += ["--feature-cols", a.feature_cols]
        if a.delimiter != "auto":
            flags += ["--delimiter", a.delimiter]
        if a.no_header:
            flags.append("--no-header")
        if a.limit_rows is not None:
            flags += ["--limit-rows", str(a.limit_rows)]
        if a.neg_label is not None:
            flags += ["--neg-label", a.neg_label]
        if a.pos_label is not None:
            flags += ["--pos-label", a.pos_label]
        if a.class_labels:
            flags += ["--class-labels", a.class_labels]
        if a.title:
            flags += ["--title", a.title]
        if a.description:
            flags += ["--description", a.description]
        flags += ["--chain-id", str(a.chain_id)]
        return flags

    def params_flags(self, params: Dict[str, Any], *, refit: bool = False) -> List[str]:
        a = self.args
        p = copy.deepcopy(params)
        flags: List[str] = []

        flags += ["--trees", str(int(p["trees"]))]
        flags += ["--depth", str(int(p["depth"]))]
        flags += ["--lr", str(float(p["lr"]))]
        flags += ["--min-leaf", str(int(p["minLeaf"]))]
        flags += ["--seed", str(int(p["seed"]))]
        flags += ["--patience", str(int(p.get("patience", 25)))]
        flags += ["--scaleQ", str(a.scaleQ)]
        flags += ["--bins", str(int(p["bins"]))]
        flags += ["--binning", str(p["binning"])]
        flags += ["--split-train", str(float(p["splitTrain"]))]
        flags += ["--split-val", str(float(p["splitVal"]))]

        # Web refit is a separate final training: earlyStop false, Train+Val true.
        if refit:
            flags.append("--refit-train-val")
        elif bool(p.get("earlyStop")):
            flags.append("--early-stop")

        imb = p.get("imbalance") or {}
        if isinstance(imb, dict):
            mode = str(imb.get("mode") or "none")
            if mode != "none":
                flags += ["--imbalance-mode", mode]
                if imb.get("cap") is not None:
                    flags += ["--imbalance-cap", str(imb.get("cap"))]
                if imb.get("normalize"):
                    flags.append("--imbalance-normalize")
                if imb.get("stratify"):
                    flags.append("--stratify")
                if imb.get("w0") is not None:
                    flags += ["--w0", str(imb.get("w0"))]
                if imb.get("w1") is not None:
                    flags += ["--w1", str(imb.get("w1"))]
                if imb.get("classWeights"):
                    flags += ["--class-weights", str(imb.get("classWeights"))]
                if imb.get("posWeights"):
                    flags += ["--pos-weights", str(imb.get("posWeights"))]

        lrs = p.get("lrSchedule")
        if isinstance(lrs, dict):
            mode = str(lrs.get("mode") or "none").lower()
            if mode in ("plateau", "piecewise"):
                flags += ["--lr-schedule", mode]
            if mode == "plateau":
                flags += ["--lr-patience", str(int(lrs.get("patience") or 1))]
                flags += ["--lr-drop-pct", str(int(lrs.get("dropPct") or 10))]
                flags += ["--lr-min", str(float(lrs.get("minLR") or 0.0))]
            elif mode == "piecewise":
                segs = lrs.get("segments") or ""
                if segs:
                    flags += ["--lr-segments", str(segs)]

        if a.extra:
            flags += shlex.split(a.extra)
        return flags

    def evaluate(self, round_no: int, params: Dict[str, Any], *, refit: bool = False) -> Trial:
        key = param_hash(params, prefix=f"round={round_no};refit={int(refit)};")
        out_path = os.path.join(self.work, f"{'refit' if refit else 'round'}_{round_no:04d}_{key}.gl1f")
        cmd = self.engine.base_cmd() + self.data_flags() + ["--out", out_path] + self.params_flags(params, refit=refit)

        t = Trial(round=round_no, params=copy.deepcopy(params), cmd=" ".join(shlex.quote(c) for c in cmd), out_path=out_path, refit=refit)
        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=self.args.trial_timeout)
            t.seconds = time.time() - start
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace")
                t.err = err[-4000:].strip()
                return t

            pkg = parse_gl1x_package(out_path)
            raw = metric_from_package(pkg, self.args.metric_key)
            if raw is None and self.args.metric_key == "bestValMetric":
                raw = scrape_best_from_stderr(proc.stderr)
            if raw is None:
                meta = meta_from_package(pkg)
                available = ", ".join(sorted(meta.keys())) if meta else "none"
                t.err = (
                    f"could not parse objective {self.args.metric_key} from GL1X footer; "
                    f"available trainMeta keys: {available}"
                )
                return t

            t.raw = float(raw)
            t.score = t.raw if self.args.direction == "min" else -t.raw
            t.ok = math.isfinite(t.score)
            t.meta = meta_from_package(pkg)
            t.curve = curve_from_package(pkg)
            return t
        except subprocess.TimeoutExpired:
            t.seconds = time.time() - start
            t.err = f"timeout after {self.args.trial_timeout}s"
            return t
        except Exception as e:
            t.seconds = time.time() - start
            t.err = f"{type(e).__name__}: {e}"
            return t


# ---------------------------------------------------------------------------
# Web-exact search loop
# ---------------------------------------------------------------------------
class WebExactSearch:
    def __init__(self, evaluator: Evaluator, args: argparse.Namespace):
        self.ev = evaluator
        self.args = args
        self.leaderboard_path = os.path.join(args.work, "leaderboard.json")
        self.trials: List[Trial] = []
        self.best: Optional[Trial] = None
        self.refit_trial: Optional[Trial] = None
        self.t0 = time.time()
        self.stop = False
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, *_: Any) -> None:
        sys.stderr.write("\n[!] Ctrl-C — stopping after current trainer process returns, then saving.\n")
        self.stop = True

    def save(self) -> None:
        ok_sorted = sorted((t for t in self.trials if t.ok), key=lambda x: x.score)
        board = {
            "mode": "web-exact",
            "metric": self.args.metric_key,
            "direction": self.args.direction,
            "engine": self.ev.engine.kind,
            "n_eval": len(self.trials),
            "elapsed_s": round(time.time() - self.t0, 3),
            "best": trial_dict(self.best) if self.best else None,
            "refit": trial_dict(self.refit_trial) if self.refit_trial else None,
            "trials": [trial_dict(t) for t in self.trials],
            "trials_sorted": [trial_dict(t) for t in ok_sorted],
            "failed": sum(1 for t in self.trials if not t.ok),
        }
        tmp = self.leaderboard_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(board, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.leaderboard_path)

    def maybe_copy_best(self) -> None:
        src = None
        if self.refit_trial and self.refit_trial.ok and os.path.exists(self.refit_trial.out_path):
            src = self.refit_trial.out_path
        elif self.best and self.best.ok and os.path.exists(self.best.out_path):
            src = self.best.out_path
        if src and self.args.out:
            os.makedirs(os.path.dirname(os.path.abspath(self.args.out)) or ".", exist_ok=True)
            shutil.copyfile(src, self.args.out)

    def record(self, t: Trial) -> None:
        self.trials.append(t)
        improved = False
        if t.ok and (self.best is None or t.score < self.best.score):
            self.best = t
            improved = True
        flag = "*" if improved else " "
        raw_s = f"{t.raw:.6g}" if t.raw is not None else "----"
        best_s = f"{self.best.raw:.6g}" if self.best and self.best.raw is not None else "----"
        msg = (
            f"[{t.round:>4}]{flag} {self.args.metric_key}={raw_s:>10} "
            f"best={best_s:>10} {t.seconds:6.1f}s  {short_params(t.params)}"
        )
        if not t.ok and t.err:
            msg += f"  ERR:{t.err[:120]}"
        print(msg, flush=True)
        if improved:
            self.maybe_copy_best()
        self.save()

    def should_stop_before_round(self, round_no: int) -> bool:
        if self.stop:
            return True
        if self.args.time_budget and (time.time() - self.t0) >= self.args.time_budget:
            return True
        if self.args.target is not None and self.best and self.best.raw is not None:
            if self.args.direction == "min" and self.best.raw <= self.args.target:
                return True
            if self.args.direction == "max" and self.best.raw >= self.args.target:
                return True
        if self.args.no_improve_patience and self.best:
            last_best_round = self.best.round
            if (round_no - last_best_round) > self.args.no_improve_patience:
                return True
        return False

    def run(self) -> int:
        base_params = read_base_train_params(self.args)
        rng = XorShift32(web_rng_seed(int(base_params["seed"])))
        max_rounds = clamp_int(self.args.trials, 1, 1000)

        print(
            f"# engine={self.ev.engine.kind} mode=web-exact rounds={max_rounds} "
            f"metric={self.args.metric_key}({self.args.direction})"
        )
        print(f"# base: {short_params(base_params)}")
        print(f"# rng seed: seed ^ 0x9e3779b9 = {web_rng_seed(int(base_params['seed']))}")
        print(f"# leaderboard -> {self.leaderboard_path}")
        print("-" * 100)

        for round_no in range(1, max_rounds + 1):
            if self.should_stop_before_round(round_no):
                break

            if round_no == 1:
                params = copy.deepcopy(base_params)
            else:
                params = generate_heuristic_candidate(
                    base_params=base_params,
                    best_params=(self.best.params if self.best else None),
                    round_no=round_no,
                    rng=rng,
                )
            t = self.ev.evaluate(round_no, params, refit=False)
            self.record(t)

        if not self.best or not self.best.ok:
            self.save()
            return 2

        # Web behavior: optional final refit on Train+Val using usedTrees and earlyStop=false.
        if self.args.refit_train_val and not self.stop:
            refit_params = copy.deepcopy(self.best.params)
            used_trees = self.best.meta.get("usedTrees", self.best.params.get("trees"))
            try:
                used_trees = int(used_trees)
            except Exception:
                used_trees = int(self.best.params.get("trees"))
            refit_params["trees"] = used_trees
            refit_params["earlyStop"] = False
            print("-" * 100)
            print(f"# refit: training on Train+Val for {used_trees} trees")
            self.refit_trial = self.ev.evaluate(self.best.round, refit_params, refit=True)
            raw_s = f"{self.refit_trial.raw:.6g}" if self.refit_trial.raw is not None else "----"
            msg = f"[refit] {self.args.metric_key}={raw_s:>10} {self.refit_trial.seconds:6.1f}s  {short_params(refit_params)}"
            if not self.refit_trial.ok and self.refit_trial.err:
                msg += f"  ERR:{self.refit_trial.err[:120]}"
            print(msg, flush=True)

        self.maybe_copy_best()
        self.save()
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Web-exact heuristic search over GL1F hyperparameters."
    )

    g = ap.add_argument_group("engine")
    g.add_argument("--engine", choices=["python", "cpp", "auto"], default="auto")
    g.add_argument("--python-exe", default=sys.executable)
    g.add_argument("--train-script", default="train_gl1f.py")
    g.add_argument("--cpp-bin", default="./train_gl1f_cpp")

    g = ap.add_argument_group("data / task")
    g.add_argument("--task", required=True, choices=["regression", "binary_classification", "multiclass_classification", "multilabel_classification"])
    g.add_argument("--input", required=True, help="CSV path, or NPZ path with --npz")
    g.add_argument("--npz", action="store_true")
    g.add_argument("--npz-x-key", default="X")
    g.add_argument("--npz-y-key", default="y")
    g.add_argument("--npy-x", default=None)
    g.add_argument("--npy-y", default=None)
    g.add_argument("--mmap", action="store_true")
    g.add_argument("--label-col", default=None)
    g.add_argument("--label-cols", default=None)
    g.add_argument("--feature-cols", default=None)
    g.add_argument("--delimiter", default="auto")
    g.add_argument("--no-header", action="store_true")
    g.add_argument("--limit-rows", type=int, default=None)
    g.add_argument("--neg-label", default=None)
    g.add_argument("--pos-label", default=None)
    g.add_argument("--class-labels", default=None)
    g.add_argument("--n-classes", type=int, default=2, help="Class/label count for web size clamp; set for multiclass/multilabel.")

    g = ap.add_argument_group("base hyperparameters: same defaults as create.html UI")
    g.add_argument("--trees", type=int, default=250)
    g.add_argument("--depth", type=int, default=4)
    g.add_argument("--lr", type=float, default=0.05)
    g.add_argument("--min-leaf", dest="min_leaf", type=int, default=10)
    g.add_argument("--bins", type=int, default=32)
    g.add_argument("--binning", choices=["linear", "quantile"], default="linear")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--split-train", type=float, default=0.7)
    g.add_argument("--split-val", type=float, default=0.2)
    g.add_argument("--scaleQ", default="auto")
    g.add_argument("--chain-id", type=int, default=29)
    g.add_argument("--early-stop", dest="early_stop", action="store_true", default=True)
    g.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    g.add_argument("--patience", type=int, default=25)
    g.add_argument("--initial-params", default=None, help="Inline JSON, @file.json, or file path for exact web-style round-1 params. Overrides the base hyperparameter flags.")
    g.add_argument("--print-initial-params", action="store_true", help="Print the clamped round-1 params JSON and exit.")
    g.add_argument("--refit-train-val", action="store_true", help="Web-style final refit after search; not used during search rounds.")

    g = ap.add_argument_group("learning-rate schedule")
    g.add_argument("--lr-schedule", choices=["none", "plateau", "piecewise"], default="none")
    g.add_argument("--lr-patience", type=int, default=25)
    g.add_argument("--lr-drop-pct", type=float, default=10.0)
    g.add_argument("--lr-min", type=float, default=0.0)
    g.add_argument("--lr-segments", default="", help="Trainer format: start:end:lr,start:end:lr")

    g = ap.add_argument_group("class imbalance / split weighting")
    g.add_argument("--imbalance-mode", choices=["none", "auto", "manual"], default="none")
    g.add_argument("--imbalance-cap", type=float, default=20.0)
    g.add_argument("--imbalance-normalize", action="store_true", default=True)
    g.add_argument("--no-imbalance-normalize", dest="imbalance_normalize", action="store_false")
    g.add_argument("--stratify", action="store_true", default=False)
    g.add_argument("--w0", type=float, default=None)
    g.add_argument("--w1", type=float, default=None)
    g.add_argument("--class-weights", default=None)
    g.add_argument("--pos-weights", default=None)

    g = ap.add_argument_group("objective / stopping")
    g.add_argument(
        "--metric-key", "--best-by", "--objective-metric", "--target-metric",
        dest="metric_key",
        default="bestValMetric",
        help=(
            "trainMeta metric used to choose the best round. Aliases: val, test, train, "
            "val_acc, test_acc, train_acc. Raw keys also work: bestValMetric, "
            "bestTestMetric, bestTrainMetric, bestValAcc, bestTestAcc, bestTrainAcc."
        ),
    )
    g.add_argument("--direction", choices=["auto", "min", "max"], default="auto", help="auto=max for accuracy-like metrics, otherwise min")
    g.add_argument("--target", type=float, default=None, help="Optional threshold on the chosen metric; direction-aware.")
    g.add_argument("--no-improve-patience", type=int, default=0)

    g = ap.add_argument_group("search")
    g.add_argument("--trials", type=int, default=10, help="Web heuristic rounds, clamped to 1..1000. Round 1 is base params.")
    g.add_argument("--time-budget", type=float, default=0.0, help="Optional seconds budget; 0 = off. Web UI has no automatic time budget.")
    g.add_argument("--trial-timeout", type=int, default=1800)
    g.add_argument("--extra", default="", help='verbatim trainer passthrough flags, e.g. "--quantile-samples 50000"')

    g = ap.add_argument_group("metadata / output")
    g.add_argument("--title", default="")
    g.add_argument("--description", default="")
    g.add_argument("--out", default="best_model.gl1f")
    g.add_argument("--work", default="gl1f_search_runs")

    args = ap.parse_args(argv)

    if args.engine == "cpp" and (args.input.endswith(".npz") or args.input.endswith(".npy") or args.npz or args.npy_x or args.npy_y):
        sys.exit("[fatal] cpp engine supports CSV only; use --engine python for NPZ/NPY.")
    if args.trials < 1:
        sys.exit("[fatal] --trials must be >= 1 for web-exact search.")
    if args.trials > 1000:
        args.trials = 1000

    args.metric_key = normalize_metric_key(args.metric_key)
    args.direction = infer_metric_direction(args.metric_key, args.direction)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.print_initial_params:
        print(json.dumps(read_base_train_params(args), indent=2, ensure_ascii=False))
        return 0

    engine = resolve_engine(args)
    evaluator = Evaluator(engine, args)
    search = WebExactSearch(evaluator, args)
    rc = search.run()

    print("-" * 100)
    if search.best and search.best.ok:
        final_note = " (refit copied)" if (search.refit_trial and search.refit_trial.ok) else ""
        print(f"BEST {args.metric_key}={search.best.raw:.6g}  round={search.best.round}  {short_params(search.best.params)}")
        print(f"  repro: {search.best.cmd}")
        if search.refit_trial:
            if search.refit_trial.ok:
                print(f"  refit: {search.refit_trial.cmd}")
            else:
                print(f"  refit failed: {search.refit_trial.err[:400]}")
        print(f"  model: {args.out}{final_note}")
    else:
        print("No successful trial. Check --engine / dataset / trainer build / labels.")
    print(
        f"  evals={len(search.trials)} failed={sum(1 for t in search.trials if not t.ok)} "
        f"elapsed={time.time() - search.t0:.1f}s leaderboard={search.leaderboard_path}"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
