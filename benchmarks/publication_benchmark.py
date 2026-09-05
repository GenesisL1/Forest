#!/usr/bin/env python3
"""Reproducible, end-to-end GL1F trainer benchmark.

This benchmark reports operational wall-clock time, including process startup,
input parsing, training, serialization, and output writing. Python and C++ read
the same CSV. The browser worker is executed under Node and reads an equivalent
JSON matrix, so its timing is useful operationally but is not a pure
language-kernel comparison.

Example:

    python3 benchmarks/publication_benchmark.py \
      --rows 3000 --features 12 --trees 60 --repeats 30 \
      --out benchmarks/results/publication_benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PY_TRAINER = REPO / "train_gl1f.py"
CPP_SOURCE = REPO / "cpp" / "train_gl1f_cpp.cpp"
JS_RUNNER = REPO / "tests" / "publication" / "js_worker_runner.mjs"


def checked_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=REPO,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def git_value(arguments: list[str], fallback: str = "unavailable") -> str:
    try:
        return checked_output(["git", *arguments])
    except (OSError, subprocess.CalledProcessError):
        return fallback


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def make_matrix(n_rows: int, n_features: int) -> tuple[list[list[float]], dict[str, list[float]]]:
    X: list[list[float]] = []
    regression: list[float] = []
    binary: list[float] = []
    for row_idx in range(n_rows):
        row: list[float] = []
        state = (0x9E3779B9 ^ row_idx) & 0xFFFFFFFF
        for feature_idx in range(n_features):
            state = (1664525 * state + 1013904223 + feature_idx * 97) & 0xFFFFFFFF
            uniform = state / 4294967296.0
            value = (uniform - 0.5) * 8.0 + 0.35 * math.sin(
                (row_idx + 1) * (feature_idx + 3) * 0.007
            )
            row.append(float(f"{value:.8g}"))
        X.append(row)
        signal = sum(
            ((-1.0 if j % 3 == 1 else 1.0) / (j + 1)) * value
            for j, value in enumerate(row)
        )
        nonlinear = 0.18 * row[0] * row[min(3, n_features - 1)]
        regression.append(float(f"{signal + nonlinear:.8g}"))
        binary.append(float(signal + nonlinear > 0.15))
    return X, {"regression": regression, "binary_classification": binary}


def write_csv(path: Path, X: list[list[float]], y: list[float]) -> list[str]:
    features = [f"f{i}" for i in range(len(X[0]))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([*features, "target"])
        for row, target in zip(X, y, strict=True):
            writer.writerow([*row, target])
    return features


def timed(command: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(
        command,
        cwd=REPO,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - start


def core_sha(path: Path) -> tuple[str, int]:
    # The benchmark writes --no-package/core-only outputs.
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def run_case(
    work: Path,
    cpp_binary: Path,
    X: list[list[float]],
    y: list[float],
    task: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case = work / task
    case.mkdir(parents=True)
    csv_path = case / "data.csv"
    features = write_csv(csv_path, X, y)

    common = [
        "--task",
        task,
        "--input",
        str(csv_path),
        "--feature-cols",
        ",".join(features),
        "--label-col",
        "target",
        "--trees",
        str(args.trees),
        "--depth",
        str(args.depth),
        "--lr",
        str(args.learning_rate),
        "--min-leaf",
        str(args.min_leaf),
        "--seed",
        str(args.seed),
        "--scaleQ",
        str(args.scale_q),
        "--bins",
        str(args.bins),
        "--binning",
        args.binning,
        "--split-train",
        "0.7",
        "--split-val",
        "0.2",
        "--no-package",
    ]
    py_out = case / "python.gl1f"
    cpp_out = case / "cpp.gl1f"
    js_out = case / "javascript.gl1f"
    commands = {
        "python": [sys.executable, str(PY_TRAINER), *common, "--out", str(py_out)],
        "cpp": [str(cpp_binary), *common, "--out", str(cpp_out)],
    }

    spec = {
        "X": X,
        "y": y,
        "params": {
            "task": task,
            "trees": args.trees,
            "depth": args.depth,
            "lr": args.learning_rate,
            "minLeaf": args.min_leaf,
            "seed": args.seed,
            "scaleQ": args.scale_q,
            "bins": args.bins,
            "binning": args.binning,
            "splitTrain": 0.7,
            "splitVal": 0.2,
            "earlyStop": False,
            "imbalance": {"mode": "none", "stratify": False},
        },
        "output": js_out.name,
    }
    spec_path = case / "worker-case.json"
    spec_path.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    commands["javascript_node"] = ["node", str(JS_RUNNER), str(spec_path)]

    # One untimed warm-up per executable/page-worker path.
    for command in commands.values():
        timed(command)

    samples: dict[str, list[float]] = {name: [] for name in commands}
    # Round-robin order reduces slow thermal/load drift favoring one engine.
    names = list(commands)
    for repeat_idx in range(args.repeats):
        order = names[repeat_idx % len(names) :] + names[: repeat_idx % len(names)]
        for name in order:
            samples[name].append(timed(commands[name]))

    outputs = {"python": py_out, "cpp": cpp_out, "javascript_node": js_out}
    hashes = {name: core_sha(path) for name, path in outputs.items()}
    if len({value[0] for value in hashes.values()}) != 1:
        raise RuntimeError(f"benchmark outputs are not byte-identical: {hashes}")

    timings: dict[str, dict[str, Any]] = {}
    for name, values in samples.items():
        median = statistics.median(values)
        if len(values) >= 2:
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        else:
            q1 = q3 = median
        timings[name] = {
            "seconds": values,
            "median_seconds": median,
            "q1_seconds": q1,
            "q3_seconds": q3,
            "iqr_seconds": q3 - q1,
            "median_absolute_deviation_seconds": statistics.median(
                abs(value - median) for value in values
            ),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
    timings["cpp"]["speedup_vs_python_median"] = (
        timings["python"]["median_seconds"] / timings["cpp"]["median_seconds"]
    )
    timings["javascript_node"]["speedup_vs_python_median"] = (
        timings["python"]["median_seconds"] / timings["javascript_node"]["median_seconds"]
    )

    digest, byte_length = next(iter(hashes.values()))
    return {
        "task": task,
        "core_sha256": digest,
        "core_bytes": byte_length,
        "timings": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--trees", type=int, default=60)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.075)
    parser.add_argument("--min-leaf", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--scale-q", type=int, default=100000)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--binning", choices=("linear", "quantile"), default="linear")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "benchmarks" / "results" / "publication_benchmark.json",
    )
    args = parser.parse_args()
    if args.rows < 30 or args.features < 1 or args.trees < 1 or args.repeats < 1:
        parser.error("rows>=30, features>=1, trees>=1, and repeats>=1 are required")

    for command in ("g++", "node"):
        if shutil.which(command) is None:
            raise SystemExit(f"{command} is required")

    X, targets = make_matrix(args.rows, args.features)
    with tempfile.TemporaryDirectory(prefix="gl1f-benchmark-") as tmp:
        work = Path(tmp)
        cpp_binary = work / "train_gl1f_cpp"
        subprocess.run(
            [
                "g++",
                "-O3",
                "-DNDEBUG",
                "-std=c++17",
                "-ffp-contract=off",
                "-fno-fast-math",
                str(CPP_SOURCE),
                "-o",
                str(cpp_binary),
            ],
            cwd=REPO,
            check=True,
        )
        cases = [
            run_case(work, cpp_binary, X, targets[task], task, args)
            for task in ("regression", "binary_classification")
        ]

    try:
        import numpy

        numpy_version = numpy.__version__
    except Exception:
        numpy_version = "unavailable"

    result = {
        "schema": "gl1f-publication-benchmark-v1",
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "revision": git_value(["rev-parse", "HEAD"]),
            "dirty": git_value(["status", "--porcelain"], "") != "",
        },
        "methodology": {
            "clock": "time.perf_counter wall time",
            "scope": "process startup + input parse + train + serialize + write",
            "warmups_per_engine": 1,
            "timed_repeats": args.repeats,
            "order": "round-robin rotated",
            "python_cpp_input": "identical CSV",
            "javascript_input": "equivalent JSON numeric matrix; timing is not a pure kernel comparison",
            "compiler_flags": "-O3 -DNDEBUG -std=c++17 -ffp-contract=off -fno-fast-math",
            "threads": "single-threaded trainer code; host scheduling not pinned",
        },
        "parameters": {
            "rows": args.rows,
            "features": args.features,
            "trees": args.trees,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "min_leaf": args.min_leaf,
            "seed": args.seed,
            "scaleQ": args.scale_q,
            "bins": args.bins,
            "binning": args.binning,
        },
        "environment": {
            "platform": platform.platform(),
            "cpu": cpu_model(),
            "logical_cpu_count_visible": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": numpy_version,
            "node": checked_output(["node", "--version"]),
            "compiler": checked_output(["g++", "--version"]).splitlines()[0],
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"{json.dumps(result, indent=2)}\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
