#!/usr/bin/env python3
"""Publication-grade cross-implementation and format tests for GL1F.

Run from the repository root:

    python3 -m unittest -v tests.publication.test_publication

The suite compiles the production C++ trainer in a temporary directory and
executes the production Python CLI and browser WebWorker source.  It compares
core GL1F bytes, not GL1X packaging metadata (which intentionally contains a
creation timestamp and differs in scope among front ends).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tests.publication.model_format import (
    FormatError,
    Package,
    parse,
    parse_path,
    predict_q,
    quantize_js,
)


REPO = Path(__file__).resolve().parents[2]
PY_TRAINER = REPO / "train_gl1f.py"
CPP_SOURCE = REPO / "cpp" / "train_gl1f_cpp.cpp"
JS_WORKER_RUNNER = REPO / "tests" / "publication" / "js_worker_runner.mjs"
JS_INFER_RUNNER = REPO / "tests" / "publication" / "js_infer_runner.mjs"
JS_CHAIN_LOADER_RUNNER = REPO / "tests" / "publication" / "js_chain_loader_runner.mjs"
PARITY_SOURCE_FILES = (
    ".nvmrc",
    "benchmarks/generate_parity_evidence.py",
    "package-lock.json",
    "train_gl1f.py",
    "gl1f_search.py",
    "gl1f_search_web_exact_v2.py",
    "cpp/train_gl1f_cpp.cpp",
    "src/train_worker.js",
    "tests/publication/js_worker_runner.mjs",
    "tests/publication/model_format.py",
    "tests/publication/test_publication.py",
)
DESIGNATED_MATRIX_SUFFIXES = {
    "linear-all",
    "softmax-precision",
    "quantile-all",
    "weighted-stratified",
    "weighted",
    "manual-weighted",
    "early-stop",
    "refit",
    "plateau",
    "piecewise",
}


@dataclass(frozen=True)
class Dataset:
    feature_names: tuple[str, ...]
    X: tuple[tuple[float, ...], ...]
    targets: dict[str, tuple[Any, ...]]


def make_dataset(n_rows: int = 240) -> Dataset:
    """Deterministic, finite, mixed-scale data with no external dependency."""
    X: list[tuple[float, ...]] = []
    regression: list[float] = []
    binary: list[int] = []
    multiclass: list[int] = []
    multilabel: list[tuple[int, int, int]] = []

    for i in range(n_rows):
        x0 = (((i * 37) % 211) - 105) / 17.0
        x1 = (((i * 61 + 7) % 197) - 98) / 23.0
        x2 = math.sin(i * 0.173) * 2.75
        x3 = ((i % 13) - 6) / 3.0
        x4 = (((i * i + 19 * i + 3) % 149) - 74) / 29.0
        row = tuple(float(f"{v:.8g}") for v in (x0, x1, x2, x3, x4))
        X.append(row)

        reg = 0.65 * x0 - 0.35 * x1 + 0.18 * x2 * x3 + 0.12 * x4
        regression.append(float(f"{reg:.8g}"))

        binary.append(int(0.9 * x0 - 0.6 * x1 + 0.25 * x2 + 0.15 * x4 > 0.2))

        scores = (
            0.7 * x0 - 0.1 * x3,
            -0.25 * x0 + 0.8 * x1 + 0.1 * x4,
            -0.4 * x1 + 0.65 * x2 + 0.2 * x3,
        )
        multiclass.append(max(range(3), key=scores.__getitem__))

        multilabel.append(
            (
                int(x0 + 0.3 * x2 > 0.0),
                int(x1 - 0.4 * x4 > -0.25),
                int(x2 * x3 + 0.2 * x0 > 0.1),
            )
        )

    return Dataset(
        feature_names=("f0", "f1", "f2", "f3", "f4"),
        X=tuple(X),
        targets={
            "regression": tuple(regression),
            "binary_classification": tuple(binary),
            "multiclass_classification": tuple(multiclass),
            "multilabel_classification": tuple(multilabel),
        },
    )


def make_softmax_precision_dataset() -> Dataset:
    """Adversarial four-class fixture for the binary32/binary64 softmax boundary.

    This generated dataset exposed a one-Q-unit Python/native leaf divergence
    when Python rounded each exponential to binary32 before accumulating the
    softmax normalizer. Six-significant-digit rows reproduce the exact CSV
    values used by the regression witness without storing a large fixture.
    """
    rng = np.random.default_rng(20260724)
    n_rows, n_features = 4_000, 12
    raw_x = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    raw_y = (
        1.5 * raw_x[:, 0]
        - raw_x[:, 1] ** 2
        + 0.5 * raw_x[:, 2] * raw_x[:, 3]
        + rng.normal(scale=0.3, size=n_rows)
    ).astype(np.float32)
    classes = np.clip(
        (((raw_y - raw_y.min()) / (raw_y.max() - raw_y.min())) * 4).astype(int),
        0,
        3,
    )
    rows = tuple(
        tuple(float(f"{value:.6g}") for value in row)
        for row in raw_x
    )
    return Dataset(
        feature_names=tuple(f"f{index}" for index in range(n_features)),
        X=rows,
        targets={
            "multiclass_classification": tuple(int(value) for value in classes),
        },
    )


def write_csv(path: Path, dataset: Dataset, task: str) -> tuple[str, tuple[str, ...]]:
    if task == "multilabel_classification":
        labels = ("label_a", "label_b", "label_c")
    else:
        labels = ("target",)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow((*dataset.feature_names, *labels))
        for row, target in zip(dataset.X, dataset.targets[task], strict=True):
            if isinstance(target, tuple):
                writer.writerow((*row, *target))
            else:
                writer.writerow((*row, target))
    return ",".join(dataset.feature_names), labels


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def ieee754_linear_binning_operation_order_witness() -> dict[str, Any]:
    """Return the standalone arithmetic witness for linear-bin operation order.

    These operands document a concrete binary floating-point boundary.  They
    are not a separate trainer input: the v0.2.2 matrix accidentally counted
    the ordinary regression training profile twice under a second label.
    """
    minimum = -5.882352828979492
    feature_range = 0.03448295593261719
    value = -5.873732089996338
    bins = 16
    division_bin = math.floor(((value - minimum) / feature_range) * bins)
    reciprocal_bin = math.floor(
        ((value - minimum) * (1.0 / feature_range)) * bins
    )
    return {
        "witnessId": "ieee754-linear-binning-operation-order",
        "status": "PASS" if (division_bin, reciprocal_bin) == (4, 3) else "FAIL",
        "inputs": {
            "minimum": minimum,
            "featureRange": feature_range,
            "value": value,
            "bins": bins,
        },
        "results": {
            "divideThenMultiplyBin": division_bin,
            "reciprocalMultiplyBin": reciprocal_bin,
        },
        "scope": (
            "Standalone arithmetic boundary witness; these operands were not "
            "supplied as a distinct three-engine training fixture."
        ),
    }


def build_parity_evidence(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_records = sorted(
        records,
        key=lambda record: (
            record["evidenceClass"],
            record["profileId"],
        ),
    )
    ordered = [
        {
            "profileId": record["profileId"],
            "evidenceClass": record["evidenceClass"],
            "task": record["task"],
            "binning": record["binning"],
            "rows": record["rows"],
            "features": record["features"],
            "engines": ["python", "cpp", "javascript"],
            "coreSha256": record["coreSha256"]["python"],
            "coreBytes": record["coreBytes"]["python"],
            "directCoreBytesEqual": record["directCoreBytesEqual"],
            "trainingProfileSha256": record["trainingProfileSha256"],
        }
        for record in ordered_records
    ]
    matrix_count = sum(
        record["evidenceClass"] == "designated-matrix" for record in ordered
    )
    control_count = sum(
        record["evidenceClass"] == "auxiliary-control" for record in ordered
    )
    designated_fingerprints = {
        record["trainingProfileSha256"]
        for record in ordered
        if record["evidenceClass"] == "designated-matrix"
    }
    if len(designated_fingerprints) != matrix_count:
        raise ValueError(
            "designated training matrix contains duplicate input/configuration profiles"
        )
    arithmetic_witnesses = [ieee754_linear_binning_operation_order_witness()]
    return {
        "schema": "gl1f-training-parity-matrix/v1",
        "status": "PASS",
        "command": (
            "python3 benchmarks/generate_parity_evidence.py "
            "--out benchmarks/results/parity_matrix.json"
        ),
        "scope": (
            "Three separately implemented trainer paths under the recorded finite "
            "profiles; not exhaustive path coverage or external replication."
        ),
        "sourceDigests": {
            relative: sha256((REPO / relative).read_bytes())
            for relative in sorted(PARITY_SOURCE_FILES)
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "node": subprocess.run(
                ["node", "--version"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip(),
            "g++": subprocess.run(
                ["g++", "--version"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.splitlines()[0],
        },
        "matrixProfileCount": matrix_count,
        "matrixProfilesAreDistinct": True,
        "auxiliaryControlCount": control_count,
        "totalProfileExecutions": len(ordered),
        "standaloneArithmeticWitnessCount": len(arithmetic_witnesses),
        "standaloneArithmeticWitnesses": arithmetic_witnesses,
        "profiles": ordered,
    }


class PublicationParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for command in ("g++", "node"):
            if shutil.which(command) is None:
                raise RuntimeError(
                    f"{command} is required; publication parity must not be skipped"
                )
        cls._tmp = tempfile.TemporaryDirectory(prefix="gl1f-publication-")
        cls.work = Path(cls._tmp.name)
        cls.cpp = cls.work / "train_gl1f_cpp"
        subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-Wall",
                "-Wextra",
                "-pedantic",
                str(CPP_SOURCE),
                "-o",
                str(cls.cpp),
            ],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cls.dataset = make_dataset()
        cls.parity_records: list[dict[str, Any]] = []

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _train_all(
        self,
        task: str,
        *,
        binning: str = "linear",
        early_stop: bool = False,
        imbalance_mode: str = "none",
        imbalance_normalize: bool = False,
        stratify: bool = False,
        refit_train_val: bool = False,
        lr_schedule: str = "none",
        manual_weights: bool = False,
        piecewise_schedule: bool = False,
        seed: int = 20260724,
        dataset: Dataset | None = None,
        trees: int = 18,
        depth: int = 3,
        learning_rate: float = 0.075,
        min_leaf: int = 4,
        scale_q: int = 100_000,
        bins: int = 16,
        quantile_samples: int = 160,
        suffix: str = "",
    ) -> dict[str, Package]:
        case_dir = self.work / f"{task}-{binning}-{suffix or 'base'}"
        case_dir.mkdir(parents=True, exist_ok=True)
        csv_path = case_dir / "data.csv"
        selected_dataset = dataset or self.dataset
        feature_cols, label_cols = write_csv(csv_path, selected_dataset, task)

        common = [
            "--task",
            task,
            "--input",
            str(csv_path),
            "--feature-cols",
            feature_cols,
            "--trees",
            str(trees),
            "--depth",
            str(depth),
            "--lr",
            str(learning_rate),
            "--min-leaf",
            str(min_leaf),
            "--seed",
            str(seed),
            "--scaleQ",
            str(scale_q),
            "--bins",
            str(bins),
            "--binning",
            binning,
            "--quantile-samples",
            str(quantile_samples),
            "--split-train",
            "0.7",
            "--split-val",
            "0.2",
            "--patience",
            "5",
            "--imbalance-mode",
            imbalance_mode,
            "--imbalance-cap",
            "9",
            "--lr-schedule",
            lr_schedule,
            "--lr-patience",
            "5",
            "--lr-drop-pct",
            "20",
            "--lr-min",
            "0.01",
        ]
        if piecewise_schedule:
            common += ["--lr-segments", "1:6:0.08,7:18:0.03"]
        if task == "multilabel_classification":
            common += ["--label-cols", ",".join(label_cols)]
        else:
            common += ["--label-col", label_cols[0]]
        if manual_weights:
            if task == "binary_classification":
                common += ["--w0", "0.75", "--w1", "1.8"]
            elif task == "multiclass_classification":
                common += ["--class-weights", "0.7,1.2,1.8"]
            elif task == "multilabel_classification":
                common += ["--pos-weights", "1.5,0.8,2.0"]
        if early_stop:
            common.append("--early-stop")
        if imbalance_normalize:
            common.append("--imbalance-normalize")
        if stratify:
            common.append("--stratify")
        if refit_train_val:
            common.append("--refit-train-val")

        py_out = case_dir / "python.gl1f"
        cpp_out = case_dir / "cpp.gl1f"
        subprocess.run(
            [sys.executable, str(PY_TRAINER), *common, "--out", str(py_out), "--no-package"],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [str(self.cpp), *common, "--out", str(cpp_out), "--no-package"],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        worker_params: dict[str, Any] = {
            "task": task,
            "trees": trees,
            "depth": depth,
            "lr": learning_rate,
            "minLeaf": min_leaf,
            "seed": seed,
            "scaleQ": scale_q,
            "bins": bins,
            "binning": binning,
            "quantileSamples": quantile_samples,
            "splitTrain": 0.7,
            "splitVal": 0.2,
            "earlyStop": early_stop,
            "patience": 5,
            "refitTrainVal": refit_train_val,
            "imbalance": {
                "mode": imbalance_mode,
                "cap": 9,
                "normalize": imbalance_normalize,
                "stratify": stratify,
            },
            "lrSchedule": {
                "mode": lr_schedule,
                "patience": 5,
                "dropPct": 20,
                "minLR": 0.01,
                "segments": (
                    [
                        {"start": 1, "end": 6, "lr": 0.08},
                        {"start": 7, "end": 18, "lr": 0.03},
                    ]
                    if piecewise_schedule
                    else []
                ),
            },
        }
        if manual_weights:
            if task == "binary_classification":
                worker_params["imbalance"].update({"w0": 0.75, "w1": 1.8})
            elif task == "multiclass_classification":
                worker_params["imbalance"]["classWeights"] = [0.7, 1.2, 1.8]
            elif task == "multilabel_classification":
                worker_params["imbalance"]["posWeights"] = [1.5, 0.8, 2.0]
        if task in ("multiclass_classification", "multilabel_classification"):
            if task == "multiclass_classification":
                worker_params["nClasses"] = (
                    max(int(value) for value in selected_dataset.targets[task]) + 1
                )
            else:
                worker_params["nClasses"] = len(selected_dataset.targets[task][0])
        js_out = case_dir / "js.gl1f"
        spec_path = case_dir / "worker-case.json"
        spec_path.write_text(
            json.dumps(
                {
                    "X": selected_dataset.X,
                    "y": selected_dataset.targets[task],
                    "params": worker_params,
                    "output": js_out.name,
                    "metaOutput": "js-meta.json",
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        subprocess.run(
            ["node", str(JS_WORKER_RUNNER), str(spec_path)],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        packages = {
            "python": parse_path(py_out),
            "cpp": parse_path(cpp_out),
            "javascript": parse_path(js_out),
        }
        hashes = {engine: sha256(pkg.core) for engine, pkg in packages.items()}
        self.assertEqual(
            len(set(hashes.values())),
            1,
            f"core-byte mismatch for {task}/{binning}: {hashes}",
        )
        cores = {engine: pkg.core for engine, pkg in packages.items()}
        reference_engine = "python"
        for engine, core in cores.items():
            self.assertEqual(
                len(core),
                len(cores[reference_engine]),
                f"core-length mismatch for {task}/{binning}: "
                f"{reference_engine} vs {engine}",
            )
            self.assertEqual(
                core,
                cores[reference_engine],
                f"direct core-byte mismatch for {task}/{binning}: "
                f"{reference_engine} vs {engine}",
            )
        self.__class__.parity_records.append(
            {
                "profileId": f"{suffix}:{task}:{binning}",
                "evidenceClass": (
                    "designated-matrix"
                    if suffix in DESIGNATED_MATRIX_SUFFIXES
                    else "auxiliary-control"
                ),
                "task": task,
                "binning": binning,
                "rows": len(selected_dataset.X),
                "features": len(selected_dataset.feature_names),
                "parameters": {
                    "trees": trees,
                    "depth": depth,
                    "learningRate": learning_rate,
                    "minLeaf": min_leaf,
                    "seed": seed,
                    "scaleQ": scale_q,
                    "bins": bins,
                    "quantileSamples": quantile_samples,
                    "earlyStop": early_stop,
                    "imbalanceMode": imbalance_mode,
                    "imbalanceNormalize": imbalance_normalize,
                    "stratify": stratify,
                    "refitTrainVal": refit_train_val,
                    "learningRateSchedule": lr_schedule,
                    "manualWeights": manual_weights,
                    "piecewiseSchedule": piecewise_schedule,
                },
                "trainingProfileSha256": sha256(
                    json.dumps(
                        {
                            "featureNames": selected_dataset.feature_names,
                            "features": selected_dataset.X,
                            "targets": selected_dataset.targets[task],
                            "parameters": worker_params,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ),
                "coreSha256": hashes,
                "coreBytes": {
                    engine: len(core) for engine, core in cores.items()
                },
                "directCoreBytesEqual": True,
            }
        )
        return packages

    def test_linear_core_bytes_all_tasks(self) -> None:
        for task in (
            "regression",
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
        ):
            with self.subTest(task=task):
                self._train_all(task, suffix="linear-all")

    def test_ieee754_linear_binning_operation_order_regression(self) -> None:
        # Standalone arithmetic witness for the defect fixed in
        # src/train_worker.js.  The operands are not supplied to the trainers,
        # so this method is intentionally not counted as a training profile.
        witness = ieee754_linear_binning_operation_order_witness()
        self.assertEqual(witness["status"], "PASS")
        self.assertEqual(
            witness["results"],
            {
                "divideThenMultiplyBin": 4,
                "reciprocalMultiplyBin": 3,
            },
        )

    def test_multiclass_softmax_precision_regression(self) -> None:
        packages = self._train_all(
            "multiclass_classification",
            dataset=make_softmax_precision_dataset(),
            trees=30,
            depth=4,
            learning_rate=0.05,
            min_leaf=5,
            scale_q=1_000_000,
            bins=32,
            quantile_samples=50_000,
            seed=42,
            suffix="softmax-precision",
        )
        expected = "58e3d9e63c5dd1313872e0c5a07c229b071272f1137f393a0384601329a432eb"
        self.assertEqual(
            {engine: sha256(package.core) for engine, package in packages.items()},
            {
                "python": expected,
                "cpp": expected,
                "javascript": expected,
            },
        )

    def test_quantile_core_bytes_all_tasks(self) -> None:
        for task in (
            "regression",
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
        ):
            with self.subTest(task=task):
                self._train_all(task, binning="quantile", suffix="quantile-all")

    def test_weighted_stratified_classification(self) -> None:
        for task in ("binary_classification", "multiclass_classification"):
            with self.subTest(task=task):
                self._train_all(
                    task,
                    imbalance_mode="auto",
                    imbalance_normalize=True,
                    stratify=True,
                    suffix="weighted-stratified",
                )

    def test_multilabel_weighting(self) -> None:
        self._train_all(
            "multilabel_classification",
            imbalance_mode="auto",
            imbalance_normalize=True,
            suffix="weighted",
        )

    def test_manual_weight_profiles(self) -> None:
        for task in (
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
        ):
            with self.subTest(task=task):
                self._train_all(
                    task,
                    imbalance_mode="manual",
                    imbalance_normalize=True,
                    manual_weights=True,
                    suffix="manual-weighted",
                )

    def test_early_stopping_all_tasks(self) -> None:
        # Compare all three engines with early stopping enabled.  The frozen
        # fixture does not assert that the stopping branch reduced tree count.
        for task in (
            "regression",
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
        ):
            with self.subTest(task=task):
                self._train_all(task, early_stop=True, suffix="early-stop")

    def test_refit_all_tasks_fixed_budget(self) -> None:
        # Refit is tested only with early stopping disabled: once Train+Val is
        # the fitting set, Val is not an independent stopping set.
        for task in (
            "regression",
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
        ):
            with self.subTest(task=task):
                self._train_all(task, refit_train_val=True, suffix="refit")

    def test_plateau_lr_schedule(self) -> None:
        # Compare an enabled plateau configuration.  The frozen fixture does
        # not assert that a learning-rate reduction was triggered.
        self._train_all(
            "binary_classification",
            early_stop=True,
            lr_schedule="plateau",
            suffix="plateau",
        )

    def test_piecewise_lr_schedule(self) -> None:
        self._train_all(
            "regression",
            lr_schedule="piecewise",
            piecewise_schedule=True,
            suffix="piecewise",
        )

    def test_same_engine_repeatability(self) -> None:
        first = self._train_all("regression", binning="quantile", suffix="repeat-a")
        second = self._train_all("regression", binning="quantile", suffix="repeat-b")
        for engine in first:
            with self.subTest(engine=engine):
                self.assertEqual(first[engine].core, second[engine].core)

    def test_explicit_zero_seed_three_way_parity(self) -> None:
        # Xorshift cannot use an all-zero state. All implementations normalize
        # an explicit seed=0 to the same documented internal fallback.
        self._train_all("regression", seed=0, suffix="seed-zero")

    def test_reference_and_production_js_inference_agree(self) -> None:
        packages = self._train_all("multiclass_classification", suffix="inference")
        package = packages["python"]
        rows = [
            self.dataset.X[0],
            self.dataset.X[17],
            self.dataset.X[103],
            (0.5, -0.5, 1.25, 0.0, -1.0),
        ]
        case_dir = self.work / "inference-check"
        case_dir.mkdir(exist_ok=True)
        model_path = case_dir / "model.gl1f"
        cases_path = case_dir / "rows.json"
        model_path.write_bytes(package.core)
        cases_path.write_text(json.dumps(rows), encoding="utf-8")
        proc = subprocess.run(
            [
                "node",
                str(JS_INFER_RUNNER),
                str(model_path),
                str(cases_path),
            ],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        actual = json.loads(proc.stdout)["predictions"]
        expected = [list(predict_q(package, row)) for row in rows]
        self.assertEqual(actual, expected)

    def test_rounding_and_strict_greater_than_boundary(self) -> None:
        from train_gl1f import clamp_i32, js_round, quantize_to_i32
        from gl1f_search import js_round as search_round
        from gl1f_search_web_exact_v2 import js_round as search_v2_round

        values = [
            -1e308, -2147483648.5, -2.5, -1.5,
            math.nextafter(-0.5, -math.inf), -0.5,
            math.nextafter(-0.5, math.inf), -0.0,
            math.nextafter(0.5, 0.0), 0.5, math.nextafter(0.5, math.inf),
            1.5, 2.5, 2147483647.5, float(2**52 + 1), 1e308,
        ]
        proc = subprocess.run(
            ["node", "-e",
             "console.log(JSON.stringify(JSON.parse(process.argv[1]).map(Math.round)))",
             json.dumps(values)],
            check=True, capture_output=True, text=True,
        )
        rounded = json.loads(proc.stdout)
        for helper in (js_round, search_round, search_v2_round):
            self.assertEqual([helper(value) for value in values], rounded)
        expected = [clamp_i32(value) for value in rounded]
        self.assertEqual(quantize_to_i32(np.array(values), 1).tolist(), expected)
        self.assertEqual([quantize_js(value, 1) for value in values], expected)
        self.assertEqual(
            quantize_to_i32(np.array([-1e308, 1e308]), 2).tolist(),
            [-2147483648, 2147483647],
        )
        self.assertEqual(
            [quantize_js(value, 2) for value in (-1e308, 1e308)],
            [-2147483648, 2147483647],
        )

        # Compile the production helpers themselves, including their int64
        # conversion bounds, and compare their clamped results with Node.
        rounding_source = self.work / "rounding-helper.cpp"
        rounding_binary = self.work / "rounding-helper"
        rounding_source.write_text(
            '#define main gl1f_cli_main\n#include '
            + json.dumps(str(CPP_SOURCE))
            + '\n#undef main\nint main(int argc, char** argv) {\n'
              '  for (int i = 1; i < argc; ++i) {\n'
              '    double x = std::strtod(argv[i], nullptr);\n'
              '    std::cout << clamp_i32(js_round_double(x)) << " "\n'
              '              << quantize_to_i32(x, 1) << " "\n'
              '              << quantize_to_i32(x, 2) << "\\n";\n'
              '  }\n}\n',
            encoding="utf-8",
        )
        subprocess.run(
            ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math",
             str(rounding_source), "-o", str(rounding_binary)],
            check=True, capture_output=True, text=True,
        )
        cpp_result = subprocess.run(
            [str(rounding_binary), *map(repr, values)],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(
            [list(map(int, line.split())) for line in cpp_result.stdout.splitlines()],
            [[expected[index], expected[index], quantize_js(value, 2)]
             for index, value in enumerate(values)],
        )

        # An actual training control: a +1 residual, Q=2, and a learning rate
        # just below 1/4 yield a leaf immediately below +1/2. The old Python
        # and C++ add-then-floor helpers emitted 1; JavaScript correctly emits 0.
        boundary_data = Dataset(
            feature_names=("x",),
            X=tuple((float(index % 2),) for index in range(240)),
            targets={"regression": tuple(float(2 * (index % 2)) for index in range(240))},
        )
        trained = self._train_all(
            "regression", dataset=boundary_data, trees=1, depth=1,
            learning_rate=math.nextafter(0.25, 0.0), min_leaf=1,
            scale_q=2, bins=8, suffix="half-boundary",
        )
        for trained_package in trained.values():
            self.assertEqual(trained_package.header.base_q, (2,))
            self.assertEqual(struct.unpack_from("<ii", trained_package.core, 32), (0, 0))

        # v1, one feature, depth=1, one tree, base=3, scale=10,
        # threshold=5, leaves=(-7, 11).
        raw = bytearray(24 + 8 + 8)
        raw[:4] = b"GL1F"
        raw[4] = 1
        struct.pack_into("<HHI", raw, 6, 1, 1, 1)
        struct.pack_into("<iI", raw, 14, 3, 10)
        struct.pack_into("<HiH", raw, 24, 0, 5, 0)
        struct.pack_into("<ii", raw, 32, -7, 11)
        package = parse(bytes(raw))

        # Equality goes left; 0.55*10 rounds to 6 and goes right.
        self.assertEqual(predict_q(package, (0.5,)), (-4,))
        self.assertEqual(predict_q(package, (0.55,)), (14,))

        # JavaScript half rounding for negative values: -0.15*10 -> -1.
        raw_neg = bytearray(raw)
        struct.pack_into("<i", raw_neg, 26, -1)
        neg_package = parse(bytes(raw_neg))
        self.assertEqual(predict_q(neg_package, (-0.15,)), (-4,))

    def test_strict_parser_rejects_malformed_models(self) -> None:
        valid = bytearray(24 + 8 + 8)
        valid[:4] = b"GL1F"
        valid[4] = 1
        struct.pack_into("<HHI", valid, 6, 1, 1, 1)
        struct.pack_into("<iI", valid, 14, 0, 100)
        struct.pack_into("<HiH", valid, 24, 0, 0, 0)

        mutations: dict[str, bytes] = {}
        short = bytes(valid[:-1])
        mutations["truncated"] = short
        bad_feature = bytearray(valid)
        struct.pack_into("<H", bad_feature, 24, 1)
        mutations["feature-index"] = bytes(bad_feature)
        bad_reserved = bytearray(valid)
        bad_reserved[5] = 1
        mutations["header-reserved"] = bytes(bad_reserved)
        bad_node_reserved = bytearray(valid)
        bad_node_reserved[30] = 1
        mutations["node-reserved"] = bytes(bad_node_reserved)
        zero_scale = bytearray(valid)
        struct.pack_into("<I", zero_scale, 18, 0)
        mutations["zero-scale"] = bytes(zero_scale)
        bad_depth = bytearray(valid)
        struct.pack_into("<H", bad_depth, 8, 0)
        mutations["zero-depth"] = bytes(bad_depth)
        mutations["unframed-trailing"] = bytes(valid) + b"x"
        zero_v2_trees = bytearray(24 + 2 * 4)
        zero_v2_trees[:4] = b"GL1F"
        zero_v2_trees[4] = 2
        struct.pack_into("<HHI", zero_v2_trees, 6, 1, 1, 0)
        struct.pack_into("<iIH", zero_v2_trees, 14, 0, 100, 2)
        mutations["v2-zero-trees"] = bytes(zero_v2_trees)
        for constant in ("NaN", "Infinity", "-Infinity"):
            payload = ('{"nested":[' + constant + ']}').encode("utf-8")
            mutations[f"footer-constant-{constant}"] = (
                bytes(valid) + b"GL1X\x01\0\0\0"
                + struct.pack("<I", len(payload)) + payload
            )

        for name, malformed in mutations.items():
            with self.subTest(name=name), self.assertRaises(FormatError):
                parse(malformed)

    def test_production_js_rejects_malformed_core_and_manifest(self) -> None:
        valid = bytearray(24 + 8 + 8)
        valid[:4] = b"GL1F"
        valid[4] = 1
        struct.pack_into("<HHI", valid, 6, 1, 1, 1)
        struct.pack_into("<iI", valid, 14, 0, 100)
        struct.pack_into("<HiH", valid, 24, 0, 0, 0)

        malformed: dict[str, bytes] = {"truncated": bytes(valid[:-1])}
        bad_feature = bytearray(valid)
        struct.pack_into("<H", bad_feature, 24, 1)
        malformed["feature-index"] = bytes(bad_feature)
        bad_header_reserved = bytearray(valid)
        bad_header_reserved[5] = 1
        malformed["header-reserved"] = bytes(bad_header_reserved)
        bad_node_reserved = bytearray(valid)
        bad_node_reserved[30] = 1
        malformed["node-reserved"] = bytes(bad_node_reserved)
        zero_scale = bytearray(valid)
        struct.pack_into("<I", zero_scale, 18, 0)
        malformed["zero-scale"] = bytes(zero_scale)
        zero_features = bytearray(valid)
        struct.pack_into("<H", zero_features, 6, 0)
        malformed["zero-features"] = bytes(zero_features)
        unsafe_depth = bytearray(valid)
        struct.pack_into("<H", unsafe_depth, 8, 31)
        malformed["unsafe-depth"] = bytes(unsafe_depth)
        malformed["unframed-trailing"] = bytes(valid) + b"x"
        zero_v2_trees = bytearray(24 + 2 * 4)
        zero_v2_trees[:4] = b"GL1F"
        zero_v2_trees[4] = 2
        struct.pack_into("<HHI", zero_v2_trees, 6, 1, 1, 0)
        struct.pack_into("<iIH", zero_v2_trees, 14, 0, 100, 2)
        malformed["v2-zero-trees"] = bytes(zero_v2_trees)
        for constant in ("NaN", "Infinity", "-Infinity"):
            payload = ('{"nested":[' + constant + ']}').encode("utf-8")
            malformed[f"footer-constant-{constant}"] = (
                bytes(valid) + b"GL1X\x01\0\0\0"
                + struct.pack("<I", len(payload)) + payload
            )

        case_dir = self.work / "malformed-production-js"
        case_dir.mkdir(exist_ok=True)
        rows_path = case_dir / "rows.json"
        rows_path.write_text("[[0.0]]\n", encoding="utf-8")
        for name, raw in malformed.items():
            with self.subTest(name=name):
                model_path = case_dir / f"{name}.gl1f"
                model_path.write_bytes(raw)
                proc = subprocess.run(
                    ["node", str(JS_INFER_RUNNER), str(model_path), str(rows_path)],
                    cwd=REPO,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertNotEqual(
                    proc.returncode,
                    0,
                    f"production decoder accepted malformed model {name}: {proc.stdout}",
                )

        # Exercise the production GL1C chain loader with an exact three-chunk
        # manifest, then mutate one invariant at a time. This covers the
        # canonical table-word shape and exact final-chunk length in addition
        # to the core decoder checks above.
        table_address = "0x0000000000000000000000000000000000000100"
        chunk_addresses = [
            "0x0000000000000000000000000000000000000201",
            "0x0000000000000000000000000000000000000202",
            "0x0000000000000000000000000000000000000203",
        ]
        chunk_size = 17
        core = bytes(valid)

        def gl1c(payload: bytes) -> str:
            return "0x" + (b"GL1C" + payload).hex()

        def pointer_payload(addresses: list[str]) -> bytes:
            return b"".join(
                b"\x00" * 12 + bytes.fromhex(address.removeprefix("0x"))
                for address in addresses
            )

        base_code = {
            table_address: gl1c(pointer_payload(chunk_addresses)),
            **{
                address: gl1c(core[index * chunk_size : (index + 1) * chunk_size])
                for index, address in enumerate(chunk_addresses)
            },
        }
        base_info: list[Any] = [table_address, chunk_size, len(chunk_addresses), len(core)]

        chain_dir = self.work / "production-js-chain-loader"
        chain_dir.mkdir(exist_ok=True)

        def run_chain_case(
            name: str,
            *,
            info: list[Any] | None = None,
            code: dict[str, str] | None = None,
        ) -> subprocess.CompletedProcess[str]:
            case_path = chain_dir / f"{name}.json"
            case_path.write_text(
                json.dumps(
                    {
                        "info": base_info if info is None else info,
                        "code": base_code if code is None else code,
                        "expectedHex": "0x" + core.hex(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return subprocess.run(
                ["node", str(JS_CHAIN_LOADER_RUNNER), str(case_path)],
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        accepted = run_chain_case("valid")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(json.loads(accepted.stdout)["hex"], "0x" + core.hex())

        malformed_manifests: dict[str, tuple[list[Any], dict[str, str]]] = {}

        malformed_manifests["zero-table-pointer"] = (
            [
                "0x0000000000000000000000000000000000000000",
                chunk_size,
                len(chunk_addresses),
                len(core),
            ],
            base_code,
        )
        malformed_manifests["bad-table-magic"] = (
            base_info,
            {
                **base_code,
                table_address: "0x" + (b"BAD!" + pointer_payload(chunk_addresses)).hex(),
            },
        )
        malformed_manifests["pointer-table-short"] = (
            base_info,
            {
                **base_code,
                table_address: gl1c(pointer_payload(chunk_addresses[:-1])),
            },
        )
        high_order_table = bytearray(b"GL1C" + pointer_payload(chunk_addresses))
        high_order_table[4] = 1
        malformed_manifests["pointer-high-order-byte"] = (
            base_info,
            {**base_code, table_address: "0x" + high_order_table.hex()},
        )
        malformed_manifests["pointer-table-extra-entry"] = (
            base_info,
            {
                **base_code,
                table_address: gl1c(pointer_payload(chunk_addresses) + b"\x00" * 32),
            },
        )
        zero_pointer_table = bytearray(pointer_payload(chunk_addresses))
        zero_pointer_table[12:32] = b"\x00" * 20
        malformed_manifests["zero-pointer"] = (
            base_info,
            {**base_code, table_address: gl1c(bytes(zero_pointer_table))},
        )
        malformed_manifests["chunk-size-too-small"] = (
            [table_address, 3, len(chunk_addresses), len(core)],
            base_code,
        )
        malformed_manifests["chunk-size-too-large"] = (
            [table_address, 24_573, len(chunk_addresses), len(core)],
            base_code,
        )
        malformed_manifests["zero-chunks"] = (
            [table_address, chunk_size, 0, len(core)],
            base_code,
        )
        malformed_manifests["too-many-chunks"] = (
            [table_address, chunk_size, 768, len(core)],
            base_code,
        )
        malformed_manifests["zero-total-bytes"] = (
            [table_address, chunk_size, len(chunk_addresses), 0],
            base_code,
        )
        malformed_manifests["chunk-count-mismatch"] = (
            [table_address, chunk_size, 2, len(core)],
            base_code,
        )
        malformed_manifests["short-nonfinal-chunk"] = (
            base_info,
            {**base_code, chunk_addresses[0]: gl1c(core[: chunk_size - 1])},
        )
        malformed_manifests["long-nonfinal-chunk"] = (
            base_info,
            {
                **base_code,
                chunk_addresses[0]: gl1c(core[:chunk_size] + b"\x00"),
            },
        )
        malformed_manifests["bad-chunk-magic"] = (
            base_info,
            {
                **base_code,
                chunk_addresses[0]: "0x" + (b"BAD!" + core[:chunk_size]).hex(),
            },
        )
        malformed_manifests["short-final-chunk"] = (
            base_info,
            {
                **base_code,
                chunk_addresses[-1]: gl1c(core[2 * chunk_size : -1]),
            },
        )
        malformed_manifests["long-final-chunk"] = (
            base_info,
            {
                **base_code,
                chunk_addresses[-1]: gl1c(core[2 * chunk_size :] + b"\x00"),
            },
        )
        mutated_core = bytearray(core)
        mutated_core[-1] ^= 1
        malformed_manifests["same-length-content-mutation"] = (
            base_info,
            {
                **base_code,
                chunk_addresses[-1]: gl1c(bytes(mutated_core[2 * chunk_size :])),
            },
        )

        for name, (info, code) in malformed_manifests.items():
            with self.subTest(manifest=name):
                rejected = run_chain_case(name, info=info, code=code)
                self.assertNotEqual(
                    rejected.returncode,
                    0,
                    f"production chain loader accepted malformed manifest {name}: "
                    f"{rejected.stdout}",
                )

    def test_gl1x_footer_framing(self) -> None:
        case_dir = self.work / "footer"
        case_dir.mkdir(exist_ok=True)
        csv_path = case_dir / "data.csv"
        features, labels = write_csv(csv_path, self.dataset, "regression")
        out = case_dir / "packaged.gl1f"
        subprocess.run(
            [
                sys.executable,
                str(PY_TRAINER),
                "--task",
                "regression",
                "--input",
                str(csv_path),
                "--feature-cols",
                features,
                "--label-col",
                labels[0],
                "--trees",
                "2",
                "--depth",
                "2",
                "--scaleQ",
                "10000",
                "--out",
                str(out),
            ],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        package = parse_path(out)
        self.assertIsNotNone(package.footer)
        self.assertEqual(package.footer["kind"], "GL1F_PACKAGE")
        self.assertEqual(package.footer["model"]["bytes"], len(package.core))
        self.assertEqual(package.footer["model"]["gl1fVersion"], package.header.version)

        rows_path = case_dir / "rows.json"
        rows_path.write_text(json.dumps([self.dataset.X[0]]) + "\n", encoding="utf-8")
        js_result = subprocess.run(
            ["node", str(JS_INFER_RUNNER), str(out), str(rows_path)],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        js_payload = json.loads(js_result.stdout)
        self.assertTrue(js_payload["hasFooter"])
        self.assertEqual(js_payload["modelLength"], len(package.core))


if __name__ == "__main__":
    unittest.main(verbosity=2)
