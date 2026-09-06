#!/usr/bin/env python3
"""Check the public paper against its machine-readable evidence records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TITLE = "GL1F: Reproducible Integer Tree-Ensemble Inference on the EVM"


class InvariantFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_source_digests(record: dict, name: str) -> None:
    sources = record.get("sourceDigests")
    require(isinstance(sources, dict) and sources, f"{name}: sourceDigests missing")
    for relative, expected in sources.items():
        path = Path(relative)
        require(
            isinstance(relative, str)
            and relative
            and not path.is_absolute()
            and ".." not in path.parts,
            f"{name}: invalid source path {relative!r}",
        )
        require(
            isinstance(expected, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"{name}: invalid digest for {relative}",
        )
        target = ROOT / path
        require(target.is_file(), f"{name}: source missing: {relative}")
        require(sha256(target) == expected, f"{name}: source drift: {relative}")


def check_formal_notation() -> None:
    for relative in ("paper/formal_results.tex", "paper/main.tex"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        require("KRd" in text, f"{relative}: missing v2 KRd decision count")
        require(
            re.search(r"(?<![A-Za-z])KR(?![A-Za-z])", text) is not None,
            f"{relative}: missing v2 KR tree/leaf count",
        )
        require(
            re.search(r"(?<![A-Za-z])KTd(?![A-Za-z])", text) is None,
            f"{relative}: stale v2 KTd notation",
        )


def check_paper_surfaces() -> None:
    manuscript = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    research_page = (ROOT / "research.html").read_text(encoding="utf-8")
    normalized = [
        " ".join(text.replace("\\\\", " ").split())
        for text in (manuscript, citation, research_page)
    ]
    require(all(TITLE in text for text in normalized), "paper title drift")
    require("not peer reviewed" in manuscript.lower(), "paper status missing")
    require("twocolumn" not in manuscript, "paper requests two columns")
    require(r"\onecolumn" not in manuscript, "paper switches column mode")
    require("table*" not in manuscript, "paper contains two-column table floats")
    pdf = ROOT / "GL1F.pdf"
    require(pdf.is_file() and pdf.read_bytes().startswith(b"%PDF-"), "GL1F.pdf invalid")


def check_historical_summary() -> None:
    original = (ROOT / "benchmarks/results/LIVE_CHAIN_WITNESS.md").read_text(
        encoding="utf-8"
    )
    extended = (
        ROOT / "benchmarks/results/LIVE_CHAIN_WITNESS_EXTENDED_V2.md"
    ).read_text(encoding="utf-8")
    combined = original + "\n" + extended
    for value in ("31,185,324", "12/12", "108/108", "13,342,043"):
        require(value in combined, f"historical summary drift: {value}")
    require("provider-attested" in combined.lower(), "provider boundary missing")


def check_evm_scaling() -> None:
    record = json.loads(
        (ROOT / "benchmarks/results/evm_scaling_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    require(record.get("status") == "PASS", "EVM scaling status is not PASS")
    require(record.get("totalComparisons") == 72, "EVM comparison count drift")
    require(record.get("mismatches") == 0, "EVM scaling mismatch")
    require(len(record.get("profiles", [])) == 6, "EVM profile count drift")



def check_storage_comparison() -> None:
    record = json.loads((ROOT / "benchmarks/results/storage_comparison.json").read_text())
    require(record.get("status") == "PASS", "storage comparison status")
    sources = record.get("source", {}).get("sha256", {})
    require_source_digests({"sourceDigests": sources}, "storage comparison")
    require(record.get("threeWayComparisons") == 72, "storage comparison count")
    require(record.get("referenceEvmComparisons") == 144, "storage reference count")
    require(record.get("mismatches") == 0, "storage mismatch count")
    profiles = record.get("profiles", [])
    require(len(profiles) == 6, "storage shape count")
    for profile in profiles:
        observations = profile.get("observations", [])
        require(len(observations) == 12, "storage vector count")
        for item in observations:
            require(
                item["referenceOutputQ"] == item["codeOutputQ"] == item["storageOutputQ"],
                "storage recorded output disagreement",
            )
        for backend in ("code", "storage"):
            values = [int(item[backend + "EstimatedGas"]) for item in observations]
            stats = profile["inferenceEstimates"][backend]
            require(abs(sum(values) / len(values) - stats["mean"]) < 1e-7,
                    "storage gas mean disagrees with raw observations")
        gas = profile["perModelGas"]
        writes = sum(map(int, gas["codeChunkWriteReceipts"])) + int(gas["codeTableWriteReceipt"])
        require(writes == int(gas["codeMaterialization"]), "storage publication receipt sum")



def check_deployment_archive() -> None:
    record = json.loads((ROOT / "benchmarks/results/live_chain_replay_13602838.json").read_text())
    require(record.get("schema") == "gl1f-live-chain-replay-result/v1", "archive schema")
    chain = record["chain"]
    require(chain["chainId"] == 29 and chain["blockNumber"] == 13602838, "archive chain pin")
    require(chain["blockHash"] == "0xfd3da1020c37ee3c1fe7cd0a6060dbc5ec3ec5fb90c0b812256dc33e467dace3", "archive block hash")
    archive = ROOT / "benchmarks/results/live_chain_archive_13602838.tar.gz"
    require(archive.stat().st_size == record["archive"]["bytes"], "archive size")
    require(sha256(archive) == record["archive"]["sha256"], "archive digest")
    verification = record["independentVerification"]
    require(verification["status"] == "verified", "archive replay status")
    require(verification["modelsChecked"] == 12, "archive model count")
    require(verification["vectorsChecked"] == 108, "archive vector count")
    require(verification["coreBytesChecked"] == 31185324, "archive core bytes")
    require(record["summary"]["dataChunks"] == 1306, "archive chunk count")


def check_evm_integration() -> None:
    record = json.loads(
        (ROOT / "benchmarks/results/evm_integration.json").read_text(encoding="utf-8")
    )
    require(record.get("schema") == "gl1f-local-evm-integration/v1", "EVM schema drift")
    require(record.get("status") == "PASS", "local-EVM status is not PASS")
    require_source_digests(record, "local-EVM integration")
    compiler = record.get("compiler", {})
    require(str(compiler.get("version", "")).startswith("0.8.20"), "Solidity drift")
    require(compiler.get("viaIR") is True, "viaIR drift")
    require(compiler.get("optimizerRuns") == 200, "optimizer drift")
    require(compiler.get("evmVersion") == "istanbul", "EVM target drift")
    comparisons = record.get("comparisons", {})
    require(comparisons.get("viewResults") == 18, "view comparison drift")
    require(comparisons.get("transactionResults") == 2, "transaction comparison drift")
    require(comparisons.get("mismatches") == 0, "local-EVM mismatch")


def check_parity() -> None:
    record = json.loads(
        (ROOT / "benchmarks/results/parity_matrix.json").read_text(encoding="utf-8")
    )
    require(record.get("schema") == "gl1f-training-parity-matrix/v1", "parity schema drift")
    require(record.get("status") == "PASS", "parity status is not PASS")
    require_source_digests(record, "training parity")
    require(record.get("matrixProfileCount") == 25, "parity profile count drift")
    require(record.get("matrixProfilesAreDistinct") is True, "profiles not distinct")
    require(record.get("auxiliaryControlCount") == 5, "auxiliary control count drift")
    require(record.get("totalProfileExecutions") == 30, "profile execution count drift")
    require(
        record.get("standaloneArithmeticWitnessCount") == 1,
        "arithmetic witness count drift",
    )
    profiles = record.get("profiles", [])
    require(len(profiles) == 30, "parity record count drift")
    for profile in profiles:
        label = profile.get("profileId", "unnamed")
        require(profile.get("directCoreBytesEqual") is True, f"{label}: bytes differ")
        require(profile.get("engines") == ["python", "cpp", "javascript"], f"{label}: engines drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", action="store_true", help=argparse.SUPPRESS)
    parser.parse_args()
    try:
        check_formal_notation()
        check_paper_surfaces()
        check_historical_summary()
        check_evm_scaling()
        check_storage_comparison()
        check_deployment_archive()
        check_evm_integration()
        check_parity()
    except (InvariantFailure, OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"SCIENTIFIC INVARIANTS: FAIL: {error}")
        return 1
    print(
        "SCIENTIFIC INVARIANTS: PASS "
        "(paper/formal notation; checked-in parity and local-EVM records; "
        "storage comparison; pinned archive digest; historical summary consistency)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
