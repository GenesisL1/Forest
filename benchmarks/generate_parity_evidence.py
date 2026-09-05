#!/usr/bin/env python3
"""Run the full publication parity suite and freeze its machine record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.publication.test_publication import (
    PublicationParityTests,
    build_parity_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/parity_matrix.json"),
    )
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PublicationParityTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if result.testsRun != 18 or result.skipped:
        raise RuntimeError(
            "publication parity preflight requires exactly 18 tests with no skips; "
            f"ran={result.testsRun}, skipped={len(result.skipped)}"
        )

    evidence = build_parity_evidence(PublicationParityTests.parity_records)
    if evidence["matrixProfileCount"] != 25:
        raise RuntimeError(
            "expected 25 distinct designated matrix profiles, received "
            f"{evidence['matrixProfileCount']}"
        )
    if evidence.get("matrixProfilesAreDistinct") is not True:
        raise RuntimeError("designated matrix profile fingerprints are not distinct")
    if evidence["auxiliaryControlCount"] != 4:
        raise RuntimeError(
            "expected four auxiliary controls, received "
            f"{evidence['auxiliaryControlCount']}"
        )
    if evidence.get("standaloneArithmeticWitnessCount") != 1:
        raise RuntimeError("expected one standalone arithmetic witness")
    arithmetic_witnesses = evidence.get("standaloneArithmeticWitnesses")
    if (
        not isinstance(arithmetic_witnesses, list)
        or len(arithmetic_witnesses) != 1
        or arithmetic_witnesses[0].get("status") != "PASS"
    ):
        raise RuntimeError("standalone arithmetic witness did not pass")

    rendered = json.dumps(evidence, indent=2).replace(
        '      "engines": [\n'
        '        "python",\n'
        '        "cpp",\n'
        '        "javascript"\n'
        "      ],",
        '      "engines": ["python", "cpp", "javascript"],',
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered + "\n", encoding="utf-8")
    print(
        "PARITY EVIDENCE: PASS "
        f"({evidence['matrixProfileCount']} distinct matrix profiles; "
        f"{evidence['standaloneArithmeticWitnessCount']} arithmetic witness; "
        f"{evidence['auxiliaryControlCount']} controls; {args.out})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
