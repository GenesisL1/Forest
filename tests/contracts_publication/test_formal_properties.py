#!/usr/bin/env python3
"""Executable witnesses for the contract/formal properties.

These tests do not replace the proofs in ``paper/formal_results.tex``.  They
exercise boundary cases and make the proof assumptions easy to regression-test.
"""

from __future__ import annotations

import itertools
import math
import unittest
from fractions import Fraction
from pathlib import Path

from tests.publication.model_format import parse_path


REPO = Path(__file__).resolve().parents[2]
I32_ABS_BOUND = 1 << 31
JS_EXACT_BOUND = 1 << 53
STORE_PAYLOAD_MAX = 24_572
POINTER_SLOT_BYTES = 32


def traverse(decisions: tuple[bool, ...]) -> int:
    index = 0
    for go_right in decisions:
        index = 2 * index + (2 if go_right else 1)
    return index


def chunk_payloads(raw: bytes, chunk_size: int) -> list[bytes]:
    return [
        raw[offset : offset + chunk_size]
        for offset in range(0, len(raw), chunk_size)
    ]


def runtime_style_read(chunks: list[bytes], chunk_size: int, offset: int, length: int) -> bytes:
    """Pure model of ForestRuntime._readBytes, including EVM zero padding."""
    chunk_index, in_chunk = divmod(offset, chunk_size)
    first = chunks[chunk_index] if chunk_index < len(chunks) else b""

    def extcodecopy_like(payload: bytes, start: int, count: int) -> bytes:
        selected = payload[start : start + count]
        return selected + b"\0" * (count - len(selected))

    if in_chunk + length <= chunk_size:
        return extcodecopy_like(first, in_chunk, length)

    first_count = chunk_size - in_chunk
    second_count = length - first_count
    second = chunks[chunk_index + 1] if chunk_index + 1 < len(chunks) else b""
    return (
        extcodecopy_like(first, in_chunk, first_count)
        + extcodecopy_like(second, 0, second_count)
    )


def round_half_toward_positive_infinity(value: Fraction) -> int:
    """Exact counterpart of floor(x + 1/2)."""
    shifted = value + Fraction(1, 2)
    return shifted.numerator // shifted.denominator


class TraversalProperties(unittest.TestCase):
    def test_every_path_maps_bijectively_to_the_leaf_interval(self) -> None:
        for depth in range(0, 13):
            with self.subTest(depth=depth):
                reached = {
                    traverse(tuple(path))
                    for path in itertools.product((False, True), repeat=depth)
                }
                internal = (1 << depth) - 1
                expected = set(range(internal, internal + (1 << depth)))
                self.assertEqual(reached, expected)

    def test_level_interval_invariant(self) -> None:
        for depth in range(0, 13):
            for path in itertools.product((False, True), repeat=depth):
                index = traverse(tuple(path))
                self.assertGreaterEqual(index, (1 << depth) - 1)
                self.assertLessEqual(index, (1 << (depth + 1)) - 2)


class StorageAddressingProperties(unittest.TestCase):
    def test_one_to_four_byte_reads_reconstruct_exact_bytes_when_chunk_size_at_least_four(self) -> None:
        raw = bytes((index * 73 + 19) % 256 for index in range(100_003))
        for chunk_size in (4, 5, 7, 31, 32, 127, 23_999, 24_000, 24_572):
            chunks = chunk_payloads(raw, chunk_size)
            boundary_offsets = {
                0,
                1,
                len(raw) - 4,
                max(0, chunk_size - 3),
                max(0, chunk_size - 2),
                max(0, chunk_size - 1),
                min(len(raw) - 4, chunk_size),
                min(len(raw) - 4, 2 * chunk_size - 1),
            }
            for offset in sorted(boundary_offsets):
                for length in (1, 2, 4):
                    if offset + length > len(raw):
                        continue
                    with self.subTest(chunk_size=chunk_size, offset=offset, length=length):
                        self.assertEqual(
                            runtime_style_read(chunks, chunk_size, offset, length),
                            raw[offset : offset + length],
                        )

    def test_chunk_size_below_four_has_a_concrete_three_chunk_counterexample(self) -> None:
        raw = b"ABCDEFGH"
        chunks = chunk_payloads(raw, 2)
        # Offset one, length four spans logical chunks 0, 1, and 2.  The
        # contract reader consults only chunks 0 and 1, so EVM zero padding
        # replaces the byte from chunk 2.
        self.assertEqual(raw[1:5], b"BCDE")
        self.assertEqual(runtime_style_read(chunks, 2, 1, 4), b"BCD\0")

    def test_pointer_table_capacity(self) -> None:
        self.assertEqual(STORE_PAYLOAD_MAX // POINTER_SLOT_BYTES, 767)
        self.assertLessEqual(767 * POINTER_SLOT_BYTES, STORE_PAYLOAD_MAX)
        self.assertGreater(768 * POINTER_SLOT_BYTES, STORE_PAYLOAD_MAX)


class NumericProperties(unittest.TestCase):
    def test_non_saturating_quantization_bound_on_exact_rationals(self) -> None:
        for scale in (1, 2, 3, 10, 1_000, 1_000_000):
            for numerator in range(-401, 402):
                for denominator in (2, 3, 5, 7, 16, 31):
                    value = Fraction(numerator, denominator)
                    quantized = round_half_toward_positive_infinity(value * scale)
                    error = abs(Fraction(quantized, scale) - value)
                    with self.subTest(scale=scale, value=value):
                        self.assertLessEqual(error, Fraction(1, 2 * scale))

        # Production binary64 multiplication can cross an abstract
        # half-integer boundary. The exact-real bound therefore needs the
        # product-rounding term documented in the formal supplement.
        scale = 100_000
        value_float = -15_988.123335
        value_exact = Fraction.from_float(value_float)
        product_float = value_float * scale
        product_exact = value_exact * scale
        production_q = math.floor(product_float + 0.5)
        abstract_q = round_half_toward_positive_infinity(product_exact)
        self.assertNotEqual(production_q, abstract_q)
        production_error = abs(Fraction(production_q, scale) - value_exact)
        self.assertGreater(production_error, Fraction(1, 2 * scale))
        delta = abs(Fraction.from_float(product_float) - product_exact)
        self.assertLessEqual(
            production_error,
            (Fraction(1, 2) + delta) / scale,
        )

    def test_registry_and_store_bounds_imply_exact_javascript_accumulation(self) -> None:
        scalar_bound = (65_535 + 1) * I32_ABS_BOUND
        self.assertEqual(scalar_bound, 1 << 47)
        self.assertLess(scalar_bound, JS_EXACT_BOUND)

        max_core = 767 * STORE_PAYLOAD_MAX
        min_v2_outputs = 2
        min_tree_bytes_at_depth_one = 16
        max_trees_per_output = (
            max_core - (24 + 4 * min_v2_outputs)
        ) // (min_v2_outputs * min_tree_bytes_at_depth_one)
        vector_bound = (max_trees_per_output + 1) * I32_ABS_BOUND
        self.assertLess(max_trees_per_output, 589_000)
        self.assertLess(vector_bound, JS_EXACT_BOUND)


class CheckedInModelWitnesses(unittest.TestCase):
    def test_checked_in_python_cpp_pairs_are_canonical_and_identical(self) -> None:
        pairs = (
            ("py.gl1f", "cpp.gl1f"),
            ("py_bin.gl1f", "cpp_bin.gl1f"),
            ("py_mc.gl1f", "cpp_mc.gl1f"),
        )
        for left_name, right_name in pairs:
            with self.subTest(left=left_name, right=right_name):
                left = parse_path(REPO / left_name)
                right = parse_path(REPO / right_name)
                self.assertEqual(left.core, right.core)
                self.assertEqual(left.header, right.header)
                self.assertEqual(len(left.core), left.header.core_length)


if __name__ == "__main__":
    unittest.main()
