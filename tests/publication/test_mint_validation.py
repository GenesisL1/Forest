#!/usr/bin/env python3
"""Regression tests for the pre-transaction minting validation boundary."""

from __future__ import annotations

import json
import struct
import unittest

from gl1f_validate import (
    FormatError,
    Header,
    MAX_POINTERS,
    Package,
    parse_gl1f_package,
    validate_deployed_registry_profile,
)


def canonical_v1() -> bytes:
    raw = bytearray(40)
    raw[:4] = b"GL1F"
    raw[4] = 1
    struct.pack_into("<HHI", raw, 6, 1, 1, 1)
    struct.pack_into("<iI", raw, 14, 3, 100)
    struct.pack_into("<HiH", raw, 24, 0, 5, 0)
    struct.pack_into("<ii", raw, 32, -7, 11)
    return bytes(raw)


def with_footer(core: bytes, payload: object) -> bytes:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return core + b"GL1X" + b"\x01\0\0\0" + struct.pack("<I", len(encoded)) + encoded


def canonical_v2() -> bytes:
    raw = bytearray(64)
    raw[:4] = b"GL1F"
    raw[4] = 2
    struct.pack_into("<HHI", raw, 6, 1, 1, 1)
    struct.pack_into("<iIH", raw, 14, 0, 100, 2)
    struct.pack_into("<ii", raw, 24, -3, 4)
    for offset in (32, 48):
        struct.pack_into("<HiH", raw, offset, 0, 5, 0)
        struct.pack_into("<ii", raw, offset + 8, -7, 11)
    return bytes(raw)


class MintValidationTests(unittest.TestCase):
    def test_canonical_core_and_footer(self) -> None:
        core = canonical_v1()
        parsed = parse_gl1f_package(core)
        self.assertEqual(parsed.core, core)
        self.assertIsNone(parsed.footer)
        self.assertEqual(parsed.header.registry_n_trees, 1)
        self.assertEqual(parsed.header.registry_base_q, 3)
        self.assertEqual(
            validate_deployed_registry_profile(parsed, chunk_size=24_000),
            1,
        )

        packaged = parse_gl1f_package(with_footer(core, {"kind": "GL1F_PACKAGE"}))
        self.assertEqual(packaged.core, core)
        self.assertEqual(packaged.footer, {"kind": "GL1F_PACKAGE"})
        string_values = {"labels": ["NaN", "Infinity", "-Infinity"]}
        self.assertEqual(
            parse_gl1f_package(with_footer(core, string_values)).footer,
            string_values,
        )

        vector = parse_gl1f_package(canonical_v2())
        self.assertEqual(vector.header.trees_per_output, 1)
        self.assertEqual(vector.header.n_outputs, 2)
        self.assertEqual(vector.header.registry_n_trees, 2)
        self.assertEqual(vector.header.registry_base_q, 0)

    def test_malformed_cores_are_rejected(self) -> None:
        valid = canonical_v1()
        cases: dict[str, bytes] = {
            "truncated": valid[:-1],
            "unframed trailing": valid + b"x",
        }

        header_reserved = bytearray(valid)
        header_reserved[5] = 1
        cases["header reserved"] = bytes(header_reserved)

        zero_features = bytearray(valid)
        struct.pack_into("<H", zero_features, 6, 0)
        cases["zero features"] = bytes(zero_features)

        zero_scale = bytearray(valid)
        struct.pack_into("<I", zero_scale, 18, 0)
        cases["zero scale"] = bytes(zero_scale)

        unsafe_depth = bytearray(valid)
        struct.pack_into("<H", unsafe_depth, 8, 21)
        cases["unsafe depth"] = bytes(unsafe_depth)

        bad_feature = bytearray(valid)
        struct.pack_into("<H", bad_feature, 24, 1)
        cases["feature index"] = bytes(bad_feature)

        node_reserved = bytearray(valid)
        struct.pack_into("<H", node_reserved, 30, 1)
        cases["node reserved"] = bytes(node_reserved)

        for name, raw in cases.items():
            with self.subTest(name=name), self.assertRaises(FormatError):
                parse_gl1f_package(raw)

    def test_malformed_v2_and_footers_are_rejected(self) -> None:
        zero_trees = bytearray(32)
        zero_trees[:4] = b"GL1F"
        zero_trees[4] = 2
        struct.pack_into("<HHI", zero_trees, 6, 1, 1, 0)
        struct.pack_into("<iIH", zero_trees, 14, 0, 100, 2)

        one_output = bytearray(44)
        one_output[:4] = b"GL1F"
        one_output[4] = 2
        struct.pack_into("<HHI", one_output, 6, 1, 1, 1)
        struct.pack_into("<iIH", one_output, 14, 0, 100, 1)

        core = canonical_v1()
        valid_footer = with_footer(core, {"ok": True})
        footer_version = bytearray(valid_footer)
        footer_version[len(core) + 4] = 2
        footer_reserved = bytearray(valid_footer)
        footer_reserved[len(core) + 5] = 1
        footer_length = bytearray(valid_footer)
        struct.pack_into("<I", footer_length, len(core) + 8, 1)
        non_object = with_footer(core, ["not", "an", "object"])

        cases = {
            "v2 zero trees": bytes(zero_trees),
            "v2 one output": bytes(one_output),
            "footer version": bytes(footer_version),
            "footer reserved": bytes(footer_reserved),
            "footer length": bytes(footer_length),
            "footer root": non_object,
        }
        for constant in ("NaN", "Infinity", "-Infinity"):
            cases[f"footer constant {constant}"] = with_footer(
                core, {"nested": [float(constant)]}
            )
        for name, raw in cases.items():
            with self.subTest(name=name), self.assertRaises(FormatError):
                parse_gl1f_package(raw)

    def test_deployed_registry_and_pointer_table_bounds(self) -> None:
        parsed = parse_gl1f_package(canonical_v1())
        with self.assertRaises(FormatError):
            validate_deployed_registry_profile(parsed, chunk_size=3)
        with self.assertRaises(FormatError):
            validate_deployed_registry_profile(parsed, chunk_size=24_573)

        too_many_trees = Package(
            header=Header(
                version=1,
                n_features=1,
                depth=1,
                trees_per_output=65_536,
                n_outputs=1,
                base_q=(0,),
                scale_q=1,
                leaves_per_tree=2,
                internal_per_tree=1,
                bytes_per_tree=16,
                core_length=24 + 65_536 * 16,
            ),
            core=b"x",
            footer=None,
        )
        with self.assertRaises(FormatError):
            validate_deployed_registry_profile(too_many_trees, chunk_size=24_000)

        oversized = Package(
            header=parsed.header,
            core=b"x" * ((MAX_POINTERS * 24_000) + 1),
            footer=None,
        )
        with self.assertRaises(FormatError):
            validate_deployed_registry_profile(oversized, chunk_size=24_000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
