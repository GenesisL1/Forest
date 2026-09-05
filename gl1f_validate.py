"""Strict, dependency-free validation for GL1F model packages.

The browser decoder and the EVM runtime remain the authoritative execution
implementations.  This module is the pre-deployment validation boundary used
by ``mint_model.py``: it rejects malformed or non-canonical bytes before any
chunk-write transaction is submitted.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any


MAX_SAFE_DEPTH = 20
MAX_CORE_BYTES = (1 << 31) - 1
MAX_JS_SAFE_INTEGER = (1 << 53) - 1
MAX_STORE_PAYLOAD_BYTES = 24_572
POINTER_WORD_BYTES = 32
MAX_POINTERS = MAX_STORE_PAYLOAD_BYTES // POINTER_WORD_BYTES


class FormatError(ValueError):
    """The byte stream violates a canonical GL1F/GL1X invariant."""


@dataclass(frozen=True)
class Header:
    version: int
    n_features: int
    depth: int
    trees_per_output: int
    n_outputs: int
    base_q: tuple[int, ...]
    scale_q: int
    leaves_per_tree: int
    internal_per_tree: int
    bytes_per_tree: int
    core_length: int

    @property
    def total_trees(self) -> int:
        return self.trees_per_output * self.n_outputs

    @property
    def registry_n_trees(self) -> int:
        # The deployed UI records total trees in the registry: T for v1 and
        # K*R for v2, while the v2 wire header stores only R at offset 10.
        return self.total_trees

    @property
    def registry_base_q(self) -> int:
        return self.base_q[0] if self.version == 1 else 0


@dataclass(frozen=True)
class Package:
    header: Header
    core: bytes
    footer: dict[str, Any] | None


def _tree_geometry(depth: int) -> tuple[int, int, int]:
    if not 1 <= depth <= MAX_SAFE_DEPTH:
        raise FormatError(
            f"depth {depth} is outside the supported range 1..{MAX_SAFE_DEPTH}"
        )
    leaves = 1 << depth
    internal = leaves - 1
    per_tree = internal * 8 + leaves * 4
    return leaves, internal, per_tree


def parse_gl1f_package(raw: bytes) -> Package:
    """Parse and validate one canonical GL1F core with an optional GL1X footer."""
    if len(raw) < 24:
        raise FormatError("shorter than the 24-byte GL1F header")
    if raw[:4] != b"GL1F":
        raise FormatError("bad GL1F magic")

    version = raw[4]
    if version not in (1, 2):
        raise FormatError(f"unsupported GL1F version {version}")
    if raw[5] != 0:
        raise FormatError("non-zero GL1F header reserved byte")

    n_features, depth = struct.unpack_from("<HH", raw, 6)
    if n_features < 1:
        raise FormatError("nFeatures must be positive")
    leaves, internal, per_tree = _tree_geometry(depth)

    trees_per_output = struct.unpack_from("<I", raw, 10)[0]
    scale_q = struct.unpack_from("<I", raw, 18)[0]
    if scale_q < 1:
        raise FormatError("scaleQ must be positive")

    if version == 1:
        if raw[22:24] != b"\0\0":
            raise FormatError("non-zero GL1F v1 reserved field")
        n_outputs = 1
        base_q = (struct.unpack_from("<i", raw, 14)[0],)
        tree_offset = 24
    else:
        if raw[14:18] != b"\0\0\0\0":
            raise FormatError("non-zero GL1F v2 reserved field")
        if trees_per_output < 1:
            raise FormatError("v2 treesPerOutput must be positive")
        n_outputs = struct.unpack_from("<H", raw, 22)[0]
        if n_outputs < 2:
            raise FormatError("v2 nClasses/nOutputs must be at least two")
        tree_offset = 24 + 4 * n_outputs
        if len(raw) < tree_offset:
            raise FormatError("truncated v2 base-logit vector")
        base_q = tuple(struct.unpack_from(f"<{n_outputs}i", raw, 24))

    total_trees = trees_per_output * n_outputs
    core_length = tree_offset + total_trees * per_tree
    if core_length > MAX_CORE_BYTES:
        raise FormatError(
            f"declared core length {core_length} exceeds {MAX_CORE_BYTES}"
        )
    if len(raw) < core_length:
        raise FormatError(f"truncated GL1F core ({len(raw)} < {core_length})")

    accumulator_bounds = [abs(value) for value in base_q]
    offset = tree_offset
    for tree_index in range(total_trees):
        for node_index in range(internal):
            feature = struct.unpack_from("<H", raw, offset)[0]
            if feature >= n_features:
                raise FormatError(
                    f"tree {tree_index} node {node_index}: "
                    f"feature {feature} >= nFeatures {n_features}"
                )
            if raw[offset + 6 : offset + 8] != b"\0\0":
                raise FormatError(
                    f"tree {tree_index} node {node_index}: non-zero reserved bytes"
                )
            offset += 8

        max_abs_leaf = 0
        for _ in range(leaves):
            max_abs_leaf = max(
                max_abs_leaf,
                abs(struct.unpack_from("<i", raw, offset)[0]),
            )
            offset += 4

        output_index = (
            0 if version == 1 else tree_index // trees_per_output
        )
        accumulator_bounds[output_index] += max_abs_leaf
        if accumulator_bounds[output_index] > MAX_JS_SAFE_INTEGER:
            raise FormatError(
                f"output {output_index} can exceed JavaScript's exact integer range"
            )

    if offset != core_length:
        raise FormatError("internal GL1F length mismatch")

    footer: dict[str, Any] | None = None
    trailing = raw[core_length:]
    if trailing:
        if len(trailing) < 12:
            raise FormatError("trailing bytes are too short to be a GL1X footer")
        if trailing[:4] != b"GL1X":
            raise FormatError("trailing bytes do not begin with GL1X")
        if trailing[4] != 1:
            raise FormatError(f"unsupported GL1X version {trailing[4]}")
        if trailing[5:8] != b"\0\0\0":
            raise FormatError("non-zero GL1X reserved bytes")
        json_length = struct.unpack_from("<I", trailing, 8)[0]
        if len(trailing) != 12 + json_length:
            raise FormatError(
                "GL1X length mismatch "
                f"({len(trailing)} != {12 + json_length})"
            )
        try:
            decoded = json.loads(trailing[12:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatError(f"invalid GL1X JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise FormatError("GL1X JSON root must be an object")
        footer = decoded

    return Package(
        header=Header(
            version=version,
            n_features=n_features,
            depth=depth,
            trees_per_output=trees_per_output,
            n_outputs=n_outputs,
            base_q=base_q,
            scale_q=scale_q,
            leaves_per_tree=leaves,
            internal_per_tree=internal,
            bytes_per_tree=per_tree,
            core_length=core_length,
        ),
        core=raw[:core_length],
        footer=footer,
    )


def validate_deployed_registry_profile(
    package: Package,
    *,
    chunk_size: int,
) -> int:
    """Validate fields that must fit the currently deployed registry/store."""
    if not 4 <= chunk_size <= MAX_STORE_PAYLOAD_BYTES:
        raise FormatError(
            f"chunkSize {chunk_size} is outside 4..{MAX_STORE_PAYLOAD_BYTES}"
        )

    header = package.header
    if header.registry_n_trees > 0xFFFF:
        raise FormatError(
            "total tree count cannot be encoded by the deployed registry's uint16 nTrees"
        )

    num_chunks = (len(package.core) + chunk_size - 1) // chunk_size
    if not 1 <= num_chunks <= MAX_POINTERS:
        raise FormatError(
            f"model needs {num_chunks} chunks; the deployed pointer table supports "
            f"1..{MAX_POINTERS}"
        )
    return num_chunks
