"""Strict, dependency-free GL1F/GL1X parser used by publication tests.

This is deliberately independent of all production decoders.  It therefore
serves as a structural oracle rather than confirming an implementation with
itself.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1
I32_ABS_MAX = 1 << 31
JS_SAFE_INTEGER = (1 << 53) - 1


class FormatError(ValueError):
    """The byte stream violates a GL1F/GL1X structural invariant."""


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
    def worst_case_abs_accumulator(self) -> int:
        return I32_ABS_MAX + self.trees_per_output * I32_ABS_MAX


@dataclass(frozen=True)
class Package:
    header: Header
    core: bytes
    footer: dict[str, Any] | None
    trailing: bytes


def _checked_tree_geometry(depth: int, *, max_depth: int) -> tuple[int, int, int]:
    if not (1 <= depth <= max_depth):
        raise FormatError(f"depth {depth} outside publication profile 1..{max_depth}")
    leaves = 1 << depth
    internal = leaves - 1
    per_tree = internal * 8 + leaves * 4
    return leaves, internal, per_tree


def parse(
    raw: bytes,
    *,
    require_canonical_reserved: bool = True,
    validate_feature_indices: bool = True,
    max_depth: int = 20,
    max_core_bytes: int = (1 << 31) - 1,
) -> Package:
    if len(raw) < 24:
        raise FormatError("shorter than the 24-byte GL1F header")
    if raw[:4] != b"GL1F":
        raise FormatError("bad GL1F magic")

    version = raw[4]
    if version not in (1, 2):
        raise FormatError(f"unsupported GL1F version {version}")
    if require_canonical_reserved and raw[5] != 0:
        raise FormatError("non-zero header reserved byte")

    n_features, depth = struct.unpack_from("<HH", raw, 6)
    if n_features < 1:
        raise FormatError("nFeatures must be positive")
    leaves, internal, per_tree = _checked_tree_geometry(depth, max_depth=max_depth)
    trees_per_output = struct.unpack_from("<I", raw, 10)[0]
    scale_q = struct.unpack_from("<I", raw, 18)[0]
    if scale_q < 1:
        raise FormatError("scaleQ must be positive")

    if version == 1:
        if require_canonical_reserved and raw[22:24] != b"\0\0":
            raise FormatError("non-zero v1 reserved bytes")
        n_outputs = 1
        base_q = (struct.unpack_from("<i", raw, 14)[0],)
        tree_offset = 24
    else:
        if require_canonical_reserved and raw[14:18] != b"\0\0\0\0":
            raise FormatError("non-zero v2 reserved field")
        if trees_per_output < 1:
            raise FormatError("v2 treesPerOutput must be positive")
        n_outputs = struct.unpack_from("<H", raw, 22)[0]
        if n_outputs < 2:
            raise FormatError("v2 nClasses/nOutputs must be at least two")
        tree_offset = 24 + 4 * n_outputs
        if len(raw) < tree_offset:
            raise FormatError("truncated v2 base-logit vector")
        base_q = struct.unpack_from(f"<{n_outputs}i", raw, 24)

    total_trees = trees_per_output * n_outputs
    core_length = tree_offset + total_trees * per_tree
    if core_length > max_core_bytes:
        raise FormatError(f"declared core length {core_length} exceeds safety cap")
    if len(raw) < core_length:
        raise FormatError(f"truncated core: {len(raw)} < {core_length}")

    if validate_feature_indices:
        off = tree_offset
        for tree_idx in range(total_trees):
            for node_idx in range(internal):
                feature = struct.unpack_from("<H", raw, off)[0]
                if feature >= n_features:
                    raise FormatError(
                        f"tree {tree_idx} node {node_idx}: feature {feature} >= nFeatures {n_features}"
                    )
                if require_canonical_reserved and raw[off + 6 : off + 8] != b"\0\0":
                    raise FormatError(f"tree {tree_idx} node {node_idx}: non-zero reserved bytes")
                off += 8
            off += 4 * leaves

    header = Header(
        version=version,
        n_features=n_features,
        depth=depth,
        trees_per_output=trees_per_output,
        n_outputs=n_outputs,
        base_q=tuple(base_q),
        scale_q=scale_q,
        leaves_per_tree=leaves,
        internal_per_tree=internal,
        bytes_per_tree=per_tree,
        core_length=core_length,
    )

    footer: dict[str, Any] | None = None
    trailing = raw[core_length:]
    if trailing:
        if len(trailing) < 12:
            raise FormatError("trailing bytes are too short to be a GL1X footer")
        if trailing[:4] != b"GL1X":
            raise FormatError("trailing bytes do not begin with GL1X")
        if trailing[4] != 1:
            raise FormatError(f"unsupported GL1X version {trailing[4]}")
        if require_canonical_reserved and trailing[5:8] != b"\0\0\0":
            raise FormatError("non-zero GL1X reserved bytes")
        json_len = struct.unpack_from("<I", trailing, 8)[0]
        if len(trailing) != 12 + json_len:
            raise FormatError(
                f"GL1X length mismatch: trailing={len(trailing)}, declared={12 + json_len}"
            )
        try:
            decoded = json.loads(trailing[12:].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatError(f"invalid GL1X JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise FormatError("GL1X JSON root must be an object")
        footer = decoded

    return Package(header=header, core=raw[:core_length], footer=footer, trailing=trailing)


def parse_path(path: str | Path, **kwargs: Any) -> Package:
    return parse(Path(path).read_bytes(), **kwargs)


def quantize_js(value: float, scale_q: int) -> int:
    if not math.isfinite(value):
        raise ValueError("publication profile requires finite feature values")
    rounded = math.floor(value * scale_q + 0.5)
    return min(I32_MAX, max(I32_MIN, rounded))


def predict_q(package: Package, features: Sequence[float]) -> tuple[int, ...]:
    """Reference integer inference for either model version."""
    h = package.header
    if len(features) != h.n_features:
        raise ValueError(f"expected {h.n_features} features, got {len(features)}")
    xq = [quantize_js(float(x), h.scale_q) for x in features]
    tree_offset = 24 if h.version == 1 else 24 + 4 * h.n_outputs
    outputs: list[int] = []

    for output_idx in range(h.n_outputs):
        acc = h.base_q[output_idx]
        class_base = tree_offset + output_idx * h.trees_per_output * h.bytes_per_tree
        for tree_idx in range(h.trees_per_output):
            tree_base = class_base + tree_idx * h.bytes_per_tree
            idx = 0
            for _ in range(h.depth):
                node_off = tree_base + idx * 8
                feature, threshold = struct.unpack_from("<Hi", package.core, node_off)
                idx = 2 * idx + (2 if xq[feature] > threshold else 1)
            leaf_idx = idx - h.internal_per_tree
            leaf_off = tree_base + h.internal_per_tree * 8 + leaf_idx * 4
            acc += struct.unpack_from("<i", package.core, leaf_off)[0]
        outputs.append(acc)
    return tuple(outputs)


def assert_js_safe_accumulation(header: Header) -> None:
    if header.worst_case_abs_accumulator > JS_SAFE_INTEGER:
        raise FormatError(
            "header permits an accumulator outside JavaScript's exact integer range; "
            "prediction parity therefore needs an additional value-level bound"
        )


def mutate_i32(raw: bytes, offset: int, value: int) -> bytes:
    out = bytearray(raw)
    struct.pack_into("<i", out, offset, value)
    return bytes(out)


def mutate_u16(raw: bytes, offset: int, value: int) -> bytes:
    out = bytearray(raw)
    struct.pack_into("<H", out, offset, value)
    return bytes(out)
