#!/usr/bin/env python3
"""Offline, independent verification of a self-contained GL1F chain archive.

This module intentionally depends only on the Python standard library.  It
does not import the GL1F trainers, browser decoder, contract test harness, or
JavaScript witness generator.  It treats the archive as untrusted input and
checks the binary format, storage reconstruction, commitments, metadata
relations, and every archived integer inference result.

It does not verify a block-header chain, consensus finality, or the provider's
account-proof paths.  Those are outside this offline artifact-to-replay check.

The default input is the versioned live-chain archive produced by
``benchmarks/archive_live_chain_state.mjs``.  A different manifest can be
selected with ``--manifest``.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import io
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "results"
    / "live_chain_archive_v3"
    / "manifest.json"
)

GL1F_MAGIC = b"GL1F"
GL1C_MAGIC = b"GL1C"
MAX_ARCHIVE_DEPTH = 12
MAX_CORE_BYTES = (1 << 31) - 1
MAX_CHUNK_PAYLOAD = 24_572
MIN_ARCHIVE_CHUNK_SIZE = 4
MAX_TABLE_POINTERS = MAX_CHUNK_PAYLOAD // 32
EVM_ADDRESS_BYTES = 20
ADDRESS_SLOT_BYTES = 32
MAX_TRANSCRIPT_BYTES = 512 * 1024 * 1024

_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_MASK64 = (1 << 64) - 1


class VerificationError(ValueError):
    """An archive or model violates a checked invariant."""


@dataclass(frozen=True)
class CoreHeader:
    """The inference-relevant fields derived from one strict GL1F core."""

    version: int
    n_features: int
    depth: int
    trees_per_output: int
    n_outputs: int
    base_q: tuple[int, ...]
    scale_q: int
    internal_nodes: int
    leaves: int
    bytes_per_tree: int
    trees_offset: int
    core_bytes: int

    @property
    def total_trees(self) -> int:
        return self.trees_per_output * self.n_outputs


@dataclass(frozen=True)
class ParsedCore:
    header: CoreHeader
    data: bytes


@dataclass(frozen=True)
class VerificationReport:
    manifest: str
    archive_id: str
    files_checked: int
    models_checked: int
    vectors_checked: int
    core_bytes_checked: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "archiveId": self.archive_id,
            "filesChecked": self.files_checked,
            "modelsChecked": self.models_checked,
            "vectorsChecked": self.vectors_checked,
            "coreBytesChecked": self.core_bytes_checked,
            "status": "verified",
        }


@dataclass(frozen=True)
class RpcObservation:
    entry_sequence: int
    request_id: int
    response_id: int
    batch_index: int
    method: str
    params: Any
    result: Any


# Rotation offsets and round constants from the Keccak-f[1600]
# specification.  Lanes are indexed as x + 5*y and serialized little-endian.
_KECCAK_ROTATION = (
    0,
    1,
    62,
    28,
    27,
    36,
    44,
    6,
    55,
    20,
    3,
    10,
    43,
    25,
    39,
    41,
    45,
    15,
    21,
    8,
    18,
    2,
    61,
    56,
    14,
)
_KECCAK_ROUND_CONSTANTS = (
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
)


def _rotl64(value: int, amount: int) -> int:
    if amount == 0:
        return value & _MASK64
    return ((value << amount) | (value >> (64 - amount))) & _MASK64


def _keccak_f1600(state: list[int]) -> None:
    """Apply Keccak-f[1600] in place.

    This compact implementation follows the theta, rho/pi, chi, and iota
    steps directly.  It is deliberately local rather than delegated to an
    Ethereum or GL1F dependency.
    """

    for round_constant in _KECCAK_ROUND_CONSTANTS:
        columns = [
            state[x]
            ^ state[x + 5]
            ^ state[x + 10]
            ^ state[x + 15]
            ^ state[x + 20]
            for x in range(5)
        ]
        deltas = [
            columns[(x - 1) % 5] ^ _rotl64(columns[(x + 1) % 5], 1)
            for x in range(5)
        ]
        for y in range(5):
            row = 5 * y
            for x in range(5):
                state[row + x] ^= deltas[x]

        rotated = [0] * 25
        for y in range(5):
            for x in range(5):
                destination_x = y
                destination_y = (2 * x + 3 * y) % 5
                rotated[destination_x + 5 * destination_y] = _rotl64(
                    state[x + 5 * y], _KECCAK_ROTATION[x + 5 * y]
                )

        for y in range(5):
            row = 5 * y
            for x in range(5):
                state[row + x] = (
                    rotated[row + x]
                    ^ ((~rotated[row + ((x + 1) % 5)])
                       & rotated[row + ((x + 2) % 5)])
                ) & _MASK64

        state[0] ^= round_constant


@functools.lru_cache(maxsize=64)
def keccak256(data: bytes) -> bytes:
    """Return legacy Keccak-256 (Ethereum), not standardized SHA3-256."""

    rate = 136
    padded = bytearray(data)
    padding_bytes = rate - (len(data) % rate)
    if padding_bytes == 1:
        # Both delimited-suffix bits occupy the final byte of this block.
        padded.append(0x81)
    else:
        padded.append(0x01)
        padded.extend(b"\0" * (padding_bytes - 2))
        padded.append(0x80)

    state = [0] * 25
    for block_start in range(0, len(padded), rate):
        block = padded[block_start : block_start + rate]
        for lane in range(rate // 8):
            offset = lane * 8
            state[lane] ^= int.from_bytes(block[offset : offset + 8], "little")
        _keccak_f1600(state)

    return b"".join(lane.to_bytes(8, "little") for lane in state)[:32]


def _fail(location: str, message: str) -> None:
    raise VerificationError(f"{location}: {message}")


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(location, "expected an object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(location, "expected an array")
    return value


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        _fail(location, f"missing required field {key!r}")
    return mapping[key]


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool):
        _fail(location, "Boolean is not an integer value")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            if text.startswith(("-0x", "0x")):
                sign = -1 if text.startswith("-") else 1
                digits = text[3:] if sign < 0 else text[2:]
                if not digits:
                    raise ValueError
                return sign * int(digits, 16)
            return int(text, 10)
        except ValueError:
            pass
    _fail(location, f"expected an exact integer or integer string, got {value!r}")


def _hex_bytes(value: Any, location: str, *, exact_bytes: int | None = None) -> bytes:
    if not isinstance(value, str) or not value.startswith("0x"):
        _fail(location, "expected a 0x-prefixed hexadecimal string")
    digits = value[2:]
    if len(digits) % 2:
        _fail(location, "hexadecimal string has an odd number of digits")
    try:
        decoded = bytes.fromhex(digits)
    except ValueError:
        _fail(location, "invalid hexadecimal string")
    if exact_bytes is not None and len(decoded) != exact_bytes:
        _fail(location, f"expected {exact_bytes} bytes, got {len(decoded)}")
    return decoded


def _address(value: Any, location: str) -> bytes:
    return _hex_bytes(value, location, exact_bytes=EVM_ADDRESS_BYTES)


def _digest_hex(value: Any, location: str) -> str:
    return "0x" + _hex_bytes(value, location, exact_bytes=32).hex()


def _relative_file(root: Path, value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(location, "expected a non-empty relative file path")
    relative = Path(value)
    if relative.is_absolute():
        _fail(location, "absolute paths are forbidden")
    root_resolved = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        _fail(location, "path escapes the archive directory")
    if not resolved.is_file():
        _fail(location, f"file does not exist: {value}")
    return resolved


def _file_reference(value: Any, location: str) -> str:
    if isinstance(value, str):
        return value
    obj = _require_mapping(value, location)
    for key in ("file", "artifactFile", "coreFile", "path"):
        candidate = obj.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    for nested_name in ("runtimeBytecode", "runtime"):
        runtime = obj.get(nested_name)
        if runtime is not None:
            return _file_reference(runtime, f"{location}.{nested_name}")
    _fail(location, "missing a file/path reference")


def _decode_artifact_file(path: Path, location: str) -> bytes:
    """Read either raw bytes or a strict single 0x-hex text artifact."""

    raw = path.read_bytes()
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return raw
    if not text.startswith("0x"):
        return raw
    return _hex_bytes(text, location)


def _declared_artifact_checks(
    descriptor: Any, data: bytes, location: str
) -> None:
    if not isinstance(descriptor, Mapping):
        return
    obj: Mapping[str, Any] = descriptor
    for nested_name in ("runtimeBytecode", "runtime"):
        runtime = obj.get(nested_name)
        if isinstance(runtime, Mapping):
            obj = runtime
            location = f"{location}.{nested_name}"
            break
    if "size" in obj:
        expected_size = _integer(obj["size"], f"{location}.size")
        if expected_size != len(data):
            _fail(location, f"declared size {expected_size} != actual {len(data)}")
    if "sha256" in obj:
        expected_sha = _digest_hex_or_plain(obj["sha256"], f"{location}.sha256")
        actual_sha = hashlib.sha256(data).hexdigest()
        if expected_sha != actual_sha:
            _fail(location, f"SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    if "keccak256" in obj:
        expected_keccak = _digest_hex(obj["keccak256"], f"{location}.keccak256")
        actual_keccak = "0x" + keccak256(data).hex()
        if expected_keccak != actual_keccak:
            _fail(
                location,
                f"Keccak-256 mismatch: expected {expected_keccak}, got {actual_keccak}",
            )


def _digest_hex_or_plain(value: Any, location: str) -> str:
    if not isinstance(value, str):
        _fail(location, "expected a hexadecimal digest string")
    text = value[2:] if value.startswith("0x") else value
    if len(text) != 64:
        _fail(location, f"expected 64 hexadecimal digits, got {len(text)}")
    try:
        bytes.fromhex(text)
    except ValueError:
        _fail(location, "invalid hexadecimal digest")
    return text.lower()


def parse_core(data: bytes, *, location: str = "core") -> ParsedCore:
    """Strictly parse a bare first-party GL1F v1 or v2 core."""

    if len(data) < 24:
        _fail(location, f"truncated fixed header ({len(data)} < 24 bytes)")
    if data[:4] != GL1F_MAGIC:
        _fail(location, "bad magic (expected GL1F)")

    version = data[4]
    if version not in (1, 2):
        _fail(location, f"unsupported version {version}")
    if data[5] != 0:
        _fail(location, "header reserved byte at offset 5 is non-zero")

    n_features = _U16.unpack_from(data, 6)[0]
    depth = _U16.unpack_from(data, 8)[0]
    trees_per_output = _U32.unpack_from(data, 10)[0]
    scale_q = _U32.unpack_from(data, 18)[0]

    if n_features == 0:
        _fail(location, "nFeatures must be positive")
    if not 1 <= depth <= MAX_ARCHIVE_DEPTH:
        _fail(
            location,
            f"depth {depth} is outside the archived first-party profile 1..{MAX_ARCHIVE_DEPTH}",
        )
    if scale_q == 0:
        _fail(location, "scaleQ must be positive")

    leaves = 1 << depth
    internal_nodes = leaves - 1
    bytes_per_tree = 8 * internal_nodes + 4 * leaves

    if version == 1:
        if data[22:24] != b"\0\0":
            _fail(location, "v1 reserved bytes at offsets 22..23 are non-zero")
        n_outputs = 1
        base_q = (_I32.unpack_from(data, 14)[0],)
        trees_offset = 24
    else:
        if data[14:18] != b"\0\0\0\0":
            _fail(location, "v2 reserved field at offsets 14..17 is non-zero")
        if trees_per_output == 0:
            _fail(location, "v2 treesPerOutput must be positive")
        n_outputs = _U16.unpack_from(data, 22)[0]
        if n_outputs < 2:
            _fail(location, "v2 nOutputs must be at least two")
        trees_offset = 24 + 4 * n_outputs
        if len(data) < trees_offset:
            _fail(location, "truncated v2 base vector")
        base_q = tuple(
            _I32.unpack_from(data, 24 + 4 * output)[0]
            for output in range(n_outputs)
        )

    total_trees = trees_per_output * n_outputs
    expected_length = trees_offset + total_trees * bytes_per_tree
    if expected_length > MAX_CORE_BYTES:
        _fail(
            location,
            f"declared core length {expected_length} exceeds {MAX_CORE_BYTES}-byte cap",
        )
    if len(data) != expected_length:
        relation = "truncated" if len(data) < expected_length else "has trailing bytes"
        _fail(
            location,
            f"{relation}: actual {len(data)} bytes, expected exactly {expected_length}",
        )

    for tree_index in range(total_trees):
        tree_start = trees_offset + tree_index * bytes_per_tree
        for node_index in range(internal_nodes):
            node_start = tree_start + node_index * 8
            feature_index = _U16.unpack_from(data, node_start)[0]
            if feature_index >= n_features:
                _fail(
                    location,
                    f"tree {tree_index}, node {node_index}: feature index "
                    f"{feature_index} >= nFeatures {n_features}",
                )
            if data[node_start + 6 : node_start + 8] != b"\0\0":
                _fail(
                    location,
                    f"tree {tree_index}, node {node_index}: reserved bytes are non-zero",
                )

    header = CoreHeader(
        version=version,
        n_features=n_features,
        depth=depth,
        trees_per_output=trees_per_output,
        n_outputs=n_outputs,
        base_q=base_q,
        scale_q=scale_q,
        internal_nodes=internal_nodes,
        leaves=leaves,
        bytes_per_tree=bytes_per_tree,
        trees_offset=trees_offset,
        core_bytes=expected_length,
    )
    return ParsedCore(header=header, data=data)


def decode_packed_i32(
    packed: bytes, n_features: int, *, location: str = "packedFeaturesQ"
) -> tuple[int, ...]:
    expected = 4 * n_features
    if len(packed) != expected:
        _fail(location, f"expected exactly {expected} bytes, got {len(packed)}")
    return tuple(_I32.unpack_from(packed, 4 * index)[0] for index in range(n_features))


def evaluate_packed(
    parsed: ParsedCore, packed: bytes, *, location: str = "packedFeaturesQ"
) -> tuple[int, ...]:
    """Evaluate exact integer logits/scores using unbounded Python integers."""

    header = parsed.header
    features_q = decode_packed_i32(packed, header.n_features, location=location)
    outputs: list[int] = []

    for output_index in range(header.n_outputs):
        accumulator = int(header.base_q[output_index])
        output_start = (
            header.trees_offset
            + output_index
            * header.trees_per_output
            * header.bytes_per_tree
        )
        for tree_index in range(header.trees_per_output):
            tree_start = output_start + tree_index * header.bytes_per_tree
            heap_index = 0
            for _level in range(header.depth):
                node_start = tree_start + heap_index * 8
                feature_index = _U16.unpack_from(parsed.data, node_start)[0]
                threshold_q = _I32.unpack_from(parsed.data, node_start + 2)[0]
                if features_q[feature_index] > threshold_q:
                    heap_index = 2 * heap_index + 2
                else:
                    heap_index = 2 * heap_index + 1
            leaf_index = heap_index - header.internal_nodes
            leaf_start = (
                tree_start + 8 * header.internal_nodes + 4 * leaf_index
            )
            accumulator += int(_I32.unpack_from(parsed.data, leaf_start)[0])
        outputs.append(accumulator)

    return tuple(outputs)


def _header_field(
    header_manifest: Mapping[str, Any],
    names: Sequence[str],
    location: str,
) -> Any:
    present = [name for name in names if name in header_manifest]
    if not present:
        _fail(location, f"missing required field (one of {', '.join(names)})")
    first = header_manifest[present[0]]
    for alias in present[1:]:
        if header_manifest[alias] != first:
            _fail(location, f"conflicting aliases {present[0]!r} and {alias!r}")
    return first


def _compare_header_manifest(
    manifest_header: Any, parsed: ParsedCore, location: str
) -> None:
    obj = _require_mapping(manifest_header, location)
    header = parsed.header
    integer_relations = (
        (("version",), header.version),
        (("nFeatures", "n_features"), header.n_features),
        (("depth",), header.depth),
        (("totalTrees", "nTrees"), header.total_trees),
        (("treesPerOutput",), header.trees_per_output),
        (("outputs", "nOutputs", "nClasses"), header.n_outputs),
        (("scaleQ",), header.scale_q),
        (("coreBytes", "coreSize"), header.core_bytes),
    )
    for aliases, expected in integer_relations:
        field_name = aliases[0]
        actual = _integer(
            _header_field(obj, aliases, location), f"{location}.{field_name}"
        )
        if actual != expected:
            _fail(
                f"{location}.{field_name}",
                f"manifest value {actual} != decoded value {expected}",
            )

    raw_base = _header_field(obj, ("baseQ", "base_q"), location)
    if isinstance(raw_base, list):
        actual_base = tuple(
            _integer(value, f"{location}.baseQ[{index}]")
            for index, value in enumerate(raw_base)
        )
    else:
        actual_base = (_integer(raw_base, f"{location}.baseQ"),)
    if actual_base != header.base_q:
        _fail(
            f"{location}.baseQ",
            f"manifest value {actual_base} != decoded value {header.base_q}",
        )


def _registry_object(model: Mapping[str, Any], location: str) -> Mapping[str, Any]:
    registry = model.get("registry")
    if registry is None:
        _fail(
            location,
            "missing raw registry object; precomputed agreement booleans are insufficient",
        )
    return _require_mapping(registry, f"{location}.registry")


def _compare_registry(
    registry: Mapping[str, Any],
    parsed: ParsedCore,
    model_id: str,
    table_address: bytes,
    chunk_size: int,
    num_chunks: int,
    total_bytes: int,
    location: str,
) -> None:
    header = parsed.header
    relations: tuple[tuple[tuple[str, ...], int], ...] = (
        (("nFeatures",), header.n_features),
        (("nTrees", "totalTrees"), header.total_trees),
        (("depth",), header.depth),
        (("baseQ",), header.base_q[0] if header.version == 1 else 0),
        (("scaleQ",), header.scale_q),
        (("chunkSize",), chunk_size),
        (("numChunks",), num_chunks),
        (("totalBytes",), total_bytes),
    )
    for aliases, expected in relations:
        actual = _integer(
            _header_field(registry, aliases, location),
            f"{location}.{aliases[0]}",
        )
        if actual != expected:
            _fail(
                f"{location}.{aliases[0]}",
                f"registry value {actual} != independently derived value {expected}",
            )

    registry_model_id = _digest_hex(
        _header_field(registry, ("modelId",), location), f"{location}.modelId"
    )
    if registry_model_id != model_id:
        _fail(
            f"{location}.modelId",
            f"registry value {registry_model_id} != content commitment {model_id}",
        )
    registry_table = _address(
        _header_field(registry, ("tablePtr", "tableAddress"), location),
        f"{location}.tablePtr",
    )
    if registry_table != table_address:
        _fail(f"{location}.tablePtr", "registry pointer != archived table address")


def _artifact_descriptor(model: Mapping[str, Any], kind: str, location: str) -> Any:
    if kind == "core":
        if "core" in model:
            return model["core"]
        if "coreFile" in model:
            return model["coreFile"]
    _fail(location, f"missing {kind} artifact descriptor")


def _table_descriptor(model: Mapping[str, Any], location: str) -> Any:
    if "table" in model:
        return model["table"]
    tables = model.get("tables")
    if tables is not None:
        array = _require_list(tables, f"{location}.tables")
        if len(array) != 1:
            _fail(f"{location}.tables", f"expected exactly one table, got {len(array)}")
        return array[0]
    storage = model.get("storage")
    if isinstance(storage, Mapping) and "table" in storage:
        return storage["table"]
    _fail(location, "missing table artifact descriptor")


def _chunk_descriptors(model: Mapping[str, Any], location: str) -> list[Any]:
    chunks = model.get("chunks")
    if chunks is None:
        storage = model.get("storage")
        if isinstance(storage, Mapping):
            chunks = storage.get("chunks")
    return _require_list(chunks, f"{location}.chunks")


def _descriptor_address(descriptor: Any, location: str) -> bytes:
    obj = _require_mapping(descriptor, location)
    for key in ("address", "pointer", "tablePtr"):
        if key in obj:
            decoded = _address(obj[key], f"{location}.{key}")
            if decoded == b"\0" * EVM_ADDRESS_BYTES:
                _fail(f"{location}.{key}", "zero address is forbidden")
            return decoded
    _fail(location, "missing EVM address")


def _runtime_bytes(
    archive_root: Path, descriptor: Any, location: str
) -> tuple[bytes, Path, int, int]:
    obj = _require_mapping(descriptor, location)
    range_obj: Mapping[str, Any] = obj
    for nested_name in ("runtimeBytecode", "runtime"):
        nested = obj.get(nested_name)
        if isinstance(nested, Mapping):
            range_obj = nested
            break
    file_ref = _file_reference(descriptor, location)
    path = _relative_file(archive_root, file_ref, f"{location}.file")
    container = _decode_artifact_file(path, f"{location}.file")
    offset = (
        _integer(range_obj["offset"], f"{location}.offset")
        if "offset" in range_obj
        else 0
    )
    if offset < 0 or offset > len(container):
        _fail(location, f"runtime offset {offset} is outside a {len(container)}-byte file")
    if "offset" in range_obj:
        if "size" not in range_obj:
            _fail(location, "a ranged runtime descriptor requires size")
        size = _integer(range_obj["size"], f"{location}.size")
        if size < 0 or offset + size > len(container):
            _fail(
                location,
                f"runtime range [{offset},{offset + size}) is outside "
                f"a {len(container)}-byte file",
            )
        data = container[offset : offset + size]
    else:
        data = container
    _declared_artifact_checks(descriptor, data, location)
    return data, path, offset, len(container)


def _reconstruct_storage(
    archive_root: Path,
    model: Mapping[str, Any],
    location: str,
) -> tuple[bytes, bytes, int, int, int, dict[bytes, str]]:
    chunk_size = _integer(
        _required(model, "chunkSize", location), f"{location}.chunkSize"
    )
    num_chunks = _integer(
        _required(model, "numChunks", location), f"{location}.numChunks"
    )
    total_bytes = _integer(
        _required(model, "totalBytes", location), f"{location}.totalBytes"
    )
    if not MIN_ARCHIVE_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_PAYLOAD:
        _fail(
            f"{location}.chunkSize",
            f"{chunk_size} is outside the documented storage profile "
            f"{MIN_ARCHIVE_CHUNK_SIZE}..{MAX_CHUNK_PAYLOAD}",
        )
    if not 1 <= num_chunks <= MAX_TABLE_POINTERS:
        _fail(
            f"{location}.numChunks",
            f"{num_chunks} is outside the one-level pointer-table profile "
            f"1..{MAX_TABLE_POINTERS}",
        )
    if total_bytes <= 0:
        _fail(f"{location}.totalBytes", "must be positive")
    derived_chunks = (total_bytes + chunk_size - 1) // chunk_size
    if num_chunks != derived_chunks:
        _fail(
            f"{location}.numChunks",
            f"registry value {num_chunks} != ceil(totalBytes/chunkSize) {derived_chunks}",
        )

    chunks = _chunk_descriptors(model, location)
    if len(chunks) != num_chunks:
        _fail(
            f"{location}.chunks",
            f"contains {len(chunks)} objects, expected numChunks={num_chunks}",
        )

    chunk_addresses: list[bytes] = []
    payloads: list[bytes] = []
    chunk_ranges: list[tuple[Path, int, int, int]] = []
    code_hashes: dict[bytes, str] = {}
    for index, descriptor in enumerate(chunks):
        chunk_location = f"{location}.chunks[{index}]"
        obj = _require_mapping(descriptor, chunk_location)
        if "index" in obj:
            declared_index = _integer(obj["index"], f"{chunk_location}.index")
            if declared_index != index:
                _fail(
                    f"{chunk_location}.index",
                    f"declared {declared_index}, expected ordered index {index}",
                )
        chunk_address = _descriptor_address(descriptor, chunk_location)
        chunk_addresses.append(chunk_address)
        runtime, path, offset, container_size = _runtime_bytes(
            archive_root, descriptor, chunk_location
        )
        if not runtime.startswith(GL1C_MAGIC):
            _fail(chunk_location, "runtime bytecode does not begin with GL1C")
        code_hashes[chunk_address] = "0x" + keccak256(runtime).hex()
        payload = runtime[4:]
        expected_payload = (
            chunk_size
            if index < num_chunks - 1
            else total_bytes - chunk_size * (num_chunks - 1)
        )
        if len(payload) != expected_payload:
            _fail(
                chunk_location,
                f"payload has {len(payload)} bytes, expected {expected_payload}",
            )
        payloads.append(payload)
        if "payloadSize" in obj:
            declared_payload = _integer(
                obj["payloadSize"], f"{chunk_location}.payloadSize"
            )
            if declared_payload != len(payload):
                _fail(
                    f"{chunk_location}.payloadSize",
                    f"declared {declared_payload} != actual {len(payload)}",
                )
        chunk_ranges.append((path, offset, len(runtime), container_size))
    table_descriptor = _table_descriptor(model, location)
    table_address = _descriptor_address(table_descriptor, f"{location}.table")
    table_runtime, table_path, table_offset, table_container_size = _runtime_bytes(
        archive_root, table_descriptor, f"{location}.table"
    )
    if not table_runtime.startswith(GL1C_MAGIC):
        _fail(f"{location}.table", "runtime bytecode does not begin with GL1C")
    code_hashes[table_address] = "0x" + keccak256(table_runtime).hex()
    pointer_payload = table_runtime[4:]
    expected_table_bytes = num_chunks * ADDRESS_SLOT_BYTES
    if len(pointer_payload) != expected_table_bytes:
        _fail(
            f"{location}.table",
            f"pointer payload has {len(pointer_payload)} bytes, "
            f"expected exactly {expected_table_bytes}",
        )
    decoded_pointers: list[bytes] = []
    for index in range(num_chunks):
        slot = pointer_payload[
            index * ADDRESS_SLOT_BYTES : (index + 1) * ADDRESS_SLOT_BYTES
        ]
        if slot[:12] != b"\0" * 12:
            _fail(
                f"{location}.table.pointer[{index}]",
                "high 12 address-slot bytes are non-zero",
            )
        decoded_pointers.append(slot[12:])
    if decoded_pointers != chunk_addresses:
        _fail(f"{location}.table", "ordered table pointers != chunk addresses")
    table_obj = _require_mapping(table_descriptor, f"{location}.table")
    if "pointerCount" in table_obj:
        pointer_count = _integer(
            table_obj["pointerCount"], f"{location}.table.pointerCount"
        )
        if pointer_count != num_chunks:
            _fail(
                f"{location}.table.pointerCount",
                f"declared {pointer_count} != numChunks {num_chunks}",
            )

    # The v3 archive stores one table range followed by all ordered chunk
    # ranges in a single exact byte container.  Validate that stronger shape
    # whenever ranged descriptors are present.
    if model.get("storageRuntimeFile") is not None:
        expected_container = model.get("storageRuntimeFile")
        if expected_container is not None:
            expected_path = _relative_file(
                archive_root,
                expected_container,
                f"{location}.storageRuntimeFile",
            )
            if table_path != expected_path:
                _fail(f"{location}.table", "table range is outside storageRuntimeFile")
        if table_offset != 0:
            _fail(
                f"{location}.table",
                f"table runtime must begin at offset 0, got {table_offset}",
            )
        next_offset = len(table_runtime)
        for index, (path, offset, size, container_size) in enumerate(chunk_ranges):
            if path != table_path:
                _fail(
                    f"{location}.chunks[{index}]",
                    "chunk range and table range use different container files",
                )
            if offset != next_offset:
                _fail(
                    f"{location}.chunks[{index}]",
                    f"non-contiguous runtime offset {offset}; expected {next_offset}",
                )
            if container_size != table_container_size:
                _fail(f"{location}.chunks[{index}]", "inconsistent container size")
            next_offset += size
        if next_offset != table_container_size:
            _fail(
                location,
                f"storage runtime ranges cover {next_offset} bytes, "
                f"container has {table_container_size}",
            )

    for pointer_field in ("chunkPointers", "orderedChunkPointers"):
        declared_pointers = model.get(pointer_field)
        if declared_pointers is None:
            continue
        pointer_list = _require_list(
            declared_pointers, f"{location}.{pointer_field}"
        )
        normalized = [
            _address(value, f"{location}.{pointer_field}[{index}]")
            for index, value in enumerate(pointer_list)
        ]
        if normalized != chunk_addresses:
            _fail(
                f"{location}.{pointer_field}",
                "manifest pointer order != archived chunk order",
            )

    reconstructed = b"".join(payloads)
    if len(reconstructed) != total_bytes:
        _fail(location, "internal error: reconstructed byte length mismatch")
    return (
        reconstructed,
        table_address,
        chunk_size,
        num_chunks,
        total_bytes,
        code_hashes,
    )


def _output_vector(value: Any, location: str) -> tuple[int, ...]:
    if isinstance(value, list):
        return tuple(
            _integer(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        )
    return (_integer(value, location),)


def _verify_vectors(
    model: Mapping[str, Any],
    parsed: ParsedCore,
    location: str,
) -> int:
    vectors_value = model.get("vectors")
    if vectors_value is None:
        corpus = model.get("conformanceStudy")
        if isinstance(corpus, Mapping):
            vectors_value = corpus.get("vectors")
    vectors = _require_list(vectors_value, f"{location}.vectors")
    if not vectors:
        _fail(f"{location}.vectors", "at least one replay vector is required")

    for index, raw_vector in enumerate(vectors):
        vector_location = f"{location}.vectors[{index}]"
        vector = _require_mapping(raw_vector, vector_location)
        packed_field = (
            "packedFeaturesQHex"
            if "packedFeaturesQHex" in vector
            else "packedFeaturesHex"
        )
        packed = _hex_bytes(
            _required(vector, packed_field, vector_location),
            f"{vector_location}.{packed_field}",
        )
        if "packedBytes" in vector:
            declared_packed_bytes = _integer(
                vector["packedBytes"], f"{vector_location}.packedBytes"
            )
            if declared_packed_bytes != len(packed):
                _fail(
                    f"{vector_location}.packedBytes",
                    f"declared {declared_packed_bytes} != actual {len(packed)}",
                )
        decoded_features = decode_packed_i32(
            packed,
            parsed.header.n_features,
            location=f"{vector_location}.{packed_field}",
        )
        if "featuresQ" in vector:
            manifest_features = _output_vector(
                vector["featuresQ"], f"{vector_location}.featuresQ"
            )
            if manifest_features != decoded_features:
                _fail(
                    f"{vector_location}.featuresQ",
                    f"manifest values {manifest_features} != packed values {decoded_features}",
                )

        evaluated = evaluate_packed(
            parsed, packed, location=f"{vector_location}.{packed_field}"
        )

        output_fields = [
            field
            for field in ("expectedOutputQ", "localPredictionQ", "chainPredictionQ")
            if field in vector
        ]
        evm_read = vector.get("evmRead")
        if isinstance(evm_read, Mapping) and "outputQ" in evm_read:
            output_fields.append("evmRead.outputQ")
        if not output_fields:
            _fail(vector_location, "no expected local or chain output is archived")

        compared_outputs: list[tuple[str, tuple[int, ...]]] = []
        for field in output_fields:
            if field == "evmRead.outputQ":
                assert isinstance(evm_read, Mapping)
                raw_output = evm_read["outputQ"]
                field_location = f"{vector_location}.evmRead.outputQ"
            else:
                raw_output = vector[field]
                field_location = f"{vector_location}.{field}"
            expected = _output_vector(raw_output, field_location)
            if len(expected) != parsed.header.n_outputs:
                _fail(
                    field_location,
                    f"contains {len(expected)} outputs, expected {parsed.header.n_outputs}",
                )
            if expected != evaluated:
                _fail(
                    field_location,
                    f"archived output {expected} != independent evaluation {evaluated}",
                )
            compared_outputs.append((field, expected))

        if any(output != compared_outputs[0][1] for _name, output in compared_outputs):
            _fail(vector_location, "archived output fields disagree with each other")

        status = vector.get("status")
        if status is not None and status != "compared":
            _fail(f"{vector_location}.status", f"expected 'compared', got {status!r}")
        exact_match = vector.get("exactMatch")
        if exact_match is not None and exact_match is not True:
            _fail(f"{vector_location}.exactMatch", "expected true")
        if isinstance(evm_read, Mapping):
            if evm_read.get("status") not in (None, "compared"):
                _fail(
                    f"{vector_location}.evmRead.status",
                    f"expected 'compared', got {evm_read.get('status')!r}",
                )
            if evm_read.get("exactMatch") not in (None, True):
                _fail(f"{vector_location}.evmRead.exactMatch", "expected true")

    return len(vectors)


def _verify_file_inventory(manifest: Mapping[str, Any], root: Path) -> int:
    inventory = _required(manifest, "files", "manifest")
    entries: Iterable[tuple[str, Any]]
    if isinstance(inventory, list):
        normalized: list[tuple[str, Any]] = []
        for index, raw_entry in enumerate(inventory):
            entry = _require_mapping(raw_entry, f"manifest.files[{index}]")
            path_value = _required(entry, "path", f"manifest.files[{index}]")
            if not isinstance(path_value, str):
                _fail(f"manifest.files[{index}].path", "expected a string")
            normalized.append((path_value, entry))
        entries = normalized
    elif isinstance(inventory, Mapping):
        entries = inventory.items()
    else:
        _fail("manifest.files", "expected an array or path-keyed object")

    seen: set[str] = set()
    checked = 0
    for path_value, descriptor in entries:
        location = f"manifest.files[{path_value!r}]"
        if path_value in seen:
            _fail(location, "duplicate inventory path")
        seen.add(path_value)
        path = _relative_file(root, path_value, f"{location}.path")
        raw = path.read_bytes()
        if isinstance(descriptor, str):
            expected_sha = _digest_hex_or_plain(descriptor, location)
            expected_size = None
        else:
            entry = _require_mapping(descriptor, location)
            expected_sha = _digest_hex_or_plain(
                _required(entry, "sha256", location), f"{location}.sha256"
            )
            expected_size = (
                _integer(entry["size"], f"{location}.size")
                if "size" in entry
                else None
            )
        if expected_size is not None and expected_size != len(raw):
            _fail(location, f"declared size {expected_size} != actual {len(raw)}")
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            _fail(
                location,
                f"SHA-256 mismatch: expected {expected_sha}, got {actual_sha}",
            )
        checked += 1
    return checked


def _read_json_artifact(
    root: Path, file_value: Any, location: str
) -> tuple[Any, bytes, Path]:
    path = _relative_file(root, file_value, f"{location}.file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{location}: invalid UTF-8 JSON: {exc}") from exc
    return value, raw, path


def _read_gzip(path: Path, location: str) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(path.read_bytes())) as stream:
            decompressed = stream.read(MAX_TRANSCRIPT_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise VerificationError(f"{location}: invalid gzip stream: {exc}") from exc
    if len(decompressed) > MAX_TRANSCRIPT_BYTES:
        _fail(location, f"decompressed content exceeds {MAX_TRANSCRIPT_BYTES} bytes")
    return decompressed


def _verify_chain_and_block(
    manifest: Mapping[str, Any], root: Path
) -> tuple[int, int, str, str, str]:
    location = "manifest.chain"
    chain = _require_mapping(_required(manifest, "chain", "manifest"), location)
    chain_id = _integer(_required(chain, "chainId", location), f"{location}.chainId")
    block_number = _integer(
        _required(chain, "blockNumber", location), f"{location}.blockNumber"
    )
    block_tag_value = _required(chain, "blockTag", location)
    if not isinstance(block_tag_value, str):
        _fail(f"{location}.blockTag", "expected a string")
    expected_block_tag = hex(block_number)
    if block_tag_value.lower() != expected_block_tag:
        _fail(
            f"{location}.blockTag",
            f"{block_tag_value!r} != canonical {expected_block_tag!r}",
        )
    block_hash = _digest_hex(
        _required(chain, "blockHash", location), f"{location}.blockHash"
    )
    state_root = _digest_hex(
        _required(chain, "blockStateRoot", location),
        f"{location}.blockStateRoot",
    )
    block_file = _required(chain, "fullBlockFile", location)
    block_value, _raw, _path = _read_json_artifact(
        root, block_file, f"{location}.fullBlock"
    )
    block = _require_mapping(block_value, f"{location}.fullBlock")
    block_relations = (
        ("number", block_number),
        ("timestamp", _integer(_required(chain, "blockTimestamp", location),
                               f"{location}.blockTimestamp")),
    )
    for name, expected in block_relations:
        actual = _integer(
            _required(block, name, f"{location}.fullBlock"),
            f"{location}.fullBlock.{name}",
        )
        if actual != expected:
            _fail(
                f"{location}.fullBlock.{name}",
                f"block value {actual} != manifest value {expected}",
            )
    archived_hash = _digest_hex(
        _required(block, "hash", f"{location}.fullBlock"),
        f"{location}.fullBlock.hash",
    )
    archived_state_root = _digest_hex(
        _required(block, "stateRoot", f"{location}.fullBlock"),
        f"{location}.fullBlock.stateRoot",
    )
    if archived_hash != block_hash:
        _fail(f"{location}.fullBlock.hash", "does not match manifest blockHash")
    if archived_state_root != state_root:
        _fail(f"{location}.fullBlock.stateRoot", "does not match manifest state root")
    return chain_id, block_number, block_tag_value.lower(), block_hash, state_root


def _verify_collector_provenance(
    manifest: Mapping[str, Any], root: Path
) -> None:
    location = "manifest.collector"
    collector = _require_mapping(
        _required(manifest, "collector", "manifest"), location
    )
    script = _required(collector, "script", location)
    if not isinstance(script, str) or not script.endswith(".mjs"):
        _fail(f"{location}.script", "expected a non-empty .mjs repository path")
    archived_file = _required(collector, "archivedSourceFile", location)
    path = _relative_file(root, archived_file, f"{location}.archivedSourceFile")
    source = path.read_bytes()
    declared_bytes = _integer(
        _required(collector, "archivedSourceBytes", location),
        f"{location}.archivedSourceBytes",
    )
    if declared_bytes != len(source):
        _fail(
            f"{location}.archivedSourceBytes",
            f"declared {declared_bytes} != actual {len(source)}",
        )
    declared_sha = _digest_hex_or_plain(
        _required(collector, "archivedSourceSha256", location),
        f"{location}.archivedSourceSha256",
    )
    actual_sha = hashlib.sha256(source).hexdigest()
    if actual_sha != declared_sha:
        _fail(
            f"{location}.archivedSourceSha256",
            f"declared {declared_sha} != actual {actual_sha}",
        )
    try:
        source_text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(
            f"{location}.archivedSourceFile: invalid UTF-8: {exc}"
        ) from exc
    if not source_text.strip():
        _fail(
            f"{location}.archivedSourceFile",
            "embedded collector source is empty",
        )
    revision = _required(collector, "repositoryRevision", location)
    if not isinstance(revision, str) or len(revision) != 40:
        _fail(f"{location}.repositoryRevision", "expected a 40-digit Git object ID")
    try:
        bytes.fromhex(revision)
    except ValueError:
        _fail(f"{location}.repositoryRevision", "invalid hexadecimal Git object ID")
    worktree_state = _required(collector, "repositoryWorktreeState", location)
    if not isinstance(worktree_state, str) or not worktree_state.strip():
        _fail(f"{location}.repositoryWorktreeState", "expected a non-empty statement")


def _verify_contracts(
    manifest: Mapping[str, Any], root: Path
) -> tuple[dict[str, bytes], dict[bytes, str]]:
    contracts = _require_list(
        _required(manifest, "contracts", "manifest"), "manifest.contracts"
    )
    required_roles = {"store", "registry", "nft", "runtime", "marketplace"}
    role_addresses: dict[str, bytes] = {}
    code_hashes: dict[bytes, str] = {}
    for index, raw_contract in enumerate(contracts):
        location = f"manifest.contracts[{index}]"
        contract = _require_mapping(raw_contract, location)
        role = _required(contract, "role", location)
        if not isinstance(role, str) or not role:
            _fail(f"{location}.role", "expected a non-empty string")
        if role in role_addresses:
            _fail(f"{location}.role", f"duplicate role {role!r}")
        address = _address(
            _required(contract, "address", location), f"{location}.address"
        )
        if address == b"\0" * EVM_ADDRESS_BYTES:
            _fail(f"{location}.address", "zero address is forbidden")
        runtime, _path, offset, container_size = _runtime_bytes(
            root, contract, location
        )
        if offset != 0 or len(runtime) != container_size:
            _fail(location, "contract runtime must occupy its complete artifact file")
        digest = "0x" + keccak256(runtime).hex()
        if address in code_hashes:
            _fail(f"{location}.address", "duplicate contract address")
        role_addresses[role] = address
        code_hashes[address] = digest
    missing = required_roles - set(role_addresses)
    if missing:
        _fail("manifest.contracts", f"missing roles: {', '.join(sorted(missing))}")
    return role_addresses, code_hashes


def _verify_registry_topology(
    manifest: Mapping[str, Any], role_addresses: Mapping[str, bytes]
) -> None:
    location = "manifest.registrySnapshot"
    snapshot = _require_mapping(
        _required(manifest, "registrySnapshot", "manifest"), location
    )
    configured_nft = _address(
        _required(snapshot, "modelNFT", location), f"{location}.modelNFT"
    )
    deployed_nft = role_addresses["nft"]
    if configured_nft != deployed_nft:
        _fail(
            f"{location}.modelNFT",
            "registry-configured NFT address != archived NFT deployment address",
        )
    if snapshot.get("modelNFTMatchesDeploymentAddress") is not True:
        _fail(
            f"{location}.modelNFTMatchesDeploymentAddress",
            "expected the independently checked relation to be true",
        )


def _verify_source_witness(
    manifest: Mapping[str, Any],
    root: Path,
    chain_id: int,
    block_number: int,
    block_hash: str,
) -> Mapping[str, Any]:
    location = "manifest.sourceWitness"
    descriptor = _require_mapping(
        _required(manifest, "sourceWitness", "manifest"), location
    )
    file_value = _required(descriptor, "file", location)
    value, raw, _path = _read_json_artifact(root, file_value, location)
    expected_sha = _digest_hex_or_plain(
        _required(descriptor, "sha256", location), f"{location}.sha256"
    )
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        _fail(location, f"SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    witness = _require_mapping(value, location)
    schema = _required(witness, "schema", location)
    if schema != _required(descriptor, "schema", location):
        _fail(f"{location}.schema", "descriptor and embedded witness disagree")
    if schema != "gl1f-live-chain-witness/v2-extended":
        _fail(f"{location}.schema", f"unsupported witness schema {schema!r}")
    if _integer(_required(witness, "chainId", location), f"{location}.chainId") != chain_id:
        _fail(f"{location}.chainId", "does not match archive chain")
    witness_block = _require_mapping(
        _required(witness, "block", location), f"{location}.block"
    )
    if _integer(
        _required(witness_block, "number", f"{location}.block"),
        f"{location}.block.number",
    ) != block_number:
        _fail(f"{location}.block.number", "does not match archive block")
    witness_hash = _digest_hex(
        _required(witness_block, "hash", f"{location}.block"),
        f"{location}.block.hash",
    )
    if witness_hash != block_hash:
        _fail(f"{location}.block.hash", "does not match archive block hash")
    return witness


def _load_transcript(
    manifest: Mapping[str, Any], root: Path
) -> tuple[dict[int, RpcObservation], str]:
    read_only = _require_mapping(
        _required(manifest, "readOnlyRpc", "manifest"), "manifest.readOnlyRpc"
    )
    transcript_descriptor = _require_mapping(
        _required(read_only, "rawTranscript", "manifest.readOnlyRpc"),
        "manifest.readOnlyRpc.rawTranscript",
    )
    file_value = _required(
        transcript_descriptor, "file", "manifest.readOnlyRpc.rawTranscript"
    )
    path = _relative_file(
        root, file_value, "manifest.readOnlyRpc.rawTranscript.file"
    )
    decompressed = _read_gzip(path, "manifest.readOnlyRpc.rawTranscript")
    try:
        text = decompressed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(
            f"manifest.readOnlyRpc.rawTranscript: invalid UTF-8: {exc}"
        ) from exc
    lines = text.splitlines()
    expected_entries = _integer(
        _required(
            transcript_descriptor,
            "requestAttempts",
            "manifest.readOnlyRpc.rawTranscript",
        ),
        "manifest.readOnlyRpc.rawTranscript.requestAttempts",
    )
    if len(lines) != expected_entries:
        _fail(
            "manifest.readOnlyRpc.rawTranscript",
            f"contains {len(lines)} entries, manifest declares {expected_entries}",
        )

    successful: dict[int, RpcObservation] = {}
    all_request_ids: set[int] = set()
    method_counts: dict[str, int] = {}
    previous_sequence = -1
    for line_index, line in enumerate(lines):
        location = f"transcript.line[{line_index + 1}]"
        try:
            entry_value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"{location}: invalid JSON: {exc}") from exc
        entry = _require_mapping(entry_value, location)
        sequence = _integer(
            _required(entry, "sequence", location), f"{location}.sequence"
        )
        if sequence <= previous_sequence:
            _fail(f"{location}.sequence", "entries are not strictly ordered")
        previous_sequence = sequence
        request_body = _required(entry, "requestBody", location)
        if not isinstance(request_body, str):
            _fail(f"{location}.requestBody", "expected an exact JSON string")
        try:
            request_value = json.loads(request_body)
        except json.JSONDecodeError as exc:
            raise VerificationError(
                f"{location}.requestBody: invalid JSON: {exc}"
            ) from exc
        requests = request_value if isinstance(request_value, list) else [request_value]
        if not requests:
            _fail(f"{location}.requestBody", "empty JSON-RPC batch")
        request_by_id: dict[int, tuple[int, Mapping[str, Any]]] = {}
        for batch_index, raw_request in enumerate(requests):
            request = _require_mapping(
                raw_request, f"{location}.request[{batch_index}]"
            )
            if request.get("jsonrpc") != "2.0":
                _fail(f"{location}.request[{batch_index}]", "jsonrpc must be '2.0'")
            request_id = _integer(
                _required(request, "id", location),
                f"{location}.request[{batch_index}].id",
            )
            if request_id in all_request_ids:
                _fail(f"{location}.request[{batch_index}].id", "duplicate request ID")
            all_request_ids.add(request_id)
            method = _required(request, "method", location)
            if not isinstance(method, str):
                _fail(f"{location}.request[{batch_index}].method", "expected string")
            method_counts[method] = method_counts.get(method, 0) + 1
            request_by_id[request_id] = (batch_index, request)
        if sequence != min(request_by_id):
            _fail(f"{location}.sequence", "must equal the first batch request ID")

        if entry.get("httpStatus") != 200 or entry.get("responseBody") is None:
            continue
        response_body = entry["responseBody"]
        if not isinstance(response_body, str):
            _fail(f"{location}.responseBody", "expected an exact JSON string")
        try:
            response_value = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise VerificationError(
                f"{location}.responseBody: invalid JSON: {exc}"
            ) from exc
        responses = (
            response_value if isinstance(response_value, list) else [response_value]
        )
        response_by_id: dict[int, Mapping[str, Any]] = {}
        for response_index, raw_response in enumerate(responses):
            response = _require_mapping(
                raw_response, f"{location}.response[{response_index}]"
            )
            if response.get("jsonrpc") != "2.0":
                _fail(f"{location}.response[{response_index}]", "jsonrpc must be '2.0'")
            response_id = _integer(
                _required(response, "id", location),
                f"{location}.response[{response_index}].id",
            )
            if response_id in response_by_id:
                _fail(f"{location}.response[{response_index}].id", "duplicate ID")
            response_by_id[response_id] = response
        if set(response_by_id) != set(request_by_id):
            _fail(location, "successful response IDs do not equal request IDs")
        for request_id, (batch_index, request) in request_by_id.items():
            response = response_by_id[request_id]
            if "error" in response:
                continue
            if "result" not in response:
                _fail(location, f"response {request_id} has neither result nor error")
            successful[request_id] = RpcObservation(
                entry_sequence=sequence,
                request_id=request_id,
                response_id=request_id,
                batch_index=batch_index,
                method=str(request["method"]),
                params=request.get("params"),
                result=response["result"],
            )

    declared_counts = _require_mapping(
        _required(read_only, "methodRequestCounts", "manifest.readOnlyRpc"),
        "manifest.readOnlyRpc.methodRequestCounts",
    )
    normalized_counts = {
        str(method): _integer(
            count, f"manifest.readOnlyRpc.methodRequestCounts.{method}"
        )
        for method, count in declared_counts.items()
    }
    if normalized_counts != dict(sorted(method_counts.items())):
        _fail(
            "manifest.readOnlyRpc.methodRequestCounts",
            "does not match exact transcript request bodies",
        )
    methods_used = _require_list(
        _required(read_only, "methodsUsed", "manifest.readOnlyRpc"),
        "manifest.readOnlyRpc.methodsUsed",
    )
    if methods_used != sorted(method_counts):
        _fail("manifest.readOnlyRpc.methodsUsed", "does not match transcript methods")
    prohibited = _require_list(
        _required(read_only, "prohibitedAndUnusedMethods", "manifest.readOnlyRpc"),
        "manifest.readOnlyRpc.prohibitedAndUnusedMethods",
    )
    overlap = set(prohibited) & set(method_counts)
    if overlap:
        _fail(
            "manifest.readOnlyRpc.prohibitedAndUnusedMethods",
            f"methods present in transcript: {', '.join(sorted(overlap))}",
        )
    return successful, str(file_value)


def _verify_transcript_chain(
    calls: Mapping[int, RpcObservation],
    chain_id: int,
    block_number: int,
    block_hash: str,
) -> None:
    chain_results = [
        call.result for call in calls.values() if call.method == "eth_chainId"
    ]
    if not chain_results:
        _fail("transcript", "no successful eth_chainId call")
    if any(_integer(value, "transcript.eth_chainId") != chain_id
           for value in chain_results):
        _fail("transcript.eth_chainId", "result does not match manifest chainId")
    matching_blocks = 0
    for call in calls.values():
        if call.method != "eth_getBlockByNumber":
            continue
        params = _require_list(call.params, "transcript.eth_getBlockByNumber.params")
        if len(params) != 2 or _integer(params[0], "transcript.blockTag") != block_number:
            continue
        result = _require_mapping(call.result, "transcript.eth_getBlockByNumber.result")
        result_hash = _digest_hex(
            _required(result, "hash", "transcript.eth_getBlockByNumber.result"),
            "transcript.eth_getBlockByNumber.result.hash",
        )
        if result_hash == block_hash:
            matching_blocks += 1
    if matching_blocks < 2:
        _fail(
            "transcript.eth_getBlockByNumber",
            "expected successful initial and final pinned-hash observations",
        )


def _source_model(
    witness: Mapping[str, Any], token_id: int, location: str
) -> Mapping[str, Any]:
    models = _require_list(_required(witness, "models", "sourceWitness"), "sourceWitness.models")
    matches = [
        _require_mapping(value, f"sourceWitness.models[{index}]")
        for index, value in enumerate(models)
        if isinstance(value, Mapping)
        and _integer(value.get("tokenId"), f"sourceWitness.models[{index}].tokenId")
        == token_id
    ]
    if len(matches) != 1:
        _fail(location, f"expected exactly one source-witness model for token {token_id}")
    return matches[0]


def _verify_source_model_equivalence(
    model: Mapping[str, Any],
    source: Mapping[str, Any],
    location: str,
) -> list[Mapping[str, Any]]:
    hex_relations = ("modelId", "tablePtr")
    for field in hex_relations:
        if str(model.get(field)).lower() != str(source.get(field)).lower():
            _fail(f"{location}.{field}", "archive and embedded source witness differ")
    for field in ("chunkSize", "numChunks", "totalBytes"):
        left = _integer(_required(model, field, location), f"{location}.{field}")
        right = _integer(
            _required(source, field, f"{location}.source"),
            f"{location}.source.{field}",
        )
        if left != right:
            _fail(f"{location}.{field}", "archive and embedded source witness differ")
    archive_vectors = _require_list(
        _required(model, "vectors", location), f"{location}.vectors"
    )
    source_study = _require_mapping(
        _required(source, "conformanceStudy", f"{location}.source"),
        f"{location}.source.conformanceStudy",
    )
    source_vectors_raw = _require_list(
        _required(source_study, "vectors", f"{location}.source.conformanceStudy"),
        f"{location}.source.conformanceStudy.vectors",
    )
    if len(archive_vectors) != len(source_vectors_raw):
        _fail(f"{location}.vectors", "archive and source vector counts differ")
    source_vectors: list[Mapping[str, Any]] = []
    for index, (raw_archive, raw_source) in enumerate(
        zip(archive_vectors, source_vectors_raw)
    ):
        vector_location = f"{location}.vectors[{index}]"
        archive_vector = _require_mapping(raw_archive, vector_location)
        source_vector = _require_mapping(
            raw_source, f"{vector_location}.source"
        )
        source_vectors.append(source_vector)
        scalar_relations = ("vectorId", "packedFeaturesQHex", "packedBytes")
        for field in scalar_relations:
            if archive_vector.get(field) != source_vector.get(field):
                _fail(
                    f"{vector_location}.{field}",
                    "archive and embedded source witness differ",
                )
        source_local = _output_vector(
            _required(source_vector, "localPredictionQ", f"{vector_location}.source"),
            f"{vector_location}.source.localPredictionQ",
        )
        source_evm = _require_mapping(
            _required(source_vector, "evmRead", f"{vector_location}.source"),
            f"{vector_location}.source.evmRead",
        )
        source_chain = _output_vector(
            _required(source_evm, "outputQ", f"{vector_location}.source.evmRead"),
            f"{vector_location}.source.evmRead.outputQ",
        )
        for field, expected in (
            ("expectedOutputQ", source_local),
            ("localPredictionQ", source_local),
            ("chainPredictionQ", source_chain),
            ("sourceWitnessChainPredictionQ", source_chain),
        ):
            actual = _output_vector(
                _required(archive_vector, field, vector_location),
                f"{vector_location}.{field}",
            )
            if actual != expected:
                _fail(
                    f"{vector_location}.{field}",
                    "archive and embedded source witness differ",
                )
        if archive_vector.get("status") != source_evm.get("status"):
            _fail(f"{vector_location}.status", "source-witness status mismatch")
        if archive_vector.get("exactMatch") != source_evm.get("exactMatch"):
            _fail(f"{vector_location}.exactMatch", "source-witness flag mismatch")
    return source_vectors


def _abi_call_data(method: str, model_id: bytes, packed: bytes) -> bytes:
    if method == "predictView":
        signature = b"predictView(bytes32,bytes)"
    elif method == "predictMultiView":
        signature = b"predictMultiView(bytes32,bytes)"
    else:
        _fail("ABI", f"unsupported inference contract method {method!r}")
    selector = keccak256(signature)[:4]
    padding = (-len(packed)) % 32
    return (
        selector
        + model_id
        + (64).to_bytes(32, "big")
        + len(packed).to_bytes(32, "big")
        + packed
        + b"\0" * padding
    )


def _decode_i256(word: bytes) -> int:
    unsigned = int.from_bytes(word, "big")
    return unsigned - (1 << 256) if unsigned >= (1 << 255) else unsigned


def _decode_inference_result(
    raw_result: Any, n_outputs: int, location: str
) -> tuple[int, ...]:
    data = _hex_bytes(raw_result, location)
    if n_outputs == 1:
        if len(data) != 32:
            _fail(location, f"scalar ABI result must be 32 bytes, got {len(data)}")
        return (_decode_i256(data),)
    expected_bytes = 64 + 32 * n_outputs
    if len(data) != expected_bytes:
        _fail(
            location,
            f"vector ABI result must be {expected_bytes} bytes, got {len(data)}",
        )
    if int.from_bytes(data[:32], "big") != 32:
        _fail(location, "noncanonical dynamic-array offset")
    if int.from_bytes(data[32:64], "big") != n_outputs:
        _fail(location, "dynamic-array length does not match model outputs")
    return tuple(
        _decode_i256(data[64 + 32 * index : 96 + 32 * index])
        for index in range(n_outputs)
    )


def _verify_vector_transcript_bindings(
    model: Mapping[str, Any],
    parsed: ParsedCore,
    source_vectors: Sequence[Mapping[str, Any]],
    calls: Mapping[int, RpcObservation],
    transcript_file: str,
    runtime_address: bytes,
    block_tag: str,
    location: str,
) -> None:
    vectors = _require_list(
        _required(model, "vectors", location), f"{location}.vectors"
    )
    model_id = _hex_bytes(
        _required(model, "modelId", location),
        f"{location}.modelId",
        exact_bytes=32,
    )
    for index, (raw_vector, source_vector) in enumerate(zip(vectors, source_vectors)):
        vector_location = f"{location}.vectors[{index}]"
        vector = _require_mapping(raw_vector, vector_location)
        binding = _require_mapping(
            _required(vector, "archiveChainCall", vector_location),
            f"{vector_location}.archiveChainCall",
        )
        request_id = _integer(
            _required(binding, "requestId", f"{vector_location}.archiveChainCall"),
            f"{vector_location}.archiveChainCall.requestId",
        )
        call = calls.get(request_id)
        if call is None:
            _fail(
                f"{vector_location}.archiveChainCall.requestId",
                "does not identify a successful transcript call",
            )
        binding_integer_relations = (
            ("transcriptSequence", call.entry_sequence),
            ("responseId", call.response_id),
            ("batchIndex", call.batch_index),
        )
        for field, expected in binding_integer_relations:
            actual = _integer(
                _required(binding, field, f"{vector_location}.archiveChainCall"),
                f"{vector_location}.archiveChainCall.{field}",
            )
            if actual != expected:
                _fail(
                    f"{vector_location}.archiveChainCall.{field}",
                    f"binding value {actual} != transcript value {expected}",
                )
        if binding.get("transcriptFile") != transcript_file:
            _fail(
                f"{vector_location}.archiveChainCall.transcriptFile",
                "does not identify the archived raw transcript",
            )
        if binding.get("rpcMethod") != "eth_call" or call.method != "eth_call":
            _fail(f"{vector_location}.archiveChainCall.rpcMethod", "must be eth_call")

        source_rpc = _require_mapping(
            _required(source_vector, "rpcRequest", f"{vector_location}.source"),
            f"{vector_location}.source.rpcRequest",
        )
        contract_method = _required(
            binding, "contractMethod", f"{vector_location}.archiveChainCall"
        )
        if contract_method != source_rpc.get("method"):
            _fail(
                f"{vector_location}.archiveChainCall.contractMethod",
                "does not match embedded source witness",
            )
        packed = _hex_bytes(
            _required(vector, "packedFeaturesQHex", vector_location),
            f"{vector_location}.packedFeaturesQHex",
        )
        calldata = _abi_call_data(str(contract_method), model_id, packed)
        calldata_hex = "0x" + calldata.hex()
        calldata_hash = "0x" + keccak256(calldata).hex()
        if _integer(
            _required(binding, "calldataBytes", f"{vector_location}.archiveChainCall"),
            f"{vector_location}.archiveChainCall.calldataBytes",
        ) != len(calldata):
            _fail(f"{vector_location}.archiveChainCall.calldataBytes", "mismatch")
        if _digest_hex(
            _required(
                binding, "calldataKeccak256", f"{vector_location}.archiveChainCall"
            ),
            f"{vector_location}.archiveChainCall.calldataKeccak256",
        ) != calldata_hash:
            _fail(f"{vector_location}.archiveChainCall.calldataKeccak256", "mismatch")
        if _digest_hex(
            _required(source_rpc, "calldataKeccak256", f"{vector_location}.source.rpcRequest"),
            f"{vector_location}.source.rpcRequest.calldataKeccak256",
        ) != calldata_hash:
            _fail(f"{vector_location}.source.rpcRequest", "calldata digest mismatch")

        params = _require_list(call.params, f"{vector_location}.transcript.params")
        if len(params) != 2:
            _fail(f"{vector_location}.transcript.params", "eth_call needs two params")
        transaction = _require_mapping(
            params[0], f"{vector_location}.transcript.params[0]"
        )
        call_to = _address(
            _required(transaction, "to", f"{vector_location}.transcript.params[0]"),
            f"{vector_location}.transcript.params[0].to",
        )
        if call_to != runtime_address:
            _fail(f"{vector_location}.transcript.params[0].to", "runtime mismatch")
        call_data = _hex_bytes(
            _required(transaction, "data", f"{vector_location}.transcript.params[0]"),
            f"{vector_location}.transcript.params[0].data",
        )
        if call_data != calldata:
            _fail(f"{vector_location}.transcript.params[0].data", "calldata mismatch")
        if not isinstance(params[1], str) or params[1].lower() != block_tag:
            _fail(f"{vector_location}.transcript.params[1]", "block tag mismatch")
        if _address(
            _required(binding, "to", f"{vector_location}.archiveChainCall"),
            f"{vector_location}.archiveChainCall.to",
        ) != runtime_address:
            _fail(f"{vector_location}.archiveChainCall.to", "runtime mismatch")
        if str(binding.get("blockTag", "")).lower() != block_tag:
            _fail(f"{vector_location}.archiveChainCall.blockTag", "block mismatch")

        decoded = _decode_inference_result(
            call.result,
            parsed.header.n_outputs,
            f"{vector_location}.transcript.result",
        )
        local = _output_vector(
            _required(vector, "localPredictionQ", vector_location),
            f"{vector_location}.localPredictionQ",
        )
        if decoded != local:
            _fail(
                f"{vector_location}.transcript.result",
                f"ABI-decoded output {decoded} != local evaluation {local}",
            )
        if _hex_bytes(
            _required(binding, "rawResultHex", f"{vector_location}.archiveChainCall"),
            f"{vector_location}.archiveChainCall.rawResultHex",
        ) != _hex_bytes(call.result, f"{vector_location}.transcript.result"):
            _fail(f"{vector_location}.archiveChainCall.rawResultHex", "mismatch")
        binding_output = _output_vector(
            _required(
                binding, "decodedOutputQ", f"{vector_location}.archiveChainCall"
            ),
            f"{vector_location}.archiveChainCall.decodedOutputQ",
        )
        if binding_output != decoded:
            _fail(f"{vector_location}.archiveChainCall.decodedOutputQ", "mismatch")
        for field in ("matchesLocalPrediction", "matchesEmbeddedSourceWitness"):
            if binding.get(field) is not True:
                _fail(f"{vector_location}.archiveChainCall.{field}", "expected true")


def _merge_code_hashes(
    destination: dict[bytes, str],
    additions: Mapping[bytes, str],
    location: str,
) -> None:
    for address, digest in additions.items():
        prior = destination.get(address)
        if prior is not None and prior != digest:
            _fail(location, "same address is bound to different runtime code")
        destination[address] = digest


def _verify_proofs(
    manifest: Mapping[str, Any],
    root: Path,
    code_hashes: Mapping[bytes, str],
) -> None:
    location = "manifest.proofEvidence"
    evidence = _require_mapping(
        _required(manifest, "proofEvidence", "manifest"), location
    )
    available = evidence.get("available")
    if available is False:
        if evidence.get("file") is not None:
            _fail(f"{location}.file", "must be null when proofs are unavailable")
        return
    if available is not True:
        _fail(f"{location}.available", "expected a Boolean")
    file_value = _required(evidence, "file", location)
    path = _relative_file(root, file_value, f"{location}.file")
    raw = _read_gzip(path, location)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{location}: invalid proof JSON: {exc}") from exc
    records = _require_list(value, location)
    declared_count = _integer(
        _required(evidence, "accountProofs", location),
        f"{location}.accountProofs",
    )
    if declared_count != len(records):
        _fail(location, f"contains {len(records)} proofs, declares {declared_count}")
    seen: set[bytes] = set()
    for index, raw_record in enumerate(records):
        record_location = f"{location}.proofs[{index}]"
        record = _require_mapping(raw_record, record_location)
        address = _address(
            _required(record, "address", record_location),
            f"{record_location}.address",
        )
        if address in seen:
            _fail(f"{record_location}.address", "duplicate proof address")
        seen.add(address)
        proof = _require_mapping(
            _required(record, "proof", record_location),
            f"{record_location}.proof",
        )
        proof_address = _address(
            _required(proof, "address", f"{record_location}.proof"),
            f"{record_location}.proof.address",
        )
        if proof_address != address:
            _fail(f"{record_location}.proof.address", "record/proof mismatch")
        expected = code_hashes.get(address)
        if expected is None:
            _fail(f"{record_location}.address", "has no archived runtime artifact")
        actual = _digest_hex(
            _required(proof, "codeHash", f"{record_location}.proof"),
            f"{record_location}.proof.codeHash",
        )
        if actual != expected:
            _fail(
                f"{record_location}.proof.codeHash",
                f"provider codeHash {actual} != Keccak(runtime) {expected}",
            )
        storage_proofs = _require_list(
            _required(proof, "storageProof", f"{record_location}.proof"),
            f"{record_location}.proof.storageProof",
        )
        if storage_proofs:
            _fail(f"{record_location}.proof.storageProof", "expected no slot proofs")
    if seen != set(code_hashes):
        missing = len(set(code_hashes) - seen)
        extra = len(seen - set(code_hashes))
        _fail(location, f"proof/runtime address sets differ ({missing} missing, {extra} extra)")
    declared_matched = _integer(
        _required(evidence, "codeHashesMatched", location),
        f"{location}.codeHashesMatched",
    )
    if declared_matched != len(records):
        _fail(f"{location}.codeHashesMatched", "does not equal verified proof count")


def verify_manifest(manifest_path: str | Path) -> VerificationReport:
    path = Path(manifest_path).resolve()
    try:
        manifest_raw = path.read_bytes()
    except OSError as exc:
        raise VerificationError(f"manifest: cannot read {path}: {exc}") from exc
    try:
        manifest_value = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"manifest: invalid UTF-8 JSON: {exc}") from exc
    manifest = _require_mapping(manifest_value, "manifest")
    root = path.parent

    schema_version = manifest.get("schemaVersion", manifest.get("schema"))
    if schema_version is None:
        _fail("manifest", "missing required field 'schema' or 'schemaVersion'")
    if schema_version not in (3, "3", "gl1f-live-chain-archive/v3"):
        _fail("manifest.schemaVersion", f"unsupported archive schema {schema_version!r}")
    archive_id_value = _required(manifest, "archiveId", "manifest")
    if not isinstance(archive_id_value, str) or not archive_id_value:
        _fail("manifest.archiveId", "expected a non-empty string")

    files_checked = _verify_file_inventory(manifest, root)
    _verify_collector_provenance(manifest, root)
    chain_id, block_number, block_tag, block_hash, _state_root = (
        _verify_chain_and_block(manifest, root)
    )
    role_addresses, code_hashes = _verify_contracts(manifest, root)
    _verify_registry_topology(manifest, role_addresses)
    source_witness = _verify_source_witness(
        manifest, root, chain_id, block_number, block_hash
    )
    transcript_calls, transcript_file = _load_transcript(manifest, root)
    _verify_transcript_chain(
        transcript_calls, chain_id, block_number, block_hash
    )
    models = _require_list(_required(manifest, "models", "manifest"), "manifest.models")
    if not models:
        _fail("manifest.models", "at least one model is required")

    vector_count = 0
    core_bytes = 0
    seen_tokens: set[int] = set()
    seen_model_ids: set[str] = set()
    for index, raw_model in enumerate(models):
        location = f"manifest.models[{index}]"
        model = _require_mapping(raw_model, location)
        token_id = _integer(
            _required(model, "tokenId", location), f"{location}.tokenId"
        )
        if token_id <= 0:
            _fail(f"{location}.tokenId", "must be positive")
        if token_id in seen_tokens:
            _fail(f"{location}.tokenId", f"duplicate token ID {token_id}")
        seen_tokens.add(token_id)

        core_descriptor = _artifact_descriptor(model, "core", location)
        core_file = _file_reference(core_descriptor, f"{location}.core")
        core_path = _relative_file(root, core_file, f"{location}.core.file")
        core = _decode_artifact_file(core_path, f"{location}.core.file")
        _declared_artifact_checks(core_descriptor, core, f"{location}.core")
        parsed = parse_core(core, location=f"{location}.core")
        if not 1 <= parsed.header.total_trees <= 65_535:
            _fail(
                f"{location}.core",
                f"total tree count {parsed.header.total_trees} is outside "
                "the deployed registry profile 1..65535",
            )
        _compare_header_manifest(
            _required(model, "header", location),
            parsed,
            f"{location}.header",
        )

        computed_model_id = "0x" + keccak256(core).hex()
        model_id = _digest_hex(
            _required(model, "modelId", location), f"{location}.modelId"
        )
        if model_id != computed_model_id:
            _fail(
                f"{location}.modelId",
                f"registry commitment {model_id} != Keccak(core) {computed_model_id}",
            )
        if "computedModelId" in model:
            archived_computed = _digest_hex(
                model["computedModelId"], f"{location}.computedModelId"
            )
            if archived_computed != computed_model_id:
                _fail(
                    f"{location}.computedModelId",
                    f"archived value {archived_computed} != recomputed {computed_model_id}",
                )
        if model_id in seen_model_ids:
            _fail(f"{location}.modelId", f"duplicate model ID {model_id}")
        seen_model_ids.add(model_id)

        (
            reconstructed,
            table_address,
            chunk_size,
            num_chunks,
            total_bytes,
            model_code_hashes,
        ) = _reconstruct_storage(root, model, location)
        _merge_code_hashes(code_hashes, model_code_hashes, location)
        if reconstructed != core:
            _fail(location, "ordered GL1C chunk reconstruction != archived core bytes")
        if total_bytes != len(core):
            _fail(
                f"{location}.totalBytes",
                f"manifest value {total_bytes} != strict core length {len(core)}",
            )

        top_table_ptr = _address(
            _required(model, "tablePtr", location), f"{location}.tablePtr"
        )
        if top_table_ptr != table_address:
            _fail(f"{location}.tablePtr", "manifest pointer != table address")

        registry = _registry_object(model, location)
        _compare_registry(
            registry,
            parsed,
            model_id,
            table_address,
            chunk_size,
            num_chunks,
            total_bytes,
            f"{location}.registry",
        )

        agreement = model.get("registryHeaderAgreement")
        if agreement is not None:
            agreement_obj = _require_mapping(
                agreement, f"{location}.registryHeaderAgreement"
            )
            for name, value in agreement_obj.items():
                if value is not True:
                    _fail(
                        f"{location}.registryHeaderAgreement.{name}",
                        "archived precomputed relation is not true",
                    )

        vector_count += _verify_vectors(model, parsed, location)
        source = _source_model(source_witness, token_id, location)
        source_vectors = _verify_source_model_equivalence(
            model, source, location
        )
        _verify_vector_transcript_bindings(
            model,
            parsed,
            source_vectors,
            transcript_calls,
            transcript_file,
            role_addresses["runtime"],
            block_tag,
            location,
        )
        core_bytes += len(core)

    _verify_proofs(manifest, root, code_hashes)
    return VerificationReport(
        manifest=str(path),
        archive_id=archive_id_value,
        files_checked=files_checked,
        models_checked=len(models),
        vectors_checked=vector_count,
        core_bytes_checked=core_bytes,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently verify the self-contained GL1F live-chain archive "
            "without network access or production decoders."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"archive manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the verification report as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_manifest(args.manifest)
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(
            "VERIFIED "
            f"{report.models_checked} models, "
            f"{report.vectors_checked} vectors, "
            f"{report.core_bytes_checked} core bytes, "
            f"{report.files_checked} inventory files "
            f"(archive {report.archive_id})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
