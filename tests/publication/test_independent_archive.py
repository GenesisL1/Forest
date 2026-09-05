"""Focused tests for the offline independent archive verifier."""

from __future__ import annotations

import hashlib
import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path

from benchmarks.independent_archive_verify import (
    VerificationError,
    evaluate_packed,
    keccak256,
    parse_core,
    verify_manifest,
)


def _node(feature: int, threshold: int, reserved: int = 0) -> bytes:
    return struct.pack("<HiH", feature, threshold, reserved)


def _v1_core() -> bytes:
    header = struct.pack(
        "<4sBBHHIiIH",
        b"GL1F",
        1,
        0,
        2,
        1,
        2,
        10,
        100,
        0,
    )
    tree_zero = _node(0, 5) + struct.pack("<ii", -2, 7)
    tree_one = _node(1, -1) + struct.pack("<ii", 4, -6)
    return header + tree_zero + tree_one


def _v2_core() -> bytes:
    header = struct.pack(
        "<4sBBHHIiIH",
        b"GL1F",
        2,
        0,
        1,
        1,
        1,
        0,
        100,
        2,
    )
    bases = struct.pack("<ii", -10, 20)
    output_zero = _node(0, 0) + struct.pack("<ii", 1, 2)
    output_one = _node(0, 0) + struct.pack("<ii", 3, 4)
    return header + bases + output_zero + output_one


def _address(byte: int) -> str:
    return "0x" + (bytes([byte]) * 20).hex()


class IndependentCoreTests(unittest.TestCase):
    def test_ethereum_keccak_known_answers(self) -> None:
        self.assertEqual(
            keccak256(b"").hex(),
            "c5d2460186f7233c927e7db2dcc703c0e"
            "500b653ca82273b7bfad8045d85a470",
        )
        self.assertEqual(
            keccak256(b"abc").hex(),
            "4e03657aea45a94fc7d47ba826c8d667c"
            "0d1e6e33a64a036ec44f58fa12d6c45",
        )
        rate_minus_one = bytes((index * 37 + 135) & 0xFF for index in range(135))
        self.assertEqual(
            keccak256(rate_minus_one).hex(),
            "1ca68768bdc448a167f00e4e610a911cd"
            "1bc3b3958b3cf8973366ecb233a0906",
        )
        one_full_rate = bytes((index * 37 + 136) & 0xFF for index in range(136))
        self.assertEqual(
            keccak256(one_full_rate).hex(),
            "71b476ce5fb1fc98a82a5a26d39bab4e"
            "c416e0d3b8481bed4126ae6893987d3b",
        )

    def test_v1_signed_little_endian_and_equality_left(self) -> None:
        parsed = parse_core(_v1_core())
        packed = struct.pack("<ii", 5, 0)
        self.assertEqual(evaluate_packed(parsed, packed), (2,))
        self.assertEqual(parsed.header.total_trees, 2)

    def test_v2_output_major_integer_evaluation(self) -> None:
        parsed = parse_core(_v2_core())
        self.assertEqual(evaluate_packed(parsed, struct.pack("<i", 0)), (-9, 23))
        self.assertEqual(evaluate_packed(parsed, struct.pack("<i", 1)), (-8, 24))

    def test_accumulation_does_not_wrap_at_int32(self) -> None:
        maximum = (1 << 31) - 1
        header = struct.pack(
            "<4sBBHHIiIH",
            b"GL1F",
            1,
            0,
            1,
            1,
            2,
            maximum,
            1,
            0,
        )
        tree = _node(0, 0) + struct.pack("<ii", 0, maximum)
        parsed = parse_core(header + tree + tree)
        self.assertEqual(
            evaluate_packed(parsed, struct.pack("<i", 1)),
            (3 * maximum,),
        )

    def test_strict_length_reserved_and_feature_checks(self) -> None:
        core = _v1_core()
        with self.assertRaisesRegex(VerificationError, "truncated"):
            parse_core(core[:-1])
        with self.assertRaisesRegex(VerificationError, "trailing bytes"):
            parse_core(core + b"\0")

        bad_header_reserved = bytearray(core)
        bad_header_reserved[5] = 1
        with self.assertRaisesRegex(VerificationError, "reserved byte"):
            parse_core(bytes(bad_header_reserved))

        bad_node_reserved = bytearray(core)
        bad_node_reserved[30] = 1
        with self.assertRaisesRegex(VerificationError, "reserved bytes"):
            parse_core(bytes(bad_node_reserved))

        bad_feature = bytearray(core)
        struct.pack_into("<H", bad_feature, 24, 2)
        with self.assertRaisesRegex(VerificationError, "feature index"):
            parse_core(bytes(bad_feature))

    def test_packed_vector_must_have_exact_width(self) -> None:
        parsed = parse_core(_v1_core())
        with self.assertRaisesRegex(VerificationError, "exactly 8 bytes"):
            evaluate_packed(parsed, b"\0" * 4)


class IndependentArchiveTests(unittest.TestCase):
    def _make_archive(self, root: Path) -> Path:
        core = _v1_core()
        split = 31
        payloads = (core[:split], core[split:])
        chunk_addresses = (_address(0x11), _address(0x22))
        table_address = _address(0x33)
        chunk_codes = tuple(b"GL1C" + payload for payload in payloads)
        table_code = b"GL1C" + b"".join(
            b"\0" * 12 + bytes.fromhex(address[2:])
            for address in chunk_addresses
        )
        storage_runtime = table_code + b"".join(chunk_codes)
        packed = struct.pack("<ii", 5, 0)
        model_id = "0x" + keccak256(core).hex()
        runtime_address = _address(0x44)
        block_hash = "0x" + "aa" * 32
        state_root = "0x" + "bb" * 32
        block = {
            "number": "0x2a",
            "timestamp": "0x7b",
            "hash": block_hash,
            "stateRoot": state_root,
            "transactions": [],
        }
        selector = keccak256(b"predictView(bytes32,bytes)")[:4]
        calldata = (
            selector
            + bytes.fromhex(model_id[2:])
            + (64).to_bytes(32, "big")
            + len(packed).to_bytes(32, "big")
            + packed
            + b"\0" * ((-len(packed)) % 32)
        )
        calldata_hex = "0x" + calldata.hex()
        result_hex = "0x" + (2).to_bytes(32, "big").hex()

        (root / "model.gl1f").write_bytes(core)
        (root / "storage-runtime.bin").write_bytes(storage_runtime)
        (root / "block.json").write_text(json.dumps(block), encoding="utf-8")
        collector_source = b"// synthetic archive collector\n"
        (root / "collector.mjs").write_bytes(collector_source)

        contracts = []
        for index, role in enumerate(
            ("store", "registry", "nft", "runtime", "marketplace")
        ):
            address = runtime_address if role == "runtime" else _address(0x50 + index)
            runtime = f"synthetic-{role}-runtime".encode()
            filename = f"contract-{role}.bin"
            (root / filename).write_bytes(runtime)
            contracts.append(
                {
                    "role": role,
                    "address": address,
                    "runtime": {
                        "file": filename,
                        "size": len(runtime),
                        "sha256": hashlib.sha256(runtime).hexdigest(),
                        "keccak256": "0x" + keccak256(runtime).hex(),
                    },
                }
            )

        source_vector = {
            "vectorId": "equality-left-and-signed",
            "packedFeaturesQHex": "0x" + packed.hex(),
            "packedBytes": len(packed),
            "localPredictionQ": ["2"],
            "evmRead": {
                "status": "compared",
                "outputQ": ["2"],
                "exactMatch": True,
            },
            "rpcRequest": {
                "to": runtime_address,
                "method": "predictView",
                "blockTag": "0x2a",
                "calldataBytes": len(calldata),
                "calldataKeccak256": "0x" + keccak256(calldata).hex(),
            },
        }
        source_witness = {
            "schema": "gl1f-live-chain-witness/v2-extended",
            "chainId": 29,
            "block": {"number": 42, "hash": block_hash},
            "models": [
                {
                    "tokenId": 1,
                    "active": True,
                    "modelId": model_id,
                    "tablePtr": table_address,
                    "chunkSize": split,
                    "numChunks": 2,
                    "totalBytes": len(core),
                    "conformanceStudy": {"vectors": [source_vector]},
                }
            ],
        }
        source_bytes = (json.dumps(source_witness, indent=2) + "\n").encode()
        (root / "source-witness.json").write_bytes(source_bytes)

        def rpc_entry(
            request_id: int,
            method: str,
            params: list[object],
            result: object,
        ) -> dict[str, object]:
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            return {
                "sequence": request_id,
                "attempt": 1,
                "context": f"synthetic {method}",
                "httpStatus": 200,
                "requestBody": json.dumps(request, separators=(",", ":")),
                "responseBody": json.dumps(response, separators=(",", ":")),
            }

        transcript_entries = [
            rpc_entry(1, "eth_chainId", [], "0x1d"),
            rpc_entry(2, "eth_getBlockByNumber", ["0x2a", True], block),
            rpc_entry(
                3,
                "eth_call",
                [{"to": runtime_address, "data": calldata_hex}, "0x2a"],
                result_hex,
            ),
            rpc_entry(4, "eth_getBlockByNumber", ["0x2a", False], block),
        ]
        transcript_plain = (
            "\n".join(json.dumps(entry) for entry in transcript_entries) + "\n"
        ).encode()
        (root / "transcript.ndjson.gz").write_bytes(
            gzip.compress(transcript_plain, mtime=0)
        )

        archive_vector = {
            "vectorId": "equality-left-and-signed",
            "packedFeaturesQHex": "0x" + packed.hex(),
            "packedBytes": len(packed),
            "featuresQ": [5, 0],
            "expectedOutputQ": ["2"],
            "localPredictionQ": ["2"],
            "chainPredictionQ": ["2"],
            "sourceWitnessChainPredictionQ": ["2"],
            "status": "compared",
            "exactMatch": True,
            "archiveChainCall": {
                "transcriptFile": "transcript.ndjson.gz",
                "transcriptSequence": 3,
                "requestId": 3,
                "responseId": 3,
                "batchIndex": 0,
                "rpcMethod": "eth_call",
                "contractMethod": "predictView",
                "to": runtime_address,
                "blockTag": "0x2a",
                "calldataBytes": len(calldata),
                "calldataKeccak256": "0x" + keccak256(calldata).hex(),
                "rawResultHex": result_hex,
                "decodedOutputQ": ["2"],
                "matchesLocalPrediction": True,
                "matchesEmbeddedSourceWitness": True,
            },
        }
        manifest = {
            "schemaVersion": 3,
            "archiveId": "synthetic-test-archive",
            "collector": {
                "script": "benchmarks/archive_live_chain_state.mjs",
                "archivedSourceFile": "collector.mjs",
                "archivedSourceBytes": len(collector_source),
                "archivedSourceSha256": hashlib.sha256(
                    collector_source
                ).hexdigest(),
                "repositoryRevision": "ab" * 20,
                "repositoryWorktreeState": "clean synthetic fixture",
            },
            "chain": {
                "chainId": 29,
                "blockNumber": 42,
                "blockTag": "0x2a",
                "blockHash": block_hash,
                "blockTimestamp": 123,
                "blockStateRoot": state_root,
                "fullBlockFile": "block.json",
            },
            "readOnlyRpc": {
                "methodsUsed": [
                    "eth_call",
                    "eth_chainId",
                    "eth_getBlockByNumber",
                ],
                "methodRequestCounts": {
                    "eth_call": 1,
                    "eth_chainId": 1,
                    "eth_getBlockByNumber": 2,
                },
                "prohibitedAndUnusedMethods": [
                    "eth_sendRawTransaction",
                    "eth_sendTransaction",
                ],
                "rawTranscript": {
                    "file": "transcript.ndjson.gz",
                    "requestAttempts": 4,
                },
            },
            "sourceWitness": {
                "file": "source-witness.json",
                "schema": "gl1f-live-chain-witness/v2-extended",
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            },
            "contracts": contracts,
            "registrySnapshot": {
                "modelNFT": _address(0x52),
                "modelNFTMatchesDeploymentAddress": True,
            },
            "proofEvidence": {
                "available": False,
                "file": None,
                "accountProofs": 0,
                "codeHashesMatched": 0,
            },
            "models": [
                {
                    "tokenId": 1,
                    "modelId": model_id,
                    "computedModelId": model_id,
                    "tablePtr": table_address,
                    "chunkPointers": list(chunk_addresses),
                    "chunkSize": split,
                    "numChunks": 2,
                    "totalBytes": len(core),
                    "storageRuntimeFile": "storage-runtime.bin",
                    "core": {
                        "file": "model.gl1f",
                        "size": len(core),
                        "sha256": hashlib.sha256(core).hexdigest(),
                        "keccak256": model_id,
                    },
                    "header": {
                        "version": 1,
                        "nFeatures": 2,
                        "depth": 1,
                        "totalTrees": 2,
                        "treesPerOutput": 2,
                        "outputs": 1,
                        "baseQ": [10],
                        "scaleQ": 100,
                        "coreBytes": len(core),
                    },
                    "registry": {
                        "modelId": model_id,
                        "tablePtr": table_address,
                        "chunkSize": split,
                        "numChunks": 2,
                        "totalBytes": len(core),
                        "nFeatures": 2,
                        "nTrees": 2,
                        "depth": 1,
                        "baseQ": 10,
                        "scaleQ": 100,
                    },
                    "registryHeaderAgreement": {
                        "nFeatures": True,
                        "nTrees": True,
                        "depth": True,
                        "baseQ": True,
                        "scaleQ": True,
                        "tablePtr": True,
                        "totalBytes": True,
                        "numChunks": True,
                        "chunkSize": True,
                    },
                    "tables": [
                        {
                            "index": 0,
                            "address": table_address,
                            "pointerCount": 2,
                            "runtime": {
                                "file": "storage-runtime.bin",
                                "offset": 0,
                                "size": len(table_code),
                                "sha256": hashlib.sha256(table_code).hexdigest(),
                                "keccak256": "0x" + keccak256(table_code).hex(),
                            },
                        }
                    ],
                    "chunks": [
                        {
                            "index": index,
                            "address": chunk_addresses[index],
                            "payloadSize": len(payloads[index]),
                            "runtime": {
                                "file": "storage-runtime.bin",
                                "offset": len(table_code)
                                + sum(len(value) for value in chunk_codes[:index]),
                                "size": len(chunk_codes[index]),
                                "sha256": hashlib.sha256(
                                    chunk_codes[index]
                                ).hexdigest(),
                                "keccak256": "0x"
                                + keccak256(chunk_codes[index]).hex(),
                            },
                        }
                        for index in range(2)
                    ],
                    "vectors": [archive_vector],
                }
            ],
        }
        inventory: dict[str, dict[str, object]] = {}
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.name == "manifest.json":
                continue
            raw = path.read_bytes()
            inventory[path.name] = {
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        manifest["files"] = inventory
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest_path

    def test_complete_archive_verifies_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = verify_manifest(self._make_archive(Path(temporary)))
        self.assertEqual(report.models_checked, 1)
        self.assertEqual(report.vectors_checked, 1)
        self.assertEqual(report.core_bytes_checked, len(_v1_core()))
        self.assertEqual(report.files_checked, 11)

    def test_collector_source_binding_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_archive(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["collector"]["archivedSourceSha256"] = "00" * 32
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                VerificationError, "archivedSourceSha256"
            ):
                verify_manifest(manifest_path)

    def test_registry_topology_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_archive(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["registrySnapshot"]["modelNFT"] = _address(0x77)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "NFT address"):
                verify_manifest(manifest_path)

        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_archive(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["registrySnapshot"][
                "modelNFTMatchesDeploymentAddress"
            ] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                VerificationError, "independently checked relation"
            ):
                verify_manifest(manifest_path)

    def test_wrong_archived_prediction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_archive(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["models"][0]["vectors"][0]["chainPredictionQ"] = ["3"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                VerificationError, "independent evaluation"
            ):
                verify_manifest(manifest_path)

    def test_chunk_order_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_archive(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["models"][0]["chunks"].reverse()
            for index, chunk in enumerate(manifest["models"][0]["chunks"]):
                chunk["index"] = index
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                VerificationError, "payload|table pointers"
            ):
                verify_manifest(manifest_path)

    def test_raw_registry_values_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_archive(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["models"][0]["registry"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "raw registry"):
                verify_manifest(manifest_path)

    def test_storage_profile_bounds_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_archive(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["models"][0]["chunkSize"] = 3
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "storage profile"):
                verify_manifest(manifest_path)

        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_archive(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["models"][0]["numChunks"] = 768
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "pointer-table profile"):
                verify_manifest(manifest_path)

    def test_embedded_source_witness_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._make_archive(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_path = root / "source-witness.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            vector = source["models"][0]["conformanceStudy"]["vectors"][0]
            vector["localPredictionQ"] = ["3"]
            vector["evmRead"]["outputQ"] = ["3"]
            source_bytes = (json.dumps(source, indent=2) + "\n").encode()
            source_path.write_bytes(source_bytes)
            source_sha = hashlib.sha256(source_bytes).hexdigest()
            manifest["sourceWitness"]["sha256"] = source_sha
            manifest["files"]["source-witness.json"] = {
                "size": len(source_bytes),
                "sha256": source_sha,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                VerificationError, "embedded source witness differ"
            ):
                verify_manifest(manifest_path)

    def test_raw_transcript_result_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._make_archive(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            transcript_path = root / "transcript.ndjson.gz"
            lines = gzip.decompress(transcript_path.read_bytes()).decode().splitlines()
            entry = json.loads(lines[2])
            response = json.loads(entry["responseBody"])
            response["result"] = "0x" + (3).to_bytes(32, "big").hex()
            entry["responseBody"] = json.dumps(response, separators=(",", ":"))
            lines[2] = json.dumps(entry)
            transcript_bytes = gzip.compress(
                ("\n".join(lines) + "\n").encode(), mtime=0
            )
            transcript_path.write_bytes(transcript_bytes)
            manifest["files"]["transcript.ndjson.gz"] = {
                "size": len(transcript_bytes),
                "sha256": hashlib.sha256(transcript_bytes).hexdigest(),
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                VerificationError, "ABI-decoded output"
            ):
                verify_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
