#!/usr/bin/env python3
"""Focused tests for fail-closed, resumable mint filesystem state."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from mint_workflow import (
    MINT_STATE_SCHEMA,
    MODEL_STORE_MAGIC,
    OwnerKeyError,
    WorkflowStateError,
    acquire_owner_key,
    bind_owner_key,
    empty_mint_state,
    load_mint_state,
    obtain_registration_receipt,
    safe_rpc_endpoint_identity,
    save_json_artifact_exclusive,
    save_mint_state,
    validate_resume_chain_state,
    validate_mint_state,
)


MODEL_ID = "0x" + "11" * 32
OTHER_MODEL_ID = "0x" + "22" * 32
OWNER_ADDRESS = "0x" + "33" * 20
OTHER_OWNER_ADDRESS = "0x" + "44" * 20
POINTER = "0x" + "55" * 20
TABLE_POINTER = "0x" + "66" * 20
CHUNK_TX_HASH = "0x" + "77" * 32
REGISTER_TX_HASH = "0x" + "88" * 32
REGISTRY_ADDRESS = "0x" + "99" * 20
SIGNER_ADDRESS = "0x" + "bb" * 20
STORE_ADDRESS = "0x" + "cc" * 20
PRIVATE_KEY = "0x" + "aa" * 32


class _HexKey:
    def __init__(self, value: str = PRIVATE_KEY) -> None:
        self.value = value

    def hex(self) -> str:
        return self.value


class _FakeAccount:
    def __init__(
        self,
        address: str = OWNER_ADDRESS,
        private_key: str = PRIVATE_KEY,
    ) -> None:
        self.address = address
        self.key = _HexKey(private_key)


def _chunk(index: int = 0, *, model_id: str = MODEL_ID) -> dict[str, object]:
    return {
        "index": index,
        "pointer": POINTER,
        "tx_hash": CHUNK_TX_HASH,
        "model_id": model_id,
    }


class MintOwnerKeyTests(unittest.TestCase):
    def test_nonresume_dry_run_generates_nothing_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "owner_key.txt")
            calls: list[str] = []

            def create_account() -> _FakeAccount:
                calls.append("create")
                return _FakeAccount()

            def account_from_key(_key: str) -> _FakeAccount:
                calls.append("load")
                return _FakeAccount()

            material = acquire_owner_key(
                path,
                expected_model_id=MODEL_ID,
                title="Dry run",
                generated_at="2026-08-27 00:00:00",
                resume=False,
                dry_run=True,
                create_account=create_account,
                account_from_key=account_from_key,
            )

            self.assertIsNone(material)
            self.assertEqual(calls, [])
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_key_is_created_exclusively_at_mode_0600_and_safely_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "owner_key.txt")
            generated = acquire_owner_key(
                path,
                expected_model_id=MODEL_ID,
                title="Release model",
                generated_at="2026-08-27 00:00:00",
                resume=False,
                dry_run=False,
                create_account=_FakeAccount,
                account_from_key=lambda _key: _FakeAccount(),
            )
            self.assertIsNotNone(generated)
            assert generated is not None
            self.assertTrue(generated.created)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            original = path.read_bytes()
            self.assertIn(f"# modelId:    {MODEL_ID}".encode(), original)

            with self.assertRaisesRegex(OwnerKeyError, "refusing to overwrite"):
                acquire_owner_key(
                    path,
                    expected_model_id=MODEL_ID,
                    title="Replacement",
                    generated_at="later",
                    resume=False,
                    dry_run=False,
                    create_account=lambda: _FakeAccount(OTHER_OWNER_ADDRESS),
                    account_from_key=lambda _key: _FakeAccount(),
                )
            self.assertEqual(path.read_bytes(), original)

            create_calls: list[str] = []
            reused = acquire_owner_key(
                path,
                expected_model_id=MODEL_ID,
                title="Ignored on resume",
                generated_at="later",
                resume=True,
                dry_run=False,
                create_account=lambda: create_calls.append("create"),
                account_from_key=lambda key: _FakeAccount(
                    OWNER_ADDRESS, private_key=key
                ),
            )
            self.assertIsNotNone(reused)
            assert reused is not None
            self.assertFalse(reused.created)
            self.assertEqual(create_calls, [])
            self.assertEqual(reused.address.lower(), OWNER_ADDRESS)
            self.assertEqual(path.read_bytes(), original)

    def test_resume_rejects_wrong_key_model_address_and_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "owner_key.txt")
            acquire_owner_key(
                path,
                expected_model_id=MODEL_ID,
                title="Release model",
                generated_at="now",
                resume=False,
                dry_run=False,
                create_account=_FakeAccount,
                account_from_key=lambda _key: _FakeAccount(),
            )

            common = {
                "path": path,
                "title": "ignored",
                "generated_at": "ignored",
                "resume": True,
                "dry_run": False,
                "create_account": lambda: self.fail("resume generated a key"),
            }
            with self.assertRaisesRegex(OwnerKeyError, "modelId does not match"):
                acquire_owner_key(
                    expected_model_id=OTHER_MODEL_ID,
                    account_from_key=lambda _key: _FakeAccount(),
                    **common,
                )
            with self.assertRaisesRegex(OwnerKeyError, "does not match"):
                acquire_owner_key(
                    expected_model_id=MODEL_ID,
                    account_from_key=lambda _key: _FakeAccount(OTHER_OWNER_ADDRESS),
                    **common,
                )

            path.chmod(0o644)
            with self.assertRaisesRegex(OwnerKeyError, "permissions are too broad"):
                acquire_owner_key(
                    expected_model_id=MODEL_ID,
                    account_from_key=lambda _key: _FakeAccount(),
                    **common,
                )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_resume_rejects_owner_key_symlink_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real-owner-key.txt"
            acquire_owner_key(
                target,
                expected_model_id=MODEL_ID,
                title="Release model",
                generated_at="now",
                resume=False,
                dry_run=False,
                create_account=_FakeAccount,
                account_from_key=lambda _key: _FakeAccount(),
            )
            link = root / "owner_key.txt"
            link.symlink_to(target)

            with self.assertRaisesRegex(OwnerKeyError, "not a regular file"):
                acquire_owner_key(
                    link,
                    expected_model_id=MODEL_ID,
                    title="ignored",
                    generated_at="ignored",
                    resume=True,
                    dry_run=False,
                    create_account=lambda: self.fail("resume generated a key"),
                    account_from_key=lambda _key: _FakeAccount(),
                )

    def test_state_owner_binding_cannot_be_replaced(self) -> None:
        state = bind_owner_key(empty_mint_state(MODEL_ID), OWNER_ADDRESS)
        self.assertEqual(state["owner_key_address"], OWNER_ADDRESS)
        with self.assertRaisesRegex(WorkflowStateError, "does not match"):
            bind_owner_key(state, OTHER_OWNER_ADDRESS)


class MintStateTests(unittest.TestCase):
    def test_state_round_trip_is_model_bound_atomic_and_mode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "mint_state.json")
            state = empty_mint_state(MODEL_ID)
            state["completed_chunks"] = [_chunk()]
            state["table_ptr"] = TABLE_POINTER
            state["register_tx_hash"] = REGISTER_TX_HASH

            normalized = save_mint_state(
                path,
                state,
                expected_model_id=MODEL_ID,
                n_chunks=1,
            )
            loaded = load_mint_state(
                path,
                expected_model_id=MODEL_ID,
                n_chunks=1,
            )

            self.assertEqual(loaded, normalized)
            self.assertEqual(loaded["schema"], MINT_STATE_SCHEMA)
            self.assertEqual(loaded["model_id"], MODEL_ID)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(
                [item.name for item in Path(temporary).iterdir()],
                ["mint_state.json"],
            )

    def test_missing_corrupt_and_unbound_legacy_state_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "mint_state.json")
            with self.assertRaisesRegex(WorkflowStateError, "does not exist"):
                load_mint_state(
                    path,
                    expected_model_id=MODEL_ID,
                    n_chunks=1,
                )

            corrupt = b'{"completed_chunks": ['
            path.write_bytes(corrupt)
            with self.assertRaisesRegex(WorkflowStateError, "cannot parse"):
                load_mint_state(
                    path,
                    expected_model_id=MODEL_ID,
                    n_chunks=1,
                )
            self.assertEqual(path.read_bytes(), corrupt)

            path.write_text(
                json.dumps(
                    {
                        "completed_chunks": [],
                        "table_ptr": None,
                        "register_tx_hash": None,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(WorkflowStateError, "not bound"):
                load_mint_state(
                    path,
                    expected_model_id=MODEL_ID,
                    n_chunks=1,
                )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_resume_rejects_state_symlink_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real-state.json"
            save_mint_state(
                target,
                empty_mint_state(MODEL_ID),
                expected_model_id=MODEL_ID,
                n_chunks=1,
            )
            link = root / "mint_state.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(WorkflowStateError, "not a regular file"):
                load_mint_state(
                    link,
                    expected_model_id=MODEL_ID,
                    n_chunks=1,
                )

    def test_legacy_state_is_accepted_only_when_a_chunk_binds_the_model(self) -> None:
        legacy = {
            "completed_chunks": [_chunk()],
            "table_ptr": POINTER,
            "register_tx_hash": None,
        }
        normalized = validate_mint_state(
            legacy,
            expected_model_id=MODEL_ID,
            n_chunks=1,
        )
        self.assertEqual(normalized["schema"], MINT_STATE_SCHEMA)
        self.assertEqual(normalized["model_id"], MODEL_ID)

    def test_inconsistent_state_is_rejected(self) -> None:
        base = empty_mint_state(MODEL_ID)
        cases: dict[str, tuple[dict[str, object], str, int]] = {
            "wrong model": (
                {**base, "model_id": OTHER_MODEL_ID},
                "does not match",
                1,
            ),
            "missing model": (
                {**base, "model_id": None},
                "missing model_id",
                1,
            ),
            "noncontiguous chunks": (
                {**base, "completed_chunks": [_chunk(index=1)]},
                "contiguous",
                2,
            ),
            "wrong chunk model": (
                {**base, "completed_chunks": [_chunk(model_id=OTHER_MODEL_ID)]},
                "does not match",
                1,
            ),
            "premature table": (
                {**base, "table_ptr": TABLE_POINTER},
                "before every chunk",
                1,
            ),
            "zero owner": (
                {**base, "owner_key_address": "0x" + "00" * 20},
                "zero address",
                1,
            ),
            "register without table": (
                {**base, "register_tx_hash": REGISTER_TX_HASH},
                "without table_ptr",
                1,
            ),
        }
        for name, (state, message, n_chunks) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                WorkflowStateError, message
            ):
                validate_mint_state(
                    state,
                    expected_model_id=MODEL_ID,
                    n_chunks=n_chunks,
                )


class MintRegistrationResumeTests(unittest.TestCase):
    def _receipt(self, tx_hash: object = REGISTER_TX_HASH) -> dict[str, object]:
        return {
            "transactionHash": tx_hash,
            "status": 1,
            "to": REGISTRY_ADDRESS,
            "from": SIGNER_ADDRESS,
        }

    def test_recorded_registration_is_recovered_without_resubmission(self) -> None:
        state = {"register_tx_hash": REGISTER_TX_HASH}
        calls: list[tuple[str, str | None]] = []

        recovered = obtain_registration_receipt(
            state,
            fetch_receipt=lambda tx_hash: (
                calls.append(("fetch", tx_hash)) or self._receipt(tx_hash)
            ),
            submit_registration=lambda: (
                calls.append(("submit", None)) or self._receipt()
            ),
            expected_registry=REGISTRY_ADDRESS,
            expected_sender=SIGNER_ADDRESS,
        )

        self.assertTrue(recovered.recovered)
        self.assertEqual(recovered.transaction_hash, REGISTER_TX_HASH)
        self.assertEqual(calls, [("fetch", REGISTER_TX_HASH)])

    def test_unavailable_recorded_receipt_refuses_to_resubmit(self) -> None:
        submit_calls: list[str] = []
        with self.assertRaisesRegex(WorkflowStateError, "refusing to resubmit"):
            obtain_registration_receipt(
                {"register_tx_hash": REGISTER_TX_HASH},
                fetch_receipt=lambda _tx_hash: None,
                submit_registration=lambda: submit_calls.append("submit"),
                expected_registry=REGISTRY_ADDRESS,
                expected_sender=SIGNER_ADDRESS,
            )
        self.assertEqual(submit_calls, [])

    def test_new_registration_submits_once_and_accepts_unprefixed_hash_object(self) -> None:
        class HashWithoutPrefix:
            def hex(self) -> str:
                return REGISTER_TX_HASH[2:]

        submit_calls: list[str] = []
        submitted = obtain_registration_receipt(
            {"register_tx_hash": None},
            fetch_receipt=lambda _tx_hash: self.fail("new tx was fetched"),
            submit_registration=lambda: (
                submit_calls.append("submit")
                or self._receipt(HashWithoutPrefix())
            ),
            expected_registry=REGISTRY_ADDRESS,
            expected_sender=SIGNER_ADDRESS,
        )
        self.assertFalse(submitted.recovered)
        self.assertEqual(submitted.transaction_hash, REGISTER_TX_HASH)
        self.assertEqual(submit_calls, ["submit"])

    def test_receipt_must_be_successful_and_target_the_registry(self) -> None:
        for name, receipt, message in (
            (
                "reverted",
                {**self._receipt(), "status": 0},
                "did not succeed",
            ),
            (
                "wrong target",
                {**self._receipt(), "to": OTHER_OWNER_ADDRESS},
                "configured registry",
            ),
            (
                "wrong sender",
                {**self._receipt(), "from": OTHER_OWNER_ADDRESS},
                "configured signer",
            ),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                WorkflowStateError, message
            ):
                obtain_registration_receipt(
                    {"register_tx_hash": REGISTER_TX_HASH},
                    fetch_receipt=lambda _tx_hash, value=receipt: value,
                    submit_registration=lambda: self.fail("resume resubmitted"),
                    expected_registry=REGISTRY_ADDRESS,
                    expected_sender=SIGNER_ADDRESS,
                )


class MintResumeChainStateTests(unittest.TestCase):
    def _state_and_chain(self) -> tuple[dict[str, object], bytes, dict[str, bytes], dict[str, object]]:
        core = b"exact selected GL1F core bytes"
        state = empty_mint_state(MODEL_ID)
        state["completed_chunks"] = [_chunk()]
        state["table_ptr"] = TABLE_POINTER
        table_payload = b"\0" * 12 + bytes.fromhex(POINTER[2:])
        code = {
            POINTER: MODEL_STORE_MAGIC + core,
            TABLE_POINTER: MODEL_STORE_MAGIC + table_payload,
        }
        receipt = {
            "transactionHash": CHUNK_TX_HASH,
            "status": 1,
            "to": STORE_ADDRESS,
            "from": SIGNER_ADDRESS,
            "pointer": POINTER,
        }
        return state, core, code, receipt

    def _validate(
        self,
        state: dict[str, object],
        core: bytes,
        code: dict[str, bytes],
        receipt: dict[str, object] | None,
    ) -> None:
        validate_resume_chain_state(
            state,
            core=core,
            chunk_size=len(core),
            fetch_code=lambda pointer: code[pointer],
            fetch_receipt=lambda _tx_hash: receipt,
            pointer_from_receipt=lambda value: str(value["pointer"]),
            expected_store=STORE_ADDRESS,
            expected_sender=SIGNER_ADDRESS,
        )

    def test_completed_chunk_receipt_code_and_table_are_verified(self) -> None:
        state, core, code, receipt = self._state_and_chain()
        self._validate(state, core, code, receipt)

    def test_tampered_or_stale_resume_chain_state_fails_closed(self) -> None:
        for name in ("chunk bytes", "table bytes", "receipt pointer", "receipt missing"):
            state, core, code, receipt = self._state_and_chain()
            if name == "chunk bytes":
                code[POINTER] = MODEL_STORE_MAGIC + b"wrong core"
            elif name == "table bytes":
                code[TABLE_POINTER] = MODEL_STORE_MAGIC + b"wrong table"
            elif name == "receipt pointer":
                assert receipt is not None
                receipt["pointer"] = OTHER_OWNER_ADDRESS
            else:
                receipt = None
            with self.subTest(name=name), self.assertRaises(WorkflowStateError):
                self._validate(state, core, code, receipt)


class MintArtifactTests(unittest.TestCase):
    def test_artifact_is_published_exclusively_and_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "S1.json")
            save_json_artifact_exclusive(path, {"model_id": MODEL_ID})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"model_id": MODEL_ID},
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            with self.assertRaisesRegex(WorkflowStateError, "refusing to overwrite"):
                save_json_artifact_exclusive(path, {"model_id": OTHER_MODEL_ID})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"model_id": MODEL_ID},
            )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_artifact_symlink_target_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("sentinel", encoding="utf-8")
            link = root / "S1.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(WorkflowStateError, "refusing to overwrite"):
                save_json_artifact_exclusive(link, {"secret": "overwrite"})
            self.assertEqual(target.read_text(encoding="utf-8"), "sentinel")


class MintModelIntegrationGuards(unittest.TestCase):
    def test_rpc_artifact_identity_discards_credentials_and_provider_tokens(self) -> None:
        raw = (
            "https://alice:USERINFO_SECRET@RPC.Example.org:8545/"
            "v3/PROVIDER_TOKEN?api_key=QUERY_SECRET#FRAGMENT_SECRET"
        )
        identity = safe_rpc_endpoint_identity(raw)
        self.assertEqual(identity, "https://rpc.example.org:8545")
        for secret in (
            "alice",
            "USERINFO_SECRET",
            "PROVIDER_TOKEN",
            "QUERY_SECRET",
            "FRAGMENT_SECRET",
        ):
            self.assertNotIn(secret, identity)
        self.assertEqual(
            safe_rpc_endpoint_identity("https://rpc.genesisl1.org/"),
            "https://rpc.genesisl1.org",
        )
        self.assertEqual(
            safe_rpc_endpoint_identity("http://[::1]:8545/private/token"),
            "http://[::1]:8545",
        )
        self.assertEqual(
            safe_rpc_endpoint_identity("not-a-url?secret=do-not-copy"),
            "[redacted RPC endpoint]",
        )

    def test_cli_uses_guarded_workflow_and_has_no_direct_key_write(self) -> None:
        source = Path("mint_model.py").read_text(encoding="utf-8")
        self.assertNotIn("owner_path.write_text", source)
        self.assertLess(source.index("load_mint_state("), source.index("Account.create"))
        self.assertIn("obtain_registration_receipt(", source)
        self.assertIn("submit_registration=submit_registration", source)
        self.assertIn('"rpc":                  rpc_identity', source)
        self.assertNotIn('"rpc":                  rpc_url', source)
        self.assertNotIn('cannot reach RPC {rpc_url}', source)
        self.assertNotIn('node says: {e}', source)
        self.assertNotIn('wait failed: {e}', source)
        self.assertNotIn('registry model identity: {exc}', source)
        self.assertIn("def _main() -> int:", source)
        self.assertIn("provider errors can contain RPC credentials", source)
        self.assertNotIn("Path(STATE_FILE).resolve()", source)
        self.assertNotIn("Path(OWNER_KEY_FILE).resolve()", source)
        self.assertNotIn("Path(ARTIFACTS_FILE).resolve()", source)
        self.assertIn("output_directory / STATE_FILE", source)
        self.assertIn("output_directory / ARTIFACTS_FILE", source)
        self.assertIn("artifact_path.exists() or artifact_path.is_symlink()", source)
        self.assertLess(
            source.index("validate_resume_chain_state("),
            source.index("Account.create"),
        )
        self.assertIn("save_json_artifact_exclusive(artifact_path, artifact)", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
