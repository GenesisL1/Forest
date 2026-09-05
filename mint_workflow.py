"""Fail-closed state and owner-key handling for ``mint_model.py``.

This module intentionally uses only the Python standard library.  The command
line publisher supplies the account factory/converter from ``eth-account``;
keeping the filesystem state machine dependency-free makes the release-critical
resume behavior directly testable in the ordinary publication suite.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit


MINT_STATE_SCHEMA = "gl1f-mint-state/v2"
MODEL_STORE_MAGIC = b"GL1C"


class WorkflowStateError(ValueError):
    """A mint state or recovered transaction is unsafe to resume."""


class OwnerKeyError(ValueError):
    """An owner-key file cannot be created or safely reused."""


@dataclass(frozen=True)
class OwnerKeyMaterial:
    account: Any
    address: str
    private_key: str
    created: bool


@dataclass(frozen=True)
class RegistrationReceipt:
    transaction_hash: str
    receipt: Any
    recovered: bool


def safe_rpc_endpoint_identity(rpc_url: Any) -> str:
    """Return an origin-only RPC identity safe to persist in artifacts.

    Authentication userinfo, paths (which commonly contain provider tokens),
    queries, and fragments are deliberately discarded.  Malformed or
    non-HTTP endpoint strings are replaced wholesale so an error path cannot
    accidentally copy a credential into ``S1.json``.
    """
    try:
        parsed = urlsplit(str(rpc_url or ""))
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except (TypeError, ValueError):
        return "[redacted RPC endpoint]"
    if scheme not in {"http", "https"} or not hostname:
        return "[redacted RPC endpoint]"
    if any(character.isspace() or ord(character) < 32 for character in hostname):
        return "[redacted RPC endpoint]"
    if any(character in "/@?#" for character in hostname):
        return "[redacted RPC endpoint]"
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    rendered_port = "" if port is None else f":{port}"
    return f"{scheme}://{rendered_host}{rendered_port}"


def save_json_artifact_exclusive(path: Path, payload: Any) -> None:
    """Atomically publish a JSON artifact without following or replacing a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), 0o644)
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise WorkflowStateError(
                f"refusing to overwrite existing artifact path: {path}"
            ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _hex_value(value: Any, nbytes: int, label: str) -> str:
    text = str(value or "")
    if not re.fullmatch(rf"0x[0-9a-fA-F]{{{2 * nbytes}}}", text):
        raise WorkflowStateError(f"{label} must be a 0x-prefixed {nbytes}-byte hex value")
    return text.lower()


def _address(value: Any, label: str) -> str:
    normalized = _hex_value(value, 20, label)
    if normalized == "0x" + "00" * 20:
        raise WorkflowStateError(f"{label} must not be the zero address")
    return normalized


def _model_id(value: Any, label: str = "model_id") -> str:
    return _hex_value(value, 32, label)


def _transaction_hash(value: Any, label: str) -> str:
    normalized = _hex_value(value, 32, label)
    if normalized == "0x" + "00" * 32:
        raise WorkflowStateError(f"{label} must not be zero")
    return normalized


def _receipt_transaction_hash(value: Any, label: str) -> str:
    if hasattr(value, "hex"):
        value = value.hex()
    text = str(value or "")
    if re.fullmatch(r"[0-9a-fA-F]{64}", text):
        text = "0x" + text
    return _transaction_hash(text, label)


def empty_mint_state(model_id: str) -> dict[str, Any]:
    """Return a new state object already bound to one model commitment."""
    return {
        "schema": MINT_STATE_SCHEMA,
        "model_id": _model_id(model_id),
        "owner_key_address": None,
        "completed_chunks": [],
        "table_ptr": None,
        "register_tx_hash": None,
    }


def validate_mint_state(
    state: Any,
    *,
    expected_model_id: str,
    n_chunks: int,
) -> dict[str, Any]:
    """Validate and normalize a current or legacy mint-state document.

    Legacy v0.2.2 state did not have a top-level schema/model identifier.  It
    can be resumed only when at least one completed chunk independently binds
    the file to ``expected_model_id``.  Empty unbound legacy files fail closed.
    """
    expected = _model_id(expected_model_id, "expected model_id")
    if not isinstance(n_chunks, int) or isinstance(n_chunks, bool) or n_chunks < 1:
        raise WorkflowStateError("n_chunks must be a positive integer")
    if not isinstance(state, dict):
        raise WorkflowStateError("mint state root must be a JSON object")

    schema = state.get("schema")
    if schema not in (None, MINT_STATE_SCHEMA):
        raise WorkflowStateError(f"unsupported mint state schema: {schema!r}")
    completed = state.get("completed_chunks")
    if not isinstance(completed, list):
        raise WorkflowStateError("completed_chunks must be a JSON array")
    if len(completed) > n_chunks:
        raise WorkflowStateError(
            f"state has {len(completed)} completed chunks but model needs {n_chunks}"
        )

    normalized_chunks: list[dict[str, Any]] = []
    chunk_model_ids: set[str] = set()
    for expected_index, record in enumerate(completed):
        if not isinstance(record, dict):
            raise WorkflowStateError(f"completed chunk {expected_index} is not an object")
        index = record.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index != expected_index:
            raise WorkflowStateError(
                "completed chunk indices must be contiguous and start at zero "
                f"(expected {expected_index}, got {index!r})"
            )
        pointer = _address(record.get("pointer"), f"completed chunk {index} pointer")
        tx_hash = _transaction_hash(
            record.get("tx_hash"), f"completed chunk {index} transaction hash"
        )
        chunk_model_id = _model_id(
            record.get("model_id"), f"completed chunk {index} model_id"
        )
        chunk_model_ids.add(chunk_model_id)
        normalized_chunks.append(
            {
                **record,
                "index": index,
                "pointer": pointer,
                "tx_hash": tx_hash,
                "model_id": chunk_model_id,
            }
        )

    if len(chunk_model_ids) > 1:
        raise WorkflowStateError("completed chunks refer to different model IDs")

    state_model_id = state.get("model_id")
    if schema == MINT_STATE_SCHEMA and state_model_id is None:
        raise WorkflowStateError("current mint state is missing model_id")
    if state_model_id is None:
        if not chunk_model_ids:
            raise WorkflowStateError(
                "legacy mint state is not bound to a model; refuse to resume"
            )
        normalized_model_id = next(iter(chunk_model_ids))
    else:
        normalized_model_id = _model_id(state_model_id)
    if normalized_model_id != expected or any(mid != expected for mid in chunk_model_ids):
        raise WorkflowStateError("mint state modelId does not match the selected .gl1f")

    table_ptr_raw = state.get("table_ptr")
    table_ptr = None if table_ptr_raw is None else _address(table_ptr_raw, "table_ptr")
    register_raw = state.get("register_tx_hash")
    register_tx_hash = (
        None
        if register_raw is None
        else _transaction_hash(register_raw, "register_tx_hash")
    )
    owner_raw = state.get("owner_key_address")
    owner_key_address = (
        None if owner_raw is None else _address(owner_raw, "owner_key_address")
    )

    if table_ptr is not None and len(normalized_chunks) != n_chunks:
        raise WorkflowStateError("table_ptr is present before every chunk is recorded")
    if register_tx_hash is not None and table_ptr is None:
        raise WorkflowStateError("register_tx_hash is present without table_ptr")

    return {
        **state,
        "schema": MINT_STATE_SCHEMA,
        "model_id": expected,
        "owner_key_address": owner_key_address,
        "completed_chunks": normalized_chunks,
        "table_ptr": table_ptr,
        "register_tx_hash": register_tx_hash,
    }


def load_mint_state(
    path: Path,
    *,
    expected_model_id: str,
    n_chunks: int,
) -> dict[str, Any]:
    """Load a required resume file; missing/corrupt state is never reset."""
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise WorkflowStateError(f"resume state file does not exist: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise WorkflowStateError(f"resume state is not a regular file: {path}")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowStateError(f"cannot parse resume state {path}: {exc}") from exc
    return validate_mint_state(
        state,
        expected_model_id=expected_model_id,
        n_chunks=n_chunks,
    )


def save_mint_state(
    path: Path,
    state: Any,
    *,
    expected_model_id: str,
    n_chunks: int,
) -> dict[str, Any]:
    """Validate and atomically save mode-0600 workflow state."""
    normalized = validate_mint_state(
        state,
        expected_model_id=expected_model_id,
        n_chunks=n_chunks,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(normalized, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            os.chmod(path, 0o600, follow_symlinks=False)
        except (NotImplementedError, TypeError):
            os.chmod(path, 0o600)
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return normalized


def _private_key(value: Any) -> str:
    text = str(value or "")
    if not text.startswith("0x"):
        text = "0x" + text
    if not re.fullmatch(r"0x[0-9a-fA-F]{64}", text):
        raise OwnerKeyError("OWNER_PRIVATE_KEY must be a 32-byte hexadecimal key")
    return text.lower()


def _owner_address(value: Any) -> str:
    try:
        return _address(value, "OWNER_ADDRESS")
    except WorkflowStateError as exc:
        raise OwnerKeyError(str(exc)) from exc


def _read_owner_key(path: Path, *, expected_model_id: str) -> tuple[str, str]:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise OwnerKeyError(f"owner key file does not exist: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise OwnerKeyError(f"owner key path is not a regular file: {path}")
    if os.name == "posix" and info.st_mode & 0o077:
        raise OwnerKeyError(
            f"owner key permissions are too broad ({oct(info.st_mode & 0o777)}); "
            f"run chmod 600 {path} before resuming"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OwnerKeyError(f"cannot read owner key file {path}: {exc}") from exc

    fields: dict[str, str] = {}
    file_model_id: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("# modelid:"):
            if file_model_id is not None:
                raise OwnerKeyError("owner key file contains duplicate modelId records")
            file_model_id = line.split(":", 1)[1].strip()
        elif line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            name = name.strip()
            if name in fields:
                raise OwnerKeyError(f"owner key file contains duplicate {name}")
            fields[name] = value.strip()

    try:
        normalized_file_model_id = _model_id(file_model_id, "owner key modelId")
        normalized_expected = _model_id(expected_model_id, "expected modelId")
    except WorkflowStateError as exc:
        raise OwnerKeyError(str(exc)) from exc
    if normalized_file_model_id != normalized_expected:
        raise OwnerKeyError("owner key file modelId does not match the selected .gl1f")
    return _owner_address(fields.get("OWNER_ADDRESS")), _private_key(
        fields.get("OWNER_PRIVATE_KEY")
    )


def _write_owner_key_exclusive(
    path: Path,
    *,
    title: str,
    model_id: str,
    address: str,
    private_key: str,
    generated_at: str,
) -> None:
    normalized_model_id = _model_id(model_id)
    normalized_address = _owner_address(address)
    normalized_private_key = _private_key(private_key)
    content = (
        "# Forest Model NFT owner API access keypair\n"
        f"# Generated:  {generated_at}\n"
        f"# Title:      {title}\n"
        f"# modelId:    {normalized_model_id}\n"
        "#\n"
        f"OWNER_ADDRESS={normalized_address}\n"
        f"OWNER_PRIVATE_KEY={normalized_private_key}\n"
    )

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise OwnerKeyError(
            f"refusing to overwrite existing owner key file: {path}; "
            "use --resume to reuse it or move it aside deliberately"
        ) from exc
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def acquire_owner_key(
    path: Path,
    *,
    expected_model_id: str,
    title: str,
    generated_at: str,
    resume: bool,
    dry_run: bool,
    create_account: Callable[[], Any],
    account_from_key: Callable[[str], Any],
) -> OwnerKeyMaterial | None:
    """Create once or safely reuse the owner key.

    A non-resume dry run deliberately returns before account generation or any
    filesystem write.  Resume dry runs may read and validate an existing key.
    """
    if resume:
        address, private_key = _read_owner_key(
            path, expected_model_id=expected_model_id
        )
        try:
            account = account_from_key(private_key)
        except Exception as exc:
            raise OwnerKeyError(
                f"cannot load owner private key ({type(exc).__name__})"
            ) from exc
        account_address = _owner_address(getattr(account, "address", None))
        if account_address != address:
            raise OwnerKeyError("OWNER_ADDRESS does not match OWNER_PRIVATE_KEY")
        return OwnerKeyMaterial(account, getattr(account, "address"), private_key, False)

    if dry_run:
        return None

    try:
        account = create_account()
    except Exception as exc:
        raise OwnerKeyError(
            f"cannot generate owner account ({type(exc).__name__})"
        ) from exc
    address = _owner_address(getattr(account, "address", None))
    key = getattr(account, "key", None)
    if hasattr(key, "hex"):
        key = key.hex()
    private_key = _private_key(key)
    _write_owner_key_exclusive(
        path,
        title=title,
        model_id=expected_model_id,
        address=getattr(account, "address"),
        private_key=private_key,
        generated_at=generated_at,
    )
    return OwnerKeyMaterial(account, getattr(account, "address"), private_key, True)


def bind_owner_key(state: dict[str, Any], address: str) -> dict[str, Any]:
    """Bind one owner-key address to state without permitting replacement."""
    normalized = _address(address, "owner key address")
    existing = state.get("owner_key_address")
    if existing is not None and _address(existing, "state owner_key_address") != normalized:
        raise WorkflowStateError("resume owner key does not match mint state")
    return {**state, "owner_key_address": normalized}


def validate_resume_chain_state(
    state: dict[str, Any],
    *,
    core: bytes,
    chunk_size: int,
    fetch_code: Callable[[str], Any],
    fetch_receipt: Callable[[str], Any],
    pointer_from_receipt: Callable[[Any], str],
    expected_store: str,
    expected_sender: str,
) -> None:
    """Verify every skipped write against immutable ModelStore runtime bytes."""
    if not isinstance(core, bytes) or not core:
        raise WorkflowStateError("resume verification requires non-empty core bytes")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise WorkflowStateError("resume verification requires a positive chunk size")
    store = _address(expected_store, "expected ModelStore")
    sender = _address(expected_sender, "expected chunk sender")
    completed = state.get("completed_chunks")
    if not isinstance(completed, list):
        raise WorkflowStateError("completed_chunks must be validated before chain checks")

    for record in completed:
        index = record["index"]
        pointer = _address(record["pointer"], f"completed chunk {index} pointer")
        tx_hash = _transaction_hash(
            record["tx_hash"], f"completed chunk {index} transaction hash"
        )
        expected_payload = core[index * chunk_size : (index + 1) * chunk_size]
        try:
            deployed_code = bytes(fetch_code(pointer))
        except Exception as exc:
            raise WorkflowStateError(
                f"cannot verify deployed code for completed chunk {index}"
            ) from exc
        if deployed_code != MODEL_STORE_MAGIC + expected_payload:
            raise WorkflowStateError(
                f"completed chunk {index} pointer does not contain the selected model bytes"
            )

        try:
            receipt = fetch_receipt(tx_hash)
        except Exception as exc:
            raise WorkflowStateError(
                f"cannot verify transaction receipt for completed chunk {index}"
            ) from exc
        if receipt is None:
            raise WorkflowStateError(
                f"transaction receipt is missing for completed chunk {index}"
            )
        status = receipt.get("status")
        if not isinstance(status, int) or isinstance(status, bool) or status != 1:
            raise WorkflowStateError(
                f"completed chunk {index} transaction did not succeed"
            )
        if _address(receipt.get("to"), "chunk receipt target") != store:
            raise WorkflowStateError(
                f"completed chunk {index} transaction target is not ModelStore"
            )
        if _address(receipt.get("from"), "chunk receipt sender") != sender:
            raise WorkflowStateError(
                f"completed chunk {index} transaction sender is not the configured signer"
            )
        receipt_hash = _receipt_transaction_hash(
            receipt.get("transactionHash"), "chunk receipt transaction hash"
        )
        if receipt_hash != tx_hash:
            raise WorkflowStateError(
                f"completed chunk {index} receipt hash does not match mint state"
            )
        try:
            receipt_pointer = _address(
                pointer_from_receipt(receipt), "ChunkWritten pointer"
            )
        except Exception as exc:
            raise WorkflowStateError(
                f"cannot verify ChunkWritten event for completed chunk {index}"
            ) from exc
        if receipt_pointer != pointer:
            raise WorkflowStateError(
                f"completed chunk {index} receipt pointer does not match mint state"
            )

    table_pointer_raw = state.get("table_ptr")
    if table_pointer_raw is not None:
        table_pointer = _address(table_pointer_raw, "table_ptr")
        table_payload = b"".join(
            b"\0" * 12 + bytes.fromhex(str(record["pointer"])[2:])
            for record in completed
        )
        try:
            table_code = bytes(fetch_code(table_pointer))
        except Exception as exc:
            raise WorkflowStateError("cannot verify deployed pointer table") from exc
        if table_code != MODEL_STORE_MAGIC + table_payload:
            raise WorkflowStateError(
                "recorded table_ptr does not contain the reconstructed pointer table"
            )


def obtain_registration_receipt(
    state: dict[str, Any],
    *,
    fetch_receipt: Callable[[str], Any],
    submit_registration: Callable[[], Any],
    expected_registry: str,
    expected_sender: str,
) -> RegistrationReceipt:
    """Recover a recorded receipt or submit exactly once when none exists."""
    expected_to = _address(expected_registry, "expected registry")
    expected_from = _address(expected_sender, "expected registration sender")
    existing_hash = state.get("register_tx_hash")
    recovered = existing_hash is not None
    if recovered:
        transaction_hash = _transaction_hash(existing_hash, "register_tx_hash")
        try:
            receipt = fetch_receipt(transaction_hash)
        except Exception as exc:
            raise WorkflowStateError(
                "recorded registration receipt is unavailable; refusing to resubmit"
            ) from exc
        if receipt is None:
            raise WorkflowStateError(
                "recorded registration receipt is unavailable; refusing to resubmit"
            )
    else:
        receipt = submit_registration()
        transaction_hash = _receipt_transaction_hash(
            receipt.get("transactionHash"),
            "submitted registration transaction hash",
        )

    status = receipt.get("status")
    if not isinstance(status, int) or isinstance(status, bool) or status != 1:
        raise WorkflowStateError(
            f"registration transaction {transaction_hash} did not succeed (status={status!r})"
        )
    receipt_to = receipt.get("to")
    if receipt_to is None or _address(receipt_to, "registration receipt to") != expected_to:
        raise WorkflowStateError("registration receipt target is not the configured registry")
    receipt_from = receipt.get("from")
    if (
        receipt_from is None
        or _address(receipt_from, "registration receipt sender") != expected_from
    ):
        raise WorkflowStateError("registration receipt sender is not the configured signer")
    return RegistrationReceipt(transaction_hash, receipt, recovered)
