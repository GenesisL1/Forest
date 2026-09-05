#!/usr/bin/env python3
"""
mint_model.py — Mint a Forest Model NFT on GenesisL1 from the command line.

This script replicates the behavior of the Forest studio's Mint tab
(create.html → src/create_page.js, deployBtn handler at lines 5761–5950)
without any browser, MetaMask, or per-chunk popup. It signs every
transaction with a private key from a .env file and deploys all model
chunks sequentially, one transaction per block, exactly the same way the
UI does — just without 600 manual MetaMask confirmations.

Mirrors UI flow exactly:
  1. Load .gl1f, strip optional GL1X JSON footer (UI: parseGl1fPackage)
  2. modelId = keccak256(core_bytes)              (UI: applyTrainedModel)
  3. Read deployFeeWei, sizeFeeWeiPerByte,
     requiredDeployFeeWei, activeLicenseId, tosVersion from registry
  4. For each chunk of CHUNK_SIZE = 24000 bytes:
       store.write(chunk) → wait → parse ChunkWritten → record pointer
  5. Build 32 bytes/pointer table, store.write(table) → tablePtr
  6. registry.registerModel(... payable requiredFeeWei ...)

Inputs (interactive prompts):
  - Title (≥3 chars; must produce ≥1 word ≥2 chars)
  - Description (≥8 chars)
  - Icon path (must validate as 128×128 PNG)
  - Pricing mode (free / tips / paid) and feeWei
  - License + ToS acceptance (must type the exact phrase, not Y/N)

Inputs (CLI flags):
  --gl1f                path to .gl1f file (required)
  --env                 path to .env file (default: ./.env)
  --resume              load mint_state.json and skip completed chunks
  --dry-run             read everything, build all data, do NOT send any tx
  --pricing-mode        0 (free, default) / 1 (tips) / 2 (paid required)
  --pricing-fee-eth     L1 fee for inference, used when mode != 0 (default 0.001)
  --pricing-recipient   address to receive inference fees (default: signer)

  Footer-aware behavior:
  When the .gl1f file contains a GL1X JSON footer (the Python and C++
  trainers in this repo write one by default; only --no-package suppresses
  it), the script auto-extracts the on-chain featuresPacked string from
  the footer's pkg.nft.featuresPacked field. This includes task type,
  feature names, label name, and (for classification) class labels --
  byte-identical to what the trainer constructed.

  If the footer also contains pkg.nft.title / description / iconPngB64
  values, those are offered as defaults during the interactive prompts
  (the user can press Enter to accept or type a new value to override).

  The following flags are only consulted when the .gl1f has NO footer
  or its featuresPacked field is missing/malformed:

  --task                regression | binary_classification |
                        multiclass_classification | multilabel_classification
  --label-name          name of the target column (default 'target')
  --feature-names-file  newline-separated file of feature names; count
                        must match the model's declared nFeatures

.env file (place in cwd or pass --env):
  PRIVATE_KEY=0x...                          # wallet with L1 to pay gas + deploy fee
  GAS_PRICE_GWEI=1                           # gas price for every transaction
  RPC_URL=https://rpc.genesisl1.org          # optional; default if not set

Outputs:
  - mint_state.json   resumable per-chunk state + pointer addresses
  - owner_key.txt     freshly-generated owner API keypair (SAVE THIS)
  - S1.json           machine-readable deployment record

Dependencies:
  python -m pip install -r requirements-mint.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Optional

from gl1f_validate import (
    FormatError,
    parse_gl1f_package,
    validate_deployed_registry_profile,
)
from mint_workflow import (
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
)

try:
    from web3 import Web3
    from web3.exceptions import TransactionNotFound, ContractLogicError
    from web3.logs import DISCARD
    from eth_account import Account
    from eth_utils import keccak, to_checksum_address, to_bytes
    from dotenv import dotenv_values
    from PIL import Image
except ImportError as e:
    print(f"\n[fatal] Missing dependency: {e}\n", file=sys.stderr)
    print("Install with:\n  python -m pip install -r requirements-mint.txt\n",
          file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Constants — copied verbatim from src/common.js loadSystem() defaults
# and src/create_page.js CHUNK_SIZE.
# ============================================================================

DEFAULTS = {
    "rpc":      "https://rpc.genesisl1.org",
    "chain_id": 29,
    "store":    "0x9CdbC23392648Bd27B4A5eD09c0fEa9452454B54",
    "registry": "0x33c9844F77a07e36B98f0FFf8201B8A8b02c2a69",
    "nft":      "0x44Dc1c54B8D579B42d78cC21cf8260DC0A3279fA",
    "runtime":  "0xD2fD0cf461a6cb56Fc08d9aEc120833D8E79044E",
    "market":   "0xA9bfa0a719b7F73cE85CA1E7f23af626D383fB46",
}

# CHUNK_SIZE = 24000 (src/create_page.js line 38). ModelStore enforces ≤ 24572.
CHUNK_SIZE = 24000

# Per-tx limits matching the UI
GAS_LIMIT_CHUNK_WRITE  = 30_000_000
GAS_LIMIT_TABLE_WRITE  = 30_000_000
GAS_LIMIT_REGISTER     = 35_000_000

# Retry policy
TX_TIMEOUT_SECONDS     = 90       # if not confirmed in 90 s, resubmit
TX_POLL_INTERVAL       = 2.0      # seconds between receipt checks
TX_MAX_RESUBMITS       = 5        # per chunk
TX_GAS_BUMP_FACTOR     = 1.25     # bump gasPrice on resubmit

# Files written in cwd
STATE_FILE     = "mint_state.json"
OWNER_KEY_FILE = "owner_key.txt"
ARTIFACTS_FILE = "S1.json"


# ============================================================================
# Minimal ABIs — the only functions/events we actually call.
# These are subsets of src/abis.js, kept as JSON for web3.py.
# ============================================================================

ABI_STORE = [
    {
        "type": "function", "name": "write", "stateMutability": "nonpayable",
        "inputs":  [{"type": "bytes",   "name": "data"}],
        "outputs": [{"type": "address", "name": "pointer"}],
    },
    {
        "type": "event", "name": "ChunkWritten", "anonymous": False,
        "inputs": [
            {"type": "address", "name": "pointer", "indexed": True},
            {"type": "uint256", "name": "size",    "indexed": False},
        ],
    },
]

ABI_REGISTRY = [
    {"type": "function", "name": "deployFeeWei",         "stateMutability": "view",
     "inputs": [], "outputs": [{"type": "uint256"}]},
    {"type": "function", "name": "sizeFeeWeiPerByte",    "stateMutability": "view",
     "inputs": [], "outputs": [{"type": "uint256"}]},
    {"type": "function", "name": "requiredDeployFeeWei", "stateMutability": "view",
     "inputs":  [{"type": "uint32", "name": "totalBytes"}],
     "outputs": [{"type": "uint256"}]},
    {"type": "function", "name": "activeLicenseId",      "stateMutability": "view",
     "inputs": [], "outputs": [{"type": "uint256"}]},
    {"type": "function", "name": "tosVersion",           "stateMutability": "view",
     "inputs": [], "outputs": [{"type": "uint256"}]},
    {"type": "function", "name": "getLicense",           "stateMutability": "view",
     "inputs":  [{"type": "uint256", "name": "id"}],
     "outputs": [{"type": "string", "name": "name"}, {"type": "string", "name": "url"}]},
    # registerModel — full signature copied from src/abis.js ABI_REGISTRY.
    {
        "type": "function", "name": "registerModel", "stateMutability": "payable",
        "inputs": [
            {"type": "bytes32",   "name": "modelId"},
            {"type": "address",   "name": "tablePtr"},
            {"type": "uint32",    "name": "chunkSize"},
            {"type": "uint32",    "name": "numChunks"},
            {"type": "uint32",    "name": "totalBytes"},
            {"type": "uint16",    "name": "nFeatures"},
            {"type": "uint16",    "name": "nTrees"},
            {"type": "uint16",    "name": "depth"},
            {"type": "int32",     "name": "baseQ"},
            {"type": "uint32",    "name": "scaleQ"},
            {"type": "string",    "name": "title"},
            {"type": "string",    "name": "description"},
            {"type": "bytes",     "name": "iconPng32"},
            {"type": "string",    "name": "featuresPacked"},
            {"type": "bytes32[]", "name": "titleWordHashes"},
            {"type": "uint8",     "name": "pricingMode"},
            {"type": "uint256",   "name": "feeWei"},
            {"type": "address",   "name": "recipient"},
            {"type": "uint32",    "name": "tosVersionAccepted"},
            {"type": "uint32",    "name": "licenseIdAccepted"},
            {"type": "address",   "name": "ownerKey"},
        ],
        "outputs": [],
    },
    {
        "type": "function", "name": "getModelSummary", "stateMutability": "view",
        "inputs":  [{"type": "uint256", "name": "tokenId"}],
        "outputs": [
            {"type": "bool"},    {"type": "bytes32"}, {"type": "address"},
            {"type": "uint16"},  {"type": "uint16"},  {"type": "uint16"},
            {"type": "int32"},   {"type": "uint8"},   {"type": "uint256"},
            {"type": "address"}, {"type": "bool"},    {"type": "address"},
            {"type": "uint32"},  {"type": "string"},  {"type": "string"},
        ],
    },
]

# ERC-721 Transfer event — used to recover the freshly-minted tokenId from the
# registerModel receipt (the UI doesn't show it, but we want it for S1).
ABI_NFT_TRANSFER_EVENT = [{
    "type": "event", "name": "Transfer", "anonymous": False,
    "inputs": [
        {"type": "address", "name": "from",    "indexed": True},
        {"type": "address", "name": "to",      "indexed": True},
        {"type": "uint256", "name": "tokenId", "indexed": True},
    ],
}]


def unpack_nft_features_string(packed: str) -> tuple[Optional[dict], list[str]]:
    """
    Parse a featuresPacked string back into (meta, featureNames).
    Mirrors src/common.js unpackNftFeatures. Used to extract task/label info
    from a footer's pkg.nft.featuresPacked when we want to validate against
    the model's nFeatures.
    """
    raw = str(packed or "")
    lines = [s.strip() for s in raw.split("\n") if s and s.strip()]
    meta = None
    start = 0
    if lines and lines[0].startswith("#meta="):
        try:
            meta = json.loads(lines[0][6:])
            start = 1
        except json.JSONDecodeError:
            meta = None
            start = 0
    return meta, lines[start:]


# ============================================================================
# Helpers — UI parity
# ============================================================================

def title_word_hashes(title: str) -> list[bytes]:
    """
    Mirror src/create_page.js titleWordHashes(title):
      - lowercase
      - split on any whitespace or comma
      - drop empty / single-char words
      - dedup preserving order
      - keccak256(utf8(word))
    """
    import re
    words = [w for w in re.split(r"[\s,]+", title.lower())
             if len(w.strip()) >= 2]
    seen, ordered = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            ordered.append(w)
    return [keccak(text=w) for w in ordered]


def pack_nft_features(task: str, feature_names: list[str],
                      label_name: Optional[str] = None,
                      labels:      Optional[list[str]] = None,
                      label_names: Optional[list[str]] = None) -> str:
    """
    Mirror src/common.js packNftFeatures.
    Output format:
        #meta={"v":1,"task":"<task>"[,"labelName":...][,"labels":[...]]}
        feat1
        feat2
        ...
    """
    raw = (task or "regression").strip()
    if raw in ("binary_classification", "binary"):
        t = "binary_classification"
    elif raw in ("multiclass_classification", "multiclass"):
        t = "multiclass_classification"
    elif raw in ("multilabel_classification", "multilabel"):
        t = "multilabel_classification"
    else:
        t = "regression"
    meta: dict = {"v": 1, "task": t}
    if label_name:
        meta["labelName"] = str(label_name)

    if t == "multilabel_classification":
        if label_names and len(label_names) >= 1:
            meta["labelNames"] = [str(x) for x in label_names]
        elif labels and len(labels) >= 1:
            meta["labelNames"] = [str(x) for x in labels]
        if labels and len(labels) >= 2:
            meta["labels"] = [str(labels[0]), str(labels[1])]
        else:
            meta["labels"] = ["0", "1"]
    elif t in ("binary_classification", "multiclass_classification"):
        if labels and len(labels) >= 2:
            meta["labels"] = [str(x) for x in labels]

    lines = ["#meta=" + json.dumps(meta, separators=(",", ":"))]
    for f in (feature_names or []):
        s = str(f or "").strip()
        if s:
            lines.append(s)
    return "\n".join(lines)


def validate_icon_128(path: Path) -> bytes:
    """
    Mirror src/create_page.js validateIcon128: must be PNG with valid signature
    and exactly 128×128 pixels. Returns raw bytes for on-chain storage.
    """
    if not path.exists():
        raise FileNotFoundError(f"Icon file not found: {path}")
    raw = path.read_bytes()
    sig = bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    if raw[:8] != sig:
        raise ValueError("Invalid PNG signature")
    img = Image.open(path)
    if img.format != "PNG":
        raise ValueError(f"Icon must be PNG (got {img.format})")
    if img.size != (128, 128):
        raise ValueError(f"Icon must be 128×128 (got {img.size[0]}×{img.size[1]})")
    return raw


# ============================================================================
# Logging
# ============================================================================

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str = "") -> None:
    if msg:
        print(f"[{now_ts()}] {msg}", flush=True)
    else:
        print(flush=True)


def log_err(msg: str) -> None:
    print(f"[{now_ts()}] [error] {msg}", file=sys.stderr, flush=True)


# ============================================================================
# Transaction helpers
# ============================================================================

def build_legacy_tx(w3: Web3, account, *, to: Optional[str], data: bytes,
                    gas: int, gas_price_wei: int, value: int = 0) -> dict:
    """
    Build a legacy (pre-EIP-1559) transaction. Forest's chain config in
    src/eth.js doesn't specify EIP-1559 fields, and `gasPrice` is universally
    accepted, so we use legacy form.
    """
    nonce = w3.eth.get_transaction_count(account.address, "pending")
    tx: dict = {
        "from":     account.address,
        "nonce":    nonce,
        "gas":      gas,
        "gasPrice": gas_price_wei,
        "value":    value,
        "data":     data,
        "chainId":  DEFAULTS["chain_id"],
    }
    if to is not None:
        tx["to"] = to_checksum_address(to)
    return tx


def send_with_retry(w3: Web3, account, *, build_tx_fn,
                    tag: str, gas_price_wei: int) -> dict:
    """
    Send a transaction and wait for receipt. On timeout, resubmit with the
    same nonce and a bumped gasPrice (replaces the stuck transaction).
    On revert (status != 1), abort. On network error, poll first to see if
    the tx made it to mempool before resending.

    Returns the receipt (a dict-like AttributeDict with status=1).
    """
    current_gas_price = gas_price_wei
    last_tx_hash:   Optional[str] = None
    last_nonce:     Optional[int] = None

    for attempt in range(1, TX_MAX_RESUBMITS + 2):
        # Build the tx. On a retry we deliberately reuse the same nonce so the
        # network treats this as a replacement (mempool rule: same {from,nonce}
        # with higher gasPrice replaces the earlier one).
        try:
            tx = build_tx_fn(current_gas_price)
            if last_nonce is not None:
                tx["nonce"] = last_nonce
            else:
                last_nonce = tx["nonce"]

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            last_tx_hash = tx_hash.hex()
            log(f"  [{tag}] tx.hash {last_tx_hash}  "
                f"(nonce={last_nonce} gasPrice={current_gas_price} attempt={attempt})")
        except ValueError as exc:
            # Common: "already known" / "replacement transaction underpriced" /
            # "nonce too low". The first means our previous send went through;
            # the second means our bump wasn't big enough; the third means the
            # tx already mined. Try to recover by polling for the previous
            # hash, otherwise rebuild with a higher bump.
            msg = str(exc).lower()
            if "already known" in msg or "nonce too low" in msg:
                log(f"  [{tag}] node reports a known/consumed nonce; polling prior tx")
                if last_tx_hash:
                    try:
                        rcpt = w3.eth.wait_for_transaction_receipt(
                            last_tx_hash, timeout=TX_TIMEOUT_SECONDS,
                            poll_latency=TX_POLL_INTERVAL)
                        return _check_receipt(rcpt, tag)
                    except Exception:
                        pass
                # Bump and retry
                current_gas_price = int(current_gas_price * TX_GAS_BUMP_FACTOR)
                continue
            if "replacement transaction underpriced" in msg or "underpriced" in msg:
                log(f"  [{tag}] underpriced replacement; bumping gas")
                current_gas_price = max(
                    int(current_gas_price * TX_GAS_BUMP_FACTOR),
                    current_gas_price + 1,
                )
                continue
            raise

        # Wait for the receipt with timeout
        try:
            rcpt = w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=TX_TIMEOUT_SECONDS,
                poll_latency=TX_POLL_INTERVAL)
            return _check_receipt(rcpt, tag)
        except Exception as exc:
            # Timeout, network drop, etc. Poll once for the receipt directly
            # (handles flaky RPCs that miss the wait window).
            log(
                f"  [{tag}] receipt wait failed ({type(exc).__name__}); "
                "polling once before resubmit"
            )
            try:
                rcpt = w3.eth.get_transaction_receipt(tx_hash)
                if rcpt is not None:
                    return _check_receipt(rcpt, tag)
            except (TransactionNotFound, Exception):
                pass

            if attempt > TX_MAX_RESUBMITS:
                raise RuntimeError(
                    f"[{tag}] giving up after {TX_MAX_RESUBMITS} resubmits; "
                    f"last hash {last_tx_hash}") from exc
            current_gas_price = int(current_gas_price * TX_GAS_BUMP_FACTOR)
            log(f"  [{tag}] timeout; resubmitting at gasPrice={current_gas_price}")

    raise RuntimeError(f"[{tag}] unreachable: exhausted retries")


def _check_receipt(rcpt, tag: str):
    status = rcpt.get("status", None) if isinstance(rcpt, dict) else rcpt.status
    gas_used = rcpt.get("gasUsed", None) if isinstance(rcpt, dict) else rcpt.gasUsed
    log(f"  [{tag}] mined status={status} gasUsed={gas_used}")
    if status != 1:
        raise RuntimeError(f"[{tag}] transaction reverted (status=0)")
    return rcpt


def parse_chunk_written(w3: Web3, store_contract, rcpt) -> str:
    """
    Find ChunkWritten event in receipt logs and return the pointer address.
    """
    logs = store_contract.events.ChunkWritten().process_receipt(
        rcpt, errors=DISCARD)
    if not logs:
        raise RuntimeError("ChunkWritten event not found in receipt")
    return to_checksum_address(logs[0]["args"]["pointer"])


def parse_minted_token_id(w3: Web3, nft_address: str, rcpt) -> Optional[int]:
    """
    Find an ERC-721 Transfer(from=0x0, to=signer, tokenId) emitted by the NFT
    contract during registerModel. Returns the new tokenId, or None if not
    found.
    """
    nft = w3.eth.contract(address=to_checksum_address(nft_address),
                           abi=ABI_NFT_TRANSFER_EVENT)
    try:
        evs = nft.events.Transfer().process_receipt(rcpt, errors=DISCARD)
    except Exception:
        return None
    for ev in evs:
        if int(ev["args"]["from"], 16) == 0:
            return int(ev["args"]["tokenId"])
    return None


# ============================================================================
# Interactive prompts
# ============================================================================

def prompt(message: str, *, validator=None, password: bool = False) -> str:
    while True:
        try:
            if password:
                import getpass
                value = getpass.getpass(message).strip()
            else:
                value = input(message).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            log_err("aborted by user")
            sys.exit(1)
        if validator is None:
            return value
        ok, err = validator(value)
        if ok:
            return value
        print(f"  {err}")


def confirm(message: str, *, exact_phrase: Optional[str] = None) -> bool:
    """
    If exact_phrase is given, user must type it verbatim.
    Otherwise, accept y / yes / n / no.
    """
    while True:
        try:
            v = input(message).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if exact_phrase is not None:
            return v == exact_phrase
        if v.lower() in ("y", "yes"):
            return True
        if v.lower() in ("n", "no", ""):
            return False
        print("  please answer yes or no")


# ============================================================================
# Main flow
# ============================================================================

def _main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gl1f", type=Path, required=True,
                        help="path to .gl1f file")
    parser.add_argument("--env", type=Path, default=Path(".env"),
                        help="path to env file with PRIVATE_KEY, "
                             "GAS_PRICE_GWEI, and (optional) RPC_URL "
                             "(default: ./.env)")
    parser.add_argument("--rpc", type=str, default=None,
                        help="RPC endpoint. Overrides RPC_URL from env if "
                             "given; falls back to env, then to "
                             f"{DEFAULTS['rpc']}")
    parser.add_argument("--store-addr",    type=str, default=DEFAULTS["store"])
    parser.add_argument("--registry-addr", type=str, default=DEFAULTS["registry"])
    parser.add_argument("--nft-addr",      type=str, default=DEFAULTS["nft"])
    parser.add_argument("--task", type=str, default="regression",
                        choices=("regression", "binary_classification",
                                 "multiclass_classification",
                                 "multilabel_classification"),
                        help="task type. ONLY consulted if the .gl1f has no "
                             "GL1X footer; otherwise the footer's task is used "
                             "(default: regression)")
    parser.add_argument("--label-name", type=str, default="target",
                        help="label-column name embedded in the on-chain "
                             "featuresPacked metadata. ONLY consulted if the "
                             ".gl1f has no GL1X footer (default: 'target')")
    parser.add_argument("--feature-names-file", type=Path, default=None,
                        help="newline-separated file of feature names. ONLY "
                             "consulted if the .gl1f has no GL1X footer; "
                             "otherwise the footer's feature names are used. "
                             "If neither is available, falls back to "
                             "'feat_0..feat_{n-1}' placeholders.")
    parser.add_argument("--pricing-mode", type=int, default=0, choices=(0, 1, 2),
                        help="0=free (default), 1=tips, 2=paid required")
    parser.add_argument("--pricing-fee-eth", type=str, default="0.001",
                        help="L1 fee per inference call, used when "
                             "pricing-mode != 0 (default 0.001)")
    parser.add_argument("--pricing-recipient", type=str, default=None,
                        help="address to receive inference fees "
                             "(default: signer address)")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted mint from mint_state.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate everything; do NOT send any tx")
    args = parser.parse_args()

    # ------------------------------------------------------------------ env
    if not args.env.exists():
        log_err(f"env file not found: {args.env}")
        log_err("expected lines:  PRIVATE_KEY=0x...   GAS_PRICE_GWEI=...   RPC_URL=...")
        return 1
    env = dotenv_values(args.env)
    pk = env.get("PRIVATE_KEY", "").strip()
    gp = env.get("GAS_PRICE_GWEI", "").strip()
    rpc_env = env.get("RPC_URL", "").strip()
    if not pk:
        log_err("PRIVATE_KEY missing from env")
        return 1
    if not gp:
        log_err("GAS_PRICE_GWEI missing from env")
        return 1
    if not pk.startswith("0x"):
        pk = "0x" + pk
    try:
        gas_price_wei = int(float(gp) * 10**9)
    except ValueError:
        log_err(f"GAS_PRICE_GWEI not numeric: {gp!r}")
        return 1

    # RPC resolution priority: --rpc flag > RPC_URL in env > DEFAULTS["rpc"]
    if args.rpc:
        rpc_url = args.rpc
        rpc_source = "--rpc flag"
    elif rpc_env:
        rpc_url = rpc_env
        rpc_source = f"RPC_URL in {args.env}"
    else:
        rpc_url = DEFAULTS["rpc"]
        rpc_source = "default"

    # ------------------------------------------------------------------ web3
    rpc_identity = safe_rpc_endpoint_identity(rpc_url)
    log(f"Connecting to {rpc_identity}  (source: {rpc_source})")
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        log_err(f"cannot reach RPC {rpc_identity}")
        return 1
    chain_id = w3.eth.chain_id
    if chain_id != DEFAULTS["chain_id"]:
        log_err(f"wrong chain: connected to chainId={chain_id}, "
                f"expected {DEFAULTS['chain_id']} (GenesisL1)")
        return 1

    account = Account.from_key(pk)
    bal_wei = w3.eth.get_balance(account.address)
    log(f"Signer:      {account.address}")
    log(f"Balance:     {Web3.from_wei(bal_wei, 'ether')} L1")
    log(f"Gas price:   {gp} gwei  ({gas_price_wei} wei)")
    log(f"Chain id:    {chain_id}")

    # ------------------------------------------------------------------ load .gl1f
    if not args.gl1f.exists():
        log_err(f".gl1f file not found: {args.gl1f}")
        return 1
    raw = args.gl1f.read_bytes()
    try:
        package = parse_gl1f_package(raw)
        n_chunks = validate_deployed_registry_profile(
            package,
            chunk_size=CHUNK_SIZE,
        )
    except FormatError as exc:
        log_err(f"invalid or non-canonical .gl1f package: {exc}")
        return 1

    core = package.core
    core_len = len(core)
    has_footer = package.footer is not None
    model_id = keccak(core)
    model_id_hex = "0x" + model_id.hex()
    header = package.header

    # Load and validate resumable state before any owner-key generation or
    # transaction work.  A normal run refuses to trample a prior workflow;
    # --resume requires an existing, model-bound, internally consistent file.
    # Resolve only the output directory.  Resolving the complete path would
    # dereference a hostile final-component symlink before no-follow checks.
    output_directory = Path.cwd().resolve()
    state_path = output_directory / STATE_FILE
    artifact_path = output_directory / ARTIFACTS_FILE
    try:
        if args.resume:
            state = load_mint_state(
                state_path,
                expected_model_id=model_id_hex,
                n_chunks=n_chunks,
            )
        else:
            if state_path.exists() or state_path.is_symlink():
                raise WorkflowStateError(
                    f"mint state already exists: {state_path}; use --resume or "
                    "move it aside deliberately"
                )
            state = empty_mint_state(model_id_hex)
    except WorkflowStateError as exc:
        log_err(str(exc))
        return 1
    if not args.dry_run and (artifact_path.exists() or artifact_path.is_symlink()):
        log_err(
            f"artifact path already exists: {artifact_path}; move it aside "
            "deliberately before minting"
        )
        return 1

    def persist_state() -> None:
        nonlocal state
        state = save_mint_state(
            state_path,
            state,
            expected_model_id=model_id_hex,
            n_chunks=n_chunks,
        )

    log()
    log(f"Loaded .gl1f:        {args.gl1f}  ({len(raw):,} bytes total)")
    log(f"Core model bytes:    {core_len:,}  (footer present: {has_footer})")
    log(f"modelId (keccak256): {model_id_hex}")
    log(f"GL1F version:        {header.version}")
    log(f"nFeatures:           {header.n_features}")
    log(f"depth:               {header.depth}")
    log(f"nTrees (registry):    {header.registry_n_trees}")
    if header.version == 2:
        log(f"trees/output:         {header.trees_per_output}")
    log(f"nOutputs:             {header.n_outputs}")
    log(f"baseQ (registry):     {header.registry_base_q}")
    log(f"scaleQ:              {header.scale_q}")

    # ------------------------------------------------------------------ extract metadata from GL1X footer
    # Mirrors the UI's "Load .gl1f" path (src/create_page.js _loadGl1fFile,
    # lines 1180-1250): when the footer exists, we use its featuresPacked
    # string verbatim so the on-chain bytes are byte-identical to what the
    # trainer constructed. We also surface footer-stored title / description /
    # icon as prompt defaults that the user can accept or override.
    pkg = package.footer
    footer_features_packed: Optional[str] = None
    footer_task: Optional[str] = None
    footer_feature_names: Optional[list[str]] = None
    footer_label_name: Optional[str] = None
    footer_title: Optional[str] = None
    footer_description: Optional[str] = None

    if pkg is not None:
        nft_obj = pkg.get("nft") or {}
        local_obj = pkg.get("local") or {}
        train_meta = local_obj.get("trainMeta") or {}

        fp_raw = nft_obj.get("featuresPacked")
        if isinstance(fp_raw, str) and fp_raw.strip():
            footer_features_packed = fp_raw
            meta, names = unpack_nft_features_string(fp_raw)
            if meta and isinstance(meta.get("task"), str):
                footer_task = meta["task"]
            if meta and isinstance(meta.get("labelName"), str):
                footer_label_name = meta["labelName"]
            if names:
                footer_feature_names = names

        # Fallback: trainMeta.task if featuresPacked didn't carry it
        if footer_task is None and isinstance(train_meta.get("task"), str):
            footer_task = train_meta["task"]

        if isinstance(nft_obj.get("title"), str) and nft_obj["title"].strip():
            footer_title = nft_obj["title"].strip()
        if isinstance(nft_obj.get("description"), str) and nft_obj["description"].strip():
            footer_description = nft_obj["description"].strip()

        log()
        log("GL1X footer detected — extracted metadata:")
        log(f"  task:           {footer_task or '(missing)'}")
        log(f"  feature count:  {len(footer_feature_names) if footer_feature_names else 0}"
            f"  (model declares {header.n_features})")
        log(f"  label name:     {footer_label_name or '(missing)'}")
        log(f"  title:          {footer_title or '(missing — will prompt)'}")
        log(f"  description:    "
            f"{(footer_description[:60] + '…') if footer_description and len(footer_description) > 60 else (footer_description or '(missing — will prompt)')}")

        # Sanity check: footer feature count should match model header
        if footer_feature_names is not None and len(footer_feature_names) != header.n_features:
            log_err(f"footer featuresPacked has {len(footer_feature_names)} feature "
                    f"names but model header declares {header.n_features}; refusing "
                    f"to proceed with mismatched metadata")
            return 1

    # ------------------------------------------------------------------ resolve metadata sources
    # Priority for each field:
    #   featuresPacked  : footer -> rebuilt from CLI flags
    #   feature_names   : footer -> --feature-names-file -> placeholders
    #   task            : footer -> --task
    #   label_name      : footer -> --label-name
    if footer_features_packed:
        features_packed = footer_features_packed
        task = footer_task or args.task
        feature_names = footer_feature_names or [f"feat_{i}" for i in range(header.n_features)]
        label_name = footer_label_name or args.label_name
        log()
        log(f"Using on-chain featuresPacked from .gl1f footer "
            f"(task={task}, nFeatures={len(feature_names)})")
    else:
        # No footer (or footer had no featuresPacked) — fall back to CLI flags.
        if args.feature_names_file is not None:
            if not args.feature_names_file.exists():
                log_err(f"feature names file not found: {args.feature_names_file}")
                return 1
            feature_names = [
                line.strip() for line in args.feature_names_file.read_text().splitlines()
                if line.strip()
            ]
            if len(feature_names) != header.n_features:
                log_err(f"feature count mismatch: file has {len(feature_names)}, "
                        f"model declares {header.n_features}")
                return 1
        else:
            feature_names = [f"feat_{i}" for i in range(header.n_features)]
            log()
            log("Note: .gl1f has no GL1X footer and --feature-names-file not given;")
            log(f"      using placeholder names feat_0..feat_{header.n_features-1}.")
            log( "      These will be permanently embedded in the on-chain NFT.")
        task = args.task
        label_name = args.label_name
        features_packed = pack_nft_features(
            task=task,
            feature_names=feature_names,
            label_name=label_name,
        )



    # ------------------------------------------------------------------ contracts
    store_addr    = to_checksum_address(args.store_addr)
    registry_addr = to_checksum_address(args.registry_addr)
    nft_addr      = to_checksum_address(args.nft_addr)
    store    = w3.eth.contract(address=store_addr,    abi=ABI_STORE)
    registry = w3.eth.contract(address=registry_addr, abi=ABI_REGISTRY)

    log()
    log(f"Store:    {store_addr}")
    log(f"Registry: {registry_addr}")
    log(f"NFT:      {nft_addr}")

    if args.resume:
        try:
            validate_resume_chain_state(
                state,
                core=core,
                chunk_size=CHUNK_SIZE,
                fetch_code=w3.eth.get_code,
                fetch_receipt=w3.eth.get_transaction_receipt,
                pointer_from_receipt=lambda receipt: parse_chunk_written(
                    w3, store, receipt
                ),
                expected_store=store_addr,
                expected_sender=account.address,
            )
        except WorkflowStateError as exc:
            log_err(str(exc))
            return 1
        log("Resume pointers, receipts, and deployed ModelStore bytes verified.")

    # ------------------------------------------------------------------ fees
    log()
    log("Reading registry fees and license/ToS metadata...")
    deploy_fee_wei         = registry.functions.deployFeeWei().call()
    size_fee_wei_per_byte  = registry.functions.sizeFeeWeiPerByte().call()
    try:
        required_fee_wei = registry.functions.requiredDeployFeeWei(core_len).call()
    except Exception:
        required_fee_wei = deploy_fee_wei + size_fee_wei_per_byte * core_len
    license_id  = registry.functions.activeLicenseId().call()
    tos_version = registry.functions.tosVersion().call()
    try:
        license_name, license_url = registry.functions.getLicense(license_id).call()
    except Exception:
        license_name, license_url = "(unknown)", ""

    log(f"Deploy fee (base):    {Web3.from_wei(deploy_fee_wei, 'ether')} L1")
    log(f"Size fee:             {Web3.from_wei(size_fee_wei_per_byte, 'ether')} L1 per byte")
    log(f"Required deploy fee:  {Web3.from_wei(required_fee_wei, 'ether')} L1")
    log(f"Active license:       id={license_id}  name='{license_name}'  url={license_url}")
    log(f"ToS version:          {tos_version}")

    if bal_wei < required_fee_wei:
        log_err(f"insufficient balance: have "
                f"{Web3.from_wei(bal_wei, 'ether')} L1, "
                f"need at least {Web3.from_wei(required_fee_wei, 'ether')} L1 "
                f"plus gas")
        return 1

    # ------------------------------------------------------------------ chunking
    log()
    log(f"Chunking: total={core_len:,} chunkSize={CHUNK_SIZE} chunks={n_chunks}")
    log(f"Estimated transactions: {n_chunks} chunk writes + 1 table write + 1 register = {n_chunks + 2}")

    # ------------------------------------------------------------------ interactive prompts
    log()
    print("=" * 72)
    print("MINT METADATA")
    print("=" * 72)
    if footer_title or footer_description:
        print("(Footer values shown in [brackets] — press Enter to accept,")
        print(" or type a new value to override.)")
        print()

    title_prompt = ("Title (≥3 chars, ≥1 word ≥2 chars)"
                    + (f"  [default: {footer_title}]" if footer_title else "")
                    + ":  ")
    title = prompt(
        title_prompt,
        validator=lambda v: (
            (True, "") if (not v and footer_title)
            else (False, "title must be ≥3 chars") if len(v) < 3
            else (False, "title must contain at least one word ≥2 chars")
                 if not title_word_hashes(v)
            else (True, "")
        ),
    )
    if not title and footer_title:
        title = footer_title
    word_hashes = title_word_hashes(title)
    log(f"  title resolved to: {title!r}")
    log(f"  {len(word_hashes)} indexed title word(s): "
        + ", ".join(f"0x{h.hex()[:8]}…" for h in word_hashes))

    desc_prompt = ("Description (≥8 chars)"
                   + (f"  [default: {footer_description[:40]}…]"
                      if footer_description and len(footer_description) > 40
                      else (f"  [default: {footer_description}]"
                            if footer_description else ""))
                   + ":  ")
    desc = prompt(
        desc_prompt,
        validator=lambda v: (
            (True, "") if (not v and footer_description)
            else (False, "description must be ≥8 chars") if len(v) < 8
            else (True, "")
        ),
    )
    if not desc and footer_description:
        desc = footer_description

    # Icon is never embedded in train_gl1f.py-produced footers (iconPngB64 is
    # always null at export time — it's only filled in by the studio UI's mint
    # tab, which doesn't go through this code path). So we always prompt.
    icon_path_str = prompt(
        "Icon path (128×128 PNG):             ",
        validator=lambda v: (
            (False, "file does not exist") if not Path(v).exists()
            else (True, "")
        ),
    )
    icon_bytes = validate_icon_128(Path(icon_path_str))
    log(f"  icon OK: {len(icon_bytes)} bytes")

    # ------------------------------------------------------------------ pricing
    pricing_mode = args.pricing_mode
    if pricing_mode == 0:
        fee_wei = 0
    else:
        try:
            fee_eth_clamped = max(0.001, min(1.0, float(args.pricing_fee_eth)))
        except ValueError:
            log_err(f"invalid --pricing-fee-eth: {args.pricing_fee_eth!r}")
            return 1
        fee_wei = w3.to_wei(fee_eth_clamped, "ether")
    if args.pricing_recipient:
        recipient = to_checksum_address(args.pricing_recipient)
    else:
        recipient = account.address

    # ------------------------------------------------------------------ legal
    log()
    print("=" * 72)
    print("LEGAL")
    print("=" * 72)
    print(f"Active license: id={license_id}  name='{license_name}'  url={license_url}")
    print(f"ToS version:    {tos_version}")
    print()
    print("By proceeding you accept the on-chain Terms (active version above)")
    print("AND the active Creative Commons license attached to your Model NFT")
    print("and its metadata, allowing third parties to copy/adapt/remix.")
    print()
    if not confirm(
        'Type exactly "I AGREE" (uppercase) to accept Terms and License:  ',
        exact_phrase="I AGREE",
    ):
        log_err("agreement not given; aborting")
        return 1

    # ------------------------------------------------------------------ final summary
    # NOTE: features_packed, task, feature_names, label_name are already
    # set above (either from GL1X footer or from CLI flags).

    log()
    print("=" * 72)
    print("DEPLOYMENT SUMMARY")
    print("=" * 72)
    print(f"Title:                {title}")
    print(f"Description:          {desc[:60]}{'…' if len(desc) > 60 else ''}")
    print(f"Icon:                 {len(icon_bytes)} bytes")
    print(f"Task:                 {task}")
    print(f"Features:             {header.n_features}")
    print(f"Trees (registry):     {header.registry_n_trees}")
    print(f"Depth:                {header.depth}")
    print(f"Model bytes:          {core_len:,} ({n_chunks} chunks of "
          f"{CHUNK_SIZE} bytes)")
    print(f"modelId:              {model_id_hex}")
    if state.get("owner_key_address"):
        owner_summary = f"reuse {state['owner_key_address']}"
    elif args.resume:
        owner_summary = "validate and reuse owner_key.txt"
    elif args.dry_run:
        owner_summary = "not generated (dry run)"
    else:
        owner_summary = "generate after final confirmation"
    print(f"Owner key:            {owner_summary}")
    print(f"Pricing mode:         {pricing_mode}  ({['free','tips','paid'][pricing_mode]})")
    print(f"Pricing feeWei:       {fee_wei}")
    print(f"Pricing recipient:    {recipient}")
    print(f"License id accepted:  {license_id}")
    print(f"ToS version accepted: {tos_version}")
    print(f"Required deploy fee:  {Web3.from_wei(required_fee_wei, 'ether')} L1")
    print(f"Estimated tx count:   {n_chunks + 2}")
    print(f"Gas price per tx:     {gp} gwei")
    print()
    owner_path = output_directory / OWNER_KEY_FILE
    if args.dry_run:
        try:
            dry_owner = acquire_owner_key(
                owner_path,
                expected_model_id=model_id_hex,
                title=title,
                generated_at=now_ts(),
                resume=args.resume,
                dry_run=True,
                create_account=Account.create,
                account_from_key=Account.from_key,
            )
            if dry_owner is not None:
                bind_owner_key(state, dry_owner.address)
                log(f"DRY RUN: validated resume owner key {dry_owner.address}.")
        except (OwnerKeyError, WorkflowStateError) as exc:
            log_err(str(exc))
            return 1
        log("DRY RUN: stopping before sending any transactions.")
        return 0
    if not confirm(
        'Type exactly "MINT" (uppercase) to start the deployment:  ',
        exact_phrase="MINT",
    ):
        log_err("not confirmed; aborting")
        return 1

    # ------------------------------------------------------------------ owner key
    # No key is generated before the dry-run/final-confirmation boundary.
    # Resume always reuses a model-bound key and never overwrites it.
    try:
        owner_material = acquire_owner_key(
            owner_path,
            expected_model_id=model_id_hex,
            title=title,
            generated_at=now_ts(),
            resume=args.resume,
            dry_run=False,
            create_account=Account.create,
            account_from_key=Account.from_key,
        )
        if owner_material is None:  # defensive; dry_run=False always returns one
            raise OwnerKeyError("owner key acquisition unexpectedly returned no key")
        owner_acct = owner_material.account
        state = bind_owner_key(state, owner_material.address)
        try:
            persist_state()
        except Exception:
            # A newly created key has not been used by any transaction yet. If
            # the initial state reservation cannot be persisted, remove that
            # otherwise-unresumable key rather than leave a dangerous orphan.
            if owner_material.created:
                try:
                    owner_path.unlink()
                except FileNotFoundError:
                    pass
            raise
    except (OSError, OwnerKeyError, WorkflowStateError) as exc:
        log_err(str(exc))
        return 1

    log()
    print("=" * 72)
    print("OWNER API ACCESS KEY")
    print("=" * 72)
    if owner_material.created:
        print("A fresh owner API key was created with mode 0600.")
    else:
        print("The existing model-bound owner API key was validated and reused.")
    print("The PUBLIC ADDRESS grants perpetual API-key inference access.")
    print("The PRIVATE KEY is needed to sign predictAccessView calls later.")
    print("The private key cannot be recovered; store it securely. The Model NFT")
    print("owner can register a replacement later with setOwnerAccessKey().")
    print()
    log(f"Owner address:      {owner_acct.address}")
    log(f"Owner private key:  {'written to' if owner_material.created else 'loaded from'} "
        f"{owner_path} (mode 0600)")
    log()
    if not confirm(
        'Type exactly "I saved it" to continue:  ',
        exact_phrase="I saved it",
    ):
        log_err("owner private key not confirmed; no transaction was sent; "
                "rerun with --resume")
        return 1

    # ------------------------------------------------------------------ resume state
    completed = list(state.get("completed_chunks") or [])
    if args.resume:
        log()
        log(f"Resuming from chunk {len(completed)}/{n_chunks} "
            f"(table_ptr={state.get('table_ptr')}, "
            f"registered={bool(state.get('register_tx_hash'))})")

    # ------------------------------------------------------------------ chunk loop
    log()
    log(f"Deploying {n_chunks} chunks (one tx per block)...")
    pointers: list[str] = [c["pointer"] for c in completed]

    for i in range(len(completed), n_chunks):
        start = i * CHUNK_SIZE
        end = min(core_len, start + CHUNK_SIZE)
        chunk = core[start:end]
        log()
        log(f"Chunk {i + 1}/{n_chunks}: store.write({len(chunk)} bytes)")

        encoded = store.encode_abi("write", args=[chunk])
        encoded_bytes = bytes.fromhex(encoded[2:] if encoded.startswith("0x") else encoded)

        def build_chunk_tx(gp_wei, *, _to=store_addr, _data=encoded_bytes):
            return build_legacy_tx(
                w3, account, to=_to, data=_data,
                gas=GAS_LIMIT_CHUNK_WRITE, gas_price_wei=gp_wei, value=0,
            )

        rcpt = send_with_retry(
            w3, account, build_tx_fn=build_chunk_tx,
            tag=f"chunk {i+1}/{n_chunks}", gas_price_wei=gas_price_wei,
        )
        ptr = parse_chunk_written(w3, store, rcpt)
        log(f"  pointer: {ptr}")
        pointers.append(ptr)

        chunk_tx_hash = (rcpt["transactionHash"].hex()
                         if hasattr(rcpt["transactionHash"], "hex")
                         else str(rcpt["transactionHash"]))
        if not chunk_tx_hash.startswith("0x"):
            chunk_tx_hash = "0x" + chunk_tx_hash
        completed.append({
            "index":     i,
            "pointer":   ptr,
            "tx_hash":   chunk_tx_hash,
            "model_id":  model_id_hex,
        })
        state["completed_chunks"] = completed
        persist_state()

    # ------------------------------------------------------------------ pointer table
    table = bytearray(32 * n_chunks)
    for i, ptr in enumerate(pointers):
        addr_bytes = bytes.fromhex(ptr[2:]) if ptr.startswith("0x") else bytes.fromhex(ptr)
        if len(addr_bytes) != 20:
            raise RuntimeError(f"unexpected pointer length: {ptr}")
        # right-aligned in 32-byte slot — UI: table.set(ab, i*32 + 12)
        table[i * 32 + 12 : i * 32 + 32] = addr_bytes
    table_bytes = bytes(table)

    if state.get("table_ptr"):
        table_ptr = state["table_ptr"]
        log()
        log(f"Pointer table already deployed: {table_ptr}")
    else:
        log()
        log(f"Writing pointer table: {len(table_bytes)} bytes "
            f"({n_chunks} pointers × 32 bytes)")
        encoded = store.encode_abi("write", args=[table_bytes])
        encoded_bytes = bytes.fromhex(encoded[2:] if encoded.startswith("0x") else encoded)

        def build_table_tx(gp_wei, *, _to=store_addr, _data=encoded_bytes):
            return build_legacy_tx(
                w3, account, to=_to, data=_data,
                gas=GAS_LIMIT_TABLE_WRITE, gas_price_wei=gp_wei, value=0,
            )

        rcpt = send_with_retry(
            w3, account, build_tx_fn=build_table_tx,
            tag="table", gas_price_wei=gas_price_wei,
        )
        table_ptr = parse_chunk_written(w3, store, rcpt)
        log(f"  table pointer: {table_ptr}")
        state["table_ptr"] = table_ptr
        persist_state()

    # ------------------------------------------------------------------ register
    log()
    if state.get("register_tx_hash"):
        log("Recovering the recorded registration receipt; no transaction will be resubmitted...")
    else:
        log("Registering model (single tx, includes mint of ERC-721 Model NFT)...")

    def submit_registration():
        register_args = [
            model_id,                          # bytes32 modelId
            to_checksum_address(table_ptr),    # address tablePtr
            CHUNK_SIZE,                        # uint32  chunkSize
            n_chunks,                          # uint32  numChunks
            core_len,                          # uint32  totalBytes
            header.n_features,                 # uint16  nFeatures
            header.registry_n_trees,           # uint16  nTrees (v2 stores K*R)
            header.depth,                      # uint16  depth
            header.registry_base_q,            # int32   baseQ (v2 reserved zero)
            header.scale_q,                    # uint32  scaleQ
            title,                             # string  title
            desc,                              # string  description
            icon_bytes,                        # bytes   iconPng32
            features_packed,                   # string  featuresPacked
            word_hashes,                       # bytes32[] titleWordHashes
            pricing_mode,                      # uint8   pricingMode
            fee_wei,                           # uint256 feeWei
            recipient,                         # address recipient
            tos_version,                       # uint32  tosVersionAccepted
            license_id,                        # uint32  licenseIdAccepted
            owner_acct.address,                # address ownerKey
        ]
        encoded = registry.encode_abi("registerModel", args=register_args)
        encoded_bytes = bytes.fromhex(
            encoded[2:] if encoded.startswith("0x") else encoded
        )

        def build_register_tx(gp_wei, *, _to=registry_addr, _data=encoded_bytes,
                              _value=required_fee_wei):
            return build_legacy_tx(
                w3, account, to=_to, data=_data,
                gas=GAS_LIMIT_REGISTER, gas_price_wei=gp_wei, value=_value,
            )

        return send_with_retry(
            w3, account, build_tx_fn=build_register_tx,
            tag="register", gas_price_wei=gas_price_wei,
        )

    try:
        registration = obtain_registration_receipt(
            state,
            fetch_receipt=w3.eth.get_transaction_receipt,
            submit_registration=submit_registration,
            expected_registry=registry_addr,
            expected_sender=account.address,
        )
    except WorkflowStateError as exc:
        log_err(str(exc))
        return 1

    rcpt = registration.receipt
    reg_tx_hash = registration.transaction_hash
    if registration.recovered:
        log(f"Recovered successful registration receipt: {reg_tx_hash}")
    else:
        state["register_tx_hash"] = reg_tx_hash
        persist_state()

    token_id = parse_minted_token_id(w3, nft_addr, rcpt)
    if token_id is None:
        log_err("registration receipt has no ModelNFT mint event; refusing to write S1.json")
        return 1
    try:
        registered_summary = registry.functions.getModelSummary(token_id).call()
        registered_model_id = bytes(registered_summary[1])
    except Exception as exc:
        log_err(
            "cannot verify recovered registry model identity "
            f"({type(exc).__name__}) via {rpc_identity}"
        )
        return 1
    try:
        registered_table_ptr = to_checksum_address(registered_summary[2])
        registered_creator = to_checksum_address(registered_summary[11])
    except Exception as exc:
        log_err(f"registered model summary is malformed ({type(exc).__name__})")
        return 1
    if (
        not registered_summary[0]
        or registered_model_id != bytes(model_id)
        or registered_table_ptr != to_checksum_address(table_ptr)
        or registered_creator != to_checksum_address(account.address)
    ):
        log_err(
            "registration receipt does not resolve to the selected "
            "modelId, pointer table, and signer"
        )
        return 1

    # ------------------------------------------------------------------ artifacts (S1.json)
    log()
    log("=" * 72)
    log("✅ DEPLOYMENT COMPLETE")
    log("=" * 72)
    artifact = {
        "deployer_address":     account.address,
        "owner_key_address":    owner_acct.address,
        "model_id":             model_id_hex,
        "title":                title,
        "description":          desc,
        "task":                 task,
        "n_features":           header.n_features,
        "n_trees":              header.registry_n_trees,
        "n_outputs":            header.n_outputs,
        "depth":                header.depth,
        "base_q":               header.registry_base_q,
        "scale_q":              header.scale_q,
        "version":              header.version,
        "core_bytes":           core_len,
        "core_sha256":          hashlib.sha256(core).hexdigest(),
        "chunk_size":           CHUNK_SIZE,
        "n_chunks":             n_chunks,
        "first_chunk_pointer":  pointers[0],
        "last_chunk_pointer":   pointers[-1],
        "table_pointer":        table_ptr,
        "store_address":        store_addr,
        "registry_address":     registry_addr,
        "nft_address":          nft_addr,
        "runtime_address":      DEFAULTS["runtime"],
        "register_tx_hash":     reg_tx_hash,
        "register_block_number": rcpt["blockNumber"],
        "token_id":             token_id,
        "license_id_accepted":  license_id,
        "tos_version_accepted": tos_version,
        "pricing_mode":         pricing_mode,
        "pricing_fee_wei":      fee_wei,
        "pricing_recipient":    recipient,
        "chain_id":             chain_id,
        "rpc":                  rpc_identity,
        "feature_names":        feature_names,
    }
    try:
        save_json_artifact_exclusive(artifact_path, artifact)
    except (OSError, WorkflowStateError) as exc:
        log_err(str(exc) if isinstance(exc, WorkflowStateError) else
                "cannot safely publish S1.json")
        return 1
    log(f"Artifacts written: {artifact_path}")
    log()
    log(f"modelId:        {model_id_hex}")
    log(f"tokenId:        {token_id if token_id is not None else '(parse failed; check explorer)'}")
    log(f"register tx:    {reg_tx_hash}")
    log(f"table pointer:  {table_ptr}")
    log(f"first chunk:    {pointers[0]}")
    log(f"last chunk:     {pointers[-1]}")
    log()
    log("Owner API key (private) is in owner_key.txt — back it up now.")
    log()
    return 0


def main() -> int:
    """Run the publisher without allowing provider exceptions to leak URLs."""
    try:
        return _main()
    except Exception as exc:
        log_err(
            "mint aborted safely after an unexpected "
            f"{type(exc).__name__}; no exception details were printed because "
            "provider errors can contain RPC credentials"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
