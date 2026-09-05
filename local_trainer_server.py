#!/usr/bin/env python3
"""
local_trainer_server.py (stdlib-only)

Serves the Forest "create" UI (static files) and a small HTTP API for Python training.

Endpoints:
  POST /api/upload?filename=...   (body: raw bytes, Content-Type: application/octet-stream)
    -> { ok, datasetId, filename, sizeBytes, columns }

  POST /api/train  (JSON)
    -> { ok, modelBytesB64, meta, curve }

  POST /api/stop
    -> { ok, stopped }

  GET /api/ping
    -> { ok, version }

Notes:
- The server code itself is stdlib-only. Training is executed by spawning `train_gl1f.py`
  as a subprocess (which may use numpy).
- Dataset caching: upload once, get a datasetId, then reuse that id for all training rounds.
- Open the UI through this server (normally http://127.0.0.1:8787). The API rejects
  cross-origin browser requests and unapproved Host names. The process refuses to bind
  beyond loopback because this local bridge has no remote-user authentication layer.
- Static serving is restricted to the browser UI and explicitly public research links;
  repository internals, dotfiles, source tooling, and local credentials are not served.
- Upload, JSON-request, and trainer-output sizes are bounded and configurable by CLI.
"""

from __future__ import annotations

import argparse
import base64
import csv
import ipaddress
import io
import json
import os
import secrets
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple


VERSION = "1.1.0"
DEFAULT_MAX_UPLOAD_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_JSON_BYTES = 1024 * 1024
DEFAULT_MAX_MODEL_BYTES = 256 * 1024 * 1024


# ----------------------------
# In-memory dataset registry
# ----------------------------

_DATASETS_LOCK = threading.Lock()
_DATASETS: Dict[str, Dict[str, Any]] = {}  # datasetId -> {path, filename, sizeBytes, columns, createdAt}

_ACTIVE_PROC_LOCK = threading.Lock()
_ACTIVE_PROC: Optional[subprocess.Popen] = None
_TRAIN_SLOT_LOCK = threading.Lock()


class RequestBodyError(ValueError):
    """The declared HTTP request body could not be read completely."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = int(status)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_loopback_bind_host(host: str) -> bool:
    value = str(host).strip().lower().strip("[]")
    if value == "localhost":
        return True
    try:
        address = ipaddress.ip_address(value)
        return address.version == 4 and address.is_loopback
    except ValueError:
        return False


def _safe_filename(name: str) -> str:
    # Keep it simple and cross-platform.
    name = (name or "dataset.csv").strip().replace("\\", "_").replace("/", "_")
    # Avoid extremely long names.
    if len(name) > 120:
        root, ext = os.path.splitext(name)
        name = root[:100] + ext[:20]
    return name or "dataset.csv"


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _read_body_stream(
    rfile,
    length: int,
    out_path: Path,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Write exactly ``length`` bytes to a new file or leave no file behind."""
    if length < 0:
        raise RequestBodyError("negative request body length")

    written = 0
    try:
        # The caller supplies a random path in a private cache directory.  "xb"
        # prevents a local symlink/pre-creation race from redirecting the write.
        with out_path.open("xb") as f:
            remaining = length
            while remaining > 0:
                chunk = rfile.read(min(chunk_size, remaining))
                if not chunk:
                    raise RequestBodyError(
                        f"incomplete request body ({written} of {length} bytes)"
                    )
                f.write(chunk)
                written += len(chunk)
                remaining -= len(chunk)
            f.flush()
            os.fsync(f.fileno())
        return written
    except Exception:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _private_cache_subdir(cache_dir: Path, name: str) -> Path:
    """Return an owner-private, non-symlink cache subdirectory."""
    root = cache_dir.resolve(strict=True)
    candidate = root / name
    candidate.mkdir(mode=0o700, parents=False, exist_ok=True)
    resolved = candidate.resolve(strict=True)
    if resolved.parent != root:
        raise RuntimeError(f"unsafe cache subdirectory: {candidate}")
    if not resolved.is_dir():
        raise RuntimeError(f"cache path is not a directory: {resolved}")
    try:
        resolved.chmod(0o700)
    except OSError:
        # Some platforms/filesystems do not implement POSIX permissions.
        pass
    return resolved



def _autodetect_delimiter_from_lines(lines: list[str], fallback: str = ",") -> str:
    candidates = [",", ";", "\t", "|"]
    best = fallback
    best_mode = 1
    best_freq = 0
    best_penalty = 10**18

    for d in candidates:
        freq: dict[int, int] = {}
        penalty = 0
        try:
            reader = csv.reader(lines, delimiter=d, quotechar='"', escapechar="\\")
            parsed = list(reader)
        except Exception:
            parsed = []

        for row in parsed:
            n = len(row)
            freq[n] = freq.get(n, 0) + 1
            for cell in row:
                for od in candidates:
                    if od == d:
                        continue
                    penalty += str(cell).count(od)

        mode_n, mode_f = 1, 0
        for k, v in freq.items():
            if v > mode_f or (v == mode_f and k > mode_n):
                mode_n, mode_f = k, v
        if mode_n < 2:
            continue

        if (mode_f > best_freq) or (mode_f == best_freq and mode_n > best_mode) or (mode_f == best_freq and mode_n == best_mode and penalty < best_penalty):
            best_freq = mode_f
            best_mode = mode_n
            best_penalty = penalty
            best = d

    return best


def _csv_columns_from_file(path: Path, max_bytes: int = 2 * 1024 * 1024) -> Optional[list]:
    """
    Read the first CSV row (header) with stdlib csv.
    We only read a small prefix for safety.
    """
    try:
        with path.open("rb") as f:
            raw = f.read(max_bytes)
        # Find first newline to avoid huge header scanning; but if header is long,
        # csv.reader can still parse it from this chunk.
        text = raw.decode("utf-8", errors="replace")
        # Ensure we only feed up to first newline if present (keeps snappy).
        nl = text.find("\n")
        if nl != -1:
            text = text[: nl + 1]
        # csv.reader expects file-like.
        sio = io.StringIO(text)
        # Auto-detect delimiter for header
        header_lines = [text.strip("\r\n")]
        delim = _autodetect_delimiter_from_lines(header_lines, fallback=",")
        reader = csv.reader(sio, delimiter=delim, quotechar='"', escapechar="\\")
        for row in reader:
            if row and isinstance(row[0], str) and row[0].startswith("\ufeff"):
                row[0] = row[0].lstrip("\ufeff")
            return row
        return None
    except Exception:
        return None


def _gl1f_model_len(gl1f_bytes: bytes) -> int:
    """
    Compute model byte length for GL1F v1/v2 (enough to strip any GL1X footer).
    Mirrors logic in src/local_infer.js (decodeModel).
    """
    if len(gl1f_bytes) < 24:
        raise ValueError("GL1F bytes too short")
    if gl1f_bytes[0:4] != b"GL1F":
        raise ValueError("Missing GL1F magic")
    ver = gl1f_bytes[4]
    depth = struct.unpack_from("<H", gl1f_bytes, 8)[0]
    pow2 = 1 << depth
    internal = pow2 - 1
    per_tree = internal * 8 + pow2 * 4

    if ver == 1:
        # headerSize = 24, nTrees at offset 10 (u32)
        n_trees = struct.unpack_from("<I", gl1f_bytes, 10)[0]
        return 24 + n_trees * per_tree

    if ver == 2:
        # headerSize = 24 (+ nClasses*4 offsets), treesPerClass at 10 (u32), nClasses at 22 (u16)
        trees_per_class = struct.unpack_from("<I", gl1f_bytes, 10)[0]
        n_classes = struct.unpack_from("<H", gl1f_bytes, 22)[0]
        trees_off = 24 + n_classes * 4
        return trees_off + (trees_per_class * n_classes) * per_tree

    raise ValueError(f"Unsupported GL1F version: {ver}")


def _parse_gl1x_footer(gl1f_bytes: bytes) -> Tuple[bytes, Optional[dict]]:
    """
    Returns (model_bytes_without_footer, pkg_json_dict_or_None)
    """
    model_len = _gl1f_model_len(gl1f_bytes)
    if len(gl1f_bytes) < model_len + 12:
        return gl1f_bytes, None
    if gl1f_bytes[model_len : model_len + 4] != b"GL1X":
        return gl1f_bytes, None
    # ver = gl1f_bytes[model_len+4]
    json_len = struct.unpack_from("<I", gl1f_bytes, model_len + 8)[0]
    start = model_len + 12
    end = start + int(json_len)
    if end > len(gl1f_bytes):
        return gl1f_bytes[:model_len], None
    raw = gl1f_bytes[start:end]
    try:
        pkg = json.loads(raw.decode("utf-8"))
    except Exception:
        pkg = None
    return gl1f_bytes[:model_len], pkg


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """
    Best-effort terminate/kill; cross-platform-ish.
    """
    try:
        if proc.poll() is not None:
            return
        # The trainer starts in its own POSIX session, so terminate any helper
        # process it may have spawned as well as the direct child.
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def _train_subprocess(
    *,
    engine: str,
    train_script: Path,
    train_bin: Optional[str],
    dataset_path: Path,
    out_path: Path,
    req: dict,
    python_exe: str,
) -> Tuple[int, str, str]:
    """Run the local trainer (python or cpp); return (exit_code, stdout, stderr)."""
    task = (req.get("task") or "regression").strip()
    feature_cols = req.get("featureCols") or []
    label_col = req.get("labelCol")
    label_cols = req.get("labelCols") or []
    neg_label = req.get("negLabel")
    pos_label = req.get("posLabel")
    class_labels = req.get("classLabels") or []
    params = req.get("params") or {}

    engine = (engine or "python").strip().lower()
    if engine not in ("python", "cpp"):
        raise ValueError(f"Unknown engine: {engine}")

    if engine == "cpp":
        if not train_bin:
            raise RuntimeError("C++ trainer binary path not configured")
        argv = [str(train_bin)]
    else:
        argv = [python_exe, str(train_script)]

    argv += ["--task", str(task), "--input", str(dataset_path), "--out", str(out_path)]

    # Columns
    if feature_cols:
        argv += ["--feature-cols", ",".join(map(str, feature_cols))]
    if label_cols:
        argv += ["--label-cols", ",".join(map(str, label_cols))]
    elif label_col is not None:
        argv += ["--label-col", str(label_col)]

    if neg_label is not None:
        argv += ["--neg-label", str(neg_label)]
    if pos_label is not None:
        argv += ["--pos-label", str(pos_label)]
    if class_labels:
        argv += ["--class-labels", ",".join(map(str, class_labels))]

    # Hyperparams
    def _opt_int_key(key: str, flag: str) -> None:
        if not isinstance(params, dict):
            return
        v = params.get(key)
        if v is None:
            return
        try:
            argv.extend([flag, str(int(v))])
        except Exception:
            argv.extend([flag, str(v)])
    
    def _opt_float_key(key: str, flag: str) -> None:
        if not isinstance(params, dict):
            return
        v = params.get(key)
        if v is None:
            return
        try:
            argv.extend([flag, str(float(v))])
        except Exception:
            argv.extend([flag, str(v)])
    
    def _opt_str_key(key: str, flag: str) -> None:
        if not isinstance(params, dict):
            return
        v = params.get(key)
        if v is None:
            return
        argv.extend([flag, str(v)])
    
    if isinstance(params, dict):
        _opt_int_key("trees", "--trees")
        _opt_int_key("depth", "--depth")
        _opt_float_key("lr", "--lr")
        _opt_int_key("minLeaf", "--min-leaf")
        _opt_int_key("seed", "--seed")
        _opt_int_key("bins", "--bins")
        _opt_str_key("binning", "--binning")
        _opt_float_key("splitTrain", "--split-train")
        _opt_float_key("splitVal", "--split-val")
        if params.get("refitTrainVal"):
            argv += ["--refit-train-val"]
    
        if params.get("earlyStop"):
            argv += ["--early-stop"]
        _opt_int_key("patience", "--patience")
    
        if "scaleQ" in params and params["scaleQ"] is not None:
            argv += ["--scaleQ", str(params["scaleQ"])]
    
    # Imbalance / weighting
    imb = req.get("imbalance") or {}
    if isinstance(imb, dict):
        mode = (imb.get("mode") or "none")
        if mode and mode != "none":
            argv += ["--imbalance-mode", str(mode)]
            if imb.get("cap") is not None:
                argv += ["--imbalance-cap", str(imb.get("cap"))]
            if imb.get("normalize"):
                argv += ["--imbalance-normalize"]
            if imb.get("stratify"):
                argv += ["--stratify"]
            if imb.get("w0") is not None:
                argv += ["--w0", str(imb.get("w0"))]
            if imb.get("w1") is not None:
                argv += ["--w1", str(imb.get("w1"))]

            cw = imb.get("classWeights")
            if cw:
                if isinstance(cw, (list, tuple)):
                    argv += ["--class-weights", ",".join(str(x) for x in cw)]
                else:
                    argv += ["--class-weights", str(cw)]

            pw = imb.get("posWeights")
            if pw:
                if isinstance(pw, (list, tuple)):
                    argv += ["--pos-weights", ",".join(str(x) for x in pw)]
                else:
                    argv += ["--pos-weights", str(pw)]

    # LR schedule
    lrs = req.get("lrSchedule") or {}
    if isinstance(lrs, dict):
        mode = (lrs.get("mode") or "none").lower()
        if mode in ("plateau", "piecewise"):
            argv += ["--lr-schedule", mode]
        if mode == "plateau":
            if lrs.get("patience") is not None:
                argv += ["--lr-patience", str(lrs.get("patience"))]
            if lrs.get("dropPct") is not None:
                argv += ["--lr-drop-pct", str(lrs.get("dropPct"))]
            if lrs.get("lrMin") is not None:
                argv += ["--lr-min", str(lrs.get("lrMin"))]
        if mode == "piecewise":
            segs = lrs.get("segments") or []
            parts = []
            if isinstance(segs, list):
                for s in segs:
                    if not isinstance(s, dict):
                        continue
                    try:
                        start = int(s.get("start"))
                        end = int(s.get("end"))
                        lr = float(s.get("lr"))
                        parts.append(f"{start}:{end}:{lr}")
                    except Exception:
                        continue
            if parts:
                argv += ["--lr-segments", ",".join(parts)]

    # Optional packaging metadata (title/description) if provided by UI.
    meta = req.get("nft") or {}
    if isinstance(meta, dict):
        title = meta.get("title")
        desc = meta.get("description")
        if title:
            argv += ["--title", str(title)]
        if desc:
            argv += ["--description", str(desc)]

    global _ACTIVE_PROC
    with _ACTIVE_PROC_LOCK:
        if _ACTIVE_PROC is not None and _ACTIVE_PROC.poll() is None:
            raise RuntimeError("Training already running")
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(train_script.parent),
            text=True,
            start_new_session=(os.name == "posix"),
        )
        _ACTIVE_PROC = proc
    try:
        out, err = proc.communicate()
        code = int(proc.returncode or 0)
        return code, out or "", err or ""
    finally:
        with _ACTIVE_PROC_LOCK:
            if _ACTIVE_PROC is proc:
                _ACTIVE_PROC = None
class Handler(SimpleHTTPRequestHandler):
    # `directory` is provided via functools.partial in main()

    server_version = f"local_trainer_server/{VERSION}"
    _PUBLIC_ROOT_FILES = frozenset(
        {
            "GL1F.pdf",
            "CITATION.cff",
            "FORMAT_SPEC.md",
            "LICENSE",
            "REPRODUCIBILITY.md",
            "aistore.html",
            "create.html",
            "forest.html",
            "index.html",
            "model.html",
            "my.html",
            "research.html",
            "style.css",
            "terms.html",
        }
    )
    _PUBLIC_NESTED_FILES = frozenset(
        {
            "benchmarks/live_chain_witness.mjs",
            "benchmarks/results/LIVE_CHAIN_WITNESS.md",
            "benchmarks/results/LIVE_CHAIN_WITNESS_EXTENDED_V2.md",
            "deployments/genesisl1.json",
            "docs/ARCHITECTURE.md",
            "docs/DEPLOYED_SYSTEM.md",
            "docs/ON_CHAIN_API.md",
            "paper/GL1F_Formal_Supplement.pdf",
        }
    )

    @classmethod
    def _public_relative_path(cls, relative: PurePosixPath) -> bool:
        value = relative.as_posix()
        if len(relative.parts) == 1 and value in cls._PUBLIC_ROOT_FILES:
            return True
        if value in cls._PUBLIC_NESTED_FILES:
            return True
        return (
            len(relative.parts) == 2
            and relative.parts[0] == "src"
            and relative.suffix == ".js"
        )

    def _static_request_allowed(self, url_path: str) -> bool:
        try:
            decoded = urllib.parse.unquote(url_path, errors="strict")
        except UnicodeDecodeError:
            return False
        if "\0" in decoded or "\\" in decoded or decoded.startswith("//"):
            return False
        relative = PurePosixPath(decoded.lstrip("/"))
        parts = relative.parts
        if any(part in ("", ".", "..") or part.startswith(".") for part in parts):
            return False

        base = Path(self.directory).resolve(strict=True)
        if not parts:
            return (base / "index.html").is_file()
        if not self._public_relative_path(relative):
            return False

        candidate = (base / Path(*parts)).resolve(strict=False)
        try:
            resolved_relative = PurePosixPath(candidate.relative_to(base).as_posix())
        except ValueError:
            return False
        # Apply the allowlist again after symlink resolution so an allowed name
        # cannot be used as an alias for a private source or credential file.
        return self._public_relative_path(resolved_relative)

    def _send_json(self, obj: Any, status: int = 200) -> None:
        data = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _request_length(self, *, maximum: int, required: bool = True) -> int:
        if self.headers.get("Transfer-Encoding"):
            raise RequestBodyError(
                "Transfer-Encoding is not supported",
                HTTPStatus.BAD_REQUEST,
            )
        values = self.headers.get_all("Content-Length", failobj=[])
        if len(values) != 1:
            if not values and not required:
                return 0
            raise RequestBodyError(
                "Exactly one Content-Length header is required",
                HTTPStatus.LENGTH_REQUIRED,
            )
        value = values[0].strip()
        if not value.isdecimal():
            raise RequestBodyError("Invalid Content-Length")
        length = int(value)
        if length > maximum:
            raise RequestBodyError(
                f"Request body exceeds the {maximum}-byte limit",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        return length

    def _read_json_body(self) -> dict:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise RequestBodyError(
                "Content-Type must be application/json",
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
        maximum = int(getattr(self.server, "max_json_bytes", DEFAULT_MAX_JSON_BYTES))
        length = self._request_length(maximum=maximum)
        raw = self.rfile.read(length) if length > 0 else b""
        if len(raw) != length:
            raise RequestBodyError(
                f"Incomplete request body ({len(raw)} of {length} bytes)"
            )
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RequestBodyError("Invalid JSON request body") from exc
        if not isinstance(value, dict):
            raise RequestBodyError("JSON request body must be an object")
        return value

    def _host_allowed(self) -> bool:
        host_values = self.headers.get_all("Host", failobj=[])
        if len(host_values) != 1:
            return False
        raw_host = host_values[0].strip()
        if not raw_host or "@" in raw_host:
            return False
        try:
            parsed = urllib.parse.urlsplit(f"//{raw_host}")
            hostname = (parsed.hostname or "").lower()
            port = parsed.port
        except ValueError:
            return False
        expected_port = int(self.server.server_address[1])
        effective_port = port if port is not None else 80
        if effective_port != expected_port:
            return False
        allowed = {
            str(value).lower()
            for value in getattr(
                self.server,
                "allowed_api_hosts",
                ("127.0.0.1", "localhost", "::1"),
            )
        }
        return hostname in allowed

    def _same_origin(self, value: str) -> bool:
        raw_host = (self.headers.get("Host") or "").strip()
        try:
            actual = urllib.parse.urlsplit(value)
            expected = urllib.parse.urlsplit(f"http://{raw_host}")
            actual_port = actual.port or 80
            expected_port = expected.port or 80
        except ValueError:
            return False
        return (
            actual.scheme.lower() == "http"
            and actual.username is None
            and actual.password is None
            and (actual.hostname or "").lower() == (expected.hostname or "").lower()
            and actual_port == expected_port
        )

    def _guard_api_request(self, *, mutating: bool) -> bool:
        if not self._host_allowed():
            self._send_json(
                {"ok": False, "error": "Host is not allowed for the local API"},
                status=HTTPStatus.FORBIDDEN,
            )
            return False
        if not mutating:
            return True

        fetch_site = (self.headers.get("Sec-Fetch-Site") or "").lower()
        if fetch_site in ("cross-site", "none"):
            self._send_json(
                {"ok": False, "error": "Cross-origin API requests are not allowed"},
                status=HTTPStatus.FORBIDDEN,
            )
            return False

        origin = self.headers.get("Origin")
        if origin is not None and not self._same_origin(origin):
            self._send_json(
                {"ok": False, "error": "Cross-origin API requests are not allowed"},
                status=HTTPStatus.FORBIDDEN,
            )
            return False

        referer = self.headers.get("Referer")
        if origin is None and referer is not None and not self._same_origin(referer):
            self._send_json(
                {"ok": False, "error": "Cross-origin API requests are not allowed"},
                status=HTTPStatus.FORBIDDEN,
            )
            return False
        return True

    def do_OPTIONS(self):
        self._send_json(
            {"ok": False, "error": "Cross-origin API requests are not allowed"},
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/ping":
            if not self._guard_api_request(mutating=False):
                return
            cpp_bin = getattr(self.server, "cpp_train_bin", None)
            supports_cpp = bool(cpp_bin) and os.path.exists(cpp_bin)
            self._send_json({"ok": True, "version": VERSION, "time": _now_iso(), "supportsCpp": supports_cpp})
            return
        if not self._static_request_allowed(parsed.path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        return super().do_GET()

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        if not self._static_request_allowed(parsed.path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        return super().do_HEAD()

    def list_directory(self, path):
        self.send_error(HTTPStatus.NOT_FOUND)
        return None

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if not self._guard_api_request(mutating=True):
            return
        if parsed.path == "/api/upload":
            return self._handle_upload(parsed)
        if parsed.path == "/api/train":
            return self._handle_train()
        if parsed.path == "/api/stop":
            return self._handle_stop()
        self._send_json({"ok": False, "error": f"Unknown endpoint: {parsed.path}"}, status=404)

    def _handle_upload(self, parsed_url: urllib.parse.ParseResult):
        tmp_path: Optional[Path] = None
        try:
            if self.headers.get_content_type() != "application/octet-stream":
                raise RequestBodyError(
                    "Content-Type must be application/octet-stream",
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
            qs = urllib.parse.parse_qs(parsed_url.query or "")
            filename = _safe_filename((qs.get("filename") or ["dataset.csv"])[0])

            maximum = int(
                getattr(self.server, "max_upload_bytes", DEFAULT_MAX_UPLOAD_BYTES)
            )
            length = self._request_length(maximum=maximum)
            if length <= 0:
                self._send_json({"ok": False, "error": "Missing Content-Length"}, status=400)
                return

            dataset_id = secrets.token_hex(16)
            ext = Path(filename).suffix.lower()
            if ext not in (".csv", ".tsv", ".txt"):
                ext = ".csv"
            cache_dir = Path(self.server.cache_dir)  # type: ignore[attr-defined]
            datasets_dir = _private_cache_subdir(cache_dir, "datasets")
            ds_path = datasets_dir / f"{dataset_id}{ext}"
            tmp_path = datasets_dir / f".tmp_{dataset_id}_{secrets.token_hex(8)}"

            # Publish only a fully received file.  The exclusive temporary
            # creation and same-directory replace make publication atomic.
            written = _read_body_stream(self.rfile, length, tmp_path)
            os.replace(tmp_path, ds_path)
            tmp_path = None

            columns = _csv_columns_from_file(ds_path) or []

            meta = {
                "id": dataset_id,
                "path": str(ds_path),
                "filename": filename,
                "sizeBytes": int(written),
                "columns": columns,
                "createdAt": _now_iso(),
            }
            with _DATASETS_LOCK:
                _DATASETS[dataset_id] = meta

            self._send_json({
                "ok": True,
                "datasetId": dataset_id,
                "filename": filename,
                "sizeBytes": int(written),
                "columns": columns,
            })
        except RequestBodyError as e:
            self._send_json({"ok": False, "error": str(e)}, status=e.status)
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    def _handle_stop(self):
        try:
            self._request_length(maximum=0, required=False)
        except RequestBodyError as e:
            self._send_json({"ok": False, "error": str(e)}, status=e.status)
            return
        stopped = False
        with _ACTIVE_PROC_LOCK:
            proc = _ACTIVE_PROC
        if proc is not None and proc.poll() is None:
            stopped = True
            _kill_process_tree(proc)
        self._send_json({"ok": True, "stopped": stopped})

    def _handle_train(self):
        if not _TRAIN_SLOT_LOCK.acquire(blocking=False):
            self._send_json(
                {"ok": False, "error": "Training already running"},
                status=HTTPStatus.CONFLICT,
            )
            return
        try:
            self._handle_train_reserved()
        finally:
            _TRAIN_SLOT_LOCK.release()

    def _handle_train_reserved(self):
        try:
            req = self._read_json_body()
            dataset_id = str(req.get("datasetId") or "")
            if not dataset_id:
                self._send_json({"ok": False, "error": "datasetId is required"}, status=400)
                return

            with _DATASETS_LOCK:
                ds = _DATASETS.get(dataset_id)
            if not ds:
                self._send_json({"ok": False, "error": f"Unknown datasetId: {dataset_id}"}, status=404)
                return

            train_script = Path(self.server.train_script)  # type: ignore[attr-defined]
            python_exe = str(self.server.python_exe)  # type: ignore[attr-defined]
            cache_dir = Path(self.server.cache_dir)  # type: ignore[attr-defined]
            datasets_dir = _private_cache_subdir(cache_dir, "datasets")
            dataset_path = Path(ds["path"]).resolve(strict=True)
            if dataset_path.parent != datasets_dir or not dataset_path.is_file():
                self._send_json(
                    {"ok": False, "error": "Cached dataset path is invalid"},
                    status=HTTPStatus.CONFLICT,
                )
                return
            if dataset_path.stat().st_size != int(ds["sizeBytes"]):
                self._send_json(
                    {"ok": False, "error": "Cached dataset size has changed"},
                    status=HTTPStatus.CONFLICT,
                )
                return

            runs_dir = _private_cache_subdir(cache_dir, "runs")
            out_path = runs_dir / f"{dataset_id}_{int(time.time())}_{secrets.token_hex(4)}.gl1f"

            engine = str(req.get("engine") or "python").strip().lower()
            if engine not in ("python", "cpp"):
                self._send_json({"ok": False, "error": f"Unknown engine: {engine}"}, status=400)
                return
            cpp_train_bin = getattr(self.server, "cpp_train_bin", None)
            if engine == "cpp":
                if not (cpp_train_bin and os.path.exists(cpp_train_bin)):
                    self._send_json({"ok": False, "error": "C++ trainer not available (missing binary)."}, status=400)
                    return
                if not os.access(cpp_train_bin, os.X_OK):
                    self._send_json({"ok": False, "error": f"C++ trainer is not executable: {cpp_train_bin}. Run: chmod +x {cpp_train_bin}"}, status=500)
                    return

            code, out, err = _train_subprocess(
                engine=engine,
                train_script=train_script,
                train_bin=cpp_train_bin,
                dataset_path=dataset_path,
                out_path=out_path,
                req=req,
                python_exe=python_exe,
            )

            if code != 0:
                # Return a concise error (avoid dumping huge logs).
                msg = (err or out or "").strip()
                if len(msg) > 4000:
                    msg = msg[:4000] + "…"
                self._send_json({"ok": False, "error": f"trainer exited with code {code}: {msg}"}, status=500)
                return

            if not out_path.is_file():
                self._send_json(
                    {"ok": False, "error": "trainer produced no model file"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            max_model_bytes = int(
                getattr(self.server, "max_model_bytes", DEFAULT_MAX_MODEL_BYTES)
            )
            model_size = out_path.stat().st_size
            if model_size > max_model_bytes:
                self._send_json(
                    {
                        "ok": False,
                        "error": (
                            f"trainer output exceeds the {max_model_bytes}-byte limit"
                        ),
                    },
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
                return
            gl1f_bytes = out_path.read_bytes()

            model_bytes, pkg = _parse_gl1x_footer(gl1f_bytes)
            meta = None
            curve = None
            if isinstance(pkg, dict):
                local = pkg.get("local") or {}
                if isinstance(local, dict):
                    meta = local.get("trainMeta")
                    curve = local.get("curve")

            resp = {
                "ok": True,
                "modelBytesB64": base64.b64encode(model_bytes).decode("ascii"),
                "meta": meta,
                "curve": curve,
            }
            self._send_json(resp)
        except RequestBodyError as e:
            self._send_json({"ok": False, "error": str(e)}, status=e.status)
        except FileNotFoundError:
            self._send_json(
                {"ok": False, "error": "Cached dataset file is missing"},
                status=HTTPStatus.CONFLICT,
            )
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local trainer server for Forest GL1F")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787)")
    ap.add_argument("--dir", default=None, help="Static file directory (default: directory of this script)")
    ap.add_argument("--cache-dir", default=".trainer_cache", help="Cache directory (default: .trainer_cache)")
    ap.add_argument("--train-script", default="train_gl1f.py", help="Path to train_gl1f.py (default: ./train_gl1f.py)")
    ap.add_argument("--cpp-train-bin", default="train_gl1f_cpp", help="Path to C++ trainer binary (default: ./train_gl1f_cpp)")
    ap.add_argument("--python", dest="python_exe", default=sys.executable, help="Python executable to run train_gl1f.py")
    ap.add_argument(
        "--max-upload-bytes",
        type=int,
        default=DEFAULT_MAX_UPLOAD_BYTES,
        help=f"Maximum dataset upload size (default: {DEFAULT_MAX_UPLOAD_BYTES})",
    )
    ap.add_argument(
        "--max-json-bytes",
        type=int,
        default=DEFAULT_MAX_JSON_BYTES,
        help=f"Maximum training request size (default: {DEFAULT_MAX_JSON_BYTES})",
    )
    ap.add_argument(
        "--max-model-bytes",
        type=int,
        default=DEFAULT_MAX_MODEL_BYTES,
        help=f"Maximum trainer output size (default: {DEFAULT_MAX_MODEL_BYTES})",
    )
    args = ap.parse_args()

    if not _is_loopback_bind_host(args.host):
        ap.error(
            "--host must be an IPv4 loopback address (for example 127.0.0.1); "
            "the trainer API is not an authenticated network service"
        )

    for flag, value in (
        ("--max-upload-bytes", args.max_upload_bytes),
        ("--max-json-bytes", args.max_json_bytes),
        ("--max-model-bytes", args.max_model_bytes),
    ):
        if value < 1:
            ap.error(f"{flag} must be positive")

    base_dir = Path(args.dir or Path(__file__).resolve().parent).resolve()
    if not base_dir.is_dir():
        ap.error(f"--dir is not a directory: {base_dir}")
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        # Prefer keeping cache next to the server (project dir) for consistency.
        cache_dir = (base_dir / cache_dir).resolve()

    try:
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            cache_dir.chmod(0o700)
        except OSError:
            pass
    except PermissionError:
        # If the project directory is read-only / owned by root, fall back to a user-writable cache.
        fallback = Path.home() / ".forest_trainer_cache"
        fallback.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            fallback.chmod(0o700)
        except OSError:
            pass
        print(f"[warn] Cannot create cache dir: {cache_dir} (permission). Using: {fallback}", file=sys.stderr)
        cache_dir = fallback

    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = (base_dir / train_script).resolve()
    if not train_script.is_file():
        print(f"[err] train script not found: {train_script}", file=sys.stderr)
        print("      Put train_gl1f.py next to local_trainer_server.py, or pass --train-script.", file=sys.stderr)
        return 2

    # Bind handler to a fixed directory for static serving.
    import functools
    handler_cls = functools.partial(Handler, directory=str(base_dir))

    httpd = ThreadingHTTPServer((args.host, args.port), handler_cls)
    httpd.daemon_threads = True
    httpd.cache_dir = str(cache_dir)      # type: ignore[attr-defined]
    httpd.train_script = str(train_script) # type: ignore[attr-defined]
    httpd.python_exe = str(args.python_exe) # type: ignore[attr-defined]
    cpp_train_bin = Path(args.cpp_train_bin)
    if not cpp_train_bin.is_absolute():
        cpp_train_bin = base_dir / cpp_train_bin
    httpd.cpp_train_bin = str(cpp_train_bin) # type: ignore[attr-defined]
    httpd.max_upload_bytes = int(args.max_upload_bytes) # type: ignore[attr-defined]
    httpd.max_json_bytes = int(args.max_json_bytes) # type: ignore[attr-defined]
    httpd.max_model_bytes = int(args.max_model_bytes) # type: ignore[attr-defined]

    allowed_api_hosts = {"127.0.0.1", "localhost"}
    allowed_api_hosts.add(str(args.host).strip().lower().strip("[]"))
    httpd.allowed_api_hosts = tuple(sorted(allowed_api_hosts)) # type: ignore[attr-defined]

    print(f"[ok] Serving {base_dir} on http://{args.host}:{args.port}")
    print(f"[ok] Cache dir: {cache_dir}")
    print(f"[ok] Train script: {train_script}")
    print(f"[ok] C++ trainer: {cpp_train_bin} (exists={cpp_train_bin.exists()})")
    print(f"[ok] API Host allowlist: {', '.join(httpd.allowed_api_hosts)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[bye] Shutting down…")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
