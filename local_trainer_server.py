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
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import secrets
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


VERSION = "1.0.0"


# ----------------------------
# In-memory dataset registry
# ----------------------------

_DATASETS_LOCK = threading.Lock()
_DATASETS: Dict[str, Dict[str, Any]] = {}  # datasetId -> {path, filename, sizeBytes, columns, createdAt}

_ACTIVE_PROC_LOCK = threading.Lock()
_ACTIVE_PROC: Optional[subprocess.Popen] = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _read_body_stream(rfile, length: int, out_path: Path, chunk_size: int = 1024 * 1024) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("wb") as f:
        remaining = length
        while remaining > 0:
            chunk = rfile.read(min(chunk_size, remaining))
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            remaining -= len(chunk)
    return written



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
        # Try gentle terminate first.
        proc.terminate()
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
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

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(train_script.parent),
        text=True,
    )

    global _ACTIVE_PROC
    with _ACTIVE_PROC_LOCK:
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

    def _set_cors(self) -> None:
        # Allow cross-origin usage if the UI is served from a different origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, obj: Any, status: int = 200) -> None:
        data = _json_bytes(obj)
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            raise ValueError("Invalid JSON request body")

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/ping":
            cpp_bin = getattr(self.server, "cpp_train_bin", None)
            supports_cpp = bool(cpp_bin) and os.path.exists(cpp_bin)
            self._send_json({"ok": True, "version": VERSION, "time": _now_iso(), "supportsCpp": supports_cpp})
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/upload":
            return self._handle_upload(parsed)
        if parsed.path == "/api/train":
            return self._handle_train()
        if parsed.path == "/api/stop":
            return self._handle_stop()
        self._send_json({"ok": False, "error": f"Unknown endpoint: {parsed.path}"}, status=404)

    def _handle_upload(self, parsed_url: urllib.parse.ParseResult):
        try:
            qs = urllib.parse.parse_qs(parsed_url.query or "")
            filename = _safe_filename((qs.get("filename") or ["dataset.csv"])[0])

            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._send_json({"ok": False, "error": "Missing Content-Length"}, status=400)
                return

            dataset_id = secrets.token_hex(16)
            ext = os.path.splitext(filename)[1] or ".csv"
            cache_dir = Path(self.server.cache_dir)  # type: ignore[attr-defined]
            ds_path = cache_dir / "datasets" / f"{dataset_id}{ext}"
            tmp_path = cache_dir / "datasets" / f".tmp_{dataset_id}{ext}"

            # Stream request body to disk.
            written = _read_body_stream(self.rfile, length, tmp_path)
            tmp_path.replace(ds_path)

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
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=500)

    def _handle_stop(self):
        stopped = False
        with _ACTIVE_PROC_LOCK:
            proc = _ACTIVE_PROC
        if proc is not None and proc.poll() is None:
            stopped = True
            _kill_process_tree(proc)
        self._send_json({"ok": True, "stopped": stopped})

    def _handle_train(self):
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
            runs_dir = cache_dir / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            out_path = runs_dir / f"{dataset_id}_{int(time.time())}_{secrets.token_hex(4)}.gl1f"

            # Ensure only one training proc at a time (simpler UX).
            with _ACTIVE_PROC_LOCK:
                if _ACTIVE_PROC is not None and _ACTIVE_PROC.poll() is None:
                    self._send_json({"ok": False, "error": "Training already running"}, status=409)
                    return

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
                dataset_path=Path(ds["path"]),
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
    args = ap.parse_args()

    base_dir = Path(args.dir or Path(__file__).resolve().parent).resolve()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        # Prefer keeping cache next to the server (project dir) for consistency.
        cache_dir = (base_dir / cache_dir).resolve()

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # If the project directory is read-only / owned by root, fall back to a user-writable cache.
        fallback = Path.home() / ".forest_trainer_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"[warn] Cannot create cache dir: {cache_dir} (permission). Using: {fallback}", file=sys.stderr)
        cache_dir = fallback

    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = (base_dir / train_script).resolve()
    if not train_script.exists():
        print(f"[err] train script not found: {train_script}", file=sys.stderr)
        print("      Put train_gl1f.py next to local_trainer_server.py, or pass --train-script.", file=sys.stderr)
        return 2

    # Bind handler to a fixed directory for static serving.
    import functools
    handler_cls = functools.partial(Handler, directory=str(base_dir))

    httpd = ThreadingHTTPServer((args.host, args.port), handler_cls)
    httpd.cache_dir = str(cache_dir)      # type: ignore[attr-defined]
    httpd.train_script = str(train_script) # type: ignore[attr-defined]
    httpd.python_exe = str(args.python_exe) # type: ignore[attr-defined]
    cpp_train_bin = Path(args.cpp_train_bin)
    if not cpp_train_bin.is_absolute():
        cpp_train_bin = base_dir / cpp_train_bin
    httpd.cpp_train_bin = str(cpp_train_bin) # type: ignore[attr-defined]

    print(f"[ok] Serving {base_dir} on http://{args.host}:{args.port}")
    print(f"[ok] Cache dir: {cache_dir}")
    print(f"[ok] Train script: {train_script}")
    print(f"[ok] C++ trainer: {cpp_train_bin} (exists={cpp_train_bin.exists()})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[bye] Shutting down…")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())