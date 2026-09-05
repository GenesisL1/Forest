from __future__ import annotations

import functools
import base64
import http.client
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

import local_trainer_server as server_module


class QuietHandler(server_module.Handler):
    def log_message(self, format: str, *args) -> None:
        pass


class LocalTrainerServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.cache.mkdir(mode=0o700)
        self.train_script = self.root / "trainer.py"
        self.train_script.write_text("raise SystemExit(0)\n", encoding="utf-8")

        handler = functools.partial(QuietHandler, directory=str(self.root))
        self.httpd = server_module.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.httpd.daemon_threads = True
        self.httpd.cache_dir = str(self.cache)
        self.httpd.train_script = str(self.train_script)
        self.httpd.python_exe = os.environ.get("PYTHON", "python3")
        self.httpd.cpp_train_bin = str(self.root / "missing-cpp-trainer")
        self.httpd.max_upload_bytes = 64
        self.httpd.max_json_bytes = 128
        self.httpd.max_model_bytes = 256
        self.httpd.allowed_api_hosts = ("127.0.0.1", "localhost")
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()

        with server_module._DATASETS_LOCK:
            server_module._DATASETS.clear()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        with server_module._DATASETS_LOCK:
            server_module._DATASETS.clear()
        self.temp.cleanup()

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict, dict[str, str]]:
        status, raw, result_headers = self.raw_request(
            method, path, body=body, headers=headers
        )
        return status, json.loads(raw), result_headers

    def raw_request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes, dict[str, str]]:
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.httpd.server_address[1], timeout=5
        )
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        result_headers = {key.lower(): value for key, value in response.getheaders()}
        connection.close()
        return response.status, raw, result_headers

    def test_same_origin_upload_is_atomic_and_has_no_cors_header(self) -> None:
        body = b"x,y\n1,2\n"
        status, payload, headers = self.request(
            "POST",
            "/api/upload?filename=sample.csv",
            body=body,
            headers={
                "Content-Type": "application/octet-stream",
                "Origin": self.origin,
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertNotIn("access-control-allow-origin", headers)
        with server_module._DATASETS_LOCK:
            dataset = dict(server_module._DATASETS[payload["datasetId"]])
        self.assertEqual(Path(dataset["path"]).read_bytes(), body)
        self.assertEqual(list((self.cache / "datasets").glob(".tmp_*")), [])

    def test_cross_origin_mutation_and_preflight_are_rejected(self) -> None:
        headers = {
            "Content-Type": "application/octet-stream",
            "Origin": "https://attacker.invalid",
        }
        status, _, response_headers = self.request(
            "POST", "/api/upload?filename=x.csv", body=b"x\n1\n", headers=headers
        )
        self.assertEqual(status, 403)
        self.assertNotIn("access-control-allow-origin", response_headers)

        status, _, response_headers = self.request(
            "OPTIONS",
            "/api/upload",
            headers={
                "Origin": "https://attacker.invalid",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(status, 405)
        self.assertNotIn("access-control-allow-origin", response_headers)

        status, _, _ = self.request(
            "POST", "/api/stop", headers={"Sec-Fetch-Site": "cross-site"}
        )
        self.assertEqual(status, 403)

    def test_dns_rebinding_host_is_rejected(self) -> None:
        status, payload, _ = self.request(
            "GET", "/api/ping", headers={"Host": "attacker.invalid"}
        )
        self.assertEqual(status, 403)
        self.assertFalse(payload["ok"])

    def test_only_loopback_bind_hosts_are_accepted(self) -> None:
        for value in ("127.0.0.1", "127.0.0.2", "localhost"):
            self.assertTrue(server_module._is_loopback_bind_host(value), value)
        for value in ("0.0.0.0", "::", "::1", "192.0.2.1", "trainer.example"):
            self.assertFalse(server_module._is_loopback_bind_host(value), value)

    def test_static_server_uses_public_allowlist(self) -> None:
        (self.root / "create.html").write_text("create", encoding="utf-8")
        source_dir = self.root / "src"
        source_dir.mkdir()
        (source_dir / "create_page.js").write_text("export {};", encoding="utf-8")

        for path in ("/create.html", "/src/create_page.js"):
            status, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 200, path)

        for path in (
            "/trainer.py",
            "/.git/config",
            "/MANIFEST.sha256",
            "/mint_model.py",
            "/%2egit/config",
        ):
            status, _, _ = self.raw_request("GET", path)
            self.assertEqual(status, 404, path)

        if hasattr(os, "symlink"):
            (source_dir / "exposed.js").symlink_to(self.train_script)
            status, _, _ = self.raw_request("GET", "/src/exposed.js")
            self.assertEqual(status, 404)

    def test_oversize_upload_is_rejected_before_file_creation(self) -> None:
        self.httpd.max_upload_bytes = 4
        status, _, _ = self.request(
            "POST",
            "/api/upload?filename=x.csv",
            body=b"12345",
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(status, 413)
        datasets = self.cache / "datasets"
        self.assertFalse(datasets.exists() and any(datasets.iterdir()))

    def test_short_stream_removes_partial_file(self) -> None:
        destination = self.root / "partial"
        with self.assertRaises(server_module.RequestBodyError):
            server_module._read_body_stream(io.BytesIO(b"abc"), 4, destination)
        self.assertFalse(destination.exists())

    def test_symlinked_cache_subdirectory_is_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        outside = self.root / "outside"
        outside.mkdir()
        (self.cache / "datasets").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            server_module._private_cache_subdir(self.cache, "datasets")

    def test_json_limit_and_training_slot_are_enforced(self) -> None:
        self.httpd.max_json_bytes = 4
        status, _, _ = self.request(
            "POST",
            "/api/train",
            body=b'{"datasetId":"x"}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 413)

        self.httpd.max_json_bytes = 128
        # The client can finish reading the 413 response a few microseconds
        # before the request thread reaches its finally block and releases the
        # slot. A bounded wait distinguishes that cleanup race from a leak.
        self.assertTrue(server_module._TRAIN_SLOT_LOCK.acquire(timeout=5.0))
        try:
            status, payload, _ = self.request(
                "POST",
                "/api/train",
                body=b'{"datasetId":"x"}',
                headers={"Content-Type": "application/json"},
            )
        finally:
            server_module._TRAIN_SLOT_LOCK.release()
        self.assertEqual(status, 409)
        self.assertIn("already running", payload["error"])

    def test_localhost_upload_then_training_workflow_still_operates(self) -> None:
        self.httpd.max_json_bytes = 512
        self.train_script.write_text(
            """\
import argparse
import struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args, _ = parser.parse_known_args()
header = b"GL1F" + bytes((1, 0)) + struct.pack("<HHIiIH", 1, 1, 1, 0, 1000, 0)
node = struct.pack("<HiH", 0, 0, 0)
leaves = struct.pack("<ii", -1, 1)
Path(args.out).write_bytes(header + node + leaves)
""",
            encoding="utf-8",
        )
        status, upload, _ = self.request(
            "POST",
            "/api/upload?filename=sample.csv",
            body=b"x,y\n1,2\n",
            headers={
                "Content-Type": "application/octet-stream",
                "Origin": self.origin,
            },
        )
        self.assertEqual(status, 200)

        request = json.dumps(
            {
                "datasetId": upload["datasetId"],
                "engine": "python",
                "featureCols": ["x"],
                "labelCol": "y",
            }
        ).encode("utf-8")
        status, trained, _ = self.request(
            "POST",
            "/api/train",
            body=request,
            headers={"Content-Type": "application/json", "Origin": self.origin},
        )
        self.assertEqual(status, 200)
        self.assertTrue(trained["ok"])
        model = base64.b64decode(trained["modelBytesB64"])
        self.assertEqual(model[:4], b"GL1F")
        self.assertEqual(len(model), 40)


if __name__ == "__main__":
    unittest.main()
