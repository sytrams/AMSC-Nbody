from __future__ import annotations

import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def request(url: str) -> tuple[int, bytes, dict[str, str]]:
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.status, response.read(), dict(response.headers.items())


def wait_until_ready(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output, _ = process.communicate()
            raise AssertionError(f"Viewer exited during startup:\n{output}")
        try:
            status, _, _ = request(f"{base_url}/api/health")
            if status == 200:
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise AssertionError("Viewer did not become ready")


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: test.py VIEWER FIXTURE_WRITER ASSET_DIR")
    viewer, fixture_writer, asset_dir = sys.argv[1:]

    with tempfile.TemporaryDirectory(prefix="nbody-browser-test-") as temporary:
        frame_dir = Path(temporary) / "frames"
        subprocess.run([fixture_writer, str(frame_dir)], check=True)
        port = reserve_port()
        base_url = f"http://127.0.0.1:{port}"
        process = subprocess.Popen(
            [
                viewer,
                "--frames-dir",
                str(frame_dir),
                "--assets-dir",
                asset_dir,
                "--bind",
                "127.0.0.1",
                "--port",
                str(port),
                "--poll-ms",
                "20",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_until_ready(base_url, process)

            status, html, headers = request(f"{base_url}/")
            assert status == 200
            assert b'id="simulation-canvas"' in html
            assert b'type="module"' in html
            assert "default-src 'self'" in headers["Content-Security-Policy"]

            status, parser_module, headers = request(
                f"{base_url}/frame_parser.mjs"
            )
            assert status == 200
            assert headers["Content-Type"].startswith("text/javascript")
            assert b"export function parseFrame" in parser_module

            status, body, _ = request(f"{base_url}/api/frames")
            assert status == 200
            catalog = json.loads(body)
            assert catalog["runId"] == "browser-test"
            assert catalog["complete"] is True
            assert catalog["frameCount"] == 2
            assert catalog["frames"][0]["hasTypes"] is True

            frame_name = catalog["frames"][0]["name"]
            status, frame, _ = request(f"{base_url}/api/frame/{frame_name}")
            assert status == 200
            assert frame[:8] == b"NBSNAP01"
            version, header_bytes = struct.unpack_from("<II", frame, 8)
            assert version == 3
            assert header_bytes == 128
            particle_count = struct.unpack_from("<Q", frame, 48)[0]
            type_bytes = struct.unpack_from("<Q", frame, 120)[0]
            assert particle_count == 4
            assert type_bytes == particle_count

            output_prefix = process.stdout.readline() if process.stdout else ""
            assert "N-body browser viewer is ready" in output_prefix
            output_remainder = process.stdout.readline() if process.stdout else ""
            output_remainder += process.stdout.readline() if process.stdout else ""
            assert base_url in output_remainder
        finally:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
