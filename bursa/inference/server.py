import json
import subprocess
import time
import urllib.request
import urllib.error


def build_server_args(model_path, port=8080, threads=4, ctx=2048) -> list[str]:
    return [
        "llama-server", "--model", model_path, "--ctx-size", str(ctx),
        "--threads", str(threads), "--temp", "0",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--offline",
        "--port", str(port),
    ]


class LlamaServer:
    def __init__(self, model_path, port=8080, threads=4, ctx=2048):
        self.args = build_server_args(model_path, port, threads, ctx)
        self.port = port
        self._proc = None

    def start(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(self.args, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    @property
    def process_id(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                return json.loads(r.read()).get("status") in ("ok", "ready")
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.health():
                return True
            if self._proc is not None and self._proc.poll() is not None:
                return False
            time.sleep(0.1)
        return False

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
