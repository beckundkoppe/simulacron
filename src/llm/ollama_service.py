import contextlib
import os
import shutil
import subprocess
import time
from typing import Optional

import requests


class OllamaServiceManager:
    """
    Starts/stops ``ollama serve`` on demand and only tears down instances we
    spawned ourselves, leaving user-managed daemons untouched.
    """

    def __init__(self) -> None:
        host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
        self._base_url = host if host.startswith("http") else f"http://{host}"
        self._health_url = f"{self._base_url}/api/tags"
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._started_by_us = False

    def _refresh_process_handle(self) -> None:
        """Reset the local handle when the child process has exited."""
        if self._process is not None and self._process.poll() is not None:
            self._process = None
            self._started_by_us = False

    def is_running(self) -> bool:
        """
        Lightweight readiness probe against the local Ollama endpoint.
        """
        self._refresh_process_handle()
        try:
            response = requests.get(self._health_url, timeout=0.5)
        except requests.RequestException:
            return False
        return response.ok

    def start(self, timeout_s: float = 15.0) -> bool:
        """
        Ensure ``ollama serve`` is up. Returns True if we spawned a new process
        in this call, False if it was already reachable.
        """
        self._refresh_process_handle()
        if self.is_running():
            # Preserve ownership flag if we still have a live handle.
            if self._process is None:
                self._started_by_us = False
            return False

        if not shutil.which("ollama"):
            raise RuntimeError("ollama executable not found in PATH; install Ollama or adjust PATH.")

        # Silence stdout/stderr to keep benchmark logs clean.
        self._process = subprocess.Popen(  # noqa: PLW1501 - intentional background service
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._started_by_us = True

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.is_running():
                return True
            if self._process and self._process.poll() is not None:
                break
            time.sleep(0.25)

        # Startup failed or timed out; clean up and raise.
        self.stop()
        raise RuntimeError("Timed out waiting for ollama serve to become ready.")

    def stop(self, timeout_s: float = 5.0) -> None:
        """
        Terminate a service instance we started. External services are left
        untouched.
        """
        self._refresh_process_handle()
        if not self._started_by_us or self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=timeout_s)

        self._process = None
        self._started_by_us = False

    @contextlib.contextmanager
    def ensure_running(self, needed: bool):
        """
        Context manager to wrap a run.

        Auto-start/stop is disabled: if a run requires Ollama and no service is
        reachable, we raise immediately so the caller can start it manually.
        """
        if not needed:
            # No Ollama dependency for this run; do nothing.
            yield False
            return

        if not self.is_running():
            raise RuntimeError(
                "Ollama must be running (ollama serve) before starting this run."
            )

        # Service is already up; leave it untouched.
        yield False
