"""ServerProcess — manages a single llama-server subprocess.

Handles command building, process start/stop, health polling, and
stderr log tailing for crash diagnostics. No KutAI imports — standalone.
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class ServerProcess:
    """Manages one llama-server child process."""

    def __init__(self, config, platform_helper) -> None:
        """
        Parameters
        ----------
        config:
            ``DaLLaMaConfig`` — engine-level settings (path, port, host).
        platform_helper:
            ``PlatformHelper`` — OS-specific process creation and teardown.
        """
        self._cfg = config
        self._platform = platform_helper
        self.process: Optional[subprocess.Popen] = None
        self._stderr_path: str = ""

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def api_base(self) -> str:
        """Base URL for llama-server REST API."""
        return f"http://{self._cfg.host}:{self._cfg.port}"

    def build_cmd(self, config) -> list[str]:
        """Build the llama-server command list from a ``ServerConfig``.

        Rules (in order):
        - Always: executable, --model, --alias, --port, --host, --ctx-size,
          --flash-attn auto, --metrics, --batch-size, --ubatch-size
        - Threads: os.cpu_count() // 2 - 2, minimum 2; skip if cpu_count unknown
        - Jinja: add --jinja unless --no-jinja is in extra_flags
        - Thinking: only when thinking=True AND --no-jinja NOT in extra_flags →
          add --reasoning on
        - Vision: if vision_projector non-empty → --mmproj <path>
        - Extra flags: appended verbatim at end
        """
        cmd: list[str] = [
            self._cfg.llama_server_path,
            "--model", config.model_path,
            "--alias", "local-model",
            "--port", str(self._cfg.port),
            "--host", self._cfg.host,
            "--ctx-size", str(config.context_length),
            "--flash-attn", "auto",
            "--parallel", "1",
            "--metrics",
            "--batch-size", "2048",
            "--ubatch-size", "512",
        ]

        # Threads
        cpu = os.cpu_count()
        if cpu is not None:
            threads = max(2, cpu // 2 - 2)
            cmd += ["--threads", str(threads)]

        # Jinja / thinking
        no_jinja = "--no-jinja" in (config.extra_flags or [])
        if not no_jinja:
            cmd.append("--jinja")
            if config.thinking:
                cmd += ["--reasoning", "on"]

        # Vision
        if config.vision_projector:
            cmd += ["--mmproj", config.vision_projector]

        # Extra flags (verbatim)
        if config.extra_flags:
            cmd.extend(config.extra_flags)

        return cmd

    async def start(self, config, load_timeout: float = 0.0) -> bool:
        """Launch llama-server and wait until healthy.

        Parameters
        ----------
        load_timeout:
            Caller-provided ceiling for the health-wait.  When >0 the
            actual timeout is ``min(internal_estimate, load_timeout)``.

        Returns True on success, False if the process fails to become healthy.
        """
        if self.process is not None:
            if self.process.poll() is None:
                # Process object exists AND is still alive — skip re-start
                logger.warning("start() called while process is already running (pid=%s)", self.process.pid)
                return True
            # Process object exists but process is dead — clean up and proceed
            logger.info("start() found dead process (pid=%s, rc=%s) — cleaning up",
                        self.process.pid, self.process.returncode)
            self._platform._close_stderr(self.process)
            self.process = None

        cmd = self.build_cmd(config)
        log_dir = os.environ.get("DALLAMA_LOG_DIR", ".")
        self._stderr_path = os.path.join(log_dir, "llama-server.stderr.log")

        logger.info("Starting llama-server: %s", " ".join(cmd))

        try:
            self.process = self._platform.create_process(cmd, self._stderr_path)
        except Exception as exc:
            logger.error("Failed to create llama-server process: %s", exc)
            return False

        timeout = self._estimate_load_timeout(config)
        if load_timeout > timeout:
            timeout = load_timeout
        logger.info("Waiting up to %.0fs for llama-server to become healthy", timeout)

        healthy = await self._wait_for_healthy(timeout)
        if not healthy:
            tail = self.read_stderr_tail(20)
            logger.error(
                "llama-server failed to become healthy. Last stderr:\n%s", tail
            )
            await self.stop()
            return False

        logger.info("llama-server healthy at %s (model=%s)", self.api_base, config.model_name)
        return True

    async def stop(self) -> None:
        """Gracefully stop the llama-server process."""
        if self.process is None:
            return
        proc = self.process
        self.process = None
        logger.info("Stopping llama-server (pid=%s)…", proc.pid)
        await self._platform.graceful_stop(proc)
        logger.info("llama-server stopped")

    async def health_check(self) -> bool:
        """Return True if /health returns HTTP 200, False otherwise."""
        return (await self._health_check_status()) == 200

    async def _health_check_status(self) -> int:
        """Return the HTTP status code from /health, or 0 on failure."""
        if not self.is_alive():
            return 0

        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.api_base}/health")
                return resp.status_code
        except Exception:
            return 0

    def is_alive(self) -> bool:
        """Return True if the subprocess is running (poll() is None)."""
        return self.process is not None and self.process.poll() is None

    def read_stderr_tail(self, lines: int = 10) -> str:
        """Read the last *lines* lines of the stderr log file.

        Returns an empty string if the file does not exist or cannot be read.
        """
        if not self._stderr_path:
            return ""
        try:
            with open(self._stderr_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            all_lines = content.splitlines()
            return "\n".join(all_lines[-lines:])
        except Exception:
            return ""

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _wait_for_healthy(self, timeout: float) -> bool:
        """Poll /health until HTTP 200 or timeout.

        Interval grows from 1 s to 3 s after the first few attempts to avoid
        hammering the server while it loads weights.
        """
        import asyncio

        elapsed = 0.0
        interval = 1.0

        while elapsed < timeout:
            # Check if process died while we were waiting
            if not self.is_alive():
                logger.warning("llama-server process exited during health wait")
                return False

            if await self.health_check():
                return True

            await asyncio.sleep(interval)
            elapsed += interval
            # Ramp up to 3 s after 10 s elapsed
            if elapsed >= 10.0:
                interval = 3.0

        return False

    def _estimate_load_timeout(self, config) -> float:
        """Estimate how long loading this model might take.

        Formula: ``file_size_mb / 500 + ctx_factor * 15``
        where ctx_factor scales context length (4096 baseline → 1.0).
        Result clamped to [30, 150].  Observed worst case is ~113s
        for the largest models on an 8 GB GPU.
        """
        try:
            size_bytes = os.path.getsize(config.model_path)
            size_mb = size_bytes / (1024 * 1024)
        except Exception:
            size_mb = 4096.0  # assume 4 GB if file unreadable

        ctx_factor = max(1.0, config.context_length / 4096)
        estimate = size_mb / 500 + ctx_factor * 15
        return max(30.0, min(150.0, estimate))
