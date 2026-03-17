# tools/shell.py
"""
Execute shell commands inside a Docker sandbox container.
The workspace directory is bind-mounted so file changes persist.
"""

import asyncio
import os
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("tools.shell")

# ---------------------------------------------------------------------------
# Configuration — override via environment or config import
# ---------------------------------------------------------------------------
WORKSPACE_DIR: str = os.environ.get(
    "WORKSPACE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace")),
)
CONTAINER_NAME: str = os.environ.get("SANDBOX_CONTAINER", "orchestrator-sandbox")
SANDBOX_IMAGE: str = os.environ.get("SANDBOX_IMAGE", "orchestrator-sandbox:latest")
SANDBOX_NETWORK: str = os.environ.get("SANDBOX_NETWORK", "bridge")
SANDBOX_MEMORY: str = os.environ.get("SANDBOX_MEMORY", "512m")
SANDBOX_CPUS: str = os.environ.get("SANDBOX_CPUS", "1.0")
MAX_OUTPUT_CHARS: int = int(os.environ.get("SANDBOX_MAX_OUTPUT", "8000"))
CONTAINER_WORKROOT: str = "/app/workspace"

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
BLOCKED_PATTERNS: set[str] = {
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",        # fork bomb
    "chmod -r 777 /",
    "shutdown",
    "reboot",
    "curl | sh",
    "curl | bash",
    "wget | sh",
    "wget | bash",
}


# Phase 8.2: Per-agent-type command allowlists (first token of command)
# None = no restriction beyond blocklist
AGENT_COMMAND_ALLOWLIST: dict[str, set[str] | None] = {
    "coder": {
        "python", "python3", "pip", "pip3", "npm", "node", "npx",
        "go", "cargo", "rustc", "git", "cat", "ls", "mkdir", "cp",
        "mv", "grep", "find", "head", "tail", "wc", "sort", "curl",
        "echo", "touch", "chmod", "pytest", "jest", "ruff", "mypy",
        "black", "flake8", "eslint", "tsc", "make",
    },
    "reviewer": {
        "pytest", "python", "python3", "npm", "node", "npx",
        "cat", "ls", "grep", "find", "head", "tail",
        "ruff", "mypy", "black", "flake8", "eslint", "tsc",
        "git", "wc", "sort",
    },
    "test_generator": {
        "python", "python3", "pip", "pip3", "npm", "node", "npx",
        "pytest", "jest", "cat", "ls", "grep", "find", "head", "tail",
        "git", "ruff", "mypy",
    },
    "fixer": {
        "python", "python3", "pip", "pip3", "npm", "node", "npx",
        "go", "cargo", "rustc", "git", "cat", "ls", "mkdir", "cp",
        "mv", "grep", "find", "head", "tail", "wc", "sort", "curl",
        "echo", "touch", "pytest", "jest", "ruff", "mypy",
    },
}


def _is_command_blocked(command: str) -> bool:
    """Return True if the command matches any blocked pattern."""
    lower = command.lower().strip()
    return any(pattern in lower for pattern in BLOCKED_PATTERNS)


def _is_command_allowed_for_agent(command: str, agent_type: str) -> bool:
    """Return True if command is allowed for the given agent type (Phase 8.2)."""
    allowlist = AGENT_COMMAND_ALLOWLIST.get(agent_type)
    if allowlist is None:
        return True  # no restriction
    # Extract the first token (the executable name)
    first_token = command.strip().split()[0] if command.strip() else ""
    # Strip path prefix (e.g. /usr/bin/python → python)
    first_token = first_token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if first_token in allowlist:
        return True
    logger.warning(f"Shell command '{first_token}' not in allowlist for agent '{agent_type}'"
                   f" — allowed: {sorted(allowlist)[:10]}...")
    return False


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------
async def _run_quiet(*args: str) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def ensure_container_running() -> bool:
    """Make sure the sandbox Docker container is up, restarting or creating as needed."""
    # 1. Already running?
    rc, stdout, _ = await _run_quiet(
        "docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME,
    )
    if rc == 0 and "true" in stdout.lower():
        return True

    # 2. Exists but stopped? Try restart.
    rc, _, _ = await _run_quiet("docker", "start", CONTAINER_NAME)
    if rc == 0:
        logger.info("restarted existing sandbox container", container=CONTAINER_NAME)
        return True

    # 3. Doesn't exist — create from scratch.
    logger.info("creating new sandbox container", container=CONTAINER_NAME)
    rc, _, stderr = await _run_quiet(
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--network", SANDBOX_NETWORK,
        "--add-host", "host.docker.internal:host-gateway",
        "--memory", SANDBOX_MEMORY,
        "--cpus", SANDBOX_CPUS,
        "-v", f"{WORKSPACE_DIR}:{CONTAINER_WORKROOT}",
        "-w", CONTAINER_WORKROOT,
        SANDBOX_IMAGE,
        "sleep", "infinity",
    )
    if rc != 0:
        logger.error("failed to create sandbox container", error=stderr.strip())
        return False

    logger.info("sandbox container created and running")
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_workdir(workdir: Optional[str]) -> str:
    """Ensure workdir stays under the container work-root."""
    if workdir is None:
        return CONTAINER_WORKROOT
    # Already absolute and under root
    if workdir.startswith(CONTAINER_WORKROOT):
        return workdir
    # Relative path → anchor under work-root
    return f"{CONTAINER_WORKROOT}/{workdir.lstrip('/')}"


def _format_output(
    stdout: bytes | str,
    stderr: bytes | str,
    exit_code: int,
) -> str:
    """Build a human-readable result string with optional truncation."""

    def _decode(data: bytes | str) -> str:
        return data.decode(errors="replace") if isinstance(data, bytes) else data

    parts: list[str] = []
    out_text = _decode(stdout).strip()
    err_text = _decode(stderr).strip()

    if out_text:
        parts.append(out_text)
    if err_text:
        parts.append(f"[STDERR]\n{err_text}")

    body = "\n".join(parts) if parts else "(no output)"

    # Status prefix
    if exit_code == 0:
        result = f"✅\n{body}"
    else:
        result = f"❌ (exit code {exit_code})\n{body}"

    # Truncate for context-window safety
    if len(result) > MAX_OUTPUT_CHARS:
        result = (
            result[: MAX_OUTPUT_CHARS - 120]
            + f"\n\n… [truncated — {len(result)} chars total]"
        )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def run_shell(
    command: str,
    timeout: int = 60,
    workdir: Optional[str] = None,
) -> str:
    """
    Execute a shell command inside the Docker sandbox.

    Args:
        command:  The shell command to run.
        timeout:  Max seconds to wait (default 60).
        workdir:  Working directory inside the container
                  (absolute under /app/workspace, or relative to it).

    Returns:
        Combined stdout + stderr with exit-code indicator,
        truncated to MAX_OUTPUT_CHARS.
    """
    if _is_command_blocked(command):
        return "🚫 BLOCKED: This command matched a safety filter and was not executed."

    if not await ensure_container_running():
        return (
            "❌ Docker sandbox is not available.\n"
            f"Run:\n  docker build -t {SANDBOX_IMAGE} .\n"
            f"  docker run -d --name {CONTAINER_NAME} "
            f"-v {WORKSPACE_DIR}:{CONTAINER_WORKROOT} "
            f"{SANDBOX_IMAGE} sleep infinity"
        )

    resolved_wd = _resolve_workdir(workdir)
    logger.debug("executing shell command", command=command[:120], workdir=resolved_wd)

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec",
            "-w", resolved_wd,
            CONTAINER_NAME,
            "bash", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Best-effort cleanup: kill the process tree inside the container
            await _run_quiet(
                "docker", "exec", CONTAINER_NAME,
                "bash", "-c", "kill -9 -1 2>/dev/null || true",
            )
            proc.kill()          # also kill the local docker-exec process
            await proc.wait()
            return f"⏱️ TIMEOUT: Command exceeded {timeout}s limit.\nCommand: {command[:120]}"

        exit_code = proc.returncode or 0
        logger.info("shell command completed", exit_code=exit_code)
        return _format_output(stdout, stderr, exit_code)

    except Exception as exc:
        logger.exception("unexpected shell error", error=str(exc))
        return f"❌ Shell execution error: {type(exc).__name__}: {exc}"


async def run_shell_with_stdin(
    command: str,
    stdin_data: str,
    timeout: int = 60,
    workdir: Optional[str] = None,
) -> str:
    """
    Run a command inside the sandbox and pipe *stdin_data* to its stdin.

    Useful for writing files via heredoc, feeding input to interactive
    programs, etc.
    """
    if _is_command_blocked(command):
        return "🚫 BLOCKED: This command matched a safety filter."

    if not await ensure_container_running():
        return "❌ Docker sandbox is not available."

    resolved_wd = _resolve_workdir(workdir)

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i",
            "-w", resolved_wd,
            CONTAINER_NAME,
            "bash", "-c", command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_data.encode()), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"⏱️ TIMEOUT after {timeout}s"

        return _format_output(stdout, stderr, proc.returncode or 0)

    except Exception as exc:
        logger.exception("shell with stdin error", error=str(exc))
        return f"❌ Shell execution error: {type(exc).__name__}: {exc}"


async def run_shell_sequential(
    commands: list[str],
    timeout: int = 120,
    workdir: Optional[str] = None,
    stop_on_error: bool = True,
) -> str:
    """
    Run multiple commands sequentially inside the sandbox.

    Args:
        commands:       Ordered list of shell commands.
        timeout:        Per-command timeout in seconds.
        workdir:        Working directory (same for all commands).
        stop_on_error:  If True, stop on the first non-zero exit code.

    Returns:
        Combined output of all executed commands.
    """
    outputs: list[str] = []

    for i, cmd in enumerate(commands, 1):
        header = f"── [{i}/{len(commands)}] $ {cmd}"
        result = await run_shell(cmd, timeout=timeout, workdir=workdir)
        outputs.append(f"{header}\n{result}")

        if stop_on_error and result.startswith("❌"):
            outputs.append(f"\n⛔ Stopped at command {i}/{len(commands)} due to error.")
            break

    return "\n\n".join(outputs)
