# tools/shell.py
"""
Execute shell commands inside a Docker sandbox container.
The workspace directory is bind-mounted so file changes persist.
"""

import asyncio
import os
from typing import Optional

import logging

logger = logging.getLogger(__name__)

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
# Host-local fallback (when Docker is unavailable)
# Set SANDBOX_MODE=local to always use host subprocess,
# or leave as "docker" to auto-fallback when container is unreachable.
# ---------------------------------------------------------------------------
SANDBOX_MODE: str = os.environ.get("SANDBOX_MODE", "docker")  # docker | local | none
LOCAL_SHELL_TIMEOUT: int = int(os.environ.get("LOCAL_SHELL_TIMEOUT", "60"))
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
# Commands additionally blocked in local (host) mode for safety
LOCAL_BLOCKED_PATTERNS: set[str] = BLOCKED_PATTERNS | {
    "sudo ", "su -", "systemctl", "service ",
    "docker ", "chown ", "mount ", "umount ",
    "useradd", "userdel", "passwd",
}



# Phase 8.2: Per-agent-type command allowlists (first token of command)
# None = no restriction beyond blocklist
AGENT_COMMAND_ALLOWLIST: dict[str, set[str] | None] = {
    "coder": {
        "python", "python3", "pip", "pip3", "npm", "node", "npx",
        "go", "cargo", "rustc", "git", "cat", "ls", "mkdir", "cp",
        "mv", "rm", "grep", "find", "head", "tail", "wc", "sort", "curl",
        "echo", "touch", "chmod", "pytest", "jest", "ruff", "mypy",
        "black", "flake8", "eslint", "tsc", "make",
        # Setup/scaffolding commands
        "cd", "pwd", "env", "export", "which", "whoami",
        "tar", "unzip", "sed", "awk", "tee", "xargs",
        "yarn", "pnpm", "bunx", "bun", "deno",
        "uvicorn", "gunicorn", "flask", "django-admin",
        "prisma", "drizzle-kit", "sequelize", "typeorm",
        "vite", "next", "create-react-app", "create-next-app",
        "docker-compose",
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
        "mv", "rm", "grep", "find", "head", "tail", "wc", "sort", "curl",
        "echo", "touch", "chmod", "pytest", "jest", "ruff", "mypy",
        "cd", "pwd", "sed", "awk", "tee",
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
# Host-local fallback execution
# ---------------------------------------------------------------------------
def _is_command_blocked_local(command: str) -> bool:
    """Stricter blocklist for host-local execution."""
    lower = command.lower().strip()
    return any(pattern in lower for pattern in LOCAL_BLOCKED_PATTERNS)


async def _run_local_shell(
    command: str,
    timeout: int = 60,
    workdir: Optional[str] = None,
) -> str:
    """
    Execute a shell command directly on the host as a subprocess.

    Used as fallback when Docker is unavailable. Applies stricter
    safety checks since we are running on the host machine.
    """
    if _is_command_blocked_local(command):
        return "🚫 BLOCKED: This command is not allowed in host-local mode."

    # Resolve working directory to workspace
    cwd = WORKSPACE_DIR
    if workdir:
        if os.path.isabs(workdir):
            # Only allow paths under workspace
            if not os.path.normpath(workdir).startswith(os.path.normpath(WORKSPACE_DIR)):
                cwd = WORKSPACE_DIR
            else:
                cwd = workdir
        else:
            cwd = os.path.join(WORKSPACE_DIR, workdir)

    logger.debug("executing local shell command", command=command[:120], cwd=cwd)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"⏱️ TIMEOUT: Command exceeded {timeout}s limit.\nCommand: {command[:120]}"

        exit_code = proc.returncode or 0
        logger.info("local shell command completed", exit_code=exit_code)
        return _format_output(stdout, stderr, exit_code)

    except Exception as exc:
        logger.exception("local shell error", error=str(exc))
        return f"❌ Local shell execution error: {type(exc).__name__}: {exc}"


async def _run_local_shell_with_stdin(
    command: str,
    stdin_data: str,
    timeout: int = 60,
    workdir: Optional[str] = None,
) -> str:
    """Host-local shell execution with stdin piping."""
    if _is_command_blocked_local(command):
        return "🚫 BLOCKED: This command is not allowed in host-local mode."

    cwd = WORKSPACE_DIR
    if workdir:
        candidate = os.path.join(WORKSPACE_DIR, workdir) if not os.path.isabs(workdir) else workdir
        if os.path.normpath(candidate).startswith(os.path.normpath(WORKSPACE_DIR)):
            cwd = candidate

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
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
        logger.exception("local shell with stdin error", error=str(exc))
        return f"❌ Local shell execution error: {type(exc).__name__}: {exc}"


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

    use_local = SANDBOX_MODE == "local"
    if not use_local and not await ensure_container_running():
        if SANDBOX_MODE == "none":
            return "⚠️ Shell execution skipped (SANDBOX_MODE=none)."
        # Auto-fallback to local execution
        logger.warning("Docker unavailable — falling back to host-local shell")
        use_local = True

    if use_local:
        return await _run_local_shell(command, timeout=timeout, workdir=workdir)

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

    use_local = SANDBOX_MODE == "local"
    if not use_local and not await ensure_container_running():
        if SANDBOX_MODE == "none":
            return "⚠️ Shell execution skipped (SANDBOX_MODE=none)."
        logger.warning("Docker unavailable — falling back to host-local shell (stdin)")
        use_local = True

    if use_local:
        return await _run_local_shell_with_stdin(command, stdin_data, timeout, workdir)

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
