# tools/shell.py
"""
Execute shell commands inside a Docker sandbox container.
The workspace directory is bind-mounted so file changes persist.
"""

import asyncio
import os
import re
import time as _time
from pathlib import Path
from typing import Literal, Optional

from src.infra.logging_config import get_logger

# Z10-T1B: caller-intent reversibility tag passed through from
# mr_roboto.run_cmd's payload["reversibility_override"]. Default None
# means "use the registry default for run_cmd" (= "partial").
ReversibilityIntent = Literal["full", "partial", "irreversible"]

logger = get_logger("tools.shell")

# ---------------------------------------------------------------------------
# Configuration — override via environment or config import
# ---------------------------------------------------------------------------
from src.app.config import WORKSPACE_ROOT as _DEFAULT_WORKSPACE
WORKSPACE_DIR: str = os.environ.get("WORKSPACE_DIR", _DEFAULT_WORKSPACE)
CONTAINER_NAME: str = os.environ.get("SANDBOX_CONTAINER", "orchestrator-sandbox")
SANDBOX_IMAGE: str = os.environ.get("SANDBOX_IMAGE", "orchestrator-sandbox:latest")
SANDBOX_NETWORK: str = os.environ.get("SANDBOX_NETWORK", "bridge")
# Z10-T3B: per-mission resource caps. Default bumped to 4g / 2 cpu /
# 512 pids for a working envelope when running a full agent loop in the
# container; legacy global container still picks up SANDBOX_MEMORY env
# (now defaulting to 4g instead of 512m) — keeps prior behaviour for
# anyone with the env var pinned.
SANDBOX_MEMORY: str = os.environ.get("SANDBOX_MEMORY", "4g")
SANDBOX_CPUS: str = os.environ.get("SANDBOX_CPUS", "2")
SANDBOX_PIDS_LIMIT: str = os.environ.get("SANDBOX_PIDS_LIMIT", "512")
MAX_OUTPUT_CHARS: int = int(os.environ.get("SANDBOX_MAX_OUTPUT", "8000"))
CONTAINER_WORKROOT: str = "/app/workspace"

# Z10-T3B: broader-egress confirmation TTL — once approved, the
# hostname is reachable from this mission's container for this many
# seconds without re-prompting.
EGRESS_GRANT_TTL_SECONDS: float = 300.0  # 5 minutes

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


# ---------------------------------------------------------------------------
# Z10-T3B — Semantic argv guard (defense-in-depth on top of BLOCKED_PATTERNS)
# ---------------------------------------------------------------------------
# Denylist of argv[0] tokens (basename match). Some entries are
# "command + first-arg" tuples handled below (e.g. ``nc -l``).
_BLOCKED_ARGV0: set[str] = {
    "dd", "mkfs", "fdisk", "parted", "wipefs", "shred",
    "iptables", "nft", "route",
}
# argv[0] basename matched by regex (covers ``mkfs.ext4`` etc.).
_BLOCKED_ARGV0_RE: list[re.Pattern[str]] = [
    re.compile(r"^mkfs\..+$"),
]
# Multi-token matches: (argv[0], required-flag). Empty flag → any.
_BLOCKED_ARGV0_WITH_FLAG: list[tuple[str, str]] = [
    ("nc", "-l"),
    ("ncat", "-l"),
]
# Path arguments that are dangerous regardless of command.
_BLOCKED_PATH_RE: list[re.Pattern[str]] = [
    re.compile(r"^/dev/sd[a-z]"),
    re.compile(r"^/dev/nvme"),
    re.compile(r"^/dev/disk"),
    re.compile(r"^/proc/sysrq-trigger$"),
    re.compile(r"^/sys/kernel(/|$)"),
]


def _argv0_basename(token: str) -> str:
    """Strip path prefix from argv[0]."""
    return token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _is_blocked_argv(argv: list[str]) -> tuple[bool, str]:
    """Semantic check on parsed argv list.

    Returns ``(blocked, reason)``. ``reason`` is a short human-readable
    string suitable for logging. Always applied, in both ``docker`` and
    ``local`` mode (defense-in-depth — the regex-based BLOCKED_PATTERNS
    set still runs first via :func:`_is_command_blocked`).
    """
    if not argv:
        return False, ""
    arg0 = _argv0_basename(argv[0])
    if arg0 in _BLOCKED_ARGV0:
        return True, f"argv0 in denylist: {arg0!r}"
    for pat in _BLOCKED_ARGV0_RE:
        if pat.match(arg0):
            return True, f"argv0 matched regex {pat.pattern!r}"
    # ``ip link …`` / ``ip route …`` — block only those two subcommands.
    if arg0 == "ip" and len(argv) >= 2 and argv[1] in ("link", "route"):
        return True, f"blocked ip subcommand: ip {argv[1]}"
    # ``socat … LISTEN…`` — any arg containing "LISTEN".
    if arg0 == "socat":
        if any("LISTEN" in a for a in argv[1:]):
            return True, "socat with LISTEN address"
    # Multi-token argv0+flag matches.
    for cmd, flag in _BLOCKED_ARGV0_WITH_FLAG:
        if arg0 == cmd and flag in argv[1:]:
            return True, f"blocked combo: {cmd} {flag}"
    # Path-style argument scan — also unwrap ``flag=path`` style values
    # (``if=/dev/sda``, ``of=/dev/sda``).
    for arg in argv[1:]:
        # Slice off "key=" prefix if present so paths after = are checked.
        candidate = arg.split("=", 1)[1] if "=" in arg and "/" in arg.split("=", 1)[1] else arg
        for pat in _BLOCKED_PATH_RE:
            if pat.match(candidate):
                return True, f"path argument matched {pat.pattern!r}: {candidate!r}"
    return False, ""


def _tokenize_command(command: str) -> list[str]:
    """Best-effort argv tokenisation for the semantic guard.

    Uses :mod:`shlex` to honour quoting. Shell metas (``|``, ``&&``,
    redirects, etc.) split into separate tokens — the guard then runs
    on each segment so something like ``ls; dd if=/dev/sda`` still
    triggers.
    """
    import shlex
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        # Unclosed quote — fall back to whitespace split so we still
        # get *something* to inspect rather than waving the command
        # through unchecked.
        tokens = command.split()
    return tokens


def _semantic_guard(command: str) -> tuple[bool, str]:
    """Run :func:`_is_blocked_argv` over each shell-split segment.

    Splits the joined command string on common shell separators
    (``;`` / ``&&`` / ``||`` / ``|``) then tokenises each segment.
    Any segment that trips the argv check blocks the whole command.
    """
    if not command or not command.strip():
        return False, ""
    # Split on shell separators while keeping things simple — operators
    # appear as their own tokens after this regex split.
    segments = re.split(r"\s*(?:;|&&|\|\||\|)\s*", command)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        argv = _tokenize_command(seg)
        blocked, reason = _is_blocked_argv(argv)
        if blocked:
            return True, reason
    return False, ""


# ---------------------------------------------------------------------------
# Z10-T3B — Egress whitelist (config/egress_allowlist.txt)
# ---------------------------------------------------------------------------
_EGRESS_ALLOWLIST_PATH_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "egress_allowlist.txt"
# Seed values used when the config file is missing — also written to disk
# on first import via :func:`_ensure_egress_allowlist_file` so the user can
# edit a real file rather than hunt for the constant.
_EGRESS_ALLOWLIST_SEED: tuple[str, ...] = (
    "# Z10-T3B per-mission egress whitelist",
    "# One hostname per line. Comments start with '#'.",
    "# Subdomain match: an entry of 'huggingface.co' allows 'cdn-lfs.huggingface.co' too.",
    "api.openai.com",
    "api.anthropic.com",
    "api.groq.com",
    "openrouter.ai",
    "huggingface.co",
    "pypi.org",
    "files.pythonhosted.org",
    "github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
    "registry.npmjs.org",
)


def _ensure_egress_allowlist_file(path: Path | None = None) -> Path:
    """Create the allowlist file with seed contents if it does not exist."""
    path = path or _EGRESS_ALLOWLIST_PATH_DEFAULT
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(_EGRESS_ALLOWLIST_SEED) + "\n", encoding="utf-8")
    except OSError as e:
        logger.warning("egress allowlist seed write failed", error=str(e))
    return path


def load_egress_allowlist(path: Path | None = None) -> set[str]:
    """Read whitelisted hostnames from ``config/egress_allowlist.txt``.

    Returns lowercase hostnames. Missing/empty file falls back to seed.
    """
    path = _ensure_egress_allowlist_file(path)
    hosts: set[str] = set()
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            hosts.add(line.lower())
    except OSError as e:
        logger.warning("egress allowlist read failed", error=str(e))
        hosts = {h.lower() for h in _EGRESS_ALLOWLIST_SEED if h and not h.startswith("#")}
    return hosts


def _host_in_allowlist(host: str, allowlist: set[str]) -> bool:
    """Match ``host`` against ``allowlist`` with subdomain promotion.

    ``cdn-lfs.huggingface.co`` matches an entry of ``huggingface.co``.
    Empty/None host → False (caller decides what to do).
    """
    if not host:
        return False
    h = host.lower()
    if h in allowlist:
        return True
    for entry in allowlist:
        if h.endswith("." + entry):
            return True
    return False


# Z10-T3B: in-process cache of recently-approved broader-egress grants
# per (mission_id, host). Keyed by mission_id, value is dict of
# host → expiry epoch seconds. Survives in-process only; restart-safe
# semantics is overkill for a 5-minute TTL.
_egress_grants: dict[int, dict[str, float]] = {}


def _record_egress_grant(mission_id: int, host: str, ttl_s: float | None = None) -> None:
    """Cache an approved broader_egress grant for ``ttl_s`` seconds."""
    ttl = float(ttl_s) if ttl_s is not None else EGRESS_GRANT_TTL_SECONDS
    grants = _egress_grants.setdefault(int(mission_id), {})
    grants[host.lower()] = _time.time() + ttl


def _has_active_egress_grant(mission_id: int, host: str) -> bool:
    grants = _egress_grants.get(int(mission_id))
    if not grants:
        return False
    exp = grants.get(host.lower())
    if exp is None:
        return False
    if exp < _time.time():
        # Stale — clean up.
        try:
            del grants[host.lower()]
        except KeyError:
            pass
        return False
    return True


# ---------------------------------------------------------------------------
# Z10-T3B — Per-mission container helpers
# ---------------------------------------------------------------------------
def mission_container_name(mission_id: int | None) -> str:
    """Resolve the container name for a mission.

    With ``mission_id=None`` (legacy callers without a mission context),
    falls back to the global ``CONTAINER_NAME`` and emits a WARNING —
    legacy paths keep working, but the call site should be migrated.
    """
    if mission_id is None:
        logger.warning(
            "mission_container_name called without mission_id — "
            "falling back to global container",
            container=CONTAINER_NAME,
        )
        return CONTAINER_NAME
    return f"kutai-mission-{int(mission_id)}"


def mission_network_name(mission_id: int | None) -> str:
    """Resolve the docker network name for a mission."""
    if mission_id is None:
        return SANDBOX_NETWORK
    return f"kutai-mission-{int(mission_id)}-net"


async def _resolve_mission_resource_caps(mission_id: int | None) -> dict[str, str]:
    """Return effective resource caps for this mission.

    Defaults come from env (SANDBOX_MEMORY / SANDBOX_CPUS /
    SANDBOX_PIDS_LIMIT). ``missions.sandbox_resource_overrides_json``
    can override any subset on a per-mission basis. Schema:
    ``{"memory": "8g", "cpus": "4", "pids_limit": "1024"}``.
    """
    caps = {
        "memory": SANDBOX_MEMORY,
        "cpus": SANDBOX_CPUS,
        "pids_limit": SANDBOX_PIDS_LIMIT,
    }
    if mission_id is None:
        return caps
    try:
        from src.infra.db import get_db
        import json as _json
        db = await get_db()
        cur = await db.execute(
            "SELECT sandbox_resource_overrides_json FROM missions WHERE id = ?",
            (int(mission_id),),
        )
        row = await cur.fetchone()
        if row and row[0]:
            override = _json.loads(row[0])
            if isinstance(override, dict):
                for k in ("memory", "cpus", "pids_limit"):
                    if k in override and override[k] is not None:
                        caps[k] = str(override[k])
    except Exception as e:
        logger.debug(
            "mission resource override lookup failed — using env defaults",
            mission_id=mission_id,
            error=str(e),
        )
    return caps


async def ensure_mission_network(mission_id: int) -> bool:
    """Create the mission's docker network if missing.

    v1 best-effort egress control: we create a bridge network
    ``kutai-mission-{id}-net`` and rely on the in-process whitelist
    + ``broader_egress`` confirmation gate to vet outbound traffic.
    True iptables-level egress filtering needs a sidecar / Linux netns
    setup that's out of scope for v1 — documented in
    docs/i2p-evolution/10-cross-cutting.md.
    """
    network = mission_network_name(mission_id)
    rc, stdout, _ = await _run_quiet(
        "docker", "network", "ls",
        "--filter", f"name=^{network}$",
        "--format", "{{.Name}}",
    )
    if rc == 0 and stdout.strip() == network:
        return True
    rc, _, stderr = await _run_quiet(
        "docker", "network", "create",
        "--driver", "bridge",
        network,
    )
    if rc != 0:
        logger.warning(
            "mission network create failed — falling back to default bridge",
            network=network,
            error=(stderr or "").strip(),
        )
        return False
    logger.info("created mission network", network=network)
    return True


async def ensure_mission_container(mission_id: int) -> bool:
    """Idempotent: ensure ``kutai-mission-{id}`` is running.

    - Already running → no-op, returns True.
    - Exists but stopped → ``docker start``.
    - Doesn't exist → ``docker run -d`` with per-mission caps + network.
    """
    container = mission_container_name(mission_id)
    # 1. Already running?
    rc, stdout, _ = await _run_quiet(
        "docker", "inspect", "-f", "{{.State.Running}}", container,
    )
    if rc == 0 and "true" in stdout.lower():
        return True

    # 2. Exists but stopped? Try restart.
    rc, _, _ = await _run_quiet("docker", "start", container)
    if rc == 0:
        logger.info("restarted existing mission container", container=container)
        return True

    # 3. Create from scratch — ensure network + caps + workspace mount.
    await ensure_mission_network(mission_id)
    caps = await _resolve_mission_resource_caps(mission_id)
    network = mission_network_name(mission_id)
    logger.info(
        "creating mission container",
        container=container,
        mission_id=mission_id,
        memory=caps["memory"],
        cpus=caps["cpus"],
        pids_limit=caps["pids_limit"],
    )
    rc, _, stderr = await _run_quiet(
        "docker", "run", "-d",
        "--name", container,
        "--network", network,
        "--add-host", "host.docker.internal:host-gateway",
        "--memory", caps["memory"],
        "--cpus", caps["cpus"],
        "--pids-limit", caps["pids_limit"],
        "-v", f"{WORKSPACE_DIR}:{CONTAINER_WORKROOT}",
        "-w", CONTAINER_WORKROOT,
        SANDBOX_IMAGE,
        "sleep", "infinity",
    )
    if rc != 0:
        logger.warning(
            "mission container create failed",
            container=container,
            error=(stderr or "").strip(),
        )
        return False
    return True


async def teardown_mission_container(mission_id: int) -> bool:
    """Stop+remove ``kutai-mission-{id}``. Tolerates missing container.

    Also drops the mission network. Errors are warning-logged but never
    raised — teardown is best-effort cleanup, callers don't need to
    handle failures.
    """
    container = mission_container_name(mission_id)
    network = mission_network_name(mission_id)
    # Stop first (idempotent — non-zero means already stopped/missing).
    await _run_quiet("docker", "stop", container)
    rc, _, stderr = await _run_quiet("docker", "rm", container)
    if rc != 0 and "no such container" not in (stderr or "").lower():
        logger.warning(
            "mission container rm non-fatal failure",
            container=container,
            error=(stderr or "").strip(),
        )
    # Best-effort network drop — fails if other containers still attached;
    # ignore that case.
    await _run_quiet("docker", "network", "rm", network)
    # Clear in-process egress grants for this mission.
    _egress_grants.pop(int(mission_id), None)
    return True


async def resolve_sandbox_mode(mission_id: int | None) -> str:
    """Resolve effective sandbox mode for a mission.

    Returns the *requested* mode — caller is responsible for opening a
    confirmation gate when a mission requests ``local`` while the
    system default is not ``local``.
    """
    if mission_id is None:
        return SANDBOX_MODE
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT sandbox_mode FROM missions WHERE id = ?",
            (int(mission_id),),
        )
        row = await cur.fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception as e:
        logger.debug(
            "mission sandbox_mode lookup failed — using system default",
            mission_id=mission_id,
            error=str(e),
        )
    return SANDBOX_MODE


async def _gate_local_mode_optin(mission_id: int) -> bool:
    """Open a sandbox_local_mode confirmation; return True iff approved.

    Polls :func:`check_confirmation` with a generous timeout. Rejected
    or timeout → False; caller falls back to docker.
    """
    try:
        from src.infra.db import request_confirmation, check_confirmation
    except Exception as e:
        logger.warning("local-mode opt-in: DB unavailable, denying", error=str(e))
        return False

    try:
        conf_id = await request_confirmation(
            task_id=0,  # mission-scope gate, not task-scope
            verb="sandbox_local_mode",
            reversibility="partial",
            payload_summary=(
                f"Mission {mission_id} requested host-mode shell — "
                "escapes container"
            ),
        )
    except Exception as e:
        logger.warning("local-mode opt-in: request failed, denying", error=str(e))
        return False

    elapsed = 0.0
    max_wait = 60.0
    interval = 0.5
    while elapsed < max_wait:
        try:
            res = await check_confirmation(conf_id)
        except Exception:
            return False
        verdict = res.get("verdict")
        if verdict == "approved":
            return True
        if verdict == "rejected":
            return False
        await asyncio.sleep(interval)
        elapsed += interval
    logger.info(
        "local-mode opt-in: timeout — denying", mission_id=mission_id, conf_id=conf_id,
    )
    return False


async def _gate_broader_egress(mission_id: int, host: str) -> bool:
    """Open a broader_egress confirmation for ``host``; True iff approved.

    Approval records an in-process grant for
    :data:`EGRESS_GRANT_TTL_SECONDS` seconds — re-prompting after that.
    """
    if _has_active_egress_grant(mission_id, host):
        return True
    try:
        from src.infra.db import request_confirmation, check_confirmation
    except Exception as e:
        logger.warning("broader-egress: DB unavailable, denying", error=str(e))
        return False
    try:
        conf_id = await request_confirmation(
            task_id=0,
            verb="broader_egress",
            reversibility="partial",
            payload_summary=f"shell wants to reach {host}",
        )
    except Exception as e:
        logger.warning("broader-egress: request failed, denying", error=str(e))
        return False
    elapsed = 0.0
    max_wait = 60.0
    interval = 0.5
    while elapsed < max_wait:
        try:
            res = await check_confirmation(conf_id)
        except Exception:
            return False
        verdict = res.get("verdict")
        if verdict == "approved":
            _record_egress_grant(mission_id, host)
            return True
        if verdict == "rejected":
            return False
        await asyncio.sleep(interval)
        elapsed += interval
    return False


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


# Cache: once we determine docker is unavailable, skip the (slow) docker
# calls for SANDBOX_DOCKER_DOWN_TTL seconds. Without this every shell-tool
# invocation pays the ``docker inspect`` + ``docker start`` + ``docker
# run`` round-trip and emits ``failed to create sandbox container`` to
# the ERROR log — observable noise on hosts without docker. The shell
# tool already auto-falls-back to local execution; the cache just keeps
# the failure quiet between probes. (Handoff item K.)
_SANDBOX_DOCKER_DOWN_TTL: float = 60.0
_sandbox_docker_down_until: float = 0.0


def _normalize_path_for_compare(p: str) -> str:
    """Canonicalize a Windows path string for cross-source comparison.

    Container's ``docker inspect`` reports Mount.Source as a Windows
    absolute path like ``C:\\Users\\sakir\\...``. The host-side
    ``WORKSPACE_DIR`` may have been read from .env with the same
    backslash-encoded form OR a forward-slash form depending on
    platform. Equality compare must be case-insensitive (NTFS),
    separator-agnostic, and trailing-slash-agnostic.
    """
    if not isinstance(p, str):
        return ""
    # Replace backslash with forward slash, normalize, lowercase.
    norm = os.path.normpath(p.replace("\\", "/"))
    return norm.casefold().rstrip("/").rstrip("\\")


async def validate_or_recreate_sandbox() -> None:
    """Startup-time check: container's bind-mount source must match
    current ``WORKSPACE_DIR``. If stale (e.g. WORKSPACE_DIR changed in
    .env since the container was created), remove the container so the
    next ``ensure_container_running`` call recreates it with the right
    bind.

    Process-startup only — caller should invoke this ONCE during
    orchestrator init. No locking needed; orchestrator startup is
    serial. Per-call validation was rejected as too costly (the
    `docker inspect` adds 200-500ms; with hundreds of shell calls per
    minute the overhead dominates).

    Mount source comparison is normalized: case-insensitive,
    separator-agnostic, trailing-slash-agnostic. False positives from
    raw string compare on Windows would force-recreate every startup.

    SANDBOX_MODE=local / SANDBOX_MODE=none short-circuit (no container
    to validate). Likewise when docker isn't installed — the existing
    docker-down cache (handoff item K) will trip the same way at first
    real shell call.
    """
    if SANDBOX_MODE in ("local", "none"):
        return

    rc, stdout, stderr = await _run_quiet(
        "docker", "inspect", CONTAINER_NAME,
        "--format", "{{range .Mounts}}{{if eq .Destination \"/app/workspace\"}}{{.Source}}{{end}}{{end}}",
    )
    if rc != 0:
        # Container doesn't exist OR docker isn't running. Either way,
        # nothing to validate — let ensure_container_running handle.
        logger.debug(
            "sandbox validation: docker inspect non-zero (container "
            "missing or daemon down) — skipping",
            rc=rc,
        )
        return

    actual_source = (stdout or "").strip()
    expected_source = WORKSPACE_DIR

    if not actual_source:
        # Container exists but has no /app/workspace mount — definitely stale.
        logger.warning(
            "sandbox validation: container has no /app/workspace "
            "mount — recreating",
            container=CONTAINER_NAME,
        )
    elif _normalize_path_for_compare(actual_source) == _normalize_path_for_compare(expected_source):
        logger.debug(
            "sandbox validation: bind-mount matches",
            container=CONTAINER_NAME,
            source=actual_source,
        )
        return
    else:
        logger.warning(
            "sandbox validation: bind-mount stale — recreating",
            container=CONTAINER_NAME,
            actual=actual_source,
            expected=expected_source,
        )

    # Stale or no-mount — remove. The next shell call recreates with
    # current WORKSPACE_DIR.
    rc, _, stderr = await _run_quiet("docker", "rm", "-f", CONTAINER_NAME)
    if rc != 0:
        logger.warning(
            "sandbox validation: docker rm -f failed — manual cleanup "
            "may be needed",
            error=(stderr or "").strip(),
        )


async def ensure_container_running() -> bool:
    """Make sure the sandbox Docker container is up, restarting or creating as needed."""
    import time as _time
    global _sandbox_docker_down_until

    # Short-circuit when we recently saw docker unavailable — caller
    # falls back to local execution. Probe again after the TTL expires
    # in case the user started docker mid-session.
    if _sandbox_docker_down_until > _time.time():
        return False

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
        # Docker daemon unreachable, image missing, or permissions issue
        # — none of which we can fix from here. Caller falls back to
        # local execution; suppress the ERROR-log spam from repeated
        # probes by going quiet for SANDBOX_DOCKER_DOWN_TTL seconds.
        # First failure logs as warning + reason; subsequent probes
        # before TTL skip the docker round-trip entirely.
        logger.warning(
            "sandbox container unavailable — falling back to local "
            "shell for the next %ds",
            int(_SANDBOX_DOCKER_DOWN_TTL),
            error=stderr.strip(),
        )
        _sandbox_docker_down_until = _time.time() + _SANDBOX_DOCKER_DOWN_TTL
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
    reversibility_intent: Optional[ReversibilityIntent] = None,
    mission_id: Optional[int] = None,
    request_egress_to: Optional[str] = None,
) -> str:
    """
    Execute a shell command inside the Docker sandbox.

    Args:
        command:  The shell command to run.
        timeout:  Max seconds to wait (default 60).
        workdir:  Working directory inside the container
                  (absolute under /app/workspace, or relative to it).
        reversibility_intent: Z10-T1B caller intent tag. ``None`` (default)
                  preserves prior behavior; mr_roboto's ``run_cmd``
                  passes through ``payload["reversibility_override"]``
                  so the shell log records whether the caller knew the
                  command was destructive.
        mission_id: Z10-T3B per-mission scope. When provided, routes
                  execution to the ``kutai-mission-{id}`` container
                  (created on demand). ``None`` keeps the legacy global
                  ``CONTAINER_NAME`` container, with a logged warning.
        request_egress_to: Z10-T3B explicit egress request. If set and
                  the host is not in ``config/egress_allowlist.txt``,
                  opens a ``broader_egress`` confirmation. Approved →
                  temporary 5-minute grant cached in-process; rejected
                  → command is blocked.

    Returns:
        Combined stdout + stderr with exit-code indicator,
        truncated to MAX_OUTPUT_CHARS.
    """
    # Z10-T3B: resolve effective mode + log the resolved value (not the
    # requested one) so the audit row shows what actually ran.
    requested_mode = await resolve_sandbox_mode(mission_id)
    resolved_mode = requested_mode
    if requested_mode == "local" and SANDBOX_MODE != "local":
        # Per-mission requests host-mode while system default isn't —
        # gate via founder confirmation.
        if mission_id is None or not await _gate_local_mode_optin(mission_id):
            resolved_mode = "docker"

    # Z10-T3B: broader-egress gate. If caller specified an out-of-list
    # host, ask founder. No-op when host is already in the allowlist.
    if request_egress_to:
        allowlist = load_egress_allowlist()
        if not _host_in_allowlist(request_egress_to, allowlist):
            if mission_id is None:
                logger.warning(
                    "broader-egress request without mission_id — denying",
                    host=request_egress_to,
                )
                return "🚫 BLOCKED: broader_egress requires mission_id."
            if not await _gate_broader_egress(mission_id, request_egress_to):
                return f"🚫 BLOCKED: egress to {request_egress_to} not approved."

    logger.info(
        "shell invocation",
        command=command[:120],
        reversibility_intent=reversibility_intent,
        mission_id=mission_id,
        sandbox_mode=resolved_mode,
    )
    if _is_command_blocked(command):
        return "🚫 BLOCKED: This command matched a safety filter and was not executed."

    # Z10-T3B: semantic argv guard — additive on top of the regex set.
    semantic_blocked, reason = _semantic_guard(command)
    if semantic_blocked:
        logger.warning("semantic guard blocked command", reason=reason, command=command[:120])
        return f"🚫 BLOCKED (semantic guard): {reason}"

    use_local = resolved_mode == "local"
    container = mission_container_name(mission_id) if mission_id is not None else CONTAINER_NAME

    if not use_local:
        if mission_id is not None:
            if not await ensure_mission_container(mission_id):
                if resolved_mode == "none":
                    return "⚠️ Shell execution skipped (SANDBOX_MODE=none)."
                logger.warning(
                    "mission container unavailable — falling back to host-local",
                    mission_id=mission_id,
                )
                use_local = True
        else:
            if not await ensure_container_running():
                if SANDBOX_MODE == "none":
                    return "⚠️ Shell execution skipped (SANDBOX_MODE=none)."
                logger.warning("Docker unavailable — falling back to host-local shell")
                use_local = True

    if use_local:
        return await _run_local_shell(command, timeout=timeout, workdir=workdir)

    resolved_wd = _resolve_workdir(workdir)
    logger.debug("executing shell command", command=command[:120], workdir=resolved_wd, container=container)

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec",
            "-w", resolved_wd,
            container,
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
                "docker", "exec", container,
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
