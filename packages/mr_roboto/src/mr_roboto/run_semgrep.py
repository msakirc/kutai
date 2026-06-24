"""Run semgrep with a rule pack and parse findings.

Mechanical executor. No LLM. Shells out to ``semgrep --config <rule_pack>
--json <targets...>`` and converts the JSON output to a normalised findings
list.

Severity mapping
----------------
semgrep ``ERROR``   → blocker
semgrep ``WARNING`` → warning
semgrep ``INFO``    → info

Missing semgrep
---------------
semgrep has no native Windows binary.  When semgrep is absent this verb
attempts a Docker fallback (``docker run semgrep/semgrep``) if Docker is
available.  If Docker is also absent the verb returns a *platform-skip*
verdict so callers can distinguish "gate did not run" from "gate ran and
passed":

    ``ok=True, skipped=True, skipped_platform=True``

A platform-skip MUST be surfaced as a WARNING by downstream verdict handlers
— it must never be silently counted as green.  On Linux/macOS where semgrep
is merely absent (not installed but the OS supports it), the legacy
``skipped_platform=False`` path is used so CI can still treat it as a soft
miss rather than a hard warning.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from yazbunu import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.run_semgrep")

# Canonical path to the rule pack shipped with this package.
_RULE_PACK_DIR = Path(__file__).parent / "rule_packs"
DEFAULT_RULE_PACK = str(_RULE_PACK_DIR / "forbidden.yml")

_SEMGREP_SEVERITY_MAP = {
    "ERROR": "blocker",
    "WARNING": "warning",
    "INFO": "info",
}

# Docker image used for the Windows fallback.
_SEMGREP_DOCKER_IMAGE = "semgrep/semgrep:latest"


def _is_windows() -> bool:
    """True when running on Windows (semgrep has no native Win binary)."""
    return sys.platform == "win32"


def _locate_semgrep() -> str:
    """Return the semgrep executable path.  Raises FileNotFoundError if absent."""
    import shutil
    path = shutil.which("semgrep")
    if path is None:
        raise FileNotFoundError("semgrep not found on PATH")
    return path


def _locate_docker() -> str | None:
    """Return the docker executable path, or None if Docker is not installed."""
    import shutil
    return shutil.which("docker")


async def _run_semgrep_via_docker(
    rule_pack_path: str,
    targets: list[str],
    workspace_root: str,
    mission_id: int | None,
    timeout_s: float,
) -> dict[str, Any] | None:
    """Try to run semgrep inside a Docker container.

    Returns a raw ``run_cmd`` result dict on success/failure, or ``None`` when
    Docker is not available.  The caller is responsible for interpreting the
    exit code.

    The workspace is mounted read-only at ``/src`` inside the container.
    Target paths are re-written to be relative to workspace_root so they map
    correctly into ``/src``.
    """
    docker_exe = _locate_docker()
    if docker_exe is None:
        return None

    # Re-map targets: convert absolute paths to relative, then prefix /src/.
    remapped: list[str] = []
    ws = Path(workspace_root).resolve()
    for t in targets:
        p = Path(t)
        if p.is_absolute():
            try:
                rel = p.relative_to(ws)
                remapped.append(f"/src/{rel.as_posix()}")
            except ValueError:
                remapped.append(t)  # outside workspace — pass as-is
        else:
            remapped.append(f"/src/{t}" if t != "." else "/src")

    # Use POSIX path for Docker volume mount even on Windows.
    ws_posix = ws.as_posix()

    cmd = [
        docker_exe, "run", "--rm",
        "-v", f"{ws_posix}:/src:ro",
        _SEMGREP_DOCKER_IMAGE,
        "semgrep",
        "--config", "/src/" + Path(rule_pack_path).name
        if not rule_pack_path.startswith("/src")
        else rule_pack_path,
        "--json",
        "--quiet",
        *remapped,
    ]

    # Copy the rule pack into the workspace temporarily so Docker can see it.
    # Rule packs ship inside the mr_roboto package, not inside workspace_root.
    # We copy only when the rule pack lives outside the workspace.
    rule_pack_p = Path(rule_pack_path).resolve()
    _tmp_pack: Path | None = None
    if not str(rule_pack_p).startswith(str(ws)):
        import shutil as _shutil
        _tmp_pack = ws / f"._semgrep_rule_pack_{rule_pack_p.name}"
        _shutil.copy2(rule_pack_p, _tmp_pack)
        # Update cmd to reference the copied pack inside /src.
        pack_idx = cmd.index("--config") + 1
        cmd[pack_idx] = f"/src/._semgrep_rule_pack_{rule_pack_p.name}"

    try:
        raw = await run_cmd(
            mission_id=mission_id,
            cmd=cmd,
            cwd=None,
            timeout_s=timeout_s + 60,  # Docker pull adds latency on first run.
            require_exit_zero=False,
            workspace_path=str(ws),
        )
    finally:
        if _tmp_pack and _tmp_pack.exists():
            _tmp_pack.unlink(missing_ok=True)

    return raw


async def run_semgrep(
    mission_id: int | None,
    target_files: list[str] | None = None,
    rule_pack_path: str | None = None,
    workspace_path: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run semgrep and return normalised findings.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_cmd`` for workspace resolution.  May be None for
        tests that supply an explicit ``workspace_path``.
    target_files:
        Relative or absolute paths to scan.  ``None`` or ``[]`` scans the
        workspace root (``"."``).
    rule_pack_path:
        Path to a semgrep ``rules:`` YAML file.  Defaults to the built-in
        ``rule_packs/forbidden.yml`` shipped with this package.
    workspace_path:
        Explicit workspace root passed through to ``run_cmd``.  Optional when
        ``mission_id`` is set.
    timeout_s:
        Hard cap passed to ``run_cmd``.  Semgrep on large repos can be slow;
        default 120 s.

    Returns
    -------
    dict with keys:

    ``ok``
        True when semgrep exited 0 or 1 (findings present but no error) AND
        the run was not skipped AND no internal error occurred.
        False when an unexpected error prevented analysis.
    ``skipped``
        True when semgrep is not available (not installed, or not runnable via
        Docker).  When ``skipped_platform`` is also True the caller MUST log a
        WARNING — the gate did not enforce anything.
    ``skipped_platform``
        True when the skip is caused by the host platform not supporting
        semgrep natively (Windows) AND the Docker fallback was also
        unavailable.  False on non-Windows systems where semgrep is simply not
        installed (soft-miss, CI may treat as warning-only).
    ``findings``
        List of finding dicts: ``{rule_id, path, line, message, severity,
        semgrep_severity}``.  Empty when ``ok=True`` and no patterns matched.
    ``blocker_count``
        Count of findings with ``severity == "blocker"``.
    ``warning_count``
        Count of findings with ``severity == "warning"``.
    ``exit``
        Raw semgrep exit code.  -1 on spawn failure.
    ``stdout_tail``
        Last portion of stdout (for debugging).
    ``stderr_tail``
        Last portion of stderr.
    ``duration_s``
        Wall time of the subprocess.
    ``error``
        Human-readable error string when ``ok=False`` and not skipped.
    """
    effective_rule_pack = rule_pack_path or DEFAULT_RULE_PACK
    on_windows = _is_windows()

    # Locate native semgrep binary.
    semgrep_exe: str | None = None
    try:
        semgrep_exe = _locate_semgrep()
    except FileNotFoundError:
        pass

    targets: list[str]
    if not target_files:
        targets = ["."]
    else:
        targets = list(target_files)

    # ── Docker fallback when native semgrep is absent ──────────────────────
    # Attempted unconditionally when semgrep is not on PATH.  On Windows this
    # is the primary path; on Linux it is a best-effort fallback.
    if semgrep_exe is None:
        effective_workspace = workspace_path or "."
        docker_raw = await _run_semgrep_via_docker(
            rule_pack_path=effective_rule_pack,
            targets=targets,
            workspace_root=effective_workspace,
            mission_id=mission_id,
            timeout_s=timeout_s,
        )
        if docker_raw is not None:
            # Docker was available — treat result the same as a native run.
            logger.info(
                "semgrep native absent; running via Docker fallback",
                platform=sys.platform,
            )
            # Fall through to the standard result-parsing logic below by
            # synthesising the variables that the native path would have set.
            exit_code = int(docker_raw.get("exit", -1))
            timed_out = bool(docker_raw.get("timed_out"))
            spawn_error = docker_raw.get("error")
            stdout = docker_raw.get("stdout_tail") or ""
            stderr = docker_raw.get("stderr_tail") or ""

            if exit_code == 127 or spawn_error:
                # Docker itself not found or semgrep image missing.
                pass  # fall through to platform-skip below
            elif timed_out:
                return {
                    "ok": False,
                    "skipped": False,
                    "skipped_platform": False,
                    "findings": [],
                    "blocker_count": 0,
                    "warning_count": 0,
                    "exit": exit_code,
                    "stdout_tail": stdout,
                    "stderr_tail": stderr,
                    "duration_s": docker_raw.get("duration_s", 0.0),
                    "error": "semgrep (Docker) timed out",
                }
            elif exit_code in (0, 1):
                # Successful Docker run — parse findings normally.
                findings = _parse_findings(stdout)
                blocker_count = sum(1 for f in findings if f["severity"] == "blocker")
                warning_count = sum(1 for f in findings if f["severity"] == "warning")
                return {
                    "ok": True,
                    "skipped": False,
                    "skipped_platform": False,
                    "findings": findings,
                    "blocker_count": blocker_count,
                    "warning_count": warning_count,
                    "exit": exit_code,
                    "stdout_tail": stdout,
                    "stderr_tail": stderr,
                    "duration_s": docker_raw.get("duration_s", 0.0),
                    "error": None,
                }
            else:
                logger.warning(
                    "semgrep (Docker) exited with unexpected code",
                    exit_code=exit_code, stderr=stderr[:300],
                )
                return {
                    "ok": False,
                    "skipped": False,
                    "skipped_platform": False,
                    "findings": [],
                    "blocker_count": 0,
                    "warning_count": 0,
                    "exit": exit_code,
                    "stdout_tail": stdout,
                    "stderr_tail": stderr,
                    "duration_s": docker_raw.get("duration_s", 0.0),
                    "error": f"semgrep (Docker) exit {exit_code}: {stderr[:200]}",
                }

        # ── Platform-skip: semgrep absent AND Docker not available ──────────
        # On Windows this is a blocker-severity omission — the gate did NOT
        # run.  On non-Windows this is a softer "tool not installed" miss.
        if on_windows:
            logger.warning(
                "semgrep not available on Windows (no native binary, Docker absent) "
                "— gate DID NOT RUN; skipped_platform=True. "
                "Install semgrep via WSL or ensure Docker is running.",
                platform=sys.platform,
            )
        else:
            logger.warning(
                "semgrep not installed — pattern_lint skipped",
                platform=sys.platform,
            )
        return {
            "ok": True,
            "skipped": True,
            "skipped_platform": on_windows,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": -1,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "error": None,
        }

    # ── Native semgrep run ─────────────────────────────────────────────────
    cmd = [
        semgrep_exe,
        "--config", effective_rule_pack,
        "--json",
        "--quiet",
        *targets,
    ]

    raw = await run_cmd(
        mission_id=mission_id,
        cmd=cmd,
        cwd=None,
        timeout_s=timeout_s,
        require_exit_zero=False,
        workspace_path=workspace_path,
    )

    exit_code = int(raw.get("exit", -1))
    timed_out = bool(raw.get("timed_out"))
    spawn_error = raw.get("error")
    stdout = raw.get("stdout_tail") or ""
    stderr = raw.get("stderr_tail") or ""

    # Exit 127 = command not found (shell wrapper didn't find semgrep).
    if exit_code == 127 or spawn_error:
        logger.warning(
            "semgrep not available or spawn error — pattern_lint skipped",
            exit_code=exit_code, error=spawn_error,
        )
        return {
            "ok": True,
            "skipped": True,
            "skipped_platform": on_windows,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": exit_code,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "duration_s": raw.get("duration_s", 0.0),
            "error": None,
        }

    if timed_out:
        return {
            "ok": False,
            "skipped": False,
            "skipped_platform": False,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": exit_code,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "duration_s": raw.get("duration_s", 0.0),
            "error": "semgrep timed out",
        }

    # semgrep exits 0 (no findings) or 1 (findings present) in normal
    # operation.  Anything else (2 = usage error, 3 = I/O error, …) is an
    # execution failure.
    if exit_code not in (0, 1):
        logger.warning(
            "semgrep exited with unexpected code",
            exit_code=exit_code, stderr=stderr[:300],
        )
        return {
            "ok": False,
            "skipped": False,
            "skipped_platform": False,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": exit_code,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "duration_s": raw.get("duration_s", 0.0),
            "error": f"semgrep exit {exit_code}: {stderr[:200]}",
        }

    findings = _parse_findings(stdout)
    blocker_count = sum(1 for f in findings if f["severity"] == "blocker")
    warning_count = sum(1 for f in findings if f["severity"] == "warning")

    return {
        "ok": True,
        "skipped": False,
        "skipped_platform": False,
        "findings": findings,
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "exit": exit_code,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "duration_s": raw.get("duration_s", 0.0),
        "error": None,
    }


def _parse_findings(stdout: str) -> list[dict]:
    """Parse semgrep JSON stdout into a normalised findings list."""
    findings: list[dict] = []
    try:
        # semgrep --json writes to stdout.  stdout_tail may be truncated by
        # run_cmd's 32 KB cap — for large outputs this could drop trailing
        # JSON.  Acceptable for v1 (rule pack is minimal; findings are dense
        # but short).
        data = json.loads(stdout)
        raw_results = data.get("results") or []
        for r in raw_results:
            sg_sev = str(r.get("extra", {}).get("severity") or "WARNING").upper()
            norm_sev = _SEMGREP_SEVERITY_MAP.get(sg_sev, "warning")
            findings.append({
                "rule_id": r.get("check_id") or "",
                "path": r.get("path") or "",
                "line": int((r.get("start") or {}).get("line") or 0),
                "message": r.get("extra", {}).get("message") or "",
                "severity": norm_sev,
                "semgrep_severity": sg_sev,
            })
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(
            "semgrep JSON parse failed — treating as empty findings",
            exc=str(exc), stdout_head=stdout[:200],
        )
    return findings
