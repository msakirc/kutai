"""Run Maestro mobile-QA flows — Z5 T4b adapter.

Mechanical executor. No LLM. Shells out (via :func:`run_cmd`) to the
Maestro CLI (``maestro test <flow.yaml>``) to drive an already-running
mobile app (Expo Go / dev build / emulator) through one or more declared
UI test flows.

This adapter is the verb behind the ``mobile_smoke`` post-hook: a
recipe-driven Maestro YAML flow (sign in → onboard → core action → sign
out) that gates a build step on the Maestro exit code.

Invocation
----------
Maestro ships as a standalone CLI installed on PATH (``maestro``). One
``maestro test`` invocation is issued per flow YAML so the structured
result can report pass/fail per flow rather than collapsing the whole
batch onto a single exit code.

Soft-skip
---------
When the ``maestro`` CLI is not installed, :func:`run_cmd` returns
``{ok:False, error:"executable not found: ..."}``; this verb maps that to
``skipped=True`` (a soft pass) rather than ``failed`` — Maestro is a
runtime tool, not a project dependency, and the dev box (Windows) may not
have it. Callers (the ``mobile_smoke`` post-hook) MUST treat ``skipped``
as a soft pass, never a blocker.

Reversibility: ``full`` — driving an app through UI flows writes nothing
durable to the workspace and makes no real-world change. A read-only test
run.
"""

from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger
from mr_roboto.run_cmd import run_cmd

logger = get_logger("mr_roboto.maestro_run")

_TAIL_CHARS = 4000  # keep last ~4 KB of stdout in the structured result

DEFAULT_TIMEOUT_S = 600.0


def _tail(text: str | None) -> str:
    text = text or ""
    return text[-_TAIL_CHARS:] if len(text) > _TAIL_CHARS else text


def _is_missing_exe(raw: dict[str, Any]) -> bool:
    """True when run_cmd could not find the Maestro executable to spawn."""
    err = (raw.get("error") or "").lower()
    return "executable not found" in err or "not found" in err


def _skip_result(reason: str, *, exit_code: int = -1) -> dict[str, Any]:
    """Build the soft-skip structured result (treated as a soft pass)."""
    return {
        "ok": True,
        "skipped": True,
        "flows_run": 0,
        "passed": 0,
        "failed": 0,
        "exit": exit_code,
        "stdout_tail": "",
        "reason": reason,
        "error": None,
    }


async def maestro_run(
    mission_id: int | None,
    flow_paths: list[str] | None = None,
    workspace_path: str | None = None,
    extra_args: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run one or more Maestro flow YAMLs and return a structured result.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_cmd`` for workspace resolution. May be None when
        ``workspace_path`` is supplied (tests).
    flow_paths:
        Workspace-relative (or absolute) paths to Maestro flow ``.yaml``
        files. One ``maestro test`` invocation is issued per path so the
        result reports per-flow pass/fail. Empty/None → soft-skip (nothing
        to run is not a failure).
    workspace_path:
        Explicit workspace root passed through to ``run_cmd``. Optional
        when ``mission_id`` is set.
    extra_args:
        Additional argv tokens appended after ``maestro test <flow>``
        (e.g. ``["--format", "junit"]``).
    timeout_s:
        Hard cap passed to ``run_cmd`` *per flow*. Default 600 s.

    Returns
    -------
    dict with keys:

    ``ok``
        True when every flow passed (or the run was soft-skipped). False
        when at least one flow failed or an internal error occurred.
    ``skipped``
        True when the Maestro CLI is not installed, or when no flows were
        supplied. Callers MUST treat this as a soft pass, never a blocker.
    ``flows_run``
        Count of flow YAMLs actually executed.
    ``passed``
        Count of flows that exited 0.
    ``failed``
        Count of flows that exited non-zero (or timed out).
    ``exit``
        Worst (non-zero-preferring) raw exit code across all flows. 0 when
        every flow passed. -1 on spawn failure / timeout.
    ``stdout_tail``
        Tail of the concatenated per-flow stdout (for debugging).
    ``error``
        Human-readable error string when ``ok=False`` and not skipped.
    """
    flows = [str(p).strip() for p in (flow_paths or []) if str(p).strip()]
    if not flows:
        # Nothing to run is a soft pass — a step may declare mobile_smoke
        # before any flow YAML has been generated.
        logger.warning("maestro_run: no flow paths supplied — skipped")
        return _skip_result("no flow paths supplied")

    flows_run = 0
    passed = 0
    failed = 0
    worst_exit = 0
    stdout_chunks: list[str] = []
    timed_out_any = False

    for flow in flows:
        cmd = ["maestro", "test", flow, *(list(extra_args or []))]
        raw = await run_cmd(
            mission_id=mission_id,
            cmd=cmd,
            cwd=None,
            timeout_s=timeout_s,
            require_exit_zero=False,
            workspace_path=workspace_path,
            reversibility_intent="full",
        )

        # Missing CLI — soft-skip the whole verb. Detected on the first
        # flow; no point retrying the remaining flows with the same
        # absent binary.
        if _is_missing_exe(raw):
            logger.warning(
                "maestro CLI not installed — mobile_smoke skipped",
                flow=flow,
            )
            return _skip_result(
                "maestro CLI not installed",
                exit_code=int(raw.get("exit", -1)),
            )

        flows_run += 1
        exit_code = int(raw.get("exit", -1))
        flow_timed_out = bool(raw.get("timed_out"))
        stdout_chunks.append(
            f"--- {flow} (exit={exit_code}) ---\n"
            + (raw.get("stdout_tail") or "")
        )

        if flow_timed_out:
            timed_out_any = True
            failed += 1
            worst_exit = exit_code if worst_exit == 0 else worst_exit
        elif exit_code == 0:
            passed += 1
        else:
            failed += 1
            worst_exit = exit_code  # non-zero wins over a prior 0

    stdout_tail = _tail("\n".join(stdout_chunks))
    ok = failed == 0

    if ok:
        return {
            "ok": True,
            "skipped": False,
            "flows_run": flows_run,
            "passed": passed,
            "failed": 0,
            "exit": 0,
            "stdout_tail": stdout_tail,
            "error": None,
        }

    if timed_out_any:
        error = f"maestro: {failed}/{flows_run} flow(s) failed (timeout after {timeout_s}s)"
    else:
        error = f"maestro: {failed}/{flows_run} flow(s) failed (exit {worst_exit})"

    return {
        "ok": False,
        "skipped": False,
        "flows_run": flows_run,
        "passed": passed,
        "failed": failed,
        "exit": worst_exit if worst_exit != 0 else -1,
        "stdout_tail": stdout_tail,
        "error": error,
    }
