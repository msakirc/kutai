"""yalayut.executor — mechanical recipe execution.

``run_recipe(recipe_id, args)`` is the body of the ``yalayut_recipe`` mr_roboto
executor. It loads a ``shell_recipe`` manifest row from ``yalayut_index``,
executes each ``invocation.steps[].cmd`` as a subprocess inside the mission
workspace (Windows-safe, no shell), and reports per-step results.

Arg-binding is **not** done here — intersect (Phase 2) statically binds
``inputs_schema`` fields and passes the resolved ``args`` dict in the mechanical
task payload. ``run_recipe`` substitutes those args into ``{placeholder}``
tokens inside each command string before tokenizing.

Returns a plain dict (crosses the mr_roboto / beckman boundary):
``{"ok", "recipe_id", "name", "steps": [...], "artifacts_present": [...],
   "artifacts_missing": [...], "reason": str | None}``.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from yazbunu import get_logger
from yalayut.shell_safety import (
    ShellSafetyError,
    check_shell_bin,
    tokenize_cmd,
    windows_incompat_reason,
)

logger = get_logger("yalayut.executor")

_STEP_TIMEOUT_S = 600.0          # cookiecutter scaffolds can be slow on cold uvx
_OUTPUT_TAIL = 8 * 1024


async def _load_recipe_row(recipe_id: int) -> dict[str, Any] | None:
    """Load a recipe row from yalayut_index, parsed manifest attached.

    Returns ``None`` if the row is absent, disabled, or not a shell_recipe.
    The mission workspace path is resolved from the manifest's bound args at
    call time (intersect injects ``workspace_path`` into args); we fall back to
    the current working directory only in tests.
    """
    import yaml

    from dabidabi import get_db

    db = await get_db()
    try:
        cur = await db.execute(
            "SELECT id, name, kind, manifest_path, mechanizable, vet_tier, enabled "
            "FROM yalayut_index WHERE id = ?",
            (recipe_id,),
        )
        row = await cur.fetchone()
    except Exception as e:
        logger.warning("yalayut_index query failed", recipe_id=recipe_id, err=str(e))
        return None
    if row is None:
        return None
    (rid, name, kind, manifest_path, mechanizable, vet_tier, enabled) = row
    if not enabled or kind != "shell_recipe" or not manifest_path:
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = yaml.safe_load(fh) or {}
    except OSError as e:
        logger.warning("recipe manifest unreadable", recipe_id=recipe_id, err=str(e))
        return None
    return {
        "id": rid,
        "name": name,
        "manifest": manifest,
        "mechanizable": bool(mechanizable),
        "vet_tier": vet_tier,
        "workspace_path": None,
    }


def _substitute_args(cmd: str, args: dict[str, Any]) -> str:
    """Replace ``{key}`` tokens in a command string with bound arg values.

    Only string/number/bool args are substituted; missing keys are left as-is
    so the allowlist/incompat checks still see the literal token.
    """
    out = cmd
    for key, val in (args or {}).items():
        if isinstance(val, (str, int, float, bool)):
            out = out.replace("{" + str(key) + "}", str(val))
    return out


def _tail(data: bytes) -> str:
    if len(data) > _OUTPUT_TAIL:
        data = data[-_OUTPUT_TAIL:]
    return data.decode("utf-8", errors="replace")


async def _run_step(argv: list[str], cwd: str) -> dict[str, Any]:
    """Execute a single tokenized command. No shell."""
    loop = asyncio.get_event_loop()
    started = loop.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except FileNotFoundError as e:
        return {"exit": -1, "ok": False, "stdout": "", "stderr": "",
                "error": f"executable not found: {e}", "argv": argv}
    except OSError as e:
        return {"exit": -1, "ok": False, "stdout": "", "stderr": "",
                "error": f"spawn failed: {e}", "argv": argv}
    timed_out = False
    try:
        out_b, err_b = await asyncio.wait_for(
            proc.communicate(), timeout=_STEP_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            out_b, err_b = b"", b""
    exit_code = -1 if timed_out else (proc.returncode or 0)
    return {
        "exit": exit_code,
        "ok": (not timed_out) and exit_code == 0,
        "stdout": _tail(out_b),
        "stderr": _tail(err_b),
        "duration_s": round(loop.time() - started, 3),
        "timed_out": timed_out,
        "argv": argv,
    }


async def run_recipe(recipe_id: int, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a shell_recipe artifact's invocation steps.

    Parameters
    ----------
    recipe_id : int
        ``yalayut_index.id`` of a ``kind='shell_recipe'`` artifact.
    args : dict
        Statically-bound inputs from intersect. May carry ``workspace_path``
        (the mission workspace dir the recipe should scaffold into).
    """
    row = await _load_recipe_row(recipe_id)
    if row is None:
        return {"ok": False, "recipe_id": recipe_id, "name": None,
                "steps": [], "artifacts_present": [], "artifacts_missing": [],
                "reason": f"recipe {recipe_id} not found / not a shell_recipe"}

    if not row.get("mechanizable"):
        return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                "steps": [], "artifacts_present": [], "artifacts_missing": [],
                "reason": "recipe is not mechanizable"}

    manifest = row.get("manifest") or {}
    steps = ((manifest.get("invocation") or {}).get("steps")) or []
    if not steps:
        return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                "steps": [], "artifacts_present": [], "artifacts_missing": [],
                "reason": "recipe has no invocation.steps"}

    cwd = (args or {}).get("workspace_path") or row.get("workspace_path") or os.getcwd()
    os.makedirs(cwd, exist_ok=True)

    # Pre-flight: tokenize + allowlist + incompat-check every step before
    # running anything, so a bad recipe fails fast with nothing executed.
    prepared: list[list[str]] = []
    for idx, step in enumerate(steps):
        raw = step.get("cmd") if isinstance(step, dict) else None
        if not raw:
            return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                    "steps": [], "artifacts_present": [], "artifacts_missing": [],
                    "reason": f"step {idx} has no cmd"}
        cmd = _substitute_args(raw, args)
        incompat = windows_incompat_reason(cmd)
        if incompat:
            return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                    "steps": [], "artifacts_present": [], "artifacts_missing": [],
                    "reason": f"step {idx} windows-incompat: {incompat}"}
        try:
            argv = tokenize_cmd(cmd)
        except ShellSafetyError as e:
            return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                    "steps": [], "artifacts_present": [], "artifacts_missing": [],
                    "reason": f"step {idx} untokenizable: {e}"}
        if not check_shell_bin(argv[0]):
            return {"ok": False, "recipe_id": recipe_id, "name": row.get("name"),
                    "steps": [], "artifacts_present": [], "artifacts_missing": [],
                    "reason": f"step {idx} bin not in allowlist: {argv[0]!r}"}
        prepared.append(argv)

    # Execute sequentially; stop on the first failing step.
    step_results: list[dict[str, Any]] = []
    all_ok = True
    for argv in prepared:
        result = await _run_step(argv, cwd)
        step_results.append(result)
        logger.info("recipe step done", recipe_id=recipe_id,
                     bin=argv[0], exit=result["exit"], ok=result["ok"])
        if not result["ok"]:
            all_ok = False
            break

    # Verify declared artifacts (paths relative to cwd unless absolute).
    declared = manifest.get("artifacts") or []
    present, missing = [], []
    for art in declared:
        path = art if os.path.isabs(art) else os.path.join(cwd, art)
        (present if os.path.exists(path) else missing).append(art)

    ok = all_ok and not missing
    reason = None
    if not all_ok:
        reason = f"step {len(step_results) - 1} failed (exit {step_results[-1]['exit']})"
    elif missing:
        reason = f"missing declared artifacts: {missing}"

    return {
        "ok": ok,
        "recipe_id": recipe_id,
        "name": row.get("name"),
        "steps": step_results,
        "artifacts_present": present,
        "artifacts_missing": missing,
        "reason": reason,
    }
