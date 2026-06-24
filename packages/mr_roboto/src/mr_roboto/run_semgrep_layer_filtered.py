"""run_semgrep_layer_filtered — layer-aware semgrep wrapper.

Pre-filters *target_files* to only those whose ``inspect_layer()`` result
matches *required_layer*, then delegates to ``run_semgrep``.

This is NOT a new auto-wired post-hook kind.  It is an opt-in helper that
step contexts can invoke via ``context.layer_rule_pack``.  The pattern_lint
hook detects that key and routes to this wrapper instead of plain
``run_semgrep``.

Usage example (step context)::

    "context": {
        "layer_rule_pack": "forbidden_in_domain.yml",
        "post_hooks": ["pattern_lint"]
    }

Design constraints
------------------
- No new hook kind.  The existing ``pattern_lint`` hook reads
  ``layer_rule_pack`` and calls this wrapper.
- ``inspect_layer`` is imported lazily inside the function to avoid circular
  imports at module load time.
- Files that fail layer-resolution (return "unknown") are excluded when
  *required_layer* is specified, erring on the side of silence.
"""
from __future__ import annotations

from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.run_semgrep_layer_filtered")

_RULE_PACK_DIR_NAME = "rule_packs"


def _rule_pack_path(pack_name: str) -> str:
    """Resolve *pack_name* relative to the mr_roboto rule_packs directory."""
    from pathlib import Path
    return str(Path(__file__).parent / _RULE_PACK_DIR_NAME / pack_name)


async def run_semgrep_layer_filtered(
    mission_id: int | None,
    target_files: list[str],
    rule_pack_path: str,
    required_layer: str,
    workspace_path: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Run semgrep only on files matching *required_layer*.

    Parameters
    ----------
    mission_id:
        Forwarded to ``run_semgrep``.
    target_files:
        Candidate files to check.  Files whose layer != *required_layer* are
        silently excluded before invoking semgrep.
    rule_pack_path:
        Path to the semgrep rule YAML (absolute or relative; relative paths
        are resolved relative to the mr_roboto rule_packs directory).
    required_layer:
        Only files classified as this layer are scanned
        (e.g. ``"domain"``).
    workspace_path:
        Mission workspace root passed to ``inspect_layer`` and ``run_semgrep``.
    timeout_s:
        Semgrep timeout forwarded to ``run_semgrep``.

    Returns
    -------
    Same dict shape as ``run_semgrep`` — ``ok``, ``skipped``, ``findings``,
    ``blocker_count``, ``warning_count``, ``exit``, ``stdout_tail``,
    ``stderr_tail``, ``duration_s``, ``error``.

    When no target files match the required layer the function returns
    ``ok=True, skipped=True, findings=[]`` — there is nothing to scan.
    """
    from src.tools.inspect_layer import inspect_layer

    # Resolve rule pack path — if not absolute, look in rule_packs dir.
    import os
    if not os.path.isabs(rule_pack_path):
        resolved_pack = _rule_pack_path(rule_pack_path)
    else:
        resolved_pack = rule_pack_path

    # Pre-filter: keep only files whose layer matches required_layer.
    filtered: list[str] = []
    for fp in target_files:
        layer = await inspect_layer(fp, workspace_path=workspace_path)
        if layer == required_layer:
            filtered.append(fp)

    if not filtered:
        logger.debug(
            "run_semgrep_layer_filtered: no files matched layer=%s — skipping",
            required_layer,
        )
        return {
            "ok": True,
            "skipped": True,
            "findings": [],
            "blocker_count": 0,
            "warning_count": 0,
            "exit": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_s": 0.0,
            "error": None,
        }

    logger.debug(
        "run_semgrep_layer_filtered: scanning %d/%d files for layer=%s",
        len(filtered), len(target_files), required_layer,
    )

    from mr_roboto.run_semgrep import run_semgrep
    return await run_semgrep(
        mission_id=mission_id,
        target_files=filtered,
        rule_pack_path=resolved_pack,
        workspace_path=workspace_path,
        timeout_s=timeout_s,
    )
