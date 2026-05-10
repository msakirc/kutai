"""verify_against_paraflow_goldens — Z1 Tier 7B (C21).

Mechanical wrapper around :func:`c21_paraflow_diff.diff_bundle`.

NOT auto-wired into any i2p step. Invoked manually via Telegram
``/paraflow_check <mission_id>`` or by a future standing audit job.
Persists each run to ``paraflow_diff_log`` for trend analysis.
"""
from __future__ import annotations

import json
from typing import Any

from c21_paraflow_diff import diff_bundle, GoldenNotFoundError


async def verify_against_paraflow_goldens(
    mission_id: int,
    archetype: str = "truthrate",
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Diff a mission's workspace against a paraflow golden bundle.

    Parameters
    ----------
    mission_id:
        Mission row id; resolved to ``workspace/mission_<id>/`` unless
        ``workspace_path`` is supplied.
    archetype:
        Golden archetype name (defaults to ``"truthrate"``).
    workspace_path:
        Override for the mission workspace (test harnesses pass tmp dirs).

    Returns
    -------
    dict
        ``{ok, verdict, gaps, coverage, coherence, design_fitness,
        score, archetype, mission_id}``.

        ``ok`` is ``True`` for ``paraflow_par`` / ``paraflow_partial``,
        ``False`` for ``paraflow_gap`` or any error. Mission verdict is
        always returned in ``verdict`` so callers can surface partial.
    """
    # Resolve workspace.
    if workspace_path is None:
        try:
            from src.tools.workspace import get_mission_workspace
            workspace_path = get_mission_workspace(int(mission_id))
        except Exception:
            workspace_path = f"workspace/mission_{mission_id}"

    try:
        diff = diff_bundle(workspace_path, archetype)
    except GoldenNotFoundError as e:
        return {
            "ok": False,
            "error": f"unknown archetype {archetype!r}: {e}",
            "archetype": archetype,
            "mission_id": int(mission_id),
            "verdict": "paraflow_gap",
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "archetype": archetype,
            "mission_id": int(mission_id),
            "verdict": "paraflow_gap",
        }

    # Best-effort persist to paraflow_diff_log.
    try:
        await _persist(int(mission_id), archetype, diff)
    except Exception:
        # Logging failure is never fatal.
        pass

    return {
        "ok": diff["verdict"] != "paraflow_gap",
        "verdict": diff["verdict"],
        "score": diff["score"],
        "gaps": diff["gaps"],
        "coverage": diff["coverage"],
        "coherence": diff["coherence"],
        "design_fitness": diff["design_fitness"],
        "archetype": archetype,
        "mission_id": int(mission_id),
        "workspace_path": diff["mission_workspace_path"],
    }


async def _persist(
    mission_id: int, archetype: str, diff: dict[str, Any]
) -> None:
    """Insert a row into ``paraflow_diff_log``. Best-effort."""
    try:
        from src.infra.db import get_db
    except Exception:
        return
    db = await get_db()
    await db.execute(
        """
        INSERT INTO paraflow_diff_log
            (mission_id, archetype, verdict, score,
             gaps_json, coverage_json, coherence_json, design_fitness_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mission_id,
            archetype,
            diff["verdict"],
            diff["score"],
            json.dumps(diff.get("gaps") or []),
            json.dumps(diff.get("coverage") or {}),
            json.dumps(diff.get("coherence") or {}),
            json.dumps(diff.get("design_fitness") or {}),
        ),
    )
    await db.commit()
