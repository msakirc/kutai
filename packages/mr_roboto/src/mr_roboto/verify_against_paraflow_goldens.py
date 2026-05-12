"""verify_against_paraflow_goldens — Z1 Tier 7B (C21).

Mechanical wrapper around :func:`c21_paraflow_diff.diff_bundle`.

NOT auto-wired into any i2p step. Invoked by:

* Manual ``/paraflow_check <mission_id> [archetype]`` Telegram cmd.
* Weekly ``paraflow_audit_all`` cron — scans the workspace for
  ``mission_<id>/.paraflow_archetype`` markers (a one-line file
  containing the archetype name) and runs the diff per match.

Persists each run to ``paraflow_diff_log`` for trend analysis.
"""
from __future__ import annotations

import json
import os
import re
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


_MISSION_DIR_RE = re.compile(r"^mission_(\d+)$")
_ARCHETYPE_MARKER = ".paraflow_archetype"


async def paraflow_audit_all(
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Run verify_against_paraflow_goldens for every mission that opts in.

    Opt-in mechanism: a one-line file at
    ``mission_<id>/.paraflow_archetype`` whose content is the archetype
    name (e.g. ``truthrate``). Missions without the marker are skipped.

    Returns a summary ``{audited, gaps, partial, par, errors, results}``
    so the cron telemetry can chart drift over time.
    """
    if workspace_root is None:
        try:
            from src.tools.workspace import WORKSPACE_DIR
            workspace_root = WORKSPACE_DIR
        except Exception:
            workspace_root = os.path.join(os.getcwd(), "workspace")

    results: list[dict[str, Any]] = []
    audited = par = partial = gaps = errors = 0
    if not os.path.isdir(workspace_root):
        return {
            "audited": 0,
            "par": 0,
            "partial": 0,
            "gaps": 0,
            "errors": 0,
            "results": [],
            "workspace_root": workspace_root,
        }

    for entry in sorted(os.listdir(workspace_root)):
        m = _MISSION_DIR_RE.match(entry)
        if not m:
            continue
        mid = int(m.group(1))
        mdir = os.path.join(workspace_root, entry)
        marker = os.path.join(mdir, _ARCHETYPE_MARKER)
        if not os.path.isfile(marker):
            continue
        try:
            with open(marker, encoding="utf-8") as fh:
                archetype = fh.read().strip().splitlines()[0].strip()
        except Exception:
            errors += 1
            continue
        if not archetype:
            errors += 1
            continue
        try:
            res = await verify_against_paraflow_goldens(
                mission_id=mid, archetype=archetype, workspace_path=mdir,
            )
        except Exception as exc:
            errors += 1
            results.append({
                "mission_id": mid,
                "archetype": archetype,
                "error": str(exc),
            })
            continue
        audited += 1
        verdict = res.get("verdict")
        if verdict == "paraflow_par":
            par += 1
        elif verdict == "paraflow_partial":
            partial += 1
        elif verdict == "paraflow_gap":
            gaps += 1
        results.append({
            "mission_id": mid,
            "archetype": archetype,
            "verdict": verdict,
            "score": res.get("score"),
        })
    return {
        "audited": audited,
        "par": par,
        "partial": partial,
        "gaps": gaps,
        "errors": errors,
        "results": results,
        "workspace_root": workspace_root,
    }
