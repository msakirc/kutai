"""Z8 T4C — phase 13.3.playbooks generator.

Reads the mission's spec.tech_stack (passed in via payload OR inferred from
mission row) and emits an ``incident_playbooks`` artifact listing every
matching incident_playbook recipe. The on-call agent (T4A) consults this
artifact at alert time so it knows which playbooks have been pre-blessed
for the mission.

Output artifact shape::

    {
      "mission_id": <int>,
      "tech_stack": ["fastapi", "postgres", ...],
      "playbooks": [
        {"id": "incident_playbook_db_disk_full_v1",
         "description": "...",
         "tech_stack": [...],
         "alerts": [{integration, event}, ...],
         "action_sequence": [...],
         "on_failure": [...]},
        ...
      ]
    }

No filesystem writes — the artifact is returned in the action result and
the orchestrator persists it via the standard produces/artifact path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger
from src.ops.playbooks import match_playbooks_for_stack, to_dict

logger = get_logger("mr_roboto.generate_playbooks")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    mission_id = task.get("mission_id")
    recipes_dir = str(payload.get("recipes_dir") or "recipes")

    tech_stack = await _resolve_tech_stack(mission_id, payload)
    playbooks = match_playbooks_for_stack(tech_stack, recipes_dir=recipes_dir)
    artifact = {
        "mission_id": mission_id,
        "tech_stack": [str(s).lower() for s in tech_stack],
        "playbooks": [to_dict(pb) for pb in playbooks],
    }
    logger.info(
        "generate_playbooks emitted",
        mission_id=mission_id,
        tech_stack=tech_stack,
        playbook_count=len(playbooks),
    )
    return {
        "status": "ok",
        "artifact_name": "incident_playbooks",
        "artifact": artifact,
        # Also emit a flat playbook_id list so the on-call agent can
        # ``in artifact["playbook_ids"]``-check at alert time without
        # re-walking the nested structure.
        "playbook_ids": [pb.id for pb in playbooks],
    }


async def _resolve_tech_stack(mission_id: int | None, payload: dict) -> list[str]:
    """Resolve the mission's tech_stack from (in order):

    1. explicit ``payload["tech_stack"]`` (list or "+"-joined string),
    2. ``payload["spec_path"]`` JSON file → ``tech_stack`` key,
    3. mission row's spec JSON if mission_id is set.

    Returns an empty list when nothing resolves — generate_playbooks does
    not fail in this case; the artifact will simply be empty and the
    phase 13 step will record that no playbooks could be pre-blessed.
    """
    raw = payload.get("tech_stack")
    if raw:
        if isinstance(raw, str):
            return [p.strip().lower() for p in raw.split("+") if p.strip()]
        return [str(p).strip().lower() for p in raw if str(p).strip()]

    spec_path = payload.get("spec_path")
    if spec_path:
        try:
            text = Path(str(spec_path)).read_text(encoding="utf-8")
            data = json.loads(text)
            stack = data.get("tech_stack") or []
            if isinstance(stack, str):
                return [p.strip().lower() for p in stack.split("+") if p.strip()]
            return [str(p).strip().lower() for p in stack if str(p).strip()]
        except Exception as exc:
            logger.warning(
                "spec_path read failed",
                spec_path=spec_path,
                error=str(exc),
            )

    if mission_id is not None:
        try:
            from dabidabi import get_db
            db = await get_db()
            async with db.execute(
                "SELECT spec FROM missions WHERE id = ?", (int(mission_id),)
            ) as cur:
                row = await cur.fetchone()
            if row and row[0]:
                spec = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                stack = (spec or {}).get("tech_stack") or []
                if isinstance(stack, str):
                    return [p.strip().lower() for p in stack.split("+") if p.strip()]
                return [str(p).strip().lower() for p in stack if str(p).strip()]
        except Exception as exc:
            logger.debug("mission spec resolve skipped: %s", exc)

    return []
