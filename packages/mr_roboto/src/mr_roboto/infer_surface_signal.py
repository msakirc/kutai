"""Deterministic surface-signal inference (i2p step 3.5z, Stage 3).

The product surface (mobile / web / desktop / admin) is the founder's own
decision — stated at intake ("build an app", "a web sitesi"). Historically the
intake answer evaporated into freeform markdown and step 3.6 re-derived
``target_platform`` from PRD prose, which could drift from what the founder
actually said.

This mechanical step reads the founder's words (mission title + description,
plus the idea brief / charter / PRD summary as enrichment), infers the surface
set deterministically (``surface_infer``), projects it onto ``target_platform``,
and persists a structured ``surface_signal.json``. Step 3.6 honors it when
confidence is high, so the canonical build signal is grounded in the founder's
intent rather than re-derived from prose.

LLM-free and Telegram-free (unit-testable; no heavy imports). See
docs/superpowers/specs/2026-06-17-surface-single-source-design.md.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from mr_roboto.surface_infer import (
    infer_surfaces,
    target_platform_from_surfaces,
)

logger = logging.getLogger(__name__)

# Artifacts (in priority order) folded into the inference text. Mission
# title/description is the rawest founder signal; the rest is enrichment.
_TEXT_ARTIFACTS = ("idea_brief_final", "product_charter", "prd_final_summary")


async def _gather_founder_text(mission_id: int | None) -> str:
    """The founder's OWN words — mission title + description. This is the
    authoritative surface signal: when the founder said "app", the founder
    means a mobile app, and no LLM-generated prose may overrule that."""
    parts: list[str] = []
    if mission_id is None:
        return ""
    try:
        from dabidabi import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT title, description FROM missions WHERE id = ?", (mission_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        if row:
            for cell in row:
                if isinstance(cell, str) and cell.strip():
                    parts.append(cell)
    except Exception as exc:
        logger.debug("infer_surface_signal: missions lookup failed: %s", exc)
    return "\n".join(parts)


async def _gather_enrichment_text(mission_id: int | None) -> str:
    """LLM-generated enrichment (idea brief / charter / PRD summary). Consulted
    ONLY when the founder named no surface — its prose can carry strong platform
    words ('dashboard', 'SaaS', 'web platform') that previously outranked the
    founder's own colloquial 'app' and flipped target_platform to web."""
    parts: list[str] = []
    if mission_id is None:
        return ""
    for name in _TEXT_ARTIFACTS:
        try:
            from src.workflows.engine.hooks import get_artifact_store
            store = get_artifact_store()
            raw = await store.retrieve(mission_id, name)
            if isinstance(raw, str) and raw.strip():
                parts.append(raw)
            elif isinstance(raw, dict):
                parts.append(json.dumps(raw, ensure_ascii=False))
        except Exception as exc:
            logger.debug("infer_surface_signal: %s lookup failed: %s", name, exc)
    return "\n".join(parts)


async def infer_surface_signal(
    task: dict,
    *,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Infer + persist ``.charter/surface_signal.json``. Always completes.

    Returns the signal dict ``{_schema_version, mission_id, surfaces,
    primary_surface, target_platform, confidence, source}``. Low confidence /
    no signal still writes (confidence='low', target_platform=null) so 3.6
    sees the absence explicitly and derives from the PRD itself.
    """
    mission_id = task.get("mission_id")
    # Founder words win: infer from the founder's OWN title+description first.
    founder_text = await _gather_founder_text(mission_id)
    inf = infer_surfaces(founder_text)
    source = "founder_words"
    if inf["confidence"] == "low":
        # The founder named no surface — fall back to the full text (founder +
        # LLM enrichment), the pre-2026-06-27 behavior, so a silent founder
        # still gets a best-effort guess instead of null. The founder text is
        # kept in the blend so a weak founder hint still counts here.
        enrichment = await _gather_enrichment_text(mission_id)
        combined = (founder_text + "\n" + enrichment).strip()
        inf = infer_surfaces(combined)
        source = "enrichment_fallback"
    surfaces = inf["surfaces"]

    signal: dict[str, Any] = {
        "_schema_version": "1",
        "mission_id": int(mission_id) if mission_id is not None else None,
        "surfaces": surfaces,
        "primary_surface": inf["primary_surface"],
        "target_platform": target_platform_from_surfaces(surfaces),
        "confidence": inf["confidence"],
        "source": source,
    }

    if workspace_path is None and mission_id is not None:
        try:
            from src.tools.workspace import get_mission_workspace
            workspace_path = str(get_mission_workspace(int(mission_id)))
        except Exception as exc:
            logger.warning(
                "infer_surface_signal: workspace resolve failed for mission=%s: "
                "%s — signal computed but not persisted", mission_id, exc,
            )

    if workspace_path:
        charter_dir = os.path.join(workspace_path, ".charter")
        os.makedirs(charter_dir, exist_ok=True)
        path = os.path.join(charter_dir, "surface_signal.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(signal, f, ensure_ascii=False, indent=2)

    logger.info(
        "infer_surface_signal: mission=%s surfaces=%s target_platform=%r (%s)",
        mission_id, surfaces, signal["target_platform"], inf["confidence"],
    )
    return {"status": "completed", **signal}
