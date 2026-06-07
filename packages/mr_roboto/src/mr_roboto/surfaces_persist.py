"""Persist the founder's surface pick to ``.charter/surfaces.json`` (5.0b).

The 5.0b ``surfaces_lock`` step asks the founder which platforms a product
targets via a reply keyboard ("mobile + web", ...). This module turns the
chosen label into canonical surface tokens and materialises the
``surfaces.json`` the ``verify_surfaces_shape`` post-step asserts on.

Kept LLM-free and Telegram-free so it is unit-testable and so the Telegram
callback handler can call it directly when the founder taps a button.
"""
from __future__ import annotations

import json
import os
from typing import Any

# Mirror verify_surfaces_shape.VALID_SURFACES — single source of truth for
# what a legal surface token is.
VALID_SURFACES = ("mobile", "web", "desktop", "admin")


def parse_surface_choice(option_label: str) -> list[str]:
    """Turn a reply-keyboard label into canonical surface tokens.

    "mobile only"            -> ["mobile"]
    "mobile + web + admin"   -> ["mobile", "web", "admin"]

    Unknown tokens (e.g. "tablet") and duplicates are dropped. Order of
    first appearance is preserved so ``primary_surface`` can default to the
    first item.
    """
    tokens: list[str] = []
    for part in str(option_label).split("+"):
        words = part.strip().split()
        if not words:
            continue
        tok = words[0].lower()
        if tok in VALID_SURFACES and tok not in tokens:
            tokens.append(tok)
    return tokens


async def write_surfaces_json(
    *,
    mission_id: int,
    option_label: str,
    primary_surface: str | None = None,
    confirmed_at: str | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Write ``<workspace>/.charter/surfaces.json`` and return its contents.

    Schema (matches ``verify_surfaces_shape``):
    ``{_schema_version, mission_id, surfaces, primary_surface, founder_confirmed_at}``.

    Raises ``ValueError`` if no valid surface token can be parsed — a human
    confirmation gate must never persist an empty/garbage selection.
    """
    surfaces = parse_surface_choice(option_label)
    if not surfaces:
        raise ValueError(f"no valid surfaces parsed from {option_label!r}")

    primary = primary_surface if primary_surface in surfaces else surfaces[0]

    if confirmed_at is None:
        from datetime import datetime, timezone
        confirmed_at = datetime.now(timezone.utc).isoformat()

    data: dict[str, Any] = {
        "_schema_version": "1",
        "mission_id": int(mission_id),
        "surfaces": surfaces,
        "primary_surface": primary,
        "founder_confirmed_at": confirmed_at,
    }

    if workspace_path is None:
        from src.tools.workspace import get_mission_workspace
        workspace_path = str(get_mission_workspace(int(mission_id)))

    charter_dir = os.path.join(workspace_path, ".charter")
    os.makedirs(charter_dir, exist_ok=True)
    path = os.path.join(charter_dir, "surfaces.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
