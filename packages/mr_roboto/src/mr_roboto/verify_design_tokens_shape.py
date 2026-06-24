"""Mechanical shape-verifier for Z1 Tier 3 ``design_tokens.json``.

Pure parse + structural assertions; no LLM. Wired as the sibling
``verify`` step of i2p_v3 ``5.0a design_tokens_generation``.

Schema (``_schema_version: "1"``)::

    {
      "_schema_version": "1",
      "mission_id": <int>,
      "variants": {
        "light": {color, typography, border_radius, spacing,
                  borders, shadows, icon_library},
        "dark":  {<same shape>}
      },
      "tag_signature": "<emphasis>"
    }

Rejects: missing variants (light or dark), placeholder hex
(``#XXXXXX``, ``TBD``), missing top-level token block, malformed hex
codes, empty font_family_base.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from src.tools.workspace import get_mission_workspace
from yazbunu import get_logger

logger = get_logger("mr_roboto.verify_design_tokens_shape")

SCHEMA_VERSION = "1"

REQUIRED_VARIANTS = ("light", "dark")
OPTIONAL_VARIANTS = ("compact", "comfortable")
REQUIRED_VARIANT_BLOCKS = (
    "color",
    "typography",
    "border_radius",
    "spacing",
    "borders",
    "shadows",
)
REQUIRED_COLOR_GROUPS = ("primary", "background", "text")
REQUIRED_TYPOGRAPHY_KEYS = ("font_family_base", "scale")

# 3-, 6-, or 8-digit hex (with alpha) — case insensitive.
_HEX_RE = re.compile(r"^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")
_PLACEHOLDER_HEX_RE = re.compile(r"#X{3,8}", re.IGNORECASE)
_PLACEHOLDER_TOKENS = ("TBD", "TODO", "FIXME")


def _resolve_under(workspace_root: str, rel_path: str) -> str | None:
    if not isinstance(rel_path, str) or not rel_path:
        return None
    if os.path.isabs(rel_path):
        return None
    joined = os.path.normpath(os.path.join(workspace_root, rel_path))
    root_real = os.path.realpath(workspace_root)
    joined_real = os.path.realpath(joined)
    if not (joined_real == root_real or joined_real.startswith(root_real + os.sep)):
        return None
    return joined_real


def _walk_strings(node: Any):
    """Yield every str leaf in a nested dict/list."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_strings(v)


def _check_variant(variant_name: str, variant: Any) -> str | None:
    """Return error string or None on pass."""
    if not isinstance(variant, dict):
        return f"variants.{variant_name} must be object, got {type(variant).__name__}"

    for blk in REQUIRED_VARIANT_BLOCKS:
        if blk not in variant:
            return f"variants.{variant_name} missing block {blk!r}"
        if not isinstance(variant[blk], dict):
            return f"variants.{variant_name}.{blk} must be object"
        if not variant[blk]:
            return f"variants.{variant_name}.{blk} must not be empty"

    color = variant["color"]
    for grp in REQUIRED_COLOR_GROUPS:
        if grp not in color:
            return f"variants.{variant_name}.color missing group {grp!r}"

    typography = variant["typography"]
    for k in REQUIRED_TYPOGRAPHY_KEYS:
        if k not in typography:
            return f"variants.{variant_name}.typography missing {k!r}"
    fam = typography.get("font_family_base")
    if not isinstance(fam, str) or not fam.strip():
        return f"variants.{variant_name}.typography.font_family_base must be non-empty string"
    scale = typography.get("scale")
    if not isinstance(scale, dict) or not scale:
        return f"variants.{variant_name}.typography.scale must be non-empty object"

    # Walk every string leaf — collect placeholder + malformed hex.
    for s in _walk_strings(variant):
        for ph in _PLACEHOLDER_TOKENS:
            if ph in s:
                return (
                    f"variants.{variant_name} contains placeholder token "
                    f"{ph!r} (value: {s!r})"
                )
        if _PLACEHOLDER_HEX_RE.search(s):
            return (
                f"variants.{variant_name} contains placeholder hex "
                f"(value: {s!r})"
            )
        # Hex check: only validate strings that *look* like hex (start with #
        # and are short enough). Avoids flagging shadow strings like
        # "0 1px 3px rgba(...)".
        if (
            s.startswith("#")
            and len(s) <= 9
            and not _HEX_RE.match(s)
        ):
            return f"variants.{variant_name} malformed hex {s!r}"

    return None


def verify_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate parsed design_tokens dict. Returns ``{"ok": bool, ...}``."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": f"payload must be object, got {type(payload).__name__}"}

    if str(payload.get("_schema_version")) != SCHEMA_VERSION:
        return {
            "ok": False,
            "error": (
                f"_schema_version mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {payload.get('_schema_version')!r}"
            ),
        }

    if "mission_id" not in payload:
        return {"ok": False, "error": "missing mission_id"}

    if "tag_signature" not in payload:
        return {"ok": False, "error": "missing tag_signature"}
    tag = payload.get("tag_signature")
    if not isinstance(tag, str) or not tag.strip():
        return {"ok": False, "error": "tag_signature must be non-empty string"}

    variants = payload.get("variants")
    if not isinstance(variants, dict):
        return {"ok": False, "error": "variants must be an object"}

    for required in REQUIRED_VARIANTS:
        if required not in variants:
            return {"ok": False, "error": f"missing required variant {required!r}"}

    # Validate every variant present (required + any optional density).
    for name, body in variants.items():
        err = _check_variant(name, body)
        if err is not None:
            return {"ok": False, "error": err}

    return {
        "ok": True,
        "variants": sorted(variants.keys()),
        "tag_signature": tag,
    }


async def verify_design_tokens_shape(
    mission_id: int | None,
    path: str = ".style/design_tokens.json",
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Resolve, parse, and validate a design_tokens.json artifact."""
    if workspace_path is None:
        if mission_id is None:
            return {"ok": False, "error": "no mission_id and no workspace_path"}
        workspace_path = get_mission_workspace(mission_id)

    abs_path = _resolve_under(workspace_path, path)
    if abs_path is None:
        return {"ok": False, "path": path, "error": "path rejected (absolute or traversal)"}
    if not os.path.isfile(abs_path):
        return {"ok": False, "path": path, "error": "file not found"}

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        return {"ok": False, "path": path, "error": f"invalid JSON: {e}"}
    except OSError as e:
        return {"ok": False, "path": path, "error": f"read failed: {e}"}

    res = verify_payload(payload)
    res["path"] = path
    return res
