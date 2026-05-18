"""Ingest founder-uploaded visuals (sketches, screenshots, moodboards).

This is the mechanical action behind B7 + C16 of the Z1 i2p evolution: when
the founder drops one or more images into a mission's ``.intake/visuals/``
directory (typically via the Telegram photo handler), this executor sends each
image to a vision-capable model, extracts a small structured brief
(intent / structural elements / palette / style / readable text / confidence),
and aggregates the results into a single ``visual_brief.md`` artifact under
``mission_{mission_id}/.intake/visual_brief.md``.

Honest scope:
- We do NOT do pixel-perfect color clustering — palette is whatever the vision
  model surfaces. Same for OCR.
- We do NOT call any image-generation model. This is ingest-only; generation is
  Z2 work (``gorsel_ustasi``).
- We require a vision-capable model to be reachable. If none is registered,
  we fail loudly with ``vision_capability_unavailable`` rather than text-only
  hallucinating.

Schema (frontmatter ``_schema_version: "1"``) is tracked separately so future
schema migrations don't break existing ``visual_brief.md`` consumers.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.ingest_visual")


SCHEMA_VERSION = "1"

ALLOWED_PURPOSES = (
    "competitor_screenshot",
    "moodboard",
    "wireframe_sketch",
    "inspiration",
    "screenshot_of_existing_product",
)

# Question we send to the vision model per image. Asks for a JSON object so we
# can parse it back into structured fields.
_PROMPT = (
    "Analyze this image as a UI / product reference. Return ONLY a single JSON "
    "object (no prose, no code fences) with these exact keys:\n"
    "  - inferred_intent: string, one paragraph describing what the founder is "
    "showing and why it's likely useful as a reference.\n"
    "  - structural_elements: array of strings naming visible structural blocks "
    "(e.g. 'header', 'hero', 'cta', 'grid', 'card', 'nav', 'footer', 'form', "
    "'list', 'sidebar').\n"
    "  - color_palette_inferred: array of 3-5 hex color codes (e.g. '#1a73e8') "
    "you see dominating the image.\n"
    "  - style_keywords: array of short descriptors (e.g. 'clean', 'dense', "
    "'minimal', 'playful', 'clinical').\n"
    "  - text_excerpts: array of strings, the readable text in the image (best "
    "effort OCR; empty array if none).\n"
    "  - confidence: float 0.0-1.0, your self-assessed confidence in the above.\n"
    "Return strictly valid JSON, nothing else."
)


@dataclass
class _ImageBrief:
    path: str
    inferred_intent: str = ""
    structural_elements: list[str] = field(default_factory=list)
    color_palette_inferred: list[str] = field(default_factory=list)
    style_keywords: list[str] = field(default_factory=list)
    text_excerpts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None

    def as_frontmatter_dict(self) -> dict:
        return {
            "path": self.path,
            "inferred_intent": self.inferred_intent,
            "structural_elements": list(self.structural_elements),
            "color_palette_inferred": list(self.color_palette_inferred),
            "style_keywords": list(self.style_keywords),
            "text_excerpts": list(self.text_excerpts),
            "confidence": float(self.confidence),
        }


def _parse_vision_response(raw: str) -> dict:
    """Pull a JSON object out of a vision model response.

    Tolerates leading/trailing prose and code fences. Returns ``{}`` if no
    parseable JSON is found.
    """
    if not raw:
        return {}
    text = raw.strip()
    # Strip code fence
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    # Last-ditch: grab the first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def _coerce_str_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _coerce_confidence(v: Any) -> float:
    try:
        f = float(v)
    except Exception:
        return 0.0
    if f != f:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, f))


async def _call_vision(filepath: str) -> tuple[str, str | None]:
    """Call the existing vision tool. Returns ``(response, error)``.

    ``error`` is ``None`` on success. If the underlying tool returns a string
    starting with ``"Error"`` we treat it as failure (the existing convention
    in :func:`src.tools.vision.analyze_image`).
    """
    try:
        from src.tools.vision import analyze_image
    except Exception as exc:
        return "", f"vision_import_failed: {exc}"
    try:
        resp = await analyze_image(filepath, question=_PROMPT)
    except Exception as exc:
        return "", f"vision_call_raised: {exc}"
    if isinstance(resp, str) and resp.lstrip().lower().startswith("error"):
        return "", resp.strip()
    return resp or "", None


def _is_vision_capability_unavailable(error: str) -> bool:
    """Detect the 'no vision-capable model' failure mode.

    Fatih Hoca's selector returns no candidate when ``needs_vision=True`` and
    no registered model has ``has_vision=True``. The dispatcher surfaces this
    as ``no eligible model`` / ``selection failed``. We map a few known error
    substrings to the explicit ``vision_capability_unavailable`` reason so
    callers (and the failure surface) get a sharp signal.
    """
    if not error:
        return False
    e = error.lower()
    needles = (
        "no eligible model",
        "no candidates",
        "selection failed",
        "no model available",
        "vision capability",
        "vision model",
    )
    return any(n in e for n in needles)


def _build_visual_brief_md(
    *,
    mission_id: int,
    purpose: str,
    briefs: list[_ImageBrief],
) -> str:
    """Render the aggregated artifact (YAML frontmatter + per-image sections)."""
    import yaml  # local import to keep module import cheap

    successful = [b for b in briefs if b.error is None]
    frontmatter = {
        "_schema_version": SCHEMA_VERSION,
        "mission_id": int(mission_id),
        "purpose": purpose,
        "images": [b.as_frontmatter_dict() for b in successful],
        "evidence_refs": [b.path for b in successful],
    }
    fm_yaml = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()

    body_parts: list[str] = []
    for idx, b in enumerate(briefs, start=1):
        fname = os.path.basename(b.path)
        if b.error is not None:
            body_parts.append(
                f"## Image {idx} — {fname}\n\n_Skipped: {b.error}_\n"
            )
            continue
        struct = ", ".join(b.structural_elements) or "_(none surfaced)_"
        palette = ", ".join(b.color_palette_inferred) or "_(none surfaced)_"
        style = ", ".join(b.style_keywords) or "_(none surfaced)_"
        if b.text_excerpts:
            text_block = "\n".join(f"- {t!r}" for t in b.text_excerpts)
        else:
            text_block = "_(no readable text)_"
        body_parts.append(
            f"## Image {idx} — {fname}\n\n"
            f"**Inferred intent:** {b.inferred_intent or '_(no intent surfaced)_'}\n\n"
            f"**Structural elements:** {struct}\n\n"
            f"**Color palette:** {palette}\n\n"
            f"**Style:** {style}\n\n"
            f"**Readable text:**\n{text_block}\n\n"
            f"**Confidence:** {b.confidence:.2f}\n"
        )

    body = "\n".join(body_parts).rstrip() + "\n"
    return f"---\n{fm_yaml}\n---\n\n# Visual brief — mission {mission_id}\n\n{body}"


async def ingest_visual(
    *,
    mission_id: int,
    file_paths: list[str],
    purpose: str,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Run the visual-ingest pipeline.

    Returns a dict shaped for the mr_roboto router:
    - On success: ``{"ok": True, "artifact_path": ..., "image_count": N,
      "purpose": ..., "skipped": [...]}``
    - On failure: ``{"ok": False, "reason": <slug>, "detail": <str>}``
    """
    if not file_paths:
        return {"ok": False, "reason": "no_images_at_paths", "detail": "empty file_paths"}
    if purpose not in ALLOWED_PURPOSES:
        return {
            "ok": False,
            "reason": "invalid_purpose",
            "detail": f"purpose={purpose!r} not in {list(ALLOWED_PURPOSES)}",
        }

    # Filter to existing files; absent files count as "no_images_at_paths" if
    # NONE survive the filter.
    existing: list[str] = []
    missing: list[str] = []
    for p in file_paths:
        if isinstance(p, str) and os.path.isfile(p):
            existing.append(p)
        else:
            missing.append(str(p))
    if not existing:
        return {
            "ok": False,
            "reason": "no_images_at_paths",
            "detail": f"none of {len(file_paths)} paths exist on disk",
            "missing": missing,
        }

    # Resolve mission workspace lazily (some tests patch this).
    if workspace_path is None:
        from src.tools.workspace import get_mission_workspace
        workspace_path = get_mission_workspace(int(mission_id))
    intake_dir = os.path.join(workspace_path, ".intake")
    os.makedirs(intake_dir, exist_ok=True)

    briefs: list[_ImageBrief] = []
    skipped: list[dict] = []
    for fp in existing:
        raw, err = await _call_vision(fp)
        if err is not None:
            if _is_vision_capability_unavailable(err):
                # Hard fail — vision capability missing is a system-level
                # problem, not a per-image issue.
                logger.error(
                    "ingest_visual: vision capability unavailable",
                    mission_id=mission_id,
                    error=err,
                )
                return {
                    "ok": False,
                    "reason": "vision_capability_unavailable",
                    "detail": err,
                }
            # Per-image failure: log + continue (P1 evidence: reason captured)
            logger.warning(
                "ingest_visual: skipping unreadable image",
                path=fp,
                error=err,
            )
            briefs.append(_ImageBrief(path=fp, error=err))
            skipped.append({"path": fp, "error": err})
            continue

        parsed = _parse_vision_response(raw)
        brief = _ImageBrief(
            path=fp,
            inferred_intent=str(parsed.get("inferred_intent") or "").strip(),
            structural_elements=_coerce_str_list(parsed.get("structural_elements")),
            color_palette_inferred=_coerce_str_list(
                parsed.get("color_palette_inferred")
            ),
            style_keywords=_coerce_str_list(parsed.get("style_keywords")),
            text_excerpts=_coerce_str_list(parsed.get("text_excerpts")),
            confidence=_coerce_confidence(parsed.get("confidence")),
        )
        briefs.append(brief)

    successful = [b for b in briefs if b.error is None]
    if not successful:
        # Every image failed for non-capability reasons (corrupted, etc.).
        return {
            "ok": False,
            "reason": "no_images_readable",
            "detail": f"{len(briefs)} attempted, all failed",
            "skipped": skipped,
        }

    artifact_path = os.path.join(intake_dir, "visual_brief.md")
    md = _build_visual_brief_md(
        mission_id=int(mission_id),
        purpose=purpose,
        briefs=briefs,
    )
    # Atomic-ish write via tmp + replace
    tmp_path = artifact_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    os.replace(tmp_path, artifact_path)

    logger.info(
        "ingest_visual: visual_brief.md written",
        mission_id=mission_id,
        artifact_path=artifact_path,
        image_count=len(successful),
        skipped_count=len(skipped),
        purpose=purpose,
    )

    return {
        "ok": True,
        "artifact_path": artifact_path,
        "image_count": len(successful),
        "purpose": purpose,
        "skipped": skipped,
        "_schema_version": SCHEMA_VERSION,
    }
