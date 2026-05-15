"""Z7 T3B A3.r1 — ``demo/accessibility_pass`` mr_roboto verb.

Generates an accessibility manifest for the demo pipeline covering:

  1. **alt_texts** — one entry per scene describing what appears on screen
     (derived from viewport_state + narrator_text).
  2. **audio_description_track** — entries for visual-only scenes (scenes
     where ``visual_only=True`` or ``narrator_text`` is empty), describing
     what a screen-reader user would miss.
  3. **keyboard_nav_variant** — ordered steps for each scene describing how
     a keyboard-only user navigates the same flow (Tab / Enter walkthrough).

No real accessibility tooling (axe-core etc.) is run here — this verb
produces the *manifest scaffold* that founders fill in / approve.  The
``demo_accessibility_check`` posthook validates the manifest is complete.

Reversibility: ``full`` — writes a local JSON file; git-reversible.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.demo_accessibility_pass")


def _alt_text_for_scene(scene: dict) -> str:
    """Derive a descriptive alt text from scene metadata.

    Combines viewport_state + narrator_text to give a concise description
    of what appears on screen during this scene.
    """
    viewport = scene.get("viewport_state", "").replace("_", " ").strip()
    narrator = (scene.get("narrator_text") or "").strip()
    title = (scene.get("title") or "").strip()

    parts = []
    if title:
        parts.append(title)
    if viewport:
        parts.append(f"screen: {viewport}")
    if narrator:
        # Use first sentence of narrator as supplemental description
        first_sentence = narrator.split(".")[0].strip()
        if first_sentence:
            parts.append(first_sentence)

    alt = ". ".join(parts) if parts else f"Demo scene {scene.get('id', 'unknown')}"
    return alt[:200]


def _audio_description_for_visual_scene(scene: dict) -> str:
    """Generate a placeholder audio description for a visual-only scene.

    Describes what a sighted user sees that a blind user would miss.
    This is a scaffold — founders are expected to edit/approve.
    """
    viewport = scene.get("viewport_state", "").replace("_", " ").strip()
    title = (scene.get("title") or "").strip()
    if viewport:
        return (
            f"[Audio description] The screen displays the {viewport}. "
            f"{title + ': ' if title else ''}"
            "The cursor moves to highlight key interface elements. "
            "(Founder: please describe specific UI actions visible in this scene.)"
        )
    return (
        "[Audio description] Visual demonstration. "
        "(Founder: please describe what appears on screen during this scene.)"
    )


def _keyboard_nav_step_for_scene(scene: dict, step_number: int) -> dict:
    """Derive a keyboard navigation step for the scene."""
    viewport = scene.get("viewport_state", "").replace("_", " ").strip()
    title = (scene.get("title") or "").strip()
    narrator = (scene.get("narrator_text") or "").strip()

    if navigator := narrator:
        step_desc = (
            f"Tab to navigate to the {viewport or 'page'} section. "
            f"Press Enter to activate. (Scene: {title or scene.get('id', 'unknown')})"
        )
    else:
        step_desc = (
            f"Tab to focus the {viewport or 'page'} area. "
            f"Use arrow keys to explore content. "
            f"(Scene: {title or scene.get('id', 'unknown')})"
        )

    return {
        "step": step_number,
        "scene_id": scene.get("id", f"scene_{step_number}"),
        "step_description": step_desc,
    }


async def run(
    mission_id: int,
    workspace_path: str,
    storyboard_path: str,
) -> dict[str, Any]:
    """Generate demo accessibility manifest.

    Returns::

        {"ok": True, "manifest_path": str, "alt_text_count": int,
         "audio_desc_count": int, "keyboard_nav_steps": int}
        {"ok": False, "error": str}
    """
    if not os.path.exists(storyboard_path):
        return {
            "ok": False,
            "error": f"storyboard.json not found at {storyboard_path}",
        }

    try:
        with open(storyboard_path, encoding="utf-8") as f:
            storyboard = json.load(f)
    except Exception as exc:
        return {"ok": False, "error": f"storyboard.json parse error: {exc}"}

    scenes = storyboard.get("scenes") or []
    if not scenes:
        return {"ok": False, "error": "storyboard has no scenes"}

    # 1. Alt texts — one per scene
    alt_texts = [
        {
            "scene_id": scene.get("id", f"scene_{i+1}"),
            "alt_text": _alt_text_for_scene(scene),
        }
        for i, scene in enumerate(scenes)
    ]

    # 2. Audio description track — only for visual-only scenes
    audio_description_track = []
    for scene in scenes:
        is_visual_only = (
            scene.get("visual_only") is True
            or not (scene.get("narrator_text") or "").strip()
        )
        if is_visual_only:
            audio_description_track.append({
                "scene_id": scene.get("id", "scene_unknown"),
                "description": _audio_description_for_visual_scene(scene),
            })

    # 3. Keyboard nav variant — one step per scene
    keyboard_nav_variant = [
        _keyboard_nav_step_for_scene(scene, i + 1)
        for i, scene in enumerate(scenes)
    ]

    manifest = {
        "mission_id": mission_id,
        "alt_texts": alt_texts,
        "audio_description_track": audio_description_track,
        "keyboard_nav_variant": keyboard_nav_variant,
        "note": (
            "This manifest is a scaffold. Founder review and approval required "
            "before publishing. Edit alt_texts and audio_description_track entries "
            "to accurately describe what appears on screen."
        ),
    }

    # Write manifest
    demo_dir = os.path.join(workspace_path, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    manifest_path = os.path.join(demo_dir, "accessibility_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "demo_accessibility_pass: manifest written",
        mission_id=mission_id,
        alt_text_count=len(alt_texts),
        audio_desc_count=len(audio_description_track),
        keyboard_nav_steps=len(keyboard_nav_variant),
        manifest_path=manifest_path,
    )

    return {
        "ok": True,
        "manifest_path": manifest_path,
        "alt_text_count": len(alt_texts),
        "audio_desc_count": len(audio_description_track),
        "keyboard_nav_steps": len(keyboard_nav_variant),
    }
