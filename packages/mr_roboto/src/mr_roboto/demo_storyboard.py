"""Z7 T3B — ``demo/storyboard`` mr_roboto verb.

Generates a demo storyboard from a product spec text by calling an LLM
(via beckman.enqueue OVERHEAD lane).  The storyboard is a structured JSON
with ordered scenes, each containing:

  - id: str                 — e.g. "scene_1", "scene_2"
  - title: str              — short scene label
  - target_seconds: int     — how long this scene should run in the demo
  - viewport_state: str     — e.g. "home_page", "dashboard", "feature_xyz"
  - narrator_text: str      — script text for caption / audio-description
  - visual_only: bool       — True when scene has no narrator_text

Reversibility: ``full`` — writes storyboard.json to workspace; git-reversible.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.demo_storyboard")


# ---------------------------------------------------------------------------
# Subprocess helper (shared with demo_record / demo_edit)
# ---------------------------------------------------------------------------

async def _run_subprocess(cmd: list[str], timeout: float = 300.0) -> tuple[int, str, str]:
    import asyncio
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return -1, "", f"timeout after {timeout}s"
    return (proc.returncode or 0), (out or b"").decode("utf-8", "replace"), (err or b"").decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# LLM storyboard prompt
# ---------------------------------------------------------------------------

_STORYBOARD_SYSTEM = (
    "You are a demo video scriptwriter for software products. "
    "Given a product description, generate a JSON storyboard for a product demo video. "
    "The storyboard must have:\n"
    '  "title": string (demo title),\n'
    '  "total_target_seconds": int (sum of scene target_seconds),\n'
    '  "scenes": array of scene objects, each with:\n'
    '    "id": "scene_N",\n'
    '    "title": string (short scene label),\n'
    '    "target_seconds": int (3–120 per scene),\n'
    '    "viewport_state": string (what the UI shows, snake_case),\n'
    '    "narrator_text": string (caption / script; empty string if visual-only),\n'
    '    "visual_only": boolean (true only when narrator_text is empty).\n'
    "Use 3–6 scenes for a concise demo. "
    "Reply ONLY with a single valid JSON object — no preamble, no code fences."
)

_STORYBOARD_USER_TMPL = (
    "Product description:\n{spec_text}\n\n"
    "Generate the storyboard JSON now."
)


async def _enqueue_storyboard_llm(
    spec: dict,
    *,
    parent_id: int | None = None,
    await_inline: bool = True,
) -> Any:
    """Call beckman.enqueue to run the LLM storyboard call.

    Extracted as a module-level function so tests can monkeypatch it without
    mocking the entire beckman module.
    """
    import general_beckman
    return await general_beckman.enqueue(spec, parent_id=parent_id, await_inline=await_inline)


def _parse_storyboard_response(content: str) -> dict | None:
    """Extract a JSON storyboard dict from LLM response content."""
    if not content:
        return None
    content = content.strip()
    # Strip code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "scenes" in parsed:
            return parsed
        return None
    except (json.JSONDecodeError, ValueError):
        # Try to find a JSON object in the text
        import re
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, dict) and "scenes" in parsed:
                    return parsed
            except Exception:
                pass
        return None


async def run(
    mission_id: int,
    spec_text: str,
    workspace_path: str,
    parent_task_id: int | None = None,
) -> dict[str, Any]:
    """Generate a demo storyboard via LLM and write it to workspace.

    Returns::

        {"ok": True, "storyboard": {...}, "storyboard_path": str}
        {"ok": False, "error": str}
    """
    if not spec_text or not spec_text.strip():
        return {"ok": False, "error": "spec_text is required but missing or empty"}

    messages = [
        {"role": "system", "content": _STORYBOARD_SYSTEM},
        {"role": "user", "content": _STORYBOARD_USER_TMPL.format(spec_text=spec_text[:6000])},
    ]

    _suffix = f"{time.monotonic_ns() % 1_000_000:06d}-{uuid.uuid4().hex[:6]}"
    spec = {
        "title": f"demo_storyboard:mission#{mission_id}:{_suffix}",
        "description": "LLM storyboard generation for demo pipeline",
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 2,
        "context": {
            "llm_call": {
                "raw_dispatch": True,
                "call_category": "overhead",
                "task": "reviewer",
                "agent_type": "reviewer",
                "difficulty": 4,
                "messages": messages,
                "failures": [],
                "estimated_input_tokens": 600,
                "estimated_output_tokens": 500,
            },
        },
    }

    try:
        task_result = await _enqueue_storyboard_llm(spec, parent_id=parent_task_id, await_inline=True)
    except Exception as exc:
        logger.warning("demo_storyboard: LLM enqueue raised: %r", exc)
        return {"ok": False, "error": f"LLM call failed: {exc}"}

    if task_result.status == "failed":
        return {"ok": False, "error": f"LLM storyboard call failed: {task_result.error}"}

    # Extract content from task_result
    result_data = getattr(task_result, "result", None) or {}
    content = result_data.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    content = str(content or "").strip()

    storyboard = _parse_storyboard_response(content)
    if storyboard is None:
        return {"ok": False, "error": f"LLM returned unparseable storyboard: {content[:200]!r}"}

    # Normalise scenes
    scenes = storyboard.get("scenes") or []
    for i, scene in enumerate(scenes):
        scene.setdefault("id", f"scene_{i + 1}")
        scene.setdefault("visual_only", not bool(scene.get("narrator_text", "").strip()))

    # Write to workspace
    demo_dir = os.path.join(workspace_path, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    storyboard_path = os.path.join(demo_dir, "storyboard.json")
    with open(storyboard_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)

    logger.info(
        "demo_storyboard: storyboard written",
        mission_id=mission_id,
        scene_count=len(scenes),
        storyboard_path=storyboard_path,
    )

    return {
        "ok": True,
        "storyboard": storyboard,
        "storyboard_path": storyboard_path,
        "scene_count": len(scenes),
    }
