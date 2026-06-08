"""Plan 3 — swap placehold.co <img> tags for real diffusion-generated PNGs.

Pipeline:
  1. Recursively walk `mission_{id}/.web/**/*.html`.
  2. Scan each file for <img> whose src is placehold.co.
  3. Enqueue ONE prompt_writer task (beckman.enqueue, await_inline) with
     the design context + placeholder list. Receive a placeholder_id ->
     prompt map. Robust to JSON-string TaskResult.result (recon-verified
     shape).
  4. Per placeholder: enqueue one image task (context.image_call.raw_dispatch).
     Beckman routes via Plan 1 v2's _select_for_admission(needs_image=True).
  5. PNG lands directly under .web/assets/<placeholder_id>.png (paintress
     writes the file; mechanic asks paintress to put it there).
  6. HTML <img src> rewritten to relative "assets/<id>.png".
  7. Graceful degrade: per-placeholder failure keeps the placehold.co URL.

Mirrors marketing_copy.py's mechanical shape — internally enqueues LLM +
image work through beckman, never calls dispatcher/HK/paintress directly
(feedback_singular_dispatcher_caller). Reversibility: "full"."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.swap_placeholder_images")


# ── Placeholder detection ──────────────────────────────────────────────

_PLACEHOLDER_HOST_RE = re.compile(r"^https?://placehold\.co/", re.IGNORECASE)
_IMG_RE = re.compile(r"<img\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r'(\b[a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*"([^"]*)"')
_DIM_RE = re.compile(r"/(\d{2,4})x(\d{2,4})", re.IGNORECASE)


def _parse_attrs(tag_inner: str) -> dict[str, str]:
    return {k.lower(): v for k, v in _ATTR_RE.findall(tag_inner)}


def _slug_from_path(html_path: str) -> str:
    return Path(html_path).stem


def _section_from_alt(alt: str) -> str:
    a = (alt or "").lower()
    if any(t in a for t in ("hero", "header", "banner")):
        return "hero"
    if any(t in a for t in ("avatar", "portrait", "headshot")):
        return "avatar"
    if "icon" in a:
        return "icon"
    if "background" in a or "bg" in a:
        return "background"
    return "feature"


def _scan_placeholders(html_path: str) -> list[dict[str, Any]]:
    try:
        with open(html_path, encoding="utf-8") as fh:
            html = fh.read()
    except OSError:
        return []
    slug = _slug_from_path(html_path)
    out: list[dict[str, Any]] = []
    occ = 0
    for m in _IMG_RE.finditer(html):
        attrs = _parse_attrs(m.group(1) or "")
        src = (attrs.get("src") or "").strip()
        if not _PLACEHOLDER_HOST_RE.search(src):
            continue
        alt = (attrs.get("alt") or "").strip()
        dim_m = _DIM_RE.search(src)
        w, h = (int(dim_m.group(1)), int(dim_m.group(2))) if dim_m else (512, 512)
        out.append({
            "placeholder_id": f"{slug}__{occ}",
            "alt": alt, "width": w, "height": h,
            "section": _section_from_alt(alt),
            "original_src": src,
            "tag_span": (m.start(), m.end()),
            "html_path": html_path,
        })
        occ += 1
    return out


# ── Workspace + assets helpers ─────────────────────────────────────────

def _web_root(workspace_path: str) -> str:
    return os.path.join(workspace_path, ".web")


def _assets_dir(workspace_path: str) -> str:
    p = os.path.join(workspace_path, ".web", "assets")
    os.makedirs(p, exist_ok=True)
    return p


def _list_html_files(workspace_path: str) -> list[str]:
    """v2 fix: recursive walk of <ws>/.web/**/*.html (Plan 3 v1 was flat
    and missed subdirectory screens)."""
    root = _web_root(workspace_path)
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".html"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


# ── TaskResult.result parser (the recon-confirmed v2 fix) ──────────────

def _parse_task_result(result_obj) -> dict:
    """TaskResult.result is a JSON STRING in production (recon: orchestrator
    json.dumps at :63). Mirror dispatcher's _task_result_to_request_response
    — accept both string and dict; json.loads the string FIRST before any
    isinstance check on the decoded value."""
    raw = getattr(result_obj, "result", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}
    return {}


# ── beckman wrapper (test-patchable, mirrors marketing_copy) ───────────

async def _enqueue_beckman(spec: dict, **kwargs):
    from general_beckman import enqueue as _enqueue
    return await _enqueue(spec, **kwargs)


# ── Main entry ─────────────────────────────────────────────────────────

async def swap_placeholder_images(
    mission_id: int,
    workspace_path: str | None = None,
    design_tokens: dict | None = None,
    brand_voice: str | None = None,
) -> dict[str, Any]:
    """Best-effort: per-placeholder failures keep the original placeholder.
    Returns:
      {ok: bool, replaced_count, skipped_count, html_files_seen,
       html_files_changed, errors: list[str]}
    """
    from src.tools.workspace import get_mission_workspace
    workspace_path = workspace_path or get_mission_workspace(int(mission_id))
    logger.info(
        "swap_placeholder_images: starting (mission_id=%s, workspace_path=%s)",
        mission_id, workspace_path,
    )

    html_files = _list_html_files(workspace_path)
    if not html_files:
        return {
            "ok": True, "replaced_count": 0, "skipped_count": 0,
            "html_files_seen": 0, "html_files_changed": 0, "errors": [],
        }

    all_placeholders: list[dict[str, Any]] = []
    for h in html_files:
        all_placeholders.extend(_scan_placeholders(h))

    if not all_placeholders:
        return {
            "ok": True, "replaced_count": 0, "skipped_count": 0,
            "html_files_seen": len(html_files), "html_files_changed": 0,
            "errors": [],
        }

    # Task 5 fills prompt_writer; Task 6 fills fanout. Scaffold returns scan.
    return {
        "ok": True, "replaced_count": 0, "skipped_count": len(all_placeholders),
        "html_files_seen": len(html_files), "html_files_changed": 0,
        "errors": [],
    }
