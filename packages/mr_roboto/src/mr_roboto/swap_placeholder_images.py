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
  6. HTML <img src> rewritten to a relative ref computed from EACH html
     file's own dir to the flat asset (os.path.relpath): root HTML gets
     "assets/<id>.png", a subdir screen gets "../assets/<id>.png", so the
     ref resolves correctly from any subdir in a static file server.
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

    prompt_map = await _call_prompt_writer(
        mission_id=int(mission_id), placeholders=all_placeholders,
        design_tokens=design_tokens, brand_voice=brand_voice,
    )
    if prompt_map is None:
        return {
            "ok": True, "replaced_count": 0,
            "skipped_count": len(all_placeholders),
            "html_files_seen": len(html_files), "html_files_changed": 0,
            "errors": ["prompt_writer task did not return a usable prompt map"],
        }
    fanout = await _fanout_and_rewrite(
        workspace_path, all_placeholders, prompt_map,
        mission_id=int(mission_id),
    )
    return {
        "ok": True,
        "replaced_count": fanout.get("replaced", 0),
        "skipped_count": fanout.get("skipped", 0),
        "html_files_seen": len(html_files),
        "html_files_changed": fanout.get("html_files_changed", 0),
        "errors": fanout.get("errors", []),
    }


# ── prompt_writer enqueue (JSON-string-safe) ───────────────────────────

async def _call_prompt_writer(
    *, mission_id: int,
    placeholders: list[dict[str, Any]],
    design_tokens: dict | None,
    brand_voice: str | None,
) -> dict[str, str] | None:
    """Enqueue one prompt_writer task. Returns placeholder_id -> prompt map,
    or None on failure. Robust to JSON-string TaskResult.result.

    ``artifact_schema`` and ``is_workflow_step`` are set in the task context
    for two reasons:
    - ``artifact_schema`` arms the constrained_emit POSTHOOK (wired by
      ``general_beckman/posthooks.py::determine_posthooks`` when it finds a
      constrainable artifact_schema in the task context).
    - ``is_workflow_step`` triggers ``post_execute_workflow_step`` on the
      beckman apply path.

    IMPORTANT: the constrained_emit posthook runs as a CHILD task AFTER
    ``await_inline`` returns. This executor therefore consumes the RAW LLM
    emit — the posthook has NOT yet run when the result is read here.
    Malformed output degrades gracefully: _parse_task_result returns {} →
    no prompts list → the function returns None → all placeholders are
    skipped with their original placehold.co URLs intact. The executor's own
    graceful-degrade (not the posthook) is the real safety net for bad JSON."""
    from src.agents.prompt_writer import (
        PROMPT_WRITER_ARTIFACT_SCHEMA,
        load_diffusion_prompt_template,
    )

    visible = [
        {"placeholder_id": p["placeholder_id"], "alt": p["alt"],
         "width": p["width"], "height": p["height"], "section": p["section"]}
        for p in placeholders
    ]
    try:
        template_text = load_diffusion_prompt_template()
    except Exception:
        template_text = None

    spec = {
        "title": f"prompt_writer:mission#{mission_id}",
        "description": "Enrich placeholder <img> intents into diffusion prompts.",
        "agent_type": "prompt_writer",
        "kind": "main_work",
        "priority": 5,
        "mission_id": mission_id,
        "context": {
            "design_tokens": design_tokens or {},
            "brand_voice": brand_voice or "",
            "placeholders": visible,
            "diffusion_template": template_text or "",
            # Arm the constrained-emit safety net (constrained_emit.maybe_apply
            # reads both of these off the task context). Without them the
            # post-emit structured pass is a no-op.
            "is_workflow_step": True,
            "artifact_schema": PROMPT_WRITER_ARTIFACT_SCHEMA,
        },
    }
    try:
        result = await _enqueue_beckman(spec, await_inline=True)
    except Exception as exc:
        logger.warning("prompt_writer enqueue raised: %s", exc)
        return None

    if getattr(result, "status", "") != "completed":
        logger.warning(
            "prompt_writer task did not complete (status=%r, error=%r)",
            getattr(result, "status", ""), getattr(result, "error", ""),
        )
        return None

    parsed = _parse_task_result(result)
    # Tolerate both shapes: top-level prompts OR nested under "result".
    prompts = parsed.get("prompts")
    if prompts is None and isinstance(parsed.get("result"), dict):
        prompts = parsed["result"].get("prompts")
    if not isinstance(prompts, list):
        logger.warning("prompt_writer returned no prompts list (parsed=%r)", parsed)
        return None

    out: dict[str, str] = {}
    for entry in prompts:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("placeholder_id")
        prompt = entry.get("prompt")
        if isinstance(pid, str) and isinstance(prompt, str) and prompt.strip():
            out[pid] = prompt.strip()
    return out or None


# ── Per-placeholder image fanout + HTML rewrite ────────────────────────

async def _generate_one_image(
    *, placeholder: dict[str, Any], prompt: str, out_dir: str,
    mission_id: int,
) -> str | None:
    """Enqueue one image task; return the PNG path or None.

    Mirrors the /image cmd shape (raw_dispatch=True, prompt, out_dir,
    width/height, filename_hint, quality_tier). Enqueued via beckman —
    Plan 1 v2's admission routes agent_type=image → dispatch_image →
    paintress. JSON-string-safe result parse.

    mission_id MUST be set so compute_task_hash includes it — without it
    two concurrent missions with same-named HTML files (e.g. home.html)
    share the same dedup hash, causing the 2nd mission's await_inline
    to wait on a task owned by the 1st mission → 600 s timeout."""
    pid = placeholder["placeholder_id"]
    spec = {
        "title": f"image:{pid}",
        "description": f"Generate image for placeholder {pid}",
        "agent_type": "image",
        "kind": "image",
        "runner": "direct",
        "priority": 5,
        "mission_id": mission_id,
        "context": {
            "image_call": {
                "raw_dispatch": True,
                "prompt": prompt,
                "out_dir": out_dir,
                "width": int(placeholder.get("width") or 512),
                "height": int(placeholder.get("height") or 512),
                "quality_tier": "fast",
                "filename_hint": pid,
            },
        },
    }
    try:
        result = await _enqueue_beckman(spec, await_inline=True)
    except Exception as exc:
        logger.warning("image enqueue raised for %s: %s", pid, exc)
        return None
    if getattr(result, "status", "") != "completed":
        logger.info("image task for %s did not complete (status=%r)",
                    pid, getattr(result, "status", ""))
        return None
    payload = _parse_task_result(result)
    path = payload.get("path") or payload.get("content")
    if not (path and os.path.isfile(path)):
        logger.warning("image task for %s returned no usable path", pid)
        return None
    return path


def _rename_to_pid(src_path: str, assets_dir: str, placeholder_id: str) -> str:
    """Rename the timestamp-named PNG to a stable <pid>.png inside assets/.
    Single rename — no copy fallback unless rename fails."""
    os.makedirs(assets_dir, exist_ok=True)
    final = os.path.join(assets_dir, f"{placeholder_id}.png")
    if os.path.abspath(src_path) == os.path.abspath(final):
        return f"{placeholder_id}.png"
    try:
        if os.path.exists(final):
            os.remove(final)
        os.replace(src_path, final)
        return f"{placeholder_id}.png"
    except OSError as exc:
        logger.warning("rename failed for %s: %s", placeholder_id, exc)
        try:
            import shutil
            shutil.copyfile(src_path, final)
            return f"{placeholder_id}.png"
        except OSError:
            return ""


def _swap_src_in_tag(tag: str, new_src: str) -> str:
    """Replace src="..." inside a single <img> tag, preserving other attrs."""
    return re.sub(r'src\s*=\s*"[^"]*"', f'src="{new_src}"', tag, count=1,
                  flags=re.IGNORECASE)


def _rewrite_html_srcs(
    html_path: str, rewrites: dict[tuple[int, int], str],
) -> bool:
    if not rewrites:
        return False
    try:
        with open(html_path, encoding="utf-8") as fh:
            html = fh.read()
    except OSError:
        return False
    # Apply tail-first so earlier spans keep their offsets.
    ordered = sorted(rewrites.items(), key=lambda kv: kv[0][0], reverse=True)
    changed = False
    for (start, end), new_src in ordered:
        old_tag = html[start:end]
        new_tag = _swap_src_in_tag(old_tag, new_src)
        if new_tag != old_tag:
            html = html[:start] + new_tag + html[end:]
            changed = True
    if changed:
        tmp = html_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(html)
        os.replace(tmp, html_path)
    return changed


async def _fanout_and_rewrite(
    workspace_path: str,
    placeholders: list[dict[str, Any]],
    prompt_map: dict[str, str],
    mission_id: int,
) -> dict[str, Any]:
    """Sequential per-placeholder: image enqueue → rename to <pid>.png →
    record rewrite. Per-placeholder failures kept in errors; original
    placehold.co URL survives in HTML for that slot."""
    assets_dir = _assets_dir(workspace_path)
    rewrites_per_file: dict[str, dict[tuple[int, int], str]] = {}
    errors: list[str] = []
    replaced = 0
    skipped = 0

    for ph in placeholders:
        pid = ph["placeholder_id"]
        prompt = prompt_map.get(pid)
        if not prompt:
            skipped += 1
            errors.append(f"no prompt for {pid}")
            continue
        path = await _generate_one_image(
            placeholder=ph, prompt=prompt, out_dir=assets_dir,
            mission_id=mission_id,
        )
        if not path:
            skipped += 1
            errors.append(f"image gen failed for {pid}")
            continue
        final_name = _rename_to_pid(path, assets_dir, pid)
        if not final_name:
            skipped += 1
            errors.append(f"rename failed for {pid}")
            continue
        # Compute the rewritten src as the path FROM THIS HTML FILE'S OWN
        # DIRECTORY to the flat asset file, so a static file server resolves
        # it correctly from ANY subdir. Root HTML (.web/home.html) → the file
        # at .web/assets/<pid>.png is reached via "assets/<pid>.png"; a subdir
        # screen (.web/screens/onboarding.html) reaches it via
        # "../assets/<pid>.png". Browsers resolve relative <img src> against
        # the document's own location, so a flat "assets/<pid>.png" would 404
        # for subdir screens (resolved → .web/screens/assets/<pid>.png).
        asset_abs = os.path.join(assets_dir, final_name)
        html_dir = os.path.dirname(ph["html_path"])
        new_src = os.path.relpath(asset_abs, html_dir).replace(os.sep, "/")
        rewrites_per_file.setdefault(ph["html_path"], {})[
            tuple(ph["tag_span"])
        ] = new_src
        replaced += 1

    files_changed = 0
    for path, rewrites in rewrites_per_file.items():
        if _rewrite_html_srcs(path, rewrites):
            files_changed += 1

    return {
        "replaced": replaced, "skipped": skipped,
        "html_files_changed": files_changed, "errors": errors,
    }
