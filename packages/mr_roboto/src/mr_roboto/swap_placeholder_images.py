"""Plan 3 — swap placehold.co <img> tags for real diffusion-generated PNGs.

CPS chain (SP5: migrated off the deleted ``await_inline`` 2026-06-11):
  1. KICKOFF (the 5.35 mechanic, ``swap_placeholder_images``): recursively
     walk ``mission_{id}/.web/**/*.html``, scan for placehold.co <img>, write
     a durable chain ledger to ``<ws>/.swap_state/swap_chain.json`` (OUTSIDE
     the served/published ``.web`` root — it carries prompts, absolute paths
     and exception strings), enqueue ONE
     prompt_writer child with ``on_complete``/``on_error`` continuations, and
     return immediately (``{ok, queued, chain: "started", ...}``). When there
     is nothing to swap it returns the old completed shape with
     ``chain: "none"``.
  2. ``prompts_done`` continuation: parses the placeholder_id -> prompt map
     from the child result, stores it in the ledger, enqueues the FIRST image
     child (sequential chain — one at a time, warm-batch friendly).
  3. ``image_done``/``image_err`` continuations: rename the PNG to a stable
     ``<pid>.png`` under ``.web/assets/`` (or record the per-placeholder
     error), then advance: next pending placeholder -> next image child;
     none left -> finalize.
  4. ``_finalize`` (plain function, not a continuation): applies ALL HTML
     <img src> rewrites in one pass from the recorded scan-time tag spans
     (valid because no rewrite happens before finalize), runs the deep shape
     check (no surviving placehold.co beyond the recorded skips; no broken
     relative asset refs), and writes the summary into the ledger.

Timing: a continuation fires when the child reaches a TRUE terminal status —
AFTER the child's constrained_emit/grade posthook chain — so ``prompts_done``
receives the POST-REPAIR prompt_writer result (strictly better than the old
``await_inline`` timing, which raced the posthook child).

Preview interaction: 5.40 (emit_preview_url) may surface the preview BEFORE
all images land — the preview serves disk files live, so swapped HTML simply
appears on refresh as the chain completes.

HTML rewrite semantics (unchanged): <img src> is rewritten to a relative ref
computed from EACH html file's own dir to the flat asset
(``os.path.relpath``): root HTML gets "assets/<id>.png", a subdir screen gets
"../assets/<id>.png". Graceful degrade: per-placeholder failure keeps the
placehold.co URL.

Mirrors marketing_copy.py's mechanical shape — internally enqueues LLM +
image work through beckman, never calls dispatcher/HK/paintress directly
(feedback_singular_dispatcher_caller). Reversibility: "full"."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.swap_placeholder_images")


# ── Continuation names (durable — referenced from the continuations table) ─

ON_PROMPTS_DONE = "mr_roboto.swap_images.prompts_done"
ON_PROMPTS_ERR = "mr_roboto.swap_images.prompts_err"
ON_IMAGE_DONE = "mr_roboto.swap_images.image_done"
ON_IMAGE_ERR = "mr_roboto.swap_images.image_err"


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


# ── Chain ledger (durable chain state on disk; cont_state stays SMALL) ─

# The ledger carries diffusion prompts, absolute filesystem paths and raw
# exception strings. It must live OUTSIDE the served .web root: .web/ is
# tunnel-served live and copytree'd to a PUBLIC gh-pages repo by
# publish_preview_pages.
LEDGER_DIRNAME = ".swap_state"
LEDGER_FILENAME = "swap_chain.json"


def _ledger_path(workspace_path: str) -> str:
    return os.path.join(workspace_path, LEDGER_DIRNAME, LEDGER_FILENAME)


def _load_ledger(workspace_path: str) -> dict | None:
    try:
        with open(_ledger_path(workspace_path), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_ledger(workspace_path: str, ledger: dict) -> None:
    path = _ledger_path(workspace_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ── Tolerant child-result parsing (CPS handlers receive plain dicts) ───

def _coerce_result_dict(result: Any) -> dict:
    """A continuation handler's ``result`` should arrive as a dict, but be
    defensive: a JSON-string body (restart-reconcile decodes only the TOP
    level; tests may fabricate strings) coerces via json.loads FIRST before
    any isinstance check on the decoded value. Adapted from the deleted
    ``_parse_task_result`` (TaskResult is gone — SP5)."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            decoded = json.loads(result)
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_prompts(result: Any) -> dict[str, str]:
    """Pull the placeholder_id -> prompt map out of a prompt_writer child
    result. Tolerates: top-level ``prompts``; nested under ``result`` (dict or
    JSON string); nested under ``content`` (dict or JSON string). Returns {}
    when no usable prompts exist (callers degrade)."""
    parsed = _coerce_result_dict(result)
    candidates: list[dict] = [parsed]
    for key in ("result", "content"):
        nested = _coerce_result_dict(parsed.get(key))
        if nested:
            candidates.append(nested)
    prompts = None
    for cand in candidates:
        if isinstance(cand.get("prompts"), list):
            prompts = cand["prompts"]
            break
    if prompts is None:
        return {}
    out: dict[str, str] = {}
    for entry in prompts:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("placeholder_id")
        prompt = entry.get("prompt")
        if isinstance(pid, str) and isinstance(prompt, str) and prompt.strip():
            out[pid] = prompt.strip()
    return out


def _extract_image_path(result: Any) -> str | None:
    """Pull the generated PNG path out of an image child result. The image
    lane returns ``{content, path, provider, ...}``; restart-reconcile may
    nest it under ``result``."""
    parsed = _coerce_result_dict(result)
    candidates: list[dict] = [parsed]
    nested = _coerce_result_dict(parsed.get("result"))
    if nested:
        candidates.append(nested)
    for cand in candidates:
        path = cand.get("path") or cand.get("content")
        if isinstance(path, str) and path.strip():
            return path
    return None


# ── beckman wrapper (test-patchable, mirrors marketing_copy) ───────────

async def _enqueue_beckman(spec: dict, **kwargs):
    from general_beckman import enqueue as _enqueue
    return await _enqueue(spec, **kwargs)


# ── Kickoff (the 5.35 mechanic) ────────────────────────────────────────

async def swap_placeholder_images(
    mission_id: int,
    workspace_path: str | None = None,
    design_tokens: dict | None = None,
    brand_voice: str | None = None,
    task_id: int | None = None,
) -> dict[str, Any]:
    """Kickoff: scan, write the chain ledger, enqueue the prompt_writer child
    with CPS continuations, return immediately.

    Returns (Action-compatible completed shapes):
      - nothing to swap:
        {ok: True, replaced_count: 0, skipped_count: 0, html_files_seen,
         html_files_changed: 0, errors: [], chain: "none"}
      - chain started:
        {ok: True, queued: True, chain: "started", placeholder_count,
         html_files_seen}
      - prompt_writer enqueue raised (graceful degrade — all placeholders keep
        their placehold.co URLs):
        the chain:"none" shape with skipped_count=N and the error recorded.
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
            "chain": "none",
        }

    all_placeholders: list[dict[str, Any]] = []
    for h in html_files:
        all_placeholders.extend(_scan_placeholders(h))

    if not all_placeholders:
        return {
            "ok": True, "replaced_count": 0, "skipped_count": 0,
            "html_files_seen": len(html_files), "html_files_changed": 0,
            "errors": [], "chain": "none",
        }

    # The ledger ON DISK is the chain state; cont_state stays SMALL
    # ({mission_id, workspace_path} + pid for image children).
    ledger: dict[str, Any] = {
        "mission_id": int(mission_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prompts_pending",
        "placeholders": [dict(p, tag_span=list(p["tag_span"]))
                         for p in all_placeholders],
        "prompt_map": {},
        "results": {},
    }
    _save_ledger(workspace_path, ledger)

    state = {"mission_id": int(mission_id), "workspace_path": workspace_path}
    spec = _build_prompt_writer_spec(
        mission_id=int(mission_id), placeholders=all_placeholders,
        design_tokens=design_tokens, brand_voice=brand_voice,
    )
    try:
        await _enqueue_beckman(
            spec,
            **({"parent_id": int(task_id)} if task_id is not None else {}),
            on_complete=ON_PROMPTS_DONE,
            on_error=ON_PROMPTS_ERR,
            cont_state=state,
        )
    except Exception as exc:
        logger.warning("prompt_writer enqueue raised: %s", exc)
        err = f"prompt_writer enqueue raised: {exc}"
        for ph in all_placeholders:
            ledger["results"][ph["placeholder_id"]] = {
                "status": "skipped", "error": err,
            }
        ledger["status"] = "done"
        ledger["errors"] = [err]
        _save_ledger(workspace_path, ledger)
        return {
            "ok": True, "replaced_count": 0,
            "skipped_count": len(all_placeholders),
            "html_files_seen": len(html_files), "html_files_changed": 0,
            "errors": [err], "chain": "none",
        }

    return {
        "ok": True, "queued": True, "chain": "started",
        "placeholder_count": len(all_placeholders),
        "html_files_seen": len(html_files),
    }


# ── Child spec builders ────────────────────────────────────────────────

def _build_prompt_writer_spec(
    *, mission_id: int,
    placeholders: list[dict[str, Any]],
    design_tokens: dict | None,
    brand_voice: str | None,
) -> dict:
    """ONE prompt_writer task spec (placeholder list -> prompt map).

    ``artifact_schema`` and ``is_workflow_step`` are set in the task context:
    - ``artifact_schema`` arms the constrained_emit POSTHOOK child (wired by
      ``general_beckman/posthooks.py::determine_posthooks``).
    - ``is_workflow_step`` triggers ``post_execute_workflow_step`` on the
      beckman apply path.

    The continuation fires only at TRUE terminal — AFTER the posthook chain —
    so ``prompts_done`` receives the POST-REPAIR result. Output still
    malformed after repair degrades gracefully: ``_extract_prompts`` returns
    {} → all placeholders are skipped with their placehold.co URLs intact."""
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

    return {
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
            # Arm the constrained-emit safety net: artifact_schema arms the
            # constrained_emit POSTHOOK child (beckman's determine_posthooks
            # reads it off the task context); is_workflow_step routes the
            # apply path through post_execute_workflow_step. Without the
            # schema the constrained re-emit never fires.
            "is_workflow_step": True,
            "artifact_schema": PROMPT_WRITER_ARTIFACT_SCHEMA,
        },
    }


def _build_image_spec(
    *, placeholder: dict[str, Any], prompt: str, out_dir: str, mission_id: int,
) -> dict:
    """One image task spec (context.image_call.raw_dispatch). Mirrors the
    /image cmd shape. Beckman's admission routes agent_type=image →
    fatih_hoca.select(needs_image=True) → paintress.

    mission_id MUST be set so compute_task_hash includes it — without it two
    concurrent missions with same-named HTML files (e.g. home.html) share the
    same dedup hash and the 2nd mission's child collapses onto the 1st's."""
    pid = placeholder["placeholder_id"]
    return {
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


# ── Continuation handlers ──────────────────────────────────────────────

async def _on_prompts_done(task_id: int, result: dict, state: dict) -> None:
    """Continuation: the prompt_writer child reached TRUE terminal (post
    constrained_emit/grade posthooks). Store the prompt map and start the
    sequential image chain; no usable prompts → finalize-with-degrade."""
    workspace_path = state.get("workspace_path") or ""
    ledger = _load_ledger(workspace_path)
    if ledger is None:
        logger.warning(
            "swap chain: prompts_done fired but ledger missing (ws=%s)",
            workspace_path,
        )
        return
    prompt_map = _extract_prompts(result)
    if not prompt_map:
        logger.warning(
            "swap chain: prompt_writer returned no usable prompt map "
            "(task_id=%s) — degrading, all placeholders skipped", task_id,
        )
        _degrade_all(ledger, "prompt_writer task did not return a usable prompt map")
        await _finalize(workspace_path, ledger)
        return
    ledger["prompt_map"] = prompt_map
    ledger["status"] = "images_pending"
    _save_ledger(workspace_path, ledger)
    await _advance(workspace_path, ledger, state)


async def _on_prompts_err(task_id: int, result: dict, state: dict) -> None:
    """on_error continuation: prompt_writer child failed (or its continuation
    TTL-expired). Whole chain degrades — every placeholder keeps its
    placehold.co URL."""
    workspace_path = state.get("workspace_path") or ""
    ledger = _load_ledger(workspace_path)
    if ledger is None:
        logger.warning(
            "swap chain: prompts_err fired but ledger missing (ws=%s)",
            workspace_path,
        )
        return
    err = str(_coerce_result_dict(result).get("error") or "prompt_writer task failed")
    logger.warning("swap chain: prompt_writer child failed: %s", err)
    _degrade_all(ledger, f"prompt_writer failed: {err}")
    await _finalize(workspace_path, ledger)


def _degrade_all(ledger: dict, error: str) -> None:
    for ph in ledger.get("placeholders") or []:
        ledger.setdefault("results", {}).setdefault(
            ph["placeholder_id"], {"status": "skipped", "error": error},
        )


async def _on_image_done(task_id: int, result: dict, state: dict) -> None:
    """Continuation: one image child finished. Rename to the stable
    ``<pid>.png``, record, advance the chain."""
    workspace_path = state.get("workspace_path") or ""
    pid = state.get("pid") or ""
    ledger = _load_ledger(workspace_path)
    if ledger is None:
        logger.warning(
            "swap chain: image_done fired but ledger missing (ws=%s, pid=%s)",
            workspace_path, pid,
        )
        return
    path = _extract_image_path(result)
    if path and os.path.isfile(path):
        final_name = _rename_to_pid(path, _assets_dir(workspace_path), pid)
        if final_name:
            entry: dict[str, Any] = {"status": "done", "asset": final_name}
        else:
            entry = {"status": "error", "error": f"rename failed for {pid}"}
    else:
        entry = {"status": "error",
                 "error": f"image task for {pid} returned no usable path"}
    ledger.setdefault("results", {})[pid] = entry
    _save_ledger(workspace_path, ledger)
    await _advance(workspace_path, ledger, state)


async def _on_image_err(task_id: int, result: dict, state: dict) -> None:
    """on_error continuation: one image child failed — record and advance.
    Graceful degrade: that placeholder keeps its placehold.co URL."""
    workspace_path = state.get("workspace_path") or ""
    pid = state.get("pid") or ""
    ledger = _load_ledger(workspace_path)
    if ledger is None:
        logger.warning(
            "swap chain: image_err fired but ledger missing (ws=%s, pid=%s)",
            workspace_path, pid,
        )
        return
    err = str(_coerce_result_dict(result).get("error") or "image task failed")
    logger.info("swap chain: image child for %s failed: %s", pid, err)
    ledger.setdefault("results", {})[pid] = {
        "status": "error", "error": f"image gen failed for {pid}: {err}",
    }
    _save_ledger(workspace_path, ledger)
    await _advance(workspace_path, ledger, state)


async def _advance(workspace_path: str, ledger: dict, state: dict) -> None:
    """Enqueue the next pending placeholder's image child (sequential — one
    at a time, warm-batch friendly), or finalize when none are left."""
    results = ledger.setdefault("results", {})
    prompt_map = ledger.get("prompt_map") or {}
    mission_id = int(ledger.get("mission_id") or state.get("mission_id") or 0)

    for ph in ledger.get("placeholders") or []:
        pid = ph["placeholder_id"]
        if pid in results:
            continue
        prompt = prompt_map.get(pid)
        if not prompt:
            results[pid] = {"status": "skipped", "error": f"no prompt for {pid}"}
            continue
        spec = _build_image_spec(
            placeholder=ph, prompt=prompt,
            out_dir=_assets_dir(workspace_path), mission_id=mission_id,
        )
        try:
            await _enqueue_beckman(
                spec,
                on_complete=ON_IMAGE_DONE,
                on_error=ON_IMAGE_ERR,
                cont_state={"mission_id": mission_id,
                            "workspace_path": workspace_path, "pid": pid},
            )
        except Exception as exc:
            logger.warning("image enqueue raised for %s: %s", pid, exc)
            results[pid] = {"status": "error",
                            "error": f"image enqueue raised for {pid}: {exc}"}
            continue
        ledger["status"] = "images_pending"
        _save_ledger(workspace_path, ledger)
        return

    await _finalize(workspace_path, ledger)


def register_continuations() -> None:
    """Register the swap-chain CPS handlers (idempotent; called at import +
    by general_beckman.continuations.register_startup_handlers)."""
    try:
        from general_beckman.continuations import register_resume
        register_resume(ON_PROMPTS_DONE, _on_prompts_done)
        register_resume(ON_PROMPTS_ERR, _on_prompts_err)
        register_resume(ON_IMAGE_DONE, _on_image_done)
        register_resume(ON_IMAGE_ERR, _on_image_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("swap chain continuation registration deferred: %s", exc)


# Register at import so the handlers are present for restart reconcile.
register_continuations()


# ── Rename + HTML rewrite helpers ──────────────────────────────────────

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


# ── Finalize (chain tail — plain function, not a continuation) ─────────

async def _finalize(workspace_path: str, ledger: dict) -> None:
    """Apply ALL HTML rewrites from the ledger in one pass, run the deep
    shape check, write the summary into the ledger.

    The rewrites use scan-time tag spans, which is safe because the HTML is
    never rewritten between scan and finalize — all rewrites happen exactly
    once, HERE. Per-placeholder failure keeps the placehold.co URL (graceful
    degrade). No Telegram notify."""
    assets_dir = _assets_dir(workspace_path)
    results = ledger.setdefault("results", {})
    rewrites_per_file: dict[str, dict[tuple[int, int], str]] = {}
    errors: list[str] = []
    replaced = 0
    skipped = 0

    for ph in ledger.get("placeholders") or []:
        pid = ph["placeholder_id"]
        entry = results.get(pid)
        if entry is None:
            entry = {"status": "skipped", "error": f"no result recorded for {pid}"}
            results[pid] = entry
        if entry.get("status") == "done" and entry.get("asset"):
            # Compute the rewritten src as the path FROM THIS HTML FILE'S OWN
            # DIRECTORY to the flat asset file, so a static file server
            # resolves it from ANY subdir: root HTML → "assets/<pid>.png",
            # subdir screen → "../assets/<pid>.png".
            asset_abs = os.path.join(assets_dir, entry["asset"])
            html_dir = os.path.dirname(ph["html_path"])
            new_src = os.path.relpath(asset_abs, html_dir).replace(os.sep, "/")
            rewrites_per_file.setdefault(ph["html_path"], {})[
                tuple(ph["tag_span"])
            ] = new_src
            replaced += 1
        else:
            skipped += 1
            if entry.get("error"):
                errors.append(str(entry["error"]))

    files_changed = 0
    for path, rewrites in rewrites_per_file.items():
        if _rewrite_html_srcs(path, rewrites):
            files_changed += 1

    # Deep shape check (lives HERE, not in 5.35.verify — the verify step may
    # run mid-flight): every surviving placehold.co must be accounted for by
    # a recorded skip/error, and no rewritten relative ref may point at a
    # missing file.
    shape_errors: list[str] = []
    surviving = 0
    broken_refs: list[str] = []
    try:
        from mr_roboto.verify_swap_placeholder_images_shape import (
            _scan_html, _walk_html,
        )
        surviving, broken_refs = _scan_html(_walk_html(workspace_path))
        if broken_refs:
            shape_errors.append(f"broken asset ref: {broken_refs[0]}")
        if surviving != skipped:
            shape_errors.append(
                f"inconsistent: surviving placeholders={surviving} but "
                f"skipped={skipped}"
            )
    except Exception as exc:  # noqa: BLE001
        shape_errors.append(f"shape check raised: {exc}")

    ledger["status"] = "done"
    ledger["replaced"] = replaced
    ledger["skipped"] = skipped
    ledger["errors"] = errors + shape_errors
    ledger["html_files_changed"] = files_changed
    ledger["shape_check"] = {
        "ok": not shape_errors,
        "surviving_placeholders": surviving,
        "broken_asset_refs": broken_refs,
    }
    _save_ledger(workspace_path, ledger)

    if skipped or errors or shape_errors:
        logger.warning(
            "swap chain finalized PARTIAL (ws=%s): replaced=%d skipped=%d "
            "errors=%s", workspace_path, replaced, skipped, errors + shape_errors,
        )
    else:
        logger.info(
            "swap chain finalized (ws=%s): replaced=%d, html_files_changed=%d",
            workspace_path, replaced, files_changed,
        )
