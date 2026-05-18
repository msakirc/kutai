"""Z4 T4A — visual-review founder-loop Telegram notification.

After ``visual_review`` runs, the founder must be able to:
  1. See the captured screenshots as a Telegram album (≤10 per album, one per route).
  2. Approve individual frames as baselines (per-breakpoint granularity).
  3. Calibrate severity ("this colour variation is fine").

Callback-data scheme
--------------------
``visrev:{mission_id}:{token}``
    Compact (~23 bytes) — the token resolves against the per-mission
    cbmap sidecar (``mission_{id}/.visual/.cbmap.json``) to an entry:
      - ``{kind: "approve", step_id, frame}`` — per-breakpoint baseline approval
      - ``{kind: "cal", verdict, lesson_pattern}`` — severity calibration

Notes
-----
- The token scheme keeps ``callback_data`` well under Telegram's 64-byte
  cap regardless of step_id / filename length, so no button is ever
  silently dropped.
- Thumbnails are WebP ≤80KB, max dimension 600px, written to
  ``mission_{id}/.visual/thumbs/{step_id}/``.
- Media-groups (albums) can't carry inline buttons — buttons go on a
  follow-up text message.
- Soft-skips when Telegram is not configured.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("mr_roboto.visual_review_notify")

# Max thumbnail size (longest edge) and target file size.
_THUMB_MAX_DIM = 600
_THUMB_MAX_BYTES = 80 * 1024  # 80 KB
_ALBUM_MAX = 10  # Telegram media-group limit


# ---------------------------------------------------------------------------
# Thumbnail compression
# ---------------------------------------------------------------------------

def _compress_to_webp(src_png: str, dst_webp: str) -> str:
    """Compress *src_png* to a WebP thumbnail ≤80 KB at *dst_webp*.

    Returns *dst_webp* on success.  Re-raises on import or write failure.
    """
    from PIL import Image  # type: ignore[import]

    img = Image.open(src_png).convert("RGB")

    # Resize to max dimension while preserving aspect ratio.
    w, h = img.size
    longest = max(w, h)
    if longest > _THUMB_MAX_DIM:
        scale = _THUMB_MAX_DIM / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Tune quality down until file is ≤80 KB.
    for quality in (80, 65, 50, 35, 20):
        img.save(dst_webp, format="WEBP", quality=quality)
        if os.path.getsize(dst_webp) <= _THUMB_MAX_BYTES:
            break

    return dst_webp


def _build_thumb_dir(workspace_path: str, mission_id: int, step_id: str) -> str:
    thumb_dir = os.path.join(
        workspace_path,
        f"mission_{mission_id}",
        ".visual",
        "thumbs",
        step_id,
    )
    os.makedirs(thumb_dir, exist_ok=True)
    return thumb_dir


def _make_thumbnails(
    captured_paths: list[str],
    workspace_path: str,
    mission_id: int,
    step_id: str,
) -> list[tuple[str, str]]:
    """Return [(thumb_path, original_basename), ...] for each captured frame.

    Silently skips frames where Pillow fails (no blocker on notification).
    """
    thumb_dir = _build_thumb_dir(workspace_path, mission_id, step_id)
    results: list[tuple[str, str]] = []
    for src in captured_paths:
        basename = os.path.basename(src)
        stem, _ = os.path.splitext(basename)
        dst = os.path.join(thumb_dir, f"{stem}.webp")
        try:
            _compress_to_webp(src, dst)
            results.append((dst, basename))
        except Exception as exc:
            logger.warning("visual_review_notify: thumb failed for %s: %s", src, exc)
    return results


# ---------------------------------------------------------------------------
# Callback-data helpers
# ---------------------------------------------------------------------------

def _cbmap_path(workspace_path: str, mission_id: int) -> str:
    """Path to the per-mission callback-token map.

    Telegram caps ``callback_data`` at 64 bytes — packing step_id + a
    frame filename (or a ``route:component:kind`` lesson pattern) blows
    that budget for late-phase steps with long IDs.  Instead each button
    gets a short token; the real fields live in this JSON sidecar.
    """
    return os.path.join(
        workspace_path, f"mission_{mission_id}", ".visual", ".cbmap.json"
    )


def _register_cb(workspace_path: str, mission_id: int, entry: dict) -> str:
    """Persist *entry* under a fresh token; return the compact callback_data.

    Returned form ``visrev:{mission_id}:{token}`` is ~23 bytes — always
    well under Telegram's 64-byte ceiling regardless of step_id length.
    """
    import json
    import secrets

    path = _cbmap_path(workspace_path, mission_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cbmap: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                cbmap = loaded
        except Exception as exc:  # noqa: BLE001
            logger.debug("visual_review_notify: cbmap read failed: %s", exc)
    token = secrets.token_hex(4)  # 8 hex chars
    cbmap[token] = entry
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cbmap, fh)
    os.replace(tmp, path)
    return f"visrev:{mission_id}:{token}"


# ---------------------------------------------------------------------------
# Inline keyboard builder
# ---------------------------------------------------------------------------

def _build_inline_keyboard(
    thumbs: list[tuple[str, str]],
    findings: list[dict[str, Any]],
    mission_id: int,
    step_id: str,
    workspace_path: str,
) -> list[list[dict[str, str]]]:
    """Build a list-of-rows for inline buttons.

    Each row = list of button dicts ``{label, callback_data}``.

    Layout:
    - One approve button per frame (≤8 frames → ≤8 buttons, 2 per row).
    - Two calibration buttons at the end.

    Every button's ``callback_data`` is a compact ``visrev:{mid}:{token}``
    string; the step_id / frame / lesson-pattern payload is persisted in
    the per-mission cbmap sidecar (see :func:`_register_cb`).
    """
    rows: list[list[dict[str, str]]] = []

    # Approve buttons (2 per row).
    approve_buttons: list[dict[str, str]] = []
    for _thumb_path, original_basename in thumbs:
        cb = _register_cb(
            workspace_path, mission_id,
            {"kind": "approve", "step_id": step_id, "frame": original_basename},
        )
        label = f"✅ {original_basename[:20]}"
        approve_buttons.append({"label": label, "callback_data": cb})

    for i in range(0, len(approve_buttons), 2):
        rows.append(approve_buttons[i: i + 2])

    # Calibration buttons — derive a pattern from the first finding if available.
    top_finding = findings[0] if findings else {}
    route = top_finding.get("route") or "all"
    component = top_finding.get("component") or "all"
    kind = top_finding.get("kind") or "other"
    lesson_pattern = f"{route}:{component}:{kind}"

    cb_fine = _register_cb(
        workspace_path, mission_id,
        {"kind": "cal", "verdict": "fine", "lesson_pattern": lesson_pattern},
    )
    cb_broken = _register_cb(
        workspace_path, mission_id,
        {"kind": "cal", "verdict": "broken", "lesson_pattern": lesson_pattern},
    )
    rows.append([
        {"label": "🟢 This is fine", "callback_data": cb_fine},
        {"label": "🔴 Genuinely broken", "callback_data": cb_broken},
    ])

    return rows


# ---------------------------------------------------------------------------
# Summary message
# ---------------------------------------------------------------------------

def _build_summary_text(
    mission_id: int,
    step_id: str,
    verdict: str,
    findings: list[dict[str, Any]],
) -> str:
    blockers = [f for f in findings if f.get("severity") == "blocker"]
    warnings = [f for f in findings if f.get("severity") == "warning"]
    infos = [f for f in findings if f.get("severity") == "info"]

    emoji = "🔴" if verdict == "fail" else "🟢"
    lines = [
        f"{emoji} *Visual review — mission #{mission_id} / step {step_id}*",
        "",
        f"Verdict: `{verdict}`  |  "
        f"blockers: {len(blockers)}  warnings: {len(warnings)}  info: {len(infos)}",
    ]
    if findings:
        lines.append("")
        lines.append("*Top findings:*")
        for f in findings[:5]:
            sev = f.get("severity", "?")
            sev_sym = {"blocker": "🔴", "warning": "🟡", "info": "ℹ️"}.get(sev, "·")
            desc = str(f.get("description") or "")[:80]
            comp = str(f.get("component") or "")
            bp = str(f.get("breakpoint") or "")
            lines.append(f"  {sev_sym} [{bp}] {comp}: {desc}")

    lines.append("")
    lines.append("_Tap a frame thumbnail to approve it as the new baseline._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

def _get_tg():
    """Return TelegramInterface or None."""
    try:
        from src.app.telegram_bot import get_telegram
        return get_telegram()
    except Exception:
        return None


async def _send_album(bot, chat_id: int, thumb_paths: list[str]) -> None:
    """Send ≤10 thumbnails as a Telegram media group (album)."""
    from telegram import InputMediaPhoto  # type: ignore[import]

    media = []
    for tp in thumb_paths[:_ALBUM_MAX]:
        try:
            with open(tp, "rb") as fh:
                media.append(InputMediaPhoto(fh.read()))
        except Exception as exc:
            logger.warning("visual_review_notify: cannot read thumb %s: %s", tp, exc)

    if not media:
        return

    try:
        await bot.send_media_group(chat_id=chat_id, media=media)
    except Exception as exc:
        logger.warning("visual_review_notify: send_media_group failed: %s", exc)


async def _send_text_with_buttons(
    bot,
    chat_id: int,
    text: str,
    rows: list[list[dict[str, str]]],
) -> None:
    """Send a text message with an inline keyboard."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup  # type: ignore[import]

    keyboard = []
    for row in rows:
        kb_row = []
        for btn in row:
            label = btn.get("label", "")
            cb = btn.get("callback_data", "")
            if label and cb:
                kb_row.append(InlineKeyboardButton(label, callback_data=cb))
        if kb_row:
            keyboard.append(kb_row)

    markup = InlineKeyboardMarkup(keyboard) if keyboard else None

    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
            reply_markup=markup,
        )
    except Exception as exc:
        # Fallback without markdown
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=text.replace("*", "").replace("`", "").replace("_", ""),
                reply_markup=markup,
            )
        except Exception as exc2:
            logger.warning("visual_review_notify: send text failed: %s | %s", exc, exc2)


# ---------------------------------------------------------------------------
# Public entry point (mirrors enqueue_similar_review_notice style)
# ---------------------------------------------------------------------------

async def enqueue_visual_review_notice(
    *,
    mission_id: int,
    step_id: str,
    verdict: str,
    findings: list[dict[str, Any]],
    captured_paths: list[str],
    workspace_path: str | None = None,
) -> None:
    """Fire-and-forget: compress thumbnails, send album + follow-up with buttons.

    Soft-skips when:
    - captured_paths is empty
    - Telegram is not configured / unreachable
    """
    if not captured_paths:
        logger.debug("visual_review_notify: no captured_paths — skipping notification")
        return

    tg = _get_tg()
    if tg is None:
        logger.debug("visual_review_notify: Telegram not configured — skipping")
        return

    # Resolve workspace_path
    if workspace_path is None:
        try:
            from src.tools.workspace import WORKSPACE_DIR
            workspace_path = WORKSPACE_DIR
        except Exception:
            workspace_path = os.getcwd()

    # 1. Build thumbnails.
    thumbs = _make_thumbnails(captured_paths, workspace_path, mission_id, step_id)
    if not thumbs:
        logger.warning("visual_review_notify: thumbnail generation failed for all frames")
        return

    # 2. Resolve chat_id (admin chat).
    try:
        from src.app.config import TELEGRAM_ADMIN_CHAT_ID
        chat_id = int(TELEGRAM_ADMIN_CHAT_ID)
    except Exception as exc:
        logger.warning("visual_review_notify: no admin chat_id: %s", exc)
        return

    bot = tg.app.bot

    # 3. Group frames by route — one album per route (8 frames max → 1 album).
    from collections import defaultdict
    by_route: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for thumb_path, orig_basename in thumbs:
        # Parse route from basename: {route}_{mode}_{breakpoint}.webp
        stem = os.path.splitext(orig_basename)[0]
        parts = stem.split("_")
        route = parts[0] if parts else "unknown"
        by_route[route].append((thumb_path, orig_basename))

    all_thumb_paths = [tp for tp, _ in thumbs]

    # 4. Send album(s) — one per route.
    for route, route_thumbs in by_route.items():
        await _send_album(bot, chat_id, [tp for tp, _ in route_thumbs])

    # 5. Build and send follow-up text + inline buttons.
    text = _build_summary_text(mission_id, step_id, verdict, findings)
    rows = _build_inline_keyboard(
        thumbs, findings, mission_id, step_id, workspace_path
    )
    await _send_text_with_buttons(bot, chat_id, text, rows)

    logger.info(
        "visual_review_notify: sent album (%d thumbs) + buttons for mission=%s step=%s",
        len(thumbs), mission_id, step_id,
    )
