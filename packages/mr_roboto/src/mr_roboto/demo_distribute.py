"""Z7 T6D — ``demo/distribute`` mr_roboto verb.

Distributes the three demo cuts (30s / 60s / 3min) produced by ``demo/edit``:

  1. Uploads each cut to YouTube as **unlisted** via Data API v3.
     The founder flips to public via a ``founder_action`` — never auto-public.
  2. Extracts a thumbnail still per cut via a single ffmpeg frame grab.
  3. Generates an ``og:video`` meta tag snippet for the product homepage.
  4. Writes ``demo/distribute_result.json`` to workspace.
  5. Emits a ``founder_action`` "review demo cuts → flip to public?".

YouTube client
--------------
``_youtube_upload`` uses ``googleapiclient`` (``google-api-python-client``)
when available.  If the library is **not installed**, ``_youtube_upload``
raises ``RuntimeError("youtube client not installed: ...")``.  This keeps
the dependency pluggable — callers mock ``_youtube_upload`` in tests and
the verb degrades gracefully to ``ok=False`` with a clear message.

Public-flip gate
----------------
``distribute`` only ever uploads as ``unlisted``.  After all uploads
complete, it emits a single ``founder_action`` card: "review demo cuts →
flip to public?".  The flip itself is a separate action
(``demo/distribute/flip_to_public``) that requires explicit founder approval.
``distribute`` never sets ``privacy="public"``.

ffmpeg invocation
-----------------
``_extract_thumbnail`` follows the same subprocess style as ``demo_edit.py``:
uses ``asyncio.create_subprocess_exec`` via a shared ``_run_subprocess``
helper so tests can monkeypatch it.

Reversibility: ``partial``
  — Unlisted YouTube video can be deleted or edited within reason, but the
    upload itself has a real-world side-effect.  ``flip_to_public`` is
    classified ``irreversible`` (see ``reversibility.py``).
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.demo_distribute")

# ---------------------------------------------------------------------------
# ffmpeg subprocess helper (same style as demo_edit.py)
# ---------------------------------------------------------------------------


async def _run_subprocess(cmd: list[str], timeout: float = 300.0) -> tuple[int, str, str]:
    """Run a subprocess; return (rc, stdout, stderr)."""
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
    return (
        proc.returncode or 0,
        (out or b"").decode("utf-8", "replace"),
        (err or b"").decode("utf-8", "replace"),
    )


# ---------------------------------------------------------------------------
# Thumbnail extraction
# ---------------------------------------------------------------------------


async def _extract_thumbnail(cut_path: str, out_path: str, *, at_second: float = 2.0) -> bool:
    """Extract a single frame from ``cut_path`` at ``at_second`` into ``out_path`` (PNG).

    Returns ``True`` on success, ``False`` on failure.
    Tests monkeypatch ``_run_subprocess``.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(at_second),
        "-i", cut_path,
        "-frames:v", "1",
        "-q:v", "2",
        out_path,
    ]
    rc, _stdout, stderr = await _run_subprocess(cmd, timeout=30.0)
    if rc != 0 or not os.path.exists(out_path):
        logger.warning(
            "demo_distribute: thumbnail extraction failed",
            cut_path=cut_path,
            rc=rc,
            stderr=stderr[:200],
        )
        return False
    return True


# ---------------------------------------------------------------------------
# YouTube upload — pluggable
# ---------------------------------------------------------------------------


def _youtube_upload_real(
    path: str,
    title: str,
    description: str,
    privacy: str,
) -> dict[str, Any]:
    """Upload ``path`` to YouTube using google-api-python-client.

    Raises ``RuntimeError`` if the library is not installed.
    """
    try:
        # Lazy import so the module loads even without the library.
        from googleapiclient.discovery import build  # type: ignore[import]
        from googleapiclient.http import MediaFileUpload  # type: ignore[import]
        import google.auth  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "youtube client not installed: pip install google-api-python-client "
            "google-auth google-auth-httplib2 google-auth-oauthlib"
        ) from exc

    # Build service using application default credentials.
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/youtube.upload"]
    )
    service = build("youtube", "v3", credentials=credentials)

    body = {
        "snippet": {
            "title": title,
            "description": description,
        },
        "status": {
            "privacyStatus": privacy,
        },
    }
    media = MediaFileUpload(path, chunksize=-1, resumable=True)
    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )
    response = request.execute()
    video_id = response["id"]
    return {
        "video_id": video_id,
        "embed_url": f"https://www.youtube.com/embed/{video_id}",
        "watch_url": f"https://www.youtube.com/watch?v={video_id}",
        "privacy": privacy,
    }


# Public name: tests/callers monkeypatch THIS name.
_youtube_upload = _youtube_upload_real


# ---------------------------------------------------------------------------
# og:video snippet generator
# ---------------------------------------------------------------------------


def _build_og_video_snippet(
    uploads: dict[str, dict[str, Any]],
    product_name: str,
) -> str:
    """Build an og:video meta tag block for the product homepage.

    Uses the 30s cut as the primary embed, falls back to 60s then 3min.
    """
    primary_label = next(
        (lbl for lbl in ("30s", "60s", "3min") if lbl in uploads),
        None,
    )
    if not primary_label:
        return ""

    primary = uploads[primary_label]
    embed_url = primary.get("embed_url", "")
    lines = [
        f'<meta property="og:video" content="{embed_url}" />',
        f'<meta property="og:video:type" content="text/html" />',
        f'<meta property="og:video:width" content="1280" />',
        f'<meta property="og:video:height" content="720" />',
        f'<!-- Additional cut lengths: {", ".join(lbl for lbl in uploads if lbl != primary_label)} -->',
    ]
    # Add per-cut embed references as HTML comments for developer convenience.
    for lbl, info in uploads.items():
        if lbl != primary_label:
            lines.append(
                f'<!-- {lbl}: <meta property="og:video" content="{info.get("embed_url", "")}" /> -->'
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Founder-action emitter
# ---------------------------------------------------------------------------


async def _emit_flip_to_public_action(
    *,
    mission_id: int,
    product_name: str,
    uploads: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Surface a founder_action 'review demo cuts → flip to public?'."""
    try:
        from src.founder_actions import create as fa_create

        video_links = "\n".join(
            f"  - {lbl}: {info.get('watch_url', info.get('embed_url', ''))}"
            for lbl, info in uploads.items()
        )
        instructions = [
            f"Review the {len(uploads)} uploaded demo cut(s) for '{product_name}' (all unlisted).",
            "Watch each cut and confirm it's ready for public release.",
            "To flip to public, run demo/distribute/flip_to_public with the video_ids from the result.",
            f"YouTube unlisted links:\n{video_links}",
        ]
        return await fa_create(
            mission_id=mission_id,
            kind="generic",
            title=f"Review demo cuts → flip to public? ({product_name})",
            why=(
                f"Demo cuts for '{product_name}' have been uploaded as unlisted. "
                "Public release requires founder approval."
            ),
            instructions=instructions,
            expected_output_kind="ack_only",
            notify_telegram=False,
        )
    except Exception as exc:
        logger.warning("demo_distribute: _emit_flip_to_public_action failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

# Cut priority for thumbnail selection (30s is the primary marketing asset).
_THUMBNAIL_OFFSET: dict[str, float] = {
    "30s": 2.0,
    "60s": 3.0,
    "3min": 5.0,
}


async def run(
    mission_id: int,
    workspace_path: str,
    cuts: dict[str, str],
    product_name: str,
    *,
    description: str = "",
) -> dict[str, Any]:
    """Upload demo cuts to YouTube (unlisted), extract thumbnails, generate og:video.

    Parameters
    ----------
    mission_id:
        Mission identifier (used for logging + founder_action).
    workspace_path:
        Root workspace directory (used for thumbnail output + result JSON).
    cuts:
        ``{label: path}`` dict as returned by ``demo/edit``.
        Expected labels: ``"30s"``, ``"60s"``, ``"3min"``.
    product_name:
        Human-readable product name (used in YouTube title + og meta).
    description:
        Optional description for YouTube videos.

    Returns::

        {
            "ok": True,
            "uploads": {
                "30s": {"video_id": str, "embed_url": str, "watch_url": str,
                        "privacy": "unlisted", "thumbnail_path": str | None},
                ...
            },
            "og_video_snippet": str,          # HTML <meta> block
            "flip_to_public_action_id": int,  # founder_action.id
        }
        {"ok": False, "error": str}
    """
    if not cuts:
        return {"ok": False, "error": "cuts is empty; nothing to distribute"}

    thumbnails_dir = os.path.join(workspace_path, "demo", "thumbnails")
    os.makedirs(thumbnails_dir, exist_ok=True)

    uploads: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for label, cut_path in cuts.items():
        # --- Upload to YouTube as unlisted ---
        title = f"{product_name} — Demo ({label})"
        try:
            # _youtube_upload is a synchronous (blocking) network upload —
            # run it off the event loop so the demo pipeline doesn't stall.
            upload_info = await asyncio.to_thread(
                _youtube_upload,
                cut_path,
                title,
                description or f"Demo video for {product_name} ({label} cut).",
                "unlisted",
            )
        except Exception as exc:
            error_msg = str(exc)
            logger.error(
                "demo_distribute: YouTube upload failed",
                label=label,
                error=error_msg,
            )
            errors.append(f"{label}: {error_msg[:200]}")
            continue

        # --- Extract thumbnail ---
        thumb_out = os.path.join(thumbnails_dir, f"{label}.png")
        thumb_ok = await _extract_thumbnail(
            cut_path,
            thumb_out,
            at_second=_THUMBNAIL_OFFSET.get(label, 2.0),
        )
        upload_info["thumbnail_path"] = thumb_out if thumb_ok else None

        uploads[label] = upload_info
        logger.info(
            "demo_distribute: cut uploaded",
            label=label,
            video_id=upload_info.get("video_id"),
        )

    if errors:
        return {
            "ok": False,
            "error": f"upload failed: {'; '.join(errors)}",
            "uploads": uploads,
        }

    # --- og:video snippet ---
    og_snippet = _build_og_video_snippet(uploads, product_name)

    # --- Emit founder_action for the public-flip decision ---
    fa = await _emit_flip_to_public_action(
        mission_id=mission_id,
        product_name=product_name,
        uploads=uploads,
    )
    flip_action_id = (fa or {}).get("id") if fa else None

    # --- Persist result JSON ---
    result_data = {
        "uploads": uploads,
        "og_video_snippet": og_snippet,
        "flip_to_public_action_id": flip_action_id,
    }
    result_path = os.path.join(workspace_path, "demo", "distribute_result.json")
    try:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
    except Exception as exc:
        logger.warning("demo_distribute: failed to write result JSON: %s", exc)

    logger.info(
        "demo_distribute: distribution complete",
        mission_id=mission_id,
        cuts=list(uploads.keys()),
        flip_action_id=flip_action_id,
    )

    return {
        "ok": True,
        "uploads": uploads,
        "og_video_snippet": og_snippet,
        "flip_to_public_action_id": flip_action_id,
    }
