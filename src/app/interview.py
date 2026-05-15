"""Z7 T4 B7 — Customer interview / call notes pipeline.

Pipeline:
  1. Transcribe — Whisper on CPU (pluggable: openai-whisper or faster-whisper).
     Raises RuntimeError with clear message when neither is installed.
  2. Summarize — LLM-bound (OVERHEAD lane via beckman.enqueue).
     Structured output: bullets per topic, verbatim quotes, insights, action items.
  3. Cross-link — non-LLM:
     - Appends summary to A10 interactions (kind='interview').
     - Enqueues action items as candidate backlog tasks.
     - Pushes quotes to press_kit_quotes ONLY when crm.has_consent(product_id,
       contact_id, 'quote_use') is True; otherwise emits a founder_action
       requesting quote consent.
  4. A0 briefing surface — emits founder_action "review interview note" after
     the pipeline completes.

Public API (used by Telegram /interview command + mr_roboto verbs):
  _transcribe(audio_path, model_size='small') -> str
  transcribe_interview(note_id, product_id) -> dict
  summarize_interview(note_id, product_id) -> dict
  cross_link_interview(note_id, product_id, contact_id, *, mission_id=None) -> dict
  emit_review_founder_action(note_id, product_id, mission_id) -> dict

DB: interview_notes (see infra/db.py migration 2026-05-16-z7-interview-notes)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("app.interview")

_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"

DEFAULT_MODEL_SIZE = "small"


# ---------------------------------------------------------------------------
# Transcription (pluggable Whisper)
# ---------------------------------------------------------------------------


def _transcribe(audio_path: str, model_size: str = DEFAULT_MODEL_SIZE) -> str:
    """Transcribe an audio file to Markdown text using Whisper (CPU).

    Attempts:
      1. openai-whisper (``import whisper``)
      2. faster-whisper (``from faster_whisper import WhisperModel``)

    Raises RuntimeError with a clear "whisper not installed" message when
    neither backend is available.  Tests should mock this function or inject
    a sys.modules stub — no real audio decode is needed in tests.

    Parameters
    ----------
    audio_path:
        Absolute path to the audio file (.mp3, .wav, .ogg, .m4a, etc.)
    model_size:
        Whisper model size string (``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, ``"large"``). Defaults to ``"small"`` for accuracy /
        speed balance on CPU. Override via env ``WHISPER_MODEL_SIZE``.

    Returns
    -------
    str
        Raw transcript as plain text.
    """
    import importlib
    import sys

    # 1. Try openai-whisper
    whisper_mod = sys.modules.get("whisper")
    if whisper_mod is None:
        # Not pre-loaded — try a real import
        try:
            whisper_mod = importlib.import_module("whisper")
        except ImportError:
            whisper_mod = None

    if whisper_mod is not None:
        model = whisper_mod.load_model(model_size)
        result = model.transcribe(audio_path)
        return result.get("text", "")

    # 2. Try faster-whisper
    fw_mod = sys.modules.get("faster_whisper")
    if fw_mod is None:
        try:
            fw_mod = importlib.import_module("faster_whisper")
        except ImportError:
            fw_mod = None

    if fw_mod is not None:
        model = fw_mod.WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(audio_path)
        return "".join(seg.text for seg in segments)

    # Neither backend available
    raise RuntimeError(
        "whisper not installed: install openai-whisper (`pip install openai-whisper`) "
        "or faster-whisper (`pip install faster-whisper`) to enable interview transcription."
    )


# ---------------------------------------------------------------------------
# Beckman shim (lazy import; mocked in tests)
# ---------------------------------------------------------------------------


async def beckman_enqueue(spec: dict, **kw):
    """Thin wrapper around general_beckman.enqueue — importable for mocking."""
    from general_beckman import enqueue
    return await enqueue(spec, **kw)


# ---------------------------------------------------------------------------
# Step 1: Transcribe (mr_roboto verb: interview/transcribe)
# ---------------------------------------------------------------------------


async def transcribe_interview(
    note_id: int,
    product_id: str,
    *,
    model_size: str | None = None,
) -> dict:
    """Run Whisper transcription on the audio_path stored in interview_notes.

    Reads audio_path from the DB, calls _transcribe(), writes transcript_md
    back. Returns ``{"ok": True, "note_id": note_id, "transcript_length": N}``.
    """
    import os
    from src.infra.db import get_db

    size = model_size or os.getenv("WHISPER_MODEL_SIZE", DEFAULT_MODEL_SIZE)

    db = await get_db()
    cur = await db.execute(
        "SELECT audio_path FROM interview_notes WHERE note_id=? AND product_id=?",
        (note_id, product_id),
    )
    row = await cur.fetchone()
    if row is None:
        return {"ok": False, "error": f"interview_notes row not found: note_id={note_id}"}

    audio_path = row[0]
    if not audio_path:
        return {"ok": False, "error": "audio_path is empty; cannot transcribe"}

    try:
        transcript = _transcribe(audio_path, model_size=size)
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}

    await db.execute(
        "UPDATE interview_notes SET transcript_md=? WHERE note_id=?",
        (transcript, note_id),
    )
    await db.commit()
    logger.info("interview: transcribed", note_id=note_id, length=len(transcript))
    return {"ok": True, "note_id": note_id, "transcript_length": len(transcript)}


# ---------------------------------------------------------------------------
# Step 2: Summarize (mr_roboto verb: interview/summarize — OVERHEAD lane)
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = """\
You are an expert at extracting structured insights from customer interview transcripts.

Given the following transcript, produce a JSON object with exactly these keys:
- "bullets": list[str] — 3-10 bullet points summarising key topics discussed
- "quotes": list[str] — verbatim quotes from the interviewee (exact words)
- "insights": str — founder-level interpretation (2-4 sentences, first-person)
- "action_items": list[str] — concrete follow-up tasks arising from the interview

Respond ONLY with valid JSON. No preamble, no markdown fences.

Transcript:
{transcript}
"""


async def summarize_interview(note_id: int, product_id: str) -> dict:
    """LLM-bound (OVERHEAD lane): read transcript, write structured summary to DB.

    Returns ``{"ok": True/False, ...}``.
    """
    from src.infra.db import get_db

    db = await get_db()
    cur = await db.execute(
        "SELECT transcript_md FROM interview_notes WHERE note_id=? AND product_id=?",
        (note_id, product_id),
    )
    row = await cur.fetchone()
    if row is None:
        return {"ok": False, "error": f"note not found: note_id={note_id}"}
    transcript = row[0] or ""
    if not transcript.strip():
        return {"ok": False, "error": "transcript_md is empty; run transcribe first"}

    prompt = _SUMMARIZE_PROMPT.format(transcript=transcript)

    # Enqueue to Beckman (OVERHEAD lane) — inline for simplicity
    task_result = await beckman_enqueue(
        {
            "title": f"Interview summary note_id={note_id}",
            "agent_type": "summarizer",
            "kind": "overhead",
            "context": json.dumps({"prompt": prompt, "note_id": note_id}),
        },
        await_inline=True,
        lane="overhead",
    )

    # Parse structured output
    output_str = getattr(task_result, "output", None) or ""
    try:
        structured: dict[str, Any] = json.loads(output_str)
    except (json.JSONDecodeError, TypeError):
        # Try to find JSON in the output
        import re
        match = re.search(r"\{.*\}", output_str, re.DOTALL)
        if match:
            try:
                structured = json.loads(match.group())
            except json.JSONDecodeError:
                structured = {}
        else:
            structured = {}

    bullets = structured.get("bullets") or []
    quotes = structured.get("quotes") or []
    insights = structured.get("insights") or ""
    action_items = structured.get("action_items") or []

    # Compose summary_md from bullets
    summary_md = "## Interview Summary\n\n"
    if bullets:
        summary_md += "\n".join(f"- {b}" for b in bullets)

    await db.execute(
        "UPDATE interview_notes "
        "SET summary_md=?, quotes_json=?, insights_md=?, action_items_json=? "
        "WHERE note_id=?",
        (
            summary_md,
            json.dumps(quotes),
            insights,
            json.dumps(action_items),
            note_id,
        ),
    )
    await db.commit()
    logger.info(
        "interview: summarized",
        note_id=note_id,
        bullets=len(bullets),
        quotes=len(quotes),
        action_items=len(action_items),
    )
    return {
        "ok": True,
        "note_id": note_id,
        "bullets": len(bullets),
        "quotes": len(quotes),
        "action_items": len(action_items),
    }


# ---------------------------------------------------------------------------
# Step 3: Cross-link (mr_roboto verb: interview/cross_link — non-LLM)
# ---------------------------------------------------------------------------


async def cross_link_interview(
    note_id: int,
    product_id: str,
    contact_id: int,
    *,
    mission_id: int | None = None,
) -> dict:
    """Non-LLM: populate A10 interactions, backlog tasks, and A4 press_kit_quotes.

    Actions:
      a. Write an interactions row (kind='interview') via crm.log_interaction.
      b. Enqueue each action item as a candidate backlog task via beckman.enqueue.
      c. For each quote:
         - If crm.has_consent(product_id, contact_id, 'quote_use') → insert
           into press_kit_quotes.
         - Else → emit a founder_action requesting quote consent.
    """
    from src.infra.db import get_db
    from src.app import crm

    db = await get_db()
    cur = await db.execute(
        "SELECT summary_md, quotes_json, action_items_json "
        "FROM interview_notes WHERE note_id=? AND product_id=?",
        (note_id, product_id),
    )
    row = await cur.fetchone()
    if row is None:
        return {"ok": False, "error": f"note not found: note_id={note_id}"}

    summary_md = row[0] or "(no summary)"
    quotes: list[str] = json.loads(row[1] or "[]")
    action_items: list[str] = json.loads(row[2] or "[]")

    # a. Log CRM interaction
    interaction_id = await crm.log_interaction(
        product_id,
        contact_id,
        "interview",
        summary_md[:1000],
        next_action="Review interview insights",
        mission_id=mission_id,
    )

    # b. Enqueue action items as candidate backlog tasks
    for item in action_items:
        try:
            await beckman_enqueue(
                {
                    "title": f"[Interview follow-up] {item}",
                    "agent_type": "planner",
                    "kind": "main_work",
                    "context": json.dumps({
                        "source": "interview",
                        "note_id": note_id,
                        "product_id": product_id,
                    }),
                },
            )
        except Exception as _enq_exc:
            logger.warning(
                "interview: failed to enqueue action item",
                item=item,
                error=str(_enq_exc),
            )

    # c. Quote consent gate
    quotes_pushed = 0
    now_str = datetime.now(timezone.utc).strftime(_DATETIME_FMT)

    # Fetch contact display_name for founder_action title
    contact = await crm.get_contact_by_id(contact_id)
    contact_name = (contact or {}).get("display_name") or f"contact#{contact_id}"

    for quote_body in quotes:
        if not quote_body:
            continue
        has_consent = await crm.has_consent(product_id, contact_id, "quote_use")
        if has_consent:
            await db.execute(
                "INSERT INTO press_kit_quotes "
                "(product_id, kit_id, source_kind, speaker, body, approved, created_at) "
                "VALUES (?, NULL, 'interview', ?, ?, 0, ?)",
                (product_id, contact_name, quote_body, now_str),
            )
            quotes_pushed += 1
        else:
            # Emit a founder_action requesting consent
            try:
                from src.founder_actions import create as fa_create
                await fa_create(
                    mission_id=mission_id or 0,
                    kind="generic",
                    title=f"Request quote consent from {contact_name}",
                    why=(
                        f"Interview note #{note_id} contains a quote from {contact_name} "
                        f"that could go to the press kit, but quote_use consent is not "
                        "on file. Request consent before publishing."
                    ),
                    instructions=[
                        f"Contact {contact_name} and obtain written consent for quote use.",
                        "Once granted, run: crm/grant_consent "
                        f"product_id={product_id} contact_id={contact_id} purpose=quote_use",
                        "Then re-run interview/cross_link to push the approved quotes.",
                    ],
                )
            except Exception as _fa_exc:
                logger.warning(
                    "interview: failed to emit consent founder_action",
                    error=str(_fa_exc),
                )

    await db.commit()
    logger.info(
        "interview: cross-linked",
        note_id=note_id,
        interaction_id=interaction_id,
        action_items=len(action_items),
        quotes_pushed=quotes_pushed,
    )
    return {
        "ok": True,
        "note_id": note_id,
        "interaction_id": interaction_id,
        "action_items_enqueued": len(action_items),
        "quotes_pushed": quotes_pushed,
    }


# ---------------------------------------------------------------------------
# A0 briefing surface: "review interview note" founder_action
# ---------------------------------------------------------------------------


async def emit_review_founder_action(
    note_id: int,
    product_id: str,
    mission_id: int,
) -> dict:
    """Emit a founder_action card: 'review interview note' after pipeline completes.

    The founder should edit insights and approve/deny quote consent requests.
    """
    from src.founder_actions import create as fa_create

    try:
        fa = await fa_create(
            mission_id=mission_id,
            kind="generic",
            title=f"Review interview note #{note_id}",
            why=(
                f"Interview pipeline completed for note #{note_id} "
                f"(product={product_id}). Review the AI-generated summary, "
                "edit insights as needed, and approve quote consent requests."
            ),
            instructions=[
                f"Open interview note #{note_id} in the KutAI interface.",
                "Review the bullet summary and edit insights if needed.",
                "Decide which quotes may go to the press kit (grant quote_use consent).",
                "Check action items added to the backlog.",
            ],
        )
        return {"ok": True, "founder_action_id": fa.id if fa else None}
    except Exception as exc:
        logger.warning("interview: failed to emit review founder_action", error=str(exc))
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers for Telegram /interview command
# ---------------------------------------------------------------------------


async def start_interview(
    product_id: str,
    handle: str,
    audio_path: str | None = None,
) -> dict:
    """Create an interview_notes row for the given contact handle.

    Looks up the contact by handle; if not found, auto-creates with category='customer'.
    Returns the new note_id and contact_id.
    """
    from src.infra.db import get_db
    from src.app import crm

    contact = await crm.get_contact_by_handle(product_id, handle)
    if contact is None:
        contact_id = await crm.add_contact(
            product_id,
            handle,
            handle,
            category="customer",
        )
    else:
        contact_id = contact["contact_id"]

    db = await get_db()
    now_str = datetime.now(timezone.utc).strftime(_DATETIME_FMT)
    cur = await db.execute(
        "INSERT INTO interview_notes "
        "(product_id, contact_id, started_at, audio_path) "
        "VALUES (?, ?, ?, ?)",
        (product_id, contact_id, now_str, audio_path or ""),
    )
    await db.commit()
    note_id = cur.lastrowid
    logger.info("interview: started", note_id=note_id, handle=handle, product_id=product_id)
    return {"ok": True, "note_id": note_id, "contact_id": contact_id}


async def stop_interview(
    note_id: int,
    product_id: str,
    *,
    audio_path: str | None = None,
    duration_minutes: float | None = None,
) -> dict:
    """Mark interview as stopped. Optionally set audio_path + duration.

    Does NOT run the pipeline — pipeline steps are dispatched separately.
    """
    from src.infra.db import get_db

    db = await get_db()
    updates = []
    params: list = []
    if audio_path is not None:
        updates.append("audio_path=?")
        params.append(audio_path)
    if duration_minutes is not None:
        updates.append("duration_minutes=?")
        params.append(duration_minutes)

    if updates:
        params.append(note_id)
        await db.execute(
            f"UPDATE interview_notes SET {', '.join(updates)} WHERE note_id=?",
            params,
        )
        await db.commit()

    logger.info("interview: stopped", note_id=note_id)
    return {"ok": True, "note_id": note_id}


async def list_interviews(
    product_id: str,
    handle: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List interview_notes rows, optionally filtered by contact handle."""
    from src.infra.db import get_db

    db = await get_db()
    if handle:
        cur = await db.execute(
            "SELECT n.note_id, n.product_id, n.contact_id, n.started_at, "
            "n.duration_minutes, n.audio_path, r.handle, r.display_name "
            "FROM interview_notes n "
            "LEFT JOIN relationships r ON r.contact_id = n.contact_id "
            "WHERE n.product_id=? AND r.handle=? "
            "ORDER BY n.started_at DESC LIMIT ?",
            (product_id, handle, limit),
        )
    else:
        cur = await db.execute(
            "SELECT n.note_id, n.product_id, n.contact_id, n.started_at, "
            "n.duration_minutes, n.audio_path, r.handle, r.display_name "
            "FROM interview_notes n "
            "LEFT JOIN relationships r ON r.contact_id = n.contact_id "
            "WHERE n.product_id=? "
            "ORDER BY n.started_at DESC LIMIT ?",
            (product_id, limit),
        )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in await cur.fetchall()]
