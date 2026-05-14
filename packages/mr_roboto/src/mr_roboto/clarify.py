"""Mechanical clarify executor: sends clarification prompt via Telegram."""
from __future__ import annotations

import logging

from src.infra.db import update_task
from src.app.telegram_bot import get_telegram

logger = logging.getLogger(__name__)


async def send_variant_keyboard(
    mission_id: int,
    task_id: int,
    chat_id: int | None,
    base_label: str,
    options: list[dict],
) -> bool:
    """Send the variant-choice inline keyboard via Telegram.

    Previously a stub (logged only, no keyboard sent) — shopping missions
    hit this, emitted a clarify_choice artifact that downstream synth_one
    skipped against, and the mission wrapped up with "Sonuç bulunamadı"
    because the user never saw the buttons. Now delegates to the
    existing TelegramInterface.send_variant_keyboard implementation
    which renders an InlineKeyboardMarkup + registers the pending
    variant_choice state for the callback handler.

    Returns True if the keyboard was sent, False if Telegram is not
    available or chat_id is missing (caller should fall back to a plain
    compare-all view in that case).
    """
    if not chat_id:
        logger.warning(
            "send_variant_keyboard: no chat_id for mission=%s task=%s — skipping",
            mission_id, task_id,
        )
        return False
    try:
        tg = get_telegram()
    except Exception as exc:
        logger.warning("send_variant_keyboard: Telegram unavailable: %s", exc)
        return False
    if tg is None:
        return False
    try:
        await tg.send_variant_keyboard(
            chat_id=int(chat_id),
            mission_id=mission_id,
            task_id=task_id,
            base_label=base_label,
            options=options,
        )
        return True
    except Exception as exc:
        logger.exception(
            "send_variant_keyboard failed for mission=%s task=%s: %s",
            mission_id, task_id, exc,
        )
        return False


async def clarify(task: dict) -> dict:
    payload = task.get("payload") or {}
    kind = payload.get("kind")

    # Z1 Tier 5A (A5) — founder attention budget gate.
    # When the mission has a budget set AND remaining < reserve_minutes
    # (default 5), defer this clarify to deferred_questions.md instead of
    # firing on Telegram. Budget is unset by default, so existing missions
    # are unaffected.
    if not bool(payload.get("attention_skip", False)):
        try:
            mission_id = task.get("mission_id")
            if mission_id is not None:
                from mr_roboto.attention_check import (
                    attention_check, write_deferred_question,
                )
                reserve = int(payload.get("attention_reserve_minutes", 5))
                check = await attention_check(
                    mission_id=int(mission_id),
                    reserve_minutes=reserve,
                )
                if not check.get("ok"):
                    qtxt = (
                        payload.get("question")
                        or payload.get("kind")
                        or task.get("title")
                        or ""
                    )
                    step_id = (task.get("context") or {}).get(
                        "workflow_step_id", str(task.get("id"))
                    )
                    deferred = await write_deferred_question(
                        mission_id=int(mission_id),
                        step_id=str(step_id),
                        question_text=str(qtxt),
                    )
                    return {
                        "status": "deferred",
                        "reason": "attention_budget_exhausted",
                        "remaining": check.get("remaining"),
                        "deferred_path": deferred.get("path"),
                    }
        except Exception as exc:
            # Never block clarify on a budget-check error — log and proceed.
            logger.warning("clarify: attention_check failed: %s", exc)

    if kind == "variant_choice":
        payload_from = payload.get("payload_from", "gate_result")
        # Load the source artifact (gate_result) from the store. Task
        # dispatch path didn't populate task["artifacts"] — it only
        # carries the payload. Without this, base_label + options were
        # always empty, making the keyboard (now wired) still useless.
        mission_id = task.get("mission_id")
        source: dict = {}
        if mission_id is not None:
            # Primary: ArtifactStore (cache-first, blackboard fallback).
            try:
                from src.workflows.engine.artifacts import get_artifact_store
                import json as _json
                store = get_artifact_store()
                raw = await store.retrieve(mission_id, payload_from)
                if isinstance(raw, str) and raw.strip():
                    source = _json.loads(raw)
                elif isinstance(raw, dict):
                    source = raw
            except Exception as exc:
                logger.warning(
                    "clarify variant_choice: artifact-store lookup failed for %r: %s",
                    payload_from, exc,
                )
            # Fallback: read the blackboard directly. ArtifactStore can
            # miss under certain timing (cache not yet populated + store
            # instance divergence across coroutines). Mission 49 proved
            # this path silently returned empty source → empty
            # base_label → the keyboard prompt was the generic "Hangi
            # model?" instead of "Xiaomi 15T ... için hangi model?".
            if not source:
                try:
                    from src.collaboration.blackboard import read_blackboard
                    artifacts = await read_blackboard(int(mission_id), "artifacts")
                    if isinstance(artifacts, dict):
                        raw2 = artifacts.get(payload_from)
                        if isinstance(raw2, str) and raw2.strip():
                            import json as _json
                            source = _json.loads(raw2)
                        elif isinstance(raw2, dict):
                            source = raw2
                except Exception as exc:
                    logger.warning(
                        "clarify variant_choice: blackboard fallback failed for %r: %s",
                        payload_from, exc,
                    )
        if not source:
            logger.warning(
                "clarify variant_choice: no source artifact found for mission=%s key=%r",
                mission_id, payload_from,
            )
        base_label = source.get("base_label", "")
        options = source.get("clarify_options") or []
        # chat_id lives in the blackboard's artifact bag (set by the
        # Telegram handler that spawned the shopping mission), NOT in
        # missions.context. Mission 49 proved this: keyboard_sent=false
        # because my earlier code only looked at missions.context and
        # got None. Fall back to task.chat_id if blackboard isn't
        # reachable for any reason.
        chat_id = None
        if mission_id is not None:
            try:
                from src.collaboration.blackboard import read_blackboard
                artifacts = await read_blackboard(mission_id, "artifacts")
                if isinstance(artifacts, dict):
                    chat_id = artifacts.get("chat_id")
            except Exception as exc:
                logger.debug("clarify chat_id (blackboard) lookup failed: %s", exc)
        if chat_id is None:
            # Secondary: task row may carry chat_id (orchestrator
            # injects it on standalone /task flows).
            chat_id = task.get("chat_id")
        if chat_id is None:
            # Tertiary: some seed paths still use missions.context.
            try:
                from src.infra.db import get_db as _get_db
                _db = await _get_db()
                _cur = await _db.execute(
                    "SELECT context FROM missions WHERE id = ?", (mission_id,),
                )
                _row = await _cur.fetchone()
                await _cur.close()
                if _row and _row[0]:
                    import json as _json2
                    _mctx = _json2.loads(_row[0])
                    if isinstance(_mctx, str):
                        _mctx = _json2.loads(_mctx)
                    chat_id = (_mctx or {}).get("chat_id")
            except Exception as exc:
                logger.debug("clarify chat_id (missions) lookup failed: %s", exc)

        sent = await send_variant_keyboard(
            mission_id,
            task["id"],
            chat_id,
            base_label,
            options,
        )
        if sent:
            await update_task(task["id"], status="waiting_human")
        return {
            "status": "needs_clarification",
            "kind": "variant_choice",
            "prompt": f"{base_label} için hangi model?" if base_label else "Hangi model?",
            "keyboard_sent": sent,
        }

    # Artifact-confirm shape: inline the file content + send inline
    # keyboard with OK / Regenerate / Edit buttons. Founder never has
    # to open the file. Triggered by `attach_file_paths` in payload.
    attach_paths = payload.get("attach_file_paths") or []
    if attach_paths:
        from src.tools.workspace import WORKSPACE_DIR
        import os.path as _osp
        mission_id_v = task.get("mission_id")
        # Resolve chat_id (mirror variant_choice fallback chain).
        chat_id = None
        if mission_id_v is not None:
            try:
                from src.collaboration.blackboard import read_blackboard
                arts = await read_blackboard(int(mission_id_v), "artifacts")
                if isinstance(arts, dict):
                    chat_id = arts.get("chat_id")
            except Exception as exc:
                logger.debug("clarify(attach) chat_id (blackboard) lookup failed: %s", exc)
        if chat_id is None:
            chat_id = task.get("chat_id")
        if chat_id is None and mission_id_v is not None:
            try:
                from src.infra.db import get_db as _get_db
                import json as _json2
                _db = await _get_db()
                _cur = await _db.execute(
                    "SELECT context FROM missions WHERE id = ?", (mission_id_v,),
                )
                _row = await _cur.fetchone()
                await _cur.close()
                if _row and _row[0]:
                    _mctx = _json2.loads(_row[0])
                    if isinstance(_mctx, str):
                        _mctx = _json2.loads(_mctx)
                    chat_id = (_mctx or {}).get("chat_id")
            except Exception as exc:
                logger.debug("clarify(attach) chat_id (missions) lookup failed: %s", exc)

        contents: list[tuple[str, str]] = []
        for rel in attach_paths:
            if not isinstance(rel, str) or not rel.strip():
                continue
            abs_p = rel if _osp.isabs(rel) else _osp.join(WORKSPACE_DIR, rel)
            try:
                with open(abs_p, encoding="utf-8") as fh:
                    contents.append((rel, fh.read()))
            except OSError as exc:
                logger.warning("clarify(attach): read failed for %s: %s", abs_p, exc)
                contents.append((rel, f"(could not read file: {exc})"))

        question_text = payload.get("question") or "Confirm the draft below."
        kind_tag = str(payload.get("kind") or "artifact_confirm")
        regen_step = payload.get("regenerate_step_id") or ""
        source_id = task.get("parent_task_id") or task["id"]

        sent = False
        if chat_id is not None:
            try:
                tg = get_telegram()
                if tg is not None:
                    await tg.send_artifact_confirm_keyboard(
                        chat_id=int(chat_id),
                        mission_id=int(mission_id_v) if mission_id_v else 0,
                        task_id=int(source_id),
                        kind=kind_tag,
                        question=str(question_text),
                        files=contents,
                        regenerate_step_id=str(regen_step),
                    )
                    sent = True
            except Exception as exc:
                logger.exception("clarify(attach): send_artifact_confirm_keyboard failed: %s", exc)
        if sent:
            await update_task(int(source_id), status="waiting_human")
        return {
            "status": "needs_clarification",
            "kind": kind_tag,
            "attach_file_paths": attach_paths,
            "keyboard_sent": sent,
            "prompt": question_text,
        }

    # Default: plain question clarify
    question = payload.get("question")
    if not question:
        raise ValueError("clarify payload requires 'question'")
    # Register the SOURCE (blocked LLM) task with Telegram, not this
    # mechanical executor row. apply._apply_clarify set the source to
    # waiting_human and spawned us as its child (parent_task_id=source).
    # Using task["id"] (mechanical) caused the user's reply to miss its
    # target: orchestrator overwrote the mechanical row back to
    # status=completed on return, leaving no waiting_human match for
    # the reply handler (observed 2026-04-23: user's "C" answer
    # rerouted to the generic LLM classifier). parent_task_id falls
    # back to task["id"] when absent — safe for test fixtures that
    # don't model the full spawn graph.
    source_id = task.get("parent_task_id") or task["id"]
    tg = get_telegram()
    await tg.request_clarification(source_id, task.get("title", ""), question)
    return {"sent": True, "question": question, "source_task_id": source_id}
