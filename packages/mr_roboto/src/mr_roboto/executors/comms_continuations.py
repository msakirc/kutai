"""SP4b Plan 3 — crisis/incident/press_kit CPS mechanical sinks.

MECHANICAL: no LLM call, no dispatcher import. Receive the already-produced
LLM output (*.resume) or fire on_error with the verb's canned fallback
(*.resume_err), then perform the founder-facing side-effect. Registered in
``general_beckman.continuations._HANDLER_MODULES`` so handlers survive restart.

Handler signature: async def handler(child_task_id: int, result: dict, state: dict) -> None
"""
from __future__ import annotations

from yazbunu import get_logger

logger = get_logger("mr_roboto.executors.comms_continuations")


def _extract_content(result: dict) -> str:
    """Dual-shape decode (normal terminal vs restart-reconcile)."""
    result = result or {}
    inner = result.get("result")
    if isinstance(inner, dict):
        content = inner.get("content", "")
    elif inner is not None:
        content = inner
    else:
        content = result.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


async def _emit_crisis_card(*, event_id, product_id, tier, variants):
    """Surface holding-statement variants to the founder. Never auto-posts."""
    try:
        from src.founder_actions import create as fa_create
        await fa_create(
            mission_id=None, kind="generic",
            title=f"Crisis holding statements ready (event #{event_id}, Tier {tier}) — pick one",
            why=("KutAI drafted holding-statement variants for the crisis. "
                 "NEVER auto-posted — select/edit and post manually."),
            instructions=[f"Variant {chr(65+i)}:\n\n{v}" for i, v in enumerate(variants)]
                         + ["Pick one, edit as needed, post manually."],
            expected_output_kind="ack_only", notify_telegram=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("crisis card emit failed: %s", exc)


async def _crisis_resume(child_task_id, result, state):
    from mr_roboto.crisis_draft_holding import parse_variants, canned_variants
    variants = parse_variants(_extract_content(result))
    if not variants:
        variants = canned_variants(int(state.get("tier") or 1), state.get("product_id") or "")
    await _emit_crisis_card(event_id=state.get("event_id"), product_id=state.get("product_id") or "",
                            tier=int(state.get("tier") or 1), variants=variants)


async def _crisis_resume_err(child_task_id, result, state):
    from mr_roboto.crisis_draft_holding import canned_variants
    logger.warning("crisis holding child failed (%s) — canned fallback", (result or {}).get("error"))
    variants = canned_variants(int(state.get("tier") or 1), state.get("product_id") or "")
    await _emit_crisis_card(event_id=state.get("event_id"), product_id=state.get("product_id") or "",
                            tier=int(state.get("tier") or 1), variants=variants)


async def _emit_incident_card(*, incident_id, product_id, draft):
    """Surface the status-update draft to the founder (the old incident_update_review
    gate, now inline in the sink — see general_beckman posthook_handlers/incident_update_review).
    NEVER auto-publishes."""
    try:
        from src.founder_actions import create as fa_create
        await fa_create(
            mission_id=None, kind="generic",
            title=f"Incident status update drafted (incident #{incident_id}) — review before publishing",
            why=("KutAI drafted a customer-facing status update. NEVER auto-published — "
                 "review/edit, then publish manually."),
            instructions=[f"Draft:\n\n{draft[:1000]}",
                          "Edit if needed, then publish manually.",
                          "NEVER publish automatically — draft only."],
            expected_output_kind="ack_only", notify_telegram=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("incident card emit failed: %s", exc)


async def _incident_resume(child_task_id, result, state):
    from mr_roboto.incident_draft_update import fallback_draft, finalize_redaction
    draft_raw = _extract_content(result).strip()
    if not draft_raw:
        draft_raw = fallback_draft(state.get("status_kind") or "investigating",
                                   state.get("affected_components") or [])
    draft = finalize_redaction(draft_raw)
    await _emit_incident_card(incident_id=state.get("incident_id"),
                              product_id=state.get("product_id") or "", draft=draft)


async def _incident_resume_err(child_task_id, result, state):
    from mr_roboto.incident_draft_update import fallback_draft, finalize_redaction
    logger.warning("incident update child failed (%s) — canned draft", (result or {}).get("error"))
    draft = finalize_redaction(fallback_draft(state.get("status_kind") or "investigating",
                                              state.get("affected_components") or []))
    await _emit_incident_card(incident_id=state.get("incident_id"),
                              product_id=state.get("product_id") or "", draft=draft)


async def _enqueue_press_kit_audience(*, audience, state):
    from src.comms.producers import _enqueue_press_kit_audience as _p
    return await _p(audience=audience, state=state)


async def _assemble_press_kit(*, mission_id, product_id, version, workspace_path,
                              spec_text, staged, source):
    from mr_roboto.press_kit_assemble import assemble_from_drafts
    await assemble_from_drafts(
        mission_id=mission_id, product_id=product_id, version=version,
        workspace_path=workspace_path, spec_text=spec_text, one_pagers=staged,
        logo_path=source.get("logo_path", ""),
        screenshot_paths=source.get("screenshot_paths", ()),
        founder_bio=source.get("founder_bio", ""),
        fact_sheet_md=source.get("fact_sheet_md", ""),
        quotes=source.get("quotes", ()),
        past_mentions=source.get("past_mentions", ()),
    )


async def _press_kit_advance(state, one_pager_text):
    staged = dict(state.get("staged") or {})
    staged[state["current"]] = one_pager_text
    remaining = list(state.get("remaining") or [])
    if remaining:
        nxt = remaining[0]
        new_state = dict(state)
        new_state["staged"] = staged
        new_state["remaining"] = remaining[1:]
        new_state["current"] = nxt
        await _enqueue_press_kit_audience(audience=nxt, state=new_state)
    else:
        await _assemble_press_kit(
            mission_id=state.get("mission_id"), product_id=state.get("product_id"),
            version=state.get("version"), workspace_path=state.get("workspace_path") or "",
            spec_text=state.get("spec_text") or "", staged=staged,
            source=state.get("source") or {},
        )


async def _press_kit_resume(child_task_id, result, state):
    text = _extract_content(result).strip()
    if not text:
        from mr_roboto.press_kit_assemble import audience_stub
        text = audience_stub(state["current"], state.get("spec_text") or "")
    await _press_kit_advance(state, text)


async def _press_kit_resume_err(child_task_id, result, state):
    from mr_roboto.press_kit_assemble import audience_stub
    logger.warning("press_kit %s child failed (%s) — stub", state.get("current"),
                   (result or {}).get("error"))
    await _press_kit_advance(state, audience_stub(state["current"], state.get("spec_text") or ""))


def register_continuations() -> None:
    """Register Plan-3 comms CPS sinks. Idempotent."""
    try:
        from general_beckman.continuations import register_resume
        register_resume("comms.crisis_holding.resume", _crisis_resume)
        register_resume("comms.crisis_holding.resume_err", _crisis_resume_err)
        register_resume("comms.incident_update.resume", _incident_resume)
        register_resume("comms.incident_update.resume_err", _incident_resume_err)
        register_resume("comms.press_kit.resume", _press_kit_resume)
        register_resume("comms.press_kit.resume_err", _press_kit_resume_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("comms continuation registration deferred: %s", exc)


register_continuations()
