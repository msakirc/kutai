"""SP4b Plan 3 — crisis/incident/press_kit CPS producers (LLM out of mr_roboto).

Each function builds the verb-specific prompt + a raw_dispatch OVERHEAD spec and
enqueues it as an admitted Beckman task with a durable continuation
(on_complete -> mechanical sink). NO await_inline. Prompts live HERE.
"""
from __future__ import annotations

import time
import uuid

from general_beckman import enqueue  # module-level for test patching
from general_beckman.lanes import LANE_ONESHOT
from src.infra.logging_config import get_logger

logger = get_logger("comms.producers")


def _suffix() -> str:
    return f"{time.monotonic_ns() % 1_000_000:06d}-{uuid.uuid4().hex[:6]}"


def _overhead_spec(title: str, description: str, prompt: str,
                   in_tok: int, out_tok: int) -> dict:
    return {
        "title": title,
        "description": description,
        "agent_type": "reviewer",
        "kind": "overhead",
        "priority": 2,
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "reviewer",
            "agent_type": "reviewer",
            "difficulty": 3,
            "messages": [{"role": "user", "content": prompt}],
            "failures": [],
            "estimated_input_tokens": in_tok,
            "estimated_output_tokens": out_tok,
        }},
    }

_CRISIS_TIER_LABELS = {1: "brand misstep / pile-on", 2: "outage / data issue",
                       3: "security incident / breach", 4: "existential / legal"}


async def enqueue_crisis_holding(*, event_id: int, product_id: str, tier: int,
                                 summary: str, playbook_excerpt: str = "") -> int | None:
    """Enqueue the crisis holding-statement producer; comms.crisis_holding.resume
    parses + emits the founder card. Returns task id (or None)."""
    tier = int(tier or 1)
    label = _CRISIS_TIER_LABELS.get(tier, f"tier {tier}")
    excerpt = (playbook_excerpt or "")[:1200] or "(playbook not available)"
    prompt = (
        f"You are drafting holding statements for a Tier {tier} crisis ({label}).\n\n"
        f"Crisis summary: {summary or 'Crisis event opened.'}\n\n"
        f"Playbook context (holding statement shape section):\n{excerpt}\n\n"
        "Produce EXACTLY 2 holding-statement variants:\n"
        "- Variant A: More formal / corporate tone.\n"
        "- Variant B: More human / conversational tone.\n\n"
        "Rules:\n- Each variant: 2-4 sentences. Under 280 characters.\n"
        "- No internal hostnames, team names, or technical jargon.\n"
        "- No speculation about root cause.\n"
        "- Tier 3+: do NOT confirm 'breach' — use 'incident' or 'security event'.\n"
        "- Tier 4: extremely minimal — confirm aware and investigating; NO details.\n\n"
        'Return ONLY a JSON array of 2 strings: ["Variant A text", "Variant B text"]'
    )
    spec = _overhead_spec(f"crisis_holding:llm:{_suffix()}",
                          f"Draft Tier {tier} crisis holding variants.", prompt, 500, 200)
    return await enqueue(
        spec, lane=LANE_ONESHOT,
        on_complete="comms.crisis_holding.resume",
        on_error="comms.crisis_holding.resume_err",
        cont_state={"event_id": event_id, "product_id": product_id, "tier": tier,
                    "summary": summary, "playbook_excerpt": excerpt},
    )


async def enqueue_incident_update(*, incident_id: int, product_id: str, status_kind: str,
                                  severity: str, affected_components: list,
                                  safe_alert_details: dict, existing_summary: str) -> int | None:
    """Enqueue the incident status-update producer; comms.incident_update.resume
    applies the final redaction pass + surfaces the draft. Inputs already redacted."""
    import json as _json
    components_str = ", ".join(affected_components) if affected_components else "the service"
    safe_details_str = _json.dumps(safe_alert_details, ensure_ascii=False)[:800]
    prompt = (
        "You are drafting a public-facing status page update for customers.\n"
        f"Incident severity: {severity}\n"
        f"Affected components: {components_str}\n"
        f"Status kind: {status_kind} (investigating|identified|monitoring|resolved)\n"
        f"Current summary: {existing_summary or 'none'}\n"
        f"Internal alert details (already redacted, for context only):\n{safe_details_str}\n\n"
        "Write 2-4 clear, calm sentences suitable for customers.\n"
        "Rules:\n- Do NOT mention internal hostnames, IPs, stack traces, or team names.\n"
        "- Do NOT include customer PII.\n- Use plain language — no jargon.\n"
        "- Acknowledge the impact, state what you know, give next-update ETA.\n"
        "Draft only — no sign-off or signature needed."
    )
    spec = _overhead_spec(f"incident_update:llm:{_suffix()}",
                          "Draft customer-facing status update.", prompt, 400, 200)
    return await enqueue(
        spec, lane=LANE_ONESHOT,
        on_complete="comms.incident_update.resume",
        on_error="comms.incident_update.resume_err",
        cont_state={"incident_id": incident_id, "product_id": product_id,
                    "status_kind": status_kind, "affected_components": affected_components},
    )


# Per-audience one-pager instructions (moved out of mr_roboto's verb).
_AUDIENCE_INSTR: dict = {
    "investor": (
        "Write a concise investor-facing one-pager. "
        "Emphasise: traction metrics, market size, unit economics, "
        "team credentials, fundraising context. Omit culture fluff."
    ),
    "journalist": (
        "Write a journalist-facing one-pager. "
        "Lead with the news hook (what changed, why now), include "
        "3-5 concrete stats, founder quote, and a clear narrative arc. "
        "Avoid marketing jargon."
    ),
    "partner": (
        "Write a partner/integration-focused one-pager. "
        "Highlight: tech stack, API surface, customer overlap, "
        "joint integration opportunity, and go-to-market potential. "
        "Concrete and actionable."
    ),
    "candidate": (
        "Write a candidate-facing one-pager (recruiting). "
        "Highlight: mission and why it matters, team culture, "
        "growth trajectory, open roles, and why this is a compelling "
        "place to work. Warm but not hyperbolic."
    ),
}
AUDIENCE_ORDER = ("investor", "journalist", "partner", "candidate")


async def enqueue_press_kit(*, product_id: str, mission_id: int, version: int,
                            workspace_path: str, spec_text: str, source: dict) -> int | None:
    """Kick off the serial press-kit producer chain (first audience). The
    comms.press_kit.resume sink stages each draft + enqueues the next audience,
    and assembles after the last one."""
    state = {
        "product_id": product_id, "mission_id": mission_id, "version": version,
        "workspace_path": workspace_path, "spec_text": spec_text,
        "remaining": list(AUDIENCE_ORDER[1:]), "current": AUDIENCE_ORDER[0],
        "staged": {}, "source": dict(source or {}),
    }
    return await _enqueue_press_kit_audience(audience=AUDIENCE_ORDER[0], state=state)


async def _enqueue_press_kit_audience(*, audience: str, state: dict) -> int | None:
    instr = _AUDIENCE_INSTR.get(audience, "Write a one-pager.")
    prompt = (f"{instr}\n\nProduct spec:\n{state.get('spec_text') or ''}\n\n"
              "Output: Markdown prose, 200-400 words, no JSON wrapper.")
    spec = _overhead_spec(f"press_kit:{audience}:llm:{_suffix()}",
                          f"Draft {audience} one-pager.", prompt, 600, 500)
    st = dict(state); st["current"] = audience
    return await enqueue(spec, lane=LANE_ONESHOT,
                         on_complete="comms.press_kit.resume",
                         on_error="comms.press_kit.resume_err", cont_state=st)
