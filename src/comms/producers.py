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
