"""Z7 T3E — B6: crisis/draft_holding mr_roboto verb.

LLM-bound: reads the tier-specific playbook + event context, outputs
2+ holding-statement variants for founder selection.

The draft is returned as variants — it is NOT posted anywhere automatically.
Founder selects and edits via founder_action card.

Payload::

    {
        "product_id": "prod-abc",     # required
        "event_id":   42,             # required
        "tier":       2,              # 1-4, required
        "summary":    "...",          # optional; fetched from crisis_events if omitted
    }

Returns::

    {
        "status":   "ok",
        "variants": ["...", "...", ...],
        "tier":     2,
        "event_id": 42,
    }
"""
from __future__ import annotations

import os
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.crisis_draft_holding")


def _playbook_path(tier: int) -> str:
    """Return the absolute path to the crisis_comms_tier{N}.md playbook.

    Resolves relative to the project root using WORKSPACE_DIR or CWD fallback.
    """
    # Try WORKSPACE_DIR env var first (set in production)
    workspace_dir = os.environ.get("WORKSPACE_DIR", "")
    if workspace_dir:
        # WORKSPACE_DIR is the missions/ workspace subdir; go up one level to repo root
        repo_root = os.path.normpath(os.path.join(workspace_dir, ".."))
        candidate = os.path.join(repo_root, "playbooks", f"crisis_comms_tier{tier}.md")
        if os.path.isfile(candidate):
            return candidate

    # Fallback: traverse up from this file
    # packages/mr_roboto/src/mr_roboto/__file__ → up 4 dirs → repo root
    here = os.path.dirname(os.path.abspath(__file__))
    # mr_roboto/ → src/ → mr_roboto (pkg) → packages/ → repo_root
    repo_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
    candidate = os.path.join(repo_root, "playbooks", f"crisis_comms_tier{tier}.md")
    if os.path.isfile(candidate):
        return candidate

    # Last resort: current working directory (test / dev context)
    cwd_candidate = os.path.join(os.getcwd(), "playbooks", f"crisis_comms_tier{tier}.md")
    return cwd_candidate


def _read_playbook(tier: int) -> str:
    """Read the playbook for *tier*. Returns empty string on failure."""
    path = _playbook_path(tier)
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except Exception as exc:
        logger.warning(
            "crisis_draft_holding: could not read playbook",
            tier=tier,
            path=path,
            error=str(exc),
        )
        return ""


async def _call_llm_draft(
    tier: int,
    summary: str,
    playbook_text: str,
) -> list[str]:
    """Call LLM (OVERHEAD lane via beckman.enqueue) to produce holding variants.

    Returns a list of 2+ holding-statement strings.
    """
    from general_beckman import enqueue
    from general_beckman.lanes import LANE_OVERHEAD
    import asyncio
    import json

    playbook_excerpt = playbook_text[:1200] if playbook_text else "(playbook not available)"

    tier_labels = {
        1: "brand misstep / pile-on",
        2: "outage / data issue",
        3: "security incident / breach",
        4: "existential / legal",
    }
    tier_label = tier_labels.get(tier, f"tier {tier}")

    prompt = (
        f"You are drafting holding statements for a Tier {tier} crisis ({tier_label}).\n\n"
        f"Crisis summary: {summary}\n\n"
        f"Playbook context (holding statement shape section):\n{playbook_excerpt}\n\n"
        f"Produce EXACTLY 2 holding-statement variants:\n"
        f"- Variant A: More formal / corporate tone.\n"
        f"- Variant B: More human / conversational tone.\n\n"
        f"Rules:\n"
        f"- Each variant: 2-4 sentences. Under 280 characters for tweet-native delivery.\n"
        f"- No internal hostnames, team names, or technical jargon.\n"
        f"- No speculation about root cause.\n"
        f"- Acknowledge the situation; state what is known; give next-update ETA if Tier 2+.\n"
        f"- Tier 3+: Do NOT confirm 'breach' — use 'incident' or 'security event'.\n"
        f"- Tier 4: Extremely minimal — confirm you are aware and investigating; NO details.\n\n"
        f"Return ONLY a JSON array of 2 strings: [\"Variant A text\", \"Variant B text\"]"
    )

    result_holder: list[Any] = []
    done_event = asyncio.Event()

    async def _on_finish(task_result: dict) -> None:
        result_holder.append(task_result.get("output") or task_result.get("result") or "")
        done_event.set()

    await enqueue(
        {
            "title": "crisis_draft_holding:llm",
            "description": f"Draft Tier {tier} crisis holding statement variants.",
            "agent_type": "assistant",
            "kind": "overhead",
            "context": {
                "prompt": prompt,
                "_callback": _on_finish,
            },
        },
        lane=LANE_OVERHEAD,
    )

    try:
        await asyncio.wait_for(done_event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("crisis_draft_holding: LLM timed out")
        return []

    raw = result_holder[0] if result_holder else ""
    # Parse JSON array from LLM output
    import re
    import json as _json

    # Try to extract JSON array from the response
    raw_str = str(raw)
    m = re.search(r'\[.*?\]', raw_str, re.DOTALL)
    if m:
        try:
            parsed = _json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except Exception:
            pass

    # Fallback: split on obvious variant markers
    variants = []
    for line in raw_str.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("Variant"):
            if len(line) > 20:
                variants.append(line)
    return variants[:2] if variants else []


# ---------------------------------------------------------------------------
# Public run() — called by mr_roboto._run_dispatch
# ---------------------------------------------------------------------------

async def run(payload: dict) -> dict:
    """Execute crisis/draft_holding.

    Returns holding-statement variants for founder selection.
    """
    product_id = payload.get("product_id") or ""
    event_id = payload.get("event_id")
    tier = int(payload.get("tier") or 1)
    summary = payload.get("summary") or ""

    if not event_id:
        return {"status": "error", "error": "event_id is required"}
    if not product_id:
        return {"status": "error", "error": "product_id is required"}

    # Fetch summary from DB if not provided
    if not summary:
        try:
            from src.infra.db import get_db
            db = await get_db()
            async with db.execute(
                "SELECT summary, tier FROM crisis_events "
                "WHERE event_id=? AND product_id=?",
                (event_id, product_id),
            ) as cur:
                row = await cur.fetchone()
            if row:
                summary = row[0] or ""
                if not tier:
                    tier = int(row[1] or 1)
        except Exception as exc:
            logger.warning("crisis_draft_holding: could not fetch event", error=str(exc))

    # Read tier-specific playbook
    playbook_text = _read_playbook(tier)

    # Get LLM variants
    variants: list[str] = []
    try:
        variants = await _call_llm_draft(
            tier=tier,
            summary=summary or "Crisis event opened.",
            playbook_text=playbook_text,
        )
    except Exception as exc:
        logger.warning("crisis_draft_holding: LLM draft failed", error=str(exc))

    # Fallback variants when LLM is unavailable
    if not variants:
        tier_labels = {
            1: "this situation",
            2: "an ongoing service issue",
            3: "a security incident",
            4: "a critical situation",
        }
        context = tier_labels.get(tier, "this issue")
        variants = [
            (
                f"We are aware of {context} affecting {product_id}. "
                "Our team is actively working on a resolution. "
                "We will provide an update shortly."
            ),
            (
                f"We know about {context} and are on it. "
                "Thank you for your patience — more details coming soon."
            ),
        ]

    logger.info(
        "crisis_draft_holding: variants ready",
        event_id=event_id,
        product_id=product_id,
        tier=tier,
        variant_count=len(variants),
    )

    return {
        "status": "ok",
        "variants": variants,
        "tier": tier,
        "event_id": event_id,
        "product_id": product_id,
    }
