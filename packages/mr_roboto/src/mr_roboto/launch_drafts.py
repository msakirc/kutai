"""Z7 T3A (A2) — Launch channel draft verbs.

Per-channel draft verbs: ``launch_drafts/<channel>`` for each of:
  hn, ph, twitter, linkedin, reddit

Each verb is LLM-bound. It enqueues a Beckman task (OVERHEAD lane)
that ingests:
  - ``spec``         — product spec text
  - ``brand_voice``  — brand voice document
  - ``mission_lessons`` — prior launch lessons (cross-mission signal)

and produces a channel-appropriate draft for founder review.

Supported channels
------------------
- ``hn``       — Hacker News (Show HN: … style, technical, community first)
- ``ph``       — Product Hunt (tagline + key features, community-engaging)
- ``twitter``  — Twitter/X (short thread, founder-voice, launch energy)
- ``linkedin`` — LinkedIn (professional tone, founder story, milestone)
- ``reddit``   — Reddit (community-specific subreddit, no hard sell)

Usage (mr_roboto dispatch)
--------------------------
``payload["action"] = "launch_drafts/hn"``
``payload = {"product_id": "prod-abc", "launch_id": 1, "spec": "...",
             "brand_voice": "...", "mission_lessons": [...]}``

Returns ``{"status": "enqueued", "task_id": int}`` on success.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.launch_drafts")

# Supported channels and their system prompt context
CHANNEL_CONTEXTS: dict[str, str] = {
    "hn": (
        "You are drafting a Show HN: post for Hacker News. "
        "Tone: technical, honest, community-first. No marketing fluff. "
        "Max 300 words. Format: 'Show HN: <one-line title>\\n\\n<body>'."
    ),
    "ph": (
        "You are drafting a Product Hunt launch post. "
        "Tone: energetic, benefit-focused, community-engaging. "
        "Include: tagline (≤60 chars), key features (3-5 bullets), maker comment."
    ),
    "twitter": (
        "You are drafting a Twitter/X launch thread for a founder. "
        "Tone: authentic founder voice, launch energy, personal. "
        "First tweet ≤280 chars. Thread can be 3-5 tweets."
    ),
    "linkedin": (
        "You are drafting a LinkedIn launch post for a founder. "
        "Tone: professional, milestone-focused, story-driven. "
        "Max 400 words. Include a clear CTA."
    ),
    "reddit": (
        "You are drafting a Reddit launch post. "
        "Tone: community-native, no hard sell, transparent about what it is. "
        "Read the subreddit rules first (provided in spec). Max 500 words."
    ),
}


async def _enqueue(spec: dict, **kwargs):
    """Thin wrapper so tests can monkeypatch this without touching Beckman internals."""
    from general_beckman import enqueue as beckman_enqueue
    return await beckman_enqueue(spec, **kwargs)


async def run(channel: str, payload: dict) -> dict:
    """Enqueue an LLM draft task for the given channel.

    Parameters
    ----------
    channel:
        One of the CHANNEL_CONTEXTS keys (``hn``, ``ph``, ``twitter``,
        ``linkedin``, ``reddit``).
    payload:
        Full task payload from mr_roboto dispatch.

    Returns
    -------
    dict
        ``{"status": "enqueued", "task_id": int}`` on success.
        ``{"status": "error", "error": str}`` on unknown channel.
    """
    if channel not in CHANNEL_CONTEXTS:
        return {
            "status": "error",
            "error": f"unknown launch channel: {channel!r}. "
                     f"Supported: {sorted(CHANNEL_CONTEXTS)}",
        }

    product_id = payload.get("product_id") or ""
    launch_id = payload.get("launch_id")
    spec_text = payload.get("spec") or ""
    brand_voice = payload.get("brand_voice") or ""
    mission_lessons = payload.get("mission_lessons") or []

    channel_ctx = CHANNEL_CONTEXTS[channel]

    lessons_block = ""
    if mission_lessons:
        lines = []
        for lesson in mission_lessons[:5]:
            p = (lesson.get("pattern") or "")[:120]
            f = (lesson.get("fix") or "")[:120]
            if p:
                lines.append(f"- {p}" + (f": {f}" if f else ""))
        if lines:
            lessons_block = "\nPrior launch lessons:\n" + "\n".join(lines)

    system_prompt = channel_ctx
    user_message = (
        f"Product spec:\n{spec_text[:2000]}\n\n"
        f"Brand voice:\n{brand_voice[:800]}"
        f"{lessons_block}"
    )

    try:
        task_id = await _enqueue(
            spec={
                "title": f"Draft {channel.upper()} launch post for launch #{launch_id} ({product_id})",
                "description": user_message,
                "agent_type": "assistant",
                "kind": "overhead",
                "context": {
                    "system_prompt": system_prompt,
                    "product_id": product_id,
                    "launch_id": launch_id,
                    "channel": channel,
                    "draft_kind": "launch_channel_draft",
                },
            },
        )
        logger.info(
            "launch_drafts: enqueued draft task",
            channel=channel,
            product_id=product_id,
            launch_id=launch_id,
            task_id=task_id,
        )
        return {"status": "enqueued", "task_id": task_id, "channel": channel}
    except Exception as exc:
        logger.error(
            "launch_drafts: enqueue failed",
            channel=channel,
            product_id=product_id,
            error=str(exc),
        )
        return {"status": "error", "error": str(exc)}
