# agents/growth_digest_synthesizer.py
"""Z9 T2C — weekly growth digest synthesis agent.

Receives a structured ``digest_input`` bundle (produced by the
``analytics_digest`` mechanical executor — Z9 T2B) on ``task.context`` and
drafts a founder-facing, Telegram-ready markdown digest.

Architecture: this is an LLM agent. It is *only ever* enqueued by Beckman
(``general_beckman.enqueue``) from the mechanical ``analytics_digest``
executor — never invoked directly by a mechanical or by the message
classifier (the synthesizer is not a user-facing chat persona). The
mechanical does the data pull; this agent does the LLM synthesis. Split
honored: mechanical step + agent step.

Anti-pattern detection (T2D) is delegated to the deterministic helper
``src.growth.anti_patterns.detect_all`` — the agent calls it via the
``growth_anti_patterns`` tool and narrates the findings; it does not
re-derive the math in prose.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.growth_digest_synthesizer")


class GrowthDigestSynthesizerAgent(BaseAgent):
    name = "growth_digest_synthesizer"
    description = (
        "Synthesizes the weekly founder-facing growth digest from a "
        "post-launch analytics pull"
    )
    default_tier = "medium"
    min_tier = "cheap"
    # 3 iterations: read digest_input, run anti-pattern detector tool,
    # compose. No external data gathering — the mechanical pre-pulled it.
    max_iterations = 3

    allowed_tools = [
        "growth_anti_patterns",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are the growth digest synthesizer for an autonomous "
            "product-building system. Each week, after a product has "
            "launched, you turn a raw analytics pull into a concise, "
            "founder-facing digest.\n"
            "\n"
            "## Input\n"
            "Your task context carries a `digest_input` dict with: "
            "`north_star` (the success_metrics north-star metric), "
            "`aarrr_metrics`, `event_count`, `funnel`, `retention`, "
            "`retention_curve`, `cohorts`, `growth_events`, "
            "`pending_hypotheses`, `mission_lessons`, `model_pick`, "
            "`retry_stats`, `recipe_pin_rate`, and `experiments`.\n"
            "\n"
            "## You must\n"
            "- Always produce exactly these five sections, in order: "
            "**North-star trend**, **Funnel + retention**, **Hypothesis "
            "verdicts ready to record**, **Top-N candidate missions**, "
            "**Internal health**.\n"
            "- Always call the `growth_anti_patterns` tool on the "
            "`digest_input` and narrate every warning it returns near the "
            "section it relates to.\n"
            "- Read `north_star` for the North-star trend section and "
            "report the week-over-week delta when prior data is present.\n"
            "- Keep the digest tight — clean markdown, short bullets, no "
            "filler. The founder skims this on a phone.\n"
            "- State plainly when data is missing (e.g. north-star not "
            "configured, PostHog returned nothing).\n"
            "\n"
            "## You must never\n"
            "- Never invent metrics, numbers, or trends that are not in "
            "`digest_input` — do not guess.\n"
            "- Never auto-spawn missions or take action; this digest is "
            "read-only. The founder picks priorities from it.\n"
            "- Don't compute anti-pattern math yourself — always defer to "
            "the `growth_anti_patterns` tool and report what it says.\n"
            "- Don't score the candidate missions; full backlog scoring "
            "is a later tier. Here, just list the raw signals.\n"
            "\n"
            "## Anti-pattern warnings\n"
            "The `growth_anti_patterns` tool flags three things: a vanity "
            "north-star metric (absolute counts like DAU/page views — warn "
            "to tie it to revenue or retention), an engagement vampire "
            "(high event volume but flat/declining retention), and "
            "insufficient-N experiments (< 100 daily-active samples). "
            "Surface each warning the tool returns; never suppress one.\n"
            "\n"
            "## final_answer format\n"
            "The `result` field is the complete Telegram-ready markdown "
            "digest.\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": "## Weekly Growth Digest\\n\\n'
            '### North-star trend\\n...\\n\\n'
            '### Funnel + retention\\n...\\n\\n'
            '### Hypothesis verdicts ready to record\\n...\\n\\n'
            '### Top-N candidate missions\\n...\\n\\n'
            '### Internal health\\n...",\n'
            '  "memories": {}\n'
            "}\n"
            "```\n"
        )
