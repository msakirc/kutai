# agents/signal_classifier.py
"""Z9 T3B — signal classifier agent.

Reads a batch of raw external product signals (support tickets, error
reports, analytics events) from ``task.context.payload.signals`` and
assigns each one a ``label`` and a ``domain``. The label drives the
backlog scorer's revenue-impact heuristic; the domain anchors each signal
to a recipe ``lessons_domain`` so prioritization can group like with like.

The agent emits structured verdicts only. The mechanical ``classify_signals``
executor enqueues this agent via Beckman and writes the results back as
``growth_events`` rows (``kind="classified_signal"``). The agent never
touches the database or spawns missions.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.signal_classifier")


class SignalClassifierAgent(BaseAgent):
    name = "signal_classifier"
    description = "Classifies raw growth signals into label + recipe domain"
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 2
    enable_self_reflection = False
    allowed_tools = []

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a product signal classifier. You read a batch of raw "
            "external signals — support tickets, error reports, analytics "
            "events, user messages — and label each one for the growth "
            "backlog.\n"
            "\n"
            "## Labels\n"
            "Assign exactly one `label` per signal:\n"
            "- `bug` — something is broken or behaving wrong.\n"
            "- `feature_request` — user wants new capability.\n"
            "- `churn_signal` — user is leaving, cancelling, or disengaging.\n"
            "- `pricing_feedback` — complaint or comment about price/plans.\n"
            "- `praise` — positive feedback, no action needed.\n"
            "- `spam` — irrelevant, abusive, or automated noise.\n"
            "\n"
            "## Domain\n"
            "Assign a `domain` — a short slug matching a recipe lessons "
            "domain when one fits (e.g. `auth`, `search`, `pagination`, "
            "`file_upload`, `audit_log`). The candidate domains are listed "
            "in your context under `recipe_domains`. If no recipe domain "
            "fits, use a concise general slug (e.g. `onboarding`, "
            "`billing`, `performance`).\n"
            "\n"
            "## You must\n"
            "- Always classify every signal in the batch — never skip one.\n"
            "- Always echo each signal's `external_id` verbatim so results "
            "can be joined back.\n"
            "- Always include a calibrated `confidence` in [0.0, 1.0]; use "
            "below 0.6 when the signal text is ambiguous or sparse.\n"
            "\n"
            "## You must never\n"
            "- Don't invent signals that are not in the batch.\n"
            "- Don't merge two signals into one verdict.\n"
            "- Never propose fixes, missions, or priorities — labelling and "
            "domain assignment only; scoring happens downstream.\n"
            "\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": {\n'
            '    "classifications": [\n'
            '      {\n'
            '        "external_id": "<verbatim id from the signal>",\n'
            '        "label": "bug",\n'
            '        "domain": "auth",\n'
            '        "confidence": 0.82\n'
            "      }\n"
            "    ]\n"
            "  },\n"
            '  "memories": {}\n'
            "}\n"
            "```\n"
        )
