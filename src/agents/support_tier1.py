# agents/support_tier1.py
"""Z8 T5E — support tier-1 agent.

Reads a user question (from ``task.context.payload.question``), receives RAG
hits from the ``support_docs`` Chroma collection in its context, and answers
with citations. Confidence below 0.7 OR detected anger/urgency signals → the
mechanical layer escalates via ``founder_actions(kind='support_escalation')``.

The agent itself only emits text + confidence. Escalation routing lives in
``src/ops/support_rag.py`` so retries don't compound user-facing messages.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.support_tier1")


class SupportTier1Agent(BaseAgent):
    name = "support_tier1"
    description = "Tier-1 support — RAG-grounded answers with confidence scoring"
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 3
    enable_self_reflection = False
    allowed_tools = [
        "read_file",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a tier-1 support agent. You answer user questions using "
            "the retrieved support documents in context — nothing else.\n"
            "\n"
            "## You must\n"
            "- Always ground every claim in the retrieved support_docs "
            "passages; cite each source by its doc_id.\n"
            "- Always include a calibrated `confidence` ∈ [0.0, 1.0] for your "
            "answer. Use 1.0 only when the docs answer the question verbatim; "
            "use <0.7 when you're inferring or the docs don't directly cover "
            "the question.\n"
            "- Always be polite, concise, and acknowledge frustration when "
            "the user expresses it.\n"
            "\n"
            "## You must never\n"
            "- Don't fabricate. If the docs don't cover it, say so and set "
            "confidence below 0.5 so the escalation path triggers.\n"
            "- Don't promise refunds, credits, or policy changes — those "
            "always escalate to the founder.\n"
            "- Never reveal internal infrastructure, credentials, or other "
            "users' data.\n"
            "\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": {\n'
            '    "answer": "<text shown to the user>",\n'
            '    "confidence": 0.85,\n'
            '    "citations": ["doc_id_1", "doc_id_2"]\n'
            "  },\n"
            '  "memories": {}\n'
            "}\n"
            "```\n"
        )
