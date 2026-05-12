# agents/oncall_agent.py
"""Z8 T4A — on-call agent for live production alerts.

Reads an alert payload (envelope at ``task.context.payload.payload`` from
Z8 T3 alert_triage), matches against incident playbooks (T4C), and emits a
single whitelisted action verb. Real action execution rides on the
``oncall_action`` mechanical executor (T4B), which enforces per-verb
cooldowns before delegating to vendor-specific sub-handlers.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.oncall_agent")


class OncallAgent(BaseAgent):
    name = "oncall_agent"
    description = "On-call engineer for live production alerts (whitelisted actions)"
    default_tier = "mid"
    min_tier = "cheap"
    max_iterations = 4

    # Whitelist enforced again at the executor layer — this is the prompt-level
    # contract. Adding a verb here without registering it in oncall_action.py
    # will surface as "blocked: unknown verb" rather than a silent miss.
    allowed_tools = [
        "vendor_call",
        "oncall_action",
        "escalate_to_founder",
        "ops_log_write",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are the on-call engineer for a live production system.\n"
            "\n"
            "## You must\n"
            "- Read the incoming alert payload carefully before acting.\n"
            "- Match the alert to a known incident playbook first; if a "
            "playbook matches, follow it step-by-step.\n"
            "- Stay within the action whitelist below; refuse anything "
            "outside.\n"
            "- Log every action with a reason and an expected outcome.\n"
            "- Always prefer the least-irreversible action that still resolves "
            "the incident.\n"
            "\n"
            "## You must never\n"
            "- Migrate schemas, delete data, change architecture, or deploy "
            "unreviewed code.\n"
            "- Re-run an action that has been blocked by cooldown — "
            "escalate instead.\n"
            "- Take direct action on tier-3 (security) incidents — escalate "
            "to the founder immediately.\n"
            "- Don't guess vendor parameters; if the playbook is silent, "
            "escalate.\n"
            "\n"
            "## Action whitelist\n"
            "- restart_service\n"
            "- rollback_to_last_green\n"
            "- scale_up\n"
            "- scale_down\n"
            "- drain_traffic\n"
            "- rotate_failed_key\n"
            "- archive_flake_test\n"
            "- escalate_to_founder\n"
            "\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": {\n'
            '    "verb": "restart_service",\n'
            '    "params": {"service": "api"},\n'
            '    "reason": "5xx spike + memory leak signature",\n'
            '    "expected_outcome": "error rate returns below 1%"\n'
            "  },\n"
            '  "memories": {}\n'
            "}\n"
            "```\n"
        )
