# agents/oncall_agent.py
"""Z8 T4A — on-call agent for live production alerts.

Reads an alert payload (envelope at ``task.context.payload.payload`` from
Z8 T3 alert_triage), matches against incident playbooks (T4C), and emits a
single whitelisted action verb. Real action execution rides on the
``oncall_action`` mechanical executor (T4B), which enforces per-verb
cooldowns before delegating to verb-specific sub-handlers.

A11.r1 refactor: handlers are now discovered from the handler registry
(``coulson.agent_handlers.registry``), making the agent configurable for
new domains (e.g. ``'mention'``) without editing this file.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.oncall_agent")

# Ops-domain whitelist (static for prompt generation — the executor's
# registry is the live source of truth at dispatch time).
_OPS_WHITELIST = [
    "restart_service",
    "rollback_to_last_green",
    "scale_up",
    "scale_down",
    "drain_traffic",
    "rotate_failed_key",
    "archive_flake_test",
    "escalate_to_founder",
]


def _get_whitelisted_verbs(domain: str = "ops") -> list[str]:
    """Return registered verbs for *domain* from the handler registry.

    Falls back to the static ops whitelist if the registry is unavailable
    (e.g. import-time bootstrap not completed) so tests and dry-runs work.
    """
    try:
        from coulson.agent_handlers.registry import get_whitelist
        verbs = sorted(get_whitelist(domain))
        return verbs if verbs else _OPS_WHITELIST
    except Exception:
        return _OPS_WHITELIST


class OncallAgent(BaseAgent):
    name = "oncall_agent"
    description = "On-call engineer for live production alerts (registry-backed actions)"
    default_tier = "mid"
    min_tier = "cheap"
    max_iterations = 4

    allowed_tools = [
        "vendor_call",
        "oncall_action",
        "escalate_to_founder",
        "ops_log_write",
    ]

    def get_system_prompt(self, task: dict) -> str:
        domain = (task.get("context") or {}).get("domain", "ops")
        whitelist = _get_whitelisted_verbs(domain)
        whitelist_lines = "\n".join(f"- {v}" for v in whitelist)
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
            f"{whitelist_lines}\n"
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
