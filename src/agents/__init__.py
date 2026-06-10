# agents/__init__.py
from prompt_foundry import PROFILE_REGISTRY, get_profile as _get_profile
from .base import BaseAgent
from .oncall_agent import OncallAgent

AGENT_REGISTRY = {**PROFILE_REGISTRY, "oncall_agent": OncallAgent()}


def get_agent(agent_type: str):
    """Get agent/profile by type. Foundry data profiles take precedence;
    oncall_agent carve-out instance is in AGENT_REGISTRY; executor is the
    final default.

    Returns a stable per-type singleton: Foundry's PROFILE_REGISTRY is built
    once at Foundry import, and the OncallAgent instance is constructed once
    at module import — identity holds across calls for both paths.
    """
    return AGENT_REGISTRY.get(agent_type) or AGENT_REGISTRY["executor"]
