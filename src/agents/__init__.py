# agents/__init__.py
from .base import BaseAgent
from .planner import PlannerAgent
from .coder import CoderAgent
from .researcher import ResearcherAgent
from .writer import WriterAgent
from .reviewer import ReviewerAgent
from .executor import ExecutorAgent
from .architect import ArchitectAgent
from .implementer import ImplementerAgent
from .test_generator import TestGeneratorAgent
from .fixer import FixerAgent
from .error_recovery import ErrorRecoveryAgent

AGENT_REGISTRY = {
    "planner": PlannerAgent(),
    "coder": CoderAgent(),
    "researcher": ResearcherAgent(),
    "writer": WriterAgent(),
    "reviewer": ReviewerAgent(),
    "executor": ExecutorAgent(),
    "architect": ArchitectAgent(),
    "implementer": ImplementerAgent(),
    "test_generator": TestGeneratorAgent(),
    "fixer": FixerAgent(),
    "error_recovery": ErrorRecoveryAgent(),
}


def get_agent(agent_type: str) -> BaseAgent:
    """Get agent by type, fallback to executor."""
    return AGENT_REGISTRY.get(agent_type, AGENT_REGISTRY["executor"])
