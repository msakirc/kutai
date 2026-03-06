# agents/__init__.py
from agents.base import BaseAgent
from agents.planner import PlannerAgent
from agents.coder import CoderAgent
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.reviewer import ReviewerAgent
from agents.executor import ExecutorAgent
from agents.architect import ArchitectAgent
from agents.implementer import ImplementerAgent
from agents.test_generator import TestGeneratorAgent
from agents.fixer import FixerAgent
from agents.error_recovery import ErrorRecoveryAgent

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
