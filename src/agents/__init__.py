# agents/__init__.py
from .base import BaseAgent
from .planner import PlannerAgent
from .architect import ArchitectAgent
from .coder import CoderAgent
from .implementer import ImplementerAgent
from .fixer import FixerAgent
from .test_generator import TestGeneratorAgent
from .reviewer import ReviewerAgent
from .visual_reviewer import VisualReviewerAgent
from .researcher import ResearcherAgent
from .analyst import AnalystAgent
from .writer import WriterAgent
from .summarizer import SummarizerAgent
from .assistant import AssistantAgent
from .executor import ExecutorAgent
from .error_recovery import ErrorRecoveryAgent

AGENT_REGISTRY = {
    "planner": PlannerAgent(),
    "architect": ArchitectAgent(),
    "coder": CoderAgent(),
    "implementer": ImplementerAgent(),
    "fixer": FixerAgent(),
    "test_generator": TestGeneratorAgent(),
    "reviewer": ReviewerAgent(),
    "visual_reviewer": VisualReviewerAgent(),
    "researcher": ResearcherAgent(),
    "analyst": AnalystAgent(),
    "writer": WriterAgent(),
    "summarizer": SummarizerAgent(),
    "assistant": AssistantAgent(),
    "executor": ExecutorAgent(),
    "error_recovery": ErrorRecoveryAgent(),
}


def get_agent(agent_type: str) -> BaseAgent:
    """Get agent by type, fallback to executor."""
    return AGENT_REGISTRY.get(agent_type, AGENT_REGISTRY["executor"])
