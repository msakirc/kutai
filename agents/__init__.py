# agents/__init__.py
from agents.base import BaseAgent
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from agents.coder import CoderAgent
from agents.reviewer import ReviewerAgent
from agents.executor import ExecutorAgent

AGENT_REGISTRY = {
    "planner": PlannerAgent,
    "researcher": ResearcherAgent,
    "writer": WriterAgent,
    "coder": CoderAgent,
    "reviewer": ReviewerAgent,
    "executor": ExecutorAgent,
}

def get_agent(agent_type: str) -> BaseAgent:
    cls = AGENT_REGISTRY.get(agent_type, ExecutorAgent)
    return cls()
