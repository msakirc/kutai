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
from .shopping_advisor import ShoppingAdvisorAgent
from .product_researcher import ProductResearcherAgent
from .deal_analyst import DealAnalystAgent
from .shopping_clarifier import ShoppingClarifierAgent
from .integration_reviewer import IntegrationReviewerAgent
from .adr_drift_judge import AdrDriftJudgeAgent
from .oncall_agent import OncallAgent
from .support_tier1 import SupportTier1Agent
from .growth_digest_synthesizer import GrowthDigestSynthesizerAgent
from .signal_classifier import SignalClassifierAgent
from .query_planner import QueryPlannerAgent
from .prior_art_synthesizer import PriorArtSynthesizerAgent

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
    "shopping_advisor": ShoppingAdvisorAgent(),
    "product_researcher": ProductResearcherAgent(),
    "deal_analyst": DealAnalystAgent(),
    "shopping_clarifier": ShoppingClarifierAgent(),
    "integration_reviewer": IntegrationReviewerAgent(),
    "adr_drift_judge": AdrDriftJudgeAgent(),
    "oncall_agent": OncallAgent(),
    "support_tier1": SupportTier1Agent(),
    "growth_digest_synthesizer": GrowthDigestSynthesizerAgent(),
    "signal_classifier": SignalClassifierAgent(),
    "query_planner": QueryPlannerAgent(),
    "prior_art_synthesizer": PriorArtSynthesizerAgent(),
}


def get_agent(agent_type: str) -> BaseAgent:
    """Get agent by type, fallback to executor."""
    return AGENT_REGISTRY.get(agent_type, AGENT_REGISTRY["executor"])
