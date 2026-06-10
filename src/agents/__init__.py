# agents/__init__.py
from prompt_foundry import get_profile as _get_profile
from .base import BaseAgent
from .reviewer import ReviewerAgent
from .visual_reviewer import VisualReviewerAgent
from .researcher import ResearcherAgent
from .analyst import AnalystAgent
from .writer import WriterAgent
from .assistant import AssistantAgent
from .executor import ExecutorAgent
from .shopping_advisor import ShoppingAdvisorAgent
from .product_researcher import ProductResearcherAgent
from .deal_analyst import DealAnalystAgent
from .shopping_clarifier import ShoppingClarifierAgent
from .shopping_grouper import ShoppingGrouperAgent
from .shopping_labeler import ShoppingLabelerAgent
from .shopping_synthesizer import ShoppingSynthesizerAgent
from .integration_reviewer import IntegrationReviewerAgent
from .adr_drift_judge import AdrDriftJudgeAgent
from .oncall_agent import OncallAgent
from .support_tier1 import SupportTier1Agent
from .growth_digest_synthesizer import GrowthDigestSynthesizerAgent
from .signal_classifier import SignalClassifierAgent
from .query_planner import QueryPlannerAgent
from .prior_art_synthesizer import PriorArtSynthesizerAgent

AGENT_REGISTRY = {
    "reviewer": ReviewerAgent(),
    "visual_reviewer": VisualReviewerAgent(),
    "researcher": ResearcherAgent(),
    "analyst": AnalystAgent(),
    "writer": WriterAgent(),
    "assistant": AssistantAgent(),
    "executor": ExecutorAgent(),
    "shopping_advisor": ShoppingAdvisorAgent(),
    "product_researcher": ProductResearcherAgent(),
    "deal_analyst": DealAnalystAgent(),
    "shopping_clarifier": ShoppingClarifierAgent(),
    "shopping_grouper": ShoppingGrouperAgent(),
    "shopping_labeler": ShoppingLabelerAgent(),
    "shopping_synthesizer": ShoppingSynthesizerAgent(),
    "integration_reviewer": IntegrationReviewerAgent(),
    "adr_drift_judge": AdrDriftJudgeAgent(),
    "oncall_agent": OncallAgent(),
    "support_tier1": SupportTier1Agent(),
    "growth_digest_synthesizer": GrowthDigestSynthesizerAgent(),
    "signal_classifier": SignalClassifierAgent(),
    "query_planner": QueryPlannerAgent(),
    "prior_art_synthesizer": PriorArtSynthesizerAgent(),
}


def get_agent(agent_type: str):
    """Get agent/profile by type. Foundry data profiles take precedence;
    legacy class instances are the fallback; executor is the final default.

    Returns a stable per-type singleton: Foundry's get_profile returns the
    registry singleton (built once at Foundry import), and the class
    instances here are constructed once at module import — identity holds
    across calls for both paths.
    """
    p = _get_profile(agent_type)
    if p is not None:
        return p
    return AGENT_REGISTRY.get(agent_type, AGENT_REGISTRY["executor"])
