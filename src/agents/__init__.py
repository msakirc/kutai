# agents/__init__.py
from prompt_foundry import get_profile as _get_profile
from .base import BaseAgent
from .writer import WriterAgent
from .shopping_advisor import ShoppingAdvisorAgent
from .product_researcher import ProductResearcherAgent
from .deal_analyst import DealAnalystAgent
from .shopping_clarifier import ShoppingClarifierAgent
from .shopping_grouper import ShoppingGrouperAgent
from .shopping_labeler import ShoppingLabelerAgent
from .shopping_synthesizer import ShoppingSynthesizerAgent
from .oncall_agent import OncallAgent

AGENT_REGISTRY = {
    "writer": WriterAgent(),
    "shopping_advisor": ShoppingAdvisorAgent(),
    "product_researcher": ProductResearcherAgent(),
    "deal_analyst": DealAnalystAgent(),
    "shopping_clarifier": ShoppingClarifierAgent(),
    "shopping_grouper": ShoppingGrouperAgent(),
    "shopping_labeler": ShoppingLabelerAgent(),
    "shopping_synthesizer": ShoppingSynthesizerAgent(),
    "oncall_agent": OncallAgent(),
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
    result = AGENT_REGISTRY.get(agent_type)
    if result is not None:
        return result
    # executor is now a Foundry profile; fall back to it as the default.
    return _get_profile("executor")
