# shopping/integrations/__init__.py
"""
Shopping integration bridges — connect shopping workflows to existing
infrastructure (Perplexica, researcher agent, web tools).
"""
from .perplexica import search_perplexica, format_shopping_query, assess_result_quality
from .researcher_bridge import research_with_existing_agent
from .web_tools_bridge import fallback_web_search, fallback_web_extract

__all__ = [
    "search_perplexica",
    "format_shopping_query",
    "assess_result_quality",
    "research_with_existing_agent",
    "fallback_web_search",
    "fallback_web_extract",
]
