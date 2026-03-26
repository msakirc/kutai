# shopping/integrations/researcher_bridge.py
"""
Bridge between shopping product_researcher and the existing researcher agent.

Wraps calls to the existing researcher pattern so that shopping workflows
can leverage the general-purpose research infrastructure.
"""
from __future__ import annotations

from src.infra.logging_config import get_logger

logger = get_logger("shopping.integrations.researcher_bridge")


async def research_with_existing_agent(
    query: str,
    shopping_context: dict | None = None,
) -> dict:
    """
    Perform research using the existing researcher agent infrastructure.

    This bridges the shopping product_researcher to the general-purpose
    researcher agent, enriching the query with shopping context.

    Args:
        query: The research query.
        shopping_context: Optional dict with keys like product_name,
            category, price_range, location, sub_intent.

    Returns:
        dict with keys: answer (str), sources (list), metadata (dict).
        On failure returns a dict with an error key.
    """
    ctx = shopping_context or {}

    # Build an enriched query with shopping context
    enriched_parts = [query]
    if ctx.get("category"):
        enriched_parts.append(f"category: {ctx['category']}")
    if ctx.get("price_range"):
        enriched_parts.append(f"budget: {ctx['price_range']}")
    if ctx.get("location"):
        enriched_parts.append(f"market: {ctx['location']}")

    enriched_query = " | ".join(enriched_parts)

    try:
        from src.agents import get_agent

        agent = get_agent("researcher")
        # Build a minimal task dict that the researcher agent expects
        task = {
            "id": 0,
            "title": f"Shopping research: {query[:50]}",
            "description": enriched_query,
            "tier": "auto",
            "metadata": {
                "shopping_context": ctx,
                "source": "shopping_researcher_bridge",
            },
        }

        result = await agent.run(task)

        # Normalize the result
        if isinstance(result, str):
            return {"answer": result, "sources": [], "metadata": ctx}
        if isinstance(result, dict):
            return {
                "answer": result.get("result", result.get("answer", str(result))),
                "sources": result.get("sources", []),
                "metadata": {**ctx, **result.get("metadata", {})},
            }
        return {"answer": str(result), "sources": [], "metadata": ctx}

    except Exception as e:
        logger.warning(
            "researcher bridge failed, falling back to web search",
            error=str(e),
            query=query[:100],
        )
        # Fallback: use web search directly
        try:
            from src.tools.web_search import web_search
            answer = await web_search(enriched_query, max_results=5)
            return {"answer": answer, "sources": [], "metadata": {**ctx, "fallback": True}}
        except Exception as e2:
            logger.error("researcher bridge fallback also failed", error=str(e2))
            return {"answer": "", "sources": [], "error": str(e2), "metadata": ctx}
