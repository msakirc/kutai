# src/tools/relevance.py
"""BM25 relevance scoring and adaptive context budget allocation.

Standalone module — scores documents against a query and allocates
character budgets proportional to relevance.
"""

from dataclasses import dataclass

from src.infra.logging_config import get_logger
from src.tools.content_extract import ExtractedContent

logger = get_logger("tools.relevance")

_MIN_BUDGET_PER_PAGE = 200
_MAX_BUDGET_RATIO = 0.4

_INTENT_BOOSTS = {
    "product": {"prices": 0.3, "reviews": 0.0},
    "reviews": {"prices": 0.0, "reviews": 0.3},
    "market":  {"prices": 0.1, "reviews": 0.1},
    "research": {"prices": 0.0, "reviews": 0.0},
    "factual": {"prices": 0.0, "reviews": 0.0},
}


@dataclass
class BudgetedContent:
    content: ExtractedContent
    relevance_score: float
    allocated_chars: int
    truncated_text: str


def _bm25_score(contents, query):
    if not contents:
        return []
    docs = [c.text for c in contents]
    try:
        import bm25s
        query_tokens = bm25s.tokenize([query])
        doc_tokens = bm25s.tokenize(docs)
        retriever = bm25s.BM25()
        retriever.index(doc_tokens)
        results_docs, results_scores = retriever.retrieve(query_tokens, corpus=list(range(len(docs))), k=len(docs))
        score_map = {}
        for idx, score in zip(results_docs[0], results_scores[0]):
            score_map[int(idx)] = float(score)
        return [score_map.get(i, 0.0) for i in range(len(docs))]
    except Exception as e:
        logger.debug("bm25s scoring failed, using term frequency fallback", error=str(e)[:100])
        query_terms = set(query.lower().split())
        scores = []
        for doc in docs:
            doc_lower = doc.lower()
            matches = sum(1 for t in query_terms if t in doc_lower)
            scores.append(matches / max(len(query_terms), 1))
        return scores


def _truncate_at_sentence(text, max_chars):
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in [". ", ".\n", "! ", "? "]:
        last_sep = truncated.rfind(sep)
        if last_sep > max_chars * 0.5:
            return truncated[:last_sep + 1]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        return truncated[:last_space] + "..."
    return truncated + "..."


def score_and_budget(contents, query, total_budget=12000, intent="factual"):
    if not contents:
        return []

    raw_scores = _bm25_score(contents, query)

    boosts = _INTENT_BOOSTS.get(intent, {"prices": 0.0, "reviews": 0.0})
    adjusted_scores = []
    for score, content in zip(raw_scores, contents):
        adj = score
        if content.has_prices:
            adj += boosts["prices"]
        if content.has_reviews:
            adj += boosts["reviews"]
        adjusted_scores.append(max(adj, 0.01))

    total_score = sum(adjusted_scores)
    max_per_page = int(total_budget * _MAX_BUDGET_RATIO)

    budgets = []
    for score in adjusted_scores:
        raw_budget = int((score / total_score) * total_budget) if total_score > 0 else total_budget // len(contents)
        clamped = max(_MIN_BUDGET_PER_PAGE, min(raw_budget, max_per_page))
        budgets.append(clamped)

    budget_sum = sum(budgets)
    if budget_sum > total_budget:
        scale = total_budget / budget_sum
        budgets = [max(_MIN_BUDGET_PER_PAGE, int(b * scale)) for b in budgets]

    results = []
    for content, score, budget in zip(contents, adjusted_scores, budgets):
        truncated = _truncate_at_sentence(content.text, budget) if content.text else ""
        results.append(BudgetedContent(content=content, relevance_score=round(score, 4),
                                       allocated_chars=budget, truncated_text=truncated))

    results.sort(key=lambda b: b.relevance_score, reverse=True)

    logger.debug("budget allocation complete", pages=len(results), total_budget=total_budget,
                 used=sum(len(b.truncated_text) for b in results), intent=intent)
    return results
