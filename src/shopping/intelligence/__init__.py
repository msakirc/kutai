"""Shopping intelligence modules.

Provides LLM-assisted and rule-based analysis for the shopping system:
query analysis, search planning, alternatives, substitutions,
constraint checking, value scoring, and product matching.
"""

from src.shopping.intelligence.query_analyzer import analyze_query
from src.shopping.intelligence.search_planner import generate_search_plan
from src.shopping.intelligence.alternatives import generate_alternatives
from src.shopping.intelligence.substitution import suggest_substitutions
from src.shopping.intelligence.constraints import check_constraints
from src.shopping.intelligence.value_scorer import score_products
from src.shopping.intelligence.product_matcher import match_products

__all__ = [
    "analyze_query",
    "generate_search_plan",
    "generate_alternatives",
    "suggest_substitutions",
    "check_constraints",
    "score_products",
    "match_products",
]
