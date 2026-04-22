"""Pure-code filter + gate logic for variant disambiguation."""
from __future__ import annotations

from src.workflows.shopping.pipeline_v2 import ProductGroup

FILTER_AUTHENTICITY_MIN = 0.7


def step_filter(groups: list[ProductGroup]) -> list[ProductGroup]:
    """Drop groups that aren't authentic, intent-matched products."""
    return [
        g for g in groups
        if g.product_type == "authentic_product"
        and g.matches_user_intent
        and g.authenticity_confidence >= FILTER_AUTHENTICITY_MIN
    ]
