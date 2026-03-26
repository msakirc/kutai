"""Shopping output formatters.

Re-exports the public API from the three formatter modules.
"""

from .formatters import (
    format_comparison_table,
    format_installment_options,
    format_price,
    format_price_comparison,
)
from .product_cards import (
    format_combo_card,
    format_deal_card,
    format_product_card,
    format_product_cards_batch,
)
from .summary import (
    format_action_buttons,
    format_alternatives,
    format_budget_option,
    format_recommendation_summary,
    format_timing_advice,
    format_top_pick,
    format_warnings,
)

__all__ = [
    # formatters
    "format_comparison_table",
    "format_installment_options",
    "format_price",
    "format_price_comparison",
    # summary
    "format_action_buttons",
    "format_alternatives",
    "format_budget_option",
    "format_recommendation_summary",
    "format_timing_advice",
    "format_top_pick",
    "format_warnings",
    # product_cards
    "format_combo_card",
    "format_deal_card",
    "format_product_card",
    "format_product_cards_batch",
]
