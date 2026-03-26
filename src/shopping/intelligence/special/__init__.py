"""Special intelligence features for shopping analysis.

Provides advanced analysis capabilities beyond basic price comparison:
warranty analysis, fake discount detection, total cost of ownership,
seasonal buying advice, unit price calculation, seller trust scoring,
and exchange rate impact assessment.
"""

from src.shopping.intelligence.special.warranty_analyzer import (
    analyze_warranty,
    get_service_network,
    assess_gray_market_risk,
)
from src.shopping.intelligence.special.fake_discount_detector import (
    detect_fake_discount,
    check_price_inflation,
    check_cross_store_consistency,
)
from src.shopping.intelligence.special.tco_calculator import (
    calculate_tco,
    estimate_energy_cost,
    estimate_consumable_cost,
    compare_tco,
)
from src.shopping.intelligence.special.seasonal_advisor import (
    get_seasonal_advice,
    get_upcoming_sales,
    is_good_time_to_buy,
)
from src.shopping.intelligence.special.unit_price_calculator import (
    calculate_unit_price,
    compare_unit_prices,
    detect_quantity,
)
from src.shopping.intelligence.special.seller_trust import (
    score_seller,
    check_seller_age,
    check_review_authenticity,
)
from src.shopping.intelligence.special.exchange_rate import (
    get_usd_try_rate,
    get_rate_trend,
    assess_import_price_impact,
)

__all__ = [
    "analyze_warranty",
    "get_service_network",
    "assess_gray_market_risk",
    "detect_fake_discount",
    "check_price_inflation",
    "check_cross_store_consistency",
    "calculate_tco",
    "estimate_energy_cost",
    "estimate_consumable_cost",
    "compare_tco",
    "get_seasonal_advice",
    "get_upcoming_sales",
    "is_good_time_to_buy",
    "calculate_unit_price",
    "compare_unit_prices",
    "detect_quantity",
    "score_seller",
    "check_seller_age",
    "check_review_authenticity",
    "get_usd_try_rate",
    "get_rate_trend",
    "assess_import_price_impact",
]
