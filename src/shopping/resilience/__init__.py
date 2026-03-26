"""Shopping resilience modules.

Provides graceful degradation, rate budgeting, circuit breaking,
error recovery, cache fallback, and anti-detection monitoring for
the shopping system.
"""

from src.shopping.resilience.fallback_chain import (
    execute_with_fallback,
    get_product_with_fallback,
    build_fallback_chain,
)
from src.shopping.resilience.rate_budget import (
    init_rate_budget_db,
    get_remaining_budget,
    consume_budget,
    get_budget_summary,
    reset_daily_budgets,
)
from src.shopping.resilience.circuit_breaker import (
    CircuitBreaker,
    check_circuit,
    record_success,
    record_failure,
    get_circuit_status,
)
from src.shopping.resilience.error_recovery import (
    handle_scraper_error,
    handle_llm_error,
    classify_error,
)
from src.shopping.resilience.cache_fallback import (
    get_stale_product,
    get_stale_price,
    warmup_cache,
)
from src.shopping.resilience.detection_monitor import (
    DetectionMonitor,
    record_request,
    is_domain_cooled_down,
    get_success_rate,
    get_detection_metrics,
)
from src.shopping.resilience.price_verification import (
    verify_prices,
    flag_outliers,
)
from src.shopping.resilience.staleness import (
    assess_staleness,
    detect_flash_sale,
    get_recommended_ttl,
)

__all__ = [
    "execute_with_fallback",
    "get_product_with_fallback",
    "build_fallback_chain",
    "init_rate_budget_db",
    "get_remaining_budget",
    "consume_budget",
    "get_budget_summary",
    "reset_daily_budgets",
    "CircuitBreaker",
    "check_circuit",
    "record_success",
    "record_failure",
    "get_circuit_status",
    "handle_scraper_error",
    "handle_llm_error",
    "classify_error",
    "get_stale_product",
    "get_stale_price",
    "warmup_cache",
    "DetectionMonitor",
    "record_request",
    "is_domain_cooled_down",
    "get_success_rate",
    "get_detection_metrics",
    "verify_prices",
    "flag_outliers",
    "assess_staleness",
    "detect_flash_sale",
    "get_recommended_ttl",
]
