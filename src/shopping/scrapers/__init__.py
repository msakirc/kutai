"""Shopping scrapers package.

Import concrete scrapers so they register themselves via ``@register_scraper``.
"""

from .base import BaseScraper, get_scraper, list_scrapers, register_scraper
from .akakce import AkakceScraper
from .trendyol import TrendyolScraper

__all__ = [
    "BaseScraper",
    "get_scraper",
    "list_scrapers",
    "register_scraper",
    "AkakceScraper",
    "TrendyolScraper",
]
