"""Shopping scrapers package.

Import concrete scrapers so they register themselves via ``@register_scraper``.
"""

from .base import BaseScraper, get_scraper, list_scrapers, register_scraper
from .akakce import AkakceScraper
from .trendyol import TrendyolScraper
from .hepsiburada import HepsiburadaScraper
from .amazon_tr import AmazonTrScraper
from .forums import TechnopatScraper, DonanimHaberScraper
from .eksisozluk import EksiSozlukScraper
from .sikayetvar import SikayetvarScraper
from .grocery import AktuelKatalogScraper, GetirScraper, MigrosScraper
from .sahibinden import SahibindenScraper
from .home_improvement import KoctasScraper, IKEAScraper
from .google_cse import GoogleCSEScraper

__all__ = [
    "BaseScraper",
    "get_scraper",
    "list_scrapers",
    "register_scraper",
    "AkakceScraper",
    "TrendyolScraper",
    "HepsiburadaScraper",
    "AmazonTrScraper",
    "TechnopatScraper",
    "DonanimHaberScraper",
    "EksiSozlukScraper",
    "SikayetvarScraper",
    "AktuelKatalogScraper",
    "GetirScraper",
    "MigrosScraper",
    "SahibindenScraper",
    "KoctasScraper",
    "IKEAScraper",
    "GoogleCSEScraper",
]
