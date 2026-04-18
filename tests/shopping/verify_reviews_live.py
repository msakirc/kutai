"""End-to-end live verification: every scraper's get_reviews on real URLs.

Runs each scraper's get_reviews against 2 real product URLs (sibling pages
to test selector resilience). Prints a summary table.

NOT a pytest — run manually:
    python -m tests.shopping.verify_reviews_live
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

# ---------------------------------------------------------------------------
# (scraper_module, ScraperClass, domain_arg, [(url, label), (url, label)])
# Uses 2 sibling URLs per site to catch parser fragility.
# Sites with auth-walled or genuinely-absent reviews are listed but expected
# to return 0 — flagged "EXPECTED_EMPTY" in the table.
# ---------------------------------------------------------------------------

CASES: list[dict[str, Any]] = [
    # --- E-commerce with rich reviews ---
    {
        "name": "trendyol",
        "module": "src.shopping.scrapers.trendyol",
        "class": "TrendyolScraper",
        "domain": "trendyol",
        "urls": [
            ("https://www.trendyol.com/apple/iphone-16-128gb-beyaz-p-857296082", "trendyol_iphone16_white"),
            ("https://www.trendyol.com/apple/iphone-16-128gb-siyah-p-857296095", "trendyol_iphone16_black"),
        ],
    },
    {
        "name": "hepsiburada",
        "module": "src.shopping.scrapers.hepsiburada",
        "class": "HepsiburadaScraper",
        "domain": "hepsiburada",
        "urls": [
            ("https://www.hepsiburada.com/apple-iphone-15-128-gb-p-HBCV00004X9ZCH", "hb_iphone15"),
            ("https://www.hepsiburada.com/samsung-galaxy-a55-5g-256-gb-samsung-turkiye-garantili-p-HBCV00005EHV2N", "hb_a55"),
        ],
    },
    {
        "name": "amazon_tr",
        "module": "src.shopping.scrapers.amazon_tr",
        "class": "AmazonTrScraper",
        "domain": "amazon",
        "urls": [
            ("https://www.amazon.com.tr/dp/9750726324", "amzn_book1"),
            ("https://www.amazon.com.tr/dp/6053609811", "amzn_book2"),
        ],
    },
    {
        "name": "kitapyurdu",
        "module": "src.shopping.scrapers.kitapyurdu",
        "class": "KitapyurduScraper",
        "domain": "kitapyurdu",
        "urls": [
            ("https://www.kitapyurdu.com/kitap/kuyucakli-yusuf/2706.html", "ky_kuyucakli"),
            ("https://www.kitapyurdu.com/kitap/saatleri-ayarlama-enstitusu/2701.html", "ky_saatleri"),
        ],
    },
    {
        "name": "dr_com_tr",
        "module": "src.shopping.scrapers.dr_com_tr",
        "class": "DrComTrScraper",
        "domain": "dr",
        "urls": [
            ("https://www.dr.com.tr/kitap/sefiller-cilt-1/victor-hugo/edebiyat/dunya-edebiyati/urunno=0001794990001", "dr_sefiller"),
            ("https://www.dr.com.tr/kitap/uc-silahsorler/alexandre-dumas/edebiyat/dunya-edebiyati/urunno=0001795788001", "dr_silahsorler"),
        ],
    },
    {
        "name": "decathlon_tr",
        "module": "src.shopping.scrapers.decathlon_tr",
        "class": "DecathlonTrScraper",
        "domain": "decathlon",
        "urls": [
            ("https://www.decathlon.com.tr/p/yetiskinler-icin-yuruyus-ayakkabisi-mh100/_/R-p-307671", "dec_mh100"),
            ("https://www.decathlon.com.tr/p/yetiskinler-icin-su-gecirmez-rain-cut-yagmurluk-siyah/_/R-p-300100", "dec_raincut"),
        ],
    },
    # --- Aggregator / specs / classifieds ---
    {
        "name": "epey",
        "module": "src.shopping.scrapers.epey",
        "class": "EpeyScraper",
        "domain": "epey",
        "urls": [
            ("https://www.epey.com/akilli-telefonlar/xiaomi-17-ultra-1tb.html", "epey_xiaomi17"),
            ("https://www.epey.com/akilli-telefonlar/oneplus-15.html", "epey_oneplus15"),
        ],
    },
    {
        "name": "akakce",
        "module": "src.shopping.scrapers.akakce",
        "class": "AkakceScraper",
        "domain": "akakce",
        "urls": [
            ("https://www.akakce.com/akilli-telefon/en-ucuz-iphone-17-pro-max-256-gb-fiyati,370526315.html", "akakce_iphone17"),
            ("https://www.akakce.com/akilli-telefon/en-ucuz-galaxy-s24-ultra-fiyati,1543070418.html", "akakce_s24"),
        ],
    },
    {
        "name": "arabam",
        "module": "src.shopping.scrapers.arabam",
        "class": "ArabamScraper",
        "domain": "arabam",
        "urls": [
            ("https://www.arabam.com/ikinci-el/otomobil-volkswagen-golf", "arabam_golf"),
        ],
        "expected_empty": True,
        "note": "arabam.com has no per-model user-review feature (verified)",
    },
    # --- Grocery / home ---
    {
        "name": "migros",
        "module": "src.shopping.scrapers.grocery",
        "class": "MigrosScraper",
        "domain": "migros",
        "urls": [
            ("https://www.migros.com.tr/lay-s-baharatli-cips-tr-104-gr-p-2cd97d", "migros_lays"),
        ],
        "expected_empty": True,
        "note": "Migros review endpoints are session-gated",
    },
    {
        "name": "getir",
        "module": "src.shopping.scrapers.grocery",
        "class": "GetirScraper",
        "domain": "getir",
        "urls": [],
        "expected_empty": True,
        "note": "Quick commerce — no per-product reviews",
    },
    {
        "name": "koctas",
        "module": "src.shopping.scrapers.home_improvement",
        "class": "KoctasScraper",
        "domain": "koctas",
        "urls": [
            ("https://www.koctas.com.tr/p/501006193/karaca-akilli-mutfak-robotu", "koctas_robot"),
        ],
        "note": "Akamai-blocked; returns aggregate via Algolia",
    },
    {
        "name": "ikea",
        "module": "src.shopping.scrapers.home_improvement",
        "class": "IKEAScraper",
        "domain": "ikea",
        "urls": [],
        "expected_empty": True,
        "note": "IKEA TR Magic Click platform has no review widget",
    },
    # --- Review-style sites ---
    {
        "name": "eksisozluk",
        "module": "src.shopping.scrapers.eksisozluk",
        "class": "EksiSozlukScraper",
        "domain": "eksisozluk",
        "urls": [
            ("https://eksisozluk.com/iphone-15-pro-max--7438862", "eksi_iphone15"),
        ],
    },
    {
        "name": "sikayetvar",
        "module": "src.shopping.scrapers.sikayetvar",
        "class": "SikayetvarScraper",
        "domain": "sikayetvar",
        "urls": [
            ("https://www.sikayetvar.com/turkcell", "sv_turkcell"),
        ],
    },
]


async def run_one(case: dict) -> list[tuple[str, int, str]]:
    mod = __import__(case["module"], fromlist=[case["class"]])
    cls = getattr(mod, case["class"])
    try:
        scraper = cls(case["domain"]) if "domain" in case else cls()
    except TypeError:
        scraper = cls()

    results: list[tuple[str, int, str]] = []
    for url, label in case["urls"]:
        t0 = time.monotonic()
        try:
            reviews = await scraper.get_reviews(url)
            elapsed = time.monotonic() - t0
            results.append((label, len(reviews or []), f"{elapsed:.1f}s"))
        except Exception as exc:
            results.append((label, -1, f"ERR:{type(exc).__name__}:{str(exc)[:60]}"))
    return results


async def main():
    print(f"{'scraper':<14} {'url-label':<22} {'count':>7}  {'note':<8}")
    print("-" * 70)
    for case in CASES:
        if not case["urls"]:
            note = case.get("note", "")
            print(f"{case['name']:<14} {'(no urls)':<22} {'-':>7}  {note}", flush=True)
            continue
        results = await run_one(case)
        for label, count, info in results:
            tag = ""
            if count == 0 and case.get("expected_empty"):
                tag = "EXPECTED_EMPTY"
            elif count == 0:
                tag = "ZERO!!"
            elif count < 0:
                tag = "FAIL"
            else:
                tag = "OK"
            print(f"{case['name']:<14} {label:<22} {count:>7}  {tag} {info}", flush=True)
    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
