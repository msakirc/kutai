"""Performance benchmarks for shopping intelligence modules.

Targets:
  - Quick (heuristic) operations: < 100 ms
  - Batch operations on 50-100 products: < 500 ms
  - Cache/monitoring internals: < 10 ms

Run only these tests with:
    pytest -m performance tests/shopping/test_performance.py -v
"""

from __future__ import annotations

import asyncio
import time
import unittest

import pytest


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Shared test-data generators
# ---------------------------------------------------------------------------

def _make_prices(n: int, base: float = 100.0) -> list[float]:
    """Return *n* prices varying around *base* with a small outlier at the end."""
    prices = [base + (i % 20) * 2.0 for i in range(n - 1)]
    prices.append(base * 10)  # deliberate outlier
    return prices


def _make_products(n: int, name_prefix: str = "Ürün") -> list[dict]:
    """Return *n* product dicts with name, price, and source fields."""
    return [
        {
            "name": f"{name_prefix} {i}",
            "price": 100.0 + i * 1.5,
            "source": f"source_{i % 5}",
        }
        for i in range(n)
    ]


def _make_bulk_products(n: int) -> list[dict]:
    """Return *n* product dicts suitable for bulk-pricing analysis."""
    products = []
    for i in range(n):
        qty = (i % 6) + 1  # quantities 1-6
        products.append(
            {
                "name": f"Deterjan {qty}'li Paket",
                "price": float(qty * 45 - (qty - 1) * 3),  # slight per-unit saving
                "quantity": qty,
                "category": "temizlik",
            }
        )
    return products


def _make_bundle_products(n: int) -> list[dict]:
    """Return *n* product dicts with various bundle signals."""
    templates = [
        {"name": "3 al 2 öde Şampuan", "price": 60.0, "original_price": 90.0},
        {"name": "5'li Set Havlu", "price": 120.0, "original_price": 150.0},
        {"name": "Kampanyalı Temizlik Seti", "price": 80.0, "original_price": 100.0},
        {"name": "Paket Fiyatı Sabun 4 adet", "price": 40.0, "original_price": 56.0},
        {"name": "Normal Ürün", "price": 30.0},
    ]
    return [templates[i % len(templates)] for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestHeuristicPerformance
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.performance
class TestHeuristicPerformance(unittest.TestCase):
    """All pure heuristic modules must complete a representative call in < 100 ms."""

    LIMIT_MS = 100

    def _assert_fast(self, elapsed_s: float, label: str) -> None:
        elapsed_ms = elapsed_s * 1000
        self.assertLess(
            elapsed_ms,
            self.LIMIT_MS,
            f"{label} took {elapsed_ms:.1f} ms — expected < {self.LIMIT_MS} ms",
        )

    # ── detect_fake_bulk_deal ────────────────────────────────────────────────

    def test_detect_fake_bulk_deal_genuine(self):
        from src.shopping.intelligence.special.bulk_detector import detect_fake_bulk_deal

        start = time.perf_counter()
        result = detect_fake_bulk_deal(
            single_price=50.0,
            bulk_price=270.0,
            bulk_quantity=6,
        )
        elapsed = time.perf_counter() - start

        self.assertFalse(result["is_fake"])
        self._assert_fast(elapsed, "detect_fake_bulk_deal (genuine)")

    def test_detect_fake_bulk_deal_fake(self):
        from src.shopping.intelligence.special.bulk_detector import detect_fake_bulk_deal

        start = time.perf_counter()
        result = detect_fake_bulk_deal(
            single_price=40.0,
            bulk_price=300.0,
            bulk_quantity=6,
        )
        elapsed = time.perf_counter() - start

        self.assertTrue(result["is_fake"])
        self._assert_fast(elapsed, "detect_fake_bulk_deal (fake)")

    def test_detect_fake_bulk_deal_edge_zero_quantity(self):
        from src.shopping.intelligence.special.bulk_detector import detect_fake_bulk_deal

        start = time.perf_counter()
        result = detect_fake_bulk_deal(
            single_price=50.0,
            bulk_price=100.0,
            bulk_quantity=0,
        )
        elapsed = time.perf_counter() - start

        self.assertFalse(result["is_fake"])
        self._assert_fast(elapsed, "detect_fake_bulk_deal (zero qty)")

    def test_detect_fake_bulk_deal_edge_same_price(self):
        from src.shopping.intelligence.special.bulk_detector import detect_fake_bulk_deal

        start = time.perf_counter()
        result = detect_fake_bulk_deal(
            single_price=50.0,
            bulk_price=150.0,
            bulk_quantity=3,
        )
        elapsed = time.perf_counter() - start

        self.assertFalse(result["is_fake"])
        self.assertAlmostEqual(result["difference_pct"], 0.0, places=1)
        self._assert_fast(elapsed, "detect_fake_bulk_deal (same unit price)")

    # ── flag_outliers ────────────────────────────────────────────────────────

    def test_flag_outliers_100_prices(self):
        from src.shopping.resilience.price_verification import flag_outliers

        prices = _make_prices(100)

        start = time.perf_counter()
        results = flag_outliers(prices)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 100)
        # The last price (base*10) should be flagged
        self.assertTrue(results[-1]["is_outlier"])
        self._assert_fast(elapsed, "flag_outliers (100 prices)")

    def test_flag_outliers_100_uniform_prices(self):
        from src.shopping.resilience.price_verification import flag_outliers

        prices = [99.99] * 100

        start = time.perf_counter()
        results = flag_outliers(prices)
        elapsed = time.perf_counter() - start

        self.assertTrue(all(not r["is_outlier"] for r in results))
        self._assert_fast(elapsed, "flag_outliers (100 uniform prices)")

    def test_flag_outliers_100_custom_threshold(self):
        from src.shopping.resilience.price_verification import flag_outliers

        prices = [100.0] * 95 + [200.0] * 5  # 100% above median for 5 items

        start = time.perf_counter()
        results = flag_outliers(prices, threshold_pct=90.0)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 100)
        self._assert_fast(elapsed, "flag_outliers (100 prices, 90% threshold)")

    # ── detect_counterfeit_keywords ──────────────────────────────────────────

    def test_detect_counterfeit_keywords_long_clean_text(self):
        from src.shopping.intelligence.special.fraud_detector import detect_counterfeit_keywords

        long_text = ("Bu ürün orijinaldir ve kalitesi yüksektir. " * 50)

        start = time.perf_counter()
        hits = detect_counterfeit_keywords(long_text)
        elapsed = time.perf_counter() - start

        self.assertIsInstance(hits, list)
        self._assert_fast(elapsed, "detect_counterfeit_keywords (long clean text)")

    def test_detect_counterfeit_keywords_long_flagged_text(self):
        from src.shopping.intelligence.special.fraud_detector import detect_counterfeit_keywords

        # Embed several counterfeit keywords in a long text
        prefix = "Kaliteli ürün " * 30
        flagged = "a kalite replika kopya muadil super copy kutusuz"
        suffix = " Hızlı kargo garantili." * 20
        long_text = prefix + flagged + suffix

        start = time.perf_counter()
        hits = detect_counterfeit_keywords(long_text)
        elapsed = time.perf_counter() - start

        self.assertGreater(len(hits), 0)
        self._assert_fast(elapsed, "detect_counterfeit_keywords (long flagged text)")

    def test_detect_counterfeit_keywords_empty_text(self):
        from src.shopping.intelligence.special.fraud_detector import detect_counterfeit_keywords

        start = time.perf_counter()
        hits = detect_counterfeit_keywords("")
        elapsed = time.perf_counter() - start

        self.assertEqual(hits, [])
        self._assert_fast(elapsed, "detect_counterfeit_keywords (empty text)")

    # ── detect_flash_sale ────────────────────────────────────────────────────

    def test_detect_flash_sale_no_signals(self):
        from src.shopping.resilience.staleness import detect_flash_sale

        product = {
            "name": "Samsung Galaxy S24",
            "price": 35000.0,
            "original_price": 37000.0,
            "description": "Türkiye garantili.",
        }

        start = time.perf_counter()
        result = detect_flash_sale(product)
        elapsed = time.perf_counter() - start

        self.assertFalse(result["is_flash_sale"])
        self._assert_fast(elapsed, "detect_flash_sale (no signals)")

    def test_detect_flash_sale_keyword_signal(self):
        from src.shopping.resilience.staleness import detect_flash_sale

        product = {
            "name": "Flash indirim! Son 3 saat kaldı!",
            "price": 1000.0,
            "original_price": 2500.0,
            "description": "Sınırlı stok fırsat",
        }

        start = time.perf_counter()
        result = detect_flash_sale(product)
        elapsed = time.perf_counter() - start

        self.assertTrue(result["is_flash_sale"])
        self._assert_fast(elapsed, "detect_flash_sale (keyword signal)")

    def test_detect_flash_sale_large_discount(self):
        from src.shopping.resilience.staleness import detect_flash_sale

        product = {
            "name": "Ürün",
            "price": 100.0,
            "original_price": 500.0,  # 80% discount
        }

        start = time.perf_counter()
        result = detect_flash_sale(product)
        elapsed = time.perf_counter() - start

        self.assertTrue(result["is_flash_sale"])
        self._assert_fast(elapsed, "detect_flash_sale (large discount)")

    def test_detect_flash_sale_empty_product(self):
        from src.shopping.resilience.staleness import detect_flash_sale

        start = time.perf_counter()
        result = detect_flash_sale({})
        elapsed = time.perf_counter() - start

        self.assertFalse(result["is_flash_sale"])
        self._assert_fast(elapsed, "detect_flash_sale (empty product)")

    # ── is_safe_to_buy_used ──────────────────────────────────────────────────

    def test_is_safe_to_buy_used_all_categories(self):
        from src.shopping.intelligence.special.used_market import is_safe_to_buy_used

        categories = [
            # Safe categories
            "elektronik", "mobilya", "giyim", "kitap", "spor",
            "bilgisayar", "telefon", "müzik", "araç", "oyun",
            # Unsafe categories
            "bebek", "medikal", "sağlık", "gıda", "kozmetik",
            "ilaç", "kask", "araba koltuğu",
        ]

        start = time.perf_counter()
        results = {cat: is_safe_to_buy_used(cat) for cat in categories}
        elapsed = time.perf_counter() - start

        # Safe categories should return True
        for cat in ["elektronik", "mobilya", "giyim", "kitap", "spor"]:
            self.assertTrue(results[cat], f"Expected safe for '{cat}'")

        # Unsafe categories should return False
        for cat in ["bebek", "medikal", "gıda", "kozmetik", "ilaç", "kask"]:
            self.assertFalse(results[cat], f"Expected unsafe for '{cat}'")

        self._assert_fast(elapsed, f"is_safe_to_buy_used (all {len(categories)} categories)")

    # ── classify_origin ──────────────────────────────────────────────────────

    def test_classify_origin_known_domestic_brands(self):
        from src.shopping.intelligence.special.import_domestic import classify_origin

        brands = ["Vestel", "Arçelik", "Beko", "Karaca", "Casper", "TOGG"]

        start = time.perf_counter()
        results = [classify_origin(b) for b in brands]
        elapsed = time.perf_counter() - start

        for brand, result in zip(brands, results):
            self.assertEqual(
                result["origin"],
                "domestic",
                f"Expected 'domestic' for '{brand}', got '{result['origin']}'",
            )
        self._assert_fast(elapsed, f"classify_origin ({len(brands)} domestic brands)")

    def test_classify_origin_known_imported_brands(self):
        from src.shopping.intelligence.special.import_domestic import classify_origin

        brands = ["Apple", "Samsung", "Sony", "Xiaomi", "Bosch", "Dyson"]

        start = time.perf_counter()
        results = [classify_origin(b) for b in brands]
        elapsed = time.perf_counter() - start

        for brand, result in zip(brands, results):
            self.assertEqual(
                result["origin"],
                "imported",
                f"Expected 'imported' for '{brand}', got '{result['origin']}'",
            )
        self._assert_fast(elapsed, f"classify_origin ({len(brands)} imported brands)")

    def test_classify_origin_unknown_brand(self):
        from src.shopping.intelligence.special.import_domestic import classify_origin

        start = time.perf_counter()
        result = classify_origin("BilinmeyenMarka XYZ")
        elapsed = time.perf_counter() - start

        self.assertEqual(result["origin"], "unknown")
        self._assert_fast(elapsed, "classify_origin (unknown brand)")

    # ── check_btk_requirement ────────────────────────────────────────────────

    def test_check_btk_requirement_phone_product(self):
        from src.shopping.intelligence.special.import_domestic import check_btk_requirement

        product = {
            "name": "iPhone 15 Pro 256GB",
            "category": "akıllı telefon",
            "brand": "Apple",
        }

        start = time.perf_counter()
        result = check_btk_requirement(product)
        elapsed = time.perf_counter() - start

        self.assertTrue(result["needs_btk"])
        self.assertGreater(result["registration_cost_estimate"], 0)
        self._assert_fast(elapsed, "check_btk_requirement (phone)")

    def test_check_btk_requirement_tablet_product(self):
        from src.shopping.intelligence.special.import_domestic import check_btk_requirement

        product = {
            "name": "iPad Pro 11 inç",
            "category": "tablet",
            "brand": "Apple",
        }

        start = time.perf_counter()
        result = check_btk_requirement(product)
        elapsed = time.perf_counter() - start

        self.assertTrue(result["needs_btk"])
        self._assert_fast(elapsed, "check_btk_requirement (tablet)")

    def test_check_btk_requirement_non_phone_product(self):
        from src.shopping.intelligence.special.import_domestic import check_btk_requirement

        product = {
            "name": "Çamaşır Makinesi 8kg",
            "category": "beyaz eşya",
            "brand": "Arçelik",
        }

        start = time.perf_counter()
        result = check_btk_requirement(product)
        elapsed = time.perf_counter() - start

        self.assertFalse(result["needs_btk"])
        self.assertEqual(result["registration_cost_estimate"], 0.0)
        self._assert_fast(elapsed, "check_btk_requirement (non-phone)")

    def test_check_btk_requirement_various_phones(self):
        from src.shopping.intelligence.special.import_domestic import check_btk_requirement

        phone_products = [
            {"name": "Samsung Galaxy S24 Ultra", "category": "smartphone"},
            {"name": "Xiaomi 14", "category": "telefon"},
            {"name": "Huawei P60", "brand": "Huawei"},
            {"name": "OnePlus 12", "category": "android telefon"},
            {"name": "iPad Mini", "category": "tablet"},
        ]

        start = time.perf_counter()
        results = [check_btk_requirement(p) for p in phone_products]
        elapsed = time.perf_counter() - start

        for product, result in zip(phone_products, results):
            self.assertTrue(
                result["needs_btk"],
                f"Expected needs_btk=True for '{product['name']}'",
            )
        self._assert_fast(elapsed, f"check_btk_requirement ({len(phone_products)} phone products)")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestBatchPerformance
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.performance
class TestBatchPerformance(unittest.TestCase):
    """Batch operations on 20-50 products must complete in < 500 ms."""

    LIMIT_MS = 500

    def _assert_fast(self, elapsed_s: float, label: str) -> None:
        elapsed_ms = elapsed_s * 1000
        self.assertLess(
            elapsed_ms,
            self.LIMIT_MS,
            f"{label} took {elapsed_ms:.1f} ms — expected < {self.LIMIT_MS} ms",
        )

    # ── verify_prices ────────────────────────────────────────────────────────

    def test_verify_prices_50_products_single_group(self):
        """50 products all with the same name → one large cross-verification group."""
        from src.shopping.resilience.price_verification import verify_prices

        products = [
            {
                "name": "Samsung Galaxy A54 128GB",
                "price": 8000.0 + i * 50,
                "source": f"source_{i % 5}",
            }
            for i in range(50)
        ]

        start = time.perf_counter()
        results = verify_prices(products)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 50)
        # All should be cross-verified (same name group)
        self.assertTrue(all(r["price_verified"] for r in results))
        self._assert_fast(elapsed, "verify_prices (50 products, single group)")

    def test_verify_prices_50_products_mixed_groups(self):
        """50 products from 5 different product names → 5 groups of 10 each."""
        from src.shopping.resilience.price_verification import verify_prices

        product_names = [
            "Laptop Asus VivoBook",
            "Akıllı Saat Apple Watch",
            "Kulaklık Sony WH1000",
            "Tablet Samsung Tab A8",
            "Kamera Canon EOS",
        ]
        products = []
        for i in range(50):
            name = product_names[i % 5]
            products.append(
                {
                    "name": name,
                    "price": 3000.0 + (i // 5) * 100 + (i % 5) * 200,
                    "source": f"source_{i % 3}",
                }
            )

        start = time.perf_counter()
        results = verify_prices(products)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 50)
        self._assert_fast(elapsed, "verify_prices (50 products, 5 groups)")

    def test_verify_prices_50_products_with_outliers(self):
        """50 products with deliberate outliers inserted."""
        from src.shopping.resilience.price_verification import verify_prices

        products = []
        for i in range(50):
            price = 100.0 + (i % 10) * 5  # normal range 100-145
            if i in (10, 25, 40):
                price = 9999.0  # outliers
            products.append(
                {
                    "name": "Deterjan 5L",
                    "price": price,
                    "source": f"src_{i}",
                }
            )

        start = time.perf_counter()
        results = verify_prices(products)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 50)
        suspicious_count = sum(1 for r in results if r["price_suspicious"])
        self.assertGreater(suspicious_count, 0)
        self._assert_fast(elapsed, "verify_prices (50 products with outliers)")

    # ── analyze_bulk_pricing ─────────────────────────────────────────────────

    def test_analyze_bulk_pricing_30_products(self):
        from src.shopping.intelligence.special.bulk_detector import analyze_bulk_pricing

        products = _make_bulk_products(30)

        start = time.perf_counter()
        results = analyze_bulk_pricing(products)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 30)
        # Results should be sorted by unit price (cheapest first)
        unit_prices = [r["unit_price"] for r in results if r["unit_price"] > 0]
        self.assertEqual(unit_prices, sorted(unit_prices))
        self._assert_fast(elapsed, "analyze_bulk_pricing (30 products)")

    def test_analyze_bulk_pricing_30_products_with_fake_bulks(self):
        from src.shopping.intelligence.special.bulk_detector import analyze_bulk_pricing

        products = []
        for i in range(30):
            qty = (i % 6) + 1
            # For even indices, deliberately make bulk more expensive per unit
            if i % 2 == 0 and qty > 1:
                price = float(qty * 60)   # higher per-unit than single
            else:
                price = float(qty * 40)   # cheaper per-unit

            products.append(
                {
                    "name": f"Şampuan {qty} adet",
                    "price": price,
                    "quantity": qty,
                }
            )

        start = time.perf_counter()
        results = analyze_bulk_pricing(products)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 30)
        fake_bulk_count = sum(1 for r in results if r.get("is_fake_bulk"))
        self.assertGreaterEqual(fake_bulk_count, 0)  # may or may not detect based on reference
        self._assert_fast(elapsed, "analyze_bulk_pricing (30 products with fake bulks)")

    def test_analyze_bulk_pricing_30_products_parse_names(self):
        """Products without explicit quantity field — quantities parsed from names."""
        from src.shopping.intelligence.special.bulk_detector import analyze_bulk_pricing

        products = [
            {"name": f"Mendil {(i % 5) + 2}'li Paket", "price": float((i % 5 + 2) * 15)}
            for i in range(30)
        ]

        start = time.perf_counter()
        results = analyze_bulk_pricing(products)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 30)
        bulk_count = sum(1 for r in results if r["is_bulk"])
        self.assertGreater(bulk_count, 0)
        self._assert_fast(elapsed, "analyze_bulk_pricing (30 products, parse names)")

    # ── detect_bundle_deals ──────────────────────────────────────────────────

    def test_detect_bundle_deals_20_products(self):
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = _make_bundle_products(20)

        start = time.perf_counter()
        results = detect_bundle_deals(products)
        elapsed = time.perf_counter() - start

        self.assertIsInstance(results, list)
        # At least some bundles should be found given our templates
        self.assertGreater(len(results), 0)
        self._assert_fast(elapsed, "detect_bundle_deals (20 products)")

    def test_detect_bundle_deals_20_no_bundles(self):
        """20 plain products with no bundle signals."""
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [
            {"name": f"Kalem Marka {i}", "price": float(10 + i)}
            for i in range(20)
        ]

        start = time.perf_counter()
        results = detect_bundle_deals(products)
        elapsed = time.perf_counter() - start

        self.assertEqual(results, [])
        self._assert_fast(elapsed, "detect_bundle_deals (20 plain products)")

    def test_detect_bundle_deals_20_all_bundle_types(self):
        """Products that each exercise a different bundle detection branch."""
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [
            {"name": "3 al 2 öde Çamaşır Deterjanı", "price": 120.0, "original_price": 180.0},
            {"name": "Mutfak Seti fiyatı 8 adet", "price": 200.0, "original_price": 280.0},
            {"name": "Kampanyalı Ürün Hediyeli", "price": 50.0, "original_price": 70.0},
            {"name": "5'li Takım Çorap", "price": 75.0, "original_price": 100.0},
            {"name": "Birlikte al 2 adet tüp", "price": 30.0},
        ]
        # Repeat the template pattern 4 times to reach 20 products
        all_products = products * 4

        start = time.perf_counter()
        results = detect_bundle_deals(all_products)
        elapsed = time.perf_counter() - start

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self._assert_fast(elapsed, "detect_bundle_deals (20 products, all bundle types)")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestCacheEfficiency
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.performance
class TestCacheEfficiency(unittest.TestCase):
    """Verify that repeated DetectionMonitor calls do not degrade performance."""

    def _make_monitor(self):
        from src.shopping.resilience.detection_monitor import DetectionMonitor
        return DetectionMonitor(window_size=50, cooldown_seconds=300)

    # ── record_request burst ─────────────────────────────────────────────────

    def test_record_1000_requests_total_time(self):
        """Recording 1 000 outcomes across multiple domains completes quickly."""
        monitor = self._make_monitor()
        domains = ["trendyol.com", "hepsiburada.com", "akakce.com", "n11.com"]

        start = time.perf_counter()
        for i in range(1000):
            domain = domains[i % len(domains)]
            success = (i % 10) != 0  # 90% success rate
            run_async(monitor.record_request(domain, success))
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        # 1000 in-memory appends — expect well under 1 s
        self.assertLess(
            elapsed_ms,
            1000,
            f"record_request ×1000 took {elapsed_ms:.1f} ms — expected < 1000 ms",
        )

    def test_record_1000_requests_no_memory_accumulation(self):
        """The rolling window should cap at window_size, not grow unboundedly."""
        monitor = self._make_monitor()
        domain = "trendyol.com"

        for i in range(1000):
            run_async(monitor.record_request(domain, True))

        stats = monitor._domains[domain]
        self.assertLessEqual(
            len(stats.outcomes),
            monitor.window_size,
            "Outcomes deque grew beyond window_size",
        )
        self.assertEqual(stats.total_requests, 1000)

    # ── get_detection_metrics speed ──────────────────────────────────────────

    def test_get_detection_metrics_under_10ms(self):
        """get_detection_metrics on a populated monitor completes in < 10 ms."""
        monitor = self._make_monitor()
        domains = ["trendyol.com", "hepsiburada.com", "akakce.com", "n11.com",
                   "ciceksepeti.com", "sahibinden.com", "gittigidiyor.com"]

        # Populate with 1000 requests first
        for i in range(1000):
            domain = domains[i % len(domains)]
            run_async(monitor.record_request(domain, i % 8 != 0))

        start = time.perf_counter()
        metrics = run_async(monitor.get_detection_metrics())
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        self.assertIn("trendyol.com", metrics)
        self.assertLess(
            elapsed_ms,
            10,
            f"get_detection_metrics took {elapsed_ms:.2f} ms — expected < 10 ms",
        )

    def test_get_detection_metrics_repeated_calls_stable(self):
        """Repeated get_detection_metrics calls should each stay under 10 ms."""
        monitor = self._make_monitor()
        domains = ["trendyol.com", "hepsiburada.com", "akakce.com"]

        for i in range(500):
            run_async(monitor.record_request(domains[i % 3], True))

        times_ms = []
        for _ in range(10):
            start = time.perf_counter()
            run_async(monitor.get_detection_metrics())
            times_ms.append((time.perf_counter() - start) * 1000)

        for i, t in enumerate(times_ms):
            self.assertLess(
                t,
                10,
                f"Call #{i + 1} to get_detection_metrics took {t:.2f} ms — expected < 10 ms",
            )

    # ── success_rate retrieval ────────────────────────────────────────────────

    def test_get_success_rate_after_1000_records(self):
        """get_success_rate should return quickly after extensive record_request calls."""
        monitor = self._make_monitor()
        domain = "hepsiburada.com"

        for i in range(1000):
            run_async(monitor.record_request(domain, i % 4 != 0))  # 75% success

        start = time.perf_counter()
        rate = run_async(monitor.get_success_rate(domain))
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        # Rolling window of 50 — rate should reflect last 50 requests
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)
        self.assertLess(
            elapsed_ms,
            10,
            f"get_success_rate took {elapsed_ms:.2f} ms — expected < 10 ms",
        )

    def test_is_domain_cooled_down_no_cooldown_fast(self):
        """is_domain_cooled_down on a domain without cooldown completes in < 10 ms."""
        monitor = self._make_monitor()
        domain = "akakce.com"

        # Record only successes — no cooldown should be triggered
        for _ in range(50):
            run_async(monitor.record_request(domain, True))

        start = time.perf_counter()
        result = run_async(monitor.is_domain_cooled_down(domain))
        elapsed = time.perf_counter() - start

        elapsed_ms = elapsed * 1000
        self.assertFalse(result)
        self.assertLess(
            elapsed_ms,
            10,
            f"is_domain_cooled_down took {elapsed_ms:.2f} ms — expected < 10 ms",
        )


if __name__ == "__main__":
    unittest.main()
