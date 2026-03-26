"""Comprehensive tests for shopping Phase 0 modules:
models, text_utils, config, cache, and request_tracker.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from src.shopping.models import (
    Combo,
    PriceHistoryEntry,
    Product,
    ProductMatch,
    Review,
    ShoppingQuery,
    ShoppingSession,
    UserConstraint,
)
from src.shopping.text_utils import (
    detect_language,
    extract_capacity,
    extract_dimensions,
    extract_energy_rating,
    extract_material,
    extract_volume_weight_for_grocery,
    extract_weight,
    generate_search_variants,
    normalize_product_name,
    normalize_turkish,
    parse_turkish_price,
)
from src.shopping.config import get_config, get_rate_limit, load_config


# ---------------------------------------------------------------------------
# Async test helper
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestModels
# ═══════════════════════════════════════════════════════════════════════════

class TestModels(unittest.TestCase):
    """Test that all dataclasses can be instantiated, defaults are correct,
    and required fields are enforced."""

    # -- Product ----------------------------------------------------------

    def test_product_required_fields(self):
        p = Product(name="Test", url="https://example.com", source="trendyol")
        self.assertEqual(p.name, "Test")
        self.assertEqual(p.url, "https://example.com")
        self.assertEqual(p.source, "trendyol")

    def test_product_defaults(self):
        p = Product(name="X", url="u", source="s")
        self.assertIsNone(p.original_price)
        self.assertIsNone(p.discounted_price)
        self.assertIsNone(p.discount_percentage)
        self.assertEqual(p.currency, "TRY")
        self.assertIsNone(p.image_url)
        self.assertEqual(p.specs, {})
        self.assertIsNone(p.rating)
        self.assertIsNone(p.review_count)
        self.assertEqual(p.availability, "in_stock")
        self.assertIsNone(p.seller_name)
        self.assertIsNone(p.seller_rating)
        self.assertIsNone(p.seller_review_count)
        self.assertIsNone(p.shipping_cost)
        self.assertIsNone(p.shipping_time_days)
        self.assertFalse(p.free_shipping)
        self.assertIsNone(p.installment_info)
        self.assertIsNone(p.warranty_months)
        self.assertIsNone(p.category_path)
        self.assertIsNone(p.fetched_at)

    def test_product_all_fields(self):
        p = Product(
            name="Laptop",
            url="https://example.com/laptop",
            source="hepsiburada",
            original_price=15999.99,
            discounted_price=12999.99,
            discount_percentage=18.75,
            currency="TRY",
            image_url="https://img.example.com/laptop.jpg",
            specs={"ram": "16GB"},
            rating=4.5,
            review_count=120,
            availability="low_stock",
            seller_name="TechShop",
            seller_rating=4.8,
            seller_review_count=500,
            shipping_cost=0.0,
            shipping_time_days=2,
            free_shipping=True,
            installment_info={"months": 6, "rate": 0.0},
            warranty_months=24,
            category_path="Elektronik > Bilgisayar > Laptop",
            fetched_at="2026-03-26T12:00:00Z",
        )
        self.assertEqual(p.original_price, 15999.99)
        self.assertEqual(p.specs["ram"], "16GB")
        self.assertTrue(p.free_shipping)

    def test_product_missing_required_raises(self):
        with self.assertRaises(TypeError):
            Product()  # type: ignore[call-arg]

    # -- Review -----------------------------------------------------------

    def test_review_required_fields(self):
        r = Review(text="Great product", source="trendyol")
        self.assertEqual(r.text, "Great product")
        self.assertEqual(r.source, "trendyol")

    def test_review_defaults(self):
        r = Review(text="ok", source="s")
        self.assertIsNone(r.rating)
        self.assertIsNone(r.date)
        self.assertIsNone(r.author)
        self.assertFalse(r.verified_purchase)
        self.assertEqual(r.helpful_count, 0)
        self.assertIsNone(r.language)

    def test_review_missing_required_raises(self):
        with self.assertRaises(TypeError):
            Review()  # type: ignore[call-arg]

    # -- ProductMatch -----------------------------------------------------

    def test_product_match_defaults(self):
        pm = ProductMatch()
        self.assertEqual(pm.products, [])
        self.assertEqual(pm.canonical_name, "")
        self.assertEqual(pm.canonical_specs, {})
        self.assertEqual(pm.confidence_score, 0.0)

    def test_product_match_independent_lists(self):
        """Ensure default factory creates independent lists."""
        pm1 = ProductMatch()
        pm2 = ProductMatch()
        pm1.products.append(Product(name="A", url="u", source="s"))
        self.assertEqual(len(pm2.products), 0)

    # -- PriceHistoryEntry ------------------------------------------------

    def test_price_history_entry_defaults(self):
        phe = PriceHistoryEntry()
        self.assertEqual(phe.price, 0.0)
        self.assertEqual(phe.source, "")
        self.assertEqual(phe.date, "")
        self.assertFalse(phe.was_campaign)

    # -- ShoppingQuery ----------------------------------------------------

    def test_shopping_query_defaults(self):
        sq = ShoppingQuery()
        self.assertEqual(sq.raw_query, "")
        self.assertIsNone(sq.interpreted_intent)
        self.assertEqual(sq.constraints, [])
        self.assertIsNone(sq.budget)
        self.assertIsNone(sq.category)
        self.assertEqual(sq.generated_searches, [])

    # -- UserConstraint ---------------------------------------------------

    def test_user_constraint_defaults(self):
        uc = UserConstraint()
        self.assertEqual(uc.type, "")
        self.assertEqual(uc.value, "")
        self.assertEqual(uc.hard_or_soft, "hard")
        self.assertEqual(uc.source, "user-stated")

    # -- Combo ------------------------------------------------------------

    def test_combo_defaults(self):
        c = Combo()
        self.assertEqual(c.products, [])
        self.assertEqual(c.total_price, 0.0)
        self.assertEqual(c.compatibility_notes, [])
        self.assertEqual(c.value_score, 0.0)

    # -- ShoppingSession --------------------------------------------------

    def test_shopping_session_defaults(self):
        ss = ShoppingSession()
        self.assertEqual(ss.session_id, "")
        self.assertEqual(ss.user_query, "")
        self.assertIsNone(ss.analyzed_intent)
        self.assertEqual(ss.products_found, [])
        self.assertEqual(ss.recommendations_made, [])
        self.assertEqual(ss.user_actions, [])
        self.assertEqual(ss.timestamps, {})


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestTurkishTextUtils
# ═══════════════════════════════════════════════════════════════════════════

class TestTurkishTextUtils(unittest.TestCase):

    # -- parse_turkish_price ----------------------------------------------

    def test_parse_turkish_price_standard(self):
        self.assertAlmostEqual(parse_turkish_price("1.299,99 TL"), 1299.99)

    def test_parse_turkish_price_lira_symbol(self):
        self.assertAlmostEqual(parse_turkish_price("₺1299.99"), 1299.99)

    def test_parse_turkish_price_comma_decimal(self):
        self.assertAlmostEqual(parse_turkish_price("1299,99"), 1299.99)

    def test_parse_turkish_price_thousands_no_decimal(self):
        self.assertAlmostEqual(parse_turkish_price("1.299 TL"), 1299.0)

    def test_parse_turkish_price_invalid(self):
        self.assertIsNone(parse_turkish_price("not a price"))

    def test_parse_turkish_price_empty(self):
        self.assertIsNone(parse_turkish_price(""))

    def test_parse_turkish_price_none_input(self):
        self.assertIsNone(parse_turkish_price(None))  # type: ignore[arg-type]

    def test_parse_turkish_price_tl_prefix(self):
        self.assertAlmostEqual(parse_turkish_price("TL 1.299,99"), 1299.99)

    # -- normalize_turkish ------------------------------------------------

    def test_normalize_turkish_istanbul(self):
        self.assertEqual(normalize_turkish("İSTANBUL"), "istanbul")

    def test_normalize_turkish_isik(self):
        self.assertEqual(normalize_turkish("IŞIK"), "ışık")

    def test_normalize_turkish_empty(self):
        self.assertEqual(normalize_turkish(""), "")

    def test_normalize_turkish_already_lower(self):
        self.assertEqual(normalize_turkish("merhaba"), "merhaba")

    def test_normalize_turkish_mixed(self):
        self.assertEqual(normalize_turkish("GÜNEŞ"), "güneş")

    # -- generate_search_variants -----------------------------------------

    def test_generate_search_variants_ram(self):
        variants = generate_search_variants("RAM")
        lower_variants = [v.lower() for v in variants]
        self.assertIn("ram", lower_variants)
        self.assertIn("bellek", lower_variants)

    def test_generate_search_variants_washing_machine(self):
        variants = generate_search_variants("çamaşır makinesi")
        self.assertIn("çamaşır makinesi", variants)
        self.assertIn("washing machine", variants)

    def test_generate_search_variants_english_to_turkish(self):
        variants = generate_search_variants("keyboard")
        lower_variants = [v.lower() for v in variants]
        self.assertIn("keyboard", lower_variants)
        self.assertIn("klavye", lower_variants)

    def test_generate_search_variants_empty(self):
        self.assertEqual(generate_search_variants(""), [""])

    def test_generate_search_variants_no_match(self):
        variants = generate_search_variants("bilinmeyen ürün")
        self.assertEqual(variants, ["bilinmeyen ürün"])

    # -- extract_dimensions -----------------------------------------------

    def test_extract_dimensions_wxdxh(self):
        dims = extract_dimensions("60x55x45 cm")
        self.assertAlmostEqual(dims["width"], 60.0)
        self.assertAlmostEqual(dims["depth"], 55.0)
        self.assertAlmostEqual(dims["height"], 45.0)

    def test_extract_dimensions_mm(self):
        dims = extract_dimensions("600x550x450 mm")
        self.assertAlmostEqual(dims["width"], 60.0)
        self.assertAlmostEqual(dims["depth"], 55.0)
        self.assertAlmostEqual(dims["height"], 45.0)

    def test_extract_dimensions_turkish_labels(self):
        dims = extract_dimensions("genişlik: 60 cm derinlik: 55 cm yükseklik: 45 cm")
        self.assertAlmostEqual(dims["width"], 60.0)
        self.assertAlmostEqual(dims["depth"], 55.0)
        self.assertAlmostEqual(dims["height"], 45.0)

    def test_extract_dimensions_empty(self):
        self.assertEqual(extract_dimensions(""), {})

    def test_extract_dimensions_no_match(self):
        self.assertEqual(extract_dimensions("no dimensions here"), {})

    # -- extract_weight ---------------------------------------------------

    def test_extract_weight_kg(self):
        self.assertAlmostEqual(extract_weight("5 kg"), 5.0)

    def test_extract_weight_grams(self):
        self.assertAlmostEqual(extract_weight("500 g"), 0.5)

    def test_extract_weight_grams_gr(self):
        self.assertAlmostEqual(extract_weight("250 gr"), 0.25)

    def test_extract_weight_decimal(self):
        self.assertAlmostEqual(extract_weight("2,5 kg"), 2.5)

    def test_extract_weight_empty(self):
        self.assertIsNone(extract_weight(""))

    def test_extract_weight_no_match(self):
        self.assertIsNone(extract_weight("no weight"))

    # -- extract_capacity -------------------------------------------------

    def test_extract_capacity_kg(self):
        result = extract_capacity("9 kg")
        self.assertEqual(result["value"], 9.0)
        self.assertEqual(result["unit"], "kg")

    def test_extract_capacity_litre(self):
        result = extract_capacity("500 litre")
        self.assertEqual(result["value"], 500.0)
        self.assertEqual(result["unit"], "litre")

    def test_extract_capacity_fincan(self):
        result = extract_capacity("12 fincan")
        self.assertEqual(result["value"], 12.0)
        self.assertEqual(result["unit"], "fincan")

    def test_extract_capacity_empty(self):
        self.assertEqual(extract_capacity(""), {})

    def test_extract_capacity_context(self):
        result = extract_capacity("9 kg yıkama kapasitesi")
        self.assertEqual(result["value"], 9.0)
        self.assertEqual(result["unit"], "kg")

    # -- extract_energy_rating --------------------------------------------

    def test_extract_energy_rating_a_plus_plus_plus(self):
        self.assertEqual(extract_energy_rating("A+++"), "A+++")

    def test_extract_energy_rating_b_sinifi(self):
        self.assertEqual(extract_energy_rating("B sınıfı"), "B")

    def test_extract_energy_rating_best_in_text(self):
        # Should return the best rating found
        self.assertEqual(extract_energy_rating("B enerji A++ sınıfı"), "A++")

    def test_extract_energy_rating_empty(self):
        self.assertIsNone(extract_energy_rating(""))

    def test_extract_energy_rating_no_match(self):
        self.assertIsNone(extract_energy_rating("no rating"))

    # -- extract_material -------------------------------------------------

    def test_extract_material_paslanmaz_celik(self):
        self.assertEqual(extract_material("paslanmaz çelik gövde"), "paslanmaz çelik")

    def test_extract_material_stainless_steel(self):
        self.assertEqual(extract_material("stainless steel body"), "paslanmaz çelik")

    def test_extract_material_cam(self):
        self.assertEqual(extract_material("cam kapak"), "cam")

    def test_extract_material_empty(self):
        self.assertIsNone(extract_material(""))

    def test_extract_material_no_match(self):
        self.assertIsNone(extract_material("unknown material xyz"))

    # -- extract_volume_weight_for_grocery --------------------------------

    def test_extract_grocery_kg(self):
        result = extract_volume_weight_for_grocery("1 kg")
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], 1.0)
        self.assertEqual(result["unit"], "kg")
        self.assertAlmostEqual(result["per_kg_or_liter"], 1.0)

    def test_extract_grocery_ml(self):
        result = extract_volume_weight_for_grocery("500 ml")
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], 500.0)
        self.assertEqual(result["unit"], "ml")
        self.assertAlmostEqual(result["per_kg_or_liter"], 0.5)

    def test_extract_grocery_gr(self):
        result = extract_volume_weight_for_grocery("250 gr")
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], 250.0)
        self.assertEqual(result["unit"], "gr")
        self.assertAlmostEqual(result["per_kg_or_liter"], 0.25)

    def test_extract_grocery_pack(self):
        result = extract_volume_weight_for_grocery("6'lı paket")
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], 6.0)
        self.assertEqual(result["unit"], "paket")

    def test_extract_grocery_empty(self):
        self.assertIsNone(extract_volume_weight_for_grocery(""))

    def test_extract_grocery_no_match(self):
        self.assertIsNone(extract_volume_weight_for_grocery("no grocery info"))

    # -- detect_language --------------------------------------------------

    def test_detect_language_turkish(self):
        self.assertEqual(detect_language("merhaba dünya"), "tr")

    def test_detect_language_english(self):
        self.assertEqual(detect_language("hello world"), "en")

    def test_detect_language_turkish_chars(self):
        self.assertEqual(detect_language("şeker"), "tr")
        self.assertEqual(detect_language("çay"), "tr")
        self.assertEqual(detect_language("ığdır"), "tr")

    def test_detect_language_empty(self):
        self.assertEqual(detect_language(""), "en")

    # -- normalize_product_name -------------------------------------------

    def test_normalize_product_name_strips_super_firsat(self):
        result = normalize_product_name("Samsung TV süper fırsat")
        self.assertNotIn("süper fırsat", result)
        self.assertIn("Samsung TV", result)

    def test_normalize_product_name_strips_kampanyali(self):
        result = normalize_product_name("kampanyalı Arçelik Buzdolabı")
        self.assertNotIn("kampanyalı", result)
        self.assertIn("Arçelik Buzdolabı", result)

    def test_normalize_product_name_strips_ucretsiz_kargo(self):
        result = normalize_product_name("Tefal Tava ücretsiz kargo")
        self.assertNotIn("ücretsiz kargo", result)
        self.assertIn("Tefal Tava", result)

    def test_normalize_product_name_strips_multiple(self):
        result = normalize_product_name("kampanyalı Samsung TV süper fırsat hızlı kargo")
        self.assertNotIn("kampanyalı", result)
        self.assertNotIn("süper fırsat", result)
        self.assertNotIn("hızlı kargo", result)
        self.assertIn("Samsung TV", result)

    def test_normalize_product_name_collapses_whitespace(self):
        result = normalize_product_name("Samsung   TV   55 inch")
        self.assertNotIn("  ", result)

    def test_normalize_product_name_empty(self):
        self.assertEqual(normalize_product_name(""), "")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):

    def setUp(self):
        # Reset the singleton so each test gets a fresh config
        import src.shopping.config as cfg_mod
        cfg_mod._config = None

    def test_get_config_returns_dict(self):
        config = get_config()
        self.assertIsInstance(config, dict)

    def test_get_config_has_expected_keys(self):
        config = get_config()
        expected_keys = {"rate_limits", "cache_ttl", "execution", "llm", "features", "user_defaults"}
        self.assertTrue(expected_keys.issubset(config.keys()))

    def test_get_config_rate_limits_has_domains(self):
        config = get_config()
        domains = config["rate_limits"]
        self.assertIn("akakce", domains)
        self.assertIn("trendyol", domains)
        self.assertIn("hepsiburada", domains)

    def test_get_rate_limit_known_domain(self):
        rl = get_rate_limit("akakce")
        self.assertIsInstance(rl, dict)
        self.assertIn("delay_seconds", rl)
        self.assertIn("daily_budget", rl)
        self.assertEqual(rl["delay_seconds"], 10)
        self.assertEqual(rl["daily_budget"], 200)

    def test_get_rate_limit_trendyol(self):
        rl = get_rate_limit("trendyol")
        self.assertEqual(rl["delay_seconds"], 5)
        self.assertEqual(rl["daily_budget"], 100)

    def test_unknown_domain_returns_defaults(self):
        rl = get_rate_limit("unknown_domain_xyz")
        self.assertIsInstance(rl, dict)
        self.assertEqual(rl["delay_seconds"], 10)
        self.assertEqual(rl["daily_budget"], 50)

    def test_get_config_singleton(self):
        """get_config should return the same object on repeated calls."""
        c1 = get_config()
        c2 = get_config()
        self.assertIs(c1, c2)


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestCache (async)
# ═══════════════════════════════════════════════════════════════════════════

class TestCache(unittest.TestCase):
    """Async cache tests using a temp DB file."""

    def setUp(self):
        import src.shopping.cache as cache_mod
        self._cache_mod = cache_mod
        # Reset singleton connection
        cache_mod._cache_db = None

        # Create a temp file for the DB
        self._tmpdir = tempfile.mkdtemp()
        self._tmp_db = os.path.join(self._tmpdir, "test_cache.db")

        # Patch CACHE_DB_PATH
        self._patcher = patch.object(cache_mod, "CACHE_DB_PATH", self._tmp_db)
        self._patcher.start()

    def tearDown(self):
        # Close the DB connection
        run_async(self._close_db())
        self._patcher.stop()

        # Clean up temp files
        try:
            for f in os.listdir(self._tmpdir):
                os.remove(os.path.join(self._tmpdir, f))
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    async def _close_db(self):
        from src.shopping.cache import close_cache_db
        await close_cache_db()

    def test_cache_product_and_retrieve(self):
        from src.shopping.cache import (
            cache_product,
            get_cached_product,
            init_cache_db,
        )

        async def _test():
            await init_cache_db()
            product = {"name": "Test Product", "price": 99.99}
            await cache_product("https://example.com/product1", product, "trendyol")
            result = await get_cached_product("https://example.com/product1")
            self.assertIsNotNone(result)
            self.assertEqual(result["name"], "Test Product")
            self.assertEqual(result["price"], 99.99)

        run_async(_test())

    def test_expired_product_returns_none(self):
        from src.shopping.cache import (
            get_cached_product,
            get_cache_db,
            init_cache_db,
            _hash,
        )

        async def _test():
            await init_cache_db()
            db = await get_cache_db()
            # Insert a product with a very old timestamp
            import json
            product = {"name": "Old Product"}
            old_time = time.time() - (31 * 86400)  # 31 days ago, exceeds specs TTL
            await db.execute(
                "INSERT OR REPLACE INTO products (url_hash, product_json, source, fetched_at, ttl_category) VALUES (?, ?, ?, ?, ?)",
                (_hash("https://example.com/old"), json.dumps(product), "test", old_time, "specs"),
            )
            await db.commit()

            result = await get_cached_product("https://example.com/old")
            self.assertIsNone(result)

        run_async(_test())

    def test_cache_search_and_retrieve(self):
        from src.shopping.cache import (
            cache_search,
            get_cached_search,
            init_cache_db,
        )

        async def _test():
            await init_cache_db()
            results = [{"name": "Product A"}, {"name": "Product B"}]
            await cache_search("laptop", "google_cse", results)
            cached = await get_cached_search("laptop", "google_cse")
            self.assertIsNotNone(cached)
            self.assertEqual(len(cached), 2)
            self.assertEqual(cached[0]["name"], "Product A")

            # Different source should return None
            cached_other = await get_cached_search("laptop", "trendyol")
            self.assertIsNone(cached_other)

        run_async(_test())

    def test_price_history(self):
        from src.shopping.cache import (
            add_price_history,
            get_price_history,
            init_cache_db,
        )

        async def _test():
            await init_cache_db()
            url = "https://example.com/product-ph"
            await add_price_history(url, 100.0, "akakce")
            await add_price_history(url, 95.0, "trendyol")
            await add_price_history(url, 90.0, "akakce")

            history = await get_price_history(url)
            self.assertEqual(len(history), 3)
            # Should be ordered by observed_at (oldest first)
            self.assertEqual(history[0]["price"], 100.0)
            self.assertEqual(history[1]["price"], 95.0)
            self.assertEqual(history[2]["price"], 90.0)
            self.assertEqual(history[0]["source"], "akakce")

        run_async(_test())

    def test_cleanup_expired(self):
        from src.shopping.cache import (
            cleanup_expired,
            get_cache_db,
            init_cache_db,
            _hash,
        )

        async def _test():
            await init_cache_db()
            db = await get_cache_db()
            import json

            # Insert an expired product (31 days old, specs TTL is 30 days)
            old_time = time.time() - (31 * 86400)
            await db.execute(
                "INSERT INTO products (url_hash, product_json, source, fetched_at, ttl_category) VALUES (?, ?, ?, ?, ?)",
                (_hash("https://expired.com"), json.dumps({"x": 1}), "test", old_time, "specs"),
            )
            # Insert a fresh product
            await db.execute(
                "INSERT INTO products (url_hash, product_json, source, fetched_at, ttl_category) VALUES (?, ?, ?, ?, ?)",
                (_hash("https://fresh.com"), json.dumps({"x": 2}), "test", time.time(), "specs"),
            )

            # Insert an expired search (13 hours old, search TTL is 12 hours)
            old_search_time = time.time() - (13 * 3600)
            await db.execute(
                "INSERT INTO search_cache (query_hash, result_json, source, searched_at) VALUES (?, ?, ?, ?)",
                (_hash("old_query"), json.dumps([]), "test", old_search_time),
            )

            await db.commit()

            await cleanup_expired()

            # Expired product should be gone
            cursor = await db.execute(
                "SELECT COUNT(*) FROM products WHERE url_hash = ?",
                (_hash("https://expired.com"),),
            )
            row = await cursor.fetchone()
            self.assertEqual(row[0], 0)

            # Fresh product should remain
            cursor = await db.execute(
                "SELECT COUNT(*) FROM products WHERE url_hash = ?",
                (_hash("https://fresh.com"),),
            )
            row = await cursor.fetchone()
            self.assertEqual(row[0], 1)

            # Expired search should be gone
            cursor = await db.execute(
                "SELECT COUNT(*) FROM search_cache WHERE query_hash = ?",
                (_hash("old_query"),),
            )
            row = await cursor.fetchone()
            self.assertEqual(row[0], 0)

        run_async(_test())


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestRequestTracker (async)
# ═══════════════════════════════════════════════════════════════════════════

class TestRequestTracker(unittest.TestCase):
    """Async request tracker tests using a temp DB file."""

    def setUp(self):
        import src.shopping.cache as cache_mod
        self._cache_mod = cache_mod
        cache_mod._cache_db = None

        self._tmpdir = tempfile.mkdtemp()
        self._tmp_db = os.path.join(self._tmpdir, "test_tracker.db")
        self._patcher = patch.object(cache_mod, "CACHE_DB_PATH", self._tmp_db)
        self._patcher.start()

    def tearDown(self):
        run_async(self._close_db())
        self._patcher.stop()

        try:
            for f in os.listdir(self._tmpdir):
                os.remove(os.path.join(self._tmpdir, f))
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    async def _close_db(self):
        from src.shopping.cache import close_cache_db
        await close_cache_db()

    def test_log_request(self):
        from src.shopping.cache import init_cache_db, get_cache_db
        from src.shopping.request_tracker import init_request_db, log_request

        async def _test():
            await init_cache_db()
            await init_request_db()

            await log_request(
                domain="trendyol",
                url="https://www.trendyol.com/product/123",
                status_code=200,
                response_time_ms=350,
                cache_hit=False,
                scraper_used="httpx",
                session_id="sess-001",
            )

            db = await get_cache_db()
            cursor = await db.execute("SELECT * FROM request_log WHERE domain = 'trendyol'")
            rows = await cursor.fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status_code"], 200)
            self.assertEqual(rows[0]["response_time_ms"], 350)
            self.assertEqual(rows[0]["cache_hit"], 0)  # stored as int

        run_async(_test())

    def test_domain_health_tracking(self):
        from src.shopping.cache import init_cache_db
        from src.shopping.request_tracker import (
            init_request_db,
            log_request,
            get_domain_health,
        )

        async def _test():
            await init_cache_db()
            await init_request_db()

            # Log some successful requests
            for _ in range(3):
                await log_request("akakce", "https://akakce.com/x", 200, 100, False)

            # Log a failed request
            await log_request("akakce", "https://akakce.com/y", 500, 200, False)

            health = await get_domain_health("akakce")
            self.assertEqual(health["success_count_24h"], 3)
            self.assertEqual(health["failure_count_24h"], 1)
            self.assertIsNotNone(health["last_success"])
            self.assertIsNotNone(health["last_failure"])
            self.assertIn(health["current_status"], ("healthy", "degraded"))

        run_async(_test())

    def test_daily_request_count(self):
        from src.shopping.cache import init_cache_db
        from src.shopping.request_tracker import (
            init_request_db,
            log_request,
            get_daily_request_count,
        )

        async def _test():
            await init_cache_db()
            await init_request_db()

            # Log 5 non-cache requests
            for i in range(5):
                await log_request("hepsiburada", f"https://hepsiburada.com/{i}", 200, 100, False)

            # Log 2 cache-hit requests (should not count)
            for i in range(2):
                await log_request("hepsiburada", f"https://hepsiburada.com/cached/{i}", 200, 10, True)

            count = await get_daily_request_count("hepsiburada")
            self.assertEqual(count, 5)

        run_async(_test())

    def test_rate_limit_info(self):
        from src.shopping.cache import init_cache_db
        from src.shopping.request_tracker import (
            init_request_db,
            log_request,
            get_rate_limit_info,
        )

        async def _test():
            await init_cache_db()
            await init_request_db()

            # Log 10 requests to trendyol (daily_budget = 100)
            for i in range(10):
                await log_request("trendyol", f"https://trendyol.com/{i}", 200, 100, False)

            info = await get_rate_limit_info("trendyol")
            self.assertEqual(info["used"], 10)
            self.assertEqual(info["limit"], 100)
            self.assertEqual(info["remaining"], 90)

        run_async(_test())

    def test_rate_limit_info_unknown_domain(self):
        from src.shopping.cache import init_cache_db
        from src.shopping.request_tracker import init_request_db, get_rate_limit_info

        async def _test():
            await init_cache_db()
            await init_request_db()

            info = await get_rate_limit_info("unknown_domain")
            self.assertEqual(info["used"], 0)
            self.assertEqual(info["limit"], 50)  # default daily_budget
            self.assertEqual(info["remaining"], 50)

        run_async(_test())


if __name__ == "__main__":
    unittest.main()
