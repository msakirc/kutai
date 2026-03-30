"""Tests for the price watch scheduler (daily re-scrape + notifications)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import unittest

import pytest

from src.app.price_watch_checker import (
    check_price_watches,
    format_price_drop_message,
    _domain_from_source,
    _domain_from_url,
)


# ─── Unit: domain extraction ────────────────────────────────────────────────


class TestDomainFromSource:
    def test_simple_name(self):
        assert _domain_from_source("trendyol") == "trendyol"

    def test_with_com(self):
        assert _domain_from_source("trendyol.com") == "trendyol"

    def test_with_com_tr(self):
        assert _domain_from_source("hepsiburada.com.tr") == "hepsiburada"

    def test_with_www(self):
        assert _domain_from_source("www.akakce.com") == "akakce"

    def test_capitalised(self):
        assert _domain_from_source("Trendyol") == "trendyol"

    def test_none(self):
        assert _domain_from_source(None) is None

    def test_empty(self):
        assert _domain_from_source("") is None


class TestDomainFromUrl:
    def test_full_url(self):
        assert _domain_from_url("https://www.trendyol.com/p/12345") == "trendyol"

    def test_none(self):
        assert _domain_from_url(None) is None

    def test_hepsiburada(self):
        assert _domain_from_url("https://www.hepsiburada.com/foo-p-123") == "hepsiburada"


# ─── Unit: notification message formatting ──────────────────────────────────


class TestFormatPriceDropMessage:
    def test_basic_drop(self):
        msg = format_price_drop_message(
            product_name="iPhone 15 Pro Max",
            old_price=65000,
            new_price=59990,
            source="Trendyol",
            url="https://trendyol.com/p/123",
        )
        assert "iPhone 15 Pro Max" in msg
        assert "65,000" in msg or "65.000" in msg
        assert "59,990" in msg or "59.990" in msg
        assert "Trendyol" in msg
        assert "trendyol.com" in msg

    def test_drop_with_target_hit(self):
        msg = format_price_drop_message(
            product_name="Test Product",
            old_price=10000,
            new_price=8000,
            source="Trendyol",
            url=None,
            target_price=9000,
        )
        assert "Hedef fiyat" in msg
        assert "alt\u0131na" in msg  # "altına düştü"

    def test_drop_with_target_not_hit(self):
        msg = format_price_drop_message(
            product_name="Test Product",
            old_price=10000,
            new_price=9500,
            source="Trendyol",
            url=None,
            target_price=8000,
        )
        assert "Hedefe" in msg
        assert "kald\u0131" in msg  # "kaldı"

    def test_no_url(self):
        msg = format_price_drop_message(
            product_name="X",
            old_price=100,
            new_price=90,
            source="src",
            url=None,
        )
        assert "Link" not in msg

    def test_percentage_calculation(self):
        msg = format_price_drop_message(
            product_name="X",
            old_price=1000,
            new_price=900,
            source=None,
            url=None,
        )
        # 10% drop
        assert "10.0%" in msg

    def test_no_source(self):
        msg = format_price_drop_message(
            product_name="X",
            old_price=100,
            new_price=90,
            source=None,
            url=None,
        )
        assert "Kaynak" not in msg


# ─── Unit: price comparison logic (inside check_price_watches) ──────────────


def _make_watch(
    watch_id=1,
    user_id=123,
    product_name="Test Product",
    current_price=10000.0,
    target_price=None,
    source="trendyol",
    product_url=None,
):
    return {
        "id": watch_id,
        "user_id": user_id,
        "product_name": product_name,
        "current_price": current_price,
        "target_price": target_price,
        "source": source,
        "product_url": product_url,
        "historical_low": current_price,
        "created_at": 1700000000.0,
        "updated_at": 1700000000.0,
    }


class TestCheckPriceWatches(unittest.IsolatedAsyncioTestCase):
    """Integration-style tests with mocked DB and scrapers."""

    async def test_no_watches(self):
        with (
            patch(
                "src.shopping.memory.price_watch.get_all_active_watches",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_get,
            patch(
                "src.shopping.memory.price_watch.expire_old_watches",
                new_callable=AsyncMock,
            ),
        ):
            result = await check_price_watches(telegram=None)
            assert result == {"checked": 0, "drops": 0, "errors": 0}
            mock_get.assert_awaited_once()

    async def test_price_drop_detected(self):
        """When scraped price < stored price by >1%, a drop is detected."""
        watch = _make_watch(current_price=10000, target_price=8000)

        mock_product = MagicMock()
        mock_product.discounted_price = 9000.0
        mock_product.original_price = 10000.0
        mock_product.source = "trendyol"
        mock_product.url = "https://trendyol.com/p/1"

        mock_scraper = MagicMock()
        mock_scraper.search = AsyncMock(return_value=[mock_product])

        mock_scraper_cls = MagicMock(return_value=mock_scraper)

        with (
            patch(
                "src.shopping.memory.price_watch.get_all_active_watches",
                new_callable=AsyncMock,
                return_value=[watch],
            ),
            patch(
                "src.shopping.memory.price_watch.expire_old_watches",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.update_watch_price",
                new_callable=AsyncMock,
            ) as mock_update,
            patch(
                "src.shopping.memory.price_watch.trigger_watch",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.scrapers.get_scraper",
                return_value=mock_scraper_cls,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await check_price_watches(telegram=None)
            assert result["checked"] == 1
            assert result["drops"] == 1
            mock_update.assert_awaited_once()

    async def test_no_price_change(self):
        """When scraped price is roughly the same, no drop is reported."""
        watch = _make_watch(current_price=10000)

        mock_product = MagicMock()
        mock_product.discounted_price = 10000.0
        mock_product.original_price = 10000.0
        mock_product.source = "trendyol"
        mock_product.url = "https://trendyol.com/p/1"

        mock_scraper = MagicMock()
        mock_scraper.search = AsyncMock(return_value=[mock_product])
        mock_scraper_cls = MagicMock(return_value=mock_scraper)

        with (
            patch(
                "src.shopping.memory.price_watch.get_all_active_watches",
                new_callable=AsyncMock,
                return_value=[watch],
            ),
            patch(
                "src.shopping.memory.price_watch.expire_old_watches",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.update_watch_price",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.trigger_watch",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.scrapers.get_scraper",
                return_value=mock_scraper_cls,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await check_price_watches(telegram=None)
            assert result["checked"] == 1
            assert result["drops"] == 0

    async def test_price_increase_no_notification(self):
        """When price goes UP, no drop notification is sent."""
        watch = _make_watch(current_price=10000)

        mock_product = MagicMock()
        mock_product.discounted_price = 11000.0
        mock_product.original_price = 11000.0
        mock_product.source = "trendyol"
        mock_product.url = "https://trendyol.com/p/1"

        mock_scraper = MagicMock()
        mock_scraper.search = AsyncMock(return_value=[mock_product])
        mock_scraper_cls = MagicMock(return_value=mock_scraper)

        with (
            patch(
                "src.shopping.memory.price_watch.get_all_active_watches",
                new_callable=AsyncMock,
                return_value=[watch],
            ),
            patch(
                "src.shopping.memory.price_watch.expire_old_watches",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.update_watch_price",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.trigger_watch",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.scrapers.get_scraper",
                return_value=mock_scraper_cls,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await check_price_watches(telegram=None)
            assert result["drops"] == 0

    async def test_target_price_hit_triggers_watch(self):
        """When price drops below target, the watch is triggered."""
        watch = _make_watch(current_price=10000, target_price=9000)

        mock_product = MagicMock()
        mock_product.discounted_price = 8500.0
        mock_product.original_price = 10000.0
        mock_product.source = "trendyol"
        mock_product.url = "https://trendyol.com/p/1"

        mock_scraper = MagicMock()
        mock_scraper.search = AsyncMock(return_value=[mock_product])
        mock_scraper_cls = MagicMock(return_value=mock_scraper)

        with (
            patch(
                "src.shopping.memory.price_watch.get_all_active_watches",
                new_callable=AsyncMock,
                return_value=[watch],
            ),
            patch(
                "src.shopping.memory.price_watch.expire_old_watches",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.update_watch_price",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.trigger_watch",
                new_callable=AsyncMock,
            ) as mock_trigger,
            patch(
                "src.shopping.scrapers.get_scraper",
                return_value=mock_scraper_cls,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await check_price_watches(telegram=None)
            assert result["drops"] == 1
            mock_trigger.assert_awaited_once_with(1)

    async def test_scraper_failure_counts_as_error(self):
        """When scraper returns no results, it counts as an error."""
        watch = _make_watch(current_price=10000)

        mock_scraper = MagicMock()
        mock_scraper.search = AsyncMock(return_value=[])
        mock_scraper_cls = MagicMock(return_value=mock_scraper)

        with (
            patch(
                "src.shopping.memory.price_watch.get_all_active_watches",
                new_callable=AsyncMock,
                return_value=[watch],
            ),
            patch(
                "src.shopping.memory.price_watch.expire_old_watches",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.update_watch_price",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.trigger_watch",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.scrapers.get_scraper",
                return_value=mock_scraper_cls,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await check_price_watches(telegram=None)
            assert result["errors"] == 1
            assert result["checked"] == 0

    async def test_telegram_notification_sent(self):
        """When a drop is detected, a Telegram message is sent."""
        watch = _make_watch(current_price=10000, user_id=42)

        mock_product = MagicMock()
        mock_product.discounted_price = 9000.0
        mock_product.original_price = 10000.0
        mock_product.source = "trendyol"
        mock_product.url = "https://trendyol.com/p/1"

        mock_scraper = MagicMock()
        mock_scraper.search = AsyncMock(return_value=[mock_product])
        mock_scraper_cls = MagicMock(return_value=mock_scraper)

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        mock_telegram = MagicMock()
        mock_telegram.app.bot = mock_bot

        with (
            patch(
                "src.shopping.memory.price_watch.get_all_active_watches",
                new_callable=AsyncMock,
                return_value=[watch],
            ),
            patch(
                "src.shopping.memory.price_watch.expire_old_watches",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.update_watch_price",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.memory.price_watch.trigger_watch",
                new_callable=AsyncMock,
            ),
            patch(
                "src.shopping.scrapers.get_scraper",
                return_value=mock_scraper_cls,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await check_price_watches(telegram=mock_telegram)
            assert result["drops"] == 1
            mock_bot.send_message.assert_awaited_once()
            call_kwargs = mock_bot.send_message.call_args[1]
            assert call_kwargs["chat_id"] == 42
            assert "Fiyat" in call_kwargs["text"]
