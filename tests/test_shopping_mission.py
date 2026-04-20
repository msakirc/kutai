"""Tests for two-tier shopping: simple queries vs research missions."""
import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestShoppingTierDetection(unittest.TestCase):
    """Test which queries trigger missions vs simple tasks."""

    def _check(self, query, sub_intent=None):
        from src.app.telegram_bot import TelegramInterface
        return TelegramInterface._is_complex_shopping_query(query, sub_intent)

    # ── Simple (Tier 1) queries ──

    def test_price_check_is_simple(self):
        self.assertFalse(self._check("iPhone 15 fiyat"))

    def test_single_product_is_simple(self):
        self.assertFalse(self._check("RTX 4070 en ucuz"))

    def test_gift_is_simple(self):
        self.assertFalse(self._check("hediye kutu"))

    def test_price_check_sub_intent_is_simple(self):
        self.assertFalse(self._check("iPhone 15 fiyat", sub_intent="price_check"))

    def test_deal_hunt_sub_intent_is_simple(self):
        self.assertFalse(self._check("laptop indirim", sub_intent="deal_hunt"))

    # ── Complex (Tier 2) queries ──

    def test_research_sub_intent_is_mission(self):
        self.assertTrue(self._check("gaming laptop", sub_intent="research"))

    def test_exploration_sub_intent_is_mission(self):
        self.assertTrue(self._check("wireless earbuds", sub_intent="exploration"))

    def test_compare_sub_intent_is_mission(self):
        self.assertTrue(self._check("iPhone vs Samsung", sub_intent="compare"))

    def test_arastir_keyword_is_mission(self):
        self.assertTrue(self._check("araştır gaming laptop"))

    def test_karsilastir_keyword_is_mission(self):
        self.assertTrue(self._check("karşılaştır iPhone Samsung"))

    def test_analiz_et_keyword_is_mission(self):
        self.assertTrue(self._check("analiz et RTX 4070 fiyatları"))

    def test_detayli_keyword_is_mission(self):
        self.assertTrue(self._check("detaylı gaming mouse"))

    def test_compare_keyword_is_mission(self):
        self.assertTrue(self._check("compare gaming laptops under 30k"))

    def test_research_keyword_is_mission(self):
        self.assertTrue(self._check("research best headphones 2026"))

    def test_analyze_keyword_is_mission(self):
        self.assertTrue(self._check("analyze smart TV options"))

    def test_vs_with_products_is_mission(self):
        self.assertTrue(self._check("MacBook Air vs Dell XPS"))

    def test_vs_standalone_not_mission(self):
        # "vs" alone without products around it should not trigger
        self.assertFalse(self._check("best headphones"))


class TestShoppingMissionCreation(unittest.TestCase):
    """Test that _create_shopping_mission creates correct task structure."""

    @patch("src.app.telegram_bot.add_task")
    @patch("src.app.telegram_bot.add_mission")
    def test_creates_mission_and_three_tasks(self, mock_add_mission, mock_add_task):
        mock_add_mission.return_value = 100
        mock_add_task.side_effect = [201, 202, 203]

        bot = MagicMock(spec=[])
        bot._create_shopping_mission = (
            TestShoppingMissionCreation._get_unbound_method()
        )

        mission_id = run_async(
            bot._create_shopping_mission(bot, "gaming laptop araştır", 12345)
        )

        self.assertEqual(mission_id, 100)
        self.assertEqual(mock_add_mission.call_count, 1)
        self.assertEqual(mock_add_task.call_count, 3)

    @staticmethod
    def _get_unbound_method():
        from src.app.telegram_bot import TelegramInterface
        return TelegramInterface._create_shopping_mission

    @patch("src.app.telegram_bot.add_task")
    @patch("src.app.telegram_bot.add_mission")
    def test_task_priorities_correct(self, mock_add_mission, mock_add_task):
        mock_add_mission.return_value = 100
        mock_add_task.side_effect = [201, 202, 203]

        method = self._get_unbound_method()
        bot = MagicMock(spec=[])

        run_async(method(bot, "compare headphones", 12345))

        calls = mock_add_task.call_args_list
        # Task 1 (research): priority 8
        self.assertEqual(calls[0].kwargs["priority"], 8)
        # Task 2 (analysis): priority 8
        self.assertEqual(calls[1].kwargs["priority"], 8)
        # Task 3 (synthesis): priority 7
        self.assertEqual(calls[2].kwargs["priority"], 7)

    @patch("src.app.telegram_bot.add_task")
    @patch("src.app.telegram_bot.add_mission")
    def test_depends_on_chain(self, mock_add_mission, mock_add_task):
        mock_add_mission.return_value = 100
        mock_add_task.side_effect = [201, 202, 203]

        method = self._get_unbound_method()
        bot = MagicMock(spec=[])

        run_async(method(bot, "research laptops", 12345))

        calls = mock_add_task.call_args_list
        # Task 1: no depends_on
        self.assertNotIn("depends_on", calls[0].kwargs)
        # Task 2: depends on task 1
        self.assertEqual(calls[1].kwargs["depends_on"], [201])
        # Task 3: depends on task 1 and task 2
        self.assertEqual(calls[2].kwargs["depends_on"], [201, 202])

    @patch("src.app.telegram_bot.add_task")
    @patch("src.app.telegram_bot.add_mission")
    def test_silent_flag_on_intermediate_tasks(self, mock_add_mission, mock_add_task):
        mock_add_mission.return_value = 100
        mock_add_task.side_effect = [201, 202, 203]

        method = self._get_unbound_method()
        bot = MagicMock(spec=[])

        run_async(method(bot, "compare TVs", 12345))

        calls = mock_add_task.call_args_list
        # Task 1 context should have silent=True
        ctx1 = calls[0].kwargs["context"]
        self.assertTrue(ctx1["silent"])
        # Task 2 context should have silent=True
        ctx2 = calls[1].kwargs["context"]
        self.assertTrue(ctx2["silent"])
        # Task 3 (final) should NOT be silent
        ctx3 = calls[2].kwargs["context"]
        self.assertNotIn("silent", ctx3)

    @patch("src.app.telegram_bot.add_task")
    @patch("src.app.telegram_bot.add_mission")
    def test_mission_id_on_all_tasks(self, mock_add_mission, mock_add_task):
        mock_add_mission.return_value = 100
        mock_add_task.side_effect = [201, 202, 203]

        method = self._get_unbound_method()
        bot = MagicMock(spec=[])

        run_async(method(bot, "gaming keyboards", 12345))

        calls = mock_add_task.call_args_list
        for i, call in enumerate(calls):
            self.assertEqual(
                call.kwargs["mission_id"], 100,
                f"Task {i+1} missing mission_id",
            )

    @patch("src.app.telegram_bot.add_task")
    @patch("src.app.telegram_bot.add_mission")
    def test_agent_types_correct(self, mock_add_mission, mock_add_task):
        mock_add_mission.return_value = 100
        mock_add_task.side_effect = [201, 202, 203]

        method = self._get_unbound_method()
        bot = MagicMock(spec=[])

        run_async(method(bot, "gaming mice", 12345))

        calls = mock_add_task.call_args_list
        self.assertEqual(calls[0].kwargs["agent_type"], "product_researcher")
        self.assertEqual(calls[1].kwargs["agent_type"], "deal_analyst")
        self.assertEqual(calls[2].kwargs["agent_type"], "shopping_advisor")


class TestShopIntentFork(unittest.TestCase):
    def _fresh_interface(self):
        """Build a minimal TelegramInterface shell for testing."""
        from src.app.telegram_bot import TelegramInterface
        iface = TelegramInterface.__new__(TelegramInterface)
        iface._pending_action = {}
        iface._pending_shop_subintent = {}
        iface._kb_state = {}
        return iface

    def test_cmd_research_product_no_args_sends_inline_buttons(self):
        from unittest.mock import AsyncMock, MagicMock
        iface = self._fresh_interface()
        update = MagicMock()
        update.effective_chat.id = 42
        msg = MagicMock()
        msg.reply_text = AsyncMock()
        update.message = msg
        context = MagicMock()
        context.args = []

        run_async(iface.cmd_research_product(update, context))

        _args, kwargs = msg.reply_text.call_args
        self.assertIn("reply_markup", kwargs)
        markup = kwargs["reply_markup"]
        callback_values = []
        for row in markup.inline_keyboard:
            for btn in row:
                callback_values.append(btn.callback_data)
        self.assertIn("shop:specific", callback_values)
        self.assertIn("shop:category", callback_values)

    def test_callback_shop_specific_sets_pending_subintent(self):
        from unittest.mock import AsyncMock, MagicMock
        iface = self._fresh_interface()
        query = MagicMock()
        query.answer = AsyncMock()
        query.data = "shop:specific"
        query.message = MagicMock()
        query.message.reply_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        update.effective_chat.id = 42
        context = MagicMock()

        run_async(iface.handle_callback(update, context))

        self.assertIn(42, iface._pending_action)
        self.assertEqual(iface._pending_action[42]["command"], "research_product")
        self.assertEqual(iface._pending_shop_subintent.get(42), "specific")

    def test_callback_shop_category_sets_pending_subintent(self):
        from unittest.mock import AsyncMock, MagicMock
        iface = self._fresh_interface()
        query = MagicMock()
        query.answer = AsyncMock()
        query.data = "shop:category"
        query.message = MagicMock()
        query.message.reply_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        update.effective_chat.id = 42
        context = MagicMock()

        run_async(iface.handle_callback(update, context))

        self.assertEqual(iface._pending_shop_subintent.get(42), "category")


if __name__ == "__main__":
    unittest.main()
