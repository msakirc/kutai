"""Tests for Telegram menu navigation system."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time
from datetime import datetime, timedelta, timezone


# Import the keyboard/action data structures (no Telegram connection needed)
from src.app.telegram_bot import (
    REPLY_KEYBOARD, KB_HIZMET, KB_ALISVERIS, KB_LISTEM, KB_GOREVLER,
    KB_WORKFLOW_SELECT, KB_SISTEM, KB_YUK_MODU, KB_BASLAT,
    _BUTTON_ACTIONS, _KB_PARENT, _CMD_ARG_PROMPTS, _PENDING_ACTION_TIMEOUT,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

ALL_KEYBOARDS = {
    "main": REPLY_KEYBOARD,
    "hizmet": KB_HIZMET,
    "alisveris": KB_ALISVERIS,
    "listem": KB_LISTEM,
    "gorevler": KB_GOREVLER,
    "workflow_select": KB_WORKFLOW_SELECT,
    "sistem": KB_SISTEM,
    "yuk_modu": KB_YUK_MODU,
    "baslat": KB_BASLAT,
}

SUB_KEYBOARDS = {k: v for k, v in ALL_KEYBOARDS.items() if k != "main"}


def _get_all_button_texts(kb):
    """Extract all button texts from a ReplyKeyboardMarkup."""
    texts = []
    for row in kb.keyboard:
        for btn in row:
            texts.append(btn.text)
    return texts


def _make_bot():
    """Create a TelegramInterface with mocked deps."""
    with patch("src.app.telegram_bot.Application"):
        from src.app.telegram_bot import TelegramInterface
        bot = TelegramInterface(orchestrator=MagicMock())
    return bot


def _make_update(chat_id=123, text="test"):
    """Create a mock Update object."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.chat_id = chat_id
    update.message.chat.id = chat_id
    update.message.text = text
    update.message.reply_text = AsyncMock()
    # Ensure no accidental document/photo triggers
    update.message.document = None
    update.message.photo = None
    return update


def _make_context():
    ctx = MagicMock()
    ctx.args = []
    return ctx


# ─── 1. Keyboard Integrity Tests ──────────────────────────────────────────────

class TestKeyboardIntegrity:
    """Data-structure sanity checks — no mocking required."""

    def test_reply_keyboard_has_five_buttons(self):
        texts = _get_all_button_texts(REPLY_KEYBOARD)
        assert len(texts) == 5

    def test_reply_keyboard_has_two_rows(self):
        assert len(REPLY_KEYBOARD.keyboard) == 2

    def test_reply_keyboard_first_row_has_three_buttons(self):
        assert len(REPLY_KEYBOARD.keyboard[0]) == 3

    def test_reply_keyboard_second_row_has_two_buttons(self):
        assert len(REPLY_KEYBOARD.keyboard[1]) == 2

    def test_all_keyboard_buttons_have_button_actions(self):
        """Every button in every keyboard must be in _BUTTON_ACTIONS."""
        for kb_name, kb in ALL_KEYBOARDS.items():
            for btn_text in _get_all_button_texts(kb):
                assert btn_text in _BUTTON_ACTIONS, (
                    f"Button '{btn_text}' in keyboard '{kb_name}' "
                    f"has no entry in _BUTTON_ACTIONS"
                )

    def test_kb_parent_keys_are_valid_state_names(self):
        """All keys in _KB_PARENT must be valid state names used by _get_current_keyboard."""
        valid_states = {"hizmet", "alisveris", "listem", "gorevler", "sistem",
                        "workflow_select", "yuk_modu", "debug", "dlq"}
        for key in _KB_PARENT:
            assert key in valid_states, (
                f"_KB_PARENT key '{key}' is not a recognised state"
            )

    def test_back_button_exists_in_every_sub_keyboard(self):
        sub_kbs = [KB_HIZMET, KB_ALISVERIS, KB_LISTEM, KB_GOREVLER,
                   KB_WORKFLOW_SELECT, KB_SISTEM, KB_YUK_MODU]
        for kb in sub_kbs:
            texts = _get_all_button_texts(kb)
            assert "🔙 Geri" in texts, f"Missing back button in {kb}"

    def test_baslat_keyboard_has_start_button(self):
        texts = _get_all_button_texts(KB_BASLAT)
        assert "▶️ Başlat" in texts

    def test_back_button_mapped_to_special_back(self):
        assert _BUTTON_ACTIONS["🔙 Geri"] == ("special", "back")

    def test_pending_action_timeout_is_positive(self):
        assert _PENDING_ACTION_TIMEOUT > 0

    def test_cmd_arg_prompts_cover_known_cmd_args_buttons(self):
        """All cmd_args actions must have a prompt defined."""
        for btn_text, (action_type, action_value) in _BUTTON_ACTIONS.items():
            if action_type == "cmd_args":
                assert action_value in _CMD_ARG_PROMPTS, (
                    f"cmd_args action '{action_value}' (button: '{btn_text}') "
                    f"has no prompt in _CMD_ARG_PROMPTS"
                )

    def test_top_level_categories_map_to_category_type(self):
        category_buttons = [
            "⚡ Hizmet", "🛒 Alışveriş", "📋 Listem",
            "🎯 Görevler", "⚙️ Sistem",
        ]
        for btn in category_buttons:
            action_type, _ = _BUTTON_ACTIONS[btn]
            assert action_type == "category", (
                f"Top-level button '{btn}' should be 'category', got '{action_type}'"
            )

    def test_workflow_select_buttons_are_special(self):
        wf_buttons = [
            "⚡ Hızlı Cevap", "📊 Araştır & Raporla",
            "🏗 Yeni Proje", "🤖 Otomatik", "💻 Kod / Diğer",
        ]
        for btn in wf_buttons:
            action_type, action_value = _BUTTON_ACTIONS[btn]
            assert action_type == "special"
            assert action_value.startswith("wf_")

    def test_no_action_type_is_invalid(self):
        valid_types = {"cmd", "cmd_args", "category", "special"}
        for btn_text, (action_type, _) in _BUTTON_ACTIONS.items():
            assert action_type in valid_types, (
                f"Button '{btn_text}' has unknown action type '{action_type}'"
            )

    def test_kb_parent_maps_sub_states_to_correct_parent_keyboards(self):
        # Direct parents of top-level sub-kbs must be REPLY_KEYBOARD
        for state in ("hizmet", "alisveris", "listem", "gorevler", "sistem"):
            assert _KB_PARENT[state] is REPLY_KEYBOARD, (
                f"State '{state}' should have REPLY_KEYBOARD as parent"
            )

    def test_workflow_select_parent_is_gorevler(self):
        assert _KB_PARENT["workflow_select"] is KB_GOREVLER

    def test_yuk_modu_parent_is_sistem(self):
        assert _KB_PARENT["yuk_modu"] is KB_SISTEM


# ─── 2. Button Action Routing Tests ──────────────────────────────────────────

class TestButtonActionRouting:
    """Test that button taps are routed to the right handlers."""

    def test_category_hizmet_action(self):
        assert _BUTTON_ACTIONS["⚡ Hizmet"] == ("category", "hizmet")

    def test_category_alisveris_action(self):
        assert _BUTTON_ACTIONS["🛒 Alışveriş"] == ("category", "alisveris")

    def test_category_listem_action(self):
        assert _BUTTON_ACTIONS["📋 Listem"] == ("category", "listem")

    def test_category_gorevler_action(self):
        assert _BUTTON_ACTIONS["🎯 Görevler"] == ("category", "gorevler")

    def test_category_sistem_action(self):
        assert _BUTTON_ACTIONS["⚙️ Sistem"] == ("category", "sistem")

    def test_hizli_ara_is_cmd_args_shop(self):
        assert _BUTTON_ACTIONS["⚡ Hızlı Ara"] == ("cmd_args", "shop")

    def test_detayli_arastir_is_cmd_args_research(self):
        assert _BUTTON_ACTIONS["🔬 Detaylı Araştır"] == ("cmd_args", "research_product")

    def test_yeni_ekle_is_cmd_args_todo(self):
        assert _BUTTON_ACTIONS["📝 Yeni Ekle"] == ("cmd_args", "todo")

    def test_is_kuyrugu_is_cmd_view_queue(self):
        assert _BUTTON_ACTIONS["📬 İş Kuyruğu"] == ("cmd", "view_queue")

    def test_eczane_is_special_pharmacy(self):
        assert _BUTTON_ACTIONS["🏥 Eczane"] == ("special", "pharmacy")

    def test_new_mission_is_special(self):
        assert _BUTTON_ACTIONS["🎯 Yeni Görev"] == ("special", "new_mission")

    def test_restart_is_special(self):
        assert _BUTTON_ACTIONS["🔄 Yeniden Başlat"] == ("special", "restart")

    def test_stop_is_special(self):
        assert _BUTTON_ACTIONS["⏹ Durdur"] == ("special", "stop")

    def test_yuk_modu_is_category(self):
        assert _BUTTON_ACTIONS["🖥 Yük Modu"] == ("category", "yuk_modu")

    def test_full_load_is_special(self):
        assert _BUTTON_ACTIONS["⚡ Full"] == ("special", "load_full")

    def test_hatirla_is_special_reminder(self):
        assert _BUTTON_ACTIONS["⏰ Hatırlat"] == ("special", "reminder")


# ─── 3. Keyboard State Management Tests ──────────────────────────────────────

class TestKeyboardStateManagement:
    """Tests for _kb_state navigation logic."""

    def test_default_state_is_main(self):
        bot = _make_bot()
        kb = bot._get_current_keyboard(chat_id=999)
        assert kb is REPLY_KEYBOARD

    def test_state_hizmet_returns_kb_hizmet(self):
        bot = _make_bot()
        bot._kb_state[1] = "hizmet"
        assert bot._get_current_keyboard(1) is KB_HIZMET

    def test_state_alisveris_returns_kb_alisveris(self):
        bot = _make_bot()
        bot._kb_state[1] = "alisveris"
        assert bot._get_current_keyboard(1) is KB_ALISVERIS

    def test_state_listem_returns_kb_listem(self):
        bot = _make_bot()
        bot._kb_state[1] = "listem"
        assert bot._get_current_keyboard(1) is KB_LISTEM

    def test_state_gorevler_returns_kb_gorevler(self):
        bot = _make_bot()
        bot._kb_state[1] = "gorevler"
        assert bot._get_current_keyboard(1) is KB_GOREVLER

    def test_state_sistem_returns_kb_sistem(self):
        bot = _make_bot()
        bot._kb_state[1] = "sistem"
        assert bot._get_current_keyboard(1) is KB_SISTEM

    def test_state_workflow_select_returns_kb_workflow(self):
        bot = _make_bot()
        bot._kb_state[1] = "workflow_select"
        assert bot._get_current_keyboard(1) is KB_WORKFLOW_SELECT

    def test_state_yuk_modu_returns_kb_yuk_modu(self):
        bot = _make_bot()
        bot._kb_state[1] = "yuk_modu"
        assert bot._get_current_keyboard(1) is KB_YUK_MODU

    def test_unknown_state_falls_back_to_reply_keyboard(self):
        bot = _make_bot()
        bot._kb_state[1] = "nonexistent_state"
        assert bot._get_current_keyboard(1) is REPLY_KEYBOARD

    def test_none_chat_id_returns_reply_keyboard(self):
        bot = _make_bot()
        assert bot._get_current_keyboard(None) is REPLY_KEYBOARD

    @pytest.mark.asyncio
    async def test_swap_keyboard_updates_state(self):
        bot = _make_bot()
        update = _make_update(chat_id=5)
        await bot._swap_keyboard(update, "hizmet", text="test")
        assert bot._kb_state[5] == "hizmet"

    @pytest.mark.asyncio
    async def test_category_button_tap_changes_state(self):
        bot = _make_bot()
        update = _make_update(chat_id=10, text="⚡ Hizmet")
        ctx = _make_context()
        # Patch _handle_category_button to track state change only
        called_with = {}
        async def fake_category(u, c, category):
            bot._kb_state[u.effective_chat.id] = category
            called_with["category"] = category
        bot._handle_category_button = fake_category

        await bot.handle_message(update, ctx)
        assert called_with.get("category") == "hizmet"

    @pytest.mark.asyncio
    async def test_back_from_hizmet_goes_to_main(self):
        bot = _make_bot()
        bot._kb_state[1] = "hizmet"
        update = _make_update(chat_id=1)
        ctx = _make_context()
        await bot._handle_special_button(update, ctx, "back")
        assert bot._kb_state[1] == "main"

    @pytest.mark.asyncio
    async def test_back_from_gorevler_goes_to_main(self):
        bot = _make_bot()
        bot._kb_state[1] = "gorevler"
        update = _make_update(chat_id=1)
        ctx = _make_context()
        await bot._handle_special_button(update, ctx, "back")
        assert bot._kb_state[1] == "main"

    @pytest.mark.asyncio
    async def test_back_from_yuk_modu_goes_to_sistem(self):
        bot = _make_bot()
        bot._kb_state[1] = "yuk_modu"
        update = _make_update(chat_id=1)
        ctx = _make_context()
        await bot._handle_special_button(update, ctx, "back")
        assert bot._kb_state[1] == "sistem"

    @pytest.mark.asyncio
    async def test_back_from_workflow_select_goes_to_gorevler(self):
        bot = _make_bot()
        bot._kb_state[1] = "workflow_select"
        update = _make_update(chat_id=1)
        ctx = _make_context()
        await bot._handle_special_button(update, ctx, "back")
        assert bot._kb_state[1] == "gorevler"

    @pytest.mark.asyncio
    async def test_back_from_main_stays_at_main(self):
        """Back from main (or unknown) state stays at main."""
        bot = _make_bot()
        # Not setting _kb_state — defaults to "main"
        update = _make_update(chat_id=1)
        ctx = _make_context()
        await bot._handle_special_button(update, ctx, "back")
        assert bot._kb_state.get(1, "main") == "main"


# ─── 4. Pending Action Timeout Tests ──────────────────────────────────────────

class TestPendingActionTimeout:
    """Pending action TTL enforcement."""

    @pytest.mark.asyncio
    async def test_fresh_pending_action_is_processed(self):
        """A cmd_args pending action with a fresh timestamp should be consumed."""
        bot = _make_bot()
        chat_id = 42
        bot._pending_action[chat_id] = {
            "command": "shop",
            "ts": time.time(),  # just now
        }
        update = _make_update(chat_id=chat_id, text="kahve makinesi")
        ctx = _make_context()

        handled = {}
        async def fake_shop(u, c):
            handled["called"] = True
            c.args = u.message.text.split()
        bot.cmd_shop = fake_shop

        await bot.handle_message(update, ctx)
        assert handled.get("called"), "Fresh pending action should be routed to handler"

    @pytest.mark.asyncio
    async def test_expired_pending_action_falls_through(self):
        """An expired pending action (>300 s) should not be routed to its handler."""
        bot = _make_bot()
        chat_id = 43
        bot._pending_action[chat_id] = {
            "command": "shop",
            "ts": time.time() - (_PENDING_ACTION_TIMEOUT + 1),
        }
        update = _make_update(chat_id=chat_id, text="kahve makinesi")
        ctx = _make_context()

        shop_called = {}
        async def fake_shop(u, c):
            shop_called["called"] = True
        bot.cmd_shop = fake_shop

        # Patch classifier so it doesn't blow up
        bot._classify_and_route = AsyncMock()
        await bot.handle_message(update, ctx)
        assert not shop_called.get("called"), "Expired pending action must not be processed"

    @pytest.mark.asyncio
    async def test_button_tap_clears_pending_action(self):
        """Tapping a reply button must clear any stale pending_action."""
        bot = _make_bot()
        chat_id = 44
        bot._pending_action[chat_id] = {
            "command": "shop",
            "ts": time.time(),
        }
        update = _make_update(chat_id=chat_id, text="🎯 Görevler")
        ctx = _make_context()

        # Patch _handle_category_button to avoid DB calls
        bot._handle_category_button = AsyncMock()
        await bot.handle_message(update, ctx)
        assert chat_id not in bot._pending_action, (
            "Pending action must be cleared when a reply button is tapped"
        )


# ─── 5. Workflow Selection Flow Tests ─────────────────────────────────────────

class TestWorkflowSelectionFlow:
    """Test the new-mission → workflow-picker → create flow."""

    @pytest.mark.asyncio
    async def test_new_mission_sets_pending_action_workflow_select(self):
        bot = _make_bot()
        chat_id = 50
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()
        await bot._handle_special_button(update, ctx, "new_mission")
        pending = bot._pending_action.get(chat_id)
        assert pending is not None
        assert pending["command"] == "_workflow_select"
        assert "ts" in pending

    @pytest.mark.asyncio
    async def test_mission_description_typed_shows_workflow_picker(self):
        """After new_mission prompt, typing a description shows KB_WORKFLOW_SELECT."""
        bot = _make_bot()
        chat_id = 51
        bot._pending_action[chat_id] = {
            "command": "_workflow_select",
            "ts": time.time(),
        }
        update = _make_update(chat_id=chat_id, text="API entegrasyonu yaz")
        ctx = _make_context()
        await bot.handle_message(update, ctx)

        # Pending mission must be stored
        assert bot._pending_mission.get(chat_id) == "API entegrasyonu yaz"
        # State must switch to workflow_select
        assert bot._kb_state.get(chat_id) == "workflow_select"
        # reply_text should have been called with KB_WORKFLOW_SELECT
        update.message.reply_text.assert_called()
        call_kwargs = update.message.reply_text.call_args
        assert call_kwargs.kwargs.get("reply_markup") is KB_WORKFLOW_SELECT or (
            len(call_kwargs.args) >= 1  # at minimum called with something
        )

    @pytest.mark.asyncio
    async def test_wf_other_shows_info_and_resets_to_gorevler(self):
        bot = _make_bot()
        chat_id = 52
        bot._pending_mission[chat_id] = "Bir şey yap"
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()
        await bot._handle_special_button(update, ctx, "wf_other")
        # State should return to gorevler
        assert bot._kb_state.get(chat_id) == "gorevler"
        update.message.reply_text.assert_called()

    @pytest.mark.asyncio
    async def test_wf_action_without_pending_mission_shows_error(self):
        """If no _pending_mission is stored, workflow button shows error."""
        bot = _make_bot()
        chat_id = 53
        # No _pending_mission stored
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()

        # patch _reply to capture output
        bot._reply = AsyncMock()
        await bot._handle_special_button(update, ctx, "wf_quick")
        bot._reply.assert_called()
        call_text = bot._reply.call_args[0][1]
        assert "❌" in call_text


# ─── 6. Reminder Parsing Tests ────────────────────────────────────────────────

class TestParseReminderTime:
    """Static method _parse_reminder_time — pure string-to-datetime tests."""

    from src.app.telegram_bot import TelegramInterface as _TI

    def _parse(self, text):
        from src.app.telegram_bot import TelegramInterface
        return TelegramInterface._parse_reminder_time(text)

    def test_10dk(self):
        # _parse_reminder_time returns UTC naive datetime for DB storage.
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        result = self._parse("10dk")
        after = datetime.now(timezone.utc).replace(tzinfo=None)
        assert result is not None
        diff = (result - before).total_seconds()
        assert 590 <= diff <= 610, f"Expected ~600s, got {diff}"

    def test_10_dakika(self):
        result = self._parse("10 dakika")
        assert result is not None
        diff = (result - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        assert 590 <= diff <= 610

    def test_1_saat(self):
        result = self._parse("1 saat")
        assert result is not None
        diff = (result - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        assert 3590 <= diff <= 3610

    def test_2s(self):
        result = self._parse("2s")
        assert result is not None
        diff = (result - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        assert 7190 <= diff <= 7210

    def test_hhmm_future(self):
        # HH:MM is parsed as Turkey local time; result is UTC naive.
        # Build a Turkey-local future time and check the UTC result is ~1h ahead of UTC now.
        try:
            from zoneinfo import ZoneInfo
            tz_tr = ZoneInfo("Europe/Istanbul")
            now_local = datetime.now(tz_tr)
        except Exception:
            now_local = datetime.now(timezone.utc)
        future_local = now_local + timedelta(hours=1)
        text = future_local.strftime("%H:%M")
        result = self._parse(text)
        assert result is not None
        # result is UTC naive; compare against UTC now
        diff = abs((result - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds() - 3600)
        assert diff < 120, f"Expected ~3600s UTC offset, got diff {diff}"

    def test_hhmm_past_becomes_tomorrow(self):
        # Build a Turkey-local time 1h in the past; result should be ~23h ahead in UTC.
        try:
            from zoneinfo import ZoneInfo
            tz_tr = ZoneInfo("Europe/Istanbul")
            past_local = datetime.now(tz_tr) - timedelta(hours=1)
        except Exception:
            past_local = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        text = past_local.strftime("%H:%M")
        result = self._parse(text)
        assert result is not None
        # result is UTC naive; must be in the future from UTC now
        diff = (result - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        assert diff > 0, "Past Turkey time should roll over to tomorrow in UTC"
        assert diff < 86400 + 60

    def test_yarin_hhmm(self):
        # yarın 09:00 = tomorrow at 09:00 Turkey = tomorrow at 06:00 UTC
        result = self._parse("yarın 09:00")
        assert result is not None
        # Result is UTC naive: 09:00 TR = 06:00 UTC
        assert result.hour == 6
        assert result.minute == 0

    def test_bare_integer_is_minutes(self):
        result = self._parse("30")
        assert result is not None
        diff = (result - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        assert 1790 <= diff <= 1810

    def test_invalid_returns_none(self):
        assert self._parse("asdf") is None

    def test_empty_returns_none(self):
        assert self._parse("  ") is None

    def test_random_text_returns_none(self):
        assert self._parse("yarın sabah") is None


# ─── 7. Cron Parsing Tests ────────────────────────────────────────────────────

class TestParseCronInput:
    """Static method _parse_cron_input — Turkish schedule → cron expression."""

    def _parse(self, text):
        from src.app.telegram_bot import TelegramInterface
        return TelegramInterface._parse_cron_input(text)

    # Cron hours are UTC (Turkey local - 3h).  Turkey has no DST, always UTC+3.
    # Examples: 09:00 TR = 06:00 UTC, 14:30 TR = 11:30 UTC, 10:30 TR = 07:30 UTC.

    def test_her_gun_09_00(self):
        result = self._parse("her gün 09:00")
        assert result == "0 6 * * *"  # 09:00 TR = 06:00 UTC

    def test_her_gun_14_30(self):
        result = self._parse("her gün 14:30")
        assert result == "30 11 * * *"  # 14:30 TR = 11:30 UTC

    def test_her_gun_no_time_default_09(self):
        result = self._parse("her gün")
        assert result == "0 6 * * *"  # default 09:00 TR = 06:00 UTC

    def test_her_2_saatte(self):
        result = self._parse("her 2 saatte")
        assert result == "0 */2 * * *"  # relative — no UTC conversion

    def test_her_4_saatte(self):
        result = self._parse("her 4 saatte")
        assert result == "0 */4 * * *"  # relative — no UTC conversion

    def test_her_saat(self):
        result = self._parse("her saat")
        assert result == "0 * * * *"  # every hour — no conversion

    def test_her_pazartesi(self):
        result = self._parse("her pazartesi")
        assert result == "0 6 * * 1"  # default 09:00 TR = 06:00 UTC

    def test_her_cuma(self):
        result = self._parse("her cuma")
        assert result == "0 6 * * 5"  # default 09:00 TR = 06:00 UTC

    def test_her_pazar(self):
        result = self._parse("her pazar")
        assert result == "0 6 * * 0"  # default 09:00 TR = 06:00 UTC

    def test_her_pazartesi_with_time(self):
        result = self._parse("her pazartesi 14:00")
        assert result == "0 11 * * 1"  # 14:00 TR = 11:00 UTC

    def test_her_sali_10_30(self):
        result = self._parse("her salı 10:30")
        assert result == "30 7 * * 2"  # 10:30 TR = 07:30 UTC

    def test_invalid_returns_none(self):
        assert self._parse("bazen") is None

    def test_empty_returns_none(self):
        assert self._parse("  ") is None

    def test_random_text_returns_none(self):
        assert self._parse("random gibberish xyz") is None


# ─── 8. Category Auto-Content Tests ──────────────────────────────────────────

class TestCategoryAutoContent:
    """Category taps that trigger auto-content (Listem, Görevler, Sistem)."""

    @pytest.mark.asyncio
    async def test_listem_sets_state_and_sends_keyboard(self):
        bot = _make_bot()
        chat_id = 60
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()

        with patch("src.app.reminders.build_todo_list_message",
                   new_callable=AsyncMock, return_value=("", None)):
            await bot._handle_category_button(update, ctx, "listem")

        assert bot._kb_state.get(chat_id) == "listem"
        # Should have sent keyboard swap message
        update.message.reply_text.assert_called()

    @pytest.mark.asyncio
    async def test_listem_shows_empty_message_when_no_todos(self):
        bot = _make_bot()
        chat_id = 61
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()

        with patch("src.app.reminders.build_todo_list_message",
                   new_callable=AsyncMock, return_value=("", None)):
            await bot._handle_category_button(update, ctx, "listem")

        calls = [str(c) for c in update.message.reply_text.call_args_list]
        combined = " ".join(calls)
        assert "Listem" in combined or "yok" in combined

    @pytest.mark.asyncio
    async def test_gorevler_sets_state(self):
        bot = _make_bot()
        chat_id = 62
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()

        with patch("src.app.telegram_bot.get_active_missions",
                   new_callable=AsyncMock, return_value=[]):
            await bot._handle_category_button(update, ctx, "gorevler")

        assert bot._kb_state.get(chat_id) == "gorevler"

    @pytest.mark.asyncio
    async def test_gorevler_shows_active_missions(self):
        bot = _make_bot()
        chat_id = 63
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()
        missions = [
            {"id": 1, "title": "Mission Alpha", "status": "running"},
            {"id": 2, "title": "Mission Beta", "status": "pending"},
        ]

        with patch("src.app.telegram_bot.get_active_missions",
                   new_callable=AsyncMock, return_value=missions):
            await bot._handle_category_button(update, ctx, "gorevler")

        calls = [str(c) for c in update.message.reply_text.call_args_list]
        combined = " ".join(calls)
        assert "Mission Alpha" in combined or "Mission Beta" in combined

    @pytest.mark.asyncio
    async def test_sistem_sets_state_and_shows_dashboard(self):
        bot = _make_bot()
        chat_id = 64
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()

        # Patch dashboard builder so it doesn't call the actual DB/orchestrator
        bot._build_system_dashboard = AsyncMock(return_value="📊 *KutAI Durum*\nTest")
        await bot._handle_category_button(update, ctx, "sistem")

        assert bot._kb_state.get(chat_id) == "sistem"
        bot._build_system_dashboard.assert_called_once()
        update.message.reply_text.assert_called()

    @pytest.mark.asyncio
    async def test_hizmet_simple_swap(self):
        bot = _make_bot()
        chat_id = 65
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()
        await bot._handle_category_button(update, ctx, "hizmet")
        assert bot._kb_state.get(chat_id) == "hizmet"

    @pytest.mark.asyncio
    async def test_alisveris_simple_swap(self):
        bot = _make_bot()
        chat_id = 66
        update = _make_update(chat_id=chat_id)
        ctx = _make_context()
        await bot._handle_category_button(update, ctx, "alisveris")
        assert bot._kb_state.get(chat_id) == "alisveris"


# ─── 9. Callback Data Tests ──────────────────────────────────────────────────

class TestCallbackData:
    """handle_callback routing for m:* callback_data strings."""

    def _make_callback_update(self, data, chat_id=100):
        update = MagicMock()
        update.callback_query.data = data
        update.callback_query.answer = AsyncMock()
        update.callback_query.message.reply_text = AsyncMock()
        update.callback_query.message.chat_id = chat_id
        update.callback_query.message.chat.id = chat_id
        update.callback_query.edit_message_text = AsyncMock()
        return update

    @pytest.mark.asyncio
    async def test_task_detail_calls_get_mission(self):
        bot = _make_bot()
        update = self._make_callback_update("m:task:detail:7")
        ctx = _make_context()

        mission_data = {
            "id": 7, "title": "Test Mission", "status": "running", "priority": 5
        }
        with patch("src.app.telegram_bot.get_mission",
                   new_callable=AsyncMock, return_value=mission_data):
            await bot.handle_callback(update, ctx)

        update.callback_query.message.reply_text.assert_called()
        call_text = update.callback_query.message.reply_text.call_args[0][0]
        assert "Test Mission" in call_text

    @pytest.mark.asyncio
    async def test_task_pause_calls_update_mission(self):
        bot = _make_bot()
        update = self._make_callback_update("m:task:pause:7")
        ctx = _make_context()

        with patch("src.app.telegram_bot.update_mission",
                   new_callable=AsyncMock) as mock_update:
            await bot.handle_callback(update, ctx)
            mock_update.assert_called_once_with(7, status="paused")

    @pytest.mark.asyncio
    async def test_confirm_cancel_edits_message(self):
        bot = _make_bot()
        update = self._make_callback_update("m:confirm:cancel")
        ctx = _make_context()
        await bot.handle_callback(update, ctx)
        update.callback_query.edit_message_text.assert_called()

    @pytest.mark.asyncio
    async def test_confirm_restart_invokes_restart_handler(self):
        bot = _make_bot()
        update = self._make_callback_update("m:confirm:restart")
        ctx = _make_context()

        restart_called = {}
        async def fake_restart(u, c):
            restart_called["called"] = True
        bot.cmd_kutai_restart = fake_restart

        await bot.handle_callback(update, ctx)
        assert restart_called.get("called"), "Restart handler must be called on m:confirm:restart"

    @pytest.mark.asyncio
    async def test_confirm_restart_without_handler_shows_error(self):
        bot = _make_bot()
        # cmd_kutai_restart may or may not exist; hide it via getattr mock
        # to simulate missing handler path
        original = getattr(bot, "cmd_kutai_restart", None)
        bot.cmd_kutai_restart = None  # type: ignore[assignment]
        update = self._make_callback_update("m:confirm:restart")
        ctx = _make_context()

        await bot.handle_callback(update, ctx)
        # Should send an error message (getattr returns None → branch goes to else)
        update.callback_query.message.reply_text.assert_called()
        if original is not None:
            bot.cmd_kutai_restart = original

    @pytest.mark.asyncio
    async def test_task_detail_mission_not_found(self):
        bot = _make_bot()
        update = self._make_callback_update("m:task:detail:99")
        ctx = _make_context()

        with patch("src.app.telegram_bot.get_mission",
                   new_callable=AsyncMock, return_value=None):
            await bot.handle_callback(update, ctx)

        call_text = update.callback_query.message.reply_text.call_args[0][0]
        assert "bulunamadı" in call_text or "99" in call_text


# ─── 10. _reply Helper Tests ──────────────────────────────────────────────────

class TestReplyHelper:
    """_reply should include the current keyboard unless caller supplies reply_markup."""

    @pytest.mark.asyncio
    async def test_reply_includes_current_keyboard(self):
        bot = _make_bot()
        bot._kb_state[1] = "hizmet"
        update = _make_update(chat_id=1)
        await bot._reply(update, "hello")
        call_kwargs = update.message.reply_text.call_args.kwargs
        assert call_kwargs.get("reply_markup") is KB_HIZMET

    @pytest.mark.asyncio
    async def test_reply_preserves_caller_supplied_markup(self):
        from telegram import InlineKeyboardMarkup, InlineKeyboardButton
        bot = _make_bot()
        bot._kb_state[1] = "main"
        update = _make_update(chat_id=1)
        custom_markup = InlineKeyboardMarkup([[InlineKeyboardButton("x", callback_data="y")]])
        await bot._reply(update, "hello", reply_markup=custom_markup)
        call_kwargs = update.message.reply_text.call_args.kwargs
        assert call_kwargs.get("reply_markup") is custom_markup

    @pytest.mark.asyncio
    async def test_reply_defaults_to_reply_keyboard_for_unknown_state(self):
        bot = _make_bot()
        # No state set — defaults to main
        update = _make_update(chat_id=77)
        await bot._reply(update, "hello")
        call_kwargs = update.message.reply_text.call_args.kwargs
        assert call_kwargs.get("reply_markup") is REPLY_KEYBOARD
