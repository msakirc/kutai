import json
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_cmd_image_enqueues_cps_no_await_inline(monkeypatch):
    """/image must enqueue with CPS on_complete/on_error and carry chat_id +
    prompt in cont_state. It must NOT use await_inline."""
    import src.app.telegram_bot as tb

    enq = AsyncMock(return_value=123)
    monkeypatch.setattr("general_beckman.enqueue", enq)

    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._reply = AsyncMock()
    update = MagicMock()
    update.effective_chat.id = 99
    ctx = MagicMock(); ctx.args = ["a", "red", "bicycle"]

    await iface.cmd_image(update, ctx)

    assert enq.await_count == 1
    spec = enq.await_args.args[0]
    kwargs = enq.await_args.kwargs
    assert spec["context"]["image_call"]["prompt"] == "a red bicycle"
    assert spec["context"]["image_call"]["raw_dispatch"] is True
    assert spec["kind"] == "image"
    assert kwargs["on_complete"] == "image_delivery.resume"
    assert kwargs["on_error"] == "image_delivery.err"
    assert kwargs["cont_state"]["chat_id"] == 99
    assert kwargs["cont_state"]["prompt"] == "a red bicycle"
    assert "await_inline" not in kwargs


@pytest.mark.asyncio
async def test_image_delivery_resume_parses_json_string_and_sends_photo(monkeypatch, tmp_path):
    import src.app.telegram_bot as tb
    png = tmp_path / "out.png"; png.write_bytes(b"\x89PNG")

    sent = {}
    class _Bot:
        async def send_photo(self, chat_id, photo, caption=None):
            sent["chat_id"] = chat_id
            sent["bytes"] = photo.read()
            sent["caption"] = caption
    class _App: bot = _Bot()
    class _TG: app = _App()
    tb.set_telegram(_TG())

    result = {"result": json.dumps({"path": str(png), "provider": "pollinations"})}
    await tb._image_delivery_resume(123, result, {"chat_id": 99, "prompt": "a red bicycle"})

    assert sent["chat_id"] == 99
    assert sent["bytes"].startswith(b"\x89PNG")
    assert "bicycle" in (sent["caption"] or "")


@pytest.mark.asyncio
async def test_image_delivery_resume_handles_dict_result_too(monkeypatch, tmp_path):
    import src.app.telegram_bot as tb
    png = tmp_path / "out.png"; png.write_bytes(b"\x89PNG")
    sent = {}
    class _Bot:
        async def send_photo(self, chat_id, photo, caption=None): sent["ok"] = True
    class _App: bot = _Bot()
    class _TG: app = _App()
    tb.set_telegram(_TG())
    await tb._image_delivery_resume(123, {"path": str(png)}, {"chat_id": 99, "prompt": "x"})
    assert sent.get("ok") is True


@pytest.mark.asyncio
async def test_image_delivery_err_reports_failure(monkeypatch):
    import src.app.telegram_bot as tb
    msgs = {}
    class _Bot:
        async def send_message(self, chat_id, text, **kw):
            msgs["chat_id"] = chat_id; msgs["text"] = text
    class _App: bot = _Bot()
    class _TG: app = _App()
    tb.set_telegram(_TG())
    await tb._image_delivery_err(123, {"error": "selection_failure:availability"},
                                 {"chat_id": 99})
    assert msgs["chat_id"] == 99
    assert "failed" in msgs["text"].lower() or "❌" in msgs["text"]


@pytest.mark.asyncio
async def test_cmd_image_no_args_replies_usage():
    import src.app.telegram_bot as tb
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._reply = AsyncMock()
    update = MagicMock(); ctx = MagicMock(); ctx.args = []
    await iface.cmd_image(update, ctx)
    msg = iface._reply.await_args.args[1]
    assert "Usage" in msg or "/image" in msg
