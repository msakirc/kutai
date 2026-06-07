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


@pytest.fixture
def kutai_image(tmp_path):
    """A PNG inside the trusted kutai_images temp root (FIX 6c confinement)."""
    import os
    import tempfile
    root = os.path.join(tempfile.gettempdir(), "kutai_images")
    os.makedirs(root, exist_ok=True)
    png = os.path.join(root, f"test_{os.getpid()}_{id(tmp_path)}.png")
    with open(png, "wb") as f:
        f.write(b"\x89PNG")
    yield png
    try:
        os.remove(png)
    except OSError:
        pass


@pytest.mark.asyncio
async def test_image_delivery_resume_parses_json_string_and_sends_photo(monkeypatch, kutai_image):
    import src.app.telegram_bot as tb
    png = kutai_image

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
async def test_image_delivery_resume_handles_dict_result_too(monkeypatch, kutai_image):
    import src.app.telegram_bot as tb
    png = kutai_image
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


# ─── FIX 3: error-string disclosure sanitization ────────────────────────────


@pytest.mark.asyncio
async def test_image_delivery_err_does_not_leak_raw_error(monkeypatch):
    """_image_delivery_err must NOT send the raw internal error verbatim to the
    user — it must send a generic/friendly message."""
    import src.app.telegram_bot as tb
    raw = "provider_raised:RuntimeError:secret /home/x/.env detail"
    sent = {}
    monkeypatch.setattr(
        tb, "_send_telegram_via_resume",
        AsyncMock(side_effect=lambda chat_id, text, **kw: sent.update(
            chat_id=chat_id, text=text) or True),
    )
    await tb._image_delivery_err(123, {"error": raw}, {"chat_id": 99})
    assert sent["chat_id"] == 99
    # The raw internal detail must not reach the user.
    assert "secret" not in sent["text"]
    assert "/home/x/.env" not in sent["text"]
    assert "RuntimeError" not in sent["text"]
    # Still a clear failure notice.
    assert "❌" in sent["text"] or "failed" in sent["text"].lower()


# ─── FIX 6c: delivery path confinement ──────────────────────────────────────


@pytest.mark.asyncio
async def test_image_delivery_resume_rejects_path_outside_temp_root(
        monkeypatch, tmp_path):
    """A result path outside the kutai_images temp root must NOT be sent —
    confinement against arbitrary-file exfiltration. Generic failure instead."""
    import src.app.telegram_bot as tb
    rogue = tmp_path / "secret.txt"
    rogue.write_bytes(b"top secret")

    photo_called = {"n": 0}
    monkeypatch.setattr(
        tb, "_send_telegram_photo_via_resume",
        AsyncMock(side_effect=lambda *a, **k: photo_called.update(
            n=photo_called["n"] + 1) or True),
    )
    err_sent = {}
    monkeypatch.setattr(
        tb, "_send_telegram_via_resume",
        AsyncMock(side_effect=lambda chat_id, text, **kw: err_sent.update(
            chat_id=chat_id, text=text) or True),
    )
    await tb._image_delivery_resume(
        123, {"path": str(rogue)}, {"chat_id": 99, "prompt": "x"})
    assert photo_called["n"] == 0
    assert err_sent.get("chat_id") == 99


@pytest.mark.asyncio
async def test_image_delivery_resume_accepts_path_inside_temp_root(
        monkeypatch):
    """A result path UNDER the kutai_images temp root is delivered normally."""
    import os
    import tempfile
    import src.app.telegram_bot as tb

    root = os.path.join(tempfile.gettempdir(), "kutai_images")
    os.makedirs(root, exist_ok=True)
    png = os.path.join(root, "ok_inside.png")
    with open(png, "wb") as f:
        f.write(b"\x89PNG")
    try:
        photo_sent = {}
        monkeypatch.setattr(
            tb, "_send_telegram_photo_via_resume",
            AsyncMock(side_effect=lambda chat_id, path, caption=None: photo_sent.update(
                chat_id=chat_id, path=path) or True),
        )
        await tb._image_delivery_resume(
            123, {"path": png}, {"chat_id": 99, "prompt": "x"})
        assert photo_sent.get("chat_id") == 99
    finally:
        os.remove(png)


# ─── FIX 8: close remaining test gaps ───────────────────────────────────────


def test_cmd_image_registered_handler():
    """_setup_handlers must register an `image` CommandHandler."""
    import src.app.telegram_bot as tb

    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    captured = []

    class _App:
        def add_handler(self, handler, *a, **kw):
            captured.append(handler)

    iface.app = _App()
    iface._setup_handlers()

    commands = set()
    for h in captured:
        cmds = getattr(h, "commands", None)
        if cmds:
            commands |= set(cmds)
    assert "image" in commands


@pytest.mark.asyncio
async def test_image_delivery_resume_no_file_sends_error(monkeypatch):
    """_image_delivery_resume with a non-existent path → generic failure via
    _send_telegram_via_resume, send_photo (photo resume) NOT called."""
    import src.app.telegram_bot as tb

    photo_called = {"n": 0}
    monkeypatch.setattr(
        tb, "_send_telegram_photo_via_resume",
        AsyncMock(side_effect=lambda *a, **k: photo_called.update(
            n=photo_called["n"] + 1) or True),
    )
    err_sent = {}
    monkeypatch.setattr(
        tb, "_send_telegram_via_resume",
        AsyncMock(side_effect=lambda chat_id, text, **kw: err_sent.update(
            chat_id=chat_id, text=text) or True),
    )
    await tb._image_delivery_resume(
        123, {"path": "/nonexistent/does/not/exist.png"},
        {"chat_id": 99, "prompt": "x"})
    assert photo_called["n"] == 0
    assert err_sent.get("chat_id") == 99
