import pytest
from paintress.providers import pollinations as poll_mod
from paintress.providers.pollinations import PollinationsProvider
from paintress.types import ImageSpec


# ---------------------------------------------------------------------------
# Shared streaming mock helpers
# ---------------------------------------------------------------------------

def _make_stream_client(status: int, body: bytes, headers: dict | None = None):
    """Build a _Client mock where client.stream(...) is an async ctx-manager
    that yields a response with .aiter_bytes(), .status_code, .raise_for_status().
    """
    _headers = headers or {"content-type": "image/png"}

    class _Resp:
        status_code = status
        headers = _headers

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

        async def aiter_bytes(self, chunk_size=None):
            # yield body in one chunk
            yield body

    class _StreamCtx:
        def __init__(self, resp):
            self._resp = resp
        async def __aenter__(self):
            return self._resp
        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, captured=None, **k):
            self._captured = captured

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, **kw):
            if self._captured is not None:
                self._captured["url"] = url
                self._captured["method"] = method
            return _StreamCtx(_Resp())

    return _Client


# ---------------------------------------------------------------------------
# Existing tests — now adapted for streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pollinations_builds_url_returns_bytes(monkeypatch):
    captured = {}
    _Client = _make_stream_client(200, b"\x89PNG\r\n\x1a\nFAKE")

    class _ClientCapturing(_Client):
        def __init__(self, *a, **k):
            super().__init__(*a, captured=captured, **k)

    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _ClientCapturing)
    prov = PollinationsProvider()
    spec = ImageSpec(prompt="a red bicycle", out_dir="/tmp", width=512, height=512, seed=42)
    data, meta = await prov.generate(spec)
    assert data.startswith(b"\x89PNG")
    assert "a%20red%20bicycle" in captured["url"]
    assert "seed=42" in captured["url"]
    assert meta["seed_used"] == 42
    assert prov.available() is True


@pytest.mark.asyncio
async def test_pollinations_escapes_slashes(monkeypatch):
    captured = {}
    _Client = _make_stream_client(200, b"\x89PNG")

    class _ClientCapturing(_Client):
        def __init__(self, *a, **k):
            super().__init__(*a, captured=captured, **k)

    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _ClientCapturing)
    await PollinationsProvider().generate(
        ImageSpec(prompt="ai/ml art", out_dir="/tmp", seed=1)
    )
    assert "/prompt/ai%2Fml%20art" in captured["url"]


@pytest.mark.asyncio
async def test_pollinations_returns_bytes_for_renoir_to_judge(monkeypatch):
    _Client = _make_stream_client(200, b"<html>error</html>", headers={"content-type": "text/html"})
    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _Client)
    data, meta = await PollinationsProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert data == b"<html>error</html>"


# ---------------------------------------------------------------------------
# FIX 2: response size cap test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pollinations_raises_when_response_exceeds_cap(monkeypatch):
    """Stream body that exceeds monkeypatched _MAX_BYTES → RuntimeError('response_too_large')."""
    monkeypatch.setattr(poll_mod, "_MAX_BYTES", 10)

    class _Resp:
        status_code = 200
        headers = {"content-type": "image/png"}

        def raise_for_status(self):
            pass

        async def aiter_bytes(self, chunk_size=None):
            # yield more than 10 bytes
            yield b"A" * 20

    class _StreamCtx:
        async def __aenter__(self): return _Resp()
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def stream(self, method, url, **kw): return _StreamCtx()

    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError, match="response_too_large"):
        await PollinationsProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
