import pytest
from paintress.providers import huggingface as hf_mod
from paintress.providers.huggingface import HuggingFaceProvider
from paintress.types import ImageSpec


# ---------------------------------------------------------------------------
# Shared streaming mock helpers
# ---------------------------------------------------------------------------

def _make_stream_client(status: int, body: bytes, headers: dict | None = None):
    """Build a _Client mock where client.stream(...) is an async ctx-manager."""
    _headers = headers or {"content-type": "image/png"}

    class _Resp:
        status_code = status
        headers = _headers

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

        def json(self):
            return {}

        async def aiter_bytes(self, chunk_size=None):
            yield body

    class _StreamCtx:
        async def __aenter__(self): return _Resp()
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def stream(self, method, url, **kw): return _StreamCtx()

    return _Client


# ---------------------------------------------------------------------------
# Availability tests (no HTTP)
# ---------------------------------------------------------------------------

def test_hf_unavailable_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert HuggingFaceProvider().available() is False


def test_hf_available_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    assert HuggingFaceProvider().available() is True


# ---------------------------------------------------------------------------
# Existing provider tests — adapted for streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hf_posts_and_returns_image(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    _Client = _make_stream_client(200, b"\x89PNG\r\n\x1a\nIMG")
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    data, meta = await HuggingFaceProvider().generate(
        ImageSpec(prompt="a forest", out_dir="/tmp", seed=99)
    )
    assert data.startswith(b"\x89PNG")
    assert meta["seed_used"] == 99


@pytest.mark.asyncio
async def test_hf_503_model_loading_raises(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    _Client = _make_stream_client(
        503,
        b'{"error":"Model is loading","estimated_time":20}',
        headers={"content-type": "application/json"},
    )
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError) as ei:
        await HuggingFaceProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert "model_loading" in str(ei.value)


@pytest.mark.asyncio
async def test_hf_403_gated_raises_auth(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    _Client = _make_stream_client(403, b"", headers={})
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError) as ei:
        await HuggingFaceProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert "hf_auth:403" in str(ei.value)


# ---------------------------------------------------------------------------
# FIX 2: response size cap test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hf_raises_when_response_exceeds_cap(monkeypatch):
    """Stream body exceeding monkeypatched _MAX_BYTES → RuntimeError('response_too_large')."""
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setattr(hf_mod, "_MAX_BYTES", 10)

    class _Resp:
        status_code = 200
        headers = {"content-type": "image/png"}
        def raise_for_status(self): pass
        async def aiter_bytes(self, chunk_size=None):
            yield b"B" * 20  # 20 > 10

    class _StreamCtx:
        async def __aenter__(self): return _Resp()
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def stream(self, method, url, **kw): return _StreamCtx()

    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError, match="response_too_large"):
        await HuggingFaceProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
