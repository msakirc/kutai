import pytest
from paintress.providers.huggingface import HuggingFaceProvider
from paintress.types import ImageSpec


def test_hf_unavailable_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert HuggingFaceProvider().available() is False


def test_hf_available_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    assert HuggingFaceProvider().available() is True


@pytest.mark.asyncio
async def test_hf_posts_and_returns_image(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    class _Resp:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\nIMG"
        headers = {"content-type": "image/png"}
        def json(self): return {}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def post(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    data, meta = await HuggingFaceProvider().generate(
        ImageSpec(prompt="a forest", out_dir="/tmp", seed=99)
    )
    assert data.startswith(b"\x89PNG")
    assert meta["seed_used"] == 99


@pytest.mark.asyncio
async def test_hf_503_model_loading_raises(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    class _Resp:
        status_code = 503
        content = b'{"error":"Model is loading","estimated_time":20}'
        headers = {"content-type": "application/json"}
        def json(self): return {"error": "Model is loading", "estimated_time": 20}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def post(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError) as ei:
        await HuggingFaceProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert "model_loading" in str(ei.value)


@pytest.mark.asyncio
async def test_hf_403_gated_raises_auth(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    class _Resp:
        status_code = 403; content = b''
        headers = {}
        def json(self): return {}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def post(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.huggingface.httpx.AsyncClient", _Client)
    with pytest.raises(RuntimeError) as ei:
        await HuggingFaceProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert "hf_auth:403" in str(ei.value)
