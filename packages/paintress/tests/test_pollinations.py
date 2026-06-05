import pytest
from paintress.providers.pollinations import PollinationsProvider
from paintress.types import ImageSpec


@pytest.mark.asyncio
async def test_pollinations_builds_url_returns_bytes(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        content = b"\x89PNG\r\n\x1a\nFAKE"
        headers = {"content-type": "image/png"}
        def raise_for_status(self): pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, **kw):
            captured["url"] = url
            return _Resp()

    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _Client)
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
    class _Resp:
        status_code = 200; content = b"\x89PNG"; headers = {}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def get(self, url, **kw):
            captured["url"] = url
            return _Resp()
    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _Client)
    await PollinationsProvider().generate(
        ImageSpec(prompt="ai/ml art", out_dir="/tmp", seed=1)
    )
    assert "/prompt/ai%2Fml%20art" in captured["url"]


@pytest.mark.asyncio
async def test_pollinations_returns_bytes_for_renoir_to_judge(monkeypatch):
    class _Resp:
        status_code = 200
        content = b"<html>error</html>"
        headers = {"content-type": "text/html"}
        def raise_for_status(self): pass
    class _Client:
        def __init__(self,*a,**k): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): return False
        async def get(self, url, **kw): return _Resp()
    monkeypatch.setattr("paintress.providers.pollinations.httpx.AsyncClient", _Client)
    data, meta = await PollinationsProvider().generate(ImageSpec(prompt="x", out_dir="/tmp"))
    assert data == b"<html>error</html>"
