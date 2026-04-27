from pathlib import Path

import pytest

from fatih_hoca.cloud.discovery import CloudDiscovery
from fatih_hoca.cloud.types import DiscoveredModel, ProviderResult


class _StubAdapter:
    def __init__(self, name: str, result: ProviderResult):
        self.name = name
        self._result = result
        self.calls = 0

    async def fetch_models(self, api_key: str) -> ProviderResult:
        self.calls += 1
        return self._result


def _ok(provider: str, ids: list[str]) -> ProviderResult:
    return ProviderResult(
        provider=provider, status="ok", auth_ok=True,
        models=[DiscoveredModel(litellm_name=f"{provider}/{i}", raw_id=i) for i in ids],
    )


def _fail(provider: str, status: str = "auth_fail") -> ProviderResult:
    return ProviderResult(provider=provider, status=status, auth_ok=False, error="bad")


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.mark.asyncio
async def test_refresh_all_calls_each_adapter_with_its_key(cache_dir):
    adapters = {"groq": _StubAdapter("groq", _ok("groq", ["a"])),
                "openai": _StubAdapter("openai", _ok("openai", ["b"]))}
    keys = {"groq": "G", "openai": "O"}
    alerts: list = []
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir,
                       alert_fn=lambda *a, **kw: alerts.append((a, kw)))
    results = await d.refresh_all(api_keys=keys)
    assert set(results.keys()) == {"groq", "openai"}
    assert all(r.auth_ok for r in results.values())
    assert adapters["groq"].calls == 1
    assert adapters["openai"].calls == 1


@pytest.mark.asyncio
async def test_refresh_all_skips_provider_without_key(cache_dir):
    adapters = {"groq": _StubAdapter("groq", _ok("groq", ["a"]))}
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir,
                       alert_fn=lambda *a, **kw: None)
    results = await d.refresh_all(api_keys={})
    assert results == {}
    assert adapters["groq"].calls == 0


@pytest.mark.asyncio
async def test_failure_falls_back_to_fresh_cache(cache_dir):
    from fatih_hoca.cloud import cache as cache_mod
    cache_mod.save(cache_dir, "groq",
                   [DiscoveredModel(litellm_name="groq/cached", raw_id="cached")], status="ok")
    adapters = {"groq": _StubAdapter("groq", _fail("groq", "network_error"))}
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir,
                       alert_fn=lambda *a, **kw: None)
    results = await d.refresh_all(api_keys={"groq": "G"})
    assert results["groq"].auth_ok is True  # cache served, treat as enabled
    assert results["groq"].served_from_cache is True
    assert results["groq"].models[0].raw_id == "cached"


@pytest.mark.asyncio
async def test_auth_fail_alerts_via_callback(cache_dir):
    captured: list = []
    adapters = {"groq": _StubAdapter("groq", _fail("groq", "auth_fail"))}
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir,
                       alert_fn=lambda provider, status, error: captured.append((provider, status, error)))
    await d.refresh_all(api_keys={"groq": "G"})
    assert len(captured) == 1
    assert captured[0][0] == "groq"
    assert captured[0][1] == "auth_fail"


@pytest.mark.asyncio
async def test_diff_logs_added_and_removed(cache_dir, caplog):
    from fatih_hoca.cloud import cache as cache_mod
    cache_mod.save(cache_dir, "groq",
                   [DiscoveredModel(litellm_name="groq/old", raw_id="old"),
                    DiscoveredModel(litellm_name="groq/keep", raw_id="keep")], status="ok")
    adapters = {"groq": _StubAdapter("groq", _ok("groq", ["new", "keep"]))}
    d = CloudDiscovery(adapters=adapters, cache_dir=cache_dir,
                       alert_fn=lambda *a, **kw: None)
    import logging
    logging.getLogger("fatih_hoca.cloud.discovery").propagate = True
    with caplog.at_level("INFO", logger="fatih_hoca.cloud.discovery"):
        await d.refresh_all(api_keys={"groq": "G"})
    log_text = " ".join(r.message for r in caplog.records)
    assert "added=" in log_text and "groq/new" in log_text
    assert "removed=" in log_text and "groq/old" in log_text
