"""public_apis_md adapter — markdown API table -> manifest list."""
from pathlib import Path

import pytest

from yalayut.discovery.sources.public_apis_md import (
    parse_public_apis_md,
    PublicApisAdapter,
)

FIXTURE = Path(__file__).parent / "fixtures" / "public_apis_sample.md"


def test_parses_all_rows():
    manifests = parse_public_apis_md(FIXTURE.read_text())
    names = {m["name"] for m in manifests}
    assert names == {"api-coingecko", "api-alpha-vantage", "api-cat-facts"}


def test_no_auth_row_shape():
    manifests = parse_public_apis_md(FIXTURE.read_text())
    cg = next(m for m in manifests if m["name"] == "api-coingecko")
    assert cg["artifact_type"] == "api"
    assert cg["name_original"] == "CoinGecko"
    assert cg["api"]["auth_type"] == "none"
    assert cg["api"]["auth_env"] is None
    assert cg["api"]["base_url"].startswith("https://")


def test_apikey_row_carries_auth_env():
    manifests = parse_public_apis_md(FIXTURE.read_text())
    av = next(m for m in manifests if m["name"] == "api-alpha-vantage")
    assert av["api"]["auth_type"] == "apikey"
    # auth_env derived from canonical name, uppercased.
    assert av["api"]["auth_env"] == "ALPHA_VANTAGE_API_KEY"


def test_intent_keywords_from_description():
    manifests = parse_public_apis_md(FIXTURE.read_text())
    cg = next(m for m in manifests if m["name"] == "api-coingecko")
    assert "cryptocurrency" in cg["intent_keywords"]


@pytest.mark.asyncio
async def test_adapter_discover(monkeypatch):
    adapter = PublicApisAdapter()

    async def fake_fetch(url):
        return FIXTURE.read_text()

    monkeypatch.setattr(adapter, "_fetch_md", fake_fetch)
    refs = await adapter.discover({"source_id": "github:public-apis/public-apis",
                                   "endpoint": "https://example/README.md"})
    assert len(refs) == 3
    assert all(r["manifest"]["artifact_type"] == "api" for r in refs)
