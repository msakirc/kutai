import asyncio
import pytest
from src.research import prior_art


class _FakeResp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self, content_type=None):
        return self._payload

    async def text(self):
        return ""


class _FakeSession:
    """Returns HN hits for the algolia URL, empty otherwise. HEAD ok."""
    def __init__(self):
        self.closed = False

    def get(self, url, params=None, timeout=None):
        if "algolia" in url:
            return _FakeResp(200, {"hits": [
                {"title": "Habitica", "url": "https://habitica.com",
                 "objectID": "1"},
            ]})
        return _FakeResp(200, {})

    def head(self, url, timeout=None, allow_redirects=True):
        return _FakeResp(200, {})

    async def close(self):
        self.closed = True


def test_fetch_candidates_returns_real_urls_only(tmp_path):
    sess = _FakeSession()
    out = asyncio.run(prior_art.fetch_candidates(
        queries=["habit tracker app"],
        domain_keywords=["habit tracker"],
        ambition_tier="private_beta",
        k=10,
        session=sess,
        db_path=str(tmp_path / "c.db"),
    ))
    assert isinstance(out, dict)
    cands = out["candidates"]
    assert cands, "expected at least one fetched candidate"
    for c in cands:
        assert isinstance(c["url"], str) and c["url"].startswith("http")
    assert "queries_run" in out["search_summary"]
    assert out["search_summary"]["total_results_inspected"] >= 1


def test_fetch_candidates_empty_when_no_hits(tmp_path):
    class _Empty(_FakeSession):
        def get(self, url, params=None, timeout=None):
            return _FakeResp(200, {})
    out = asyncio.run(prior_art.fetch_candidates(
        queries=["nonexistent"], domain_keywords=["nope"],
        session=_Empty(), db_path=str(tmp_path / "c.db"),
    ))
    assert out["candidates"] == []
