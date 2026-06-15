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


async def _noop_web(queries, domain_keywords):
    """Hermetic web searcher: never touches the network."""
    return []


def test_fetch_candidates_returns_real_urls_only(tmp_path):
    sess = _FakeSession()
    out = asyncio.run(prior_art.fetch_candidates(
        queries=["habit tracker app"],
        domain_keywords=["habit tracker"],
        ambition_tier="private_beta",
        k=10,
        session=sess,
        web_searcher=_noop_web,
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
        session=_Empty(), web_searcher=_noop_web,
        db_path=str(tmp_path / "c.db"),
    ))
    assert out["candidates"] == []


# --- A: query relaxation -------------------------------------------------

def test_hn_relaxes_long_query_when_empty(tmp_path):
    """A long graveyard-biased phrase yields 0 HN hits; the fetcher must
    retry with a relaxed core query (graveyard/generic modifiers stripped,
    capped to a few core tokens) and surface the candidate."""
    class _RelaxSession(_FakeSession):
        def get(self, url, params=None, timeout=None):
            if "algolia" in url:
                q = (params or {}).get("query", "")
                # Only the short core query returns a hit; the long phrase 0.
                if q.strip().lower() == "gamified habit tracker":
                    return _FakeResp(200, {"hits": [
                        {"title": "Habitica", "url": "https://habitica.com",
                         "objectID": "1"},
                    ]})
                return _FakeResp(200, {"hits": []})
            return _FakeResp(200, {})

    out = asyncio.run(prior_art.fetch_candidates(
        queries=["gamified habit tracker mobile app dead startup"],
        domain_keywords=["gamified habit tracker"],
        ambition_tier="private_beta",
        session=_RelaxSession(), web_searcher=_noop_web,
        db_path=str(tmp_path / "c.db"),
    ))
    names = [c.get("name") for c in out["candidates"]]
    assert "Habitica" in names, (
        f"relaxation should have surfaced the candidate; got {names}")


# --- B: general web-search tier -----------------------------------------

def test_web_search_tier_populates_when_hn_wiki_empty(tmp_path):
    """When HN + Wikipedia return nothing, the injected web searcher's
    results must be normalized into candidates and the source recorded."""
    class _Empty(_FakeSession):
        def get(self, url, params=None, timeout=None):
            return _FakeResp(200, {})

    async def _web(queries, domain_keywords):
        return [
            {"title": "HabitForge", "href": "https://habitforge.io/",
             "body": "Gamified habit tracker app, earn XP."},
        ]

    out = asyncio.run(prior_art.fetch_candidates(
        queries=["gamified habit tracker dead startup"],
        domain_keywords=["gamified habit tracker"],
        ambition_tier="private_beta",
        session=_Empty(), web_searcher=_web,
        db_path=str(tmp_path / "c.db"),
    ))
    names = [c.get("name") for c in out["candidates"]]
    assert "HabitForge" in names, f"web tier candidate missing; got {names}"
    assert "web_search" in out["search_summary"]["sources_used"]
    assert out["search_summary"]["total_results_inspected"] >= 1
    hf = next(c for c in out["candidates"] if c.get("name") == "HabitForge")
    assert hf["url"] == "https://habitforge.io/"
