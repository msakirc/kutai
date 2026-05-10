"""Tests for vecihi.prior_art — Z1 T6B (P5)."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vecihi.prior_art import (
    DEFAULT_TTL_HOURS,
    SCHEMA_VERSION,
    _build_queries,
    _classify_verdict,
    _dedup_by_name,
    _hash_keywords,
    _read_cache,
    _write_cache,
    find_prior_art,
)


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Helpers — fake aiohttp session
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status: int = 200, json_data=None, text_data: str = ""):
        self.status = status
        self._json = json_data
        self._text = text_data
        self.headers = {"content-type": "application/json"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, content_type=None):
        if self._json is None:
            raise ValueError("not json")
        return self._json

    async def text(self):
        return self._text or (json.dumps(self._json) if self._json else "")


class _FakeSession:
    """Routes per-URL to canned responses; counts calls."""

    def __init__(self, route_map):
        self.route_map = route_map
        self.calls = []
        self.closed = False

    def _match(self, url: str):
        for prefix, resp in self.route_map.items():
            if url.startswith(prefix):
                return resp
        return _FakeResponse(404, json_data={})

    def get(self, url, params=None, timeout=None, **kw):
        self.calls.append(("GET", url, params))
        resp = self._match(url)
        if callable(resp):
            return resp()
        return resp

    def head(self, url, timeout=None, allow_redirects=True, **kw):
        self.calls.append(("HEAD", url, None))
        resp = self._match(url)
        if callable(resp):
            return resp()
        return resp

    async def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestPureHelpers:
    def test_hash_keywords_stable(self):
        a = _hash_keywords(["TrendyOL", "  reseller "])
        b = _hash_keywords(["reseller", "trendyol"])
        assert a == b

    def test_build_queries_seed_plus_suffixes(self):
        q = _build_queries("Reseller arbitrage tool for TR", ["trendyol"])
        assert q
        assert any("trendyol" in s for s in q)
        assert len(q) <= 5

    def test_classify_verdict_graveyard(self):
        attempted = [{"status": "dead"}, {"status": "dormant"}]
        v = _classify_verdict(attempted, total_inspected=30, queries=["a", "b", "c"])
        assert v == "graveyard_well_populated"

    def test_classify_verdict_blue_ocean(self):
        v = _classify_verdict([], 25, ["a", "b", "c"])
        assert v == "blue_ocean_validated"

    def test_classify_verdict_suspicious(self):
        v = _classify_verdict([], 3, ["a"])
        assert v == "blue_ocean_suspicious"

    def test_classify_verdict_thin(self):
        v = _classify_verdict([{"status": "unknown"}], 10, ["a", "b"])
        assert v == "graveyard_thin"

    def test_dedup_by_name_merges_sources(self):
        items = [
            {"name": "Foo", "sources": ["a"]},
            {"name": "foo", "sources": ["b"]},
        ]
        out = _dedup_by_name(items)
        assert len(out) == 1
        assert set(out[0]["sources"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------
class TestCache:
    def test_round_trip(self, tmp_path):
        db = str(tmp_path / "cache.db")
        report = {
            "_schema_version": "1",
            "search_summary": {"sources_used": ["hn_algolia"]},
            "verdict": "graveyard_thin",
        }
        _write_cache("abc", report, db_path=db)
        got = _read_cache("abc", db_path=db)
        assert got is not None
        assert got["verdict"] == "graveyard_thin"

    def test_stale_cache_returns_none(self, tmp_path):
        import datetime as dt
        db = str(tmp_path / "stale.db")
        # write with ttl=0 directly so any age trips it
        from vecihi.prior_art import _ensure_cache_table
        con = sqlite3.connect(db)
        _ensure_cache_table(con)
        con.execute(
            "INSERT INTO prior_art_cache VALUES (?, ?, ?, ?)",
            (
                "key",
                json.dumps({"verdict": "stale"}),
                (dt.datetime.utcnow() - dt.timedelta(hours=200)).isoformat(),
                168,
            ),
        )
        con.commit()
        con.close()
        assert _read_cache("key", db_path=db) is None


# ---------------------------------------------------------------------------
# Happy-path with all 4 sources mocked
# ---------------------------------------------------------------------------
class TestFindPriorArt:
    def _routes_with_two_dead(self):
        hn_data = {
            "hits": [
                {
                    "title": "SellerSpy.tr",
                    "url": "https://sellerspy.tr",
                    "objectID": "1",
                },
                {
                    "title": "ArbitrajPro",
                    "url": "https://arbitrajpro.io",
                    "objectID": "2",
                },
            ]
        }
        wiki_data = {
            "query": {
                "search": [
                    {"title": "Reseller", "snippet": "<i>reseller</i> arbitrage"},
                ]
            }
        }
        wayback_data = {
            "archived_snapshots": {
                "closest": {
                    "available": True,
                    "timestamp": "20190812000000",
                    "url": "http://web.archive.org/web/20190812/sellerspy.tr",
                }
            }
        }

        def hn_resp():
            return _FakeResponse(200, json_data=hn_data)

        def wiki_resp():
            return _FakeResponse(200, json_data=wiki_data)

        def wb_resp():
            return _FakeResponse(200, json_data=wayback_data)

        def head_dead():
            return _FakeResponse(404, json_data={})

        def head_ok():
            return _FakeResponse(200, json_data={})

        return {
            "https://hn.algolia.com": hn_resp,
            "https://en.wikipedia.org/w/api.php": wiki_resp,
            "https://archive.org/wayback/available": wb_resp,
            # Make sellerspy/arbitrajpro fail HEAD => marked dead
            "https://sellerspy.tr": head_dead,
            "https://arbitrajpro.io": head_dead,
            "https://en.wikipedia.org/wiki/Reseller": head_ok,
        }

    def test_happy_path_returns_schema(self, tmp_path):
        db = str(tmp_path / "happy.db")
        sess = _FakeSession(self._routes_with_two_dead())
        report = run_async(find_prior_art(
            idea_summary="Turkish reseller arbitrage tool",
            domain_keywords=["trendyol", "reseller"],
            k=10,
            ambition_tier="private_beta",
            db_path=db,
            session=sess,
        ))
        assert report["_schema_version"] == SCHEMA_VERSION
        assert isinstance(report["attempted_solutions"], list)
        assert "verdict" in report
        assert "search_summary" in report
        assert report["search_summary"]["queries_run"]
        # HN + wiki at minimum should be in sources_used
        assert "hn_algolia" in report["search_summary"]["sources_used"]
        # Cache write happened
        cached = _read_cache(_hash_keywords(["trendyol", "reseller"]), db_path=db)
        assert cached is not None

    def test_url_validation_marks_dead(self, tmp_path):
        db = str(tmp_path / "dead.db")
        sess = _FakeSession(self._routes_with_two_dead())
        report = run_async(find_prior_art(
            # Idea summary shares tokens with SellerSpy / ArbitrajPro so
            # they get classified as attempted (vs adjacent).
            idea_summary="SellerSpy ArbitrajPro reseller arbitrage tool",
            domain_keywords=["reseller"],
            ambition_tier="private_beta",
            db_path=db,
            session=sess,
        ))
        # SellerSpy + ArbitrajPro had 404 HEADs => marked dead either via
        # url-resolution sweep OR via wayback-archived-but-unreachable
        # branch. attempted_solutions or adjacent_failures must contain
        # at least one dead entry.
        all_items = report["attempted_solutions"] + report["adjacent_failures"]
        dead = [s for s in all_items if s.get("status") == "dead"]
        assert dead, f"expected dead entries, got {all_items}"


class TestRateLimitFallback:
    def test_falls_back_to_cache_on_429(self, tmp_path):
        db = str(tmp_path / "rl.db")
        # Pre-populate cache
        cached_report = {
            "_schema_version": "1",
            "search_summary": {
                "queries_run": ["x"],
                "sources_used": ["hn_algolia"],
                "rate_limit_hits": [],
                "total_results_inspected": 5,
                "cache_hit_for_sources": [],
            },
            "attempted_solutions": [{"name": "FromCache", "url": "https://x"}],
            "adjacent_failures": [],
            "key_lessons": [],
            "graveyard_count": 0,
            "verdict": "graveyard_thin",
        }
        _write_cache(_hash_keywords(["foo"]), cached_report, db_path=db)

        def rl():
            return _FakeResponse(429, json_data={})

        sess = _FakeSession({
            "https://hn.algolia.com": rl,
            "https://en.wikipedia.org/w/api.php": rl,
            "https://archive.org/wayback/available": rl,
        })
        report = run_async(find_prior_art(
            idea_summary="anything",
            domain_keywords=["foo"],
            ambition_tier="private_beta",
            db_path=db,
            session=sess,
        ))
        # Cache annotated
        assert report["attempted_solutions"][0]["name"] == "FromCache"
        assert report["search_summary"]["cache_hit_for_sources"]


class TestEmptyResultsFallback:
    def test_three_empty_sources_no_cache(self, tmp_path):
        db = str(tmp_path / "empty.db")

        def empty():
            return _FakeResponse(200, json_data={"hits": [], "query": {"search": []}})

        sess = _FakeSession({
            "https://hn.algolia.com": empty,
            "https://en.wikipedia.org/w/api.php": empty,
            "https://archive.org/wayback/available": empty,
        })
        report = run_async(find_prior_art(
            idea_summary="extremely niche idea",
            domain_keywords=["niche"],
            ambition_tier="private_beta",
            db_path=db,
            session=sess,
        ))
        # No cache exists => fresh report. With queries < 3, verdict is suspicious.
        assert report["verdict"] in (
            "blue_ocean_suspicious", "blue_ocean_validated"
        )
        # No attempted solutions (all empty)
        assert report["attempted_solutions"] == []


class TestProductHuntTier:
    def test_public_launch_includes_ph(self, tmp_path):
        db = str(tmp_path / "ph.db")

        def hn_empty():
            return _FakeResponse(200, json_data={"hits": []})

        def wiki_empty():
            return _FakeResponse(200, json_data={"query": {"search": []}})

        ph_xml = (
            '<?xml version="1.0"?><rss><channel>'
            '<item><title>Reseller Toolkit</title>'
            '<link>https://www.producthunt.com/posts/reseller-toolkit</link>'
            '</item>'
            '<item><title>Unrelated thing</title>'
            '<link>https://www.producthunt.com/posts/other</link>'
            '</item>'
            '</channel></rss>'
        )

        def ph_resp():
            return _FakeResponse(200, text_data=ph_xml)

        def wb_resp():
            return _FakeResponse(200, json_data={"archived_snapshots": {}})

        def head_ok():
            return _FakeResponse(200, json_data={})

        sess = _FakeSession({
            "https://hn.algolia.com": hn_empty,
            "https://en.wikipedia.org/w/api.php": wiki_empty,
            "https://archive.org/wayback/available": wb_resp,
            "https://www.producthunt.com/feed": ph_resp,
            "https://www.producthunt.com/posts/reseller-toolkit": head_ok,
            "https://www.producthunt.com/posts/other": head_ok,
        })
        report = run_async(find_prior_art(
            idea_summary="Reseller toolkit for marketplaces",
            domain_keywords=["reseller"],
            ambition_tier="public_launch",
            db_path=db,
            session=sess,
        ))
        # PH should be hit; at least one PH item should appear (filter on "reseller")
        assert "product_hunt" in report["search_summary"]["sources_used"]
