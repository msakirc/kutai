"""Tests for prior_art_min_coverage — Z1 T6B (P5) post-hook on step 1.0."""
from __future__ import annotations

import json

import pytest

from mr_roboto.prior_art_min_coverage import prior_art_min_coverage


def _good_report():
    return {
        "_schema_version": "1",
        "search_summary": {
            "queries_run": ["q1", "q2", "q3"],
            "sources_used": ["hn_algolia", "wikipedia"],
            "rate_limit_hits": [],
            "total_results_inspected": 30,
            "cache_hit_for_sources": [],
        },
        "attempted_solutions": [
            {
                "name": "SellerSpy",
                "url": "https://sellerspy.tr",
                "status": "dead",
                "wayback_first_capture": "20190101",
                "sources": ["https://web.archive.org/.../sellerspy.tr"],
            },
            {
                "name": "ArbitrajPro",
                "url": "https://arbitrajpro.io",
                "status": "dormant",
                "sources": ["https://news.ycombinator.com/item?id=42"],
            },
        ],
        "adjacent_failures": [],
        "key_lessons": [
            {
                "lesson_id": "L-001",
                "lesson": "ToS is the kill vector",
                "evidence_refs": ["agent_inference"],
                "applies_to_us": "Use official APIs.",
            }
        ],
        "graveyard_count": 2,
        "verdict": "graveyard_well_populated",
    }


class TestHappyPath:
    def test_well_populated_passes(self):
        res = prior_art_min_coverage(report=_good_report())
        assert res["ok"] is True
        assert res["problems"] == []
        assert res["attempted"] == 2

    def test_load_from_path(self, tmp_path):
        path = tmp_path / "report.json"
        path.write_text(json.dumps(_good_report()), encoding="utf-8")
        res = prior_art_min_coverage(report_path=str(path))
        assert res["ok"] is True


class TestRule1AttemptedCount:
    def test_blue_ocean_with_coverage_passes(self):
        rep = {
            "_schema_version": "1",
            "search_summary": {
                "queries_run": ["a", "b", "c"],
                "total_results_inspected": 25,
            },
            "attempted_solutions": [],
            "key_lessons": [],
            "verdict": "blue_ocean_validated",
        }
        res = prior_art_min_coverage(report=rep)
        assert res["ok"] is True

    def test_blue_ocean_without_coverage_fails(self):
        rep = {
            "_schema_version": "1",
            "search_summary": {
                "queries_run": ["a"],
                "total_results_inspected": 3,
            },
            "attempted_solutions": [],
            "key_lessons": [],
            "verdict": "blue_ocean_validated",
        }
        res = prior_art_min_coverage(report=rep)
        assert res["ok"] is False
        assert any("blue_ocean" in p for p in res["problems"])

    def test_no_attempted_non_blue_ocean_fails(self):
        rep = {
            "_schema_version": "1",
            "search_summary": {"queries_run": ["a", "b", "c"], "total_results_inspected": 5},
            "attempted_solutions": [],
            "key_lessons": [],
            "verdict": "blue_ocean_suspicious",
        }
        res = prior_art_min_coverage(report=rep)
        assert res["ok"] is False
        assert any("verdict" in p for p in res["problems"])


class TestRule2URLShape:
    def test_missing_url_fails(self):
        rep = _good_report()
        rep["attempted_solutions"][0]["url"] = ""
        res = prior_art_min_coverage(report=rep)
        assert res["ok"] is False
        assert any("url missing" in p for p in res["problems"])

    def test_non_http_url_fails(self):
        rep = _good_report()
        rep["attempted_solutions"][0]["url"] = "ftp://nope"
        res = prior_art_min_coverage(report=rep)
        assert res["ok"] is False


class TestRule3KeyLessons:
    def test_attempted_without_lessons_fails(self):
        rep = _good_report()
        rep["key_lessons"] = []
        res = prior_art_min_coverage(report=rep)
        assert res["ok"] is False
        assert any("key_lessons" in p for p in res["problems"])


class TestRule4DeadEvidence:
    def test_dead_without_wayback_or_hn_fails(self):
        rep = _good_report()
        # Strip the wayback evidence and HN reference from the first dead entry
        rep["attempted_solutions"][0].pop("wayback_first_capture", None)
        rep["attempted_solutions"][0]["sources"] = ["https://random.example/article"]
        # Also strip the second dormant entry's HN evidence
        rep["attempted_solutions"][1]["sources"] = ["https://other.example"]
        res = prior_art_min_coverage(report=rep)
        assert res["ok"] is False
        assert any("unverifiable" in p for p in res["problems"])

    def test_dead_with_hn_passes(self):
        rep = _good_report()
        rep["attempted_solutions"][0].pop("wayback_first_capture", None)
        rep["attempted_solutions"][0]["sources"] = [
            "https://news.ycombinator.com/item?id=99"
        ]
        res = prior_art_min_coverage(report=rep)
        assert res["ok"] is True


class TestErrorHandling:
    def test_missing_report_fails(self):
        res = prior_art_min_coverage()
        assert res["ok"] is False
        assert "missing" in res["problems"][0] or "no report" in res["problems"][0]

    def test_bad_path_fails(self):
        res = prior_art_min_coverage(report_path="/nonexistent/path.json")
        assert res["ok"] is False
