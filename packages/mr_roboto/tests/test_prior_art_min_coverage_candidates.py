import json
from mr_roboto.prior_art_min_coverage import prior_art_min_coverage


def _report(url):
    return {
        "search_summary": {"queries_run": ["a", "b", "c"],
                           "total_results_inspected": 25},
        "attempted_solutions": [
            {"name": "Habitica", "url": url, "status": "dead",
             "sources": ["https://news.ycombinator.com/item?id=1"]}],
        "key_lessons": [{"lesson": "x"}],
        "verdict": "graveyard_thin",
    }


def test_url_not_in_candidates_is_rejected(tmp_path):
    cand = tmp_path / "candidates.json"
    cand.write_text(json.dumps({"candidates": [
        {"name": "RealApp", "url": "https://real.example"}]}), encoding="utf-8")
    out = prior_art_min_coverage(
        report=_report("https://habitica.com"),  # fabricated, not fetched
        candidates_path=str(cand))
    assert out["ok"] is False
    assert any("not in fetched candidates" in p for p in out["problems"])


def test_url_in_candidates_passes(tmp_path):
    cand = tmp_path / "candidates.json"
    cand.write_text(json.dumps({"candidates": [
        {"name": "Habitica", "url": "https://habitica.com"}]}), encoding="utf-8")
    out = prior_art_min_coverage(
        report=_report("https://habitica.com"),
        candidates_path=str(cand))
    assert out["ok"] is True, out["problems"]


def test_no_candidates_path_keeps_legacy_behavior():
    out = prior_art_min_coverage(report=_report("https://habitica.com"))
    assert out["ok"] is True, out["problems"]
