"""prior_art_min_coverage must not crash on non-dict attempted_solutions items.

mission_79 (2026-05-30, post-restart): prior_art_search (#225583) DLQ'd with
"prior_art_min_coverage: 'str' object has no attribute 'get'". The on-disk
report had dict items, but the producer emitted a malformed report whose
``attempted_solutions`` was a list of bare strings (["Habitica", ...]); the
Rule-2 / Rule-4 loops called ``sol.get("url")`` on each and crashed. The
earlier search_summary guard fixed one .get-on-str; the loop items were the
other. A malformed entry must surface as a problem, never an exception.
"""
from __future__ import annotations

from mr_roboto.prior_art_min_coverage import prior_art_min_coverage


def test_string_attempted_items_do_not_crash():
    res = prior_art_min_coverage(report={
        "verdict": "red_ocean",
        "attempted_solutions": ["Habitica", "Habitify"],
        "key_lessons": ["streaks demotivate"],
    })
    assert "crashed" not in " ".join(res.get("problems", []))
    assert res["ok"] is False  # malformed entries → flagged, not passed
    assert any("attempted_solutions" in p for p in res["problems"])


def test_mixed_dict_and_string_items():
    res = prior_art_min_coverage(report={
        "verdict": "red_ocean",
        "attempted_solutions": [
            {"name": "Habitica", "url": "https://habitica.com/", "status": "Active"},
            "Habitify",  # malformed
        ],
        "key_lessons": ["x"],
    })
    assert "crashed" not in " ".join(res.get("problems", []))
    # the one bad entry is flagged; the good one is fine
    assert any("[1]" in p for p in res["problems"])


def test_all_dict_items_still_pass():
    res = prior_art_min_coverage(report={
        "verdict": "red_ocean",
        "attempted_solutions": [
            {"name": "Habitica", "url": "https://habitica.com/", "status": "Active"},
        ],
        "key_lessons": ["x"],
    })
    assert res["ok"] is True, res


def test_dead_entry_with_string_item_no_crash():
    """Rule-4 (dead/dormant) loop must also tolerate non-dict items."""
    res = prior_art_min_coverage(report={
        "verdict": "red_ocean",
        "attempted_solutions": ["just a name"],
        "key_lessons": ["x"],
    })
    assert "crashed" not in " ".join(res.get("problems", []))
