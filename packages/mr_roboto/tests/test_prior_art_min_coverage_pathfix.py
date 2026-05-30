"""prior_art_min_coverage: workspace-relative report_path + string search_summary.

mission_79 (2026-05-30) DLQ'd prior_art_search (#225583) and its post-hook
(#226311) with problems=['no report payload provided'] even though
workspace/mission_79/.research/prior_art_report.json was present and good.

Two bugs:
1. _load_report used report_path verbatim. Mechanical executors run with
   cwd = repo root, but the payload path is workspace-relative
   ('mission_79/.research/prior_art_report.json'), so os.path.isfile() is
   False -> 'no report payload provided'. Fix: resolve against WORKSPACE_DIR.
2. Latent crash: search_summary in a real report is a STRING, but the code
   did `summary.get('queries_run')` -> AttributeError once the report loads.
"""
from __future__ import annotations

import json

from mr_roboto.prior_art_min_coverage import prior_art_min_coverage, _load_report


_GOOD_REPORT = {
    "search_summary": "Saturated market; gap in shared-errand gamification.",  # STRING
    "attempted_solutions": [
        {"name": "Habitica", "url": "https://habitica.com/", "status": "Active"},
    ],
    "key_lessons": ["Streaks can demotivate."],
    "verdict": "blue_ocean_validated_with_caveats",
}


def test_string_search_summary_does_not_crash():
    """A real report has search_summary as prose (str), not an object."""
    res = prior_art_min_coverage(report=dict(_GOOD_REPORT))
    assert res["ok"] is True, res
    assert res["attempted"] == 1


def test_relative_report_path_resolved_against_workspace(tmp_path, monkeypatch):
    import src.tools.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(tmp_path), raising=False)
    rel = "mission_79/.research/prior_art_report.json"
    abs_p = tmp_path / "mission_79" / ".research" / "prior_art_report.json"
    abs_p.parent.mkdir(parents=True, exist_ok=True)
    abs_p.write_text(json.dumps(_GOOD_REPORT), encoding="utf-8")

    # The dispatch hands a workspace-relative path; cwd is NOT tmp_path.
    payload, err = _load_report(None, rel)
    assert err is None, err
    assert payload is not None
    assert payload["attempted_solutions"][0]["name"] == "Habitica"

    res = prior_art_min_coverage(report=None, report_path=rel)
    assert res["ok"] is True, res


def test_missing_everywhere_still_reports_cleanly():
    payload, err = _load_report(None, "nope/missing.json")
    assert payload is None
    assert err == "no report payload provided"


def test_empty_dict_report_falls_through_to_path(tmp_path, monkeypatch):
    """An empty inline report must not shadow a good report_path."""
    import src.tools.workspace as ws
    monkeypatch.setattr(ws, "WORKSPACE_DIR", str(tmp_path), raising=False)
    abs_p = tmp_path / "r.json"
    abs_p.write_text(json.dumps(_GOOD_REPORT), encoding="utf-8")
    payload, err = _load_report({}, "r.json")
    assert err is None
    assert payload["verdict"] == "blue_ocean_validated_with_caveats"
