from general_beckman.apply import _posthook_agent_and_payload


class _A:
    kind = "prior_art_min_coverage"
    source_task_id = 42


def test_candidates_path_threaded_from_source_ctx():
    source_ctx = {
        "produces": ["mission_80/.research/prior_art_report.json"],
        "candidates_path": "mission_80/.research/prior_art_candidates.json",
    }
    # source is passed but source_ctx is the third arg (separate from source)
    runner, payload = _posthook_agent_and_payload(_A(), {}, source_ctx)
    assert runner == "mechanical"
    assert payload["payload"]["candidates_path"] == \
        "mission_80/.research/prior_art_candidates.json"
    assert payload["payload"]["report_path"] == \
        "mission_80/.research/prior_art_report.json"


def test_candidates_path_absent_is_none():
    source_ctx = {
        "produces": ["mission_80/.research/prior_art_report.json"],
    }
    runner, payload = _posthook_agent_and_payload(_A(), {}, source_ctx)
    assert runner == "mechanical"
    assert payload["payload"].get("candidates_path") is None
    assert payload["payload"]["report_path"] == \
        "mission_80/.research/prior_art_report.json"
