from src.workflows.engine.producer_index import (
    build_producer_index, producers_for_reviewer, producer_for_artifact,
)

def _wf():
    return {"steps": [
        {"id": "a", "output_artifacts": ["x", "y"]},
        {"id": "b", "output_artifacts": ["z"]},
        {"id": "rev", "input_artifacts": ["x", "z"], "output_artifacts": ["rev_result"]},
    ]}

def test_build_index_maps_artifact_to_producers():
    idx = build_producer_index(_wf())
    assert idx["x"] == ["a"]
    assert idx["z"] == ["b"]

def test_producers_for_reviewer_dedups_and_resolves():
    idx = build_producer_index(_wf())
    assert sorted(producers_for_reviewer(_wf(), "rev", idx)) == ["a", "b"]

def test_producers_for_reviewer_excludes_self():
    wf = {"steps": [{"id": "rev", "input_artifacts": ["rev_result"], "output_artifacts": ["rev_result"]}]}
    idx = build_producer_index(wf)
    assert producers_for_reviewer(wf, "rev", idx) == []

def test_producer_for_artifact():
    idx = build_producer_index(_wf())
    assert producer_for_artifact("x", idx) == "a"
    assert producer_for_artifact("missing", idx) is None
