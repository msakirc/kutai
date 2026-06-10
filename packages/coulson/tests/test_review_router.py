from coulson.posthooks.review_router import (
    map_tagged_issues, build_router_prompt, parse_router_assignment,
)

def test_tagged_issues_group_by_producer():
    index = {"requirements_spec": ["3.4"], "prd_final": ["2.11"]}
    issues = [
        {"target_artifact": "requirements_spec", "severity": "blocker", "problem": "no traceability"},
        {"target_artifact": "prd_final", "severity": "major", "problem": "thin NFRs"},
        {"target_artifact": "requirements_spec", "severity": "minor", "problem": "typo"},
    ]
    grouped, unresolved = map_tagged_issues(issues, index)
    assert set(grouped) == {"3.4", "2.11"}
    assert len(grouped["3.4"]) == 2
    assert unresolved == []

def test_untagged_and_unmappable_go_to_unresolved():
    index = {"requirements_spec": ["3.4"]}
    issues = [
        {"target_artifact": None, "severity": "blocker", "problem": "systemic"},
        {"target_artifact": "ghost", "severity": "blocker", "problem": "no producer"},
    ]
    grouped, unresolved = map_tagged_issues(issues, index)
    assert grouped == {}
    assert len(unresolved) == 2

def test_build_router_prompt_lists_candidates():
    prompt = build_router_prompt(
        issue={"problem": "auth flow undefined", "severity": "blocker"},
        candidates=[("3.4", "requirements_spec"), ("2.11", "prd_final")],
    )
    assert "auth flow undefined" in prompt
    assert "3.4" in prompt and "2.11" in prompt
    assert "unknown" in prompt.lower()

def test_parse_router_assignment_picks_step():
    assert parse_router_assignment("STEP: 3.4", ["3.4", "2.11"]) == "3.4"

def test_parse_router_assignment_unknown():
    assert parse_router_assignment("STEP: unknown", ["3.4"]) is None

def test_parse_router_rejects_hallucinated_step():
    assert parse_router_assignment("STEP: 9.9", ["3.4"]) is None
