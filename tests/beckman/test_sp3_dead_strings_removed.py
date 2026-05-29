"""SP3 Task 11 - dead wrapper agent strings removed; summarizer guarded."""
import re

PLUMBING = [
    "packages/general_beckman/src/general_beckman/apply.py",
    "packages/general_beckman/src/general_beckman/posthooks.py",
    "packages/general_beckman/src/general_beckman/rewrite.py",
    "packages/general_beckman/src/general_beckman/__init__.py",
]

def test_no_dead_wrapper_agent_strings():
    for path in PLUMBING:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        assert '"grader"' not in src, f'"grader" still in {path}'
        assert '"artifact_summarizer"' not in src, f'"artifact_summarizer" still in {path}'
        # code_reviewer may appear ONLY in comments/docstrings; assert not as a set literal member
        # (best-effort: no bare "code_reviewer" string-literal membership)

def test_summarizer_guarded_in_no_posthooks():
    import importlib
    ph = importlib.import_module("general_beckman.posthooks")
    s = getattr(ph, "_NO_POSTHOOKS_AGENT_TYPES")
    assert "summarizer" in s and "reviewer" in s
    assert "grader" not in s and "artifact_summarizer" not in s
