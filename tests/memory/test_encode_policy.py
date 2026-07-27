import importlib
import pytest


def _fresh(monkeypatch, env=None):
    monkeypatch.delenv("KUTAI_ENCODE_POLICY", raising=False)
    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, v)
    import src.memory.encode_policy as ep
    return importlib.reload(ep)


def test_kills_implicit_accepted_feedback(monkeypatch):
    ep = _fresh(monkeypatch)
    allowed, reason = ep.should_store(
        "Feedback on task: X\nType: accepted",
        {"type": "user_feedback", "feedback_type": "accepted"},
    )
    assert allowed is False
    assert reason == "killed_implicit_feedback"


def test_preserves_explicit_correction(monkeypatch):
    ep = _fresh(monkeypatch)
    allowed, reason = ep.should_store(
        "Feedback on task: X\nType: modified\nUser correction: use snake_case",
        {"type": "user_feedback", "feedback_type": "modified"},
    )
    assert allowed is True
    assert reason == ""


def test_allows_normal_task_result(monkeypatch):
    ep = _fresh(monkeypatch)
    allowed, reason = ep.should_store(
        "Task: Build parser\nDescription: parse YAML\nOutcome: success\nResult: done",
        {"type": "task_result"},
    )
    assert allowed is True
    assert reason == ""


def test_pollution_regex_not_applied_to_task_result(monkeypatch):
    """The skill-description pollution regex is mis-fit for multi-line task
    prose and dropped good design artifacts on the live DB, so it must NOT be
    applied to task_result. A keyword-y but long-enough body passes; semantic
    degeneracy is dogru_mu_samet's job at the writer, not this gate's."""
    ep = _fresh(monkeypatch)
    allowed, reason = ep.should_store(
        "Task: X\nResult:\n* STRATEGY: use api_call\nObservation: it worked", {"type": "task_result"}
    )
    assert allowed is True
    assert reason == ""


def test_rejects_too_short_firehose_type(monkeypatch):
    ep = _fresh(monkeypatch)
    allowed, reason = ep.should_store("ok", {"type": "task_result"})
    assert allowed is False
    assert reason == "too_short"


def test_curated_types_are_never_quality_filtered(monkeypatch):
    """Blocker-1 guard: a bulleted / keyword-y fact or ingested doc must pass."""
    ep = _fresh(monkeypatch)
    for mem_type in ("fact", "ingested_document", "user_preference",
                     "cross_agent_insight", "conversation", "code_symbol"):
        allowed, reason = ep.should_store("* STRATEGY: Observation: one line", {"type": mem_type})
        assert allowed is True, f"{mem_type} was wrongly filtered ({reason})"
        assert reason == ""


def test_no_type_passes(monkeypatch):
    """data_type-only writers (web cache, shopping) have no 'type' -> pass."""
    ep = _fresh(monkeypatch)
    allowed, reason = ep.should_store("* bulleted scraped result", {"data_type": "web_result"})
    assert allowed is True
    assert reason == ""


def test_killswitch_disables_all_gating(monkeypatch):
    ep = _fresh(monkeypatch, {"KUTAI_ENCODE_POLICY": "off"})
    allowed, reason = ep.should_store("ok", {"type": "user_feedback", "feedback_type": "accepted"})
    assert allowed is True
    assert reason == ""
