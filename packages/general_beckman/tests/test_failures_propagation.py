import pytest


def test_admission_select_forwards_failed_models(monkeypatch):
    """next_task admission must read task.context['failed_models'] and pass them
    as failures= to fatih_hoca.select. Text path wraps names in Failure objects."""
    import general_beckman as gb
    captured = {}

    def _fake_select(**kw):
        captured.update(kw)
        from fatih_hoca.types import Pick
        from fatih_hoca.registry import ModelInfo
        return Pick(
            model=ModelInfo(name="x", location="cloud", provider="p", litellm_name="p/x"),
            min_time_seconds=0.0,
        )
    monkeypatch.setattr("fatih_hoca.select", _fake_select)

    spec = {
        "kind": "main_work",
        "agent_type": "coder",
        "context": {"failed_models": ["groq/oss-120b", "gemini/2.5-flash"]},
    }
    gb._select_for_admission(spec)  # helper added in this task
    raw = captured.get("failures") or []
    # TEXT path forwards real Failure dataclasses where `.model` is a STRING.
    names = []
    for f in raw:
        n = getattr(f, "model", None) if not isinstance(f, str) else f
        if isinstance(n, str) and n:
            names.append(n)
    assert "groq/oss-120b" in names
    assert "gemini/2.5-flash" in names
    from fatih_hoca.types import Failure
    assert all(isinstance(f, Failure) for f in raw)


def test_admission_select_no_failed_models_passes_empty(monkeypatch):
    import general_beckman as gb

    def _fake_select(**kw):
        assert "failures" in kw
        assert kw["failures"] == []
        from fatih_hoca.types import Pick
        from fatih_hoca.registry import ModelInfo
        return Pick(
            model=ModelInfo(name="x", location="cloud", provider="p", litellm_name="p/x"),
            min_time_seconds=0.0,
        )
    monkeypatch.setattr("fatih_hoca.select", _fake_select)
    gb._select_for_admission({"kind": "main_work", "agent_type": "coder",
                              "context": {}})
