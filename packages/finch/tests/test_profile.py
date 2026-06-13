from finch.profile import Profile

def test_profile_exposes_runtime_surface():
    p = Profile(
        name="summarizer",
        description="x",
        system_prompt="You are a summarization specialist...",
        allowed_tools=["read_file"],
        max_iterations=3,
    )
    # duck-typed surface coulson/runtime consume:
    assert p.name == "summarizer"
    assert p.allowed_tools == ["read_file"]
    assert p.max_iterations == 3
    assert p.get_system_prompt({"id": 1}) == "You are a summarization specialist..."
    # runtime-mutable attrs default correctly:
    assert p._prompt_version_override is None
    assert p._suppress_clarification is False
    assert p.progress_callback is None
    # defaults for unspecified profile fields:
    assert p.default_tier == "cheap"
    assert p.execution_pattern == "react_loop"
    assert p.enable_self_reflection is False
    assert p.confidence_gate == "fail_closed"

def test_profile_get_system_prompt_static_returns_seed():
    p = Profile(name="x", description="d", system_prompt="SEED")
    assert p.get_system_prompt({"anything": True}) == "SEED"
