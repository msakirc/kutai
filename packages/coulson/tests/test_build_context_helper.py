"""coulson.context.build_context — ported off BaseAgent._build_context (Task 5.5).

Exercises the helper directly with build_user_context mocked: it must return the
context string and apply skill-injected tools to a per-execution copy of
profile.allowed_tools (NOT mutate the shared attribute), via the
_original_allowed_tools snapshot pattern.
"""
import asyncio
import types

import coulson.context as ctx


def _make_profile(allowed_tools):
    p = types.SimpleNamespace()
    p.name = "tester"
    p.allowed_tools = allowed_tools
    return p


def test_build_context_returns_string_no_injection(monkeypatch):
    async def fake_build_user_context(profile, task, *, model_ctx=4096):
        assert model_ctx == 4096
        return "CTX", []

    monkeypatch.setattr(ctx, "build_user_context", fake_build_user_context)
    p = _make_profile(["read_file"])

    out = asyncio.run(ctx.build_context(p, {"id": 1}))

    assert out == "CTX"
    # No injection → allowed_tools untouched, no snapshot created.
    assert p.allowed_tools == ["read_file"]
    assert not hasattr(p, "_original_allowed_tools")


def test_build_context_injects_skills_into_copy(monkeypatch):
    async def fake_build_user_context(profile, task, *, model_ctx=4096):
        return "CTX", ["web_search", "extract_url"]

    monkeypatch.setattr(ctx, "build_user_context", fake_build_user_context)
    shared = ["read_file"]
    p = _make_profile(shared)

    out = asyncio.run(ctx.build_context(p, {"id": 1}))

    assert out == "CTX"
    # Injected tools landed on a per-execution copy...
    assert p.allowed_tools == ["read_file", "web_search", "extract_url"]
    # ...not on the original shared list (bug-fix invariant).
    assert shared == ["read_file"]
    # Snapshot captured the original so execute()'s finally can restore it.
    assert p._original_allowed_tools == ["read_file"]


def test_build_context_appends_to_existing_snapshot(monkeypatch):
    """If a prior setup step (tools_hint/auto-strip) already snapshotted,
    build_context appends to the existing mutable list, not re-snapshot."""
    async def fake_build_user_context(profile, task, *, model_ctx=4096):
        return "CTX", ["web_search"]

    monkeypatch.setattr(ctx, "build_user_context", fake_build_user_context)
    p = _make_profile(["read_file", "file_tree"])
    # Simulate execute()'s _apply_tools_hint snapshot already in place.
    p._original_allowed_tools = ["read_file"]

    asyncio.run(ctx.build_context(p, {"id": 1}))

    assert p.allowed_tools == ["read_file", "file_tree", "web_search"]
    # Existing snapshot preserved (not overwritten).
    assert p._original_allowed_tools == ["read_file"]
