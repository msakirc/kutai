"""Z6 T3D — real_tool_kind resolver tests."""
from __future__ import annotations

import pytest

from src.integrations import resolver as resolver_mod
from src.integrations.resolver import resolve_real_tool, _split_kinds


# ---------------------------------------------------------------------------


class _FakeRegistry:
    def __init__(self, adapters: set[str]):
        self._a = adapters

    def get(self, name):
        return object() if name in self._a else None


def _install(monkeypatch, adapters: set[str], credentials: dict[str, dict]):
    import src.integrations.registry as reg_mod
    import src.security.credential_store as cs_mod

    monkeypatch.setattr(
        reg_mod, "get_integration_registry",
        lambda: _FakeRegistry(adapters),
    )

    async def _get_credential(service_name: str):
        return credentials.get(service_name)

    monkeypatch.setattr(cs_mod, "get_credential", _get_credential)


# ---------------------------------------------------------------------------
# _split_kinds
# ---------------------------------------------------------------------------


def test_split_kinds_single():
    assert _split_kinds("vercel") == ["vercel"]


def test_split_kinds_pipe_list():
    assert _split_kinds("vercel|railway|supabase") == [
        "vercel", "railway", "supabase",
    ]


def test_split_kinds_empty():
    assert _split_kinds("") == []
    assert _split_kinds(None) == []
    assert _split_kinds([]) == []


def test_split_kinds_strips_whitespace():
    assert _split_kinds(" vercel | railway ") == ["vercel", "railway"]


def test_split_kinds_list_input():
    assert _split_kinds(["vercel", "railway"]) == ["vercel", "railway"]


# ---------------------------------------------------------------------------
# resolve_real_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_none_when_empty():
    assert (await resolve_real_tool("")) is None
    assert (await resolve_real_tool(None)) is None


@pytest.mark.asyncio
async def test_single_kind_with_adapter_and_cred(monkeypatch):
    _install(
        monkeypatch,
        adapters={"vercel"},
        credentials={"vercel": {"token": "abc"}},
    )
    assert (await resolve_real_tool("vercel")) == "vercel"


@pytest.mark.asyncio
async def test_first_match_in_pipe_list(monkeypatch):
    _install(
        monkeypatch,
        adapters={"vercel", "railway", "supabase"},
        credentials={
            "vercel": {"token": "v"}, "railway": {"token": "r"},
        },
    )
    # Both vercel and railway are eligible; first one wins.
    assert (await resolve_real_tool("vercel|railway|supabase")) == "vercel"


@pytest.mark.asyncio
async def test_fallback_when_first_lacks_creds(monkeypatch):
    """First kind has adapter but no creds; second has both — pick second."""
    _install(
        monkeypatch,
        adapters={"vercel", "railway"},
        credentials={"railway": {"token": "r"}},  # vercel missing creds
    )
    assert (await resolve_real_tool("vercel|railway")) == "railway"


@pytest.mark.asyncio
async def test_fallback_when_first_lacks_adapter(monkeypatch):
    """First kind has no adapter; second has both — pick second."""
    _install(
        monkeypatch,
        adapters={"sentry"},
        credentials={"sentry": {"token": "s"}},
    )
    assert (await resolve_real_tool("datadog|sentry|new_relic")) == "sentry"


@pytest.mark.asyncio
async def test_none_when_no_kind_qualifies(monkeypatch):
    """No adapter for any kind → None."""
    _install(
        monkeypatch,
        adapters=set(),  # nothing registered
        credentials={},
    )
    assert (await resolve_real_tool("vercel|railway")) is None


@pytest.mark.asyncio
async def test_none_when_adapter_but_no_creds_anywhere(monkeypatch):
    """Adapter exists but no creds for any kind → None (admission then
    emits credential_paste via its existing path).
    """
    _install(
        monkeypatch,
        adapters={"vercel", "railway"},
        credentials={},
    )
    assert (await resolve_real_tool("vercel|railway")) is None


# ---------------------------------------------------------------------------
# Admission wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admission_uses_resolver(monkeypatch):
    """When the first pipe-list kind has no creds but the second does, the
    admission gate matches on the second — so credential_paste isn't emitted
    against the wrong kind.
    """
    from general_beckman.z6_admission import _resolve_adapter_with_cred

    _install(
        monkeypatch,
        adapters={"vercel", "railway"},
        credentials={"railway": {"token": "r"}},
    )
    match = await _resolve_adapter_with_cred(["vercel", "railway"])
    assert match == "railway"
