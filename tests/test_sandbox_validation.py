"""Test sandbox bind-mount validation at startup (architectural fix
discussed 2026-04-26).

The container's bind-mount source can drift from current
``WORKSPACE_DIR`` if .env changes between sessions; the stale
container persists across runs. ``validate_or_recreate_sandbox`` runs
once at orchestrator startup, inspects the container, and force-
removes when stale so the next shell call recreates with the right
bind.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.tools.shell import (
    _normalize_path_for_compare,
    validate_or_recreate_sandbox,
)


# ── Path normalization ─────────────────────────────────────────────────


def test_norm_backslash_vs_forward():
    assert _normalize_path_for_compare(
        r"C:\Users\sakir\foo"
    ) == _normalize_path_for_compare("C:/Users/sakir/foo")


def test_norm_case_insensitive():
    assert _normalize_path_for_compare(r"C:\FOO") == _normalize_path_for_compare(
        r"c:\foo"
    )


def test_norm_trailing_slash():
    assert _normalize_path_for_compare("C:/foo/") == _normalize_path_for_compare(
        "C:/foo"
    )


def test_norm_different_paths_unequal():
    assert _normalize_path_for_compare("C:/foo") != _normalize_path_for_compare(
        "C:/bar"
    )


def test_norm_non_string_returns_empty():
    assert _normalize_path_for_compare(None) == ""  # type: ignore[arg-type]
    assert _normalize_path_for_compare(123) == ""  # type: ignore[arg-type]


# ── Validation behaviour ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_skips_when_sandbox_mode_local(monkeypatch):
    monkeypatch.setattr("src.tools.shell.SANDBOX_MODE", "local")
    fake_run = AsyncMock(return_value=(0, "", ""))
    with patch("src.tools.shell._run_quiet", fake_run):
        await validate_or_recreate_sandbox()
    fake_run.assert_not_called()


@pytest.mark.asyncio
async def test_validate_skips_when_sandbox_mode_none(monkeypatch):
    monkeypatch.setattr("src.tools.shell.SANDBOX_MODE", "none")
    fake_run = AsyncMock(return_value=(0, "", ""))
    with patch("src.tools.shell._run_quiet", fake_run):
        await validate_or_recreate_sandbox()
    fake_run.assert_not_called()


@pytest.mark.asyncio
async def test_validate_skips_recreate_when_container_missing(monkeypatch):
    """docker inspect non-zero -> container doesn't exist -> nothing to validate."""
    monkeypatch.setattr("src.tools.shell.SANDBOX_MODE", "docker")

    calls = []

    async def _fake_run(*args):
        calls.append(args)
        return (1, "", "No such container")

    with patch("src.tools.shell._run_quiet", _fake_run):
        await validate_or_recreate_sandbox()

    # Only the inspect call ran; no rm.
    assert len(calls) == 1
    assert "inspect" in calls[0]


@pytest.mark.asyncio
async def test_validate_skips_recreate_when_mount_matches(monkeypatch):
    monkeypatch.setattr("src.tools.shell.SANDBOX_MODE", "docker")
    monkeypatch.setattr(
        "src.tools.shell.WORKSPACE_DIR",
        r"C:\Users\sakir\workspace",
    )

    calls = []

    async def _fake_run(*args):
        calls.append(args)
        if "inspect" in args:
            # Mount source matches expected (with backslash variation).
            return (0, "C:/Users/sakir/workspace", "")
        return (0, "", "")

    with patch("src.tools.shell._run_quiet", _fake_run):
        await validate_or_recreate_sandbox()

    # Only inspect; no rm.
    assert any("inspect" in c for c in calls)
    assert not any("rm" in c for c in calls)


@pytest.mark.asyncio
async def test_validate_recreates_when_mount_stale(monkeypatch):
    monkeypatch.setattr("src.tools.shell.SANDBOX_MODE", "docker")
    monkeypatch.setattr(
        "src.tools.shell.WORKSPACE_DIR",
        r"C:\Users\sakir\workspace",
    )

    calls = []

    async def _fake_run(*args):
        calls.append(args)
        if "inspect" in args:
            return (0, r"C:\Users\sakir\src\workspace", "")  # stale path
        return (0, "", "")

    with patch("src.tools.shell._run_quiet", _fake_run):
        await validate_or_recreate_sandbox()

    # Inspect + rm both fired.
    assert any("inspect" in c for c in calls)
    rm_calls = [c for c in calls if "rm" in c]
    assert len(rm_calls) == 1


@pytest.mark.asyncio
async def test_validate_recreates_when_no_mount(monkeypatch):
    """Container exists but lacks /app/workspace mount entirely."""
    monkeypatch.setattr("src.tools.shell.SANDBOX_MODE", "docker")
    monkeypatch.setattr(
        "src.tools.shell.WORKSPACE_DIR", r"C:\Users\sakir\workspace",
    )

    calls = []

    async def _fake_run(*args):
        calls.append(args)
        if "inspect" in args:
            return (0, "", "")  # empty mount source
        return (0, "", "")

    with patch("src.tools.shell._run_quiet", _fake_run):
        await validate_or_recreate_sandbox()

    rm_calls = [c for c in calls if "rm" in c]
    assert len(rm_calls) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
