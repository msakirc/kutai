"""Tests for mr_roboto.publish_preview_pages — Unit B GitHub Pages publisher.

All subprocess calls go through _run (imported from init_mission_github_repo);
we monkeypatch that symbol plus the gh-helper booleans and
_resolve_preview_root.  No real git / gh / network calls are made.

Test naming:
  test_happy_path                  — full success: push + pages-enable
  test_pages_already_enabled_409  — Pages API 409 → still ok+url, pending False
  test_no_preview_root             — root=None → pending, no git/gh calls
  test_gh_cli_missing              — gh not on PATH → pending gh_cli_missing
  test_repo_missing_create_invoked — repo absent → _gh_repo_create called
  test_dispatch_smoke              — mr_roboto.run() routes and returns completed
"""
from __future__ import annotations

import importlib
import os
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# Import the module (not just the function) so monkeypatch.setattr works.
_ppp_mod = importlib.import_module("mr_roboto.publish_preview_pages")
_init_mod = importlib.import_module("mr_roboto.init_mission_github_repo")

from mr_roboto.publish_preview_pages import publish_preview_pages


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace_web(tmp_path):
    """Workspace with a non-empty .web/ dir (web-track preview root)."""
    web = tmp_path / ".web"
    web.mkdir()
    (web / "index.html").write_text("<html>Hello</html>", encoding="utf-8")
    return str(tmp_path)


def _make_run_stub(calls_log: list[dict]):
    """
    Return a _run replacement that records every call and returns success
    for all commands.  Special cases:
    - git init -b gh-pages   → (0, "", "")
    - git push -f -u ...     → (0, "", "")
    - gh api ... /pages      → (0, "", "")
    - everything else        → (0, "", "")
    """
    def _stub(cmd: list[str], cwd: str | None = None, timeout: float = 30.0):
        calls_log.append({"cmd": cmd, "cwd": cwd})
        return 0, "", ""
    return _stub


def _make_run_stub_pages_409(calls_log: list[dict]):
    """Like _make_run_stub but the pages-enable call returns 409."""
    def _stub(cmd: list[str], cwd: str | None = None, timeout: float = 30.0):
        calls_log.append({"cmd": cmd, "cwd": cwd})
        if "gh" in cmd and "/pages" in " ".join(cmd):
            return 1, "", "HTTP 409: already exists"
        return 0, "", ""
    return _stub


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path(workspace_web, monkeypatch):
    """Full success: repo exists, root present → push + pages-enable called, URL correct."""
    calls: list[dict] = []

    # Patch _run on the init_mission_github_repo module (where publish_preview_pages
    # imports it from).
    monkeypatch.setattr(_init_mod, "_run", _make_run_stub(calls))
    # _run is imported by name inside publish_preview_pages at module level, but
    # Python's import resolves it through the init_mission_github_repo module.
    # Also patch the symbol in publish_preview_pages' own namespace to be safe.
    monkeypatch.setattr(_ppp_mod, "_run", _make_run_stub(calls))

    monkeypatch.setattr(_ppp_mod, "_gh_available", lambda: (True, "gh 2.0"))
    monkeypatch.setattr(_ppp_mod, "_gh_authenticated", lambda: (True, "logged in"))
    monkeypatch.setattr(_ppp_mod, "_gh_current_user", lambda: "testowner")
    monkeypatch.setattr(_ppp_mod, "_gh_repo_exists", lambda owner, repo: True)

    # _resolve_preview_root: patch inside the emit_preview_url module but since
    # publish_preview_pages does a lazy import, we patch the module attribute.
    _epu_mod = importlib.import_module("mr_roboto.emit_preview_url")
    monkeypatch.setattr(_epu_mod, "_resolve_preview_root",
                        lambda ws: os.path.join(ws, ".web"))

    monkeypatch.delenv("KUTAI_GITHUB_ORG", raising=False)

    res = await publish_preview_pages(mission_id=7, workspace_path=workspace_web)

    assert res["ok"] is True
    assert res["pending"] is False
    assert res["url"] == "https://testowner.github.io/kutai-mission-7-unnamed/"
    assert "github.com/testowner" in res["repo_url"]

    # A git push -f ... gh-pages call must have occurred.
    push_calls = [c for c in calls if "push" in c["cmd"]]
    assert push_calls, "Expected at least one git push call"
    push_args = " ".join(push_calls[0]["cmd"])
    assert "-f" in push_args
    assert "gh-pages" in push_args

    # A gh api ... /pages call must have occurred.
    pages_calls = [c for c in calls if "gh" in c["cmd"] and "/pages" in " ".join(c["cmd"])]
    assert pages_calls, "Expected a gh api .../pages call"


# ---------------------------------------------------------------------------
# Pages already enabled (409) — still success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pages_already_enabled_409(workspace_web, monkeypatch):
    """Pages API returns 409 'already exists' → result is still ok+pending=False+url set."""
    calls: list[dict] = []

    monkeypatch.setattr(_init_mod, "_run", _make_run_stub_pages_409(calls))
    monkeypatch.setattr(_ppp_mod, "_run", _make_run_stub_pages_409(calls))

    monkeypatch.setattr(_ppp_mod, "_gh_available", lambda: (True, "gh 2.0"))
    monkeypatch.setattr(_ppp_mod, "_gh_authenticated", lambda: (True, "ok"))
    monkeypatch.setattr(_ppp_mod, "_gh_current_user", lambda: "myorg")
    monkeypatch.setattr(_ppp_mod, "_gh_repo_exists", lambda o, r: True)

    _epu_mod = importlib.import_module("mr_roboto.emit_preview_url")
    monkeypatch.setattr(_epu_mod, "_resolve_preview_root",
                        lambda ws: os.path.join(ws, ".web"))

    monkeypatch.delenv("KUTAI_GITHUB_ORG", raising=False)

    res = await publish_preview_pages(mission_id=9, workspace_path=workspace_web)

    assert res["ok"] is True
    assert res["pending"] is False
    assert res["url"].startswith("https://myorg.github.io/")
    # No pages_api_warning key (409 is treated as success, not a warning).
    assert "pages_api_warning" not in res


# ---------------------------------------------------------------------------
# No preview root
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_preview_root(tmp_path, monkeypatch):
    """No .prototype or .web → pending, no git/gh subprocess calls made."""
    calls: list[dict] = []
    monkeypatch.setattr(_init_mod, "_run", _make_run_stub(calls))
    monkeypatch.setattr(_ppp_mod, "_run", _make_run_stub(calls))

    monkeypatch.setattr(_ppp_mod, "_gh_available", lambda: (True, "ok"))
    monkeypatch.setattr(_ppp_mod, "_gh_authenticated", lambda: (True, "ok"))
    monkeypatch.setattr(_ppp_mod, "_gh_current_user", lambda: "u")
    monkeypatch.setattr(_ppp_mod, "_gh_repo_exists", lambda o, r: True)

    _epu_mod = importlib.import_module("mr_roboto.emit_preview_url")
    monkeypatch.setattr(_epu_mod, "_resolve_preview_root", lambda ws: None)

    res = await publish_preview_pages(mission_id=1, workspace_path=str(tmp_path))

    assert res["ok"] is True
    assert res["pending"] is True
    assert res["reason"] == "no_preview_root"
    assert res["url"] is None
    # No git / gh subprocess calls.
    assert calls == [], f"Unexpected subprocess calls: {calls}"


# ---------------------------------------------------------------------------
# gh CLI missing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gh_cli_missing(workspace_web, monkeypatch):
    """gh not available → pending gh_cli_missing, no push."""
    calls: list[dict] = []
    monkeypatch.setattr(_init_mod, "_run", _make_run_stub(calls))
    monkeypatch.setattr(_ppp_mod, "_run", _make_run_stub(calls))

    # _resolve_preview_root must return a real root so we get past step 2.
    _epu_mod = importlib.import_module("mr_roboto.emit_preview_url")
    monkeypatch.setattr(_epu_mod, "_resolve_preview_root",
                        lambda ws: os.path.join(ws, ".web"))

    monkeypatch.setattr(_ppp_mod, "_gh_available", lambda: (False, "gh not found"))

    res = await publish_preview_pages(mission_id=2, workspace_path=workspace_web)

    assert res["ok"] is True
    assert res["pending"] is True
    assert res["reason"] == "gh_cli_missing"

    # No push calls.
    push_calls = [c for c in calls if "push" in c["cmd"]]
    assert push_calls == []


# ---------------------------------------------------------------------------
# Repo missing → _gh_repo_create invoked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repo_missing_create_invoked(workspace_web, monkeypatch):
    """When repo doesn't exist, _gh_repo_create is called (and succeeds here)."""
    calls: list[dict] = []
    monkeypatch.setattr(_init_mod, "_run", _make_run_stub(calls))
    monkeypatch.setattr(_ppp_mod, "_run", _make_run_stub(calls))

    monkeypatch.setattr(_ppp_mod, "_gh_available", lambda: (True, "ok"))
    monkeypatch.setattr(_ppp_mod, "_gh_authenticated", lambda: (True, "ok"))
    monkeypatch.setattr(_ppp_mod, "_gh_current_user", lambda: "createowner")
    monkeypatch.setattr(_ppp_mod, "_gh_repo_exists", lambda o, r: False)

    create_calls: list[tuple] = []

    def _fake_create(owner, repo, vis):
        create_calls.append((owner, repo, vis))
        return True, "created"

    monkeypatch.setattr(_ppp_mod, "_gh_repo_create", _fake_create)

    _epu_mod = importlib.import_module("mr_roboto.emit_preview_url")
    monkeypatch.setattr(_epu_mod, "_resolve_preview_root",
                        lambda ws: os.path.join(ws, ".web"))

    monkeypatch.delenv("KUTAI_GITHUB_ORG", raising=False)

    res = await publish_preview_pages(mission_id=5, workspace_path=workspace_web)

    # _gh_repo_create must have been called once.
    assert len(create_calls) == 1
    owner_arg, repo_arg, vis_arg = create_calls[0]
    assert owner_arg == "createowner"
    assert vis_arg == "public"

    assert res["ok"] is True
    assert res["pending"] is False


# ---------------------------------------------------------------------------
# Dispatch smoke: mr_roboto.run() routes publish_preview_pages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_smoke(tmp_path, monkeypatch):
    """mr_roboto.run() routes 'publish_preview_pages' and returns status='completed'."""
    import mr_roboto

    # Patch publish_preview_pages inside the dispatch to fail-soft (pending) so
    # we don't need any real git/gh.  The dispatch returns Action(completed)
    # even when the inner call is pending, because pending is not an error.
    _epu_mod = importlib.import_module("mr_roboto.emit_preview_url")
    monkeypatch.setattr(_epu_mod, "_resolve_preview_root", lambda ws: None)

    # Ensure gh helpers also short-circuit cleanly (no_preview_root fires first).
    monkeypatch.setattr(_ppp_mod, "_gh_available", lambda: (True, "ok"))

    # Create a minimal .web so the workspace exists but root is None (patched above).
    ws = str(tmp_path)

    task = {
        "id": 1,
        "mission_id": 7,
        "payload": {
            "action": "publish_preview_pages",
            "workspace_path": ws,
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed", (
        f"Expected 'completed', got {action.status!r} / {action.error!r}"
    )
    result = action.result or {}
    assert result.get("ok") is True
    # pending is True here because root is None (patched).
    assert result.get("pending") is True
    assert result.get("reason") == "no_preview_root"
