"""Tests for mr_roboto.init_mission_github_repo — Z1 T6C (C18).

GitHub repo init at end of phase 6. ALL subprocess calls are mocked; this
suite never spawns a real `gh` or `git` invocation.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import mr_roboto.init_mission_github_repo as gh_mod
from mr_roboto.init_mission_github_repo import (
    init_mission_github_repo,
    _compute_repo_name,
    _kebab,
    _read_charter_title,
    _populate_export,
)
# Alias to keep module-level patch targets working even if a sibling import
# in mr_roboto.__init__ shadows the submodule.
assert gh_mod is not None


@pytest.fixture
def workspace(tmp_path):
    """Mission workspace with a charter + a couple of artifacts."""
    ws = tmp_path / "mission_42"
    ws.mkdir()
    (ws / "charter.md").write_text(
        "---\ntitle: Coffee Habit Tracker\n---\n\n# Coffee Habit Tracker\n\nbody.",
        encoding="utf-8",
    )
    (ws / "non_goals.md").write_text("- nope\n", encoding="utf-8")
    (ws / "premortem.md").write_text("died.", encoding="utf-8")
    adr_dir = ws / "adr"
    adr_dir.mkdir()
    (adr_dir / "001-stack.md").write_text("adr body", encoding="utf-8")
    return str(ws)


# ── Pure helpers ────────────────────────────────────────────────────────────


def test_kebab_strips_punctuation_and_caps():
    assert _kebab("Hello, World!!!") == "hello-world"
    assert _kebab("  Foo / Bar / Baz  ") == "foo-bar-baz"
    # Cap at 30 chars; trailing dash trimmed if cut lands on one.
    s = _kebab("a" * 50)
    assert len(s) <= 30
    assert _kebab("") == ""
    assert _kebab("!!!@@@") == ""


def test_read_charter_title_frontmatter(workspace):
    assert _read_charter_title(workspace) == "Coffee Habit Tracker"


def test_read_charter_title_h1_fallback(tmp_path):
    ws = tmp_path / "mission_99"
    ws.mkdir()
    (ws / "charter.md").write_text("# Just A Heading\n\nbody", encoding="utf-8")
    assert _read_charter_title(str(ws)) == "Just A Heading"


def test_read_charter_title_missing(tmp_path):
    ws = tmp_path / "mission_99"
    ws.mkdir()
    assert _read_charter_title(str(ws)) == ""


def test_compute_repo_name_uses_charter_slug(workspace):
    name = _compute_repo_name(42, workspace)
    assert name == "kutai-mission-42-coffee-habit-tracker"


def test_compute_repo_name_falls_back_to_unnamed(tmp_path):
    ws = tmp_path / "mission_99"
    ws.mkdir()
    name = _compute_repo_name(99, str(ws))
    assert name == "kutai-mission-99-unnamed"


# ── _populate_export whitelist ──────────────────────────────────────────────


def test_populate_export_copies_whitelist(workspace, tmp_path):
    export = str(tmp_path / "out")
    written = _populate_export(workspace, export)
    assert "charter.md" in written
    assert "non_goals.md" in written
    assert "premortem.md" in written
    assert "adr/" in written
    # Files actually exist in the destination.
    assert os.path.isfile(os.path.join(export, "charter.md"))
    assert os.path.isfile(os.path.join(export, "adr", "001-stack.md"))


def test_populate_export_skips_unwhitelisted(workspace, tmp_path):
    # Drop a hidden file + a non-whitelisted file in the workspace.
    open(os.path.join(workspace, "secrets.env"), "w").write("KEY=1")
    os.makedirs(os.path.join(workspace, ".prototype"), exist_ok=True)
    open(os.path.join(workspace, ".prototype", "x.html"), "w").write("nope")

    export = str(tmp_path / "out")
    _populate_export(workspace, export)
    assert not os.path.exists(os.path.join(export, "secrets.env"))
    assert not os.path.exists(os.path.join(export, ".prototype"))


# ── Fail-soft: gh missing ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_soft_when_gh_missing(workspace):
    def fake_run(cmd, cwd=None, timeout=30.0):
        # gh --version → not found.
        if cmd[:2] == ["gh", "--version"]:
            return 127, "", "FileNotFoundError: gh"
        return 0, "", ""

    with patch("mr_roboto.init_mission_github_repo._run", side_effect=fake_run):
        res = await init_mission_github_repo(
            mission_id=42, workspace_path=workspace
        )
    assert res["ok"] is True
    assert res["pending"] is True
    assert res["reason"] == "gh_cli_missing"
    status = open(os.path.join(workspace, "github_init_status.md"), encoding="utf-8").read()
    assert "pending: gh_cli_missing" in status
    # Did NOT create .git_export
    assert not os.path.exists(os.path.join(workspace, ".git_export"))


# ── Fail-soft: gh unauth ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_soft_when_gh_unauthenticated(workspace):
    def fake_run(cmd, cwd=None, timeout=30.0):
        if cmd[:2] == ["gh", "--version"]:
            return 0, "gh version 2.0", ""
        if cmd[:3] == ["gh", "auth", "status"]:
            return 1, "", "You are not logged into GitHub"
        return 0, "", ""

    with patch("mr_roboto.init_mission_github_repo._run", side_effect=fake_run):
        res = await init_mission_github_repo(
            mission_id=42, workspace_path=workspace
        )
    assert res["ok"] is True
    assert res["pending"] is True
    assert res["reason"] == "gh_unauthenticated"
    status = open(os.path.join(workspace, "github_init_status.md"), encoding="utf-8").read()
    assert "pending: gh_unauthenticated" in status


# ── Founder opt-out ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skip_by_founder_emits_status_without_calling_gh(workspace):
    with patch("mr_roboto.init_mission_github_repo._run") as mock_run:
        res = await init_mission_github_repo(
            mission_id=42, workspace_path=workspace, skip=True
        )
    assert res["ok"] is True
    assert res["pending"] is True
    assert res["reason"] == "skipped_by_founder"
    mock_run.assert_not_called()


# ── Happy path + idempotency ────────────────────────────────────────────────


def _happy_run_factory(repo_exists: bool):
    """Build a fake _run that emulates a successful gh+git pipeline."""
    state = {"created": False}

    def fake_run(cmd, cwd=None, timeout=30.0):
        head = cmd[:3]
        # gh --version OK.
        if cmd[:2] == ["gh", "--version"]:
            return 0, "gh version 2.0", ""
        # gh auth status OK.
        if head == ["gh", "auth", "status"]:
            return 0, "Logged in to github.com as kutaifounder", ""
        # gh api user --jq .login → owner.
        if cmd[:3] == ["gh", "api", "user"]:
            return 0, "kutaifounder", ""
        # gh repo view → exists or not.
        if head == ["gh", "repo", "view"]:
            return (0 if (repo_exists or state["created"]) else 1), "{}", ""
        # gh repo create OK.
        if head == ["gh", "repo", "create"]:
            state["created"] = True
            return 0, "https://github.com/kutaifounder/kutai-mission-42", ""
        # git everything OK.
        if cmd[0] == "git":
            if cmd[1:3] == ["rev-parse", "HEAD"]:
                return 0, "abcdef0123456789", ""
            if cmd[1:3] == ["rev-parse", "--abbrev-ref"]:
                return 0, "main", ""
            if cmd[1:2] == ["remote"] and cmd[2:3] == ["remove"]:
                return 1, "", "no such remote"  # tolerated
            return 0, "", ""
        return 0, "", ""

    return fake_run


@pytest.mark.asyncio
async def test_happy_path_creates_repo_and_returns_url(workspace, monkeypatch):
    # Avoid touching the real DB.
    async def noop(*_a, **_k):
        return None
    monkeypatch.setattr(
        "mr_roboto.init_mission_github_repo._persist_repo_url", noop
    )

    with patch(
        "mr_roboto.init_mission_github_repo._run",
        side_effect=_happy_run_factory(repo_exists=False),
    ):
        res = await init_mission_github_repo(
            mission_id=42, workspace_path=workspace
        )
    assert res["ok"] is True
    assert res["pending"] is False
    assert res["repo_url"] == (
        "https://github.com/kutaifounder/"
        "kutai-mission-42-coffee-habit-tracker"
    )
    assert res["commit_sha"] == "abcdef012345"  # truncated to 12
    assert res["owner"] == "kutaifounder"
    assert res["repo_name"] == "kutai-mission-42-coffee-habit-tracker"
    # Status file written without `pending:`.
    status = open(os.path.join(workspace, "github_init_status.md"), encoding="utf-8").read()
    assert "status: complete" in status
    assert "kutai-mission-42-coffee-habit-tracker" in status
    # .git_export populated with whitelisted files.
    assert os.path.isfile(os.path.join(workspace, ".git_export", "charter.md"))
    assert os.path.isfile(
        os.path.join(workspace, ".git_export", "adr", "001-stack.md")
    )


@pytest.mark.asyncio
async def test_idempotent_when_repo_already_exists(workspace, monkeypatch):
    async def noop(*_a, **_k):
        return None
    monkeypatch.setattr(
        "mr_roboto.init_mission_github_repo._persist_repo_url", noop
    )

    create_calls = {"n": 0}
    happy = _happy_run_factory(repo_exists=True)

    def wrap(cmd, cwd=None, timeout=30.0):
        if cmd[:3] == ["gh", "repo", "create"]:
            create_calls["n"] += 1
        return happy(cmd, cwd=cwd, timeout=timeout)

    with patch("mr_roboto.init_mission_github_repo._run", side_effect=wrap):
        res = await init_mission_github_repo(
            mission_id=42, workspace_path=workspace
        )
    assert res["pending"] is False
    # Repo already existed → never tried to create.
    assert create_calls["n"] == 0


@pytest.mark.asyncio
async def test_visibility_env_override(workspace, monkeypatch):
    monkeypatch.setenv("KUTAI_GITHUB_DEFAULT_VISIBILITY", "public")

    async def noop(*_a, **_k):
        return None
    monkeypatch.setattr(
        "mr_roboto.init_mission_github_repo._persist_repo_url", noop
    )

    seen = {"flag": None}
    happy = _happy_run_factory(repo_exists=False)

    def wrap(cmd, cwd=None, timeout=30.0):
        if cmd[:3] == ["gh", "repo", "create"] and len(cmd) > 4:
            seen["flag"] = cmd[4]
        return happy(cmd, cwd=cwd, timeout=timeout)

    with patch("mr_roboto.init_mission_github_repo._run", side_effect=wrap):
        res = await init_mission_github_repo(
            mission_id=42, workspace_path=workspace
        )
    assert res["pending"] is False
    assert res["visibility"] == "public"
    assert seen["flag"] == "--public"


@pytest.mark.asyncio
async def test_org_env_override(workspace, monkeypatch):
    monkeypatch.setenv("KUTAI_GITHUB_ORG", "kutai-org")

    async def noop(*_a, **_k):
        return None
    monkeypatch.setattr(
        "mr_roboto.init_mission_github_repo._persist_repo_url", noop
    )

    happy = _happy_run_factory(repo_exists=False)
    with patch("mr_roboto.init_mission_github_repo._run", side_effect=happy):
        res = await init_mission_github_repo(
            mission_id=42, workspace_path=workspace
        )
    assert res["owner"] == "kutai-org"
    assert "kutai-org/" in res["repo_url"]
