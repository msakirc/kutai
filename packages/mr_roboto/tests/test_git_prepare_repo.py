"""Git-prereq — git_prepare_repo mechanical executor (create repo + push scaffold).

All git ops and the github adapter are stubbed — no network, no real repo.
"""
import pytest


@pytest.mark.asyncio
async def test_git_prepare_repo_creates_and_pushes(monkeypatch, tmp_path):
    from mr_roboto.executors import git_prepare_repo as gpr
    calls = {"created": False, "pushed": False}

    async def fake_create_repo(name, token):
        calls["created"] = True
        return {"ok": True, "repo_url": f"https://github.com/kutay/{name}.git", "existed": False}

    async def fake_git_push(workspace, repo_url, token):
        calls["pushed"] = True
        return {"ok": True}

    monkeypatch.setattr(gpr, "_create_repo", fake_create_repo)
    monkeypatch.setattr(gpr, "_git_push_scaffold", fake_git_push)
    # Ensure a token is present so run() proceeds to create+push.
    async def yes_token(_service="github"):
        return "ghp_faketoken"
    monkeypatch.setattr(gpr, "_get_github_token", yes_token)

    task = {"payload": {"action": "git_prepare_repo", "repo_name": "habithub",
                        "workspace": str(tmp_path)}, "context": {"mission_id": 90}}
    res = await gpr.run(task)
    assert res["ok"] and res["pushed"] and calls["created"] and calls["pushed"]
    assert res["repo_url"].endswith("habithub.git")


@pytest.mark.asyncio
async def test_git_prepare_repo_requires_token(monkeypatch, tmp_path):
    from mr_roboto.executors import git_prepare_repo as gpr
    async def no_cred(_service="github"):
        return None
    monkeypatch.setattr(gpr, "_get_github_token", no_cred)
    task = {"payload": {"action": "git_prepare_repo", "repo_name": "x", "workspace": str(tmp_path)}}
    res = await gpr.run(task)
    assert res["ok"] is False and "credential" in (res.get("reason") or "").lower()


@pytest.mark.asyncio
async def test_git_prepare_repo_dispatches(monkeypatch, tmp_path):
    import mr_roboto
    from mr_roboto.executors import git_prepare_repo as gpr
    async def ok_run(_t):
        return {"ok": True, "repo_url": "https://github.com/kutay/x.git", "pushed": True}
    monkeypatch.setattr(gpr, "run", ok_run)
    act = await mr_roboto.run({"payload": {"action": "git_prepare_repo", "repo_name": "x", "workspace": str(tmp_path)}})
    assert act.status == "completed"
