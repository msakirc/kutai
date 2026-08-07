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


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    """Minimal httpx.AsyncClient stand-in — records calls, returns a scripted resp."""
    def __init__(self, resp, calls):
        self._resp = resp
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kwargs):
        self._calls.append(("POST", url))
        return self._resp

    async def get(self, url, **kwargs):
        self._calls.append(("GET", url))
        return self._resp


@pytest.mark.asyncio
async def test_create_repo_201_reads_clone_url(monkeypatch):
    """201 → real clone_url is returned (no registry / no mock in the loop)."""
    from mr_roboto.executors import git_prepare_repo as gpr
    import httpx
    calls = []
    resp = _FakeResp(201, {"clone_url": "https://github.com/kutay/habithub.git"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(resp, calls))
    res = await gpr._create_repo("habithub", "ghp_realtoken")
    assert res == {"ok": True, "repo_url": "https://github.com/kutay/habithub.git", "existed": False}
    # Called GitHub DIRECTLY — never the integration registry / mock adapter.
    assert calls == [("POST", "https://api.github.com/user/repos")]


@pytest.mark.asyncio
async def test_create_repo_422_resolves_owner(monkeypatch):
    """422 already-exists → resolve real owner/name clone URL via _gh_login."""
    from mr_roboto.executors import git_prepare_repo as gpr
    import httpx
    resp = _FakeResp(422, {"message": "name already exists"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(resp, []))
    async def fake_login(_tok):
        return "kutay"
    monkeypatch.setattr(gpr, "_gh_login", fake_login)
    res = await gpr._create_repo("habithub", "ghp_realtoken")
    assert res == {"ok": True, "repo_url": "https://github.com/kutay/habithub.git", "existed": True}


@pytest.mark.asyncio
async def test_create_repo_no_mock_can_trigger_push(monkeypatch, tmp_path):
    """Regression: a fake/mock clone_url must NOT drive a real credentialed push.

    The old code went through the registry which, in the live-bot mock default,
    returned a fake clone_url that then hit a real `git push`. _create_repo now
    calls GitHub directly, so a non-2xx/422 response degrades to ok=False and
    run() short-circuits BEFORE _git_push_scaffold ever runs.
    """
    from mr_roboto.executors import git_prepare_repo as gpr
    import httpx
    pushed = {"called": False}

    async def yes_token(_service="github"):
        return "ghp_realtoken"

    async def spy_push(workspace, repo_url, token):  # must never run
        pushed["called"] = True
        return {"ok": True}

    # 403 (e.g. token lacks repo scope) — the mock-response path no longer exists.
    resp = _FakeResp(403, {"message": "forbidden"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(resp, []))
    monkeypatch.setattr(gpr, "_get_github_token", yes_token)
    monkeypatch.setattr(gpr, "_git_push_scaffold", spy_push)

    task = {"payload": {"action": "git_prepare_repo", "repo_name": "habithub",
                        "workspace": str(tmp_path)}}
    res = await gpr.run(task)
    assert res["ok"] is False
    assert pushed["called"] is False  # no real push against a phantom repo


@pytest.mark.asyncio
async def test_push_scrubs_token_from_reason(monkeypatch, tmp_path):
    """A failing git command's stderr must not leak the PAT into the reason."""
    from mr_roboto.executors import git_prepare_repo as gpr

    token = "ghp_SECRETTOKEN"

    class _Proc:
        returncode = 1
        async def communicate(self):
            # stderr embeds the authed remote (token in URL) — as git does.
            return b"", f"fatal: unable to access https://x-access-token:{token}@github.com/x".encode()

    async def fake_exec(*cmd, **kwargs):
        # git remote remove would "fail" benignly first; force the push to fail.
        if cmd[:2] == ("git", "push"):
            return _Proc()
        ok = _Proc()
        ok.returncode = 0
        return ok

    monkeypatch.setattr(gpr.asyncio, "create_subprocess_exec", fake_exec)
    res = await gpr._git_push_scaffold(str(tmp_path), "https://github.com/kutay/x.git", token)
    assert res["ok"] is False
    assert token not in res["reason"]
    assert "***" in res["reason"]
