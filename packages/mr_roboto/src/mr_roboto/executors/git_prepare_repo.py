"""Mechanical: create a GitHub repo + push the mission scaffold (git-prereq for deploy)."""
from __future__ import annotations
import asyncio, os
from typing import Any


async def _get_github_token(service: str = "github") -> str | None:
    from src.security.credential_store import get_credential
    cred = await get_credential(service)
    if not cred:
        return None
    return cred.get("token") or cred.get("api_key")


async def _gh_login(token: str) -> str | None:
    """Resolve the authenticated user's login (owner) for building an owner/name clone URL."""
    import httpx
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get("https://api.github.com/user",
                        headers={"Authorization": f"Bearer {token}",
                                 "Accept": "application/vnd.github+json"})
        return r.json().get("login") if r.status_code == 200 else None


async def _create_repo(name: str, token: str) -> dict[str, Any]:
    """Create repo via the GitHub adapter; treat 'already exists' as success (idempotent)."""
    from src.integrations.registry import get_integration_registry
    gh = get_integration_registry().get("github")
    res = await gh.execute("create_repo", {"name": name})
    if res.get("status") == "ok":
        data = res.get("data", {})
        return {"ok": True, "repo_url": data.get("clone_url") or data.get("html_url"), "existed": False}
    # 422 = name already exists → idempotent: resolve the REAL owner/name clone URL
    # (do NOT return a bare-name URL — https://github.com/{name}.git is unpushable). Opus review.
    if res.get("status_code") == 422:
        login = await _gh_login(token)
        if not login:
            return {"ok": False, "reason": "repo_exists_but_owner_unresolved"}
        return {"ok": True, "repo_url": f"https://github.com/{login}/{name}.git", "existed": True}
    return {"ok": False, "reason": res.get("error", "create_repo failed")}


async def _git_push_scaffold(workspace: str, repo_url: str, token: str) -> dict[str, Any]:
    """Init + commit + push the workspace tree using a PAT-authenticated remote."""
    authed = repo_url.replace("https://", f"https://x-access-token:{token}@")
    cmds = [
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=kutai@local", "-c", "user.name=KutAI",
         "commit", "-q", "-m", "chore: initial scaffold"],
        ["git", "branch", "-M", "main"],
        ["git", "remote", "add", "origin", authed],
        ["git", "push", "-q", "-u", "origin", "main"],
    ]
    for cmd in cmds:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=workspace,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        if proc.returncode != 0 and b"nothing to commit" not in (err or b""):
            return {"ok": False, "reason": f"{cmd[1] if len(cmd)>1 else cmd[0]}: {err.decode(errors='replace')[:200]}"}
    return {"ok": True}


async def run(task: dict) -> dict[str, Any]:
    payload = (task.get("payload") or (task.get("context") or {}).get("payload") or {})
    repo_name = payload.get("repo_name")
    workspace = payload.get("workspace")
    if not repo_name or not workspace:
        return {"ok": False, "reason": "missing repo_name or workspace"}
    if not os.path.isdir(workspace):
        return {"ok": False, "reason": f"workspace not found: {workspace}"}
    token = await _get_github_token()
    if not token:
        return {"ok": False, "reason": "no github credential (add a PAT via /credential add github)"}
    created = await _create_repo(repo_name, token)
    if not created.get("ok"):
        return created
    pushed = await _git_push_scaffold(workspace, created["repo_url"], token)
    if not pushed.get("ok"):
        return {"ok": False, "reason": pushed.get("reason"), "repo_url": created["repo_url"]}
    return {"ok": True, "repo_url": created["repo_url"], "pushed": True, "existed": created.get("existed", False)}
