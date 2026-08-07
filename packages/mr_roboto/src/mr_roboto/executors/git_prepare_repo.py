"""Mechanical: create a GitHub repo + push the mission scaffold (git-prereq for deploy)."""
from __future__ import annotations
import asyncio, os
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.git_prepare_repo")


async def _get_github_token(service: str = "github") -> str | None:
    from src.security.credential_store import get_credential
    cred = await get_credential(service)
    if not cred:
        return None
    return cred.get("token") or cred.get("api_key")


async def _gh_login(token: str) -> str | None:
    """Resolve the authenticated user's login (owner) for building an owner/name clone URL.

    A network/timeout error degrades cleanly to None (→ "owner unresolved" reason)
    rather than raising a raw HTTPError out of the executor.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get("https://api.github.com/user",
                            headers={"Authorization": f"Bearer {token}",
                                     "Accept": "application/vnd.github+json"})
            return r.json().get("login") if r.status_code == 200 else None
    except Exception as e:  # timeout / transport / json decode
        logger.warning(f"_gh_login failed to resolve owner: {type(e).__name__}: {e}")
        return None


async def _create_repo(name: str, token: str) -> dict[str, Any]:
    """Create the repo by calling GitHub DIRECTLY with the PAT.

    We deliberately bypass the integration registry: in mock mode (the live-bot
    default, KUTAI_ENV unset) the registry adapter returns a FAKE clone_url —
    which would then be handed to a REAL credentialed ``git push`` (pushing the
    founder's PAT at a mock URL and reporting phantom success). Calling the API
    directly, mirroring ``_gh_login``, removes the mock-in-the-loop hazard.

    Handles: 201 created (read clone_url); 422 already-exists (resolve owner via
    _gh_login → owner/name clone URL); anything else → {"ok": False, "reason"}.
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(
                "https://api.github.com/user/repos",
                json={"name": name, "private": True},
                headers={"Authorization": f"Bearer {token}",
                         "Accept": "application/vnd.github+json"},
            )
    except Exception as e:
        logger.warning(f"create_repo request failed: {type(e).__name__}: {e}")
        return {"ok": False, "reason": f"create_repo request failed: {type(e).__name__}"}

    if r.status_code == 201:
        clone_url = (r.json() or {}).get("clone_url")
        if not clone_url:
            logger.warning("create_repo 201 but no clone_url in response")
            return {"ok": False, "reason": "create_repo ok but no clone_url"}
        return {"ok": True, "repo_url": clone_url, "existed": False}
    # 422 = name already exists → idempotent: resolve the REAL owner/name clone URL
    # (do NOT return a bare-name URL — https://github.com/{name}.git is unpushable).
    if r.status_code == 422:
        login = await _gh_login(token)
        if not login:
            logger.warning("repo exists (422) but owner could not be resolved")
            return {"ok": False, "reason": "repo_exists_but_owner_unresolved"}
        return {"ok": True, "repo_url": f"https://github.com/{login}/{name}.git", "existed": True}
    logger.warning(f"create_repo failed: HTTP {r.status_code}")
    return {"ok": False, "reason": f"create_repo failed: HTTP {r.status_code}"}


async def _git_push_scaffold(workspace: str, repo_url: str, token: str) -> dict[str, Any]:
    """Init + commit + push the workspace tree using a PAT-authenticated remote.

    Idempotent on re-run: ``git remote remove origin`` (ignore failure) precedes
    ``git remote add origin`` so a pre-existing origin doesn't fail the run.
    The PAT is scrubbed from any error text before it becomes a returned reason.
    """
    authed = repo_url.replace("https://", f"https://x-access-token:{token}@")
    # `git remote remove origin` may legitimately fail (no origin yet) — ignore.
    cmds = [
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=kutai@local", "-c", "user.name=KutAI",
         "commit", "-q", "-m", "chore: initial scaffold"],
        ["git", "branch", "-M", "main"],
        ["git", "remote", "remove", "origin"],
        ["git", "remote", "add", "origin", authed],
        ["git", "push", "-q", "-u", "origin", "main"],
    ]
    # Commands whose non-zero exit is benign (idempotency escape hatches).
    _ignore_fail = {("remote", "remove")}
    for cmd in cmds:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=workspace,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        if proc.returncode == 0:
            continue
        # `git remote remove origin` fails when origin doesn't exist yet — benign.
        if (cmd[1], cmd[2] if len(cmd) > 2 else "") in _ignore_fail:
            continue
        if b"nothing to commit" in (err or b""):
            continue
        # Scrub the PAT out of stderr before it reaches a persisted action row.
        err_text = (err or b"").decode(errors="replace").replace(token, "***")
        label = cmd[1] if len(cmd) > 1 else cmd[0]
        logger.warning(f"git_prepare_repo push failed at '{label}': {err_text[:200]}")
        return {"ok": False, "reason": f"{label}: {err_text[:200]}"}
    return {"ok": True}


async def _persist_repo_url(mission_id: int, repo_url: str) -> None:
    """Persist the pushed repo URL to ``missions.github_repo_url``.

    deploy_staging (7.13) reads this column via ``_repo_from_mission`` to locate the
    repo — without this write the live handoff is broken. Best-effort: a DB failure
    is logged but does NOT fail the git action (the repo is already created+pushed).
    Mirrors ``init_mission_github_repo._persist_repo_url``.
    """
    try:
        from general_beckman import update_mission_fields as _umf
        await _umf(int(mission_id), github_repo_url=repo_url)
    except Exception as e:
        logger.warning(f"github_repo_url DB persist failed for mission {mission_id}: "
                       f"{type(e).__name__}: {e}")


def _resolve_workspace(mission_id, payload_ws) -> str | None:
    """Absolute mission workspace. A payload-relative 'mission_{id}/' would resolve against the
    process CWD (mirrors deploy_staging._resolve_workspace) — re-root under WORKSPACE_DIR via
    get_mission_workspace when the payload value is missing or not absolute."""
    if payload_ws and os.path.isabs(payload_ws):
        return payload_ws
    if mission_id is not None:
        from src.tools.workspace import get_mission_workspace
        return get_mission_workspace(int(mission_id))
    return payload_ws


async def run(task: dict) -> dict[str, Any]:
    payload = (task.get("payload") or (task.get("context") or {}).get("payload") or {})
    ctx = task.get("context") or {}
    mission_id = ctx.get("mission_id") or payload.get("mission_id")
    repo_name = payload.get("repo_name")
    if not repo_name and mission_id is not None:
        repo_name = f"kutay-mission-{mission_id}"
    # A workflow payload passes a workspace-relative 'mission_{mission_id}/' that only gets
    # {mission_id} substituted → it resolves against the process CWD. Re-root it (or supply it
    # from mission_id when absent) exactly like deploy_staging does.
    workspace = _resolve_workspace(mission_id, payload.get("workspace"))
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
    # Persist the repo URL so deploy_staging (7.13) can resolve it from the mission row.
    if mission_id is not None:
        await _persist_repo_url(mission_id, created["repo_url"])
    return {"ok": True, "repo_url": created["repo_url"], "pushed": True, "existed": created.get("existed", False)}
