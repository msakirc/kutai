"""Unit B — GitHub Pages publisher for static mission prototypes.

Pushes the resolved preview root (``_resolve_preview_root``) onto a
dedicated ``gh-pages`` branch of the per-mission GitHub repo, then
enables GitHub Pages so the prototype is live at
``https://<owner>.github.io/<repo>/``.

Design decisions
----------------
- Fail-soft everywhere: every failure returns ``{"ok": True, "pending": True,
  "reason": ...}`` so callers (and Beckman) never see an exception.  The
  mechanical executor is never allowed to DLQ a publish — a stale prototype
  is better than a stuck mission.
- No LLM calls (mechanical-only).
- Uses the same ``_run`` / ``_gh_*`` / ``_compute_repo_name`` / ``_resolve_workspace``
  helpers as ``init_mission_github_repo`` — imported at module level so
  monkeypatch.setattr targets work correctly.
- Pages-enable is idempotent: a 409 / "already exists" from the API is
  treated as success; other non-zero exits produce a ``pages_api_warning``
  key in the result but do NOT fail the publish.
"""
from __future__ import annotations

import os
import shutil
from typing import Any

from yazbunu import get_logger
from mr_roboto.init_mission_github_repo import (
    _run,
    _gh_available,
    _gh_authenticated,
    _gh_current_user,
    _gh_repo_exists,
    _gh_repo_create,
    _compute_repo_name,
    _resolve_workspace,
)

logger = get_logger("mr_roboto.publish_preview_pages")


async def publish_preview_pages(
    mission_id: int,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Publish the mission prototype to GitHub Pages.

    Returns a dict with at least: ``ok``, ``pending``, ``reason``, ``url``.
    On success: also ``repo_url``, ``owner``, ``repo``.
    """
    # 1. Resolve workspace.
    workspace_path = _resolve_workspace(mission_id, workspace_path)

    # 2. Resolve preview root.
    from mr_roboto.emit_preview_url import _resolve_preview_root
    root = _resolve_preview_root(workspace_path)
    if root is None:
        logger.info(
            f"publish_preview_pages pending (no root) for mission {mission_id}"
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "no_preview_root",
            "url": None,
        }

    # 3. gh available?
    avail, avail_msg = _gh_available()
    if not avail:
        logger.warning(
            f"publish_preview_pages pending for mission {mission_id}: "
            f"gh missing ({avail_msg})"
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "gh_cli_missing",
            "url": None,
        }

    # 4. gh authenticated?
    authed, auth_msg = _gh_authenticated()
    if not authed:
        logger.warning(
            f"publish_preview_pages pending for mission {mission_id}: "
            f"gh not auth ({auth_msg})"
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "gh_unauthenticated",
            "url": None,
        }

    # 5. Resolve owner + repo name.
    owner = os.environ.get("KUTAI_GITHUB_ORG", "").strip() or _gh_current_user()
    if not owner:
        return {
            "ok": True,
            "pending": True,
            "reason": "gh_user_unknown",
            "url": None,
        }
    repo = _compute_repo_name(mission_id, workspace_path)

    # 6. Ensure repo exists (create if absent — we only need a Pages host).
    if not _gh_repo_exists(owner, repo):
        ok, msg = _gh_repo_create(owner, repo, "public")
        if not ok:
            logger.warning(
                f"publish_preview_pages pending for mission {mission_id}: "
                f"create failed ({msg})"
            )
            return {
                "ok": True,
                "pending": True,
                "reason": "gh_repo_create_failed",
                "url": None,
            }

    # 7. Build a fresh .pages_export/ dir — clean slate each publish.
    pages_dir = os.path.join(workspace_path, ".pages_export")
    if os.path.isdir(pages_dir):
        shutil.rmtree(pages_dir)
    # Copy the contents of root INTO pages_dir so index.html lands at the root.
    # Skip ALL dotfiles/dot-dirs (defense-in-depth): a static preview never
    # needs them, and stray internal state (chain ledgers with prompts and
    # absolute paths, .git, editor droppings) must not reach the PUBLIC repo.
    shutil.copytree(root, pages_dir, ignore=shutil.ignore_patterns(".*"))

    # 8. Init gh-pages branch in .pages_export/.
    # Try -b gh-pages flag first; fall back for older gits.
    code, _, err = _run(["git", "init", "-b", "gh-pages"], cwd=pages_dir)
    if code != 0:
        code, _, err = _run(["git", "init"], cwd=pages_dir)
        if code != 0:
            return {
                "ok": True,
                "pending": True,
                "reason": "git_init_failed",
                "url": None,
            }
        _run(["git", "checkout", "-b", "gh-pages"], cwd=pages_dir)

    # Set identity.
    _run(["git", "config", "user.email", "kutai@local"], cwd=pages_dir)
    _run(["git", "config", "user.name", "KutAI"], cwd=pages_dir)

    # Stage + commit.
    code, _, err = _run(["git", "add", "-A"], cwd=pages_dir)
    if code != 0:
        return {
            "ok": True,
            "pending": True,
            "reason": "git_add_failed",
            "url": None,
        }
    commit_msg = f"Preview publish — mission #{int(mission_id)}"
    _run(
        ["git", "commit", "-m", commit_msg, "--allow-empty"],
        cwd=pages_dir,
    )

    # Remote.
    remote_url = f"https://github.com/{owner}/{repo}.git"
    _run(["git", "remote", "remove", "origin"], cwd=pages_dir)  # ignore failure
    _run(["git", "remote", "add", "origin", remote_url], cwd=pages_dir)

    # 9. Force-push to gh-pages (generated artifact — each publish overwrites).
    code, out, err = _run(
        ["git", "push", "-f", "-u", "origin", "gh-pages"],
        cwd=pages_dir,
        timeout=60.0,
    )
    if code != 0:
        logger.warning(
            f"publish_preview_pages pending for mission {mission_id}: "
            f"push failed ({err or out})"
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "git_push_failed",
            "url": None,
        }

    # 10. Enable GitHub Pages (idempotent — 409 = already enabled → treat as success).
    pages_api_warning: str | None = None
    pages_code, pages_out, pages_err = _run(
        [
            "gh", "api",
            "-X", "POST",
            f"repos/{owner}/{repo}/pages",
            "-f", "source[branch]=gh-pages",
            "-f", "source[path]=/",
        ],
        timeout=30.0,
    )
    if pages_code != 0:
        blob = (pages_out + "\n" + pages_err).lower()
        if "409" in blob or "already exists" in blob:
            # Pages already enabled — fine.
            pass
        else:
            pages_api_warning = f"pages_enable non-zero ({pages_code}): {pages_err or pages_out}"
            logger.warning(
                f"publish_preview_pages pages_api_warning for mission {mission_id}: "
                f"{pages_api_warning}"
            )

    # 11. Compute URL.
    url = f"https://{owner}.github.io/{repo}/"

    # 12. Persist to DB (best-effort — never raise).
    await _persist_pages_to_db(mission_id, url)

    result: dict[str, Any] = {
        "ok": True,
        "pending": False,
        "url": url,
        "repo_url": f"https://github.com/{owner}/{repo}",
        "owner": owner,
        "repo": repo,
        "reason": "",
    }
    if pages_api_warning:
        result["pages_api_warning"] = pages_api_warning

    logger.info(
        f"publish_preview_pages complete for mission {mission_id}: {url}"
    )
    return result


async def _persist_pages_to_db(mission_id: int, url: str) -> None:
    """Best-effort DB persist — log + swallow any error."""
    try:
        from dabidabi import get_db
        db = await get_db()
        await db.execute(
            "INSERT INTO preview_log (mission_id, action, url, exit_code) "
            "VALUES (?, ?, ?, ?)",
            (int(mission_id), "pages", url, 0),
        )
        await db.commit()
        from general_beckman import update_mission_fields as _umf
        from dabidabi.times import db_now as _db_now
        await _umf(int(mission_id), preview_url=url, preview_started_at=_db_now())
    except Exception as e:
        logger.debug(f"publish_preview_pages DB persist skipped: {e}")
