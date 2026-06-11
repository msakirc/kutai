"""Z1 Tier 6 (C18) — GitHub repo init at end of phase 6.

Creates a remote GitHub repo for a mission via ``gh`` CLI and pushes the
phase-≤6 spec artifacts as the initial commit. Fail-soft when ``gh`` is
missing or unauthenticated — emits a ``pending:`` status file so the mission
still progresses.

Strategic locks honoured:

- Q4 (mechanical-only): no LLM calls.
- Q2 (skip MCP): uses ``gh`` CLI directly (KutAI already has it on PATH per
  paraflow P9 note "KutAI can push via gh-cli already").

Design notes (decisions surfaced in the deliverable summary):

(a) Repo name slug
    ``kutai-mission-<id>-<charter_slug>``. ``charter_slug`` is derived from
    the charter title (frontmatter ``title:`` or first H1). Converted to
    kebab-case (lowercase ASCII alphanumerics + ``-``), stripped of leading/
    trailing dashes, capped at 30 chars. If the charter cannot be read or
    yields an empty slug we fall back to ``unnamed``. Final repo name is
    capped at 80 chars (well under GitHub's 100-char limit).

(b) ``.git_export/`` strategy
    We initialise a SEPARATE git repo at ``mission_<id>/.git_export/`` rather
    than reusing any mission-internal git history. Reasoning: phase-7+ steps
    create their own working repo via ``7.1 repository_setup``; clobbering
    that would be destructive. The export repo is purely a snapshot of the
    locked phase-≤6 spec — auditable, immutable, easy to clone for review.

(c) Artifact whitelist
    Walks ``mission_<id>/`` and copies anything matching this whitelist:
    ``charter.md``, ``product_charter.md``, ``reverse_pitch.md``,
    ``non_goals.md``, ``premortem.md``, ``compliance_overlay.md``,
    ``prior_art_report.md``, ``visual_brief.md``, ``intake_todo.md``,
    plus the directories ``adr/``, ``.style/``, ``.web/`` (HTML prototypes),
    ``screen_plans/``, ``surfaces/``, ``user_flow/``, ``shared_shell/``,
    ``compliance_templates_rendered/``, ``screens/``. README.md and any
    file already at the top level of ``mission_<id>/`` matching ``*.md``
    are also copied. ``.git_export/`` itself is excluded (no recursion).
    Hidden dirs starting with ``.`` other than the whitelisted ones are
    skipped (no .prototype/, no .tunnel.pid, no .git/).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.init_mission_github_repo")


# Per-decision-(c) whitelist. Top-level files copied verbatim if present.
_WHITELIST_FILES = (
    "charter.md",
    "product_charter.md",
    "reverse_pitch.md",
    "non_goals.md",
    "premortem.md",
    "compliance_overlay.md",
    "prior_art_report.md",
    "visual_brief.md",
    "intake_todo.md",
    "spec_drift_report.md",
    "design_tokens.json",
    "taste_emphasis.json",
    "screen_inventory.md",
    "shared_shell.md",
    "user_flow.md",
    "surfaces.md",
    "README.md",
)
# Directories copied recursively if present.
_WHITELIST_DIRS = (
    "adr",
    ".style",
    ".web",
    "screen_plans",
    "screens",
    "surfaces",
    "user_flow",
    "shared_shell",
    "compliance_templates_rendered",
)

_REPO_NAME_MAX = 80
_SLUG_MAX = 30


# ---------------------------------------------------------------------------
# Slug + name derivation
# ---------------------------------------------------------------------------
def _kebab(text: str, max_len: int = _SLUG_MAX) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s[:max_len].rstrip("-") or ""


def _read_charter_title(workspace_path: str) -> str:
    """Read the charter title — frontmatter ``title:`` first, then first H1."""
    candidates = [
        os.path.join(workspace_path, "charter.md"),
        os.path.join(workspace_path, "product_charter.md"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue
        # YAML frontmatter `title:`
        fm = re.match(r"\s*---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if fm:
            tm = re.search(r"^title:\s*(.+?)\s*$", fm.group(1), re.MULTILINE)
            if tm:
                return tm.group(1).strip().strip('"').strip("'")
        # First H1
        h1 = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        if h1:
            return h1.group(1).strip()
    return ""


def _compute_repo_name(mission_id: int, workspace_path: str) -> str:
    title = _read_charter_title(workspace_path)
    slug = _kebab(title) or "unnamed"
    name = f"kutai-mission-{int(mission_id)}-{slug}"
    return name[:_REPO_NAME_MAX]


# ---------------------------------------------------------------------------
# gh CLI helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str], cwd: str | None = None, timeout: float = 30.0) -> tuple[int, str, str]:
    """Run a command. Returns (returncode, stdout, stderr).

    Windows-aware: relies on the executable being on PATH; never shells out
    via cmd.exe.
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except FileNotFoundError as e:
        return 127, "", f"FileNotFoundError: {e}"
    except subprocess.TimeoutExpired as e:
        return 124, "", f"TimeoutExpired: {e}"
    except Exception as e:  # pragma: no cover — diagnostic only
        return 1, "", f"{type(e).__name__}: {e}"


def _gh_available() -> tuple[bool, str]:
    code, out, err = _run(["gh", "--version"], timeout=10.0)
    if code != 0:
        return False, err or out or "gh --version returned non-zero"
    return True, out


def _gh_authenticated() -> tuple[bool, str]:
    # `gh auth status` writes to stderr by convention.
    code, out, err = _run(["gh", "auth", "status"], timeout=10.0)
    blob = (out + "\n" + err).strip()
    if code != 0:
        return False, blob or "gh auth status returned non-zero"
    return True, blob


def _gh_current_user() -> str | None:
    code, out, _err = _run(["gh", "api", "user", "--jq", ".login"], timeout=10.0)
    if code == 0 and out:
        return out.strip()
    return None


def _gh_repo_exists(owner: str, repo: str) -> bool:
    code, _out, _err = _run(
        ["gh", "repo", "view", f"{owner}/{repo}", "--json", "name"], timeout=15.0
    )
    return code == 0


def _gh_repo_create(owner: str, repo: str, visibility: str) -> tuple[bool, str]:
    flag = "--private" if visibility != "public" else "--public"
    code, out, err = _run(
        ["gh", "repo", "create", f"{owner}/{repo}", flag], timeout=30.0
    )
    if code != 0:
        return False, err or out
    return True, out or err


# ---------------------------------------------------------------------------
# Filesystem export
# ---------------------------------------------------------------------------
def _populate_export(workspace_path: str, export_dir: str) -> list[str]:
    """Copy whitelisted artifacts from ``workspace_path`` into ``export_dir``.

    Returns the list of relative paths written (for the status file).
    """
    written: list[str] = []
    os.makedirs(export_dir, exist_ok=True)

    for fname in _WHITELIST_FILES:
        src = os.path.join(workspace_path, fname)
        if os.path.isfile(src):
            dst = os.path.join(export_dir, fname)
            try:
                shutil.copy2(src, dst)
                written.append(fname)
            except Exception as e:  # pragma: no cover
                logger.debug(f"copy {src} -> {dst} failed: {e}")

    for dname in _WHITELIST_DIRS:
        src = os.path.join(workspace_path, dname)
        if os.path.isdir(src):
            dst = os.path.join(export_dir, dname)
            try:
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                # Track the directory itself; consumers can list its contents.
                written.append(f"{dname}/")
            except Exception as e:  # pragma: no cover
                logger.debug(f"copytree {src} -> {dst} failed: {e}")

    return sorted(written)


# ---------------------------------------------------------------------------
# Local git for the export
# ---------------------------------------------------------------------------
def _git_init_export(export_dir: str, commit_message: str) -> tuple[bool, str | None, str]:
    """Init export_dir as a fresh git repo, stage everything, commit.

    Returns (ok, sha, error_blob).
    """
    if not os.path.isdir(os.path.join(export_dir, ".git")):
        code, _, err = _run(["git", "init", "-b", "main"], cwd=export_dir)
        if code != 0:
            # Older gits don't support -b; fall back.
            code, _, err = _run(["git", "init"], cwd=export_dir)
            if code != 0:
                return False, None, err or "git init failed"
            _run(["git", "checkout", "-b", "main"], cwd=export_dir)
    # Identity (idempotent).
    _run(["git", "config", "user.email", "kutai@local"], cwd=export_dir)
    _run(["git", "config", "user.name", "KutAI"], cwd=export_dir)
    code, _, err = _run(["git", "add", "-A"], cwd=export_dir)
    if code != 0:
        return False, None, err or "git add failed"
    # Commit (allow-empty — even an empty mission still gets the marker).
    code, _, err = _run(
        ["git", "commit", "-m", commit_message, "--allow-empty"], cwd=export_dir
    )
    if code != 0:
        return False, None, err or "git commit failed"
    code, sha, err = _run(["git", "rev-parse", "HEAD"], cwd=export_dir)
    if code != 0 or not sha:
        return True, None, err or ""
    return True, sha[:12], ""


def _git_push_to_remote(export_dir: str, owner: str, repo: str) -> tuple[bool, str]:
    remote_url = f"https://github.com/{owner}/{repo}.git"
    # Set or replace remote.
    code, _, _ = _run(["git", "remote", "remove", "origin"], cwd=export_dir)
    code, _, err = _run(
        ["git", "remote", "add", "origin", remote_url], cwd=export_dir
    )
    if code != 0:
        return False, err or "git remote add failed"
    # Detect current branch (we set 'main' in init but be defensive).
    _, branch, _ = _run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=export_dir
    )
    branch = branch.strip() or "main"
    # `gh` provides authenticated git push when configured as a credential
    # helper; fall back to plain `git push -u origin <branch>`.
    code, out, err = _run(
        ["git", "push", "-u", "origin", branch], cwd=export_dir, timeout=60.0
    )
    if code != 0:
        return False, err or out or "git push failed"
    return True, branch


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------
def _write_status(
    workspace_path: str,
    *,
    pending: bool,
    reason: str = "",
    repo_url: str = "",
    sha: str = "",
    files: list[str] | None = None,
) -> str:
    path = os.path.join(workspace_path, "github_init_status.md")
    lines = ["# GitHub init status", ""]
    if pending:
        lines.append(f"pending: {reason}")
    else:
        lines.append("status: complete")
        if repo_url:
            lines.append(f"repo_url: {repo_url}")
        if sha:
            lines.append(f"commit_sha: {sha}")
    if files:
        lines.append("")
        lines.append("## Files")
        for f in files:
            lines.append(f"- {f}")
    body = "\n".join(lines) + "\n"
    os.makedirs(workspace_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------
async def _persist_repo_url(mission_id: int, repo_url: str) -> None:
    try:
        from general_beckman import update_mission_fields as _umf
        await _umf(int(mission_id), github_repo_url=repo_url)
    except Exception as e:  # pragma: no cover — best-effort
        logger.debug(f"github_repo_url DB persist skipped: {e}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def _resolve_workspace(mission_id: int, workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    from src.tools.workspace import get_mission_workspace
    return get_mission_workspace(int(mission_id))


async def init_mission_github_repo(
    mission_id: int,
    repo_visibility: str = "public",
    workspace_path: str | None = None,
    skip: bool = False,
) -> dict[str, Any]:
    """Initialise a GitHub repo for ``mission_<id>`` at end of phase 6.

    Returns a dict with at least: ``ok``, ``pending``, ``reason``,
    ``repo_url``, ``commit_sha``, ``status_path``, ``files``.
    """
    workspace_path = _resolve_workspace(mission_id, workspace_path)
    os.makedirs(workspace_path, exist_ok=True)

    # Founder-level opt-out: still emit the status file, mark skipped.
    if skip:
        status_path = _write_status(
            workspace_path, pending=True, reason="skipped_by_founder"
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "skipped_by_founder",
            "repo_url": None,
            "commit_sha": None,
            "status_path": status_path,
            "files": [],
        }

    # 1. gh available?
    avail, avail_msg = _gh_available()
    if not avail:
        status_path = _write_status(
            workspace_path, pending=True, reason="gh_cli_missing"
        )
        logger.warning(
            f"github init pending for mission {mission_id}: gh missing "
            f"({avail_msg})"
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "gh_cli_missing",
            "repo_url": None,
            "commit_sha": None,
            "status_path": status_path,
            "files": [],
        }

    # 2. gh authenticated?
    authed, auth_msg = _gh_authenticated()
    if not authed:
        status_path = _write_status(
            workspace_path, pending=True, reason="gh_unauthenticated"
        )
        logger.warning(
            f"github init pending for mission {mission_id}: gh not auth "
            f"({auth_msg})"
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "gh_unauthenticated",
            "repo_url": None,
            "commit_sha": None,
            "status_path": status_path,
            "files": [],
        }

    # 3. Configurable visibility + owner.
    visibility = (
        os.environ.get("KUTAI_GITHUB_DEFAULT_VISIBILITY")
        or repo_visibility
        or "public"
    ).strip().lower()
    if visibility not in ("public", "private"):
        visibility = "public"
    owner = os.environ.get("KUTAI_GITHUB_ORG", "").strip() or _gh_current_user()
    if not owner:
        status_path = _write_status(
            workspace_path, pending=True, reason="gh_user_unknown"
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "gh_user_unknown",
            "repo_url": None,
            "commit_sha": None,
            "status_path": status_path,
            "files": [],
        }

    # 4. Compute repo name.
    repo_name = _compute_repo_name(mission_id, workspace_path)

    # 5. Idempotent create.
    if not _gh_repo_exists(owner, repo_name):
        ok, msg = _gh_repo_create(owner, repo_name, visibility)
        if not ok:
            status_path = _write_status(
                workspace_path,
                pending=True,
                reason=f"gh_repo_create_failed: {msg[:200]}",
            )
            logger.warning(
                f"github init pending for mission {mission_id}: create failed "
                f"({msg})"
            )
            return {
                "ok": True,
                "pending": True,
                "reason": "gh_repo_create_failed",
                "repo_url": None,
                "commit_sha": None,
                "status_path": status_path,
                "files": [],
            }

    # 6. Populate .git_export/.
    export_dir = os.path.join(workspace_path, ".git_export")
    files = _populate_export(workspace_path, export_dir)

    # 7. Initial commit.
    commit_msg = (
        f"Initial spec lock — phase 6 close (mission #{int(mission_id)})"
    )
    ok, sha, err = _git_init_export(export_dir, commit_msg)
    if not ok:
        status_path = _write_status(
            workspace_path,
            pending=True,
            reason=f"git_init_failed: {err[:200]}",
            files=files,
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "git_init_failed",
            "repo_url": None,
            "commit_sha": None,
            "status_path": status_path,
            "files": files,
        }

    # 8. Push.
    ok, branch_or_err = _git_push_to_remote(export_dir, owner, repo_name)
    if not ok:
        status_path = _write_status(
            workspace_path,
            pending=True,
            reason=f"git_push_failed: {branch_or_err[:200]}",
            files=files,
        )
        logger.warning(
            f"github init pending for mission {mission_id}: push failed "
            f"({branch_or_err})"
        )
        return {
            "ok": True,
            "pending": True,
            "reason": "git_push_failed",
            "repo_url": None,
            "commit_sha": sha,
            "status_path": status_path,
            "files": files,
        }

    repo_url = f"https://github.com/{owner}/{repo_name}"
    status_path = _write_status(
        workspace_path,
        pending=False,
        repo_url=repo_url,
        sha=sha or "",
        files=files,
    )
    await _persist_repo_url(int(mission_id), repo_url)

    logger.info(
        f"github repo initialised for mission {mission_id}: {repo_url} "
        f"(sha={sha}, files={len(files)})"
    )
    return {
        "ok": True,
        "pending": False,
        "reason": "",
        "repo_url": repo_url,
        "commit_sha": sha,
        "status_path": status_path,
        "files": files,
        "repo_name": repo_name,
        "owner": owner,
        "visibility": visibility,
    }


# ---------------------------------------------------------------------------
# Manual visibility flip (Telegram /github visibility)
# ---------------------------------------------------------------------------
async def set_repo_visibility(
    mission_id: int, visibility: str, workspace_path: str | None = None
) -> dict[str, Any]:
    """Flip visibility on the existing per-mission repo via ``gh repo edit``."""
    visibility = (visibility or "").strip().lower()
    if visibility not in ("public", "private"):
        return {"ok": False, "error": "visibility must be 'public' or 'private'"}
    workspace_path = _resolve_workspace(mission_id, workspace_path)

    # Need the repo URL — pull from DB or recompute.
    repo_url: str | None = None
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT github_repo_url FROM missions WHERE id = ?",
            (int(mission_id),),
        )
        row = await cur.fetchone()
        if row and row[0]:
            repo_url = row[0]
    except Exception as e:  # pragma: no cover — best-effort
        logger.debug(f"github_repo_url lookup skipped: {e}")

    if not repo_url:
        owner = (
            os.environ.get("KUTAI_GITHUB_ORG", "").strip() or _gh_current_user()
        )
        if not owner:
            return {"ok": False, "error": "gh_user_unknown"}
        repo_name = _compute_repo_name(mission_id, workspace_path)
        repo_url = f"https://github.com/{owner}/{repo_name}"

    # Strip protocol so gh can address it as owner/repo.
    target = repo_url.replace("https://github.com/", "").rstrip("/")
    code, out, err = _run(
        ["gh", "repo", "edit", target, "--visibility", visibility,
         "--accept-visibility-change-consequences"],
        timeout=30.0,
    )
    if code != 0:
        return {"ok": False, "error": err or out or "gh repo edit failed"}
    return {"ok": True, "repo_url": repo_url, "visibility": visibility}
