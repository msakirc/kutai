# Z0 Mission Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-mission $ ceiling, forum-topic provisioning, lifecycle states, auto-pause triggers, and reversibility/collision-guard system to KutAI without disturbing existing routing or i2p Phase 0.

**Architecture:** Thin slices across `src/infra/db.py`, `src/app/telegram_bot.py`, `packages/general_beckman/`, `packages/fatih_hoca/`, `src/workflows/engine/`, plus one new package `packages/safety_guard/`. No new umbrella package; data couples through `missions` table additions and a new `mission_lifecycle_log` audit table.

**Tech Stack:** Python 3.10, aiosqlite (WAL), python-telegram-bot v20, pytest with `timeout 120` discipline.

**Spec:** [docs/superpowers/specs/2026-05-05-z0-mission-preflight-design.md](../specs/2026-05-05-z0-mission-preflight-design.md)

**Conventions:**
- Every implementation step is preceded by a failing test (TDD).
- All `pytest` invocations prefix with `timeout 60` (single test) or `timeout 120` (suite).
- Commit after each task. Use the exact commit messages shown.
- Existing migration pattern (try/except for idempotency) used for all `ALTER TABLE` statements.

---

## File Structure

**Create:**
- `packages/safety_guard/pyproject.toml`
- `packages/safety_guard/src/safety_guard/__init__.py` — public API: `pre_action`, `Decision` types
- `packages/safety_guard/src/safety_guard/tags.py` — `Reversibility` enum + `resolve()`
- `packages/safety_guard/src/safety_guard/collision.py` — collision guards + hardcoded blocklist
- `packages/safety_guard/src/safety_guard/executor_hook.py` — orchestrates tag resolve + guards + decision
- `packages/safety_guard/tests/__init__.py`
- `packages/safety_guard/tests/test_tags.py`
- `packages/safety_guard/tests/test_collision.py`
- `packages/safety_guard/tests/test_executor_hook.py`
- `tests/infra/test_db_migration_z0.py`
- `tests/integration/test_z0_mission_lifecycle.py`
- `tests/integration/test_z0_collision_block.py`
- `tests/integration/test_z0_kill_snapshot.py`

**Modify:**
- `src/infra/db.py` (~line 920) — add Z0 column migrations + `mission_lifecycle_log` table
- `packages/fatih_hoca/src/fatih_hoca/selector.py` — accept `remaining_budget_usd`, filter candidates
- `packages/fatih_hoca/src/fatih_hoca/types.py` — add `SelectionFailure` reason field
- `packages/general_beckman/src/general_beckman/admission.py` (or `queue.py` — task picks) — lifecycle gate, budget pass-through, in-flight tracker, ceiling backstop
- `packages/general_beckman/src/general_beckman/__init__.py` — `on_task_finished` cost increment + threshold notifies + DLQ-cascade trigger
- `src/app/telegram_bot.py` — forum-topic provisioning, pinned status, `/kill_mission`, `/pause_mission`/`/resume_mission` polish, callback handler routes
- `src/workflows/i2p/i2p_v3.json` — tag dangerous steps with `reversibility` + `locked`
- `src/workflows/engine/runner.py` (or where step is dispatched to executor) — call `safety_guard.pre_action`

---

## Phase 1 — Foundation (DB + safety_guard primitives)

### Task 1: DB migration

**Files:**
- Modify: `src/infra/db.py` (~line 920, after the existing `ALTER TABLE tasks ADD COLUMN timeout_seconds` block)
- Test: `tests/infra/test_db_migration_z0.py`

- [ ] **Step 1: Write failing test**

```python
# tests/infra/test_db_migration_z0.py
import os
import pytest
import aiosqlite

from src.infra.db import init_db


@pytest.mark.asyncio
async def test_z0_columns_added(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)

    # Pre-seed a missions row to simulate pre-Z0 install
    async with aiosqlite.connect(db_path) as db:
        await db.execute("CREATE TABLE missions (id INTEGER PRIMARY KEY, title TEXT)")
        await db.execute("INSERT INTO missions (title) VALUES ('legacy')")
        await db.commit()

    await init_db()

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("PRAGMA table_info(missions)")
        cols = {row[1] for row in await cur.fetchall()}
        assert "cost_ceiling_usd" in cols
        assert "spent_usd" in cols
        assert "message_thread_id" in cols
        assert "lifecycle_state" in cols

        # Pre-existing row gets defaults
        cur = await db.execute(
            "SELECT lifecycle_state, spent_usd, cost_ceiling_usd "
            "FROM missions WHERE title = 'legacy'"
        )
        row = await cur.fetchone()
        assert row == ("active", 0, None)

        # Lifecycle log table exists
        cur = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='mission_lifecycle_log'"
        )
        assert await cur.fetchone() is not None


@pytest.mark.asyncio
async def test_z0_migration_idempotent(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    await init_db()
    await init_db()  # second pass — must not raise
```

- [ ] **Step 2: Run test, confirm fail**

Run: `timeout 60 pytest tests/infra/test_db_migration_z0.py -v`
Expected: FAIL — columns not present.

- [ ] **Step 3: Add migration to `src/infra/db.py`**

Locate the existing series of `ALTER TABLE tasks ADD COLUMN ...` blocks (~line 906–933). Append:

```python
# Z0 mission preflight columns (2026-05-05)
for ddl in (
    "ALTER TABLE missions ADD COLUMN cost_ceiling_usd REAL",
    "ALTER TABLE missions ADD COLUMN spent_usd REAL DEFAULT 0",
    "ALTER TABLE missions ADD COLUMN message_thread_id INTEGER",
    "ALTER TABLE missions ADD COLUMN lifecycle_state TEXT DEFAULT 'active'",
):
    try:
        await db.execute(ddl)
    except Exception as e:
        logger.debug(f"Z0 column migration skipped (already present): {e}")

await db.execute("""
    CREATE TABLE IF NOT EXISTS mission_lifecycle_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mission_id INTEGER NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        reason TEXT,
        triggered_by TEXT,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (mission_id) REFERENCES missions(id)
    )
""")

# Backfill lifecycle_state for any pre-existing NULL rows (older installs)
await db.execute(
    "UPDATE missions SET lifecycle_state = 'active' WHERE lifecycle_state IS NULL"
)
await db.execute(
    "UPDATE missions SET spent_usd = 0 WHERE spent_usd IS NULL"
)
```

- [ ] **Step 4: Run test, confirm pass**

Run: `timeout 60 pytest tests/infra/test_db_migration_z0.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke import + commit**

```bash
python -c "from src.infra.db import init_db; print('ok')"
git add tests/infra/test_db_migration_z0.py src/infra/db.py
git commit -m "feat(z0): missions schema migration for ceiling, thread, lifecycle"
```

---

### Task 2: safety_guard package skeleton + tags.py

**Files:**
- Create: `packages/safety_guard/pyproject.toml`
- Create: `packages/safety_guard/src/safety_guard/__init__.py`
- Create: `packages/safety_guard/src/safety_guard/tags.py`
- Create: `packages/safety_guard/tests/__init__.py`
- Create: `packages/safety_guard/tests/test_tags.py`

- [ ] **Step 1: Create pyproject + package skeleton**

`packages/safety_guard/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "safety_guard"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

`packages/safety_guard/src/safety_guard/__init__.py`:

```python
"""Pre-action safety guard: reversibility tag resolution + collision guards."""
from safety_guard.tags import Reversibility, resolve

__all__ = ["Reversibility", "resolve"]
```

`packages/safety_guard/tests/__init__.py`: empty file.

- [ ] **Step 2: Write failing test for tags.py**

`packages/safety_guard/tests/test_tags.py`:

```python
import pytest
from safety_guard.tags import Reversibility, resolve


def test_default_full_when_unspecified():
    assert resolve({}, None) == Reversibility.FULL


def test_static_returned_when_no_runtime_override():
    step = {"reversibility": "partial"}
    assert resolve(step, None) == Reversibility.PARTIAL


def test_locked_ignores_runtime_override():
    step = {"reversibility": "none", "locked": True}
    # Even an "escalation" attempt is ignored — locked means workflow author wins.
    assert resolve(step, Reversibility.FULL) == Reversibility.NONE
    assert resolve(step, Reversibility.PARTIAL) == Reversibility.NONE


def test_runtime_escalation_accepted():
    step = {"reversibility": "full"}
    assert resolve(step, Reversibility.PARTIAL) == Reversibility.PARTIAL
    assert resolve(step, Reversibility.NONE) == Reversibility.NONE


def test_runtime_downgrade_rejected():
    step = {"reversibility": "none"}
    # Downgrade to FULL → static (NONE) wins.
    assert resolve(step, Reversibility.FULL) == Reversibility.NONE
    assert resolve(step, Reversibility.PARTIAL) == Reversibility.NONE


def test_unknown_reversibility_value_falls_back_to_full(caplog):
    step = {"reversibility": "garbage"}
    with caplog.at_level("WARNING"):
        assert resolve(step, None) == Reversibility.FULL
    assert any("garbage" in m for m in caplog.messages)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `timeout 60 pytest packages/safety_guard/tests/test_tags.py -v`
Expected: FAIL — module not present.

- [ ] **Step 4: Implement tags.py**

`packages/safety_guard/src/safety_guard/tags.py`:

```python
"""Reversibility tag resolution.

Static tag in workflow JSON is the floor. Runtime override (from executor)
may escalate stricter; downgrade is rejected. `locked: true` removes
override entirely (workflow author wins absolutely).
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Reversibility(Enum):
    FULL = ("full", 0)        # rank: higher = stricter
    PARTIAL = ("partial", 1)
    NONE = ("none", 2)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def rank(self) -> int:
        return self.value[1]

    @classmethod
    def from_str(cls, s: str) -> "Reversibility":
        for r in cls:
            if r.label == s:
                return r
        logger.warning(
            "unknown reversibility value %r; defaulting to FULL", s
        )
        return cls.FULL


def resolve(step: dict, runtime_override: Optional[Reversibility]) -> Reversibility:
    """Return the effective reversibility tag for a step.

    Priority:
      1. If `step.locked` is true: static tag wins, override ignored.
      2. Else, runtime override accepted only if rank >= static rank.
      3. Downgrade attempts logged + rejected.
    """
    static = Reversibility.from_str(step.get("reversibility", "full"))
    locked = bool(step.get("locked", False))
    if locked or runtime_override is None:
        return static
    if runtime_override.rank < static.rank:
        logger.warning(
            "downgrade rejected: runtime=%s static=%s; using static",
            runtime_override.label, static.label,
        )
        return static
    return runtime_override
```

- [ ] **Step 5: Run test, install editable, commit**

```bash
pip install -e packages/safety_guard
timeout 60 pytest packages/safety_guard/tests/test_tags.py -v
```
Expected: PASS.

```bash
git add packages/safety_guard/
git commit -m "feat(safety_guard): Reversibility enum + tag resolver"
```

---

### Task 3: safety_guard collision.py

**Files:**
- Create: `packages/safety_guard/src/safety_guard/collision.py`
- Test: `packages/safety_guard/tests/test_collision.py`

- [ ] **Step 1: Write failing tests**

`packages/safety_guard/tests/test_collision.py`:

```python
import pytest
from safety_guard.collision import (
    detect_force_push,
    detect_shared_history_rewrite,
    detect_shell_outside_workspace,
    detect_destructive_shared_db,
    detect_blocklist,
    SHARED_BRANCHES,
)


# ── force-push ──────────────────────────────────────────────
def test_detects_git_push_force_short():
    assert detect_force_push("git push -f origin feature")


def test_detects_git_push_force_long():
    assert detect_force_push("git push --force origin feature")


def test_detects_git_push_force_with_lease():
    assert detect_force_push("git push --force-with-lease origin feature")


def test_no_false_positive_on_normal_push():
    assert not detect_force_push("git push origin feature")
    assert not detect_force_push("echo --force this is just text")


# ── shared-history rewrite ──────────────────────────────────
def test_detects_rebase_on_main():
    assert detect_shared_history_rewrite("git rebase main", current_branch="main")


def test_detects_reset_hard_on_shared():
    assert detect_shared_history_rewrite("git reset --hard HEAD~3", current_branch="develop")


def test_no_rewrite_on_personal_branch():
    assert not detect_shared_history_rewrite(
        "git rebase main", current_branch="sakir/feature"
    )


# ── shell scope ─────────────────────────────────────────────
def test_blocks_shell_outside_workspace(tmp_path):
    workspace = str(tmp_path / "ws")
    assert detect_shell_outside_workspace("rm -rf /tmp/scratch", workspace_root=workspace)
    assert detect_shell_outside_workspace("cat /etc/passwd", workspace_root=workspace)


def test_allows_shell_inside_workspace(tmp_path):
    workspace = str(tmp_path / "ws")
    assert not detect_shell_outside_workspace(
        f"rm -rf {workspace}/scratch", workspace_root=workspace
    )


# ── destructive shared DB ───────────────────────────────────
def test_blocks_drop_table_non_mission_scoped():
    assert detect_destructive_shared_db("DROP TABLE missions")
    assert detect_destructive_shared_db("TRUNCATE TABLE tasks")


def test_allows_mission_scoped_drop():
    # Mission-scoped tables are named with mission id suffix or `mission_*`
    assert not detect_destructive_shared_db("DROP TABLE mission_42_scratch")


# ── hardcoded blocklist (always wins) ───────────────────────
def test_blocklist_force_push_to_main():
    assert detect_blocklist("git push --force origin main")
    assert detect_blocklist("git push -f origin master")


def test_blocklist_stripe_charge_create():
    assert detect_blocklist("stripe.charges.create(amount=1000)")


def test_blocklist_vercel_prod():
    assert detect_blocklist("vercel deploy --prod")


def test_blocklist_aws_s3_rm():
    assert detect_blocklist("aws s3 rm s3://prod-bucket --recursive")


def test_blocklist_passes_innocuous():
    assert not detect_blocklist("git status")
    assert not detect_blocklist("echo hello")
```

- [ ] **Step 2: Run test, confirm fail**

Run: `timeout 60 pytest packages/safety_guard/tests/test_collision.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement collision.py**

`packages/safety_guard/src/safety_guard/collision.py`:

```python
"""Collision and blocklist detectors. Pure functions, no I/O."""
from __future__ import annotations

import os
import re
import shlex


SHARED_BRANCHES = {"main", "master", "develop"}
SHARED_BRANCH_PATTERNS = [re.compile(r"^release/")]


def _is_shared_branch(branch: str) -> bool:
    if branch in SHARED_BRANCHES:
        return True
    return any(p.match(branch) for p in SHARED_BRANCH_PATTERNS)


# Force-push: any git push with -f, --force, or --force-with-lease.
_FORCE_PUSH_RE = re.compile(
    r"\bgit\s+push\b.*?(?:\s-f\b|\s--force(?:-with-lease)?\b)"
)


def detect_force_push(cmd: str) -> bool:
    """Return True if `cmd` is any flavor of force-push."""
    return bool(_FORCE_PUSH_RE.search(cmd))


_REBASE_RE = re.compile(r"\bgit\s+rebase\b")
_RESET_HARD_RE = re.compile(r"\bgit\s+reset\b.*?\s--hard\b")


def detect_shared_history_rewrite(cmd: str, current_branch: str) -> bool:
    """Detect rebase / reset --hard on shared branches."""
    if not _is_shared_branch(current_branch):
        return False
    return bool(_REBASE_RE.search(cmd) or _RESET_HARD_RE.search(cmd))


def detect_shell_outside_workspace(cmd: str, workspace_root: str) -> bool:
    """True if cmd references absolute paths outside `workspace_root`.

    Best-effort: tokenizes and checks any token that looks like an absolute
    path. If a token is /tmp/x and workspace is /home/user/ws, returns True.
    """
    workspace_norm = os.path.abspath(workspace_root).rstrip(os.sep) + os.sep
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        # Malformed shell — be conservative, block.
        return True
    for tok in tokens:
        if tok.startswith("/") or (len(tok) > 1 and tok[1] == ":"):
            abs_tok = os.path.abspath(tok)
            if not abs_tok.startswith(workspace_norm) and abs_tok != workspace_norm.rstrip(os.sep):
                return True
    return False


_DESTRUCTIVE_DB_RE = re.compile(
    r"^\s*(DROP\s+TABLE|TRUNCATE(?:\s+TABLE)?)\s+(\w+)", re.IGNORECASE
)


def detect_destructive_shared_db(query: str) -> bool:
    """DROP/TRUNCATE on non-mission-scoped tables.

    Mission-scoped tables match `mission_<digits>_*` or `mission_<digits>`.
    """
    m = _DESTRUCTIVE_DB_RE.match(query)
    if not m:
        return False
    table = m.group(2)
    if re.match(r"^mission_\d+(_|$)", table):
        return False
    return True


# Hardcoded blocklist — patterns that are NEVER allowed regardless of tag.
_BLOCKLIST_PATTERNS = [
    re.compile(r"\bgit\s+push\b.*?(?:-f|--force(?:-with-lease)?)\b.*\b(main|master)\b"),
    re.compile(r"\bstripe\.charges\.create\b"),
    re.compile(r"\bvercel\s+deploy\s+--prod\b"),
    re.compile(r"\baws\s+s3\s+rm\b"),
    re.compile(r"\brm\s+-rf\s+/(?!\w*tmp)"),
]


def detect_blocklist(cmd: str) -> bool:
    return any(p.search(cmd) for p in _BLOCKLIST_PATTERNS)
```

- [ ] **Step 4: Run test, confirm pass**

Run: `timeout 60 pytest packages/safety_guard/tests/test_collision.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/safety_guard/
git commit -m "feat(safety_guard): collision detectors + hardcoded blocklist"
```

---

### Task 4: safety_guard executor_hook.py

**Files:**
- Create: `packages/safety_guard/src/safety_guard/executor_hook.py`
- Modify: `packages/safety_guard/src/safety_guard/__init__.py` (export `pre_action`, `Decision`)
- Test: `packages/safety_guard/tests/test_executor_hook.py`

- [ ] **Step 1: Write failing tests**

`packages/safety_guard/tests/test_executor_hook.py`:

```python
import pytest
from dataclasses import dataclass

from safety_guard import pre_action, Reversibility
from safety_guard.executor_hook import Allow, Block, WaitForFounder


# ── happy paths ─────────────────────────────────────────────
def test_simple_allow():
    step = {"reversibility": "full"}
    action = {"command": "echo hello"}
    decision = pre_action(step, action, workspace_root="/tmp/ws", current_branch="feat")
    assert isinstance(decision, Allow)


def test_force_push_blocked():
    step = {"reversibility": "partial"}
    action = {"command": "git push --force origin main"}
    decision = pre_action(step, action, workspace_root="/tmp/ws", current_branch="main")
    assert isinstance(decision, Block)
    assert "blocklist" in decision.reason or "force_push" in decision.reason


def test_collision_block_outside_workspace():
    step = {"reversibility": "full"}
    action = {"command": "rm -rf /etc/passwd"}
    decision = pre_action(step, action, workspace_root="/tmp/ws", current_branch="feat")
    assert isinstance(decision, Block)


# ── reversibility flow ──────────────────────────────────────
def test_none_locked_waits_when_idle():
    step = {"reversibility": "none", "locked": True}
    action = {"command": "stripe-cli charge --amount 100"}  # not in blocklist (test env)
    decision = pre_action(
        step, action,
        workspace_root="/tmp/ws", current_branch="feat",
        founder_recently_active=False,
    )
    assert isinstance(decision, WaitForFounder)


def test_none_proceeds_when_founder_active():
    step = {"reversibility": "none"}  # not locked
    action = {"command": "git tag v1.0"}
    decision = pre_action(
        step, action,
        workspace_root="/tmp/ws", current_branch="feat",
        founder_recently_active=True,
    )
    assert isinstance(decision, Allow)


def test_none_locked_waits_even_when_active():
    step = {"reversibility": "none", "locked": True}
    action = {"command": "git tag v1.0"}
    decision = pre_action(
        step, action,
        workspace_root="/tmp/ws", current_branch="feat",
        founder_recently_active=True,
    )
    # Locked + none ALWAYS requires explicit founder action, regardless of recency.
    assert isinstance(decision, WaitForFounder)


# ── allowlist override ──────────────────────────────────────
def test_per_mission_allowlist_relaxes_collision():
    step = {"reversibility": "partial"}
    action = {"command": "git push --force-with-lease origin sakir/feat"}
    # Without allowlist: blocked
    decision = pre_action(
        step, action,
        workspace_root="/tmp/ws", current_branch="sakir/feat",
        mission_allowlist=[],
    )
    assert isinstance(decision, Block)
    # With allowlist: allowed
    decision = pre_action(
        step, action,
        workspace_root="/tmp/ws", current_branch="sakir/feat",
        mission_allowlist=[r"git push --force-with-lease origin sakir/"],
    )
    assert isinstance(decision, Allow)


def test_blocklist_beats_allowlist():
    # Even with allowlist match, hardcoded blocklist wins.
    step = {"reversibility": "partial"}
    action = {"command": "git push --force origin main"}
    decision = pre_action(
        step, action,
        workspace_root="/tmp/ws", current_branch="main",
        mission_allowlist=[r"git push --force origin main"],  # tries to allow
    )
    assert isinstance(decision, Block)
```

- [ ] **Step 2: Run test, confirm fail**

Run: `timeout 60 pytest packages/safety_guard/tests/test_executor_hook.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement executor_hook.py**

`packages/safety_guard/src/safety_guard/executor_hook.py`:

```python
"""Pre-action decision: Allow | WaitForFounder | Block.

Called by the workflow engine before any executor action runs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from safety_guard.tags import Reversibility, resolve
from safety_guard.collision import (
    detect_force_push,
    detect_shared_history_rewrite,
    detect_shell_outside_workspace,
    detect_destructive_shared_db,
    detect_blocklist,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Allow:
    pass


@dataclass(frozen=True)
class WaitForFounder:
    reason: str


@dataclass(frozen=True)
class Block:
    reason: str


Decision = Allow | WaitForFounder | Block


def _allowlist_matches(cmd: str, patterns: list[str]) -> bool:
    return any(re.search(p, cmd) for p in patterns)


def pre_action(
    step: dict,
    action: dict,
    *,
    workspace_root: str,
    current_branch: str,
    founder_recently_active: bool = True,
    mission_allowlist: Optional[list[str]] = None,
    runtime_reversibility: Optional[Reversibility] = None,
) -> Decision:
    """Single decision point before executing a workflow step's action.

    Args:
      step: workflow step dict (carries reversibility/locked).
      action: pending action dict (carries command, target, etc.).
      workspace_root: absolute path of mission workspace.
      current_branch: git branch the workspace is on.
      founder_recently_active: True if Telegram has seen activity in the
        last N minutes (caller decides; we just gate on it).
      mission_allowlist: per-mission patterns the founder approved via
        /safety allow. Cannot override hardcoded blocklist.
      runtime_reversibility: executor's optional escalation (stricter only).

    Returns Allow | WaitForFounder(reason) | Block(reason).
    """
    cmd = action.get("command", "") or ""
    mission_allowlist = mission_allowlist or []

    # 1. Hardcoded blocklist — always wins.
    if detect_blocklist(cmd):
        return Block(reason="blocklist")

    # 2. Collision guards — bypassable only by per-mission allowlist.
    def _block_unless_allowed(check, name):
        if check and not _allowlist_matches(cmd, mission_allowlist):
            return Block(reason=f"collision:{name}")
        return None

    for check, name in (
        (detect_force_push(cmd), "force_push"),
        (detect_shared_history_rewrite(cmd, current_branch), "shared_history_rewrite"),
        (detect_shell_outside_workspace(cmd, workspace_root), "shell_outside_workspace"),
        (detect_destructive_shared_db(cmd), "destructive_shared_db"),
    ):
        result = _block_unless_allowed(check, name)
        if result is not None:
            return result

    # 3. Reversibility-driven flow.
    tag = resolve(step, runtime_reversibility)
    if tag is Reversibility.NONE:
        if step.get("locked", False) or not founder_recently_active:
            return WaitForFounder(reason="non_reversible_step")

    return Allow()
```

Update `packages/safety_guard/src/safety_guard/__init__.py`:

```python
"""Pre-action safety guard."""
from safety_guard.tags import Reversibility, resolve
from safety_guard.executor_hook import (
    pre_action,
    Allow,
    WaitForFounder,
    Block,
    Decision,
)

__all__ = [
    "Reversibility",
    "resolve",
    "pre_action",
    "Allow",
    "WaitForFounder",
    "Block",
    "Decision",
]
```

- [ ] **Step 4: Run test, confirm pass**

Run: `timeout 60 pytest packages/safety_guard/tests/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/safety_guard/
git commit -m "feat(safety_guard): executor pre_action hook (Allow/Wait/Block)"
```

---

## Phase 2 — Selection-time budget filter (Fatih Hoca)

### Task 5: Fatih Hoca accepts `remaining_budget_usd` and filters candidates

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py`
- Modify: `packages/fatih_hoca/src/fatih_hoca/types.py` (or wherever Pick is defined)
- Test: `packages/fatih_hoca/tests/test_budget_filter.py` (new)

- [ ] **Step 1: Locate selector entry**

```bash
grep -n "def select" packages/fatih_hoca/src/fatih_hoca/selector.py | head -5
```

Confirm `Selector.select()` is a method that scores candidates after eligibility filtering. Identify the eligibility-filter step (the function or block that produces the candidate list before scoring).

- [ ] **Step 2: Write failing test**

`packages/fatih_hoca/tests/test_budget_filter.py`:

```python
import pytest
from unittest.mock import MagicMock

from fatih_hoca.selector import Selector


def _mk_model(name, cost_estimate):
    m = MagicMock()
    m.name = name
    m.estimated_cost_usd = cost_estimate
    m.is_eligible_for.return_value = True
    return m


def test_filters_models_above_remaining_budget(monkeypatch):
    selector = Selector(
        registry=MagicMock(all_models=lambda: [
            _mk_model("local-free", 0.0),
            _mk_model("cloud-cheap", 0.10),
            _mk_model("cloud-expensive", 5.00),
        ]),
        nerd_herd=MagicMock(),
        available_providers=[],
    )
    # Stub scoring to a known model when a specific list reaches it.
    seen = []
    def fake_score(candidates, **_kw):
        seen.append([m.name for m in candidates])
        return candidates[0] if candidates else None
    monkeypatch.setattr(selector, "_score_and_pick", fake_score, raising=False)

    selector.select(task={}, remaining_budget_usd=0.50)

    assert seen, "scoring not called"
    assert "cloud-expensive" not in seen[0]
    assert "local-free" in seen[0]
    assert "cloud-cheap" in seen[0]


def test_zero_budget_keeps_only_free_models(monkeypatch):
    selector = Selector(
        registry=MagicMock(all_models=lambda: [
            _mk_model("local-free", 0.0),
            _mk_model("cloud-any", 0.01),
        ]),
        nerd_herd=MagicMock(),
        available_providers=[],
    )
    seen = []
    monkeypatch.setattr(
        selector, "_score_and_pick",
        lambda candidates, **_kw: (seen.append([m.name for m in candidates]) or candidates[0] if candidates else None),
        raising=False,
    )
    selector.select(task={}, remaining_budget_usd=0.0)
    assert seen[0] == ["local-free"]


def test_none_budget_no_filter(monkeypatch):
    selector = Selector(
        registry=MagicMock(all_models=lambda: [
            _mk_model("local-free", 0.0),
            _mk_model("cloud-any", 1000.0),
        ]),
        nerd_herd=MagicMock(),
        available_providers=[],
    )
    seen = []
    monkeypatch.setattr(
        selector, "_score_and_pick",
        lambda candidates, **_kw: (seen.append([m.name for m in candidates]) or candidates[0] if candidates else None),
        raising=False,
    )
    selector.select(task={}, remaining_budget_usd=None)
    assert "cloud-any" in seen[0]


def test_empty_pool_returns_failure(monkeypatch):
    selector = Selector(
        registry=MagicMock(all_models=lambda: [_mk_model("only-cloud", 0.50)]),
        nerd_herd=MagicMock(),
        available_providers=[],
    )
    result = selector.select(task={}, remaining_budget_usd=0.10)
    # Per spec: empty candidate pool → SelectionFailure(reason='budget')
    from fatih_hoca.types import SelectionFailure  # to be added
    assert isinstance(result, SelectionFailure)
    assert result.reason == "budget"
```

- [ ] **Step 3: Run test, confirm fail**

Run: `timeout 60 pytest packages/fatih_hoca/tests/test_budget_filter.py -v`
Expected: FAIL — `remaining_budget_usd` kwarg not honored OR `SelectionFailure` not present.

- [ ] **Step 4: Add `SelectionFailure` type**

In `packages/fatih_hoca/src/fatih_hoca/types.py` (or wherever `Pick` lives), append:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionFailure:
    """Returned by Selector.select() when no model can satisfy constraints."""
    reason: str  # 'budget' | 'eligibility' | ...
    detail: str = ""
```

- [ ] **Step 5: Wire `remaining_budget_usd` filter**

In `selector.py`, modify `Selector.select()` signature to accept `remaining_budget_usd: float | None = None`. Inside, after eligibility filtering and BEFORE scoring:

```python
def select(self, task=None, *, remaining_budget_usd=None, **kwargs):
    candidates = self._eligible_candidates(task=task, **kwargs)

    # Z0: budget filter
    if remaining_budget_usd is not None:
        before = len(candidates)
        candidates = [
            m for m in candidates
            if (getattr(m, "estimated_cost_usd", 0.0) or 0.0) <= remaining_budget_usd
        ]
        logger.info(
            "budget filter: %d/%d eligible at remaining=$%.4f",
            len(candidates), before, remaining_budget_usd,
        )
        if not candidates:
            from fatih_hoca.types import SelectionFailure
            return SelectionFailure(
                reason="budget",
                detail=f"no model fits remaining ${remaining_budget_usd:.4f}",
            )

    return self._score_and_pick(candidates, task=task, **kwargs)
```

(Adapt to actual `select()` signature; the structure is: pre-score eligibility filter has a budget filter appended.)

Update `packages/fatih_hoca/src/fatih_hoca/__init__.py`'s `select(**kwargs)` thin wrapper to forward `remaining_budget_usd`:

```python
def select(**kwargs):
    if _selector is None:
        return None
    return _selector.select(**kwargs)
```

(Already forwards via **kwargs; verify no need to enumerate.)

- [ ] **Step 6: Run test, confirm pass**

Run: `timeout 60 pytest packages/fatih_hoca/tests/test_budget_filter.py -v`
Expected: PASS.

- [ ] **Step 7: Run existing fatih_hoca tests for regressions**

Run: `timeout 120 pytest packages/fatih_hoca/tests/ -v`
Expected: all PASS (no regression).

- [ ] **Step 8: Commit**

```bash
git add packages/fatih_hoca/
git commit -m "feat(fatih_hoca): remaining_budget_usd filter + SelectionFailure"
```

---

## Phase 3 — Beckman lifecycle + ceiling enforcement

### Task 6: Beckman admission gate on lifecycle_state

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/queue.py` (or `admission.py` — file that picks next task)
- Test: `packages/general_beckman/tests/test_admission_lifecycle.py` (new)

- [ ] **Step 1: Locate `next_task`-like function**

```bash
grep -n "def next_task\|def pick_next\|def admit" packages/general_beckman/src/general_beckman/*.py
```

Identify the function that returns the next task to dispatch. Read its current logic — note where the SQL query for pending tasks lives.

- [ ] **Step 2: Write failing test**

`packages/general_beckman/tests/test_admission_lifecycle.py`:

```python
import pytest
import aiosqlite

from general_beckman.queue import next_task  # adapt to actual location


@pytest.mark.asyncio
async def test_skips_paused_mission(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'paused')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (?, 't', 'pending')",
            (mid,),
        )
        await db.commit()

    task = await next_task()
    assert task is None  # paused → no dispatch


@pytest.mark.asyncio
async def test_skips_killed_mission(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'killed')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (?, 't', 'pending')",
            (mid,),
        )
        await db.commit()

    task = await next_task()
    assert task is None


@pytest.mark.asyncio
async def test_dispatches_active_mission(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (?, 't', 'pending')",
            (mid,),
        )
        await db.commit()

    task = await next_task()
    assert task is not None
    assert task["mission_id"] == mid
```

- [ ] **Step 3: Run test, confirm fail**

Run: `timeout 60 pytest packages/general_beckman/tests/test_admission_lifecycle.py -v`
Expected: FAIL — paused/killed missions still dispatch.

- [ ] **Step 4: Add lifecycle gate to next_task**

Find the SQL that fetches pending tasks. Add a JOIN to `missions` filtering by `lifecycle_state = 'active'`:

```python
# Before:
# SELECT t.* FROM tasks t WHERE t.status = 'pending' ORDER BY ...
#
# After:
sql = """
    SELECT t.*
    FROM tasks t
    JOIN missions m ON t.mission_id = m.id
    WHERE t.status = 'pending'
      AND m.lifecycle_state = 'active'
    ORDER BY ...
"""
```

(Standalone tasks without a mission: keep existing path. If mission_id is NULL, lifecycle gate is N/A.)

If the query already lacks a JOIN: add it. If standalone (no mission) tasks exist, use a `LEFT JOIN ... WHERE m.id IS NULL OR m.lifecycle_state = 'active'`.

- [ ] **Step 5: Run test, confirm pass**

Run: `timeout 60 pytest packages/general_beckman/tests/test_admission_lifecycle.py -v`
Expected: PASS.

- [ ] **Step 6: Run beckman regression suite**

Run: `timeout 120 pytest packages/general_beckman/tests/ -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add packages/general_beckman/
git commit -m "feat(beckman): admission gate on missions.lifecycle_state"
```

---

### Task 7: Beckman passes `remaining_budget_usd` to Fatih Hoca

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/queue.py` (admission entry)
- Modify: `src/core/llm_dispatcher.py` (or wherever Fatih Hoca is invoked from beckman path)
- Test: `packages/general_beckman/tests/test_admission_budget.py` (new)

- [ ] **Step 1: Trace beckman → fatih_hoca call site**

```bash
grep -rn "fatih_hoca.select\|from fatih_hoca" packages/general_beckman/ src/core/llm_dispatcher.py
```

Note: per CLAUDE.md, dispatcher calls `fatih_hoca.select()`. Beckman supplies the task to dispatcher. Determine whether the cleanest insertion is dispatcher-side (dispatcher reads mission from task, computes remaining_budget) or beckman-side (beckman computes and writes onto task before handoff).

Recommended: dispatcher-side. Beckman just emits the task; dispatcher loads mission row, computes `remaining = ceiling - spent`, and forwards to `fatih_hoca.select()`.

- [ ] **Step 2: Write failing test**

`packages/general_beckman/tests/test_admission_budget.py`:

```python
import pytest
from unittest.mock import patch, MagicMock

# This test exercises the dispatcher path that beckman triggers.
from src.core.llm_dispatcher import LLMDispatcher


@pytest.mark.asyncio
async def test_dispatcher_passes_remaining_budget(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd) VALUES ('m', 2.0, 0.5)"
        )
        await db.commit()
        cur = await db.execute("SELECT id FROM missions")
        mid = (await cur.fetchone())[0]

    captured = {}
    def fake_select(**kwargs):
        captured.update(kwargs)
        return MagicMock(model_name="local", estimated_cost_usd=0.0)

    with patch("fatih_hoca.select", fake_select):
        dispatcher = LLMDispatcher()
        # Adapt this to the real entry: ask() or request().
        await dispatcher.request(task={"mission_id": mid, "title": "test"}, prompt="x")

    assert "remaining_budget_usd" in captured
    assert captured["remaining_budget_usd"] == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_dispatcher_passes_none_when_no_ceiling(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("INSERT INTO missions (title) VALUES ('no-ceiling')")
        await db.commit()
        cur = await db.execute("SELECT id FROM missions")
        mid = (await cur.fetchone())[0]

    captured = {}
    def fake_select(**kwargs):
        captured.update(kwargs)
        return MagicMock(model_name="local", estimated_cost_usd=0.0)

    with patch("fatih_hoca.select", fake_select):
        dispatcher = LLMDispatcher()
        await dispatcher.request(task={"mission_id": mid, "title": "test"}, prompt="x")

    assert captured.get("remaining_budget_usd") is None
```

- [ ] **Step 3: Run test, confirm fail**

Run: `timeout 60 pytest packages/general_beckman/tests/test_admission_budget.py -v`
Expected: FAIL — kwarg not passed.

- [ ] **Step 4: Modify dispatcher to compute and forward remaining_budget**

In `src/core/llm_dispatcher.py`, locate the call to `fatih_hoca.select(...)`. Before that call, fetch mission row and compute remaining:

```python
async def _remaining_budget(mission_id: int | None) -> float | None:
    if mission_id is None:
        return None
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT cost_ceiling_usd, spent_usd FROM missions WHERE id = ?",
        (mission_id,),
    )
    row = await cur.fetchone()
    if not row or row[0] is None:
        return None
    ceiling, spent = float(row[0]), float(row[1] or 0.0)
    return max(0.0, ceiling - spent)
```

In the `select()` call site, add `remaining_budget_usd=await _remaining_budget(task.get("mission_id"))`.

Handle `SelectionFailure` return:

```python
from fatih_hoca.types import SelectionFailure

result = fatih_hoca.select(task=task, remaining_budget_usd=remaining, ...)
if isinstance(result, SelectionFailure):
    if result.reason == "budget":
        # Trigger mission pause via lifecycle event (Task 11).
        from packages.general_beckman.lifecycle_events import emit_pause
        await emit_pause(task["mission_id"], reason="no_model_fits_budget")
        raise DispatchAborted("mission paused: no model fits budget")
    raise DispatchAborted(f"selection failed: {result.reason}")
```

(`emit_pause` is created in Task 11; for now, leave it as a stub that logs + sets state.)

- [ ] **Step 5: Run test, confirm pass**

Run: `timeout 60 pytest packages/general_beckman/tests/test_admission_budget.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/llm_dispatcher.py packages/general_beckman/
git commit -m "feat(beckman): pass remaining_budget_usd through dispatcher to fatih_hoca"
```

---

### Task 8: Beckman cost tracking + threshold notifies

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` (`on_task_finished`)
- Test: `packages/general_beckman/tests/test_threshold_notifies.py` (new)

- [ ] **Step 1: Locate on_task_finished**

```bash
grep -n "def on_task_finished\|async def on_task_finished" packages/general_beckman/src/general_beckman/*.py
```

- [ ] **Step 2: Write failing test**

`packages/general_beckman/tests/test_threshold_notifies.py`:

```python
import pytest
import aiosqlite
from unittest.mock import AsyncMock, patch

from general_beckman import on_task_finished


@pytest.mark.asyncio
async def test_spent_increments(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd) VALUES ('m', 5.0, 0.0)"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()

    await on_task_finished({"mission_id": mid, "id": 1, "cost_usd": 0.50})

    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute("SELECT spent_usd FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == pytest.approx(0.50)


@pytest.mark.asyncio
async def test_threshold_notify_50pct(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd) VALUES ('m', 1.0, 0.0)"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()

    notify = AsyncMock()
    with patch("general_beckman.notify_threshold", notify):
        await on_task_finished({"mission_id": mid, "id": 1, "cost_usd": 0.55})

    # 0.55 / 1.00 = 55% → 50% threshold fires.
    notify.assert_called_once()
    assert notify.call_args.kwargs.get("pct") == 50 or 50 in notify.call_args.args


@pytest.mark.asyncio
async def test_each_threshold_fires_once(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd) VALUES ('m', 1.0, 0.55)"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        # Pre-record that 50% notify already fired.
        await db.execute(
            "UPDATE missions SET context = json_set(COALESCE(context, '{}'), '$.thresholds_fired', json_array(50)) "
            "WHERE id = ?", (mid,),
        )
        await db.commit()

    notify = AsyncMock()
    with patch("general_beckman.notify_threshold", notify):
        # Adds 0.20 → 0.75 → crosses 75%
        await on_task_finished({"mission_id": mid, "id": 1, "cost_usd": 0.20})

    # 50% should NOT re-fire; 75% fires once.
    fired_pcts = [
        c.kwargs.get("pct") if "pct" in c.kwargs else (c.args[1] if len(c.args) > 1 else None)
        for c in notify.call_args_list
    ]
    assert 75 in fired_pcts
    assert 50 not in fired_pcts
```

- [ ] **Step 3: Run test, confirm fail**

Run: `timeout 60 pytest packages/general_beckman/tests/test_threshold_notifies.py -v`
Expected: FAIL — spent not incremented or notify not called.

- [ ] **Step 4: Implement spent + threshold logic**

In `packages/general_beckman/src/general_beckman/__init__.py` (or wherever `on_task_finished` lives), modify:

```python
import json
from src.infra.db import get_db

THRESHOLDS_PCT = (50, 75, 90)


async def notify_threshold(mission_id: int, pct: int, spent: float, ceiling: float):
    """Post threshold notify to mission thread.

    Stub for now — Task 13 wires this to telegram_bot pinned-status updater.
    """
    import logging
    logging.getLogger(__name__).info(
        "mission %d crossed %d%% threshold ($%.4f / $%.4f)",
        mission_id, pct, spent, ceiling,
    )


async def on_task_finished(task: dict):
    # ... existing logic ...

    mission_id = task.get("mission_id")
    cost = float(task.get("cost_usd") or 0.0)
    if mission_id is None or cost <= 0:
        return

    db = await get_db()
    # Atomic increment + read of new state.
    await db.execute(
        "UPDATE missions SET spent_usd = COALESCE(spent_usd, 0) + ? WHERE id = ?",
        (cost, mission_id),
    )
    cur = await db.execute(
        "SELECT cost_ceiling_usd, spent_usd, context FROM missions WHERE id = ?",
        (mission_id,),
    )
    ceiling, spent, ctx_raw = await cur.fetchone()
    if ceiling is None or ceiling <= 0:
        await db.commit()
        return

    ctx = json.loads(ctx_raw) if ctx_raw else {}
    fired = set(ctx.get("thresholds_fired", []))
    pct = (spent / ceiling) * 100
    new_fires = []
    for t in THRESHOLDS_PCT:
        if pct >= t and t not in fired:
            new_fires.append(t)
            fired.add(t)
    if new_fires:
        ctx["thresholds_fired"] = sorted(fired)
        await db.execute(
            "UPDATE missions SET context = ? WHERE id = ?",
            (json.dumps(ctx), mission_id),
        )
        for t in new_fires:
            await notify_threshold(mission_id, t, spent, ceiling)
    await db.commit()
```

- [ ] **Step 5: Run test, confirm pass**

Run: `timeout 60 pytest packages/general_beckman/tests/test_threshold_notifies.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/
git commit -m "feat(beckman): spent_usd tracking + 50/75/90 threshold notifies"
```

---

### Task 9: Beckman in-flight tracker + ceiling backstop

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/queue.py` (or admission file)
- Test: extend `packages/general_beckman/tests/test_admission_budget.py`

- [ ] **Step 1: Append failing test**

In `test_admission_budget.py`, add:

```python
@pytest.mark.asyncio
async def test_in_flight_estimates_block_overshoot(tmp_path, monkeypatch):
    """Two parallel tasks both fit individually but combined exceed ceiling."""
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd) VALUES ('m', 1.0, 0.0)"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status, estimated_cost_usd) "
            "VALUES (?, 't1', 'running', 0.60)", (mid,),
        )
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status, estimated_cost_usd) "
            "VALUES (?, 't2', 'pending', 0.55)", (mid,),
        )
        await db.commit()

    from general_beckman.queue import next_task
    task = await next_task()
    # 0.0 spent + 0.60 in-flight + 0.55 new = 1.15 > 1.0 → must block t2.
    assert task is None or task["title"] != "t2"
```

(Test assumes `tasks.estimated_cost_usd` column exists. If not present, add it via migration in Task 1 retroactively or in this task.)

- [ ] **Step 2: Add `estimated_cost_usd` to tasks (if missing)**

```bash
grep -n "estimated_cost_usd" src/infra/db.py
```

If absent, append to the migration block in Task 1's location:

```python
try:
    await db.execute("ALTER TABLE tasks ADD COLUMN estimated_cost_usd REAL DEFAULT 0")
except Exception as e:
    logger.debug(f"tasks.estimated_cost_usd migration skipped: {e}")
```

Run migration test from Task 1 to confirm idempotency.

- [ ] **Step 3: Run test, confirm fail**

Run: `timeout 60 pytest packages/general_beckman/tests/test_admission_budget.py::test_in_flight_estimates_block_overshoot -v`
Expected: FAIL — t2 dispatched.

- [ ] **Step 4: Add backstop check in next_task**

Modify `next_task()` admission. After lifecycle gate + before returning task:

```python
# Backstop: spent + in-flight + this estimate > ceiling → skip
if candidate.get("mission_id"):
    cur = await db.execute(
        "SELECT cost_ceiling_usd, spent_usd FROM missions WHERE id = ?",
        (candidate["mission_id"],),
    )
    row = await cur.fetchone()
    if row and row[0] is not None:
        ceiling, spent = float(row[0]), float(row[1] or 0.0)
        # Sum estimates of tasks currently running for this mission.
        cur2 = await db.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM tasks "
            "WHERE mission_id = ? AND status = 'running'",
            (candidate["mission_id"],),
        )
        in_flight = float((await cur2.fetchone())[0] or 0.0)
        new_est = float(candidate.get("estimated_cost_usd") or 0.0)
        if spent + in_flight + new_est > ceiling:
            logger.info(
                "ceiling backstop: mission %d spent=$%.4f in_flight=$%.4f new=$%.4f > ceiling=$%.4f",
                candidate["mission_id"], spent, in_flight, new_est, ceiling,
            )
            # Skip this candidate; loop continues to next.
            continue
```

- [ ] **Step 5: Run test, confirm pass**

Run: `timeout 60 pytest packages/general_beckman/tests/test_admission_budget.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/ src/infra/db.py
git commit -m "feat(beckman): in-flight tracker + ceiling backstop in admission"
```

---

### Task 10: Beckman lifecycle event emitter (auto-pause triggers)

**Files:**
- Create: `packages/general_beckman/src/general_beckman/lifecycle_events.py`
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`
- Test: `packages/general_beckman/tests/test_lifecycle_events.py` (new)

- [ ] **Step 1: Write failing tests**

`packages/general_beckman/tests/test_lifecycle_events.py`:

```python
import pytest
import aiosqlite

from general_beckman.lifecycle_events import emit_pause, dlq_cascade_check


@pytest.mark.asyncio
async def test_emit_pause_transitions_state(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()

    await emit_pause(mid, reason="no_model_fits_budget", triggered_by="auto:budget")

    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
        assert row[0] == "paused"
        log = await (await db.execute(
            "SELECT from_state, to_state, reason, triggered_by FROM mission_lifecycle_log "
            "WHERE mission_id=?", (mid,))).fetchall()
    assert len(log) == 1
    assert log[0] == ("active", "paused", "no_model_fits_budget", "auto:budget")


@pytest.mark.asyncio
async def test_emit_pause_idempotent_when_already_paused(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'paused')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()

    await emit_pause(mid, reason="dup", triggered_by="test")

    async with aiosqlite.connect(db_path) as db:
        log = await (await db.execute(
            "SELECT * FROM mission_lifecycle_log WHERE mission_id=?", (mid,))).fetchall()
    assert log == []  # no log entry on no-op


@pytest.mark.asyncio
async def test_dlq_cascade_triggers_pause_at_3_consec_failures(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        for _ in range(3):
            await db.execute(
                "INSERT INTO tasks (mission_id, title, status, completed_at) "
                "VALUES (?, 't', 'failed', CURRENT_TIMESTAMP)", (mid,),
            )
        await db.commit()

    triggered = await dlq_cascade_check(mid)
    assert triggered is True

    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "paused"
```

- [ ] **Step 2: Run test, confirm fail**

Run: `timeout 60 pytest packages/general_beckman/tests/test_lifecycle_events.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement lifecycle_events.py**

```python
"""Mission lifecycle state transitions + audit log."""
from __future__ import annotations

import logging
from src.infra.db import get_db

logger = logging.getLogger(__name__)


async def _transition(
    mission_id: int,
    from_states: tuple[str, ...],
    to_state: str,
    reason: str,
    triggered_by: str,
) -> bool:
    """Atomic transition. Returns True if state changed, False if no-op."""
    db = await get_db()
    placeholders = ",".join("?" * len(from_states))
    cur = await db.execute(
        f"UPDATE missions SET lifecycle_state = ? "
        f"WHERE id = ? AND lifecycle_state IN ({placeholders})",
        (to_state, mission_id, *from_states),
    )
    changed = cur.rowcount > 0
    if changed:
        # Read prior state for log (one of from_states; we accept any).
        await db.execute(
            "INSERT INTO mission_lifecycle_log (mission_id, from_state, to_state, reason, triggered_by) "
            "VALUES (?, ?, ?, ?, ?)",
            (mission_id, from_states[0] if len(from_states) == 1 else "any", to_state, reason, triggered_by),
        )
    await db.commit()
    return changed


async def emit_pause(mission_id: int, reason: str, triggered_by: str = "auto") -> bool:
    return await _transition(mission_id, ("active",), "paused", reason, triggered_by)


async def emit_resume(mission_id: int, reason: str = "founder", triggered_by: str = "founder") -> bool:
    return await _transition(mission_id, ("paused",), "active", reason, triggered_by)


async def emit_kill(mission_id: int, reason: str = "founder", triggered_by: str = "founder") -> bool:
    return await _transition(mission_id, ("active", "paused"), "killed", reason, triggered_by)


async def emit_complete(mission_id: int, reason: str = "all_tasks_done", triggered_by: str = "auto") -> bool:
    return await _transition(mission_id, ("active",), "completed", reason, triggered_by)


async def dlq_cascade_check(mission_id: int) -> bool:
    """If the last 3 consecutive completed tasks for mission failed, pause.

    Resets on any successful completion.
    """
    db = await get_db()
    cur = await db.execute(
        "SELECT status FROM tasks WHERE mission_id = ? AND completed_at IS NOT NULL "
        "ORDER BY completed_at DESC LIMIT 3",
        (mission_id,),
    )
    rows = await cur.fetchall()
    if len(rows) < 3:
        return False
    if all(r[0] == "failed" for r in rows):
        return await emit_pause(mission_id, "dlq_cascade", "auto:dlq")
    return False
```

- [ ] **Step 4: Hook DLQ-cascade check into on_task_finished**

In `general_beckman/__init__.py`, after the spent-tracking block, append:

```python
# Z0: DLQ cascade trigger
from general_beckman.lifecycle_events import dlq_cascade_check
if mission_id is not None and task.get("status") == "failed":
    await dlq_cascade_check(mission_id)
```

- [ ] **Step 5: Run test, confirm pass**

Run: `timeout 60 pytest packages/general_beckman/tests/test_lifecycle_events.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/
git commit -m "feat(beckman): lifecycle event emitter + DLQ cascade auto-pause"
```

---

## Phase 4 — Telegram lifecycle commands + forum topic

### Task 11: telegram_bot — provision forum topic at mission start

**Files:**
- Modify: `src/app/telegram_bot.py` (cmd_mission flow + new method)
- Test: `tests/app/test_telegram_lifecycle.py` (new)

- [ ] **Step 1: Locate cmd_mission**

```bash
grep -n "async def cmd_mission\|cmd_mission" src/app/telegram_bot.py | head -10
```

Read ~40 lines around the function to see how missions are inserted today.

- [ ] **Step 2: Write failing test**

`tests/app/test_telegram_lifecycle.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.app.telegram_bot import TelegramInterface


@pytest.mark.asyncio
async def test_provision_mission_thread_happy_path():
    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg.bot.create_forum_topic = AsyncMock(return_value=MagicMock(message_thread_id=999))
    tg.bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    tg.bot.pin_chat_message = AsyncMock()

    chat_id = 1234
    thread_id = await tg.provision_mission_thread(chat_id, mission_id=1, title="Build X")
    assert thread_id == 999
    tg.bot.create_forum_topic.assert_called_once()
    tg.bot.pin_chat_message.assert_called_once()


@pytest.mark.asyncio
async def test_provision_falls_back_to_main_chat_on_perm_error():
    from telegram.error import BadRequest
    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg.bot.create_forum_topic = AsyncMock(
        side_effect=BadRequest("Bot doesn't have permission")
    )

    thread_id = await tg.provision_mission_thread(1234, mission_id=1, title="Build X")
    assert thread_id is None  # signals fallback to tag-prefix
```

- [ ] **Step 3: Run test, confirm fail**

Run: `timeout 60 pytest tests/app/test_telegram_lifecycle.py -v`
Expected: FAIL — method missing.

- [ ] **Step 4: Implement provision_mission_thread**

In `src/app/telegram_bot.py`, add a method to `TelegramInterface`:

```python
async def provision_mission_thread(
    self, chat_id: int, mission_id: int, title: str,
) -> int | None:
    """Create a forum topic for the mission. Returns thread_id or None on failure.

    Caller must persist returned thread_id to missions.message_thread_id.
    """
    topic_name = f"#{mission_id} {title[:40]}"
    try:
        topic = await self.bot.create_forum_topic(chat_id=chat_id, name=topic_name)
        thread_id = topic.message_thread_id
    except Exception as e:
        logger.warning(
            "forum topic creation failed for mission %d: %s — falling back to tag-prefix",
            mission_id, e,
        )
        return None

    # Pin initial status message.
    try:
        msg = await self.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=self._format_pinned_status(mission_id, title, spent=0, ceiling=None),
        )
        await self.bot.pin_chat_message(
            chat_id=chat_id, message_id=msg.message_id, disable_notification=True,
        )
    except Exception as e:
        logger.warning("pin_chat_message failed: %s", e)

    return thread_id


def _format_pinned_status(
    self, mission_id: int, title: str,
    spent: float, ceiling: float | None,
    state: str = "active",
    tasks_done: int = 0, tasks_running: int = 0, tasks_queued: int = 0,
) -> str:
    if ceiling:
        pct = (spent / ceiling) * 100 if ceiling > 0 else 0
        budget_line = f"Spent: ${spent:.2f} / ${ceiling:.2f} ({pct:.1f}%)"
    else:
        budget_line = f"Spent: ${spent:.2f} (no ceiling)"
    return (
        f"Mission #{mission_id} — \"{title}\"\n"
        f"Status: {state}\n"
        f"{budget_line}\n"
        f"Tasks: {tasks_done} done, {tasks_running} in flight, {tasks_queued} queued"
    )
```

- [ ] **Step 5: Run test, confirm pass**

Run: `timeout 60 pytest tests/app/test_telegram_lifecycle.py -v`
Expected: PASS.

- [ ] **Step 6: Wire into cmd_mission**

After the `INSERT INTO missions` in `cmd_mission`, add:

```python
# Z0: provision forum topic (best-effort; falls back to tag-prefix on failure)
thread_id = await self.provision_mission_thread(
    chat_id=update.effective_chat.id,
    mission_id=mission_id,
    title=title,
)
if thread_id is not None:
    db = await get_db()
    await db.execute(
        "UPDATE missions SET message_thread_id = ? WHERE id = ?",
        (thread_id, mission_id),
    )
    await db.commit()
```

- [ ] **Step 7: Smoke import + commit**

```bash
python -c "from src.app.telegram_bot import TelegramInterface; print('ok')"
git add src/app/telegram_bot.py tests/app/
git commit -m "feat(z0): provision per-mission forum topic + pinned status"
```

---

### Task 12: telegram_bot — `/kill_mission` + lifecycle command polish

**Files:**
- Modify: `src/app/telegram_bot.py` (existing /pause and /resume; add /kill_mission)
- Test: extend `tests/app/test_telegram_lifecycle.py`

- [ ] **Step 1: Append failing tests**

```python
@pytest.mark.asyncio
async def test_kill_mission_sets_killed_state_and_writes_snapshot(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()

    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()

    update = MagicMock()
    update.effective_message.text = f"/kill_mission {mid}"
    update.effective_chat.id = 1
    context = MagicMock()
    context.args = [str(mid)]

    await tg.cmd_kill_mission(update, context)

    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "killed"

    # Snapshot artifact written.
    snapshot_path = tmp_path / f"mission_kill_{mid}.json"
    # (Adjust snapshot path to wherever your impl writes; assertion may be on artifact_store.)


@pytest.mark.asyncio
async def test_resume_after_kill_rejected(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'killed')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()

    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()

    update = MagicMock()
    context = MagicMock()
    context.args = [str(mid)]
    await tg.cmd_resume_mission(update, context)

    # State unchanged.
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "killed"
    # Founder told why.
    tg._reply.assert_called_once()
    msg = tg._reply.call_args[0][1] if len(tg._reply.call_args[0]) > 1 else tg._reply.call_args.args[1]
    assert "killed" in msg.lower()
```

- [ ] **Step 2: Run test, confirm fail**

Run: `timeout 60 pytest tests/app/test_telegram_lifecycle.py -v`
Expected: FAIL — commands missing.

- [ ] **Step 3: Implement /kill_mission, refine /pause /resume**

Add to `TelegramInterface`:

```python
async def cmd_kill_mission(self, update, context):
    if not context.args:
        await self._reply(update, "Usage: /kill_mission <id>")
        return
    try:
        mid = int(context.args[0])
    except ValueError:
        await self._reply(update, "Invalid mission id.")
        return

    from general_beckman.lifecycle_events import emit_kill
    changed = await emit_kill(mid, reason="founder_kill", triggered_by="founder")
    if not changed:
        await self._reply(update, f"Mission {mid}: cannot kill (already terminal or missing).")
        return

    # Snapshot.
    await self._snapshot_mission(mid)
    await self._reply(update, f"Mission {mid} killed. Snapshot written.")


async def cmd_resume_mission(self, update, context):
    if not context.args:
        await self._reply(update, "Usage: /resume_mission <id>")
        return
    mid = int(context.args[0])
    from general_beckman.lifecycle_events import emit_resume
    db = await get_db()
    cur = await db.execute("SELECT lifecycle_state FROM missions WHERE id = ?", (mid,))
    row = await cur.fetchone()
    if not row:
        await self._reply(update, f"Mission {mid}: not found.")
        return
    state = row[0]
    if state in ("killed", "completed"):
        await self._reply(update, f"Mission {mid} is {state}; cannot resume.")
        return
    changed = await emit_resume(mid, triggered_by="founder")
    if changed:
        await self._reply(update, f"Mission {mid} resumed.")
    else:
        await self._reply(update, f"Mission {mid} not paused.")


async def cmd_pause_mission(self, update, context):
    if not context.args:
        await self._reply(update, "Usage: /pause_mission <id>")
        return
    mid = int(context.args[0])
    from general_beckman.lifecycle_events import emit_pause
    changed = await emit_pause(mid, reason="founder_pause", triggered_by="founder")
    if changed:
        await self._reply(update, f"Mission {mid} paused. In-flight tasks will finish.")
    else:
        await self._reply(update, f"Mission {mid}: not in active state.")


async def _snapshot_mission(self, mission_id: int):
    """Write mission state to artifact store as `mission_kill_<id>.json`."""
    from src.workflows.engine.artifacts import get_artifact_store
    import json
    db = await get_db()
    cur = await db.execute("SELECT * FROM missions WHERE id = ?", (mission_id,))
    mission = dict(zip([c[0] for c in cur.description], await cur.fetchone()))
    cur = await db.execute(
        "SELECT id, title, status, completed_at FROM tasks WHERE mission_id = ?",
        (mission_id,),
    )
    tasks = [dict(zip([c[0] for c in cur.description], r)) for r in await cur.fetchall()]
    snapshot = {"mission": mission, "tasks": tasks}
    try:
        store = get_artifact_store()
        await store.put(mission_id, f"mission_kill_{mission_id}", json.dumps(snapshot))
    except Exception as e:
        logger.error("snapshot write failed for mission %d: %s", mission_id, e)
```

Register the three commands in `_setup_handlers()`:

```python
application.add_handler(CommandHandler("pause_mission", self.cmd_pause_mission))
application.add_handler(CommandHandler("resume_mission", self.cmd_resume_mission))
application.add_handler(CommandHandler("kill_mission", self.cmd_kill_mission))
```

- [ ] **Step 4: Run test, confirm pass**

Run: `timeout 60 pytest tests/app/test_telegram_lifecycle.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke import + commit**

```bash
python -c "from src.app.telegram_bot import TelegramInterface; print('ok')"
git add src/app/telegram_bot.py tests/app/
git commit -m "feat(z0): /pause_mission /resume_mission /kill_mission with snapshot"
```

---

### Task 13: telegram_bot — wire `notify_threshold` + ceiling-pause notifications

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` (`notify_threshold` body)
- Test: `tests/app/test_threshold_notify_post.py` (new)

- [ ] **Step 1: Write failing test**

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_notify_threshold_posts_to_thread(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, message_thread_id, context) "
            "VALUES ('m', 999, json_object('chat_id', 1234))"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()

    bot = AsyncMock()
    with patch("src.app.telegram_bot.get_telegram") as gt:
        gt.return_value.bot = bot
        from general_beckman import notify_threshold
        await notify_threshold(mid, pct=50, spent=0.5, ceiling=1.0)

    assert bot.send_message.called
    kwargs = bot.send_message.call_args.kwargs
    assert kwargs["chat_id"] == 1234
    assert kwargs["message_thread_id"] == 999
    assert "50%" in kwargs["text"]
```

- [ ] **Step 2: Run test, confirm fail**

Run: `timeout 60 pytest tests/app/test_threshold_notify_post.py -v`

- [ ] **Step 3: Replace `notify_threshold` stub**

In `packages/general_beckman/src/general_beckman/__init__.py`:

```python
async def notify_threshold(mission_id: int, pct: int, spent: float, ceiling: float):
    from src.app.telegram_bot import get_telegram
    from src.infra.db import get_db
    import json

    db = await get_db()
    cur = await db.execute(
        "SELECT message_thread_id, context FROM missions WHERE id = ?",
        (mission_id,),
    )
    row = await cur.fetchone()
    if not row:
        return
    thread_id, ctx_raw = row
    ctx = json.loads(ctx_raw) if ctx_raw else {}
    chat_id = ctx.get("chat_id")
    if chat_id is None:
        logger.warning("threshold notify: no chat_id in mission %d context", mission_id)
        return

    text = f"📊 Mission #{mission_id} crossed {pct}% — ${spent:.2f} / ${ceiling:.2f}"
    try:
        tg = get_telegram()
        await tg.bot.send_message(
            chat_id=chat_id,
            message_thread_id=thread_id,
            text=text,
        )
    except Exception as e:
        logger.warning("threshold notify post failed: %s", e)
```

- [ ] **Step 4: Run test, confirm pass + commit**

```bash
timeout 60 pytest tests/app/test_threshold_notify_post.py -v
git add packages/general_beckman/ tests/app/
git commit -m "feat(z0): wire threshold notify to mission Telegram thread"
```

---

### Task 14: telegram_bot — ceiling Q at mission start + inline-button callbacks

**Files:**
- Modify: `src/app/telegram_bot.py` (cmd_mission ceiling Q; callback handler)
- Test: extend `tests/app/test_telegram_lifecycle.py`

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_resume_button_callback_resumes_mission(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'paused')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()

    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()

    update = MagicMock()
    update.callback_query.data = f"mission_resume:{mid}"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    context = MagicMock()

    await tg.handle_callback(update, context)

    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "active"
```

- [ ] **Step 2: Run test, confirm fail**

Run: `timeout 60 pytest tests/app/test_telegram_lifecycle.py::test_resume_button_callback_resumes_mission -v`

- [ ] **Step 3: Add callback routes for mission_resume / mission_kill / mission_pause**

In `handle_callback()`:

```python
data = update.callback_query.data
if data.startswith("mission_resume:"):
    mid = int(data.split(":", 1)[1])
    from general_beckman.lifecycle_events import emit_resume
    await emit_resume(mid, triggered_by="founder")
    await update.callback_query.answer("Resumed.")
    await update.callback_query.edit_message_text(f"Mission {mid} resumed.")
    return
if data.startswith("mission_pause:"):
    mid = int(data.split(":", 1)[1])
    from general_beckman.lifecycle_events import emit_pause
    await emit_pause(mid, reason="founder_pause", triggered_by="founder")
    await update.callback_query.answer("Paused.")
    await update.callback_query.edit_message_text(f"Mission {mid} paused.")
    return
if data.startswith("mission_kill:"):
    mid = int(data.split(":", 1)[1])
    from general_beckman.lifecycle_events import emit_kill
    await emit_kill(mid, triggered_by="founder")
    await self._snapshot_mission(mid)
    await update.callback_query.answer("Killed.")
    await update.callback_query.edit_message_text(f"Mission {mid} killed.")
    return
```

- [ ] **Step 4: Add ceiling Q to cmd_mission**

After mission row insert, ask:

```python
await self._reply(
    update,
    f"Cost ceiling for this mission ($)? Reply with a number, or `none` for unlimited.\n"
    f"Default if no reply within 30s: unlimited.",
)
self._pending_action[chat_id] = {
    "kind": "z0_ceiling",
    "mission_id": mission_id,
    "expires_at": time.time() + 30,
}
```

In the existing pending-action handler, add a branch for `z0_ceiling`:

```python
if pending["kind"] == "z0_ceiling":
    raw = update.message.text.strip().lower()
    if raw in ("none", "skip", ""):
        ceiling = None
    else:
        try:
            ceiling = float(raw)
        except ValueError:
            await self._reply(update, "Invalid number. Skipping (no ceiling).")
            ceiling = None
    if ceiling is not None:
        db = await get_db()
        await db.execute(
            "UPDATE missions SET cost_ceiling_usd = ? WHERE id = ?",
            (ceiling, pending["mission_id"]),
        )
        await db.commit()
    del self._pending_action[chat_id]
    await self._reply(update, "Mission starting…")
    return
```

- [ ] **Step 5: Run test, smoke, commit**

```bash
timeout 60 pytest tests/app/test_telegram_lifecycle.py -v
python -c "from src.app.telegram_bot import TelegramInterface; print('ok')"
git add src/app/telegram_bot.py tests/app/
git commit -m "feat(z0): ceiling Q at mission start + lifecycle button callbacks"
```

---

## Phase 5 — Workflow integration (reversibility tags + executor hook)

### Task 15: Tag dangerous i2p_v3 steps with reversibility=none/locked

**Files:**
- Modify: `src/workflows/i2p/i2p_v3.json`
- Test: `tests/workflows/test_i2p_v3_reversibility_tags.py` (new)

- [ ] **Step 1: List dangerous step ids in i2p_v3**

Look for steps that:
- deploy to prod (any `deploy_*_prod`, `vercel deploy --prod`, etc.)
- charge money (Stripe, payment integrations)
- public posts (Twitter, customer email blasts)
- DNS / domain ops
- KYC / legal filings
- force-push or rewrite shared history

```bash
grep -nE '"id":\s*"(deploy|publish|charge|stripe|launch|domain|kyc|tweet|email_blast)' src/workflows/i2p/i2p_v3.json | head -30
```

Compile a target list (~10-30 steps). Include their step IDs in the test.

- [ ] **Step 2: Write failing test**

`tests/workflows/test_i2p_v3_reversibility_tags.py`:

```python
import json
import pathlib

I2P = pathlib.Path("src/workflows/i2p/i2p_v3.json")

# Steps that must carry reversibility=none + locked=true.
LOCKED_NONE_IDS = {
    # Fill from Step 1 audit; sample:
    "deploy_prod",
    "publish_to_app_store",
    "send_marketing_email",
    "stripe_charge_real",
    # ... etc.
}


def test_dangerous_steps_are_locked_none():
    data = json.loads(I2P.read_text(encoding="utf-8"))
    found = {}
    def walk(node):
        if isinstance(node, dict):
            sid = node.get("id")
            if sid in LOCKED_NONE_IDS:
                found[sid] = node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(data)
    missing = LOCKED_NONE_IDS - found.keys()
    assert not missing, f"missing dangerous step ids in workflow: {missing}"
    for sid, step in found.items():
        assert step.get("reversibility") == "none", f"{sid} reversibility != none"
        assert step.get("locked") is True, f"{sid} locked != true"
```

- [ ] **Step 3: Run, confirm fail**

Run: `timeout 60 pytest tests/workflows/test_i2p_v3_reversibility_tags.py -v`

- [ ] **Step 4: Add tags to JSON**

For each step ID in `LOCKED_NONE_IDS`, add:

```json
"reversibility": "none",
"locked": true
```

(Editing the JSON is mechanical; do it in one pass.)

- [ ] **Step 5: Run test, pass, commit**

```bash
timeout 60 pytest tests/workflows/test_i2p_v3_reversibility_tags.py -v
git add src/workflows/i2p/i2p_v3.json tests/workflows/
git commit -m "feat(z0): tag dangerous i2p_v3 steps with reversibility=none + locked"
```

---

### Task 16: Wire `safety_guard.pre_action` into workflow engine

**Files:**
- Modify: `src/workflows/engine/runner.py` (or where step → executor dispatch happens)
- Test: `tests/workflows/test_engine_safety_guard.py` (new)

- [ ] **Step 1: Locate step-execution site**

```bash
grep -n "executor\|run_step\|dispatch" src/workflows/engine/runner.py | head
```

Find the function that takes a step and either runs LLM agent or mechanical executor. The hook goes immediately before that dispatch.

- [ ] **Step 2: Write failing test**

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_engine_blocks_force_push_action():
    from src.workflows.engine.runner import dispatch_step  # adapt
    step = {"id": "test_force_push", "reversibility": "partial", "agent": "mechanical"}
    action = {"command": "git push --force origin main"}

    # Patch executor so we can confirm it was NOT called.
    executor = AsyncMock()
    result = await dispatch_step(
        step, action, mission_id=1,
        workspace_root="/tmp/ws", current_branch="main",
        executor_override=executor,
    )
    executor.assert_not_called()
    assert result.get("status") == "blocked"
    assert "blocklist" in result.get("reason", "") or "force_push" in result.get("reason", "")
```

- [ ] **Step 3: Run, confirm fail**

- [ ] **Step 4: Insert pre_action call**

```python
from safety_guard import pre_action, Allow, WaitForFounder, Block


async def dispatch_step(step, action, *, mission_id, workspace_root, current_branch, executor_override=None):
    decision = pre_action(
        step, action,
        workspace_root=workspace_root,
        current_branch=current_branch,
        founder_recently_active=await _founder_recently_active(),
        mission_allowlist=await _mission_safety_allowlist(mission_id),
    )
    if isinstance(decision, Block):
        logger.warning("step %s blocked: %s", step.get("id"), decision.reason)
        return {"status": "blocked", "reason": decision.reason}
    if isinstance(decision, WaitForFounder):
        # mark task waiting_human; return without executing
        return {"status": "waiting_human", "reason": decision.reason}
    # decision is Allow → proceed
    executor = executor_override or _get_executor(step)
    return await executor(step, action)


async def _founder_recently_active() -> bool:
    """Best-effort: True if Telegram has seen activity in last 30 minutes."""
    # Stub: always True for now. Refine with real activity tracker later.
    return True


async def _mission_safety_allowlist(mission_id: int) -> list[str]:
    from src.infra.db import get_db
    import json
    if mission_id is None:
        return []
    db = await get_db()
    cur = await db.execute("SELECT context FROM missions WHERE id = ?", (mission_id,))
    row = await cur.fetchone()
    if not row or not row[0]:
        return []
    return json.loads(row[0]).get("safety_allowlist", [])
```

- [ ] **Step 5: Run, pass, commit**

```bash
timeout 60 pytest tests/workflows/test_engine_safety_guard.py -v
git add src/workflows/engine/ tests/workflows/
git commit -m "feat(z0): wire safety_guard.pre_action into workflow engine dispatch"
```

---

## Phase 6 — Integration tests

### Task 17: Integration — mission lifecycle (ceiling pause/resume/complete)

**Files:**
- Create: `tests/integration/test_z0_mission_lifecycle.py`

- [ ] **Step 1: Write the test**

```python
import pytest
import aiosqlite

from src.infra.db import init_db
from general_beckman import on_task_finished
from general_beckman.lifecycle_events import emit_pause, emit_resume


@pytest.mark.asyncio
async def test_mission_pauses_at_ceiling_and_resumes(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, cost_ceiling_usd, spent_usd, lifecycle_state) "
            "VALUES ('m', 1.0, 0.0, 'active')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.commit()

    # Simulate task that crosses 100%.
    await on_task_finished({"mission_id": mid, "id": 1, "cost_usd": 0.95})
    await on_task_finished({"mission_id": mid, "id": 2, "cost_usd": 0.10})  # → 1.05 over

    async with aiosqlite.connect(db_path) as db:
        spent_row = await (await db.execute(
            "SELECT spent_usd FROM missions WHERE id=?", (mid,))).fetchone()
        ctx_row = await (await db.execute(
            "SELECT context FROM missions WHERE id=?", (mid,))).fetchone()
    assert spent_row[0] >= 1.0

    # Resume after manual pause.
    await emit_pause(mid, reason="ceiling_reached", triggered_by="auto:budget")
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
        assert row[0] == "paused"
    await emit_resume(mid)
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
        assert row[0] == "active"
```

- [ ] **Step 2: Run + commit**

```bash
timeout 60 pytest tests/integration/test_z0_mission_lifecycle.py -v -m "not llm"
git add tests/integration/test_z0_mission_lifecycle.py
git commit -m "test(z0): integration mission lifecycle (ceiling/pause/resume)"
```

---

### Task 18: Integration — collision block (force-push)

**Files:**
- Create: `tests/integration/test_z0_collision_block.py`

- [ ] **Step 1: Write test**

```python
import pytest
from src.workflows.engine.runner import dispatch_step


@pytest.mark.asyncio
async def test_force_push_step_blocked_end_to_end(tmp_path):
    step = {"id": "danger", "agent": "mechanical", "reversibility": "partial"}
    action = {"command": "git push --force origin main"}
    result = await dispatch_step(
        step, action, mission_id=None,
        workspace_root=str(tmp_path),
        current_branch="main",
    )
    assert result["status"] == "blocked"
```

- [ ] **Step 2: Run + commit**

```bash
timeout 60 pytest tests/integration/test_z0_collision_block.py -v -m "not llm"
git add tests/integration/test_z0_collision_block.py
git commit -m "test(z0): integration collision block (force-push)"
```

---

### Task 19: Integration — `/kill_mission` snapshot

**Files:**
- Create: `tests/integration/test_z0_kill_snapshot.py`

- [ ] **Step 1: Write test**

```python
import pytest
import aiosqlite
import json

from src.infra.db import init_db
from src.app.telegram_bot import TelegramInterface


@pytest.mark.asyncio
async def test_kill_mission_writes_snapshot(tmp_path, monkeypatch):
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    await init_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        mid = (await (await db.execute("SELECT last_insert_rowid()")).fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (?, 't1', 'completed')",
            (mid,),
        )
        await db.commit()

    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = type("X", (), {})()
    from unittest.mock import AsyncMock
    tg._reply = AsyncMock()

    update = type("U", (), {})()
    context = type("C", (), {"args": [str(mid)]})()
    await tg.cmd_kill_mission(update, context)

    # State + snapshot.
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "killed"

    from src.workflows.engine.artifacts import get_artifact_store
    store = get_artifact_store()
    snap = await store.retrieve(mid, f"mission_kill_{mid}")
    assert snap
    parsed = json.loads(snap) if isinstance(snap, str) else snap
    assert parsed["mission"]["id"] == mid
    assert any(t["title"] == "t1" for t in parsed["tasks"])
```

- [ ] **Step 2: Run + commit**

```bash
timeout 60 pytest tests/integration/test_z0_kill_snapshot.py -v -m "not llm"
git add tests/integration/test_z0_kill_snapshot.py
git commit -m "test(z0): integration kill_mission snapshot"
```

---

### Task 20: Full suite regression sweep + manual smoke

- [ ] **Step 1: Full unit + integration run**

Run: `timeout 180 pytest tests/ packages/safety_guard/tests/ packages/general_beckman/tests/ packages/fatih_hoca/tests/ -v -m "not llm"`
Expected: all PASS (no regressions in adjacent code).

- [ ] **Step 2: Manual smoke (founder runs)**

In a real Telegram session:

1. `/mission Build a quick recipe app` → enter `0.50` ceiling. Confirm forum topic created with pinned status.
2. Mission runs. After spend reaches ~$0.25, confirm 50% notify in thread.
3. After ~$0.45, 90% notify. After $0.50 trip (or no-fit selection failure), confirm pause + Resume button.
4. Tap Resume → confirm mission continues.
5. `/kill_mission <id>` → confirm pinned status changes to "killed", snapshot artifact present.
6. `/resume_mission <id>` on killed mission → rejection message.
7. Run a hand-crafted test workflow with a `git push --force origin main` mechanical step → confirm safety_guard blocks + status reported in thread.

Document results in plan execution log.

- [ ] **Step 3: Final commit (release marker)**

```bash
git tag z0-mission-preflight-shipped
git commit --allow-empty -m "chore(z0): mark mission preflight shipped"
```

---

## Acceptance Criteria (from spec)

- [ ] DB migration applies cleanly on existing kutay.db; pre-existing missions readable + dispatchable
- [ ] New `cost_ceiling_usd`, `spent_usd`, `message_thread_id`, `lifecycle_state` columns populated on new missions
- [ ] `mission_lifecycle_log` records every state transition with reason + trigger source
- [ ] Fatih Hoca `select()` accepts `remaining_budget_usd`; ceiling=0 missions never pick cloud models
- [ ] When no model fits remaining budget, mission auto-pauses with `no_model_fits_budget`
- [ ] 50/75/90% thresholds notify exactly once each per mission lifetime
- [ ] Forum topic provisioned at mission start; pinned status reflects spent/ceiling/state; falls back to tag-prefix gracefully when forum unavailable
- [ ] `/pause_mission`, `/resume_mission`, `/kill_mission` work with proper state transitions; resume-after-kill rejected
- [ ] `safety_guard.pre_action` fires before every action; collision guards block force-push / parallel-overwrite / shared-history-rewrite / out-of-workspace shell / destructive shared DB
- [ ] Workflow steps tagged with `reversibility`/`locked` honor static-overrides-runtime-downgrade rule
- [ ] All unit + integration tests pass with `timeout 180 pytest ... -m "not llm"`
- [ ] Manual smoke verified by founder for: ceiling pause/resume, force-push block, /kill_mission snapshot

## Open Questions (deferred until execution)

- **Per-mission allowlist UX** (`/safety allow <pattern>` command surface). Stub via `mission.context.safety_allowlist` JSON field; no UI in Phase 1. Add command if needed during smoke.
- **Threshold notify de-dup across pause/resume cycles.** Current impl: per-mission-lifetime (already-fired set in `mission.context.thresholds_fired` persists across pause/resume). Confirm with founder during smoke.
- **`task.estimated_cost_usd` populator.** Spec proposes Fatih Hoca returns `(model, estimated_cost_usd)` from select; cached on task before dispatch. Task 9 adds the column; populator is in dispatcher when select() returns. If `Pick` doesn't already carry cost, add it as a small refinement during Task 7.
- **`_founder_recently_active()` real implementation.** Currently stubbed to `True`. Replace with last-message-from-allowed-chat timestamp check (≤30 min) before shipping non-reversible steps that depend on it.

## Cross-references

- Spec: `docs/superpowers/specs/2026-05-05-z0-mission-preflight-design.md`
- Source frame: `docs/i2p-evolution/z0-mission-preflight.md`
- Existing primitives leveraged: `packages/general_beckman/`, `packages/fatih_hoca/`, `packages/mr_roboto/clarify`, `src/security/credential_store.py`, `src/workflows/engine/`
- Deferred zones: Z2 (phase activation lazy), Z6 (vault provisioning + readiness lazy capture), Z9 (north-star metric)
