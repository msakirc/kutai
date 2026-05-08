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
