import pytest

from safety_guard import pre_action, Reversibility
from safety_guard.executor_hook import Allow, Block, WaitForFounder


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


def test_none_locked_waits_when_idle():
    step = {"reversibility": "none", "locked": True}
    action = {"command": "stripe-cli charge --amount 100"}
    decision = pre_action(
        step, action,
        workspace_root="/tmp/ws", current_branch="feat",
        founder_recently_active=False,
    )
    assert isinstance(decision, WaitForFounder)


def test_none_proceeds_when_founder_active():
    step = {"reversibility": "none"}
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
    assert isinstance(decision, WaitForFounder)


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
    step = {"reversibility": "partial"}
    action = {"command": "git push --force origin main"}
    decision = pre_action(
        step, action,
        workspace_root="/tmp/ws", current_branch="main",
        mission_allowlist=[r"git push --force origin main"],
    )
    assert isinstance(decision, Block)
