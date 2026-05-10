"""Z10-T3B — semantic argv guard.

The regex-based ``BLOCKED_PATTERNS`` set in :mod:`src.tools.shell` is
brittle: ``["dd", "if=/dev/sda", ...]`` joins to ``"dd if=/dev/sda"``
which only matches because ``"dd if="`` is in the set. List-style
invocations from automated callers (mr_roboto/run_cmd) miss the regex
entirely. The semantic guard is the additive belt-and-suspenders.
"""
from __future__ import annotations

from src.tools import shell


def test_dd_argv_blocked() -> None:
    blocked, reason = shell._is_blocked_argv(["dd", "if=/dev/sda", "of=/tmp/x"])
    assert blocked is True
    # Reason should mention either dd or the /dev/sd path.
    assert "dd" in reason or "/dev/sd" in reason


def test_echo_hi_not_blocked() -> None:
    blocked, reason = shell._is_blocked_argv(["echo", "hi"])
    assert blocked is False
    assert reason == ""


def test_ls_nvme_blocked() -> None:
    blocked, reason = shell._is_blocked_argv(["ls", "/dev/nvme0n1"])
    assert blocked is True
    assert "/dev/nvme" in reason


def test_nc_listen_blocked() -> None:
    blocked, reason = shell._is_blocked_argv(["nc", "-l", "8080"])
    assert blocked is True
    assert "nc" in reason


def test_mkfs_variant_blocked() -> None:
    blocked, reason = shell._is_blocked_argv(["mkfs.ext4", "/dev/sda1"])
    assert blocked is True


def test_socat_listen_blocked() -> None:
    blocked, reason = shell._is_blocked_argv(["socat", "TCP-LISTEN:9000", "EXEC:/bin/sh"])
    assert blocked is True
    assert "socat" in reason


def test_ip_link_blocked() -> None:
    blocked, reason = shell._is_blocked_argv(["ip", "link", "set", "eth0", "down"])
    assert blocked is True


def test_ip_addr_not_blocked() -> None:
    # ``ip addr`` (read-only) should NOT be blocked — only link/route.
    blocked, _ = shell._is_blocked_argv(["ip", "addr"])
    assert blocked is False


def test_sysrq_path_blocked() -> None:
    blocked, _ = shell._is_blocked_argv(["echo", "1", ">", "/proc/sysrq-trigger"])
    # echo ... /proc/sysrq-trigger should hit the path scan.
    assert blocked is True


def test_semantic_guard_runs_on_string() -> None:
    blocked, reason = shell._semantic_guard("dd if=/dev/sda of=/tmp/x")
    assert blocked is True


def test_semantic_guard_pipeline_segment_blocked() -> None:
    """A blocked command after ``;`` or ``&&`` still trips."""
    blocked, _ = shell._semantic_guard("ls -la && dd if=/dev/sda of=/tmp/x")
    assert blocked is True


def test_semantic_guard_clean_command() -> None:
    blocked, _ = shell._semantic_guard("python3 -c 'print(1)'")
    assert blocked is False
