"""Named-mutex single-instance authority for the hub.

The Win32 named mutex is the CWD/session-independent authority for "one hub,
never two" — immune to the historical CWD-relative-file-lock bug, kernel
auto-released on process death. See the design doc:
docs/superpowers/specs/2026-07-17-yasar-usta-always-live-singleton-design.md (§4.1).

`decide_singleton` is a pure decision tree over an injectable CreateMutexW seam
(so it is testable off-Windows). `record_fault` is the pure cross-process
circuit-breaker that keeps a permanent fault from producing a per-minute alert
storm. The acting glue (exit codes + Telegram alert) lives in the caller.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# ── Win32 error codes ────────────────────────────────────────────────────────
ERROR_ALREADY_EXISTS = 183
ERROR_ACCESS_DENIED = 5

# ── Outcomes ─────────────────────────────────────────────────────────────────
OWNED = "owned"                    # we created and hold the mutex → run
ALREADY_RUNNING = "already_running"  # another hub owns it → exit(0)
ERROR = "error"                    # ambiguous failure → fail closed (never "free")

# Keeps the OS handle referenced for the process lifetime (kernel releases it on
# death). Mirrors lock.py's module-global handle pattern. One hub → one call.
_held_handle = None


def _win32_create_mutex(qualified_name: str):
    """Real seam: CreateMutexW → (handle_or_None, GetLastError).

    On ERROR_ALREADY_EXISTS Windows still returns a valid handle (to the
    existing object); the caller distinguishes via the error code.
    """
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateMutexW.restype = wintypes.HANDLE
    k.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    handle = k.CreateMutexW(None, False, qualified_name)
    err = ctypes.get_last_error()
    return (handle or None, err)


def decide_singleton(name: str, create_mutex=_win32_create_mutex):
    """Try to become the single instance.

    Prefers the machine-wide ``Global\\`` namespace; on ACCESS_DENIED (the
    launch token lacks SeCreateGlobalPrivilege) falls back to ``Local\\``, which
    needs no privilege and dedups within the single interactive session.

    Returns ``(outcome, handle, namespace)``. A NULL handle is NEVER read as
    "lock free" — any non-ALREADY_EXISTS creation failure is ``ERROR`` so the
    caller fails closed rather than risk a second instance.
    """
    global _held_handle
    for ns in ("Global\\", "Local\\"):
        handle, err = create_mutex(ns + name)
        if not handle:                       # NULL → creation failed
            if err == ERROR_ACCESS_DENIED and ns == "Global\\":
                continue                     # no Global privilege → try Local
            return ERROR, None, ns
        if err == ERROR_ALREADY_EXISTS:      # handle to an existing object
            return ALREADY_RUNNING, None, ns
        if err == 0:                         # we created & own it
            _held_handle = handle
            return OWNED, handle, ns
        return ERROR, None, ns               # handle but unexpected error
    return ERROR, None, None


@dataclass
class FaultDecision:
    should_alert: bool
    count: int
    give_up: bool


def record_fault(
    marker_path,
    signature: str,
    now: float,
    dedup_seconds: int = 3600,
    give_up_after: int = 5,
) -> FaultDecision:
    """Cross-process circuit-breaker for fail-closed mutex faults.

    A small JSON marker survives process relaunches so an identical, permanent
    fault alerts at most once per ``dedup_seconds`` and, after
    ``give_up_after`` consecutive occurrences, signals ``give_up`` (caller then
    exits 0 to stop the Task-Scheduler retry hammer). A different signature or
    a missing/corrupt marker resets the counter.
    """
    marker_path = Path(marker_path)
    try:
        data = json.loads(marker_path.read_text())
    except Exception:
        data = {}

    same = data.get("signature") == signature
    if same:
        count = int(data.get("count", 0)) + 1
        last_alert = float(data.get("last_alert_ts", 0.0))
        should_alert = (now - last_alert) >= dedup_seconds
    else:
        count = 1
        should_alert = True

    if should_alert:
        new_last_alert = now
    elif same:
        new_last_alert = float(data.get("last_alert_ts", now))
    else:
        new_last_alert = now

    out = {
        "signature": signature,
        "count": count,
        "first_ts": data.get("first_ts", now) if same else now,
        "last_alert_ts": new_last_alert,
    }
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(out))
    except Exception:
        pass

    return FaultDecision(should_alert=should_alert, count=count, give_up=count >= give_up_after)


def _fault_marker(state_dir) -> Path:
    """Marker path in state_dir, falling back to the temp dir if state_dir is
    unusable (e.g. the fault IS an unwritable state dir — avoid a chicken-egg)."""
    try:
        p = Path(state_dir) / ".mutex_fault.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return Path(tempfile.gettempdir()) / "yasar_mutex_fault.json"


def enforce_singleton(
    name: str,
    *,
    state_dir,
    create_mutex=_win32_create_mutex,
    alert=None,
    exit_fn=sys.exit,
    now_fn=time.time,
):
    """Become the single hub or exit — the acting glue over ``decide_singleton``.

    - OWNED → return (caller proceeds; handle held for process lifetime).
    - ALREADY_RUNNING → ``exit_fn(0)`` (another hub owns it).
    - ERROR → fail CLOSED: never proceed. Circuit-breaker (``record_fault``)
      alerts at most once/hour and, after K consecutive faults, ``exit_fn(0)``
      to stop the Layer-0 retry hammer; otherwise ``exit_fn(3)`` so Layer 0
      retries a transient fault.
    """
    outcome, _handle, ns = decide_singleton(name, create_mutex=create_mutex)
    if outcome == OWNED:
        return
    if outcome == ALREADY_RUNNING:
        exit_fn(0)
        return

    d = record_fault(_fault_marker(state_dir), signature="mutex_error", now=now_fn())
    if d.give_up:
        if alert:
            alert(f"🛑 Yaşar Usta: giving up after {d.count} mutex faults ({ns}) — human needed")
        exit_fn(0)
        return
    if d.should_alert and alert:
        alert(f"⚠️ Yaşar Usta: mutex acquisition failed ({ns}) — retrying")
    exit_fn(3)
