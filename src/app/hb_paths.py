"""Single source of truth for where the orchestrator writes its heartbeat +
state snapshot. Under the Yaşar Usta hub, YASAR_USTA_STATE_DIR is authoritative
so hub (reader) and orchestrator (writer) never disagree. Falls back to the
legacy relative path for a non-hub launch."""
import os


def _state_dir() -> str | None:
    """Yaşar Usta-supplied absolute state dir, or None for a non-hub launch."""
    return os.environ.get("YASAR_USTA_STATE_DIR") or None


def heartbeat_paths() -> tuple:
    sd = _state_dir()
    if sd:
        return (os.path.join(sd, "orchestrator.heartbeat"),
                os.path.join(sd, "heartbeat"))
    return ("logs/orchestrator.heartbeat", "logs/heartbeat")


def state_snapshot_path() -> str:
    sd = _state_dir()
    if sd:
        return os.path.join(sd, "orchestrator.state.json")
    return "logs/orchestrator.state.json"


def shutdown_signal_paths() -> tuple:
    """Ordered read candidates for the hub's shutdown.signal, authoritative first.

    The Yaşar Usta hub writes the signal to ``${log_dir}/shutdown.signal``
    (supervisor.py); the path migration flips the target's registry ``log_dir``
    from ``${project_root}/logs`` to ``${state_dir}/logs`` so it leaves Dropbox.
    Hence, under the hub, the signal lands in the ``logs`` subdir *below*
    state_dir — one level deeper than the heartbeat (which sits at the state_dir
    root). We keep the legacy CWD-relative ``logs/shutdown.signal`` as a
    transition fallback so this read-side may land BEFORE the hub flips its
    registry (no ordering hazard, and a non-hub launch keeps working)."""
    sd = _state_dir()
    legacy = os.path.join("logs", "shutdown.signal")
    if sd:
        return (os.path.join(sd, "logs", "shutdown.signal"), legacy)
    return (legacy,)
