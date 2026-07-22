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
