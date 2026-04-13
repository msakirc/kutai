"""OS-specific helpers for process tree detection."""
from __future__ import annotations

import os


def get_process_tree_pids() -> set[int]:
    """Get PIDs belonging to our process tree (self + children + python parent)."""
    pids = {os.getpid()}
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        for child in proc.children(recursive=True):
            pids.add(child.pid)
        parent = proc.parent()
        if parent and "python" in (parent.name() or "").lower():
            pids.add(parent.pid)
    except Exception:
        pass
    return pids
