"""Salako — mechanical dispatcher: non-LLM task executors."""
from __future__ import annotations

from salako.workspace_snapshot import snapshot_workspace
from salako.git_commit import auto_commit

__all__ = ["Action", "run", "snapshot_workspace", "auto_commit"]
