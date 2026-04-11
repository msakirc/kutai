# security/permissions.py
"""
Phase 8.1 — Agent Permission Matrix

Enforced per-agent-type allowed tools. Before running any tool, the base
agent checks AGENT_PERMISSIONS[agent_type]. None = all tools allowed.
"""
from __future__ import annotations
from typing import Optional

from src.infra.logging_config import get_logger

logger = get_logger("security.permissions")

# None means "all tools allowed" for that agent type.
# An explicit set means only those tools are permitted.
AGENT_PERMISSIONS: dict[str, Optional[set[str]]] = {
    # Planning agents — read-only, no shell execution
    "planner": {
        "file_tree", "read_file", "read_blackboard", "write_blackboard",
        "git_log", "list_workspace", "project_info", "web_search", "smart_search",
    },
    "architect": {
        "file_tree", "read_file", "read_blackboard", "write_blackboard",
        "git_log", "list_workspace", "web_search", "smart_search",
    },
    # Coding agents — file manipulation + shell
    "coder": {
        "file_tree", "read_file", "write_file", "edit_file", "patch_file",
        "apply_diff", "shell", "shell_stdin", "shell_sequential",
        "run_code", "lint", "verify_deps", "scaffold", "recommend_stack",
        "git_log", "git_commit", "git_branch", "git_init", "git_rollback",
        "read_blackboard", "write_blackboard", "list_workspace",
    },
    "implementer": None,  # full access
    "fixer": {
        "file_tree", "read_file", "write_file", "edit_file", "patch_file",
        "apply_diff", "shell", "shell_stdin", "shell_sequential",
        "run_code", "lint", "verify_deps",
        "git_log", "read_blackboard", "write_blackboard",
    },
    "test_generator": {
        "file_tree", "read_file", "write_file", "edit_file",
        "shell", "run_code", "lint",
        "git_log", "read_blackboard", "list_workspace",
    },
    # Review agents — read-only + lint
    "reviewer": {
        "file_tree", "read_file", "shell", "lint", "run_code",
        "git_log", "read_blackboard", "list_workspace",
        "index_workspace", "query_codebase", "codebase_map",
        "project_info", "write_file",
    },
    "visual_reviewer": {
        "file_tree", "read_file", "analyze_image",
        "read_blackboard",
    },
    # Research agents — web + read
    "researcher": {
        "web_search", "smart_search", "web_extract", "file_tree", "read_file",
        "read_blackboard",
    },
    "analyst": {
        "web_search", "smart_search", "web_extract", "file_tree", "read_file", "write_file",
        "read_blackboard", "write_blackboard",
        "analyze_image", "read_document",
    },
    # Writing agents — file write only
    "writer": {
        "file_tree", "read_file", "write_file", "edit_file",
        "read_blackboard", "web_search", "smart_search",
    },
    "summarizer": {
        "file_tree", "read_file", "read_blackboard",
    },
    # General agents
    "assistant": None,  # full access
    "executor": None,   # full access
}


def check_permission(agent_type: str, tool_name: str) -> bool:
    """
    Return True if agent_type is permitted to use tool_name.

    Always permits if:
    - The agent has None (full access)
    - The agent type is unknown (fail-open for flexibility)
    """
    allowed = AGENT_PERMISSIONS.get(agent_type)
    if allowed is None:
        return True  # full access or unknown agent
    return tool_name in allowed


def get_allowed_tools(agent_type: str) -> Optional[set[str]]:
    """Return the allowed tool set for an agent type, or None for full access."""
    return AGENT_PERMISSIONS.get(agent_type)
