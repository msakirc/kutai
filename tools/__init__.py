# tools/__init__.py
"""
Tool registry — every tool an agent can invoke.

Each entry bundles the async callable, a human-readable description, and
a JSON usage example.  The full catalogue can be injected into agent
prompts via ``get_tool_descriptions()``.
"""

import inspect
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports — must match the public API of each module we built
# ---------------------------------------------------------------------------
from tools.shell import run_shell, run_shell_with_stdin, run_shell_sequential
from tools.workspace import get_file_tree, read_file, write_file, detect_project
from tools.git_ops import (
    git_init,
    git_commit,
    git_branch,
    git_log,
    git_diff,
    git_rollback,
    git_status,
)

# Optional / pre-existing tools — degrade gracefully if absent
_optional_tools: dict[str, dict[str, Any]] = {}

try:
    from tools.web_search import web_search

    _optional_tools["web_search"] = {
        "function": web_search,
        "description": "Search the web. Args: query (str)",
        "example": '{"tool": "web_search", "query": "FastAPI websocket tutorial"}',
    }
except ImportError:
    logger.info("web_search tool not available — skipping")

try:
    from tools.code_runner import run_code

    _optional_tools["run_code"] = {
        "function": run_code,
        "description": (
            "Run a code snippet directly. "
            "Args: code (str), language (str, optional, default 'python')"
        ),
        "example": '{"tool": "run_code", "code": "print(2+2)"}',
    }
except ImportError:
    logger.info("run_code tool not available — skipping")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    # ── Shell ──────────────────────────────────────────────────────────────
    "shell": {
        "function": run_shell,
        "description": (
            "Execute a shell command in the Docker sandbox. "
            "Args: command (str), timeout (int, optional, default 60), "
            "workdir (str, optional)"
        ),
        "example": '{"tool": "shell", "command": "python3 main.py", "timeout": 30}',
    },
    "shell_stdin": {
        "function": run_shell_with_stdin,
        "description": (
            "Execute a shell command and pipe data to its stdin. "
            "Args: command (str), stdin_data (str), "
            "timeout (int, optional), workdir (str, optional)"
        ),
        "example": (
            '{"tool": "shell_stdin", '
            '"command": "cat > hello.txt", "stdin_data": "Hello world"}'
        ),
    },
    "shell_sequential": {
        "function": run_shell_sequential,
        "description": (
            "Run multiple commands in order, stopping on first failure. "
            "Args: commands (list[str]), timeout (int, optional, per-command), "
            "workdir (str, optional), stop_on_error (bool, default true)"
        ),
        "example": (
            '{"tool": "shell_sequential", '
            '"commands": ["pip install -r requirements.txt", "python main.py"]}'
        ),
    },

    # ── File Operations ────────────────────────────────────────────────────
    "file_tree": {
        "function": get_file_tree,
        "description": (
            "List files and directories as a visual tree. "
            "Args: path (str, optional), max_depth (int, optional, default 5), "
            "max_items (int, optional, default 200)"
        ),
        "example": '{"tool": "file_tree", "path": "my-project"}',
    },
    "read_file": {
        "function": read_file,
        "description": (
            "Read a file with line numbers. "
            "Args: filepath (str), max_lines (int, optional, default 200)"
        ),
        "example": '{"tool": "read_file", "filepath": "src/main.py"}',
    },
    "write_file": {
        "function": write_file,
        "description": (
            "Create or update a file. "
            'Args: filepath (str), content (str), '
            'mode ("write" or "append", default "write")'
        ),
        "example": '{"tool": "write_file", "filepath": "src/main.py", "content": "print(\'hello\')"}',
    },
    "project_info": {
        "function": detect_project,
        "description": (
            "Analyze the workspace to detect languages, frameworks, "
            "dependencies, and file statistics. "
            "Args: path (str, optional)"
        ),
        "example": '{"tool": "project_info"}',
    },

    # ── Git ─────────────────────────────────────────────────────────────────
    "git_init": {
        "function": git_init,
        "description": (
            "Initialize a git repo (with .gitignore and initial commit). "
            "Idempotent — safe to call on an existing repo. "
            "Args: path (str, optional)"
        ),
        "example": '{"tool": "git_init"}',
    },
    "git_commit": {
        "function": git_commit,
        "description": (
            "Stage all changes and commit. "
            "Args: message (str), path (str, optional), "
            "add_all (bool, optional, default true)"
        ),
        "example": '{"tool": "git_commit", "message": "feat: add login endpoint"}',
    },
    "git_branch": {
        "function": git_branch,
        "description": (
            "Create and switch to a branch (or switch if it already exists). "
            "Args: branch_name (str), path (str, optional)"
        ),
        "example": '{"tool": "git_branch", "branch_name": "feat/auth"}',
    },
    "git_log": {
        "function": git_log,
        "description": (
            "Show recent commits (one-line format). "
            "Args: path (str, optional), count (int, optional, default 10)"
        ),
        "example": '{"tool": "git_log", "count": 5}',
    },
    "git_diff": {
        "function": git_diff,
        "description": (
            "Show uncommitted changes. "
            "Args: path (str, optional), stat_only (bool, optional, default false)"
        ),
        "example": '{"tool": "git_diff"}',
    },
    "git_rollback": {
        "function": git_rollback,
        "description": (
            "Soft-reset the last N commits (keeps files staged). "
            "Args: steps (int, default 1), path (str, optional)"
        ),
        "example": '{"tool": "git_rollback", "steps": 1}',
    },
    "git_status": {
        "function": git_status,
        "description": (
            "Show current branch and working-tree status. "
            "Args: path (str, optional)"
        ),
        "example": '{"tool": "git_status"}',
    },

    # ── Optional tools injected below ──────────────────────────────────────
    **_optional_tools,
}


# ---------------------------------------------------------------------------
# Pre-compute accepted parameter names per tool (once at import time)
# ---------------------------------------------------------------------------
_TOOL_PARAMS: dict[str, Optional[set[str]]] = {}

for _name, _info in TOOL_REGISTRY.items():
    try:
        _sig = inspect.signature(_info["function"])
        _TOOL_PARAMS[_name] = set(_sig.parameters.keys())
    except (ValueError, TypeError):
        # If introspection fails, don't filter — let Python raise naturally
        _TOOL_PARAMS[_name] = None

# Clean up module namespace
del _name, _info, _sig


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def get_tool_descriptions() -> str:
    """
    Format every tool's description and example into a prompt-friendly
    string suitable for injection into an agent's system message.
    """
    lines: list[str] = []
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"• **{name}**: {info['description']}")
        lines.append(f"  Example: {info['example']}")
    return "\n".join(lines)


def list_tool_names() -> list[str]:
    """Return a sorted list of all registered tool names."""
    return sorted(TOOL_REGISTRY.keys())


async def execute_tool(tool_name: str, **kwargs: Any) -> str:
    """
    Execute a registered tool by name.

    Unknown keyword arguments are silently dropped so that extra fields
    the LLM may include (e.g. ``"tool"``, ``"thought"``) don't crash
    the underlying function.

    Args:
        tool_name: Key in TOOL_REGISTRY.
        **kwargs:  Arguments forwarded to the tool function.

    Returns:
        The tool's string output, or a descriptive error message.
    """
    if tool_name not in TOOL_REGISTRY:
        available = ", ".join(sorted(TOOL_REGISTRY.keys()))
        return f"❌ Unknown tool: '{tool_name}'. Available: {available}"

    func = TOOL_REGISTRY[tool_name]["function"]

    # ── Filter kwargs to accepted parameters ──
    valid_params = _TOOL_PARAMS.get(tool_name)
    if valid_params is not None:
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    else:
        # Introspection failed at import — pass everything through
        filtered = kwargs

    logger.debug(f"Executing tool '{tool_name}' with args: {list(filtered.keys())}")

    try:
        result = await func(**filtered)
        return str(result)

    except TypeError as exc:
        # Almost always a missing required argument
        expected = (
            ", ".join(sorted(valid_params)) if valid_params else "(unknown)"
        )
        logger.error(f"Tool '{tool_name}' argument error: {exc}", exc_info=True)
        return (
            f"❌ Argument error ({tool_name}): {exc}\n"
            f"Expected parameters: {expected}"
        )

    except Exception as exc:
        logger.error(f"Tool '{tool_name}' error: {exc}", exc_info=True)
        return f"❌ Tool error ({tool_name}): {type(exc).__name__}: {exc}"
