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
from tools.edit_file import edit_file
from tools.linting import auto_lint
from tools.deps import verify_dependencies
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
        "example": '{"action": "tool_call", "tool": "web_search", "args": {"query": "FastAPI websocket tutorial"}}',
    }
except Exception as e:
    logger.warning(f"web_search tool not available — {type(e).__name__}: {e}")

try:
    from tools.code_runner import run_code

    _optional_tools["run_code"] = {
        "function": run_code,
        "description": (
            "Run a code snippet directly. "
            "Args: code (str), language (str, optional, default 'python')"
        ),
        "example": '{"action": "tool_call", "tool": "run_code", "args": {"code": "print(2+2)"}}',
    }
except ImportError as e:
    logger.warning(f"run_code tool not available — {type(e).__name__}: {e}")

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
        "example": '{"action": "tool_call", "tool": "shell", "args": {"command": "python3 main.py"}}',
    },
    "shell_stdin": {
        "function": run_shell_with_stdin,
        "description": (
            "Execute a shell command and pipe data to its stdin. "
            "Args: command (str), stdin_data (str), "
            "timeout (int, optional), workdir (str, optional)"
        ),
        "example": (
            '{"action": "tool_call", "tool": "shell_stdin", '
            '"args": {"command": "cat > hello.txt", "stdin_data": "Hello world"}}'
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
            '{"action": "tool_call", "tool": "shell_sequential", '
            '"args": {"commands": ["pip install -r requirements.txt", "python main.py"]}}'
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
        "example": '{"action": "tool_call", "tool": "file_tree", "args": {"path": "."}}',
    },
    "read_file": {
        "function": read_file,
        "description": (
            "Read a file with line numbers. "
            "Args: filepath (str), max_lines (int, optional, default 200)"
        ),
        "example": '{"action": "tool_call", "tool": "read_file", "args": {"filepath": "src/main.py"}}',
    },
    "write_file": {
        "function": write_file,
        "description": (
            "Create or update a file. "
            'Args: filepath (str), content (str), '
            'mode ("write" or "append", default "write")'
        ),
        "example": '{"action": "tool_call", "tool": "write_file", "args": {"filepath": "src/main.py", "content": "print(\'hello\')"}}',
    },
    "project_info": {
        "function": detect_project,
        "description": (
            "Analyze the workspace to detect languages, frameworks, "
            "dependencies, and file statistics. "
            "Args: path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "project_info", "args": {}}',
    },
    "edit_file": {
        "function": edit_file,
        "description": (
            "Replace a range of lines in a file. start_line and end_line are 1-indexed and inclusive. "
            "Args: filepath (str), start_line (int), end_line (int), new_content (str)"
        ),
        "example": '{"action": "tool_call", "tool": "edit_file", "args": {"filepath": "src/main.py", "start_line": 10, "end_line": 15, "new_content": "def foo():\\n    pass\\n"}}',
    },
    "lint": {
        "function": auto_lint,
        "description": (
            "Auto-lint and format a Python file using ruff. "
            "Args: filepath (str)"
        ),
        "example": '{"action": "tool_call", "tool": "lint", "args": {"filepath": "src/main.py"}}',
    },
    "verify_deps": {
        "function": verify_dependencies,
        "description": (
            "Scan Python files, extract imports, and auto-install any missing "
            "third-party packages via pip. "
            "Args: path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "verify_deps", "args": {"path": "."}}',
    },

    # ── Git ─────────────────────────────────────────────────────────────────
    "git_init": {
        "function": git_init,
        "description": (
            "Initialize a git repo (with .gitignore and initial commit). "
            "Idempotent — safe to call on an existing repo. "
            "Args: path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "git_init", "args": {}}',
    },
    "git_commit": {
        "function": git_commit,
        "description": (
            "Stage all changes and commit. "
            "Args: message (str), path (str, optional), "
            "add_all (bool, optional, default true)"
        ),
        "example": '{"action": "tool_call", "tool": "git_commit", "args": {"message": "feat: add login endpoint"}}',
    },
    "git_branch": {
        "function": git_branch,
        "description": (
            "Create and switch to a branch (or switch if it already exists). "
            "Args: branch_name (str), path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "git_branch", "args": {"branch_name": "feat/auth"}}',
    },
    "git_log": {
        "function": git_log,
        "description": (
            "Show recent commits (one-line format). "
            "Args: path (str, optional), count (int, optional, default 10)"
        ),
        "example": '{"action": "tool_call", "tool": "git_log", "args": {"count": 5}}',
    },
    "git_diff": {
        "function": git_diff,
        "description": (
            "Show uncommitted changes. "
            "Args: path (str, optional), stat_only (bool, optional, default false)"
        ),
        "example": '{"action": "tool_call", "tool": "git_diff", "args": {}}',
    },
    "git_rollback": {
        "function": git_rollback,
        "description": (
            "Soft-reset the last N commits (keeps files staged). "
            "Args: steps (int, default 1), path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "git_rollback", "args": {"steps": 1}}',
    },
    "git_status": {
        "function": git_status,
        "description": (
            "Show current branch and working-tree status. "
            "Args: path (str, optional)"
        ),
        "example": '{"action": "tool_call", "tool": "git_status", "args": {}}',
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
logger.info(f"📦 Loaded tools: {sorted(TOOL_REGISTRY.keys())}")

# ---------------------------------------------------------------------------
# LiteLLM Tool Schemas (auto-generated from TOOL_REGISTRY)
# ---------------------------------------------------------------------------
_PYTHON_TYPE_MAP = {
    int: "integer", float: "number", bool: "boolean",
    str: "string", list: "array", dict: "object",
}

TOOL_SCHEMAS: list[dict] = []

for _name, _info in TOOL_REGISTRY.items():
    try:
        _sig = inspect.signature(_info["function"])
        _properties: dict = {}
        _required: list[str] = []
        for _pname, _param in _sig.parameters.items():
            _ptype = "string"  # safe default
            _annotation = _param.annotation
            if _annotation is not inspect.Parameter.empty:
                _ptype = _PYTHON_TYPE_MAP.get(_annotation, "string")
            _properties[_pname] = {
                "type": _ptype,
                "description": f"Parameter: {_pname}",
            }
            if _param.default is inspect.Parameter.empty:
                _required.append(_pname)
        TOOL_SCHEMAS.append({
            "type": "function",
            "function": {
                "name": _name,
                "description": _info["description"],
                "parameters": {
                    "type": "object",
                    "properties": _properties,
                    "required": _required,
                },
            },
        })
    except (ValueError, TypeError):
        pass

# Pseudo-tools for structured agent actions
TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": (
            "Provide your final answer to the task. Use this when you are "
            "done and ready to submit your complete result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "Your complete answer / result for the task.",
                },
                "memories": {
                    "type": "object",
                    "description": "Optional key-value pairs to remember for future tasks.",
                },
            },
            "required": ["result"],
        },
    },
})

TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "clarify",
        "description": (
            "Ask the user a clarifying question when you need more information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarification question to ask.",
                },
            },
            "required": ["question"],
        },
    },
})

# Clean up
del _name, _info, _sig, _properties, _required, _pname, _param, _ptype, _annotation

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
