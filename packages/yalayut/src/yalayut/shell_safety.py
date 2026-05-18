"""Windows-safe shell tokenizer + bin allowlist for yalayut recipe execution.

Recipe ``invocation.steps[].cmd`` strings are split into argv lists here. No
``shell=True`` is ever used, so there is no shell-injection surface. The first
token of every command is checked against a static allowlist and the whole
string is screened for Windows-incompatible patterns documented in the yalayut
recon (chmod / sudo / apt / brew / bare .sh / symlink / $HOME / /dev/null).
"""
from __future__ import annotations

import re
import shlex

# First-token allowlist. Recipe scaffolders only — no network fetch tools,
# no destructive tools. Mirrors the yalayut_policy shell_allowlist seed.
SHELL_BIN_ALLOWLIST = frozenset(
    {
        "uvx",
        "npx",
        "npm",
        "pip",
        "pip3",
        "git",
        "cookiecutter",
        "python",
        "python3",
        "poetry",
        "pnpm",
        "yarn",
    }
)

# Windows-incompat patterns. Maps a compiled regex to the reason string.
_WIN_INCOMPAT: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bchmod\b"), "chmod"),
    (re.compile(r"\bsudo\b"), "sudo"),
    (re.compile(r"\bapt(-get)?\b"), "apt"),
    (re.compile(r"\bbrew\b"), "brew"),
    (re.compile(r"\byum\b"), "yum"),
    (re.compile(r"\bln\s+-s\b"), "symlink"),
    (re.compile(r"/dev/null"), "dev_null"),
    (re.compile(r"\$HOME\b"), "home_var"),
    # A bare .sh invocation (./foo.sh or `bash foo.sh` with no .ps1 sibling).
    (re.compile(r"(^|\s)(\./|bash\s+)\S*\.sh(\s|$)"), "bare_sh"),
]


class ShellSafetyError(ValueError):
    """Raised when a recipe command cannot be safely tokenized."""


def tokenize_cmd(cmd: str) -> list[str]:
    """Split a recipe command string into an argv list (no shell).

    Uses ``shlex`` in POSIX mode — quoting works identically on Windows since
    we never hand the string to ``cmd.exe``. Raises ``ShellSafetyError`` on an
    empty/whitespace-only command.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ShellSafetyError("empty command")
    try:
        argv = shlex.split(cmd, posix=True)
    except ValueError as e:
        raise ShellSafetyError(f"unparseable command: {e}") from e
    if not argv:
        raise ShellSafetyError("empty command after tokenize")
    return argv


def check_shell_bin(binary: str) -> bool:
    """Return True iff ``binary`` (the first argv token) is allowlisted."""
    # Strip any path component and a trailing .exe so `python.exe` matches.
    base = binary.replace("\\", "/").rsplit("/", 1)[-1]
    if base.lower().endswith(".exe"):
        base = base[:-4]
    return base.lower() in SHELL_BIN_ALLOWLIST


def windows_incompat_reason(cmd: str) -> str | None:
    """Return a reason string if ``cmd`` contains a Windows-incompat pattern.

    Returns ``None`` when the command is Windows-safe.
    """
    if not isinstance(cmd, str):
        return "not_a_string"
    for pattern, reason in _WIN_INCOMPAT:
        if pattern.search(cmd):
            return reason
    return None
