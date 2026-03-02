# tools/deps.py
"""
Dependency verification — parse Python imports from source files,
check against installed packages, auto-install missing ones.
"""

import ast
import os
import logging
import re
from tools.shell import run_shell
from tools.workspace import _safe_resolve, WORKSPACE_DIR

logger = logging.getLogger(__name__)

# Modules from the Python stdlib (3.10+) — no need to install these.
# This is a subset; we check pip as the authoritative source.
_STDLIB_MODULES: set[str] = {
    "abc", "aifc", "argparse", "array", "ast", "asyncio", "atexit",
    "base64", "binascii", "bisect", "builtins", "calendar", "cgi",
    "cmd", "code", "codecs", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy",
    "copyreg", "cProfile", "csv", "ctypes", "curses", "dataclasses",
    "datetime", "dbm", "decimal", "difflib", "dis", "distutils",
    "email", "encodings", "enum", "errno", "faulthandler", "fcntl",
    "filecmp", "fileinput", "fnmatch", "fractions", "ftplib",
    "functools", "gc", "getopt", "getpass", "gettext", "glob",
    "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "imaplib", "importlib", "inspect", "io", "ipaddress", "itertools",
    "json", "keyword", "lib2to3", "linecache", "locale", "logging",
    "lzma", "mailbox", "math", "mimetypes", "mmap", "multiprocessing",
    "netrc", "numbers", "operator", "optparse", "os", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posixpath", "pprint", "profile", "pstats",
    "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue",
    "quopri", "random", "re", "readline", "reprlib", "resource",
    "rlcompleter", "runpy", "sched", "secrets", "select", "selectors",
    "shelve", "shlex", "shutil", "signal", "site", "smtplib",
    "socket", "socketserver", "sqlite3", "ssl", "stat", "statistics",
    "string", "struct", "subprocess", "sunau", "symtable", "sys",
    "sysconfig", "syslog", "tabnanny", "tarfile", "tempfile", "termios",
    "test", "textwrap", "threading", "time", "timeit", "tkinter",
    "token", "tokenize", "tomllib", "trace", "traceback", "tracemalloc",
    "tty", "turtle", "types", "typing", "unicodedata", "unittest",
    "urllib", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml",
    "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
    # Common underscore-prefixed stdlib
    "_thread", "__future__",
}

# Common PyPI name mappings (import name → pip package name)
_IMPORT_TO_PIP: dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "gi": "PyGObject",
    "attr": "attrs",
    "serial": "pyserial",
    "usb": "pyusb",
    "magic": "python-magic",
    "jwt": "PyJWT",
    "lxml": "lxml",
}


def _extract_imports(source: str) -> set[str]:
    """
    Extract top-level import module names from Python source code.

    Returns a set of top-level module names (e.g. 'flask', 'requests').
    """
    modules: set[str] = set()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback: regex-based extraction
        for match in re.finditer(
            r"^\s*(?:import|from)\s+(\w+)", source, re.MULTILINE
        ):
            modules.add(match.group(1))
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])

    return modules


def _get_pip_name(module_name: str) -> str:
    """Map an import name to a pip package name."""
    return _IMPORT_TO_PIP.get(module_name, module_name)


async def verify_dependencies(path: str = "") -> str:
    """
    Scan Python files in the workspace, extract imports, check which
    packages are missing, and auto-install them via pip.

    Args:
        path: Subdirectory relative to workspace root (default: root).

    Returns:
        Summary of what was found and installed.
    """
    root = _safe_resolve(path) if path else os.path.realpath(WORKSPACE_DIR)
    if root is None:
        return "❌ Access denied: path is outside workspace."
    if not os.path.isdir(root):
        return f"❌ Directory not found: {path or 'workspace root'}"

    # 1. Collect all imports from .py files
    all_imports: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip common non-source directories
        dirnames[:] = [
            d for d in dirnames
            if d not in {
                "__pycache__", ".git", "node_modules", ".venv",
                "venv", "env", ".tox", ".eggs",
            }
        ]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", errors="replace") as f:
                    source = f.read()
                all_imports |= _extract_imports(source)
            except Exception as e:
                logger.debug(f"Could not parse {fpath}: {e}")

    if not all_imports:
        return "ℹ️ No Python imports found in workspace."

    # 2. Filter out stdlib and local modules
    third_party = set()
    for mod in all_imports:
        if mod in _STDLIB_MODULES:
            continue
        # Skip local project modules (files that exist in workspace)
        local_path = os.path.join(root, mod)
        if os.path.isdir(local_path) or os.path.isfile(local_path + ".py"):
            continue
        third_party.add(mod)

    if not third_party:
        return "✅ All imports are stdlib or local — no packages to install."

    # 3. Check which are installed in the sandbox
    pip_names = {mod: _get_pip_name(mod) for mod in third_party}
    check_cmd = "pip list --format=columns 2>/dev/null | tail -n +3 | awk '{print $1}'"
    try:
        installed_output = await run_shell(check_cmd, timeout=30)
        installed = {
            pkg.strip().lower()
            for pkg in installed_output.split("\n")
            if pkg.strip()
        }
    except Exception:
        installed = set()

    missing = []
    for mod, pip_name in pip_names.items():
        # Normalize for comparison (pip uses dashes, imports use underscores)
        normalized = pip_name.lower().replace("-", "_").replace(".", "_")
        pip_normalized = pip_name.lower().replace("_", "-")
        if normalized not in installed and pip_normalized not in installed:
            missing.append(pip_name)

    if not missing:
        return (
            f"✅ All {len(third_party)} third-party packages are already "
            f"installed: {', '.join(sorted(pip_names.values()))}"
        )

    # 4. Auto-install missing packages
    to_install = " ".join(missing)
    logger.info(f"📦 Auto-installing missing packages: {to_install}")

    try:
        install_output = await run_shell(
            f"pip install {to_install} 2>&1",
            timeout=120,
        )
    except Exception as e:
        return f"❌ Failed to install packages: {e}"

    # Check for errors in output
    if "error" in install_output.lower() and "successfully" not in install_output.lower():
        return (
            f"⚠️ Installation had errors:\n{install_output}\n\n"
            f"Attempted to install: {to_install}"
        )

    return (
        f"✅ Installed {len(missing)} missing package(s): {', '.join(missing)}\n"
        f"{install_output[-500:]}"
    )
