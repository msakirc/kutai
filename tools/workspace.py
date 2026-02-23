# tools/workspace.py
"""
Give agents awareness of the current project workspace:
file tree, file contents, project structure detection.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKSPACE_DIR: str = os.environ.get(
    "WORKSPACE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace")),
)

MAX_TREE_DEPTH = 5
MAX_TREE_ITEMS = 200
MAX_READ_LINES = 200
MAX_READ_SIZE = 500_000        # 500 KB
MAX_DEPENDENCY_PREVIEW = 2000  # chars to show from dependency files

# ---------------------------------------------------------------------------
# Ignore rules
# ---------------------------------------------------------------------------
SKIP_DIRS: set[str] = {
    "__pycache__", ".git", "node_modules", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".next", ".nuxt",
    "vendor", ".tox", ".eggs",
}

SKIP_FILES: set[str] = {
    ".DS_Store", "Thumbs.db",
}

SKIP_EXTENSIONS: set[str] = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib",
}

SKIP_SUFFIXES: list[str] = [
    ".egg-info",
]


def _should_skip(name: str, is_dir: bool) -> bool:
    """Return True if this entry should be hidden from listings."""
    if is_dir:
        if name in SKIP_DIRS:
            return True
        return any(name.endswith(s) for s in SKIP_SUFFIXES)
    # File
    if name in SKIP_FILES:
        return True
    _, ext = os.path.splitext(name)
    if ext.lower() in SKIP_EXTENSIONS:
        return True
    return any(name.endswith(s) for s in SKIP_SUFFIXES)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
def _safe_resolve(relative_path: str) -> Optional[str]:
    """
    Resolve *relative_path* under WORKSPACE_DIR.
    Returns the absolute path, or None if it escapes the workspace.
    """
    # Normalise first so that `..` tricks are caught
    joined = os.path.normpath(os.path.join(WORKSPACE_DIR, relative_path))
    real = os.path.realpath(joined)
    workspace_real = os.path.realpath(WORKSPACE_DIR)
    # Must be equal to or nested under workspace
    if real == workspace_real or real.startswith(workspace_real + os.sep):
        return real
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _human_size(size_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes}B" if unit == "B" else f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


# ---------------------------------------------------------------------------
# File tree
# ---------------------------------------------------------------------------
async def get_file_tree(
    path: str = "",
    max_depth: int = MAX_TREE_DEPTH,
    max_items: int = MAX_TREE_ITEMS,
) -> str:
    """
    Return an ASCII file tree of the workspace (or a subdirectory).

    Args:
        path:      Subdirectory relative to workspace root.
        max_depth: How deep to recurse.
        max_items: Maximum entries to show.

    Returns:
        A string with the visual tree representation.
    """
    root = _safe_resolve(path) if path else os.path.realpath(WORKSPACE_DIR)
    if root is None:
        return "❌ Access denied: path is outside workspace."
    if not os.path.isdir(root):
        return f"❌ Directory not found: {path or 'workspace root'}"

    lines: list[str] = []
    item_count = 0

    def _walk(dir_path: str, prefix: str, depth: int) -> None:
        nonlocal item_count
        if depth > max_depth or item_count > max_items:
            return

        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            lines.append(f"{prefix}├── [permission denied]")
            return

        dirs = [
            e for e in entries
            if os.path.isdir(os.path.join(dir_path, e)) and not _should_skip(e, True)
        ]
        files = [
            e for e in entries
            if os.path.isfile(os.path.join(dir_path, e)) and not _should_skip(e, False)
        ]

        # Interleave: dirs first, then files (both with tree connectors)
        all_entries: list[tuple[str, bool]] = [
            (d, True) for d in dirs
        ] + [
            (f, False) for f in files
        ]

        for i, (name, is_dir) in enumerate(all_entries):
            item_count += 1
            if item_count > max_items:
                lines.append(f"{prefix}├── … (truncated, >{max_items} items)")
                return

            is_last = i == len(all_entries) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            full = os.path.join(dir_path, name)

            if is_dir:
                lines.append(f"{prefix}{connector}📁 {name}/")
                _walk(full, prefix + extension, depth + 1)
            else:
                try:
                    size_str = _human_size(os.path.getsize(full))
                except OSError:
                    size_str = "?"
                lines.append(f"{prefix}{connector}{name}  ({size_str})")

    display_name = path or "workspace"
    lines.append(f"📂 {display_name}/")
    _walk(root, "  ", 1)

    if item_count == 0:
        lines.append("  (empty)")

    lines.append(f"\n({item_count} items shown)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Read file
# ---------------------------------------------------------------------------
async def read_file(filepath: str, max_lines: int = MAX_READ_LINES) -> str:
    """
    Read a file from the workspace.

    Args:
        filepath:  Path relative to workspace root.
        max_lines: Maximum number of lines to return.

    Returns:
        File contents with line numbers.
    """
    full_path = _safe_resolve(filepath)
    if full_path is None:
        return "❌ Access denied: path is outside workspace."

    if not os.path.isfile(full_path):
        return f"❌ File not found: {filepath}"

    try:
        size = os.path.getsize(full_path)
        if size > MAX_READ_SIZE:
            return (
                f"❌ File too large ({_human_size(size)}). "
                "Use shell to inspect with head/tail/less."
            )

        with open(full_path, "r", errors="replace") as f:
            all_lines = f.readlines()

        total = len(all_lines)
        shown = all_lines[:max_lines]
        numbered = "".join(f"{i + 1:4d} | {line}" for i, line in enumerate(shown))

        if total > max_lines:
            numbered += f"\n\n… [{total - max_lines} more lines, {total} total]"

        return (
            f"📄 {filepath} ({total} lines, {_human_size(size)}):\n\n{numbered}"
        )

    except Exception as exc:
        logger.error(f"Error reading {filepath}: {exc}", exc_info=True)
        return f"❌ Error reading {filepath}: {exc}"


# ---------------------------------------------------------------------------
# Write file
# ---------------------------------------------------------------------------
async def write_file(
    filepath: str,
    content: str,
    mode: str = "write",
) -> str:
    """
    Write content to a file in the workspace.

    Args:
        filepath: Path relative to workspace root.
        content:  Content to write.
        mode:     "write" (overwrite) or "append".

    Returns:
        Confirmation message.
    """
    # We need the parent directory to exist, so resolve it separately.
    # _safe_resolve may fail if the file doesn't exist yet, so we resolve
    # the *parent* and ensure it's inside the workspace.
    parent_rel = os.path.dirname(filepath)
    # If filepath is just a filename, parent_rel is "".
    parent_abs = _safe_resolve(parent_rel) if parent_rel else os.path.realpath(WORKSPACE_DIR)

    if parent_abs is None:
        return "❌ Access denied: path is outside workspace."

    # Now build the full path from the validated parent
    full_path = os.path.join(parent_abs, os.path.basename(filepath))

    # Double-check the final path is still inside workspace
    final_real = os.path.realpath(full_path)
    workspace_real = os.path.realpath(WORKSPACE_DIR)
    if not (final_real == workspace_real or final_real.startswith(workspace_real + os.sep)):
        return "❌ Access denied: resolved path is outside workspace."

    try:
        os.makedirs(parent_abs, exist_ok=True)

        file_mode = "a" if mode == "append" else "w"
        with open(full_path, file_mode) as f:
            f.write(content)

        size = os.path.getsize(full_path)
        action = "Appended to" if mode == "append" else "Wrote"
        return f"✅ {action} {filepath} ({_human_size(size)})"

    except Exception as exc:
        logger.error(f"Error writing {filepath}: {exc}", exc_info=True)
        return f"❌ Error writing {filepath}: {exc}"


# ---------------------------------------------------------------------------
# Project detection
# ---------------------------------------------------------------------------

# Maps filename → (human description, category)
_PROJECT_INDICATORS: dict[str, tuple[str, str]] = {
    "package.json":        ("Node.js / JavaScript",  "dependencies"),
    "requirements.txt":    ("Python (pip)",           "dependencies"),
    "pyproject.toml":      ("Python (modern)",        "build config"),
    "Pipfile":             ("Python (pipenv)",        "dependencies"),
    "setup.py":            ("Python package",         "packaging"),
    "Cargo.toml":          ("Rust",                   "dependencies"),
    "go.mod":              ("Go",                     "module"),
    "pom.xml":             ("Java / Maven",           "build"),
    "build.gradle":        ("Java / Kotlin (Gradle)", "build"),
    "Dockerfile":          ("Docker",                 "container"),
    "docker-compose.yml":  ("Docker Compose",         "multi-container"),
    "docker-compose.yaml": ("Docker Compose",         "multi-container"),
    ".github/workflows":   ("GitHub Actions CI",      "ci"),
    "Makefile":            ("Make build system",      "build"),
    ".env":                ("Environment config",     "config"),
    "README.md":           ("Has README",             "docs"),
    "index.html":          ("Static HTML",            "entry"),
    "app.py":              ("Python app (Flask?)",    "entry"),
    "main.py":             ("Python entry point",     "entry"),
    "manage.py":           ("Django",                 "entry"),
    "tsconfig.json":       ("TypeScript",             "config"),
    "next.config.js":      ("Next.js",               "config"),
    "next.config.mjs":     ("Next.js",               "config"),
    "vite.config.js":      ("Vite",                   "config"),
    "vite.config.ts":      ("Vite",                   "config"),
}

# Files whose content is useful for the agent to see
_DEPENDENCY_FILES: set[str] = {
    "requirements.txt", "package.json", "pyproject.toml",
    "Cargo.toml", "go.mod",
}


async def detect_project(path: str = "") -> str:
    """
    Analyze the workspace (or a subdirectory) and return a structured
    summary: detected technologies, file-type breakdown, and previews
    of key dependency files.

    Args:
        path: Subdirectory relative to workspace root.

    Returns:
        Human-readable project summary.
    """
    root = _safe_resolve(path) if path else os.path.realpath(WORKSPACE_DIR)
    if root is None:
        return "❌ Access denied: path is outside workspace."
    if not os.path.isdir(root):
        return f"❌ Directory not found: {path or 'workspace root'}"

    detected: list[str] = []
    dep_previews: dict[str, str] = {}

    for filename, (description, _category) in _PROJECT_INDICATORS.items():
        full = os.path.join(root, filename)
        if os.path.exists(full):
            detected.append(f"  • {description}  ({filename})")

            # Read dependency/config files for agent context
            if filename in _DEPENDENCY_FILES and os.path.isfile(full):
                try:
                    with open(full, "r", errors="replace") as f:
                        preview = f.read(MAX_DEPENDENCY_PREVIEW)
                    dep_previews[filename] = preview
                except Exception:
                    pass

    # Count file extensions
    ext_counts: dict[str, int] = {}
    total_files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip(d, True)]
        for fname in filenames:
            if _should_skip(fname, False):
                continue
            total_files += 1
            ext = os.path.splitext(fname)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    top_ext = sorted(ext_counts.items(), key=lambda x: -x[1])[:10]

    # ---- Build output ----
    display = path or "workspace root"
    parts: list[str] = [f"📊 Project Analysis: {display}\n"]
    parts.append(f"Total files: {total_files}\n")

    if detected:
        parts.append("Detected technologies:")
        parts.extend(detected)
    else:
        parts.append("No recognised project structure detected.")

    if top_ext:
        parts.append("\nFile types:")
        for ext, count in top_ext:
            parts.append(f"  {ext or '(no ext)':12s}: {count} files")

    for filename, content in dep_previews.items():
        # Truncate long previews
        preview = content[:1000]
        if len(content) > 1000:
            preview += "\n  … (truncated)"
        parts.append(f"\n📎 {filename}:\n```\n{preview}\n```")

    return "\n".join(parts)
