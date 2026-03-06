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

        if mode == "append":
            # Append doesn't need atomic — just write directly
            with open(full_path, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            # Phase 9: Atomic write via temp file + os.replace()
            # Write to a sibling temp file, then atomically move
            import tempfile
            fd, tmp_path = tempfile.mkstemp(
                dir=parent_abs, prefix=".tmp_write_", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp_path, full_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        size = os.path.getsize(full_path)
        action = "Appended to" if mode == "append" else "Wrote"
        res = f"✅ {action} {filepath} ({_human_size(size)})"

        # Auto-lint Python files
        if full_path.endswith(".py"):
            from tools.linting import auto_lint
            lint_res = await auto_lint(filepath)
            res += f"\n\n--- Auto-Linting ---\n{lint_res}"

        return res

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


# ---------------------------------------------------------------------------
# Phase 6: Per-goal workspace isolation
# ---------------------------------------------------------------------------

def get_goal_workspace(goal_id: int) -> str:
    """Return the workspace directory for a specific goal.

    Creates the directory if it doesn't exist.
    Structure: workspace/goal_{id}/
    """
    goal_dir = os.path.join(WORKSPACE_DIR, f"goal_{goal_id}")
    os.makedirs(goal_dir, exist_ok=True)
    return goal_dir


def get_goal_workspace_relative(goal_id: int) -> str:
    """Return the relative path (from workspace root) for a goal workspace."""
    return f"goal_{goal_id}"


def list_goal_workspaces() -> list[dict]:
    """List all goal workspaces with basic info."""
    result = []
    if not os.path.isdir(WORKSPACE_DIR):
        return result
    for entry in os.scandir(WORKSPACE_DIR):
        if entry.is_dir() and entry.name.startswith("goal_"):
            try:
                gid = int(entry.name.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            # Count files
            file_count = sum(
                len(files) for _, _, files in os.walk(entry.path)
            )
            result.append({
                "goal_id": gid,
                "path": entry.path,
                "file_count": file_count,
            })
    return result


def cleanup_goal_workspace(goal_id: int) -> bool:
    """Remove a goal's workspace directory (after goal completion).

    Returns True if cleaned up, False if not found.
    """
    import shutil
    goal_dir = os.path.join(WORKSPACE_DIR, f"goal_{goal_id}")
    if os.path.isdir(goal_dir):
        shutil.rmtree(goal_dir, ignore_errors=True)
        return True
    return False


# ---------------------------------------------------------------------------
# Phase 6: Workspace snapshots (file hashing)
# ---------------------------------------------------------------------------

def compute_workspace_hashes(workspace_path: str) -> dict[str, str]:
    """Compute SHA-256 hashes of all files in a workspace directory.

    Returns {relative_path: hash} dict.
    """
    import hashlib
    hashes: dict[str, str] = {}
    workspace_real = os.path.realpath(workspace_path)

    if not os.path.isdir(workspace_real):
        return hashes

    for dirpath, dirnames, filenames in os.walk(workspace_real):
        # Skip hidden/generated dirs
        dirnames[:] = [d for d in dirnames if not _should_skip(d, True)]
        for fname in filenames:
            if _should_skip(fname, False):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, workspace_real)
            try:
                h = hashlib.sha256()
                with open(full, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                hashes[rel] = h.hexdigest()[:16]
            except (OSError, IOError):
                pass
    return hashes


def diff_snapshots(
    before: dict[str, str], after: dict[str, str],
) -> dict[str, list[str]]:
    """Compare two file-hash snapshots.

    Returns {"added": [...], "modified": [...], "deleted": [...]}.
    """
    before_files = set(before.keys())
    after_files = set(after.keys())

    added = sorted(after_files - before_files)
    deleted = sorted(before_files - after_files)
    modified = sorted(
        f for f in before_files & after_files
        if before[f] != after[f]
    )
    return {"added": added, "modified": modified, "deleted": deleted}


# ---------------------------------------------------------------------------
# Phase 6: Multi-project support
# ---------------------------------------------------------------------------

def load_projects_config() -> list[dict]:
    """Load project configurations from projects.json.

    Returns list of project dicts:
        [{"name": "...", "path": "...", "language": "...", "conventions": "..."}]
    """
    import json
    from config import PROJECTS_CONFIG_PATH

    if not os.path.isfile(PROJECTS_CONFIG_PATH):
        return []
    try:
        with open(PROJECTS_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "projects" in data:
            return data["projects"]
    except (json.JSONDecodeError, IOError):
        pass
    return []


def get_project(name: str) -> dict | None:
    """Look up a project by name."""
    projects = load_projects_config()
    for p in projects:
        if p.get("name", "").lower() == name.lower():
            return p
    return None
