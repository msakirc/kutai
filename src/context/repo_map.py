# context/repo_map.py
"""
Phase 12.4 — Repository Map

Auto-generates a structural map of a repository:
  - Module dependency graph (who imports whom)
  - Entry points (main.py, app.py, index.js, etc.)
  - Test mapping (which test files test which modules)
  - Config files and their roles
  - Directory purpose summary

Public API:
    repo_map = generate_repo_map(root_path)
    text     = format_repo_map(repo_map)
"""
import json
import logging
import os
import re
from typing import Optional

from ..parsing.tree_sitter_parser import (
    detect_language,
    get_parseable_extensions,
    parse_file,
)

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info", ".next", ".nuxt", "target", "out",
    "bin", "obj", ".gradle", ".idea", ".vscode", "coverage",
    "chroma_data", "workspace",
}

ENTRY_POINT_NAMES = {
    "main.py", "app.py", "server.py", "index.py", "run.py", "manage.py",
    "index.js", "index.ts", "app.js", "app.ts", "server.js", "server.ts",
    "main.js", "main.ts", "main.go", "main.rs", "Main.java", "App.java",
    "main.c", "main.cpp",
}

CONFIG_FILES = {
    "package.json": "Node.js project configuration",
    "tsconfig.json": "TypeScript compiler configuration",
    "requirements.txt": "Python dependencies",
    "setup.py": "Python package setup",
    "pyproject.toml": "Python project configuration",
    "Cargo.toml": "Rust project configuration",
    "go.mod": "Go module configuration",
    "build.gradle": "Gradle build configuration",
    "build.gradle.kts": "Kotlin Gradle build configuration",
    "pom.xml": "Maven build configuration",
    "Makefile": "Build automation",
    "CMakeLists.txt": "CMake build configuration",
    "Dockerfile": "Docker container definition",
    "docker-compose.yml": "Docker Compose services",
    ".env": "Environment variables",
    ".env.example": "Environment variable template",
    ".gitignore": "Git ignore rules",
    "jest.config.js": "Jest test configuration",
    "pytest.ini": "pytest configuration",
    "setup.cfg": "Python tool configuration",
    "tox.ini": "Tox test configuration",
    ".eslintrc.js": "ESLint configuration",
    ".eslintrc.json": "ESLint configuration",
    "ruff.toml": "Ruff linter configuration",
}

TEST_PATTERNS = [
    re.compile(r"test_(\w+)\.py$"),        # Python: test_module.py
    re.compile(r"(\w+)_test\.py$"),         # Python: module_test.py
    re.compile(r"(\w+)\.test\.[jt]sx?$"),   # JS/TS: module.test.js
    re.compile(r"(\w+)\.spec\.[jt]sx?$"),   # JS/TS: module.spec.js
    re.compile(r"(\w+)_test\.go$"),         # Go: module_test.go
    re.compile(r"(\w+)Test\.java$"),        # Java: ModuleTest.java
]

TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs"}


# ─── Generate Repo Map ──────────────────────────────────────────────────────

def generate_repo_map(root_path: str) -> dict:
    """
    Generate a comprehensive repository map.

    Scans the directory, parses source files, and builds:
      - dependency_graph: {file: [imported_modules]}
      - entry_points: [files that are entry points]
      - test_mapping: {test_file: [tested_modules]}
      - config_files: {file: role_description}
      - directory_summary: {dir: {files, purpose}}
      - languages: {lang: file_count}

    Args:
        root_path: Root directory to scan.

    Returns:
        Dict with the full repo map structure.
    """
    root = os.path.normpath(root_path)
    extensions = get_parseable_extensions()

    dependency_graph: dict[str, list[str]] = {}
    entry_points: list[str] = []
    test_mapping: dict[str, list[str]] = {}
    config_found: dict[str, str] = {}
    dir_summary: dict[str, dict] = {}
    language_counts: dict[str, int] = {}
    all_files: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden/cache directories
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""

        dir_files: list[str] = []
        dir_languages: set[str] = set()

        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root).replace("\\", "/")

            # Config files
            if fname in CONFIG_FILES:
                config_found[rel_path] = CONFIG_FILES[fname]

            # Check if it's a parseable source file
            ext = os.path.splitext(fname)[1].lower()
            if ext not in extensions:
                continue

            lang = detect_language(fname)
            if lang == "unknown":
                continue

            all_files.append(rel_path)
            dir_files.append(fname)
            dir_languages.add(lang)
            language_counts[lang] = language_counts.get(lang, 0) + 1

            # Entry points
            if fname in ENTRY_POINT_NAMES:
                entry_points.append(rel_path)

            # Test files
            is_test = False
            tested_modules: list[str] = []

            # Check if in test directory
            dir_parts = rel_dir.split("/") if rel_dir else []
            if any(d in TEST_DIRS for d in dir_parts):
                is_test = True

            # Check filename patterns
            for pattern in TEST_PATTERNS:
                m = pattern.search(fname)
                if m:
                    is_test = True
                    tested_modules.append(m.group(1))

            if is_test:
                test_mapping[rel_path] = tested_modules

            # Parse for dependency graph
            try:
                result = parse_file(full_path)
                imported = []
                for imp in result.get("imports", []):
                    text = imp.get("text", "")
                    # Extract module name from import text
                    if "from " in text:
                        parts = text.split("from ")
                        if len(parts) > 1:
                            mod = parts[1].split(" import")[0].strip()
                            imported.append(mod)
                    elif "import " in text:
                        mod = text.replace("import ", "").strip().split(" as ")[0]
                        imported.append(mod)
                    elif "#include" in text:
                        imported.append(
                            text.replace("#include", "").strip().strip('<>"')
                        )
                    elif "use " in text:
                        imported.append(
                            text.replace("use ", "").strip().rstrip(";")
                        )

                if imported:
                    dependency_graph[rel_path] = imported

            except Exception:
                pass

        # Directory summary
        if dir_files:
            purpose = _infer_dir_purpose(rel_dir, dir_files, dir_languages)
            dir_summary[rel_dir or "."] = {
                "file_count": len(dir_files),
                "languages": sorted(dir_languages),
                "purpose": purpose,
            }

    return {
        "root": root,
        "dependency_graph": dependency_graph,
        "entry_points": entry_points,
        "test_mapping": test_mapping,
        "config_files": config_found,
        "directory_summary": dir_summary,
        "languages": language_counts,
        "total_files": len(all_files),
    }


def _infer_dir_purpose(dir_path: str, files: list[str], languages: set[str]) -> str:
    """Infer the purpose of a directory from its contents."""
    name = dir_path.split("/")[-1] if dir_path else "root"
    name_lower = name.lower()

    purpose_map = {
        "src": "Source code",
        "lib": "Library modules",
        "pkg": "Packages",
        "cmd": "CLI commands",
        "api": "API layer",
        "routes": "URL routes/handlers",
        "controllers": "Request controllers",
        "models": "Data models",
        "views": "View templates/components",
        "components": "UI components",
        "pages": "Page components",
        "utils": "Utility functions",
        "helpers": "Helper functions",
        "services": "Service layer",
        "middleware": "Request middleware",
        "config": "Configuration",
        "scripts": "Build/utility scripts",
        "tools": "Developer tools",
        "agents": "Agent implementations",
        "memory": "Memory system",
        "parsing": "Code parsing",
        "context": "Context assembly",
        "security": "Security modules",
        "migrations": "Database migrations",
        "fixtures": "Test fixtures",
        "static": "Static assets",
        "public": "Public assets",
        "assets": "Project assets",
        "docs": "Documentation",
    }

    if name_lower in TEST_DIRS:
        return "Test suite"

    if name_lower in purpose_map:
        return purpose_map[name_lower]

    # Infer from file contents
    if any(f.startswith("test_") or f.endswith("_test.py") for f in files):
        return "Tests"
    if any(f.endswith((".html", ".jinja2", ".ejs", ".hbs")) for f in files):
        return "Templates"
    if any(f.endswith((".css", ".scss", ".less")) for f in files):
        return "Styles"

    lang_str = "/".join(sorted(languages))
    return f"{lang_str} modules" if languages else "Mixed files"


# ─── Format for Prompt Injection ─────────────────────────────────────────────

def format_repo_map(repo_map: dict, max_lines: int = 80) -> str:
    """
    Format a repo map into a compact text block for agent prompts.

    Args:
        repo_map: Dict from generate_repo_map().
        max_lines: Maximum lines in output.

    Returns:
        Formatted text block.
    """
    lines: list[str] = []

    # Languages summary
    langs = repo_map.get("languages", {})
    if langs:
        lang_parts = [f"{lang}: {count}" for lang, count in
                      sorted(langs.items(), key=lambda x: -x[1])]
        lines.append(f"Languages: {', '.join(lang_parts)}")
        lines.append(f"Total files: {repo_map.get('total_files', 0)}")
        lines.append("")

    # Entry points
    entry_points = repo_map.get("entry_points", [])
    if entry_points:
        lines.append("Entry points: " + ", ".join(entry_points))
        lines.append("")

    # Config files
    configs = repo_map.get("config_files", {})
    if configs:
        lines.append("Config files:")
        for path, role in sorted(configs.items())[:10]:
            lines.append(f"  {path} — {role}")
        lines.append("")

    # Directory structure
    dir_summary = repo_map.get("directory_summary", {})
    if dir_summary:
        lines.append("Directory structure:")
        for dir_path, info in sorted(dir_summary.items())[:20]:
            purpose = info.get("purpose", "")
            count = info.get("file_count", 0)
            lines.append(f"  {dir_path}/ ({count} files) — {purpose}")
        lines.append("")

    # Key dependencies (top imports)
    dep_graph = repo_map.get("dependency_graph", {})
    if dep_graph:
        # Count import frequency
        import_freq: dict[str, int] = {}
        for deps in dep_graph.values():
            for dep in deps:
                import_freq[dep] = import_freq.get(dep, 0) + 1

        top_deps = sorted(import_freq.items(), key=lambda x: -x[1])[:15]
        if top_deps:
            lines.append("Most imported modules:")
            for mod, count in top_deps:
                lines.append(f"  {mod} ({count}x)")
            lines.append("")

    # Test mapping summary
    tests = repo_map.get("test_mapping", {})
    if tests:
        lines.append(f"Test files: {len(tests)}")
        for test_file, modules in sorted(tests.items())[:10]:
            if modules:
                lines.append(f"  {test_file} → tests {', '.join(modules)}")
            else:
                lines.append(f"  {test_file}")

    # Trim to max_lines
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1]
        lines.append("... (truncated)")

    return "\n".join(lines)


def save_repo_map(repo_map: dict, output_path: str) -> None:
    """Save repo map as JSON for persistence."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(repo_map, f, indent=2, default=str)


def load_repo_map(input_path: str) -> Optional[dict]:
    """Load repo map from JSON."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
