# context/assembler.py
"""
Phase 12.3 — Intelligent Context Assembly

Given a task description, assembles the most relevant code context:
  1. Embed the description and query code embeddings
  2. Pull matched symbols with imports and dependents
  3. Include related test files
  4. Include recent git changes
  5. Fit within a configurable token budget

Public API:
    context_block = await assemble_context(task, project_root, max_tokens)
"""
import os
import subprocess
from typing import Optional

from src.infra.logging_config import get_logger
from ..parsing.tree_sitter_parser import parse_file, detect_language

logger = get_logger("context.assembler")


# ─── Token Estimation ────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for code."""
    return len(text) // 4


# ─── Code Embedding Query ───────────────────────────────────────────────────

async def _query_code_embeddings(
    query_text: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Query the codebase vector collection for relevant code symbols.

    Returns list of dicts with filepath, symbol_name, etc.
    Falls back to empty list if vector store isn't available.
    """
    try:
        from ..memory.vector_store import query, is_ready
        if not is_ready():
            return []

        results = await query(
            text=query_text,
            collection="codebase",
            top_k=top_k,
        )
        return results

    except Exception as e:
        logger.debug(f"Code embedding query failed: {e}")
        return []


# ─── File Reading ────────────────────────────────────────────────────────────

def _read_file_section(
    filepath: str,
    line_start: int,
    line_end: int,
    context_lines: int = 5,
) -> str:
    """Read a specific section of a file with surrounding context."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""

    # Expand range with context
    start = max(0, line_start - 1 - context_lines)
    end = min(len(lines), line_end + context_lines)

    selected = lines[start:end]
    # Add line numbers
    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i:4d} | {line.rstrip()}")

    return "\n".join(numbered)


def _read_file_brief(filepath: str, max_lines: int = 50) -> str:
    """Read first N lines of a file for a brief overview."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[:max_lines]
    except OSError:
        return ""

    if not lines:
        return ""

    numbered = []
    for i, line in enumerate(lines, 1):
        numbered.append(f"{i:4d} | {line.rstrip()}")

    result = "\n".join(numbered)
    if len(lines) == max_lines:
        result += "\n  ... (truncated)"
    return result


# ─── Git Integration ─────────────────────────────────────────────────────────

def _get_recent_changes(
    root_path: str,
    filepaths: list[str],
    max_diff_lines: int = 50,
) -> str:
    """Get recent git changes for specific files."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-5", "--", *filepaths],
            cwd=root_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""

        commits = result.stdout.strip()
        if not commits:
            return ""

        return f"Recent changes:\n{commits}"

    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


# ─── Test File Discovery ────────────────────────────────────────────────────

def _find_related_tests(
    root_path: str,
    source_files: list[str],
) -> list[str]:
    """
    Find test files that likely test the given source files.

    Heuristics:
      - test_<module>.py for <module>.py
      - <module>.test.js for <module>.js
      - Same directory or parallel tests/ directory
    """
    test_files: list[str] = []
    test_dirs = {"test", "tests", "__tests__", "spec"}

    for src_path in source_files:
        basename = os.path.basename(src_path)
        name_no_ext = os.path.splitext(basename)[0]
        src_dir = os.path.dirname(src_path)

        # Python patterns
        candidates = [
            os.path.join(src_dir, f"test_{name_no_ext}.py"),
            os.path.join(src_dir, f"{name_no_ext}_test.py"),
            os.path.join(src_dir, "tests", f"test_{name_no_ext}.py"),
            os.path.join("tests", f"test_{name_no_ext}.py"),
            # JS/TS patterns
            os.path.join(src_dir, f"{name_no_ext}.test.js"),
            os.path.join(src_dir, f"{name_no_ext}.test.ts"),
            os.path.join(src_dir, f"{name_no_ext}.spec.js"),
            os.path.join(src_dir, f"{name_no_ext}.spec.ts"),
            os.path.join(src_dir, "__tests__", f"{name_no_ext}.test.js"),
            os.path.join(src_dir, "__tests__", f"{name_no_ext}.test.ts"),
        ]

        for candidate in candidates:
            full = os.path.join(root_path, candidate)
            if os.path.isfile(full) and candidate not in test_files:
                test_files.append(candidate)

    return test_files


# ─── Import Resolution ──────────────────────────────────────────────────────

def _resolve_imports(
    root_path: str,
    imports: list[dict],
    language: str,
) -> list[str]:
    """
    Resolve import statements to actual file paths within the project.

    Returns list of relative file paths that are imported.
    """
    resolved: list[str] = []

    for imp in imports:
        text = imp.get("text", "")

        if language == "python":
            # Extract module from "from X import Y" or "import X"
            module = ""
            if "from " in text:
                parts = text.split("from ")
                if len(parts) > 1:
                    module = parts[1].split(" import")[0].strip()
            elif "import " in text:
                module = text.replace("import ", "").strip().split(" as ")[0]

            if module:
                # Convert module path to file path
                path = module.replace(".", "/")
                candidates = [
                    f"{path}.py",
                    f"{path}/__init__.py",
                ]
                for c in candidates:
                    full = os.path.join(root_path, c)
                    if os.path.isfile(full):
                        resolved.append(c)
                        break

        elif language in ("javascript", "typescript"):
            # Extract from "import ... from 'X'" or require('X')
            import re
            match = re.search(r'["\']([./][^"\']+)["\']', text)
            if match:
                rel = match.group(1)
                # Try with various extensions
                exts = [".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"]
                if not any(rel.endswith(e) for e in [".js", ".ts", ".jsx", ".tsx"]):
                    for ext in exts:
                        full = os.path.join(root_path, rel + ext)
                        if os.path.isfile(full):
                            resolved.append(rel + ext)
                            break
                else:
                    full = os.path.join(root_path, rel)
                    if os.path.isfile(full):
                        resolved.append(rel)

    return resolved


# ─── Main Assembly Function ─────────────────────────────────────────────────

async def assemble_context(
    task: dict,
    project_root: str,
    max_tokens: int = 4000,
) -> str:
    """
    Assemble relevant code context for a task.

    Pipeline:
      1. Query code embeddings for relevant symbols
      2. Parse matched files for structure
      3. Pull imports and dependents
      4. Find related test files
      5. Include recent git changes
      6. Format within token budget

    Args:
        task:         Task dict with title, description.
        project_root: Root path of the project.
        max_tokens:   Token budget for the context block.

    Returns:
        Formatted code context string, or empty string.
    """
    title = task.get("title", "")
    description = task.get("description", "")
    query_text = f"{title}: {description[:500]}"

    if not query_text.strip() or not os.path.isdir(project_root):
        return ""

    sections: list[str] = []
    total_tokens = 0
    matched_files: set[str] = set()

    # ── 1. Query code embeddings ──
    embedding_results = await _query_code_embeddings(query_text, top_k=8)

    if embedding_results:
        code_section_lines = ["### Relevant Code"]

        for result in embedding_results:
            if total_tokens >= max_tokens:
                break

            meta = result.get("metadata", {})
            filepath = meta.get("filepath", "")
            symbol_name = meta.get("symbol_name", "")
            symbol_type = meta.get("symbol_type", "")
            line_start = meta.get("line_start", 1)
            line_end = meta.get("line_end", line_start + 10)
            language = meta.get("language", "")

            if not filepath:
                continue

            full_path = os.path.join(project_root, filepath)
            if not os.path.isfile(full_path):
                continue

            matched_files.add(filepath)

            # Read the relevant section
            section = _read_file_section(
                full_path, int(line_start), int(line_end),
            )
            if not section:
                continue

            section_tokens = _estimate_tokens(section)
            if total_tokens + section_tokens > max_tokens * 0.6:
                # Don't let code snippets take more than 60% of budget
                break

            code_section_lines.append(
                f"\n**{symbol_type} `{symbol_name}`** "
                f"in `{filepath}` (L{line_start}-{line_end}):"
            )
            code_section_lines.append(f"```{language}\n{section}\n```")
            total_tokens += section_tokens

        if len(code_section_lines) > 1:
            sections.append("\n".join(code_section_lines))

    # ── 2. Resolve imports of matched files ──
    imported_files: list[str] = []
    for filepath in list(matched_files)[:5]:
        full_path = os.path.join(project_root, filepath)
        try:
            result = parse_file(full_path)
            language = result.get("language", "")
            resolved = _resolve_imports(
                project_root, result.get("imports", []), language,
            )
            for imp_file in resolved:
                if imp_file not in matched_files:
                    imported_files.append(imp_file)
        except Exception:
            pass

    if imported_files and total_tokens < max_tokens * 0.8:
        dep_lines = ["### Dependencies"]
        for imp_file in imported_files[:5]:
            full_path = os.path.join(project_root, imp_file)
            brief = _read_file_brief(full_path, max_lines=15)
            if brief:
                tokens = _estimate_tokens(brief)
                if total_tokens + tokens > max_tokens * 0.85:
                    break
                dep_lines.append(f"\n`{imp_file}` (imported):")
                dep_lines.append(f"```\n{brief}\n```")
                total_tokens += tokens
                matched_files.add(imp_file)

        if len(dep_lines) > 1:
            sections.append("\n".join(dep_lines))

    # ── 3. Related test files ──
    test_files = _find_related_tests(project_root, list(matched_files))
    if test_files and total_tokens < max_tokens * 0.9:
        test_lines = ["### Related Tests"]
        for test_file in test_files[:3]:
            full_path = os.path.join(project_root, test_file)
            brief = _read_file_brief(full_path, max_lines=20)
            if brief:
                tokens = _estimate_tokens(brief)
                if total_tokens + tokens > max_tokens * 0.95:
                    break
                test_lines.append(f"\n`{test_file}`:")
                test_lines.append(f"```\n{brief}\n```")
                total_tokens += tokens

        if len(test_lines) > 1:
            sections.append("\n".join(test_lines))

    # ── 4. Recent git changes ──
    if matched_files and total_tokens < max_tokens:
        changes = _get_recent_changes(project_root, list(matched_files)[:10])
        if changes:
            change_tokens = _estimate_tokens(changes)
            if total_tokens + change_tokens <= max_tokens:
                sections.append(f"### {changes}")
                total_tokens += change_tokens

    if not sections:
        return ""

    return "## Code Context\n\n" + "\n\n".join(sections)


# ─── Ambient Context Assembly ──────────────────────────────────────────────────

async def assemble_ambient_context(
    goal_id: int | None = None,
    max_tokens: int = 400,
) -> str:
    """
    Assemble a short ambient context block injected into every agent execution.

    Includes: time of day, system load mode, active goals count, and recent
    blackboard decisions if goal_id is provided. Stays under max_tokens.
    """
    import datetime
    parts: list[str] = []

    # Time of day
    now = datetime.datetime.now()
    hour = now.hour
    if hour < 6:
        tod = "night"
    elif hour < 12:
        tod = "morning"
    elif hour < 17:
        tod = "afternoon"
    else:
        tod = "evening"
    parts.append(f"- Time: {now.strftime('%Y-%m-%d %H:%M')} ({tod})")

    # System load mode
    try:
        from ..infra.load_manager import get_load_mode
        mode = get_load_mode()
        parts.append(f"- System load mode: {mode}")
    except Exception:
        pass

    # Active goals
    try:
        from ..infra.db import get_db
        db = await get_db()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM goals WHERE status IN ('pending','running','in_progress')"
        )
        row = await cursor.fetchone()
        if row:
            count = row[0]
            parts.append(f"- Active goals: {count}")
    except Exception:
        pass

    # Recent blackboard decisions
    if goal_id:
        try:
            from ..collaboration.blackboard import read_blackboard
            decisions = await read_blackboard(goal_id, key="decisions")
            if decisions and isinstance(decisions, list) and len(decisions) > 0:
                recent = decisions[-2:]  # last 2
                dec_strs = []
                for d in recent:
                    if isinstance(d, dict):
                        dec_strs.append(d.get("what", str(d))[:80])
                    else:
                        dec_strs.append(str(d)[:80])
                if dec_strs:
                    parts.append("- Recent decisions: " + "; ".join(dec_strs))
        except Exception:
            pass

    if not parts:
        return ""

    text = "## Current Context\n" + "\n".join(parts)
    # Rough token trim
    if len(text) > max_tokens * 4:
        text = text[:max_tokens * 4]
    return text


# ─── Language-Aware Prompt Injection (Phase 10.2) ──────────────────────────────

async def get_language_hints(
    task: dict,
    project_root: str | None = None,
) -> str:
    """
    Detect project language and return language-specific prompt hints.

    Used by agents to get idiomatic patterns, test commands, etc.
    Returns empty string if language can't be detected.
    """
    try:
        import os
        from ..languages import get_toolkit, detect_language

        # Try to detect from project files
        language = None

        # First check task context for language hint
        task_ctx = task.get("context", {})
        if isinstance(task_ctx, str):
            import json
            try:
                task_ctx = json.loads(task_ctx)
            except Exception:
                task_ctx = {}
        if isinstance(task_ctx, dict):
            language = task_ctx.get("language")

        # Then try to detect from project_root file extensions
        if not language and project_root and os.path.isdir(project_root):
            extensions = set()
            for fname in os.listdir(project_root):
                _, ext = os.path.splitext(fname)
                if ext:
                    extensions.add(ext.lower())
            # Also check src/ subdirectory
            src_dir = os.path.join(project_root, "src")
            if os.path.isdir(src_dir):
                for fname in os.listdir(src_dir):
                    _, ext = os.path.splitext(fname)
                    if ext:
                        extensions.add(ext.lower())
            language = detect_language(list(extensions))

        if not language:
            return ""

        toolkit = get_toolkit(language)
        if not toolkit:
            return ""

        return toolkit.get_prompt_hints()

    except Exception as e:
        logger.debug(f"Language hints detection failed: {e}")
        return ""
