# parsing/code_embeddings.py
"""
Phase 12.2 — Code Embedding Index
Phase E   — Code Intelligence Enhancements

Embeds function/class signatures + docstrings + body previews into
the codebase vector collection for semantic code search.

Features:
  - Incremental re-indexing (only re-embed changed files via hash comparison)
  - Triggered after file-change tool calls (post-tool hooks)
  - Startup re-scan of changed files
  - Test-to-code linking via metadata
  - Stores: filepath, symbol_name, symbol_type, line_start, line_end, language

Public API:
    stats   = await index_codebase(root_path)
    stats   = await reindex_file(filepath)
    results = await search_code(query, top_k=10)
    stats   = await startup_rescan(root_path)
    await post_tool_reindex(filepath, root_path)
"""
import hashlib
import os
import time
from typing import Optional

from src.infra.logging_config import get_logger
from .tree_sitter_parser import (
    detect_language,
    get_parseable_extensions,
    parse_file,
)
from ..memory.vector_store import is_ready, embed_and_store, query as vs_query

logger = get_logger("parsing.code_embeddings")


# ─── File Hash Tracking ──────────────────────────────────────────────────────

_file_hashes: dict[str, str] = {}

SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "chroma_data", "workspace",
}


def _file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file's contents."""
    try:
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return ""


def _needs_reindex(filepath: str) -> bool:
    """Check if a file has changed since last indexing."""
    current_hash = _file_hash(filepath)
    if not current_hash:
        return False
    previous = _file_hashes.get(filepath)
    return previous != current_hash


# ─── Embed Symbols ──────────────────────────────────────────────────────────

async def _embed_symbol(
    text: str,
    metadata: dict,
    doc_id: str,
) -> Optional[str]:
    """Store a code symbol embedding in the codebase collection."""
    try:
        if not is_ready():
            return None

        metadata["timestamp"] = time.time()
        metadata["type"] = "code_symbol"
        metadata["importance"] = 5

        return await embed_and_store(
            text=text,
            metadata=metadata,
            collection="codebase",
            doc_id=doc_id,
        )
    except Exception as e:
        logger.debug(f"Code embedding failed: {e}")
        return None


def _detect_test_module(filepath: str, rel_path: str) -> str:
    """
    Phase E: Detect if a file is a test file and determine the module it tests.

    Heuristics:
      - test_foo.py -> tests module "foo"
      - tests/test_bar.py -> tests module "bar"
      - foo_test.py -> tests module "foo"

    Returns the tested module name, or empty string if not a test file.
    """
    basename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(basename)[0]

    if name_no_ext.startswith("test_"):
        return name_no_ext[5:]  # test_foo -> foo
    if name_no_ext.endswith("_test"):
        return name_no_ext[:-5]  # foo_test -> foo
    if "/tests/" in rel_path or "\\tests\\" in rel_path:
        # File is in a tests directory
        return name_no_ext.replace("test_", "")

    return ""


async def _index_file_symbols(filepath: str, root_path: str) -> int:
    """
    Parse a file and embed all its symbols into the vector store.

    Phase E enhancement: includes test-to-code linking metadata and
    import context in embeddings.

    Returns the number of symbols embedded.
    """
    result = parse_file(filepath)
    language = result.get("language", "unknown")
    rel_path = os.path.relpath(filepath, root_path).replace("\\", "/")
    embedded = 0

    # Phase E: Detect test file and target module
    tests_module = _detect_test_module(filepath, rel_path)

    # Phase E: Collect imports for context enrichment
    imports = result.get("imports", [])
    import_context = ""
    if imports:
        import_names = []
        for imp in imports[:10]:
            if isinstance(imp, dict):
                import_names.append(imp.get("module", imp.get("name", "")))
            elif isinstance(imp, str):
                import_names.append(imp)
        if import_names:
            import_context = f"\nimports: {', '.join(import_names)}"

    # Embed functions
    for func in result.get("functions", []):
        name = func.get("name", "")
        if not name or name.startswith("_") and name != "__init__":
            continue  # Skip private helpers (too noisy)

        sig = func.get("signature", "")
        docstring = func.get("docstring", "")
        preview = func.get("body_preview", "")

        # Compose text for embedding (Phase E: include import context)
        text = f"function {name}{sig}"
        if docstring:
            text += f"\n{docstring}"
        if preview:
            text += f"\n{preview[:200]}"
        if import_context:
            text += import_context

        doc_id = f"code-{hashlib.sha256(f'{rel_path}:{name}'.encode()).hexdigest()[:16]}"

        metadata = {
            "filepath": rel_path,
            "symbol_name": name,
            "symbol_type": "function",
            "line_start": func.get("line_start", 0),
            "line_end": func.get("line_end", 0),
            "language": language,
            "signature": sig[:200],
        }

        # Phase E: Test-to-code linking
        if tests_module:
            metadata["tests_module"] = tests_module
            metadata["is_test"] = True

        stored = await _embed_symbol(
            text=text,
            metadata=metadata,
            doc_id=doc_id,
        )
        if stored:
            embedded += 1

    # Embed classes
    for cls in result.get("classes", []):
        name = cls.get("name", "")
        if not name:
            continue

        bases = ", ".join(cls.get("bases", []))
        docstring = cls.get("docstring", "")
        methods = [m.get("name", "") for m in cls.get("methods", [])]

        text = f"class {name}"
        if bases:
            text += f"({bases})"
        if docstring:
            text += f"\n{docstring}"
        if methods:
            text += f"\nmethods: {', '.join(methods[:10])}"
        if import_context:
            text += import_context

        doc_id = f"code-{hashlib.sha256(f'{rel_path}:class:{name}'.encode()).hexdigest()[:16]}"

        metadata = {
            "filepath": rel_path,
            "symbol_name": name,
            "symbol_type": "class",
            "line_start": cls.get("line_start", 0),
            "line_end": cls.get("line_end", 0),
            "language": language,
            "methods": ", ".join(methods[:20]),
        }

        # Phase E: Test-to-code linking
        if tests_module:
            metadata["tests_module"] = tests_module
            metadata["is_test"] = True

        stored = await _embed_symbol(
            text=text,
            metadata=metadata,
            doc_id=doc_id,
        )
        if stored:
            embedded += 1

    # Update hash
    _file_hashes[filepath] = _file_hash(filepath)

    return embedded


# ─── Public API ──────────────────────────────────────────────────────────────

async def index_codebase(root_path: str) -> dict:
    """
    Index all source files in a directory tree.

    Only re-indexes files that have changed since last indexing
    (incremental via file hash comparison).

    Args:
        root_path: Root directory to scan.

    Returns:
        Dict with stats: files_scanned, files_indexed, symbols_embedded.
    """
    extensions = get_parseable_extensions()
    files_scanned = 0
    files_indexed = 0
    symbols_embedded = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in extensions:
                continue

            full_path = os.path.join(dirpath, fname)
            files_scanned += 1

            # Skip if unchanged
            if not _needs_reindex(full_path):
                continue

            count = await _index_file_symbols(full_path, root_path)
            if count > 0:
                files_indexed += 1
                symbols_embedded += count

    logger.info(
        f"Code indexing: scanned {files_scanned} files, "
        f"indexed {files_indexed}, embedded {symbols_embedded} symbols"
    )

    return {
        "files_scanned": files_scanned,
        "files_indexed": files_indexed,
        "symbols_embedded": symbols_embedded,
    }


async def reindex_file(filepath: str, root_path: str = "") -> dict:
    """
    Re-index a single file (call after write_file/edit_file/patch_file).

    Args:
        filepath:   Path to the changed file.
        root_path:  Project root (for relative path computation).

    Returns:
        Dict with stats.
    """
    if not root_path:
        root_path = os.path.dirname(filepath) or "."

    if not os.path.isfile(filepath):
        return {"symbols_embedded": 0, "error": "file not found"}

    count = await _index_file_symbols(filepath, root_path)

    return {
        "filepath": filepath,
        "symbols_embedded": count,
    }


# ─── Phase E: Startup Re-scan ──────────────────────────────────────────────

async def startup_rescan(root_path: str) -> dict:
    """
    Quick re-scan of project files on startup to detect external changes.

    Compares file hashes against the last known state and re-indexes
    any files that have changed since the last run.

    Args:
        root_path: Root directory to scan.

    Returns:
        Dict with stats: files_checked, files_changed, symbols_embedded.
    """
    extensions = get_parseable_extensions()
    files_checked = 0
    files_changed = 0
    symbols_embedded = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in extensions:
                continue

            full_path = os.path.join(dirpath, fname)
            files_checked += 1

            if _needs_reindex(full_path):
                count = await _index_file_symbols(full_path, root_path)
                if count > 0:
                    files_changed += 1
                    symbols_embedded += count

    if files_changed:
        logger.info(
            f"Startup re-scan: {files_changed}/{files_checked} files changed, "
            f"{symbols_embedded} symbols re-embedded"
        )
    else:
        logger.debug(f"Startup re-scan: {files_checked} files checked, no changes")

    return {
        "files_checked": files_checked,
        "files_changed": files_changed,
        "symbols_embedded": symbols_embedded,
    }


# ─── Phase E: Post-Tool Reindex Hook ──────────────────────────────────────

# Tools that modify files and should trigger re-indexing
FILE_MODIFY_TOOLS = {"write_file", "edit_file", "patch_file", "apply_diff"}


async def post_tool_reindex(
    filepath: str,
    root_path: str = "",
) -> None:
    """
    Re-index a file after a file-modifying tool call.

    This is a lightweight wrapper around reindex_file that silently
    catches errors. Called from the agent loop after tool execution.

    Args:
        filepath:   Path to the modified file.
        root_path:  Project root for relative path computation.
    """
    try:
        if not is_ready():
            return

        # Only index parseable file types
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in get_parseable_extensions():
            return

        result = await reindex_file(filepath, root_path)
        count = result.get("symbols_embedded", 0)
        if count > 0:
            logger.debug(
                "Post-tool reindex: %d symbols in %s",
                count, os.path.basename(filepath),
            )
    except Exception as e:
        logger.debug("Post-tool reindex failed: %s", e)


async def search_code(query: str, top_k: int = 10) -> list[dict]:
    """
    Search the code embedding index for relevant symbols.

    Args:
        query:  Natural language or code query.
        top_k:  Number of results.

    Returns:
        List of dicts with: filepath, symbol_name, symbol_type,
        line_start, line_end, language, distance.
    """
    try:
        if not is_ready():
            return []

        results = await vs_query(
            text=query,
            collection="codebase",
            top_k=top_k,
        )

        formatted = []
        for r in results:
            meta = r.get("metadata", {})
            formatted.append({
                "filepath": meta.get("filepath", ""),
                "symbol_name": meta.get("symbol_name", ""),
                "symbol_type": meta.get("symbol_type", ""),
                "line_start": meta.get("line_start", 0),
                "line_end": meta.get("line_end", 0),
                "language": meta.get("language", ""),
                "distance": r.get("distance", 1.0),
                "text": r.get("text", "")[:200],
            })

        return formatted

    except Exception as e:
        logger.debug(f"Code search failed: {e}")
        return []
