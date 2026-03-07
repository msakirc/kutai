# parsing/code_embeddings.py
"""
Phase 12.2 — Code Embedding Index

Embeds function/class signatures + docstrings + body previews into
the codebase vector collection for semantic code search.

Features:
  - Incremental re-indexing (only re-embed changed files via hash comparison)
  - Triggered after file-change tool calls
  - Stores: filepath, symbol_name, symbol_type, line_start, line_end, language

Public API:
    stats  = await index_codebase(root_path)
    stats  = await reindex_file(filepath)
    results = await search_code(query, top_k=10)
"""
import hashlib
import logging
import os
import time
from typing import Optional

from parsing.tree_sitter_parser import (
    detect_language,
    get_parseable_extensions,
    parse_file,
)

logger = logging.getLogger(__name__)


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
        from memory.vector_store import embed_and_store, is_ready
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


async def _index_file_symbols(filepath: str, root_path: str) -> int:
    """
    Parse a file and embed all its symbols into the vector store.

    Returns the number of symbols embedded.
    """
    result = parse_file(filepath)
    language = result.get("language", "unknown")
    rel_path = os.path.relpath(filepath, root_path).replace("\\", "/")
    embedded = 0

    # Embed functions
    for func in result.get("functions", []):
        name = func.get("name", "")
        if not name or name.startswith("_") and name != "__init__":
            continue  # Skip private helpers (too noisy)

        sig = func.get("signature", "")
        docstring = func.get("docstring", "")
        preview = func.get("body_preview", "")

        # Compose text for embedding
        text = f"function {name}{sig}"
        if docstring:
            text += f"\n{docstring}"
        if preview:
            text += f"\n{preview[:200]}"

        doc_id = f"code-{hashlib.sha256(f'{rel_path}:{name}'.encode()).hexdigest()[:16]}"

        stored = await _embed_symbol(
            text=text,
            metadata={
                "filepath": rel_path,
                "symbol_name": name,
                "symbol_type": "function",
                "line_start": func.get("line_start", 0),
                "line_end": func.get("line_end", 0),
                "language": language,
                "signature": sig[:200],
            },
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

        doc_id = f"code-{hashlib.sha256(f'{rel_path}:class:{name}'.encode()).hexdigest()[:16]}"

        stored = await _embed_symbol(
            text=text,
            metadata={
                "filepath": rel_path,
                "symbol_name": name,
                "symbol_type": "class",
                "line_start": cls.get("line_start", 0),
                "line_end": cls.get("line_end", 0),
                "language": language,
                "methods": ", ".join(methods[:20]),
            },
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
        from memory.vector_store import query as vs_query, is_ready
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
