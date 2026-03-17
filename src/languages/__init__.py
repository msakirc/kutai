# languages/__init__.py
"""Multi-language toolkit for coding pipeline quality (Phase 10.1)."""
from .base import LanguageToolkit
from .python import PythonToolkit
from .javascript import JavaScriptToolkit
from .typescript import TypeScriptToolkit
from .go import GoToolkit
from .rust import RustToolkit

_REGISTRY: dict[str, LanguageToolkit] = {
    "python": PythonToolkit(),
    "javascript": JavaScriptToolkit(),
    "typescript": TypeScriptToolkit(),
    "go": GoToolkit(),
    "rust": RustToolkit(),
}

def get_toolkit(language: str) -> LanguageToolkit | None:
    """Get the toolkit for a given language name (case-insensitive)."""
    return _REGISTRY.get(language.lower())

def detect_language(file_extensions: list[str]) -> str | None:
    """Detect language from a list of file extensions."""
    ext_map = {
        ".py": "python",
        ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
    }
    counts: dict[str, int] = {}
    for ext in file_extensions:
        lang = ext_map.get(ext.lower())
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return max(counts, key=counts.get) if counts else None
