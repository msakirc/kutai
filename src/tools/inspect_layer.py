"""inspect_layer — resolve the architectural layer for a source-file path.

Returns Layer ∈ {"domain", "adapter", "infra", "test", "ui", "unknown"}.

Resolution order
----------------
1. Spec-override  — reads ``layer_map.json`` from the current mission workspace
   (``mission_{mission_id}/.spec/layer_map.json`` if workspace_path is set).
   First matching glob wins.  Format::

       {"globs": {"src/domain/**": "domain", "src/infra/**": "infra", ...}}

2. Heuristic table — path segment keywords mapped to layer names (see
   ``_heuristic_layer`` below).

Soft-skip behaviour
-------------------
When no workspace_path is available the spec-override step is skipped and
only the heuristic is applied.  The function never raises — unknown paths
return ``"unknown"``.
"""
from __future__ import annotations

import json
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

Layer = Literal["domain", "adapter", "infra", "test", "ui", "unknown"]

# ──────────────────────────────────────────────────────────────────────────────
# Heuristic keyword table (priority order — first match wins)
# ──────────────────────────────────────────────────────────────────────────────

_HEURISTIC_RULES: list[tuple[list[str], Layer]] = [
    # test layer — checked first so test helpers in infra/ still classify as test
    (["test_", "_test.", ".test.", "tests/", "/tests/", "test.ts", "test.js", "test.go",
      "_test.go", ".spec.ts", ".spec.js"], "test"),
    # domain layer
    (["domain/", "/domain/"], "domain"),
    # adapter layer
    (["adapter/", "/adapter/", "gateway/", "/gateway/", "client/", "/client/"], "adapter"),
    # infra layer
    (["infra/", "/infra/", "storage/", "/storage/",
      "repo/", "/repo/", "repository/", "/repository/", "db/", "/db/"], "infra"),
    # ui layer
    (["components/", "/components/", "pages/", "/pages/",
      "views/", "/views/", "ui/", "/ui/"], "ui"),
]


def _heuristic_layer(path: str) -> Layer:
    """Apply the heuristic keyword table to *path*."""
    # Normalise separators so both / and \ work.
    norm = path.replace("\\", "/").lower()
    for keywords, layer in _HEURISTIC_RULES:
        if any(kw in norm for kw in keywords):
            return layer
    return "unknown"


def _load_layer_map(workspace_path: str) -> dict[str, Layer] | None:
    """Load globs dict from layer_map.json if it exists, else None."""
    candidates = [
        os.path.join(workspace_path, ".spec", "layer_map.json"),
        os.path.join(workspace_path, "layer_map.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, encoding="utf-8") as fh:
                    data = json.load(fh)
                globs = data.get("globs")
                if isinstance(globs, dict):
                    return globs  # type: ignore[return-value]
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _apply_spec_override(path: str, globs: dict[str, str]) -> Layer | None:
    """Return the first matching layer from *globs*, or None if no match."""
    norm = path.replace("\\", "/")
    for pattern, layer in globs.items():
        if fnmatch(norm, pattern) or fnmatch(norm, pattern.lstrip("/")):
            return layer  # type: ignore[return-value]
    return None


async def inspect_layer(
    path: str,
    workspace_path: str | None = None,
) -> Layer:
    """Return the architectural layer for *path*.

    Parameters
    ----------
    path:
        File path to classify (relative or absolute).
    workspace_path:
        Mission workspace root.  When provided the function checks
        ``<workspace_path>/.spec/layer_map.json`` for spec overrides.

    Returns
    -------
    Layer string: one of ``"domain"``, ``"adapter"``, ``"infra"``,
    ``"test"``, ``"ui"``, or ``"unknown"``.
    """
    if not path:
        return "unknown"

    # 1. Spec-override — only when we have a workspace root.
    if workspace_path:
        globs = _load_layer_map(workspace_path)
        if globs:
            override = _apply_spec_override(path, globs)
            if override is not None:
                return override

    # 2. Heuristic fallback.
    return _heuristic_layer(path)
