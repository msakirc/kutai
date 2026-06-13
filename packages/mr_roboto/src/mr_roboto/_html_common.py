"""Shared HTML-scanning primitives for the placeholder-swap chain and its
verifiers (``swap_placeholder_images``, ``verify_swap_placeholder_images_shape``,
``verify_html_prototype_shape``).

ONE definition each — these used to exist as byte-identical clones in three
modules. The regexes are pinned character-identical by tests; importers alias
them back under their historical private names so existing test imports keep
working."""
from __future__ import annotations

import json
import os
import re
from typing import Any

PLACEHOLDER_HOST_RE = re.compile(r"^https?://placehold\.co/", re.IGNORECASE)

# `<img ...>` matcher — captures the full tag text for attr extraction.
IMG_RE = re.compile(r"<img\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
ATTR_RE = re.compile(r'(\b[a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*"([^"]*)"')


def parse_attrs(tag_inner: str) -> dict[str, str]:
    return {k.lower(): v for k, v in ATTR_RE.findall(tag_inner)}


def walk_html(workspace_path: str) -> list[str]:
    """Recursive sorted walk of ``<ws>/.web/**/*.html`` (v2 fix: Plan 3 v1
    was flat and missed subdirectory screens)."""
    root = os.path.join(workspace_path, ".web")
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".html"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def coerce_result_dict(result: Any) -> dict:
    """Tolerant dict coercion for CPS handler results / persisted task
    results. Production persists JSON STRINGS (orchestrator json.dumps;
    restart-reconcile decodes only the TOP level; tests may fabricate
    strings), so a string is json.loads'd FIRST before any isinstance check
    on the decoded value. Anything non-dict degrades to {}."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            decoded = json.loads(result)
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}
    return {}
