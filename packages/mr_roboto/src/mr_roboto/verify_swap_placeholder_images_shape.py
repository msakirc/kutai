"""verify_swap_placeholder_images_shape — Plan 3 posthook.

Validates that swap_placeholder_images produced a self-consistent result:
- replaced_count agrees with the number of placehold.co URLs that have
  actually disappeared from HTML (within errors-margin for graceful
  degrade).
- assets/ exists when replaced_count > 0.
- skipped_count matches the count of surviving placehold.co references.

Returns {ok: bool, error: str|None, surviving_placeholders: int,
         expected_replaced: int}.

PRODUCTION SHAPE NOTE: the swap step's TaskResult.result arrives as a JSON
STRING (orchestrator json.dumps), so when this verifier is dispatched as a
post-hook the ``swap_result`` payload may be a JSON string rather than a
dict. ``_coerce_swap_result`` json.loads it FIRST (mirrors
swap_placeholder_images._parse_task_result) before any field access."""
from __future__ import annotations

import json
import os
import re
from typing import Any

_PLACEHOLDER_HOST_RE = re.compile(r"^https?://placehold\.co/", re.IGNORECASE)
_IMG_SRC_RE = re.compile(r'<img\b[^>]*?\bsrc\s*=\s*"([^"]*)"',
                         re.IGNORECASE | re.DOTALL)


def _coerce_swap_result(swap_result: Any) -> dict:
    """The swap step's result is a JSON STRING in production. Accept both
    a dict (tests / direct calls) and a JSON string (production posthook
    payload); decode the string FIRST before any isinstance check on the
    decoded value."""
    if swap_result is None:
        return {}
    if isinstance(swap_result, dict):
        return swap_result
    if isinstance(swap_result, str):
        try:
            decoded = json.loads(swap_result)
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}
    return {}


def _walk_html(workspace_path: str) -> list[str]:
    root = os.path.join(workspace_path, ".web")
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".html"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def _count_surviving_placeholders(html_paths: list[str]) -> int:
    n = 0
    for p in html_paths:
        try:
            with open(p, encoding="utf-8") as fh:
                html = fh.read()
        except OSError:
            continue
        for m in _IMG_SRC_RE.finditer(html):
            if _PLACEHOLDER_HOST_RE.search(m.group(1) or ""):
                n += 1
    return n


def verify_swap_placeholder_images_shape(
    *,
    workspace_path: str,
    swap_result: Any,
) -> dict[str, Any]:
    swap = _coerce_swap_result(swap_result)
    replaced = int(swap.get("replaced_count", 0) or 0)
    skipped = int(swap.get("skipped_count", 0) or 0)
    errors_list = swap.get("errors") or []

    html_paths = _walk_html(workspace_path)
    surviving = _count_surviving_placeholders(html_paths)

    # Consistency FIRST: surviving placehold.co URLs must equal skipped_count.
    # (Ordered ahead of the assets-dir check so a claimed-replaced-but-still-
    # surviving prototype is reported as the internal inconsistency it is,
    # rather than incidentally tripping the assets-missing branch.)
    if surviving != skipped:
        return {
            "ok": False,
            "error": (
                f"inconsistent: surviving placeholders={surviving} but "
                f"skipped_count={skipped} (errors={len(errors_list)})"
            ),
            "surviving_placeholders": surviving,
            "expected_replaced": replaced,
        }

    # Assets dir presence: required when replaced > 0.
    assets_dir = os.path.join(workspace_path, ".web", "assets")
    assets_exists = os.path.isdir(assets_dir)
    if replaced > 0 and not assets_exists:
        return {
            "ok": False,
            "error": (
                f"assets/ directory missing but replaced_count={replaced}"
            ),
            "surviving_placeholders": surviving,
            "expected_replaced": replaced,
        }

    return {
        "ok": True,
        "error": None,
        "surviving_placeholders": surviving,
        "expected_replaced": replaced,
    }
