"""verify_swap_placeholder_images_shape — Plan 3 posthook.

Validates the swap step's durable workspace artifacts (like
verify_charter_shape), so the gate is MEANINGFUL even when the producer's
``swap_result`` is unavailable.

CPS NOTE (2026-06-11): the 5.35 task result is now the KICKOFF shape — the
image chain runs asynchronously via durable continuations, so this verifier
may run MID-FLIGHT. It must NOT fail on surviving placehold.co URLs (the
chain may still be generating); the deep HTML verification (surviving ==
skipped consistency) lives in the chain's finalize, not here.

Layers:

1. Self-derived broken-asset-ref check (always runs; the live i2p path).
   Walks ``<ws>/.web/**/*.html``. Every ``<img src="...">`` whose src is a
   RELATIVE rewritten asset ref (e.g. ``assets/<id>.png`` — NOT a placehold.co
   URL, NOT an absolute http(s) URL) must reference a file that EXISTS, resolved
   relative to the HTML file's own directory. A rewritten ref pointing at a
   missing file is the real corruption mode → FAIL. Surviving ``placehold.co``
   ``<img>`` are ACCEPTABLE (graceful degrade / mid-flight chain) and never
   fail the gate alone.

2. swap_result validation (ADDITIONAL; only when swap_result non-empty):
   - ``ok`` must not be False.
   - ``chain == "started"`` / ``"in_flight"`` (kickoff shapes; "in_flight"
     = a re-run kickoff that found the chain mid-flight): the chain ledger
     ``<ws>/.swap_state/swap_chain.json`` must exist, ``placeholder_count`` must be
     > 0 and match the ledger, and the ledger status must be one of
     prompts_pending / images_pending / done. No surviving-placeholder check
     (chain may be mid-flight).
   - ``chain == "none"`` or legacy no-chain dict: surviving placehold.co ==
     skipped_count, and assets/ exists when replaced_count > 0 (tests /
     direct calls / future cross-step wiring).
   When swap_result is empty (the live i2p case — no cross-step injection)
   layer 2 is skipped and the verdict rests on layer 1.

Returns {ok: bool, error: str|None, surviving_placeholders: int,
         expected_replaced: int, broken_asset_refs: list[str]}.

PRODUCTION SHAPE NOTE: a persisted task result arrives as a JSON STRING
(orchestrator json.dumps), so the ``swap_result`` payload may be a JSON
string rather than a dict. ``_coerce_swap_result`` json.loads it FIRST
before any field access."""
from __future__ import annotations

import json
import os
import re
from typing import Any

_PLACEHOLDER_HOST_RE = re.compile(r"^https?://placehold\.co/", re.IGNORECASE)
_ABSOLUTE_URL_RE = re.compile(r"^(?:https?:)?//|^[a-z][a-z0-9+.\-]*:",
                              re.IGNORECASE)
_DATA_URI_RE = re.compile(r"^data:", re.IGNORECASE)
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


def _is_rewritten_asset_ref(src: str) -> bool:
    """A relative, locally-rewritten asset reference (the corruption-prone
    case). NOT a placehold.co URL, NOT an absolute/scheme/protocol-relative
    URL, NOT a data: URI, NOT a root-relative ("/..."), and non-empty.

    A leading "/" makes the ref ROOT-ANCHORED (resolved against the server
    document root, not the HTML file's own dir), so it is NOT the executor's
    output — the swap step only ever emits truly-relative refs ("assets/x.png"
    or "../assets/x.png"). Root-relative refs (e.g. a pre-existing
    "/assets/already_real.png" the executor leaves untouched) must be skipped,
    not resolved relative to the HTML dir, which would yield a bogus path and a
    false broken-asset-ref."""
    s = (src or "").strip()
    if not s:
        return False
    if _PLACEHOLDER_HOST_RE.search(s):
        return False
    if _DATA_URI_RE.match(s):
        return False
    if _ABSOLUTE_URL_RE.match(s):
        return False
    if s.startswith("/"):
        return False
    return True


def _scan_html(html_paths: list[str]) -> tuple[int, list[str]]:
    """Return (surviving_placeholder_count, broken_asset_refs).

    A broken asset ref is a relative rewritten ``<img src>`` whose target file
    does not exist when resolved against the HTML file's own directory."""
    surviving = 0
    broken: list[str] = []
    for p in html_paths:
        try:
            with open(p, encoding="utf-8") as fh:
                html = fh.read()
        except OSError:
            continue
        base_dir = os.path.dirname(p)
        for m in _IMG_SRC_RE.finditer(html):
            src = m.group(1) or ""
            if _PLACEHOLDER_HOST_RE.search(src):
                surviving += 1
                continue
            if not _is_rewritten_asset_ref(src):
                continue
            # Resolve the relative ref against the HTML file's directory.
            # Strip any query/fragment before the filesystem check.
            clean = src.split("?", 1)[0].split("#", 1)[0]
            target = os.path.normpath(os.path.join(base_dir, clean))
            if not os.path.isfile(target):
                broken.append(src)
    return surviving, broken


def verify_swap_placeholder_images_shape(
    *,
    workspace_path: str,
    swap_result: Any,
) -> dict[str, Any]:
    swap = _coerce_swap_result(swap_result)
    have_swap_result = bool(swap)
    replaced = int(swap.get("replaced_count", 0) or 0)
    skipped = int(swap.get("skipped_count", 0) or 0)
    errors_list = swap.get("errors") or []

    html_paths = _walk_html(workspace_path)
    surviving, broken_refs = _scan_html(html_paths)

    # Layer 1 (always): a rewritten asset ref pointing at a missing file is
    # the real corruption mode the live gate must catch. This is meaningful
    # even when swap_result is empty.
    if broken_refs:
        return {
            "ok": False,
            "error": f"broken asset ref: {broken_refs[0]}",
            "surviving_placeholders": surviving,
            "expected_replaced": replaced,
            "broken_asset_refs": broken_refs,
        }

    # Layer 2 (only when swap_result is non-empty): producer-result
    # validation. In the live i2p path swap_result is empty (no cross-step
    # injection), so these checks are skipped and the verdict rests on layer 1.
    if have_swap_result:
        if swap.get("ok") is False:
            return {
                "ok": False,
                "error": "producer reported ok=false",
                "surviving_placeholders": surviving,
                "expected_replaced": replaced,
                "broken_asset_refs": broken_refs,
            }

        # CPS kickoff shape: the chain runs asynchronously — validate the
        # ledger instead of the (possibly mid-flight) HTML. Surviving
        # placehold.co URLs are EXPECTED here and must NOT fail the gate.
        # "in_flight" is the re-run kickoff shape (reentrancy guard found
        # the chain mid-flight; no overwrite, no duplicate enqueue) —
        # validated identically.
        if swap.get("chain") in ("started", "in_flight"):
            ledger_path = os.path.join(workspace_path, ".swap_state",
                                       "swap_chain.json")
            ledger: dict = {}
            try:
                with open(ledger_path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                ledger = loaded if isinstance(loaded, dict) else {}
            except Exception:
                ledger = {}
            if not ledger:
                return {
                    "ok": False,
                    "error": "chain started but ledger missing/unreadable",
                    "surviving_placeholders": surviving,
                    "expected_replaced": replaced,
                    "broken_asset_refs": broken_refs,
                }
            pc = int(swap.get("placeholder_count") or 0)
            ledger_n = len(ledger.get("placeholders") or [])
            if pc <= 0 or pc != ledger_n:
                return {
                    "ok": False,
                    "error": (
                        f"chain started but placeholder_count={pc} does not "
                        f"match ledger ({ledger_n})"
                    ),
                    "surviving_placeholders": surviving,
                    "expected_replaced": replaced,
                    "broken_asset_refs": broken_refs,
                }
            if ledger.get("status") not in (
                "prompts_pending", "images_pending", "done",
            ):
                return {
                    "ok": False,
                    "error": f"chain ledger has bad status: "
                             f"{ledger.get('status')!r}",
                    "surviving_placeholders": surviving,
                    "expected_replaced": replaced,
                    "broken_asset_refs": broken_refs,
                }
            return {
                "ok": True,
                "error": None,
                "surviving_placeholders": surviving,
                "expected_replaced": replaced,
                "broken_asset_refs": broken_refs,
            }

        # Consistency FIRST: surviving placehold.co URLs must equal
        # skipped_count. (Ordered ahead of the assets-dir check so a
        # claimed-replaced-but-still-surviving prototype is reported as the
        # internal inconsistency it is, rather than incidentally tripping the
        # assets-missing branch.)
        if surviving != skipped:
            return {
                "ok": False,
                "error": (
                    f"inconsistent: surviving placeholders={surviving} but "
                    f"skipped_count={skipped} (errors={len(errors_list)})"
                ),
                "surviving_placeholders": surviving,
                "expected_replaced": replaced,
                "broken_asset_refs": broken_refs,
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
                "broken_asset_refs": broken_refs,
            }

    return {
        "ok": True,
        "error": None,
        "surviving_placeholders": surviving,
        "expected_replaced": replaced,
        "broken_asset_refs": broken_refs,
    }
