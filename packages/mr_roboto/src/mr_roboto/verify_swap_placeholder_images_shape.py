"""verify_swap_placeholder_images_shape — Plan 3 posthook.

Validates the swap step's durable workspace artifacts (like
verify_charter_shape), so the gate is MEANINGFUL even when the producer's
``swap_result`` is unavailable.

CPS NOTE (2026-06-11): the 5.35 task result is now the KICKOFF shape — the
image chain runs asynchronously via durable continuations, so this verifier
may run MID-FLIGHT. It must NOT fail on surviving placehold.co URLs (the
chain may still be generating); the deep HTML verification (surviving ==
skipped consistency) lives in the chain's finalize — and is ENFORCED here
when the verifier runs post-finalize (ledger status=done carries the
recorded ``shape_check``).

Layers:

1. Self-derived broken-asset-ref check (always runs; the live i2p path).
   Walks ``<ws>/.web/**/*.html``. Scoped to refs the swap plausibly WROTE:
   a relative ``<img src>`` matching the swap's output shape —
   ``("../")*assets/<name>.png`` (the executor emits "assets/<pid>.png" for
   root HTML and "../assets/<pid>.png" for subdir screens; assets/ is flat).
   Such a ref pointing at a missing file is the real corruption mode → FAIL.
   Agent-authored relative refs (e.g. ``img/logo.png``, ``assets/photo.jpg``)
   that are missing on disk become WARNING notes in the result, never a
   failure — the swap never touched them. Surviving ``placehold.co``
   ``<img>`` are ACCEPTABLE (graceful degrade / mid-flight chain) and never
   fail the gate alone. placehold.co / data: / absolute / root-relative
   refs are excluded as before.

2a. swap_result validation (when swap_result non-empty — tests / back-compat
   / future cross-step wiring):
   - ``ok`` must not be False.
   - ``chain == "started"`` / ``"in_flight"`` (kickoff shapes; "in_flight"
     = a re-run kickoff that found the chain mid-flight): the chain ledger
     ``<ws>/.swap_state/swap_chain.json`` must exist, ``placeholder_count`` must
     be > 0 and match the ledger, and the ledger status must be one of
     prompts_pending / images_pending / done. No surviving-placeholder check
     (chain may be mid-flight).
   - ``chain == "none"`` or legacy no-chain dict: surviving placehold.co ==
     skipped_count, and assets/ exists when replaced_count > 0.

2b. LIVE ledger validation (when swap_result is empty — the live i2p case,
   no cross-step injection): the verifier reads the chain ledger ITSELF.
   - Ledger absent → PASS with a note (the kickoff may have found no
     placeholders to swap; can't distinguish, so tolerate).
   - Ledger present → same sanity as the chain branch: status must be in
     the known vocabulary, placeholder list non-empty. status == "done"
     additionally evaluates the finalize-recorded ``shape_check`` and FAILS
     surfacing its errors — this is the enforcement consumer for the
     finalize deep check. Mid-flight statuses stay tolerant. A done ledger
     WITHOUT a shape_check is the degraded kickoff (enqueue raised) →
     tolerate with a note.

Returns {ok: bool, error: str|None, surviving_placeholders: int,
         expected_replaced: int, broken_asset_refs: list[str],
         warnings: list[str]}.

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
# The swap executor's output shape: flat "<pid>.png" under assets/, referenced
# as "assets/<pid>.png" (root HTML) or "../assets/<pid>.png" (subdir screen).
# Anything else relative is agent-authored — warn, don't fail.
_SWAP_REF_RE = re.compile(r"^(?:\.\./)*assets/[^/\\]+\.png$", re.IGNORECASE)

_LEDGER_STATUSES = ("prompts_pending", "images_pending", "done")


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


def _ledger_path(workspace_path: str) -> str:
    return os.path.join(workspace_path, ".swap_state", "swap_chain.json")


def _load_chain_ledger(workspace_path: str) -> tuple[str, dict]:
    """Load ``<ws>/.swap_state/swap_chain.json``.

    Returns (state, ledger) with state in {"absent", "unreadable", "ok"} —
    the caller's semantics differ between a ledger that never existed (the
    kickoff may have found nothing) and one that is present but corrupt."""
    path = _ledger_path(workspace_path)
    if not os.path.exists(path):
        return "absent", {}
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
    except Exception:
        return "unreadable", {}
    if not isinstance(loaded, dict) or not loaded:
        return "unreadable", {}
    return "ok", loaded


def _scan_html(html_paths: list[str]) -> tuple[int, list[str], list[str]]:
    """Return (surviving_placeholder_count, broken_swap_refs,
    missing_agent_refs).

    A broken swap ref is a relative ``<img src>`` matching the swap's output
    pattern (``("../")*assets/<name>.png``) whose target file does not exist
    when resolved against the HTML file's own directory — the gate FAILS on
    these. Any OTHER relative ref missing on disk is agent-authored (the swap
    never wrote it) and is reported as a warning only."""
    surviving = 0
    broken: list[str] = []
    agent_missing: list[str] = []
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
                if _SWAP_REF_RE.match(clean.strip()):
                    broken.append(src)
                else:
                    agent_missing.append(src)
    return surviving, broken, agent_missing


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
    warnings: list[str] = []

    html_paths = _walk_html(workspace_path)
    surviving, broken_refs, agent_missing = _scan_html(html_paths)
    for ref in agent_missing:
        warnings.append(
            f"agent-authored ref missing on disk (not swap-written; "
            f"not fatal): {ref}"
        )

    def _verdict(ok: bool, error: str | None = None,
                 *, expected: int | None = None) -> dict[str, Any]:
        return {
            "ok": ok,
            "error": error,
            "surviving_placeholders": surviving,
            "expected_replaced": replaced if expected is None else expected,
            "broken_asset_refs": broken_refs,
            "warnings": warnings,
        }

    # Layer 1 (always): a swap-written asset ref pointing at a missing file
    # is the real corruption mode the live gate must catch. This is
    # meaningful even when swap_result is empty.
    if broken_refs:
        return _verdict(False, f"broken asset ref: {broken_refs[0]}")

    # Layer 2a (when swap_result is non-empty): producer-result validation
    # (tests / direct calls / future cross-step wiring).
    if have_swap_result:
        if swap.get("ok") is False:
            return _verdict(False, "producer reported ok=false")

        # CPS kickoff shape: the chain runs asynchronously — validate the
        # ledger instead of the (possibly mid-flight) HTML. Surviving
        # placehold.co URLs are EXPECTED here and must NOT fail the gate.
        # "in_flight" is the re-run kickoff shape (reentrancy guard found
        # the chain mid-flight; no overwrite, no duplicate enqueue) —
        # validated identically.
        if swap.get("chain") in ("started", "in_flight"):
            state, ledger = _load_chain_ledger(workspace_path)
            if state != "ok":
                return _verdict(
                    False, "chain started but ledger missing/unreadable",
                )
            pc = int(swap.get("placeholder_count") or 0)
            ledger_n = len(ledger.get("placeholders") or [])
            if pc <= 0 or pc != ledger_n:
                return _verdict(
                    False,
                    f"chain started but placeholder_count={pc} does not "
                    f"match ledger ({ledger_n})",
                )
            if ledger.get("status") not in _LEDGER_STATUSES:
                return _verdict(
                    False,
                    f"chain ledger has bad status: {ledger.get('status')!r}",
                )
            return _verdict(True)

        # Consistency FIRST: surviving placehold.co URLs must equal
        # skipped_count. (Ordered ahead of the assets-dir check so a
        # claimed-replaced-but-still-surviving prototype is reported as the
        # internal inconsistency it is, rather than incidentally tripping the
        # assets-missing branch.)
        if surviving != skipped:
            return _verdict(
                False,
                f"inconsistent: surviving placeholders={surviving} but "
                f"skipped_count={skipped} (errors={len(errors_list)})",
            )

        # Assets dir presence: required when replaced > 0.
        assets_dir = os.path.join(workspace_path, ".web", "assets")
        if replaced > 0 and not os.path.isdir(assets_dir):
            return _verdict(
                False,
                f"assets/ directory missing but replaced_count={replaced}",
            )

        return _verdict(True)

    # Layer 2b (LIVE: swap_result empty — no cross-step injection). Read the
    # chain ledger ourselves so the gate stays meaningful.
    state, ledger = _load_chain_ledger(workspace_path)
    if state == "absent":
        warnings.append(
            "no swap chain ledger found — kickoff may have found no "
            "placeholders to swap"
        )
        return _verdict(True)
    if state == "unreadable":
        return _verdict(False, "swap chain ledger present but unreadable")

    status = ledger.get("status")
    if status not in _LEDGER_STATUSES:
        return _verdict(False, f"chain ledger has bad status: {status!r}")
    if not (ledger.get("placeholders") or []):
        return _verdict(False, "chain ledger has no placeholders")

    if status == "done":
        expected = int(ledger.get("replaced") or 0)
        shape_check = ledger.get("shape_check")
        if isinstance(shape_check, dict):
            if shape_check.get("ok") is False:
                led_errs = [str(e) for e in (ledger.get("errors") or [])]
                detail = "; ".join(led_errs[:3]) or "no error detail recorded"
                return _verdict(
                    False,
                    f"finalized chain failed shape check: {detail}",
                    expected=expected,
                )
        else:
            # Degraded kickoff (prompt_writer enqueue raised) writes
            # status=done WITHOUT a shape_check — every placeholder kept its
            # placehold.co URL by design. Tolerate, but note it.
            warnings.append(
                "ledger done without shape_check (degraded kickoff — "
                "nothing was swapped)"
            )
        return _verdict(True, expected=expected)

    # Mid-flight (prompts_pending / images_pending): tolerant — the chain is
    # still generating; finalize owns the deep check.
    return _verdict(True)
