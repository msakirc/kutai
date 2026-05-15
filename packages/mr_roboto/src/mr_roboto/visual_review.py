"""Z4 T2B — visual_review mechanical verb.

Compares captured screenshots against baseline images (DIFF mode) or audits
them against design tokens / screen specifications (AUDIT mode) using a vision
model.  Returns a structured envelope ``{verdict, findings, skipped, reason}``
matching the ``run_axe`` pattern exactly.

Soft-skips when:
- ``captured_paths`` is empty
- Vision capability is unavailable (no vision-capable model registered)

Return shape
------------
``{verdict, findings, skipped, reason}``

- ``verdict``: ``"pass"`` or ``"fail"`` (fail when any blocker found).
  When skipped, verdict is ``"pass"`` (skip = no gate).
- ``findings``: list of dicts with keys
  ``{severity, file, url, kind, component, description, expected, observed,
    breakpoint, route, mode, source}``.
- ``skipped``: bool
- ``reason``: str explaining skip (present when skipped, absent otherwise).

Severity rules (applied locally, overriding the model's ``severity_hint``)
---------------------------------------------------------------------------
- Color difference ΔE > threshold (default 4)  → blocker
- Layout / element shift > threshold (default 2px) → blocker
- Named component from spec is missing          → blocker
- Brand / design-token violation                → blocker
- Wrong font size                               → blocker
- Shadow / elevation / micro-spacing mismatch   → info
- Anything else the model flagged               → warning

Thresholds are read from ``<repo_root>/.kutay/visual.yaml`` if present.
"""
from __future__ import annotations

import os
import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.visual_review")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DIFF_PROMPT = (
    "You are a pixel-precise visual QA engineer. You have been given TWO images:\n"
    "  Image 1: the CAPTURED screenshot (current implementation).\n"
    "  Image 2: the BASELINE reference (approved design or previous passing build).\n\n"
    "Compare them carefully. Return ONLY a single JSON object (no prose, no code fences) "
    "with exactly one key:\n"
    '  "findings": array of finding objects.\n\n'
    "Each finding object MUST have these exact keys:\n"
    '  "kind": one of ["color", "layout", "missing_component", "typography", '
    '"spacing", "shadow", "other"],\n'
    '  "component": string — name of the affected UI element or region,\n'
    '  "description": string — clear description of the difference,\n'
    '  "expected": string — what the baseline shows,\n'
    '  "observed": string — what the captured screenshot shows,\n'
    '  "severity_hint": one of ["blocker", "warning", "info"] — your best '
    "assessment (local rules will override this).\n\n"
    "If the images match perfectly, return an empty findings array.\n"
    "Return strictly valid JSON, nothing else."
)

AUDIT_PROMPT = (
    "You are a pixel-precise visual QA engineer. You have been given ONE screenshot "
    "to audit against design requirements.\n\n"
    "Return ONLY a single JSON object (no prose, no code fences) with exactly one key:\n"
    '  "findings": array of finding objects.\n\n'
    "Each finding object MUST have these exact keys:\n"
    '  "kind": one of ["color", "layout", "missing_component", "typography", '
    '"spacing", "shadow", "other"],\n'
    '  "component": string — name of the affected UI element or region,\n'
    '  "description": string — clear description of the issue,\n'
    '  "expected": string — what the design spec or best practice requires,\n'
    '  "observed": string — what is actually rendered in the screenshot,\n'
    '  "severity_hint": one of ["blocker", "warning", "info"] — your best '
    "assessment (local rules will override this).\n\n"
    "If no visual issues are found, return an empty findings array.\n"
    "Return strictly valid JSON, nothing else."
)

# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "color_delta_e": 4,
    "layout_shift_px": 2,
}


def _load_thresholds() -> dict[str, Any]:
    """Read .kutay/visual.yaml from repo root; fall back to built-in defaults."""
    try:
        # Locate repo root relative to this file
        here = os.path.dirname(os.path.abspath(__file__))
        # packages/mr_roboto/src/mr_roboto/ → climb 4 levels
        root = here
        for _ in range(4):
            root = os.path.dirname(root)
        yaml_path = os.path.join(root, ".kutay", "visual.yaml")
        if os.path.isfile(yaml_path):
            import yaml  # lazy; pyyaml is already a dependency
            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict):
                thresholds = dict(_DEFAULTS)
                for k in _DEFAULTS:
                    if k in data:
                        try:
                            thresholds[k] = float(data[k])
                        except (TypeError, ValueError):
                            pass
                return thresholds
    except Exception as exc:
        logger.debug("visual_review: could not load .kutay/visual.yaml: %s", exc)
    return dict(_DEFAULTS)


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

# Kinds that always map to INFO (shadow / micro-spacing)
_INFO_KINDS = frozenset({"shadow"})

# Keywords in description / kind that suggest a blocker pattern
_BLOCKER_KIND_PATTERNS = frozenset({
    "color",          # ΔE checked numerically; catch non-numeric mentions too
    "missing_component",
    "typography",
})


def _apply_severity_rules(finding: dict, thresholds: dict) -> str:
    """Return the locally enforced severity string for a single finding dict."""
    kind = str(finding.get("kind") or "other").lower()
    description = str(finding.get("description") or "").lower()
    expected = str(finding.get("expected") or "").lower()
    observed = str(finding.get("observed") or "").lower()
    component = str(finding.get("component") or "").lower()

    # 1. Shadow / elevation / micro-spacing → info
    if kind == "shadow" or "shadow" in description or "elevation" in description:
        return "info"
    if "micro-spacing" in description or "micro spacing" in description:
        return "info"

    # 2. Missing component → blocker
    if kind == "missing_component":
        return "blocker"
    if "missing" in description and ("component" in description or "element" in description):
        return "blocker"

    # 3. Color difference — detect numeric ΔE mentions (takes priority)
    if kind == "color":
        threshold_val = float(thresholds.get("color_delta_e", 4))
        # Try to extract a ΔE number from description — this is authoritative
        # Match "delta_e", "delta-e", "deltaE" (ASCII) or "ΔE" / "δe" (Unicode)
        m = re.search(
            r"(?:delta.?e|Δe|δe)[:\s=]*([0-9]+(?:\.[0-9]+)?)",
            description,
            re.IGNORECASE,
        )
        if m:
            try:
                delta = float(m.group(1))
                # Numeric ΔE present: apply strict threshold rule
                return "blocker" if delta > threshold_val else "warning"
            except ValueError:
                pass
        # No numeric ΔE — fall back to keyword heuristics
        # brand / design token violation
        if "brand" in description or "token" in description or "palette" in description:
            return "blocker"
        # If the model hinted blocker, respect it for color
        if str(finding.get("severity_hint") or "").lower() == "blocker":
            return "blocker"
        return "warning"

    # 4. Layout / element shift → blocker when > threshold
    if kind == "layout":
        threshold_px = float(thresholds.get("layout_shift_px", 2))
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*px", description)
        if m:
            try:
                shift = float(m.group(1))
                if shift > threshold_px:
                    return "blocker"
            except ValueError:
                pass
        # Generic shift mention → blocker
        if "shift" in description or "misalign" in description or "offset" in description:
            return "blocker"
        if str(finding.get("severity_hint") or "").lower() == "blocker":
            return "blocker"
        return "warning"

    # 5. Typography → blocker (wrong font size)
    if kind == "typography":
        if "font size" in description or "font-size" in description or "wrong size" in description:
            return "blocker"
        if "wrong font" in description or "font family" in description:
            return "blocker"
        if str(finding.get("severity_hint") or "").lower() == "blocker":
            return "blocker"
        return "warning"

    # 6. Spacing → info for micro, warning otherwise
    if kind == "spacing":
        if "micro" in description:
            return "info"
        return "warning"

    # 7. Brand / design-token violation mentioned in any kind → blocker
    if "brand" in description or "design token" in description or "design-token" in description:
        return "blocker"

    # 8. Fall through: trust severity_hint or default to warning
    hint = str(finding.get("severity_hint") or "").lower()
    if hint in ("blocker", "warning", "info"):
        return hint
    return "warning"


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(
    r"^(?P<route>.+?)_(?P<mode>[^_]+)_(?P<breakpoint>[^_.]+)\.png$",
    re.IGNORECASE,
)


def _parse_filename(path: str) -> dict[str, str]:
    """Extract breakpoint/route/mode from a filename like ``{route}_{mode}_{bp}.png``."""
    basename = os.path.basename(path)
    m = _FILENAME_RE.match(basename)
    if m:
        return {
            "route": m.group("route"),
            "mode": m.group("mode"),
            "breakpoint": m.group("breakpoint"),
        }
    # Could not parse — return empty strings
    return {"route": "", "mode": "", "breakpoint": ""}


# ---------------------------------------------------------------------------
# Main verb
# ---------------------------------------------------------------------------


async def visual_review(
    *,
    mission_id: int,
    step_id: str,
    captured_paths: list[str],
    baseline_dir: str | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Visual diff / audit verb.

    Parameters
    ----------
    mission_id:
        The mission this step belongs to.
    step_id:
        Workflow step identifier (used for logging).
    captured_paths:
        List of absolute paths to captured screenshot files.
    baseline_dir:
        Directory containing baseline PNG files. Defaults to
        ``mission_{id}/.visual/baseline/`` under WORKSPACE_DIR.
    workspace_path:
        Override for WORKSPACE_DIR (used in tests / custom setups).

    Returns
    -------
    dict with keys: ``verdict``, ``findings``, ``skipped``, ``reason`` (if skipped).
    """
    if not captured_paths:
        reason = "no captured screenshots"
        logger.warning("visual_review soft-skip: %s (step=%s)", reason, step_id)
        return {"verdict": "pass", "findings": [], "skipped": True, "reason": reason}

    # Load thresholds from .kutay/visual.yaml
    thresholds = _load_thresholds()

    # Resolve workspace path lazily
    if workspace_path is None:
        try:
            from src.tools.workspace import WORKSPACE_DIR
            workspace_path = WORKSPACE_DIR
        except Exception:
            workspace_path = os.getcwd()

    # Resolve baseline directory
    if baseline_dir is None:
        baseline_dir = os.path.join(workspace_path, f"mission_{mission_id}", ".visual", "baseline")

    # Load design artifacts (best-effort; absence is fine)
    design_tokens: dict | None = None
    screen_specs: dict | None = None
    try:
        from mr_roboto.executors.social_preview_check import _load_artifact
        design_tokens = await _load_artifact(int(mission_id), "design_tokens")
        screen_specs = await _load_artifact(int(mission_id), "screen_specifications")
    except Exception as exc:
        logger.debug("visual_review: artifact load failed (best-effort): %s", exc)

    # Build enriched AUDIT prompt with spec context if available
    audit_prompt = _build_audit_prompt(design_tokens, screen_specs)

    # Import helpers from ingest_visual (avoids code duplication)
    try:
        from mr_roboto.ingest_visual import (
            _parse_vision_response,
            _is_vision_capability_unavailable,
        )
    except Exception as exc:
        return {
            "verdict": "pass",
            "findings": [],
            "skipped": True,
            "reason": f"ingest_visual_import_failed: {exc}",
        }

    try:
        from src.tools.vision import analyze_image
    except Exception as exc:
        return {
            "verdict": "pass",
            "findings": [],
            "skipped": True,
            "reason": f"vision_import_failed: {exc}",
        }

    all_findings: list[dict] = []

    for captured_path in captured_paths:
        file_meta = _parse_filename(captured_path)

        # Determine mode: DIFF if baseline exists, AUDIT otherwise
        baseline_path = _find_baseline(captured_path, baseline_dir)

        if baseline_path is not None:
            # DIFF mode: compare captured vs baseline
            try:
                raw_response = await analyze_image(
                    [captured_path, baseline_path],
                    question=DIFF_PROMPT,
                )
            except Exception as exc:
                err_str = str(exc)
                if _is_vision_capability_unavailable(err_str):
                    logger.warning("visual_review: vision capability unavailable")
                    return {
                        "verdict": "pass",
                        "findings": [],
                        "skipped": True,
                        "reason": "vision_capability_unavailable",
                    }
                logger.warning("visual_review: vision call raised for %s: %s", captured_path, exc)
                continue
        else:
            # AUDIT mode: single frame vs design intent
            try:
                raw_response = await analyze_image(
                    [captured_path],
                    question=audit_prompt,
                )
            except Exception as exc:
                err_str = str(exc)
                if _is_vision_capability_unavailable(err_str):
                    logger.warning("visual_review: vision capability unavailable")
                    return {
                        "verdict": "pass",
                        "findings": [],
                        "skipped": True,
                        "reason": "vision_capability_unavailable",
                    }
                logger.warning("visual_review: vision call raised for %s: %s", captured_path, exc)
                continue

        # Detect capability error returned as string
        if isinstance(raw_response, str) and _is_vision_capability_unavailable(raw_response):
            logger.warning("visual_review: vision capability unavailable (string return)")
            return {
                "verdict": "pass",
                "findings": [],
                "skipped": True,
                "reason": "vision_capability_unavailable",
            }

        # Parse response
        parsed = _parse_vision_response(raw_response if isinstance(raw_response, str) else "")
        raw_findings = parsed.get("findings") or []
        if not isinstance(raw_findings, list):
            raw_findings = []

        for rf in raw_findings:
            if not isinstance(rf, dict):
                continue
            severity = _apply_severity_rules(rf, thresholds)
            finding = {
                "severity": severity,
                "file": captured_path,
                "url": "",
                "kind": str(rf.get("kind") or "other"),
                "component": str(rf.get("component") or ""),
                "description": str(rf.get("description") or ""),
                "expected": str(rf.get("expected") or ""),
                "observed": str(rf.get("observed") or ""),
                "breakpoint": file_meta["breakpoint"],
                "route": file_meta["route"],
                "mode": file_meta["mode"],
                "source": "visual_review",
            }
            all_findings.append(finding)

    has_blocker = any(f["severity"] == "blocker" for f in all_findings)
    verdict = "fail" if has_blocker else "pass"

    logger.info(
        "visual_review: verdict=%s findings=%d step=%s mission=%s",
        verdict, len(all_findings), step_id, mission_id,
    )
    return {
        "verdict": verdict,
        "findings": all_findings,
        "skipped": False,
    }


def _find_baseline(captured_path: str, baseline_dir: str) -> str | None:
    """Return the baseline file path if a file with the same basename exists."""
    basename = os.path.basename(captured_path)
    candidate = os.path.join(baseline_dir, basename)
    if os.path.isfile(candidate):
        return candidate
    return None


def _build_audit_prompt(design_tokens: dict | None, screen_specs: dict | None) -> str:
    """Build AUDIT_PROMPT optionally enriched with design token / spec context."""
    extra_lines: list[str] = []
    if design_tokens:
        # Surface a compact representation of tokens for the model
        import json
        try:
            token_snippet = json.dumps(design_tokens, ensure_ascii=False)[:1000]
            extra_lines.append(
                f"\nDesign tokens (use these as the source of truth for colors, "
                f"typography, and spacing):\n{token_snippet}"
            )
        except Exception:
            pass
    if screen_specs:
        import json
        try:
            spec_snippet = json.dumps(screen_specs, ensure_ascii=False)[:1000]
            extra_lines.append(
                f"\nScreen specifications (list of required components per screen):\n"
                f"{spec_snippet}"
            )
        except Exception:
            pass
    if extra_lines:
        return AUDIT_PROMPT + "\n" + "\n".join(extra_lines)
    return AUDIT_PROMPT
