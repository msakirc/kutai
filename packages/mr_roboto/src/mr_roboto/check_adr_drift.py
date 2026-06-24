"""ADR drift gate — mechanical-first architectural decision record checker.

For each ADR in the mission's ``.adr/`` directory, extract the
``falsification_signal`` and run mechanical checks against the produced
files:

v2 object form
--------------
- ``forbidden_imports``: scan produced files for matching ``import`` /
  ``from ... import`` (Python) or ``import ... from`` / ``require``
  (JS/TS) statements.
- ``forbidden_patterns``: compile each as a regex; scan file contents.
- ``required_test_coverage``: if True, verify at least one produced file
  matches a test-file naming convention.

v1 string form / null
---------------------
Treat as judgment-only — skip mechanical checks, surface the ADR id in
``judgment_only_adr_ids`` for a downstream LLM-judge follow-up.  No
mechanical fail is emitted.

Soft-skip
---------
When no ADR register and no ``.adr/*.json`` files are found, return
``skipped=True, verdict="pass"`` immediately — the verb never errors on
missions without ADRs.

Return shape
------------
``{verdict, findings, judgment_only_adr_ids, skipped}``

- ``verdict``: ``"pass"`` or ``"fail"``
- ``findings``: list of ``{severity, file, why, adr_id, signal_type}``
  (blockers from mechanical violations)
- ``judgment_only_adr_ids``: ADR ids that need LLM-judge follow-up
- ``skipped``: True when no ADR data found

Only mechanical violations trigger ``verdict="fail"``; judgment-only ADRs
never block.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.check_adr_drift")

# ---------------------------------------------------------------------------
# Test-file naming conventions (required_test_coverage)
# ---------------------------------------------------------------------------

_TEST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(^|/)test_[^/]+\.py$"),
    re.compile(r"(^|/)[^/]+_test\.py$"),
    re.compile(r"(^|/)[^/]+\.test\.(ts|tsx|js|jsx)$"),
    re.compile(r"(^|/)[^/]+\.spec\.(ts|tsx|js|jsx)$"),
    re.compile(r"(^|/)[^/]+_test\.go$"),
]


def _is_test_file(path: str) -> bool:
    for pat in _TEST_PATTERNS:
        if pat.search(path):
            return True
    return False


# ---------------------------------------------------------------------------
# Forbidden-import detection
# ---------------------------------------------------------------------------

# Python: `import X` or `from X import ...` (top-level module)
_PY_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+(\S+)|from\s+(\S+)\s+import)",
    re.MULTILINE,
)
# JS/TS: `import ... from "X"` or `require("X")`
_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+.*?\bfrom\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\))""",
    re.DOTALL,
)


def _scan_forbidden_imports(
    content: str, forbidden: list[str], file_path: str, suffix: str
) -> list[dict]:
    """Return findings for each forbidden import found in *content*."""
    findings = []
    for forbidden_pkg in forbidden:
        if suffix == ".py":
            for m in _PY_IMPORT_RE.finditer(content):
                top = (m.group(1) or m.group(2) or "").split(".")[0]
                if top and top == forbidden_pkg.split(".")[0]:
                    findings.append({
                        "severity": "blocker",
                        "file": file_path,
                        "why": f"forbidden import '{forbidden_pkg}' found",
                        "signal_type": "forbidden_imports",
                    })
                    break  # one finding per forbidden pkg per file
        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            for m in _JS_IMPORT_RE.finditer(content):
                specifier = m.group(1) or m.group(2) or ""
                pkg = specifier.split("/")[0]
                if pkg == forbidden_pkg.split("/")[0]:
                    findings.append({
                        "severity": "blocker",
                        "file": file_path,
                        "why": f"forbidden import '{forbidden_pkg}' found",
                        "signal_type": "forbidden_imports",
                    })
                    break
    return findings


# ---------------------------------------------------------------------------
# Forbidden-pattern detection
# ---------------------------------------------------------------------------

def _scan_forbidden_patterns(
    content: str, patterns: list[str], file_path: str
) -> list[dict]:
    """Return findings for each forbidden regex pattern found in *content*."""
    findings = []
    for raw_pat in patterns:
        try:
            compiled = re.compile(raw_pat, re.MULTILINE)
        except re.error as exc:
            logger.warning(
                "check_adr_drift: invalid forbidden_pattern regex",
                pattern=raw_pat, error=str(exc),
            )
            continue
        if compiled.search(content):
            findings.append({
                "severity": "blocker",
                "file": file_path,
                "why": f"forbidden pattern matched: {raw_pat!r}",
                "signal_type": "forbidden_patterns",
            })
    return findings


# ---------------------------------------------------------------------------
# ADR enumeration
# ---------------------------------------------------------------------------

def _enumerate_adr_files(register_path: str, workspace_path: str | None) -> list[str]:
    """Return a list of ADR JSON file paths.

    Priority:
    1. Parse ``register_path`` (the register.md) to find ``*.json`` files in
       its parent ``.adr/`` directory.
    2. Fall back to listing all ``*.json`` files in the ``.adr/`` directory
       adjacent to register_path.
    3. If workspace_path given, also try ``{workspace_path}/.adr/*.json``.

    Returns empty list (→ soft-skip) when nothing found.
    """
    candidates: list[str] = []

    # Derive the .adr/ dir from the register path
    register_p = Path(register_path) if register_path else None
    adr_dirs: list[Path] = []

    if register_p:
        adr_dirs.append(register_p.parent)
    if workspace_path:
        adr_dirs.append(Path(workspace_path) / ".adr")

    for adr_dir in adr_dirs:
        if adr_dir.is_dir():
            for entry in adr_dir.iterdir():
                if entry.suffix == ".json" and entry.is_file():
                    p = str(entry)
                    if p not in candidates:
                        candidates.append(p)

    return candidates


# ---------------------------------------------------------------------------
# Single-ADR mechanical check
# ---------------------------------------------------------------------------

def _check_one_adr(
    adr_id: str,
    falsification: Any,
    produced_files: list[str],
    workspace_path: str | None,
) -> tuple[list[dict], bool]:
    """Check one ADR's falsification_signal against produced_files.

    Returns ``(findings, is_judgment_only)``.
    """
    # null → judgment-only
    if falsification is None:
        return [], True

    # v1 string form → judgment-only
    if isinstance(falsification, str):
        return [], True

    # v2 object form → mechanical checks
    if not isinstance(falsification, dict):
        # Unknown shape — treat conservatively as judgment-only
        return [], True

    findings: list[dict] = []
    forbidden_imports: list[str] = falsification.get("forbidden_imports") or []
    forbidden_patterns: list[str] = falsification.get("forbidden_patterns") or []
    required_coverage: bool = bool(falsification.get("required_test_coverage", False))

    # Resolve files: produced_files may be relative; resolve under workspace
    resolved: list[tuple[str, Path]] = []
    for rel in produced_files:
        p = Path(rel)
        if not p.is_absolute() and workspace_path:
            p = Path(workspace_path) / p
        resolved.append((rel, p))

    # Check forbidden imports + patterns against each file
    for (rel, p) in resolved:
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        suffix = p.suffix.lower()

        if forbidden_imports:
            for finding in _scan_forbidden_imports(
                content, forbidden_imports, rel, suffix
            ):
                finding["adr_id"] = adr_id
                findings.append(finding)

        if forbidden_patterns:
            for finding in _scan_forbidden_patterns(content, forbidden_patterns, rel):
                finding["adr_id"] = adr_id
                findings.append(finding)

    # required_test_coverage: at least one produced file must be a test file
    if required_coverage:
        has_test = any(_is_test_file(rel) for (rel, _) in resolved)
        if not has_test:
            findings.append({
                "severity": "blocker",
                "file": "<produces>",
                "why": "required_test_coverage=True but no test file found in produces",
                "adr_id": adr_id,
                "signal_type": "required_test_coverage",
            })

    return findings, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def check_adr_drift(
    adr_register_path: str,
    produced_files: list[str],
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Run the ADR drift gate against *produced_files*.

    Parameters
    ----------
    adr_register_path:
        Path to the ADR register markdown (e.g. ``mission_42/.adr/register.md``).
        Used to enumerate the ``.adr/`` directory. If missing, falls back to
        listing all ``*.json`` files in the adjacent directory.
    produced_files:
        Files emitted by the source task. These are the files checked against
        each ADR's ``falsification_signal``.
    workspace_path:
        Optional workspace root for resolving relative paths and fallback
        ``.adr/`` discovery.

    Returns
    -------
    dict with keys:
        ``verdict``              – ``"pass"`` or ``"fail"``
        ``findings``             – list of mechanical-violation dicts
        ``judgment_only_adr_ids``– ADR ids that need LLM-judge follow-up
        ``skipped``              – True when no ADR data found (soft-pass)
    """
    t0 = time.monotonic()

    adr_files = _enumerate_adr_files(adr_register_path, workspace_path)

    if not adr_files:
        logger.debug(
            "check_adr_drift: no ADR files found — soft-skip",
            register=adr_register_path,
        )
        return {
            "verdict": "pass",
            "findings": [],
            "judgment_only_adr_ids": [],
            "skipped": True,
            "duration_s": round(time.monotonic() - t0, 3),
        }

    all_findings: list[dict] = []
    judgment_only: list[str] = []

    for adr_path in adr_files:
        adr_id = Path(adr_path).stem
        try:
            with open(adr_path, encoding="utf-8") as fh:
                adr_data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "check_adr_drift: cannot load ADR file",
                path=adr_path, error=str(exc),
            )
            continue

        falsification = adr_data.get("falsification_signal")
        findings, is_jo = _check_one_adr(
            adr_id=adr_id,
            falsification=falsification,
            produced_files=produced_files,
            workspace_path=workspace_path,
        )
        all_findings.extend(findings)
        if is_jo:
            judgment_only.append(adr_id)

    blocker_count = sum(1 for f in all_findings if f.get("severity") == "blocker")
    verdict = "fail" if blocker_count > 0 else "pass"

    logger.info(
        "check_adr_drift complete",
        verdict=verdict,
        findings_count=len(all_findings),
        judgment_only_count=len(judgment_only),
        duration_s=round(time.monotonic() - t0, 3),
    )

    return {
        "verdict": verdict,
        "findings": all_findings,
        "judgment_only_adr_ids": judgment_only,
        "skipped": False,
        "duration_s": round(time.monotonic() - t0, 3),
    }
