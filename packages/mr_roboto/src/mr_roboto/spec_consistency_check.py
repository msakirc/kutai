"""Spec consistency check — Z1 Tier 5 (T5B / B5).

Augment Intent's "specs stay alive" pattern: every phase 7+ wave starts
with a mechanical re-validation that compares the locked phase-≤6 spec
artifacts against the current phase-N artifacts and surfaces drift.

Drift = a phase-N artifact contradicts a phase-≤6 spec commitment.

This is a RULE-BASED scan (not LLM). Rationale:
  1. Speed: runs at the head of every wave; cheap.
  2. Determinism: drift catalogue must be the same across runs so the
     reviewer can ack a stable list.
  3. Scope: the LLM reviewer at 6.6 already has eyes; B5's job is to make
     drift visible, not to judge.

Detection rules (v1, conservative — false negatives preferred over false
positives):

  R1. Stack drift: any phase-N file mentions a tech keyword (e.g. a
      framework name) that is explicitly excluded by a phase-4 ADR's
      ``status: rejected`` option or by the mission-wide non_goals.md.

  R2. Token drift: any phase-N CSS / Tailwind config references a hex
      color or font-family token NOT present in design_tokens.json.

  R3. Surface drift: any phase-N route / screen file uses a surface
      (mobile/web/desktop) not declared in surfaces.md.

  R4. Non-goal drift: any phase-N artifact (instruction or code) includes
      a substring that overlaps with a non_goals.md bullet (>= 2 distinct
      tokens of length >= 4).

  R5. Charter brand drift: any phase-N copy doc contradicts the brand
      keywords (uses an explicitly excluded brand keyword from charter
      "Brand Keywords" section).

When the spec artifact is missing on disk (mission predates B5 / phase 6
incomplete), we fail-soft: emit a "spec_artifact_missing" warning and
return ok=True (no drift to detect). The wave-start step in i2p_v3.json
gates with skip_when=legacy_pre_spec_alive=='1'.

Output: ``mission_<id>/spec_drift_report.md`` containing both the JSON
envelope and a human-readable bullet list. Schema:

    {
      "_schema_version": "1",
      "drift_items": [
        {"phase": "phase_N",
         "artifact": "<relative_path>",
         "conflict": "<rule_id>: <one_line_human>",
         "suggested_resolution": "<short_action>"}
      ]
    }
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


_HEX_RE = re.compile(r"#([0-9a-fA-F]{3,8})\b")
_TAG_RE = re.compile(r"^\s*[-*]\s+(.*?)\s*$", re.MULTILINE)
_BULLET_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")


def _resolve_workspace_root(workspace_path: str | None, mission_id: str | int) -> Path:
    """Return the workspace root that contains ``mission_<id>/``."""
    if workspace_path:
        return Path(workspace_path)
    # The expander does NOT wire workspace_path into the mechanical payload
    # (the writer/coulson path resolves WORKSPACE_DIR itself), so fall back to
    # the canonical WORKSPACE_DIR base — the same root used to build
    # ``mission_<id>/``. The old ``Path.cwd()`` default pointed at the server's
    # arbitrary working dir, so every consistency check found no mission dir and
    # passed vacuously (mission #71, first e2e run, 2026-05-22).
    try:
        from src.tools.workspace import WORKSPACE_DIR
        return Path(WORKSPACE_DIR)
    except Exception:
        return Path.cwd()


def _read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def _gather_spec_files(mission_dir: Path) -> dict[str, list[Path]]:
    """Collect locked phase-≤6 spec artifacts. Returns {kind: [paths]}.

    Conservative: only known relative subpaths.
    """
    out: dict[str, list[Path]] = {
        "charter": [],
        "non_goals": [],
        "design_tokens": [],
        "surfaces": [],
        "screen_plans": [],
        "adrs": [],
        "premortem": [],
        "compliance": [],
    }
    if not mission_dir.exists():
        return out

    charter = mission_dir / ".charter" / "product_charter.md"
    if charter.exists():
        out["charter"].append(charter)

    non_goals = mission_dir / "non_goals.md"
    if non_goals.exists():
        out["non_goals"].append(non_goals)

    tokens = mission_dir / ".style" / "design_tokens.json"
    if tokens.exists():
        out["design_tokens"].append(tokens)

    surfaces = mission_dir / "surfaces.md"
    if surfaces.exists():
        out["surfaces"].append(surfaces)

    screens_dir = mission_dir / ".screens"
    if screens_dir.exists():
        for plan in screens_dir.rglob("screen_plan.md"):
            out["screen_plans"].append(plan)

    adrs_dir = mission_dir / ".adrs"
    if adrs_dir.exists():
        for adr in adrs_dir.rglob("*.json"):
            out["adrs"].append(adr)

    premortem = mission_dir / "premortem.md"
    if premortem.exists():
        out["premortem"].append(premortem)
    else:
        # Also accept premortem.json envelope
        pj = mission_dir / "premortem.json"
        if pj.exists():
            out["premortem"].append(pj)

    compliance = mission_dir / "compliance_overlay.md"
    if compliance.exists():
        out["compliance"].append(compliance)

    return out


_SPEC_FILENAMES = frozenset({
    "product_charter.md",
    "non_goals.md",
    "surfaces.md",
    "premortem.md",
    "premortem.json",
    "compliance_overlay.md",
    "design_tokens.json",
    "spec_drift_report.md",
    "user_flow.md",
    "shared_shell.md",
    "screen_inventory.md",
    "screen_plan.md",
})

_SPEC_DIRS = frozenset({
    ".charter",
    ".style",
    ".adrs",
    ".screens",
    ".compliance",
})


def _is_spec_file(p: Path, mission_dir: Path) -> bool:
    """True when this path is a phase-≤6 locked spec artifact."""
    if p.name in _SPEC_FILENAMES:
        return True
    try:
        rel = p.relative_to(mission_dir)
    except ValueError:
        return False
    parts = rel.parts
    if parts and parts[0] in _SPEC_DIRS:
        return True
    return False


def _gather_phase_n_files(mission_dir: Path, current_phase: str) -> list[Path]:
    """Return files plausibly emitted by the current phase.

    Heuristic: any file under mission_dir whose path contains the phase
    number; OR any file under standard phase-N directories. We're
    intentionally conservative — false negatives are fine. Locked spec
    artifacts (charter / non_goals / design_tokens / etc.) are excluded.
    """
    out: list[Path] = []
    if not mission_dir.exists():
        return out

    # Phase number e.g. "phase_8" -> "8"
    m = re.search(r"(\d+)$", current_phase or "")
    if not m:
        return out
    n = m.group(1)

    candidate_globs = [
        f".phase_{n}/**/*",
        f"phase_{n}/**/*",
        # Generic code/doc surfaces likely produced phase 7+.
        "backend/**/*.py",
        "backend/**/*.md",
        "frontend/**/*.tsx",
        "frontend/**/*.ts",
        "frontend/**/*.css",
        "frontend/**/*.md",
        "src/**/*.py",
        "src/**/*.md",
    ]
    seen: set[Path] = set()
    for glob in candidate_globs:
        for p in mission_dir.glob(glob):
            if p.is_file() and p not in seen and not _is_spec_file(p, mission_dir):
                seen.add(p)
                out.append(p)
    return out


def _extract_design_tokens(tokens_path: Path) -> dict[str, set[str]]:
    """Return {kind: {value}} for tokens we care about (colors, fonts)."""
    out: dict[str, set[str]] = {"colors": set(), "fonts": set()}
    txt = _read_text(tokens_path)
    if not txt:
        return out
    try:
        data = json.loads(txt)
    except Exception:
        return out

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str):
                    if _HEX_RE.search(v):
                        for hexm in _HEX_RE.finditer(v):
                            out["colors"].add("#" + hexm.group(1).lower())
                    if "font" in str(k).lower() or "family" in str(k).lower():
                        out["fonts"].add(v.strip().lower())
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    return out


def _extract_non_goals_tokens(non_goals_path: Path) -> list[set[str]]:
    """Return per-bullet token sets (lowercased)."""
    out: list[set[str]] = []
    txt = _read_text(non_goals_path) or ""
    for m in _TAG_RE.finditer(txt):
        bullet = m.group(1)
        toks = {
            t.lower()
            for t in _BULLET_TOKEN_RE.findall(bullet)
            if len(t) >= 4
        }
        if toks:
            out.append(toks)
    return out


def _extract_surfaces(surfaces_path: Path) -> set[str]:
    """Return the set of declared surfaces from surfaces.md (lowercased)."""
    out: set[str] = set()
    txt = (_read_text(surfaces_path) or "").lower()
    for kw in ("mobile", "web", "desktop", "ios", "android"):
        if re.search(rf"\b{kw}\b", txt):
            out.add(kw)
    return out


_BRAND_HEADER_RE = re.compile(
    r"^#{1,6}\s*(?:🎯\s*)?brand\s*keywords?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_EXCLUDED_HEADER_RE = re.compile(
    r"^\s*[-*]?\s*(?:excluded|avoid|do not use|forbidden)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _extract_brand_excluded(charter_paths: list[Path]) -> set[str]:
    """Read charter "Brand Keywords" section; return excluded brand tokens.

    Pattern accepted:
        ## Brand Keywords
        - Excluded: foo, bar, baz
        - Avoid: corporate, enterprise
    Tokens are lowercased + stripped of whitespace.
    """
    excluded: set[str] = set()
    for p in charter_paths:
        body = _read_text(p)
        if not body:
            continue
        m = _BRAND_HEADER_RE.search(body)
        if not m:
            continue
        # Take everything until the next heading at <= same level OR EOF.
        section = body[m.end():]
        next_header = re.search(r"\n#{1,6}\s+\S", section)
        if next_header:
            section = section[:next_header.start()]
        for line in section.splitlines():
            ex = _EXCLUDED_HEADER_RE.match(line)
            if not ex:
                continue
            tail = line[ex.end():]
            for tok in re.split(r"[,;]\s*|\s+\bor\b\s+", tail):
                tok = tok.strip().strip(".").lower()
                if len(tok) >= 3 and tok.isascii():
                    excluded.add(tok)
    return excluded


def _extract_compliance_forbidden(compliance_paths: list[Path]) -> set[str]:
    """Read `compliance_overlay.json` and surface explicit forbidden tokens.

    Accepted keys (compliance_overlay schema is loose so we tolerate both
    shapes):
      * top-level ``forbidden_third_parties``: list[str]
      * each ``required_documents[i].forbidden_keywords``: list[str]
    Tokens are lowercased.
    """
    out: set[str] = set()
    for p in compliance_paths:
        # _gather_spec_files collects compliance_overlay.md; the JSON sibling
        # has the structured payload — try that too.
        candidates = [p]
        if p.suffix == ".md":
            sib = p.with_suffix(".json")
            if sib.exists():
                candidates.append(sib)
        for c in candidates:
            if c.suffix != ".json":
                continue
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                continue
            for k in data.get("forbidden_third_parties") or []:
                if isinstance(k, str) and len(k) >= 3:
                    out.add(k.strip().lower())
            for doc in data.get("required_documents") or []:
                for k in (doc or {}).get("forbidden_keywords") or []:
                    if isinstance(k, str) and len(k) >= 3:
                        out.add(k.strip().lower())
    return out


def _extract_rejected_tech(adr_paths: list[Path]) -> set[str]:
    """Return tech keywords from ADR options marked as rejected."""
    out: set[str] = set()
    for p in adr_paths:
        txt = _read_text(p)
        if not txt:
            continue
        try:
            obj = json.loads(txt)
        except Exception:
            continue
        # Walk options; gather tech labels of options whose status is rejected.
        opts = obj.get("options") or []
        if isinstance(opts, list):
            for opt in opts:
                if not isinstance(opt, dict):
                    continue
                status = str(opt.get("status") or "").lower()
                if status == "rejected":
                    name = opt.get("name") or opt.get("label") or opt.get("id")
                    if isinstance(name, str) and name.strip():
                        out.add(name.strip().lower())
    return out


_COPY_DOC_SUFFIXES = (".md", ".mdx", ".txt", ".html", ".htm")
_CODE_DOC_SUFFIXES = (
    ".py", ".js", ".ts", ".jsx", ".tsx", ".vue", ".rs", ".go", ".java",
    ".kt", ".swift", ".json", ".yml", ".yaml",
)


def _scan_phase_files(
    files: list[Path],
    mission_dir: Path,
    tokens: dict[str, set[str]],
    non_goal_buckets: list[set[str]],
    surfaces: set[str],
    rejected_tech: set[str],
    current_phase: str,
    brand_excluded: set[str] | None = None,
    compliance_forbidden: set[str] | None = None,
) -> list[dict[str, str]]:
    """Apply rules R1-R6 to phase-N files. Returns drift_items.

    R5 = brand drift (copy docs only).
    R6 = compliance forbidden tokens (copy + code).
    """
    brand_excluded = brand_excluded or set()
    compliance_forbidden = compliance_forbidden or set()
    drift: list[dict[str, str]] = []
    surface_keywords = ("mobile", "web", "desktop")
    for p in files:
        txt = _read_text(p)
        if not txt:
            continue
        rel = str(p.relative_to(mission_dir)) if mission_dir in p.parents else str(p)

        # R1 — rejected tech keywords
        if rejected_tech:
            low = txt.lower()
            for tech in rejected_tech:
                # word-boundary match to avoid spurious substring hits
                if re.search(rf"\b{re.escape(tech)}\b", low):
                    drift.append({
                        "phase": current_phase,
                        "artifact": rel,
                        "conflict": (
                            f"R1 stack_drift: references rejected tech "
                            f"{tech!r} from phase-4 ADR"
                        ),
                        "suggested_resolution": (
                            f"Remove {tech!r} or amend the ADR to accept it."
                        ),
                    })

        # R2 — token drift (CSS / Tailwind only)
        if p.suffix in (".css", ".scss") or "tokens.css" in p.name.lower():
            for hexm in _HEX_RE.finditer(txt):
                hex_val = "#" + hexm.group(1).lower()
                if tokens["colors"] and hex_val not in tokens["colors"]:
                    drift.append({
                        "phase": current_phase,
                        "artifact": rel,
                        "conflict": (
                            f"R2 token_drift: color {hex_val} not in "
                            f"design_tokens.json"
                        ),
                        "suggested_resolution": (
                            f"Add {hex_val} to design_tokens.json or replace "
                            f"with an existing token."
                        ),
                    })
                    break  # one drift per file is enough signal

        # R3 — surface drift
        if surfaces:
            low = txt.lower()
            for kw in surface_keywords:
                if re.search(rf"\b{kw}\b", low) and kw not in surfaces:
                    drift.append({
                        "phase": current_phase,
                        "artifact": rel,
                        "conflict": (
                            f"R3 surface_drift: mentions {kw!r} surface but "
                            f"not in surfaces.md (declared: "
                            f"{sorted(surfaces)})"
                        ),
                        "suggested_resolution": (
                            f"Either add {kw!r} to surfaces.md or remove "
                            f"the reference."
                        ),
                    })
                    break

        # R5 — brand drift (copy docs only)
        if brand_excluded and p.suffix.lower() in _COPY_DOC_SUFFIXES:
            low = txt.lower()
            for kw in brand_excluded:
                if re.search(rf"\b{re.escape(kw)}\b", low):
                    drift.append({
                        "phase": current_phase,
                        "artifact": rel,
                        "conflict": (
                            f"R5 brand_drift: copy uses excluded brand "
                            f"keyword {kw!r} (charter 'Brand Keywords' → "
                            f"Excluded)"
                        ),
                        "suggested_resolution": (
                            f"Rephrase to avoid {kw!r}, or amend the "
                            f"charter's brand-keyword exclusions."
                        ),
                    })
                    break

        # R6 — compliance forbidden tokens (copy + code; broader scope)
        if compliance_forbidden and p.suffix.lower() in (
            _COPY_DOC_SUFFIXES + _CODE_DOC_SUFFIXES
        ):
            low = txt.lower()
            for kw in compliance_forbidden:
                if re.search(rf"\b{re.escape(kw)}\b", low):
                    drift.append({
                        "phase": current_phase,
                        "artifact": rel,
                        "conflict": (
                            f"R6 compliance_drift: references compliance-"
                            f"forbidden token {kw!r} from compliance_overlay"
                        ),
                        "suggested_resolution": (
                            f"Remove the reference or amend "
                            f"compliance_overlay.json."
                        ),
                    })
                    break

        # R4 — non-goal drift
        if non_goal_buckets:
            file_tokens = {
                t.lower()
                for t in _BULLET_TOKEN_RE.findall(txt)
                if len(t) >= 4
            }
            for bucket in non_goal_buckets:
                overlap = bucket & file_tokens
                if len(overlap) >= 2:
                    drift.append({
                        "phase": current_phase,
                        "artifact": rel,
                        "conflict": (
                            f"R4 non_goal_drift: overlaps non_goals "
                            f"bullet on tokens {sorted(overlap)[:4]}"
                        ),
                        "suggested_resolution": (
                            "Confirm with founder this isn't a forbidden "
                            "scope expansion or amend non_goals.md."
                        ),
                    })
                    break

    return drift


def _render_report(env: dict[str, Any]) -> str:
    lines = [
        "# Spec drift report",
        "",
        f"_schema_version: {env.get('_schema_version')!r}_",
        "",
    ]
    items = env.get("drift_items") or []
    if not items:
        lines.append("No drift detected.")
    else:
        for it in items:
            lines.append(
                f"- [{it.get('phase')}] `{it.get('artifact')}` — "
                f"{it.get('conflict')}"
            )
            lines.append(f"    - resolution: {it.get('suggested_resolution')}")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(env, indent=2, sort_keys=False))
    lines.append("```")
    return "\n".join(lines) + "\n"


def spec_consistency_check(
    *,
    mission_id: str | int,
    current_phase: str,
    workspace_path: str | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Re-read phase-≤6 spec + phase-N artifacts; surface drift.

    Returns dict with keys: ok, drift_items (list), report_path,
    spec_artifacts_present (list), warnings (list).
    """
    root = _resolve_workspace_root(workspace_path, mission_id)
    mission_dir = root / f"mission_{mission_id}"

    warnings: list[str] = []
    if not mission_dir.exists():
        warnings.append(f"mission_dir_missing:{mission_dir}")
        env = {"_schema_version": "1", "drift_items": []}
        return {
            "ok": True,
            "drift_items": [],
            "report_path": None,
            "spec_artifacts_present": [],
            "warnings": warnings,
            "envelope": env,
        }

    spec = _gather_spec_files(mission_dir)
    spec_present = [k for k, v in spec.items() if v]

    # Fail-soft: charter is the minimum spec. If absent, treat as no drift
    # and warn; the wave-start step's skip_when (legacy_pre_spec_alive)
    # already guards old missions.
    if not spec["charter"]:
        warnings.append("spec_artifact_missing:charter")

    tokens = (
        _extract_design_tokens(spec["design_tokens"][0])
        if spec["design_tokens"]
        else {"colors": set(), "fonts": set()}
    )
    non_goal_buckets = (
        _extract_non_goals_tokens(spec["non_goals"][0])
        if spec["non_goals"]
        else []
    )
    surfaces = (
        _extract_surfaces(spec["surfaces"][0])
        if spec["surfaces"]
        else set()
    )
    rejected_tech = _extract_rejected_tech(spec["adrs"])
    brand_excluded = _extract_brand_excluded(spec["charter"])
    compliance_forbidden = _extract_compliance_forbidden(spec["compliance"])

    phase_files = _gather_phase_n_files(mission_dir, current_phase)
    drift_items = _scan_phase_files(
        phase_files,
        mission_dir,
        tokens,
        non_goal_buckets,
        surfaces,
        rejected_tech,
        current_phase,
        brand_excluded=brand_excluded,
        compliance_forbidden=compliance_forbidden,
    )

    env = {
        "_schema_version": "1",
        "drift_items": drift_items,
    }

    report_path = out_path or str(mission_dir / "spec_drift_report.md")
    try:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(_render_report(env), encoding="utf-8")
    except OSError as e:
        warnings.append(f"write_error:{e}")

    return {
        "ok": len(drift_items) == 0,
        "drift_items": drift_items,
        "report_path": report_path,
        "spec_artifacts_present": spec_present,
        "warnings": warnings,
        "envelope": env,
    }
