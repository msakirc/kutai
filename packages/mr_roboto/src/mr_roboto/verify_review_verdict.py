"""Read a reviewer's *_review_result verdict and classify it.

pass-class  -> the reviewer accepted the artifact; the step completes.
fail-class  -> route to general_beckman.review_routing.route_review_failure.
malformed   -> the reviewer task itself failed (no parseable verdict): normal
               DLQ, NOT the routing path.

Two reviewer output shapes are accepted:

1. STATUS shape (the 7 standard reviewers): a top-level
   ``{"status": pass|approved|needs_minor_fixes|fail, "issues": [...]}``.
   ``status`` drives the verdict; ``issues`` rides along to the router.
   Step 1.13 (research_quality_review) carries the verdict under ``verdict``
   instead of ``status`` (its artifact_schema field is ``verdict``) — either
   key is read; malformed only when NEITHER is present.

2. FINDINGS shape (step 10.5 encryption_and_logging_review): the reviewer has
   no pass/fail status. Instead it emits one or two findings arrays
   (``findings: [{check, severity, description, remediation, target_artifact?}]``)
   keyed under its artifact names (e.g. ``encryption_verification_result`` and
   ``logging_security_result``), OR a single top-level ``findings`` array.
   A finding at severity ``critical`` or ``high`` is a blocker → ``fail``;
   anything else (medium/low/info only) → ``pass``. Blocking findings are
   mapped to the standard issue shape the router consumes
   (``{target_artifact, severity, problem}``) so routing works unchanged.
"""
from __future__ import annotations

import os
import re
from typing import Any

from yazbunu import get_logger

logger = get_logger("mr_roboto.verify_review_verdict")

_PASS_CLASS = {"pass", "approved", "needs_minor_fixes"}
_FAIL_CLASS = {"fail"}

# Findings at these severities block the gate (mirrors the artifact_schema
# blockers declaration: {field: severity, levels: [critical, high]}).
_BLOCKING_SEVERITIES = {"critical", "high"}


def _iter_findings(review_result: dict) -> list[dict]:
    """Collect every finding dict from a findings-shape review_result.

    Accepts both a single top-level ``findings`` array and per-artifact
    wrappers (``{<artifact_name>: {"findings": [...]}, ...}``). Non-dict
    findings are ignored. Returns a flat list preserving order.
    """
    out: list[dict] = []
    top = review_result.get("findings")
    if isinstance(top, list):
        out.extend(f for f in top if isinstance(f, dict))
    for value in review_result.values():
        if isinstance(value, dict):
            nested = value.get("findings")
            if isinstance(nested, list):
                out.extend(f for f in nested if isinstance(f, dict))
    return out


def _has_findings_shape(review_result: dict) -> bool:
    """True iff review_result carries a findings array (top-level or nested)
    and no top-level ``status`` (status takes precedence when present)."""
    if "status" in review_result:
        return False
    if isinstance(review_result.get("findings"), list):
        return True
    return any(
        isinstance(v, dict) and isinstance(v.get("findings"), list)
        for v in review_result.values()
    )


def findings_to_issues(findings: list[dict]) -> list[dict]:
    """Map blocking (critical/high) findings to the router's issue shape.

    Each blocking finding becomes ``{target_artifact, severity, problem}``:
      * ``problem``        <- finding ``description``
      * ``severity``       <- finding ``severity`` (critical|high)
      * ``target_artifact``<- finding ``target_artifact`` (None if absent →
                              the router treats it as unresolved → founder-halt)

    Pure / deterministic — non-blocking findings are dropped (they do not
    reject the artifact). Order is preserved.
    """
    issues: list[dict] = []
    for f in findings:
        sev = str(f.get("severity") or "").lower()
        if sev not in _BLOCKING_SEVERITIES:
            continue
        issues.append({
            "target_artifact": f.get("target_artifact"),
            "severity": sev,
            "problem": f.get("description") or f.get("check") or "",
        })
    return issues


# ── Tier-1 deterministic grounding (2026-06-26) ──────────────────────────────
# A single reviewer LLM confabulates findings (invented verbatim quotes,
# rubric-example echoes, false "missing section" claims). Before a `fail` halts
# the mission, each finding is checked against its target artifact and DROPPED
# only when its cited evidence is *provably* not present/absent as claimed.
# HIGH-PRECISION: drop only on certainty; any doubt / unreadable artifact /
# exception → KEEP (never silently drop a real finding, never crash the gate).

# Absence/incompleteness claim markers ("missing X", "lacks X", "X is empty").
_ABSENCE_MARKERS = re.compile(
    r"\b(missing|does not contain|do not contain|doesn't contain|lacks?|absent|"
    r"empty|without|not present|fails? to (?:include|contain)|incomplete|"
    r"no\s+\w)\b",
    re.IGNORECASE,
)

# Quoted spans the model presents as evidence: "...", '...', smart quotes, `...`.
_QUOTE_RE = re.compile(
    r'"([^"\n]{3,200})"'
    r"|'([^'\n]{3,200})'"
    r"|“([^”\n]{3,200})”"
    r"|‘([^’\n]{3,200})’"
    r"|`([^`\n]{3,200})`"
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _extract_quotes(text: str) -> list[str]:
    out: list[str] = []
    for m in _QUOTE_RE.finditer(text):
        span = next((g for g in m.groups() if g), None)
        if span and span.strip():
            out.append(span.strip())
    return out


def _distinctive(span: str) -> bool:
    """A quoted span that is *prose evidence* — text the model claims to have
    read in the artifact, specific enough that its absence is meaningful.

    Require >=6 chars AND a space (multi-word). A single token — a snake_case /
    camelCase identifier, a JSON field name, or a config ENUM value like
    ``public_launch`` / ``graveyard_count`` / ``inconclusive`` — is a
    *reference*, not text claimed to be in the artifact. Treating such a token
    as evidence wrongly drops a genuine structured finding whose quote merely
    names a mission-config value (mission-90 Check 14, Check 17). Erring toward
    KEEP here is correct: a real single-word fabricated quote is rare and merely
    survives to the Tier-2 refuter, whereas dropping a real blocker is the
    failure this whole pass exists to prevent.
    """
    s = span.strip()
    return len(s) >= 6 and " " in s


def _norm_phrase(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _header_title(line: str) -> str | None:
    """Normalized title of a markdown header / bold-label line, else None.

    Handles both ``## Section`` (title = whole line) and ``**Label:** body``
    (title = the label PREFIX + body, matched by prefix below). Leading ``#``,
    ``*`` and numbering (``1.``) are stripped."""
    stripped = line.strip()
    if not (stripped.startswith("#") or stripped.startswith("**")):
        return None
    low = re.sub(r"^[#*\s]+", "", stripped.lower())
    low = re.sub(r"^\d+[.)]\s*", "", low)  # numbered header "1. Landscape"
    return _norm_phrase(low)


def _present_as_section_or_field(content: str, name: str) -> bool:
    """True iff *name* is present as a markdown header / bold label (the name is
    the LEADING phrase of the header/label line), or as a non-empty JSON/YAML
    field key. Used both to confirm a false-absence claim and to exclude
    section-NAME quotes from the evidence set.

    PREFIX match (not loose word-subset): a header that merely CONTAINS the
    section's words elsewhere (``## Important Notes On The Landscape`` for
    "Notes") does NOT count as present — loose subset over-drops a real
    missing-section finding (review MAJOR #2). Prefix also handles inline bold
    labels with trailing body text (``**Guiding principles:** Make streaks…``).
    Erring toward "absent" is safe (Rule A then simply doesn't fire)."""
    nm = _norm_phrase(name)
    if not nm:
        return False
    for line in content.splitlines():
        title = _header_title(line)
        if title is not None and (title == nm or title.startswith(nm + " ")):
            return True
    # JSON/YAML field key with a non-empty value.
    key = "_".join(nm.split())
    m = re.search(
        re.escape(key) + r"\s*[:=]\s*(\[[^\]]*\]|\{[^}]*\}|\"[^\"]*\"|'[^']*'|[^\s,}\]]+)",
        content.lower(),
    )
    if m:
        val = m.group(1).strip()
        if val not in ("[]", "{}", '""', "''", "null", "none", "0", ""):
            return True
    return False


def _enumerated_sections(problem: str) -> list[str]:
    """Section names enumerated in a parenthesized comma list, e.g.
    "(Landscape, Value Thesis, ... , Notes)". Strips leading e.g./such-as."""
    names: list[str] = []
    for m in re.finditer(r"\(([^()]{6,})\)", problem):
        inner = m.group(1)
        parts = [p.strip() for p in inner.split(",")]
        cleaned: list[str] = []
        for p in parts:
            p = re.sub(r'^(?:e\.?g\.?|such as|i\.?e\.?)[,:]?\s*', "", p, flags=re.IGNORECASE)
            p = p.strip().strip('"\'`')
            if p and len(p) >= 3:
                cleaned.append(p)
        if len(cleaned) >= 2:
            names.extend(cleaned)
    return names


# Rule C — falsification-triple absence claims. The reviewer names the triple
# fields / says "triple(s)" / "falsification". Paired with an absence marker,
# this is a presence claim the mechanical verifier owns.
_FALSIFICATION_MARKERS = re.compile(
    r"falsification|risk_if_wrong|validation_method|falsification_signal|"
    r"\btriples?\b",
    re.IGNORECASE,
)

# Quality/adequacy qualifiers. When present, the finding is judging the triple's
# QUALITY (is it specific / observable / testable?) — a legitimate reviewer axis
# Rule C must NOT touch — not its PRESENCE. Keeps Rule C to the confabulated
# "missing / empty" class only (m90 567426 finding [4]: "lack SPECIFIC
# falsification signals" is a specificity judgment, not an absence claim).
_QUALITY_QUALIFIERS = re.compile(
    r"\b(vague|specific|specificity|observable|measurable|testable|generic|"
    r"non-observable|unobservable|insufficient|not sufficient|too broad|"
    r"quantif\w*)\b",
    re.IGNORECASE,
)


def _split_table_row(line: str) -> list[str] | None:
    """A markdown table row's cells (``| a | b |`` → ``['a','b']``), else None."""
    s = line.strip()
    if not (s.startswith("|") and s.count("|") >= 2):
        return None
    return [c.strip() for c in s.strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and "-" in s and bool(re.match(r"^\|?[\s:|-]+\|?$", s))


def _triple_column_index(header_cells: list[str]) -> dict[str, int] | None:
    """Map a requirement table's triple columns to indices, or None.

    Requires all three of risk / validation / falsification-signal columns.
    Also records an id column when present (first ``id``/``req``-ish column)."""
    norm = [_norm_phrase(c) for c in header_cells]
    idx: dict[str, int] = {}
    wanted = (
        ("risk_if_wrong", ("risk",)),
        ("validation_method", ("validation",)),
        ("falsification_signal", ("falsification",)),
    )
    for field, keys in wanted:
        for ci, name in enumerate(norm):
            if any(k in name for k in keys):
                idx[field] = ci
                break
    if not all(f in idx for f in ("risk_if_wrong", "validation_method", "falsification_signal")):
        return None
    for ci, name in enumerate(norm):
        if name == "id" or name.endswith(" id") or "req" in name:
            idx["id"] = ci
            break
    return idx


def _parse_spec_requirement_triples(content: str) -> list[dict]:
    """Parse markdown requirement tables into triple-bearing item dicts.

    Returns one dict (``req_id`` + the three triple fields) per data row of any
    table whose header names the risk / validation / falsification-signal
    columns — the shape ``verify_falsification_present`` validates. Empty list
    when no such table exists (caller then cannot prove presence → keeps)."""
    items: list[dict] = []
    lines = content.splitlines()
    i, n = 0, len(lines)
    while i < n:
        header = _split_table_row(lines[i])
        if header and i + 1 < n and _is_table_separator(lines[i + 1]):
            idx = _triple_column_index(header)
            if idx:
                j = i + 2
                while j < n:
                    row = _split_table_row(lines[j])
                    if row is None:
                        break
                    if _is_table_separator(lines[j]):
                        j += 1
                        continue
                    item: dict = {}
                    if idx.get("id") is not None and idx["id"] < len(row):
                        item["req_id"] = row[idx["id"]]
                    for f in ("risk_if_wrong", "validation_method", "falsification_signal"):
                        ci = idx[f]
                        item[f] = row[ci] if ci < len(row) else ""
                    items.append(item)
                    j += 1
                i = j
                continue
        i += 1
    return items


def _falsification_presence_proven(artifact_content: str) -> bool:
    """True iff the spec's requirement tables populate every triple column.

    Deterministic proof that a reviewer 'missing triple / empty table' claim is
    confabulated: parse the tables and run the SAME mechanical checker the
    producers hard-gate on. ``missing`` empty AND not ``empty`` = presence
    proven. Ignores ``critical_underspecified`` — that is a QUALITY axis (a real
    reviewer finding), not a presence claim, so it must not block the drop."""
    items = _parse_spec_requirement_triples(artifact_content)
    if not items:
        return False
    try:
        from mr_roboto.verify_falsification_present import verify_falsification_present
        res = verify_falsification_present(artifacts={"spec": items})
    except Exception:  # noqa: BLE001 — grounding must never crash the gate
        return False
    return not res.get("missing") and not res.get("empty")


def classify_issue_grounding(problem: str, artifact_content: str | None) -> str:
    """Deterministically ground one reviewer finding against its artifact.

    Returns one of:
      * ``"drop"``             — confabulation proven (false-absence claim whose
                                 sections are all present, OR an evidence quote
                                 the artifact does not contain).
      * ``"keep_confirmed"``   — the finding's quoted evidence IS present in the
                                 artifact — it is grounded, keep without Tier 2.
      * ``"keep_unverifiable"``— no deterministic signal (no checkable quote or
                                 section). Keep; a blocking one is a Tier-2
                                 refuter candidate.

    Fail-safe: unreadable artifact / empty problem / any ambiguity → keep.
    """
    try:
        if not isinstance(problem, str) or not problem.strip():
            return "keep_unverifiable"
        if not isinstance(artifact_content, str) or not artifact_content.strip():
            return "keep_unverifiable"

        has_absence = bool(_ABSENCE_MARKERS.search(problem))

        # Rule A — false-absence: the finding claims a set of sections is missing,
        # but EVERY enumerated section is actually present. (>=3 to be safe.)
        sections = _enumerated_sections(problem)
        if has_absence and len(sections) >= 3:
            if all(_present_as_section_or_field(artifact_content, s) for s in sections):
                return "drop"

        # Rule C — false falsification-triple absence: the finding claims a
        # requirement's triple is missing / the requirement table is empty, but
        # the requirements_spec table populates the risk_if_wrong /
        # validation_method / falsification_signal columns for every row.
        # Presence is a mechanical fact the producers already hard-gate
        # (verify_falsification_present on 3.1/3.2/3.3/3.7); Rule A can't see it
        # because the fields are nested table CELLS, not headers/JSON keys
        # (m90 567426). Quality axes (vague critical validation) are separate
        # findings and untouched — only the confabulated ABSENCE claim is dropped.
        if (
            has_absence
            and _FALSIFICATION_MARKERS.search(problem)
            and not _QUALITY_QUALIFIERS.search(problem)
        ):
            if _falsification_presence_proven(artifact_content):
                return "drop"

        # Rule B — fabricated quote: the finding embeds distinctive evidence
        # quotes (excluding ones that are present section/field NAMES, which are
        # references). If NONE of the evidence quotes appear → fabricated → drop;
        # all present → confirmed; mixed → unverifiable (Tier 2 decides).
        quotes = [q for q in _extract_quotes(problem) if _distinctive(q)]
        evidence = [
            q for q in quotes if not _present_as_section_or_field(artifact_content, q)
        ]
        if evidence:
            norm_content = _norm(artifact_content)
            present = [q for q in evidence if _norm(q) in norm_content]
            if not present:
                return "drop"
            if len(present) == len(evidence):
                return "keep_confirmed"
            return "keep_unverifiable"

        return "keep_unverifiable"
    except Exception:  # noqa: BLE001 — grounding must never crash the gate
        return "keep_unverifiable"


# Explicitly-minor severities — the ONLY ones that let a survived `fail`
# downgrade. Anything else (blocker/major/critical/high OR a missing/unknown
# severity) is treated as blocking: we re-derive `pass` only when confabulated
# findings were dropped, never on a severity technicality.
_NON_BLOCKING_SEVERITIES = {"minor", "medium", "low", "info", "trivial", "nit"}


def _is_blocking_issue(issue: dict) -> bool:
    return str(issue.get("severity") or "").lower() not in _NON_BLOCKING_SEVERITIES


async def _maybe_await(value):
    import inspect
    if inspect.isawaitable(value):
        return await value
    return value


def _strip_ext(name: str) -> str:
    return re.sub(r"\.[a-z0-9]+$", "", name, flags=re.IGNORECASE)


def _read_from_workspace(mission_id, target_artifact: str) -> str | None:
    """Bounded basename match under the mission workspace.

    Reviewer findings name a bare filename (``competitive_positioning.md``) but
    artifacts live in dotted subdirs (``.charter/`` ``.prd/`` ``.research/``
    ``.intake/``) or at the root. Walk the workspace ROOT + only dotted subdirs
    (never the generated code tree) and return the first file whose basename
    (or stem) matches. Disk = canonical truth."""
    try:
        from src.tools.workspace import get_mission_workspace
        ws = str(get_mission_workspace(mission_id))
    except Exception:  # noqa: BLE001
        return None
    if not ws or not os.path.isdir(ws):
        return None
    basename = os.path.basename(target_artifact.replace("\\", "/"))
    if not basename:
        return None
    want = basename.lower()
    want_stem = _strip_ext(want)
    for root, dirs, files in os.walk(ws):
        # Prune: at the workspace top level descend ONLY into dotted artifact
        # dirs (.charter/.prd/.research/.intake/.adr). Skip the generated code
        # tree (backend/, app/, …) so a same-named code file can't shadow the
        # real artifact and the walk stays cheap.
        if os.path.realpath(root) == os.path.realpath(ws):
            dirs[:] = [d for d in dirs if d.startswith(".")]
        for f in files:
            fl = f.lower()
            if fl == want or _strip_ext(fl) == want_stem:
                try:
                    with open(os.path.join(root, f), encoding="utf-8") as fh:
                        return fh.read()
                except Exception:  # noqa: BLE001
                    continue
    return None


async def _resolve_artifact_content(mission_id, target_artifact) -> str | None:
    """Production resolver: ``target_artifact`` filename -> on-disk content.

    Disk is the canonical truth — the reviewer's input artifacts are
    materialized to ``<mission_workspace>/...`` by the produces pipeline. We do
    NOT fall back to the ArtifactStore (a blackboard read can open the prod DB
    and is unnecessary — disk is the single source of truth). Returns None when
    the file does not resolve; the grounding then KEEPS the finding and routes
    it normally (never drops, never sends empty content to the refuter)."""
    if not target_artifact or not isinstance(target_artifact, str):
        return None
    content = _read_from_workspace(mission_id, target_artifact)
    if content and content.strip():
        return content
    return None


def make_disk_resolver(mission_id):
    """Bind the production disk/store resolver to a mission for ground_review_verdict."""
    async def _resolve(target_artifact):
        return await _resolve_artifact_content(mission_id, target_artifact)
    return _resolve


async def ground_review_verdict(
    *, review_result: Any, resolve_artifact
) -> dict[str, Any]:
    """Tier-1 grounding over a whole reviewer verdict.

    Classifies the verdict, and — ONLY when it is ``fail`` (only ``fail``
    halts) — grounds every issue against its ``target_artifact`` via the
    ``resolve_artifact(name)`` callable (sync or async, returns content or
    None). Confabulated issues (``classify_issue_grounding`` == "drop") are
    removed; the verdict is re-derived from the survivors (no blocking issue
    left → ``pass``). Unverifiable-but-blocking survivors are surfaced as
    Tier-2 refuter candidates.

    Returns ``{ok, verdict_class, issues, dropped, tier2_candidates}``.
    Fail-safe throughout — a resolver error or any exception keeps the issue
    (never silently drops a real finding, never crashes the gate).
    """
    base = verify_review_verdict(review_result=review_result)
    if base.get("verdict_class") != "fail":
        return {**base, "dropped": [], "tier2_candidates": []}

    issues = base.get("issues") or []
    kept: list[dict] = []
    dropped: list[dict] = []
    tier2: list[dict] = []

    for issue in issues:
        if not isinstance(issue, dict):
            kept.append(issue)
            continue
        target = issue.get("target_artifact")
        problem = str(issue.get("problem") or "")
        content = None
        try:
            content = await _maybe_await(resolve_artifact(target))
        except Exception:  # noqa: BLE001 — resolver failure must never drop
            content = None

        decision = classify_issue_grounding(problem, content)
        if decision == "drop":
            dropped.append({**issue, "_drop_reason": "cited evidence not found in artifact"})
            logger.info(
                "[verdict-verify] dropped finding — evidence not found",
                target_artifact=target, problem=problem[:160],
            )
            continue

        kept.append(issue)
        # A Tier-2 refuter candidate requires that the artifact actually
        # RESOLVED — if we could not load it, Tier-1 could not ground it and the
        # refuter could not judge it either (it would see empty content and
        # wrongly mark a real finding UNSUPPORTED). Keep it and route normally.
        if (
            decision == "keep_unverifiable"
            and _is_blocking_issue(issue)
            and isinstance(content, str)
            and content.strip()
        ):
            tier2.append(issue)

    blocking_left = any(_is_blocking_issue(i) for i in kept if isinstance(i, dict))
    verdict_class = "fail" if blocking_left else "pass"
    return {
        "ok": verdict_class != "fail",
        "verdict_class": verdict_class,
        "issues": kept,
        "dropped": dropped,
        "tier2_candidates": tier2,
    }


def verify_review_verdict(*, review_result: Any) -> dict[str, Any]:
    if not isinstance(review_result, dict):
        return {"ok": False, "verdict_class": "malformed",
                "error": "no parseable review verdict", "issues": []}

    # FINDINGS shape (10.5): no status, but one or more findings arrays.
    if _has_findings_shape(review_result):
        findings = _iter_findings(review_result)
        blocking = findings_to_issues(findings)
        if blocking:
            return {"ok": False, "verdict_class": "fail", "issues": blocking}
        return {"ok": True, "verdict_class": "pass", "issues": []}

    # STATUS shape (the 7 standard reviewers use `status`; step 1.13
    # research_quality_review uses `verdict`). Accept either key — malformed
    # only when NEITHER is present (and it isn't a findings shape).
    if "status" not in review_result and "verdict" not in review_result:
        return {"ok": False, "verdict_class": "malformed",
                "error": "no parseable review verdict", "issues": []}
    status = str(review_result.get("status") or review_result.get("verdict") or "").lower()
    issues = review_result.get("issues") or []
    if status in _FAIL_CLASS:
        return {"ok": False, "verdict_class": "fail", "issues": issues}
    if status in _PASS_CLASS:
        return {"ok": True, "verdict_class": "pass", "issues": issues}
    return {"ok": False, "verdict_class": "malformed",
            "error": f"unknown verdict status {status!r}", "issues": issues}
