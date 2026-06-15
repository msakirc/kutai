"""Z1 Tier 6A (P9) — cross-mission ADR + compliance inheritance hints.

Post-hook on step ``0.5 founder_clarification``. Reads the current
mission's ``compliance_fingerprint`` + charter ``domain_keywords`` (rule-
based extraction from the charter brand keywords), queries
``mission_artifacts_index`` for the most-recent ADRs and compliance
fingerprints from the same ``founder_id`` whose ``domain_keywords_json``
overlaps the current mission's keywords (Jaccard > 0.3), and emits
``mission_<id>/prior_mission_hints.md`` for Telegram review with
"reuse" / "diverge" buttons per hint.

No schema is enforced on the founder's response — these are advisory.
The choices end up as a free-text annotation alongside the hint file.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.surface_prior_mission_hints")

DEFAULT_JACCARD_THRESHOLD = 0.3
TARGET_ARTIFACT_NAMES = ("compliance_fingerprint", "adr_register", "tech_stack_adr")


def _resolve_workspace(mission_id: int, workspace_path: str | None) -> str:
    if workspace_path:
        return workspace_path
    from src.tools.workspace import get_mission_workspace
    return get_mission_workspace(int(mission_id))


# Rule-based domain keyword extraction — picks brand-keyword bullets out
# of a charter and any "## Brand Keywords" / problem-statement section.
# Chosen over spaCy/LLM for cold-path speed (this runs as a post-hook on
# every clarify dispatch and must not block on a model load).
_BRAND_BULLET = re.compile(
    r"^\s*[-*]\s+\*\*([^*]+?)\*\*", re.MULTILINE,
)
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _extract_domain_keywords(text: str) -> list[str]:
    if not text:
        return []
    keywords: list[str] = []

    # Pull bolded bullets (the charter's `Brand Keywords` shape).
    for m in _BRAND_BULLET.finditer(text):
        kw = m.group(1).strip().lower()
        if kw and kw not in keywords:
            keywords.append(kw)

    # Pull heading words as a fallback when no brand keywords were found.
    if not keywords:
        for m in _HEADING.finditer(text):
            for tok in re.split(r"\W+", m.group(1).strip().lower()):
                if len(tok) >= 4 and tok not in keywords:
                    keywords.append(tok)
            if len(keywords) >= 8:
                break

    # Cap; the index column is JSON-serialised TEXT.
    return keywords[:16]


def _load_compliance_fingerprint(workspace_path: str) -> dict[str, Any] | None:
    path = os.path.join(workspace_path, "compliance_fingerprint.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning(
            "surface_prior_mission_hints: cf read failed %s: %s", path, e,
        )
        return None


def _load_charter_text(workspace_path: str) -> str | None:
    candidates = [
        os.path.join(workspace_path, ".charter", "product_charter.md"),
        os.path.join(workspace_path, "idea_brief.md"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return fh.read()
            except Exception as e:
                logger.warning(
                    "surface_prior_mission_hints: charter read failed %s: %s",
                    path, e,
                )
    return None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


async def surface_prior_mission_hints(
    mission_id: int,
    *,
    workspace_path: str | None = None,
    founder_id: str = "default",
    top_k: int = 3,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
) -> dict[str, Any]:
    """Surface prior-mission hints (ADR + compliance) for founder review.

    Returns dict shaped::

        {
            "ok": True,
            "hints": [
                {"prior_mission_id": int,
                 "artifact_name": str,
                 "artifact_path": str,
                 "schema_version": str | None,
                 "overlap_keywords": [str, ...],
                 "jaccard": float},
                ...
            ],
            "report_path": str | None,
            "current_keywords": [str, ...],
            "checked": bool,
        }

    Always returns ``ok=True`` — these are advisory hints, never
    blocking. ``hints`` is empty when no priors match (or the index
    table is empty).
    """
    ws = _resolve_workspace(mission_id, workspace_path)

    charter_text = _load_charter_text(ws) or ""
    fingerprint = _load_compliance_fingerprint(ws) or {}
    current_keywords: list[str] = list(_extract_domain_keywords(charter_text))

    # Compliance-fingerprint jurisdictions widen the keyword set.
    for jur in (fingerprint.get("jurisdictions") or []):
        s = str(jur).strip().lower()
        if s and s not in current_keywords:
            current_keywords.append(s)
    for cat in (fingerprint.get("data_categories_coarse") or []):
        s = str(cat).strip().lower()
        if s and s not in current_keywords:
            current_keywords.append(s)

    if not current_keywords:
        return {
            "ok": True,
            "hints": [],
            "report_path": None,
            "current_keywords": [],
            "checked": False,
        }

    rows = await _query_index(founder_id=founder_id, exclude_mission_id=mission_id)
    current_set = set(current_keywords)
    scored: list[dict[str, Any]] = []
    for r in rows:
        try:
            kws = json.loads(r.get("domain_keywords_json") or "[]")
        except Exception:
            kws = []
        prior_set = {str(k).lower() for k in kws if isinstance(k, str)}
        j = _jaccard(current_set, prior_set)
        if j >= jaccard_threshold:
            overlap = sorted(current_set & prior_set)
            scored.append({
                "prior_mission_id": r.get("mission_id"),
                "artifact_name": r.get("artifact_name"),
                "artifact_path": r.get("artifact_path"),
                "schema_version": r.get("schema_version"),
                "overlap_keywords": overlap,
                "jaccard": round(j, 3),
                "created_at": r.get("created_at"),
            })

    scored.sort(
        key=lambda h: (h["jaccard"], h.get("created_at") or ""),
        reverse=True,
    )
    hints = scored[:top_k]

    report_path: str | None = None
    if hints:
        report_path = _write_hints_report(ws, hints, current_keywords)

    return {
        "ok": True,
        "hints": hints,
        "report_path": report_path,
        "current_keywords": current_keywords,
        "checked": True,
    }


async def _query_index(
    *, founder_id: str, exclude_mission_id: int,
) -> list[dict[str, Any]]:
    """Fetch matching rows from ``mission_artifacts_index``."""
    try:
        from dabidabi import get_db
        db = await get_db()
        cur = await db.execute(
            """
            SELECT mission_id, artifact_name, artifact_path,
                   schema_version, domain_keywords_json, created_at
            FROM mission_artifacts_index
            WHERE founder_id = ?
              AND mission_id != ?
              AND artifact_name IN (?, ?, ?)
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (
                founder_id,
                int(exclude_mission_id),
                *TARGET_ARTIFACT_NAMES,
            ),
        )
        rows = await cur.fetchall()
        await cur.close()
        out = []
        for row in rows:
            out.append({
                "mission_id": row[0],
                "artifact_name": row[1],
                "artifact_path": row[2],
                "schema_version": row[3],
                "domain_keywords_json": row[4],
                "created_at": row[5],
            })
        return out
    except Exception as e:
        logger.warning("surface_prior_mission_hints: index query failed: %s", e)
        return []


def _write_hints_report(
    workspace_path: str,
    hints: list[dict[str, Any]],
    current_keywords: list[str],
) -> str:
    os.makedirs(workspace_path, exist_ok=True)
    out = os.path.join(workspace_path, "prior_mission_hints.md")
    lines: list[str] = [
        "# Prior mission hints",
        "",
        "_Cross-mission ADR + compliance inheritance (advisory)._",
        "",
        f"**Current keywords:** {', '.join(current_keywords) or '(none)'}",
        "",
        "Founder: tap **reuse** to inherit a prior decision, **diverge** to record why this mission differs.",
        "",
    ]
    for h in hints:
        lines.append(
            f"## #{h.get('prior_mission_id')} — {h.get('artifact_name')} "
            f"(jaccard {h['jaccard']:.2f})"
        )
        if h.get("schema_version"):
            lines.append(f"- **schema:** v{h['schema_version']}")
        if h.get("artifact_path"):
            lines.append(f"- **path:** `{h['artifact_path']}`")
        if h.get("overlap_keywords"):
            lines.append(
                f"- **overlap:** {', '.join(h['overlap_keywords'])}"
            )
        lines.append("")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
    except Exception as e:
        logger.warning(
            "surface_prior_mission_hints: write report %s failed: %s", out, e,
        )
    return out


async def index_mission_artifacts(
    mission_id: int,
    *,
    workspace_path: str | None = None,
    founder_id: str = "default",
) -> dict[str, Any]:
    """Walk phase-≤6 artifacts and write rows into ``mission_artifacts_index``.

    One batch pass at the phase-6 tail; cleaner than touching every
    individual emitter. Reads each artifact's ``_schema_version`` from
    its JSON envelope (when present) and derives ``domain_keywords``
    from the charter via :func:`_extract_domain_keywords`.

    Returns ``{ok, indexed: int, paths: [...]}``.
    """
    ws = _resolve_workspace(mission_id, workspace_path)
    if not os.path.isdir(ws):
        return {"ok": True, "indexed": 0, "paths": [], "reason": "no workspace"}

    charter_text = _load_charter_text(ws) or ""
    domain_keywords = _extract_domain_keywords(charter_text)
    fingerprint = _load_compliance_fingerprint(ws) or {}
    for jur in (fingerprint.get("jurisdictions") or []):
        s = str(jur).strip().lower()
        if s and s not in domain_keywords:
            domain_keywords.append(s)

    candidates = _collect_indexable_artifacts(ws)
    if not candidates:
        return {"ok": True, "indexed": 0, "paths": []}

    indexed_paths: list[str] = []
    try:
        from dabidabi import get_db
        db = await get_db()
        for art_name, art_path in candidates:
            schema_version = _read_schema_version(art_path)
            try:
                await db.execute(
                    """
                    INSERT INTO mission_artifacts_index (
                        mission_id, artifact_name, artifact_path,
                        schema_version, domain_keywords_json, founder_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(mission_id, artifact_name) DO UPDATE SET
                        artifact_path=excluded.artifact_path,
                        schema_version=excluded.schema_version,
                        domain_keywords_json=excluded.domain_keywords_json,
                        founder_id=excluded.founder_id
                    """,
                    (
                        int(mission_id),
                        art_name,
                        art_path,
                        schema_version,
                        json.dumps(domain_keywords),
                        founder_id,
                    ),
                )
                indexed_paths.append(art_path)
            except Exception as e:
                logger.warning(
                    "index_mission_artifacts: insert failed %s: %s", art_path, e,
                )
        try:
            await db.commit()
        except Exception:
            pass
    except Exception as e:
        logger.warning("index_mission_artifacts: db unavailable: %s", e)
        return {"ok": False, "indexed": 0, "paths": [], "reason": str(e)}

    return {
        "ok": True,
        "indexed": len(indexed_paths),
        "paths": indexed_paths,
        "domain_keywords": domain_keywords,
    }


def _collect_indexable_artifacts(workspace_path: str) -> list[tuple[str, str]]:
    """Return list of (artifact_name, absolute_path) pairs."""
    items: list[tuple[str, str]] = []

    def _maybe(name: str, *path_parts: str) -> None:
        full = os.path.join(workspace_path, *path_parts)
        if os.path.isfile(full):
            items.append((name, full))

    _maybe("compliance_fingerprint", "compliance_fingerprint.json")
    _maybe("compliance_overlay", "compliance_overlay.json")
    _maybe("charter", ".charter", "product_charter.md")
    _maybe("non_goals", "non_goals.md")
    _maybe("reverse_pitch", "reverse_pitch.md")

    # ADR register + every adr-*.json under .adrs/.
    adr_dir = os.path.join(workspace_path, ".adrs")
    if os.path.isdir(adr_dir):
        register = os.path.join(adr_dir, "register.md")
        if os.path.isfile(register):
            items.append(("adr_register", register))
        for fname in sorted(os.listdir(adr_dir)):
            if fname.endswith(".json"):
                items.append(
                    (f"adr_{fname[:-5]}", os.path.join(adr_dir, fname))
                )

    return items


def _read_schema_version(path: str) -> str | None:
    """Extract _schema_version from a JSON artifact (or None)."""
    if not path.endswith(".json"):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            v = data.get("_schema_version")
            if v is not None:
                return str(v)
    except Exception:
        return None
    return None
