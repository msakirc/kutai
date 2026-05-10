"""diff_bundle — rule-based bundle-quality regression vs Paraflow goldens.

Design choices (locked at Z1 Tier 7B):

- **Rule-based**, not LLM. The harness is meant to run nightly / on
  every mission complete; LLM cost would dominate. Coverage,
  section-shape coherence, and design-token-axis fitness are all
  decidable by string + JSON inspection.
- **Stdlib only.** No yaml, no jsonschema. Markdown headings parsed by
  scanning ``^## `` lines; design tokens read as JSON.
- **Verdicts are coarse.** ``paraflow_par`` (>= 0.85 score),
  ``paraflow_partial`` (>= 0.50), ``paraflow_gap`` (< 0.50). The
  numbers are intentionally generous — the regression's job is to
  spot *missing whole artifacts*, not nitpick prose.

Mission workspace layout assumptions
------------------------------------

The harness inspects, in order, well-known relative paths under the
mission workspace::

    charter.md, product_charter.md, .charter.md
    personas.md, persona_*.md
    prd.md, PRD.md
    user_flow.md
    screen_plans/  or  Feature Plan/  or  .screen_plans/
    screens/       or  Screen & Prototype/  or  .web/screens/
    style_guide_light.md, .style/style_guide_light.md
    style_guide_dark.md,  .style/style_guide_dark.md
    .style/design_tokens.json

When KutAI eventually settles its naming, additional aliases can be
appended to the ``_ALIASES`` table below — both candidate sets are
checked and the first hit wins.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_GOLDENS_ROOT = (
    Path(__file__).resolve().parents[4] / "tests" / "goldens" / "paraflow"
)

KNOWN_ARCHETYPES = ("truthrate",)

# Verdict thresholds (composite score in [0, 1])
_PAR_THRESHOLD = 0.85
_PARTIAL_THRESHOLD = 0.50

# Mission-side path aliases per artifact slot.
_ALIASES: dict[str, tuple[str, ...]] = {
    "charter": ("charter.md", "product_charter.md", ".charter.md"),
    "personas": ("personas.md",),
    "prd": ("prd.md", "PRD.md"),
    "user_flow": ("user_flow.md",),
    "style_guide_light": (
        "style_guide_light.md",
        ".style/style_guide_light.md",
    ),
    "style_guide_dark": (
        "style_guide_dark.md",
        ".style/style_guide_dark.md",
    ),
    # directories
    "screen_plans_dir": ("screen_plans", "Feature Plan", ".screen_plans"),
    "screens_dir": ("screens", "Screen & Prototype", ".web/screens"),
    # tokens (no golden equivalent, but consistency check needs it)
    "design_tokens": (
        ".style/design_tokens.json",
        "design_tokens.json",
    ),
}

_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_NUMBERED_PREFIX_RE = re.compile(r"^\s*\d+\)\s*")


class GoldenNotFoundError(KeyError):
    """Raised when ``archetype`` is not under the goldens root."""


@dataclass
class _Golden:
    """In-memory view of a golden archetype bundle."""

    archetype: str
    root: Path
    files: dict[str, Path] = field(default_factory=dict)
    screen_plan_keys: list[str] = field(default_factory=list)
    screen_keys: list[str] = field(default_factory=list)
    charter_sections: list[str] = field(default_factory=list)
    prd_sections: list[str] = field(default_factory=list)
    user_flow_sections: list[str] = field(default_factory=list)
    style_light_sections: list[str] = field(default_factory=list)
    style_dark_sections: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    """Lower + strip leading numbered prefix (``1) ``, ``2) ``...)."""
    return _NUMBERED_PREFIX_RE.sub("", s).strip().lower()


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return ""


def _section_titles(text: str) -> list[str]:
    """Return normalized ``##`` heading titles."""
    return [_norm(m.group(1)) for m in _HEADING_RE.finditer(text)]


def load_golden(
    archetype: str, goldens_root: Path | str | None = None
) -> _Golden:
    """Load a golden archetype bundle.

    Raises :class:`GoldenNotFoundError` if the archetype directory is
    missing or empty (allowing callers to skip the regression in CI
    environments without the goldens checked out).
    """
    root = Path(goldens_root or DEFAULT_GOLDENS_ROOT) / archetype
    if not root.is_dir():
        raise GoldenNotFoundError(
            f"archetype {archetype!r} not under {root.parent}"
        )

    files: dict[str, Path] = {}
    for slot in (
        "charter",
        "personas",
        "prd",
        "user_flow",
        "style_guide_light",
        "style_guide_dark",
    ):
        cand = root / f"{slot}.md"
        if cand.is_file():
            files[slot] = cand

    sp_dir = root / "screen_plans"
    sc_dir = root / "screens"
    screen_plan_keys = (
        sorted(p.stem.replace("_screen_plan", "") for p in sp_dir.glob("*.md"))
        if sp_dir.is_dir()
        else []
    )
    screen_keys = (
        sorted(p.stem for p in sc_dir.glob("*.html"))
        if sc_dir.is_dir()
        else []
    )

    g = _Golden(
        archetype=archetype,
        root=root,
        files=files,
        screen_plan_keys=screen_plan_keys,
        screen_keys=screen_keys,
        charter_sections=_section_titles(_read_text(files["charter"]))
        if "charter" in files
        else [],
        prd_sections=_section_titles(_read_text(files["prd"]))
        if "prd" in files
        else [],
        user_flow_sections=_section_titles(_read_text(files["user_flow"]))
        if "user_flow" in files
        else [],
        style_light_sections=_section_titles(
            _read_text(files["style_guide_light"])
        )
        if "style_guide_light" in files
        else [],
        style_dark_sections=_section_titles(
            _read_text(files["style_guide_dark"])
        )
        if "style_guide_dark" in files
        else [],
    )
    return g


# ---------------------------------------------------------------------------
# Mission inspection
# ---------------------------------------------------------------------------


def _resolve_mission_path(workspace: Path, slot: str) -> Path | None:
    """Return the first existing alias under ``workspace`` for ``slot``."""
    for alias in _ALIASES.get(slot, ()):
        cand = workspace / alias
        if slot.endswith("_dir"):
            if cand.is_dir():
                return cand
        else:
            if cand.is_file():
                return cand
    return None


def _design_token_axes(tokens_path: Path | None) -> set[str]:
    """Return the top-level keys of ``design_tokens.json`` (axes)."""
    if tokens_path is None or not tokens_path.is_file():
        return set()
    try:
        obj = json.loads(_read_text(tokens_path))
    except (json.JSONDecodeError, ValueError):
        return set()
    if not isinstance(obj, dict):
        return set()
    # Strip schema-version metadata.
    return {k for k in obj.keys() if not k.startswith("_")}


_PARAFLOW_TOKEN_AXES_BY_STYLE_HEADING = {
    "colors": "colors",
    "typography": "typography",
    "border radius": "border_radius",
    "layout & spacing": "spacing",
    "page layout - mobile": "layout",
}


def _expected_token_axes_from_style(g: _Golden) -> set[str]:
    """Map paraflow style-guide sections to expected design-token axes."""
    axes: set[str] = set()
    for h in g.style_light_sections + g.style_dark_sections:
        if h in _PARAFLOW_TOKEN_AXES_BY_STYLE_HEADING:
            axes.add(_PARAFLOW_TOKEN_AXES_BY_STYLE_HEADING[h])
    return axes


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def _coverage(workspace: Path, g: _Golden) -> dict[str, Any]:
    """Per-artifact presence map. Each value is bool or a dict for dirs."""
    cov: dict[str, Any] = {}
    for slot in (
        "charter",
        "personas",
        "prd",
        "user_flow",
        "style_guide_light",
        "style_guide_dark",
    ):
        cov[slot] = _resolve_mission_path(workspace, slot) is not None

    sp_dir = _resolve_mission_path(workspace, "screen_plans_dir")
    sp_count = len(list(sp_dir.glob("*.md"))) if sp_dir else 0
    cov["screen_plans"] = {
        "present": sp_dir is not None,
        "count": sp_count,
        "golden_count": len(g.screen_plan_keys),
    }

    sc_dir = _resolve_mission_path(workspace, "screens_dir")
    sc_count = len(list(sc_dir.glob("*.html"))) if sc_dir else 0
    cov["screens"] = {
        "present": sc_dir is not None,
        "count": sc_count,
        "golden_count": len(g.screen_keys),
    }

    return cov


def _coherence(workspace: Path, g: _Golden) -> dict[str, Any]:
    """Section-shape match: does mission's artifact share golden's
    ``##`` headings?

    Each entry maps to ``{"matched": [...], "missing": [...],
    "score": float}``. ``score`` = matched / max(1, len(golden_sections)).
    """
    out: dict[str, Any] = {}

    def _check(slot: str, golden_sections: list[str]) -> None:
        if not golden_sections:
            out[slot] = {
                "matched": [],
                "missing": [],
                "score": 1.0,
                "reason": "golden has no sections",
            }
            return
        p = _resolve_mission_path(workspace, slot)
        if p is None:
            out[slot] = {
                "matched": [],
                "missing": list(golden_sections),
                "score": 0.0,
                "reason": "mission artifact missing",
            }
            return
        mission_sections = set(_section_titles(_read_text(p)))
        matched = [s for s in golden_sections if s in mission_sections]
        missing = [s for s in golden_sections if s not in mission_sections]
        out[slot] = {
            "matched": matched,
            "missing": missing,
            "score": len(matched) / max(1, len(golden_sections)),
        }

    _check("charter", g.charter_sections)
    _check("prd", g.prd_sections)
    _check("user_flow", g.user_flow_sections)
    _check("style_guide_light", g.style_light_sections)
    _check("style_guide_dark", g.style_dark_sections)
    return out


def _design_fitness(workspace: Path, g: _Golden) -> dict[str, Any]:
    """Design-tokens axis match against paraflow's style-guide axes."""
    expected_axes = _expected_token_axes_from_style(g)
    tokens_path = _resolve_mission_path(workspace, "design_tokens")
    actual_axes = _design_token_axes(tokens_path)

    if not expected_axes:
        return {
            "expected_axes": [],
            "actual_axes": sorted(actual_axes),
            "missing_axes": [],
            "score": 1.0,
            "reason": "golden has no derivable axes",
        }
    missing = sorted(expected_axes - actual_axes)
    matched = sorted(expected_axes & actual_axes)
    return {
        "expected_axes": sorted(expected_axes),
        "actual_axes": sorted(actual_axes),
        "matched_axes": matched,
        "missing_axes": missing,
        "score": len(matched) / max(1, len(expected_axes)),
        "tokens_present": tokens_path is not None,
    }


def _verdict(composite: float) -> str:
    if composite >= _PAR_THRESHOLD:
        return "paraflow_par"
    if composite >= _PARTIAL_THRESHOLD:
        return "paraflow_partial"
    return "paraflow_gap"


def diff_bundle(
    mission_workspace_path: str | os.PathLike,
    archetype: str,
    goldens_root: Path | str | None = None,
) -> dict[str, Any]:
    """Compare a mission workspace to a paraflow golden bundle.

    Parameters
    ----------
    mission_workspace_path:
        Absolute or relative filesystem path to the mission workspace
        (e.g. ``workspace/mission_57/``). Need not exist; missing paths
        produce maximum gaps.
    archetype:
        Golden archetype name (e.g. ``"truthrate"``). Must match a
        directory under ``goldens_root``.
    goldens_root:
        Optional override for the goldens root. Defaults to
        ``tests/goldens/paraflow/``.

    Returns
    -------
    dict
        ``{coverage, coherence, design_fitness, gaps, verdict, score,
        archetype, mission_workspace_path, goldens_root}``.

    Raises
    ------
    GoldenNotFoundError
        Archetype directory missing under ``goldens_root``.
    """
    workspace = Path(mission_workspace_path)
    g = load_golden(archetype, goldens_root)

    coverage = _coverage(workspace, g)
    coherence = _coherence(workspace, g)
    design = _design_fitness(workspace, g)

    # Gaps: any artifact slot whose presence flag is False, plus
    # screen_plans/screens whose count is below the golden count.
    gaps: list[str] = []
    for slot in (
        "charter",
        "personas",
        "prd",
        "user_flow",
        "style_guide_light",
        "style_guide_dark",
    ):
        if not coverage.get(slot):
            gaps.append(slot)
    if not coverage["screen_plans"]["present"]:
        gaps.append("screen_plans")
    elif coverage["screen_plans"]["count"] < coverage["screen_plans"][
        "golden_count"
    ]:
        gaps.append(
            f"screen_plans:short("
            f"{coverage['screen_plans']['count']}<"
            f"{coverage['screen_plans']['golden_count']})"
        )
    if not coverage["screens"]["present"]:
        gaps.append("screens")
    elif coverage["screens"]["count"] < coverage["screens"]["golden_count"]:
        gaps.append(
            f"screens:short("
            f"{coverage['screens']['count']}<"
            f"{coverage['screens']['golden_count']})"
        )
    for slot, info in coherence.items():
        if info.get("missing"):
            gaps.append(f"coherence:{slot}({len(info['missing'])} missing)")
    if design.get("missing_axes"):
        gaps.append(f"design_tokens_axes:{','.join(design['missing_axes'])}")

    # Composite score: presence (40%) + coherence avg (40%) + design (20%).
    presence_slots = (
        "charter",
        "personas",
        "prd",
        "user_flow",
        "style_guide_light",
        "style_guide_dark",
    )
    pres_hits = sum(1 for s in presence_slots if coverage.get(s))
    pres_hits += 1 if coverage["screen_plans"]["present"] else 0
    pres_hits += 1 if coverage["screens"]["present"] else 0
    pres_total = len(presence_slots) + 2
    presence_score = pres_hits / pres_total

    coh_scores = [v["score"] for v in coherence.values()]
    coherence_score = sum(coh_scores) / max(1, len(coh_scores))

    design_score = design["score"]

    composite = (
        0.4 * presence_score + 0.4 * coherence_score + 0.2 * design_score
    )
    verdict = _verdict(composite)

    return {
        "archetype": archetype,
        "mission_workspace_path": str(workspace),
        "goldens_root": str(
            Path(goldens_root or DEFAULT_GOLDENS_ROOT).resolve()
        ),
        "coverage": coverage,
        "coherence": coherence,
        "design_fitness": design,
        "gaps": gaps,
        "score": round(composite, 4),
        "presence_score": round(presence_score, 4),
        "coherence_score": round(coherence_score, 4),
        "design_score": round(design_score, 4),
        "verdict": verdict,
    }
