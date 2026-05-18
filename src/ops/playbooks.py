"""Z8 T4C — incident playbook loader + matcher.

Playbooks live under ``recipes/incident_playbook_<name>/v<N>/playbook.yaml``
(deliberately *not* ``recipe.yaml`` so the generic recipe substrate in
``src/infra/recipes.py`` skips them — that engine requires a different
schema (``name``/``version``/``templates``) which playbooks don't share).

A playbook has the shape::

    id: incident_playbook_db_disk_full_v1
    kind: incident_playbook
    description: "..."
    requires:
      tech_stack: [postgres, sqlite, ...]
      runtime_state:
        - condition: db.disk_used_pct_gt
          threshold: 85
    match:
      alerts:
        - integration: betterstack
          event: db_disk_alert
    action_sequence:
      - verb: restart_service
        params: {...}
        reversibility: partial
    on_failure:
      - verb: escalate_to_founder
        severity: critical

Public API
----------
- :func:`load_playbook(path)` → :class:`Playbook`
- :func:`list_playbooks(recipes_dir)` → list[Playbook]
- :func:`match_playbook(alert, runtime_state, recipes_dir)` → Playbook | None
- :func:`match_playbooks_for_stack(tech_stack, recipes_dir)` → list[Playbook]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.infra.logging_config import get_logger

logger = get_logger("ops.playbooks")

# Reuse the same YAML loader the recipe substrate uses so we get pyyaml
# when present and the hand-parser fallback when not.
try:
    import yaml as _yaml  # type: ignore[import]

    def _load_yaml(text: str) -> dict:
        return _yaml.safe_load(text) or {}

except ImportError:  # pragma: no cover
    from src.infra.recipes import _load_yaml  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------

@dataclass
class Playbook:
    id: str
    description: str
    requires: dict = field(default_factory=dict)
    match: dict = field(default_factory=dict)
    action_sequence: list[dict] = field(default_factory=list)
    on_failure: list[dict] = field(default_factory=list)
    _path: Optional[str] = field(default=None, repr=False, compare=False)

    @property
    def tech_stack(self) -> list[str]:
        return [str(s).lower() for s in (self.requires.get("tech_stack") or [])]

    @property
    def alert_matchers(self) -> list[dict]:
        return list((self.match or {}).get("alerts") or [])


# ---------------------------------------------------------------------------

def load_playbook(path: str) -> Playbook:
    """Load a playbook.yaml file. Raises ValueError on bad shape."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"cannot read playbook {path}: {exc}") from exc
    try:
        data = _load_yaml(text)
    except Exception as exc:
        raise ValueError(f"YAML parse error in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"playbook.yaml must be a mapping in {path}")
    if data.get("kind") != "incident_playbook":
        raise ValueError(
            f"playbook.yaml in {path} has kind={data.get('kind')!r} "
            "(expected 'incident_playbook')"
        )
    pb_id = data.get("id")
    if not pb_id:
        raise ValueError(f"playbook.yaml in {path} missing 'id'")
    return Playbook(
        id=str(pb_id),
        description=str(data.get("description") or ""),
        requires=dict(data.get("requires") or {}),
        match=dict(data.get("match") or {}),
        action_sequence=list(data.get("action_sequence") or []),
        on_failure=list(data.get("on_failure") or []),
        _path=str(p),
    )


def list_playbooks(recipes_dir: str = "recipes") -> list[Playbook]:
    """Scan ``recipes/incident_playbook_*/v*/playbook.yaml`` and return
    each one. Invalid files are logged and skipped (don't break callers)."""
    root = Path(recipes_dir)
    if not root.is_dir():
        logger.warning("recipes_dir not found: %s", recipes_dir)
        return []
    out: list[Playbook] = []
    for name_dir in sorted(root.iterdir()):
        if not name_dir.is_dir():
            continue
        if not name_dir.name.startswith("incident_playbook_"):
            continue
        for version_dir in sorted(name_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            yaml_path = version_dir / "playbook.yaml"
            if not yaml_path.exists():
                continue
            try:
                out.append(load_playbook(str(yaml_path)))
            except ValueError as exc:
                logger.warning("skipping invalid playbook %s: %s", yaml_path, exc)
    return sorted(out, key=lambda pb: pb.id)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _alert_matches(playbook: Playbook, alert: dict) -> bool:
    """True when the alert's (integration, event) pair appears in
    ``playbook.match.alerts``. Matching is exact and case-sensitive on
    these two fields — they are vendor-defined and should be reproducible."""
    integration = str(alert.get("integration") or "")
    event = str(alert.get("event") or alert.get("event_type") or "")
    for matcher in playbook.alert_matchers:
        m_int = str(matcher.get("integration") or "")
        m_ev = str(matcher.get("event") or "")
        if m_int and integration != m_int:
            continue
        if m_ev and event != m_ev:
            continue
        return True
    return False


def _runtime_state_matches(playbook: Playbook, runtime_state: dict) -> bool:
    """True when every ``requires.runtime_state`` condition is satisfied.

    Supported condition operators (v1):
      * ``<field>.<key>_gt`` / ``_lt`` — numeric compare against ``threshold``.

    Unknown conditions are treated as ``True`` (don't block matching) but
    are logged so we notice when a playbook is depending on something we
    haven't wired yet.
    """
    conds = (playbook.requires or {}).get("runtime_state") or []
    if not conds:
        return True
    for cond in conds:
        op = str(cond.get("condition") or "")
        threshold = cond.get("threshold")
        if not op:
            continue
        # Expect "<group>.<key>_<gt|lt>"
        if "." not in op:
            logger.debug("playbook %s: condition without dot: %s", playbook.id, op)
            continue
        group, _, rest = op.partition(".")
        suffix = rest.rsplit("_", 1)
        if len(suffix) != 2 or suffix[1] not in ("gt", "lt"):
            logger.debug("playbook %s: unknown op suffix in %s", playbook.id, op)
            continue
        key, comparator = suffix
        try:
            actual = float(((runtime_state or {}).get(group) or {}).get(key))
        except (TypeError, ValueError):
            return False
        try:
            t = float(threshold) if threshold is not None else 0.0
        except (TypeError, ValueError):
            return False
        if comparator == "gt" and not (actual > t):
            return False
        if comparator == "lt" and not (actual < t):
            return False
    return True


def match_playbook(
    alert: dict,
    runtime_state: dict | None = None,
    recipes_dir: str = "recipes",
) -> Playbook | None:
    """Return the first playbook whose alert matchers and runtime-state
    requirements both fit ``alert`` and ``runtime_state``.

    Playbooks are scanned in id-sorted order so the choice is reproducible
    when multiple playbooks match — first definition wins. Callers that
    want a different priority must filter the result of :func:`list_playbooks`.
    """
    rs = runtime_state or {}
    for pb in list_playbooks(recipes_dir):
        if not _alert_matches(pb, alert):
            continue
        if not _runtime_state_matches(pb, rs):
            continue
        return pb
    return None


def match_playbooks_for_stack(
    tech_stack: list[str] | str,
    recipes_dir: str = "recipes",
) -> list[Playbook]:
    """Return every playbook whose ``requires.tech_stack`` overlaps with
    the mission's stack. Used by the phase-13 generator: we don't have a
    live alert at planning time, so we instantiate every playbook the
    stack *could* trigger.
    """
    if isinstance(tech_stack, str):
        parts = {p.strip().lower() for p in tech_stack.split("+") if p.strip()}
    else:
        parts = {str(p).strip().lower() for p in tech_stack if str(p).strip()}
    out: list[Playbook] = []
    for pb in list_playbooks(recipes_dir):
        pb_stack = set(pb.tech_stack)
        if not pb_stack:
            continue
        if pb_stack & parts:
            out.append(pb)
    return out


def to_dict(pb: Playbook) -> dict[str, Any]:
    """Serialise a Playbook for the incident_playbooks artifact."""
    return {
        "id": pb.id,
        "description": pb.description,
        "tech_stack": pb.tech_stack,
        "alerts": pb.alert_matchers,
        "action_sequence": pb.action_sequence,
        "on_failure": pb.on_failure,
    }
