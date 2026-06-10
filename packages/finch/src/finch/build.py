"""Uniform prompt assembly for overhead/husam-caller prompts.

Rubric content (system + user template) lives in rubrics/*.yaml, loaded once.
Callers pass task fields + optional dynamic blocks; this is the ONE place
overhead messages get built (unanimity). Dynamic context (mission lessons,
calibration, tools) is passed IN as extra_blocks — the leaf can't fetch it.
"""
from __future__ import annotations
from pathlib import Path
import re
import yaml

_RUBRICS_DIR = Path(__file__).parent / "rubrics"
_RUBRICS: dict[str, dict] = {}


def _load_rubrics() -> None:
    for yml in sorted(_RUBRICS_DIR.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        _RUBRICS[data["key"]] = data


def register_rubric(key: str, system: str, user_template: str) -> None:
    """Programmatic registration (tests / dynamic rubrics)."""
    _RUBRICS[key] = {"key": key, "system": system, "user_template": user_template}


def _render(template: str, fields: dict) -> str:
    """Replace explicit {key} tokens with their values in a SINGLE pass.

    Single-pass regex so a field VALUE that itself contains another field's
    {token} is NOT re-substituted (e.g. a task description containing the
    literal text "{response}" must survive verbatim, not get clobbered by the
    grading rubric's response field). Longest keys first so a short key cannot
    shadow a longer token sharing its prefix. All other braces are untouched.
    """
    if not fields:
        return template
    keys = sorted(fields, key=len, reverse=True)
    pattern = "|".join(re.escape("{" + k + "}") for k in keys)
    return re.sub(pattern, lambda m: str(fields[m.group(0)[1:-1]]), template)


def build_messages(key: str, fields: dict, extra_blocks: list[str] | None = None) -> list[dict]:
    r = _RUBRICS[key]
    system = r["system"]
    if extra_blocks:
        system = "\n\n".join([system, *extra_blocks])
    user = _render(r["user_template"], fields)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


if _RUBRICS_DIR.exists():
    _load_rubrics()
