"""Coverage: expander preserves any_of nested-list entries in `produces`.

Bug history: pre-fix, the expander filtered produces with
`isinstance(p, str)`, silently dropping nested any_of slots
(``[["a.py", "b.py"]]``). Mission would run with `produces=[]` so the
verify_artifacts post-hook degenerated to a no-op.

Fix: list-of-strings entries pass through; template substitution walks
inside each alternative.
"""
from __future__ import annotations

import json

from src.workflows.engine.expander import expand_steps_to_tasks, expand_template


def _make_step(produces, **overrides) -> dict:
    base = {
        "id": "9.99",
        "phase": "phase_9",
        "name": "scaffold_with_anyof",
        "agent": "coder",
        "difficulty": "easy",
        "tools_hint": ["write_file"],
        "depends_on": [],
        "input_artifacts": [],
        "output_artifacts": ["scaffold_result"],
        "instruction": "scaffold the thing",
        "done_when": "scaffold_result exists",
        "artifact_schema": {"scaffold_result": {"type": "object", "fields": {}}},
        "produces": produces,
    }
    base.update(overrides)
    return base


def _ctx(task: dict) -> dict:
    raw = task.get("context")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw or {}


def test_top_level_string_entry_passes_through():
    step = _make_step(produces=["Dockerfile"])
    tasks = expand_steps_to_tasks([step], mission_id=1, initial_context={})
    assert _ctx(tasks[0])["produces"] == ["Dockerfile"]


def test_top_level_any_of_nested_list_preserved():
    step = _make_step(produces=[["alembic.ini", "prisma/schema.prisma"]])
    tasks = expand_steps_to_tasks([step], mission_id=1, initial_context={})
    assert _ctx(tasks[0])["produces"] == [["alembic.ini", "prisma/schema.prisma"]]


def test_top_level_mixed_string_and_any_of():
    step = _make_step(produces=[
        "Dockerfile",
        ["a.py", "b.ts"],
        "migrations/**/*.py",
    ])
    tasks = expand_steps_to_tasks([step], mission_id=1, initial_context={})
    assert _ctx(tasks[0])["produces"] == [
        "Dockerfile",
        ["a.py", "b.ts"],
        "migrations/**/*.py",
    ]


def test_top_level_empty_or_invalid_entries_dropped():
    step = _make_step(produces=[
        "",
        "  ",
        ["", "x"],   # contains empty alt → invalid
        [],          # empty list → invalid
        42,          # wrong type → dropped
        "ok.py",
        ["a.py", "b.py"],
    ])
    tasks = expand_steps_to_tasks([step], mission_id=1, initial_context={})
    assert _ctx(tasks[0])["produces"] == ["ok.py", ["a.py", "b.py"]]


def test_template_substitution_walks_inside_any_of():
    """Per-feature placeholders inside any_of alternatives must interpolate."""
    template = {
        "template_id": "demo",
        "steps": [
            {
                "template_step_id": "feat.0",
                "name": "scaffold",
                "agent": "coder",
                "difficulty": "easy",
                "tools_hint": ["write_file"],
                "instruction": "x",
                "output_artifacts": ["scaffold_result"],
                "produces": [
                    "migrations/{feature_id}/initial.py",
                    [
                        "backend/app/{feature_id}.py",
                        "frontend/src/{feature_id}.ts",
                    ],
                ],
            },
        ],
    }
    expanded = expand_template(
        template,
        params={"feature_id": "F-007"},
        prefix="8.F-007.",
    )
    step = expanded[0]
    assert step["produces"] == [
        "migrations/F-007/initial.py",
        ["backend/app/F-007.py", "frontend/src/F-007.ts"],
    ]


def test_template_substitution_preserves_unknown_placeholders():
    """Missing param leaves the placeholder raw — validator catches downstream."""
    template = {
        "template_id": "demo",
        "steps": [
            {
                "template_step_id": "feat.0",
                "name": "scaffold",
                "agent": "coder",
                "difficulty": "easy",
                "tools_hint": ["write_file"],
                "instruction": "x",
                "output_artifacts": ["x"],
                "produces": [["{missing}/a.py", "{missing}/b.py"]],
            },
        ],
    }
    expanded = expand_template(
        template,
        params={"feature_id": "F-007"},
        prefix="8.F-007.",
    )
    assert expanded[0]["produces"] == [["{missing}/a.py", "{missing}/b.py"]]
