"""materialize_produces — sole writer of produces paths (spec 2026-06-05)."""
from __future__ import annotations

import json

import pytest

from src.workflows.engine.hooks import materialize_produces


_SCHEMA_MD = {"competitive_positioning":
              {"type": "markdown", "required_sections": ["Landscape", "Notes"]}}

_NARRATION_WRAP = (
    "## Analysis\n\n### Corrected Artifact Content\n\n"
    "```yaml\n---\nmission_id: 81\n---\n\n## Landscape\nx\n\n## Notes\ny\n```\n"
)


def _ctx(produces, schema=None):
    c = {"produces": produces}
    if schema is not None:
        c["artifact_schema"] = schema
    return c


@pytest.mark.asyncio
async def test_writes_unwrapped_canonical_and_returns_it(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path), raising=False)
    rel = "mission_81/.prd/competitive_positioning.md"
    ctx = _ctx([rel], _SCHEMA_MD)
    task = {"mission_id": 81, "agent_type": "analyst"}

    out = await materialize_produces(ctx, task, {"result": _NARRATION_WRAP}, _NARRATION_WRAP)

    disk = (tmp_path / rel).read_text(encoding="utf-8")
    assert disk.strip().startswith("---")       # front-matter at file start
    assert "## Landscape" in disk and "```" not in disk
    assert out == disk                            # returned == on-disk (gate parity)


@pytest.mark.asyncio
async def test_mechanical_executor_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path), raising=False)
    ctx = _ctx(["mission_81/x.md"], _SCHEMA_MD)
    task = {"mission_id": 81, "executor": "mechanical"}
    out = await materialize_produces(ctx, task, {}, "## Landscape\nx")
    assert out == "## Landscape\nx"               # unchanged
    assert not (tmp_path / "mission_81/x.md").exists()


@pytest.mark.asyncio
async def test_no_schema_writes_passthrough(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path), raising=False)
    rel = "mission_81/notes.md"
    ctx = _ctx([rel])                              # no artifact_schema
    task = {"mission_id": 81, "agent_type": "analyst"}
    body = "# Notes\n\nplain body"
    out = await materialize_produces(ctx, task, {}, body)
    written = (tmp_path / rel).read_text(encoding="utf-8")
    assert written.endswith(body)                  # stamped + body
    assert "mission_id: 81" in written
    assert out.endswith(body)


@pytest.mark.asyncio
async def test_json_unwrapped_and_stamped(tmp_path, monkeypatch):
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path), raising=False)
    rel = "mission_81/.intake/draft.json"
    schema = {"draft": {"type": "object", "required": ["items"]}}
    ctx = _ctx([rel], schema)
    task = {"mission_id": 81, "agent_type": "analyst"}
    wrapped = '## Summary\n```json\n{"items": [1, 2]}\n```\n'
    out = await materialize_produces(ctx, task, {}, wrapped)
    disk = json.loads((tmp_path / rel).read_text(encoding="utf-8"))
    assert disk["items"] == [1, 2]
    assert disk["mission_id"] == 81
    assert json.loads(out)["items"] == [1, 2]


@pytest.mark.asyncio
async def test_multi_produces_output_value_does_not_contaminate_sibling(tmp_path, monkeypatch):
    """Cut #2: ``output_value`` is the PRIMARY artifact's content and must never
    land in a sibling file. Repro (5.0d shape: screen_inventory.md +
    shared_shell.md): the sibling's own disk content is incomplete (fails its
    required_sections) while output_value (the primary) coincidentally passes
    the sibling's schema. Pre-fix, the sibling's candidates were
    [sibling_disk(fail), output_value(pass)] -> select_canonical returned
    output_value -> the sibling file is overwritten with the PRIMARY's content.
    Disk-only multi candidates must keep the sibling's own content."""
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path), raising=False)
    primary_rel = "mission_81/.flow/screen_inventory.md"
    sibling_rel = "mission_81/.flow/shared_shell.md"
    (tmp_path / "mission_81/.flow").mkdir(parents=True, exist_ok=True)
    # Sibling's OWN (incomplete) content — missing the "Navigation" section.
    sibling_disk = "## Shell Layout\nheader only; navigation not yet described\n"
    (tmp_path / sibling_rel).write_text(sibling_disk, encoding="utf-8")
    (tmp_path / primary_rel).write_text("## Screens\n- login\n- home\n", encoding="utf-8")

    # Schema keyed for the sibling; output_value (primary) happens to satisfy it.
    schema = {"shared_shell": {"type": "markdown",
                               "required_sections": ["Shell Layout", "Navigation"]}}
    ctx = _ctx([primary_rel, sibling_rel], schema)
    task = {"mission_id": 81, "agent_type": "designer"}
    # Primary result that coincidentally carries both sibling sections.
    output_value = "## Shell Layout\nfrom-primary\n## Navigation\nfrom-primary\n"

    out = await materialize_produces(ctx, task, {}, output_value)

    sibling_after = (tmp_path / sibling_rel).read_text(encoding="utf-8")
    assert "navigation not yet described" in sibling_after   # sibling's own content kept
    assert "from-primary" not in sibling_after               # PRIMARY content did NOT leak in
    # Multi return contract: output_value returned unchanged.
    assert out == output_value


@pytest.mark.asyncio
async def test_multi_produces_length_fallback_does_not_contaminate(tmp_path, monkeypatch):
    """Length-fallback variant: when NEITHER the sibling disk nor output_value
    passes the schema, pre-fix select_canonical fell through to the *longest*
    candidate — a long output_value would still clobber the sibling. Disk-only
    candidates must keep the sibling's own content regardless of length."""
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path), raising=False)
    primary_rel = "mission_81/.flow/screen_inventory.md"
    sibling_rel = "mission_81/.flow/shared_shell.md"
    (tmp_path / "mission_81/.flow").mkdir(parents=True, exist_ok=True)
    sibling_disk = "## Shell Layout\nshort sibling body\n"
    (tmp_path / sibling_rel).write_text(sibling_disk, encoding="utf-8")
    (tmp_path / primary_rel).write_text("## Screens\n- login\n", encoding="utf-8")
    # Schema nothing here satisfies (forces the most-substantial fallback).
    schema = {"shared_shell": {"type": "markdown",
                               "required_sections": ["Shell Layout", "Navigation", "Footer"]}}
    ctx = _ctx([primary_rel, sibling_rel], schema)
    task = {"mission_id": 81, "agent_type": "designer"}
    long_output = "## Screens\n" + ("- screen line\n" * 80)   # far longer than sibling

    await materialize_produces(ctx, task, {}, long_output)

    sibling_after = (tmp_path / sibling_rel).read_text(encoding="utf-8")
    assert "short sibling body" in sibling_after     # sibling kept
    assert "screen line" not in sibling_after        # long primary did not win


@pytest.mark.asyncio
async def test_mission81_289715_regression(tmp_path, monkeypatch):
    """The real failure: agent wrote a narration report to the produces path
    while the correct doc sat in a ```yaml fence in result. Materializer must
    overwrite disk with the unwrapped artifact AND pass both the loose schema
    and a strict front-matter check."""
    monkeypatch.setattr("src.tools.workspace.WORKSPACE_DIR", str(tmp_path), raising=False)
    rel = "mission_81/.prd/competitive_positioning.md"
    abs_p = tmp_path / rel
    abs_p.parent.mkdir(parents=True, exist_ok=True)
    abs_p.write_text("## Findings\n- listed\n## Recommendations\nready.", encoding="utf-8")  # narration on disk
    ctx = _ctx([rel], {"competitive_positioning":
                       {"type": "markdown",
                        "required_sections": ["Landscape", "Value Thesis", "Strengths",
                                              "Our Differentiators", "Switching Costs", "Notes"]}})
    task = {"mission_id": 81, "agent_type": "analyst"}
    result_text = (
        "## Analysis\n```yaml\n---\nmission_id: 81\n---\n\n"
        "## Landscape\na\n## Value Thesis\nb\n## Strengths\nc\n"
        "## Our Differentiators\nd\n## Switching Costs\ne\n## Notes\nf\n```\n"
    )
    out = await materialize_produces(ctx, task, {"result": result_text}, result_text)
    disk = abs_p.read_text(encoding="utf-8")
    assert disk.lstrip().startswith("---")            # strict front-matter gate
    assert "## Landscape" in disk and "## Notes" in disk
    assert "## Findings" not in disk                  # narration replaced
    assert out == disk
