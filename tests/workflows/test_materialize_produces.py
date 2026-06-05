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
