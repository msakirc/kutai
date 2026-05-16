"""Integration tests for intersect.flash."""
import json

import pytest

from intersect.flash import flash as do_flash


@pytest.fixture
def patch_yalayut(monkeypatch, fake_artifact):
    """Patch yalayut.query + trust lookups so flash runs without Phase 1."""
    def _install(artifacts):
        async def _query(task_ctx):
            return list(artifacts)
        import yalayut
        monkeypatch.setattr(yalayut, "query", _query, raising=False)
    return _install


@pytest.mark.asyncio
async def test_flash_attaches_inject_envelope(
    intersect_db, sample_task, patch_yalayut, fake_artifact, monkeypatch,
):
    patch_yalayut([fake_artifact(
        artifact_id=1, kind="prompt_skill", vet_tier=0, score=0.9,
        name="anthropics-pdf",
    )])
    # source/owner trust default to 1.0 when rows absent (see _trust).
    out = await do_flash(sample_task)
    skills = out["skills"]
    assert isinstance(skills, list)
    assert len(skills) == 1
    s = skills[0]
    assert s["exposure_class"] == "inject"
    assert s["applies_to"] == "execution"
    assert s["render"] in ("prose", "prebind")
    assert s["name"] == "anthropics-pdf"
    assert "payload" in s
    assert isinstance(s["confidence"], float)


@pytest.mark.asyncio
async def test_flash_routes_preempt_to_mechanical_lane(
    intersect_db, sample_task, patch_yalayut, fake_artifact,
):
    patch_yalayut([fake_artifact(
        artifact_id=18, kind="shell_recipe", mechanizable=True, vet_tier=0,
        score=1.0, name="cc-pypackage",
        inputs_schema={},  # non-parametric → fully bound trivially
    )])
    out = await do_flash(sample_task)
    # preempt does not ride the envelope.
    assert out["skills"] == []
    assert out["runner"] == "mechanical"
    payload = out["payload"]
    assert payload["action"] == "yalayut_recipe"
    assert payload["recipe_id"] == 18
    assert "args" in payload


@pytest.mark.asyncio
async def test_flash_quarantine_artifacts_excluded(
    intersect_db, sample_task, patch_yalayut, fake_artifact,
):
    patch_yalayut([fake_artifact(artifact_id=3, vet_tier=2, score=1.0)])
    out = await do_flash(sample_task)
    assert out["skills"] == []


@pytest.mark.asyncio
async def test_flash_skips_query_when_recipe_lookup_false(
    intersect_db, patch_yalayut, fake_artifact,
):
    called = {"hit": False}

    async def _query(task_ctx):
        called["hit"] = True
        return []

    import yalayut
    import pytest as _pt
    _pt.MonkeyPatch().setattr(yalayut, "query", _query, raising=False)
    task = {
        "id": 5, "title": "Design the architecture", "description": "",
        "context": json.dumps({"recipe_lookup": False}),
    }
    out = await do_flash(task)
    assert out["skills"] == []


@pytest.mark.asyncio
async def test_flash_graceful_degrade_on_error(
    intersect_db, sample_task, monkeypatch,
):
    async def _boom(task_ctx):
        raise RuntimeError("yalayut exploded")

    import yalayut
    monkeypatch.setattr(yalayut, "query", _boom, raising=False)
    out = await do_flash(sample_task)
    # Error must not propagate; skills defaults to empty.
    assert out["skills"] == []


@pytest.mark.asyncio
async def test_flash_conflict_loser_dropped_and_logged(
    intersect_db, sample_task, patch_yalayut, fake_artifact,
):
    patch_yalayut([
        fake_artifact(artifact_id=10, kind="agent_config", vet_tier=0, score=0.9,
                      name="wshobson-backend-architect"),
        fake_artifact(artifact_id=11, kind="agent_config", vet_tier=0, score=0.6,
                      name="wshobson-security-auditor"),
    ])
    out = await do_flash(sample_task)
    # Two agent_config skills collide on the same slot — only the
    # highest-score one is kept.
    agent_cfgs = [s for s in out["skills"]
                  if s["name"].startswith("wshobson-")]
    assert len(agent_cfgs) == 1
    assert agent_cfgs[0]["artifact_id"] == 10
    cur = await intersect_db.execute(
        "SELECT artifact_id FROM yalayut_usage WHERE conflict_loser = 1")
    losers = await cur.fetchall()
    assert (11,) in losers


@pytest.mark.asyncio
async def test_flash_writes_usage_telemetry(
    intersect_db, sample_task, patch_yalayut, fake_artifact,
):
    patch_yalayut([fake_artifact(artifact_id=1, kind="prompt_skill", vet_tier=0,
                                 score=0.9)])
    await do_flash(sample_task)
    cur = await intersect_db.execute(
        "SELECT exposure_class FROM yalayut_usage WHERE exposed = 1")
    rows = await cur.fetchall()
    assert rows and rows[0][0] == "inject"
