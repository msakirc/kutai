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


# ---------------------------------------------------------------------------
# P2-1 regression: flash must pass the RAW task dict to yalayut.query(),
# not the nested _build_task_ctx() dict.  The raw task has top-level "id"
# and "title"; the nested dict wraps them under a "task" key — from_task()
# would miss them and return an empty query_text() → zero results.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_flash_passes_raw_task_dict_to_yalayut_query(
    intersect_db, sample_task, fake_artifact, monkeypatch,
):
    """yalayut.query must receive the RAW task dict (top-level 'id'/'title'),
    NOT the nested binding-context dict produced by _build_task_ctx."""
    received_arg = {}

    async def _capturing_query(arg):
        received_arg.update(arg)  # store what flash actually passed
        return [fake_artifact(artifact_id=1, kind="prompt_skill", vet_tier=0,
                              score=0.9, name="anthropics-pdf")]

    import yalayut
    monkeypatch.setattr(yalayut, "query", _capturing_query, raising=False)

    await do_flash(sample_task)

    # The raw task dict has top-level "id" and "title".
    assert "id" in received_arg, (
        "yalayut.query received a dict without top-level 'id' — "
        "flash is passing the nested _build_task_ctx() dict instead of the raw task"
    )
    assert "title" in received_arg, (
        "yalayut.query received a dict without top-level 'title' — "
        "flash is passing the nested _build_task_ctx() dict instead of the raw task"
    )
    # The nested binding dict would have a 'task' key — that must NOT be the
    # top-level shape passed to yalayut.query().
    assert "task" not in received_arg, (
        "yalayut.query received the nested _build_task_ctx() dict "
        "(has top-level 'task' key) — it should receive the raw task dict"
    )


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
    intersect_db, sample_task, patch_yalayut, fake_artifact, monkeypatch,
):
    """Preempt routing to mechanical lane works when the gate is explicitly
    enabled (monkeypatched to True). This keeps the routing code covered even
    though the gate is off by default in Phase 2."""
    import sys
    flash_mod = sys.modules["intersect.flash"]
    monkeypatch.setattr(flash_mod, "PHASE2_PREEMPT_ENABLED", True)
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
async def test_flash_phase2_gate_downgrades_preempt_to_inject(
    intersect_db, sample_task, patch_yalayut, fake_artifact,
):
    """Phase 2 regression: with PHASE2_PREEMPT_ENABLED=False (the default),
    a T0 mechanizable shell_recipe artifact that would route to the mechanical
    lane must instead appear as an 'inject' entry in task['skills'].

    The yalayut_recipe executor does not exist until Phase 3 — routing to it
    now causes a guaranteed task failure in mr_roboto.  The gate must be off
    by default and the preempt routing code must NOT fire.

    This test FAILS against the pre-fix code because the gate does not exist
    and every qualifying artifact unconditionally routes to preempt.
    """
    patch_yalayut([fake_artifact(
        artifact_id=18, kind="shell_recipe", mechanizable=True, vet_tier=0,
        score=1.0, name="cc-pypackage",
        inputs_schema={},  # non-parametric → would be fully bound
    )])
    out = await do_flash(sample_task)
    # Gate is off — must NOT route to mechanical lane.
    assert "runner" not in out or out.get("runner") != "mechanical", (
        "PHASE2_PREEMPT_ENABLED is False; flash must not set runner=mechanical"
    )
    # Must appear in the skills envelope as inject instead.
    skills = out["skills"]
    assert len(skills) >= 1, (
        "preempt downgraded to inject must appear in task['skills']"
    )
    inject_entries = [s for s in skills if s["exposure_class"] == "inject"]
    assert inject_entries, (
        f"expected at least one inject entry in skills; got {skills}"
    )
    assert inject_entries[0]["artifact_id"] == 18


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
