"""End-to-end Phase 2: task → intersect.flash → envelope → coulson prompt.

Verifies the full consumer path is wired with no dangling fragment:
  - intersect.flash attaches task["skills"]
  - coulson.build_user_context renders it into the agent prompt
  - the skills.py shim returns the same inject slice
  - a preempt artifact routes the task to the mechanical lane
"""
import json

import pytest


class _Profile:
    name = "coder"
    allowed_tools = ["read_file", "write_file"]
    max_iterations = 5
    _prompt_version_override = None
    _suppress_clarification = False

    def get_system_prompt(self, task):
        return "You are a coder."


class _Art:
    """Stub artifact with addendum-corrected attribute names."""
    def __init__(self, **kw):
        self.artifact_id = kw.get("artifact_id", kw.get("id", 1))
        self.artifact_type = kw.get("artifact_type", "skill")
        self.kind = kw.get("kind", "prompt_skill")
        self.name = kw.get("name", "anthropics-pdf")
        self.source = kw.get("source", "github:anthropics/skills@/skills")
        self.owner = kw.get("owner", "anthropics")
        self.vet_tier = kw.get("vet_tier", 0)
        self.score = kw.get("score", kw.get("vector_sim", 0.9))
        self.mechanizable = kw.get("mechanizable", False)
        self.applies_to = "execution"
        self.intent_keywords = kw.get("intent_keywords", [])
        self.inputs_schema = kw.get("inputs_schema", {})
        self.body_excerpt = kw.get("body_excerpt", kw.get("body", "PDF skill body"))
        self.env_status = "ready"


@pytest.fixture
async def e2e_db(monkeypatch):
    import aiosqlite
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(
        """
        CREATE TABLE yalayut_sources (source_id TEXT, trust_score REAL);
        CREATE TABLE yalayut_owners (owner_id TEXT, trust_score REAL);
        CREATE TABLE yalayut_usage (
          id INTEGER PRIMARY KEY, artifact_id INTEGER, task_id TEXT,
          exposure_class TEXT, bind_args_json TEXT, exposed BOOLEAN,
          called BOOLEAN, succeeded BOOLEAN, latency_ms INTEGER,
          conflict_loser BOOLEAN, would_have_used INTEGER,
          escape_reason TEXT, occurred_at TIMESTAMP);
        CREATE TABLE yalayut_bind_cache (
          id INTEGER PRIMARY KEY, manifest_id INTEGER, ctx_embedding BLOB,
          bound_args_json TEXT, hit_count INTEGER DEFAULT 0,
          created_at TIMESTAMP, last_used_at TIMESTAMP);
        """
    )
    await conn.execute(
        "INSERT INTO yalayut_sources VALUES "
        "('github:anthropics/skills@/skills', 1.0)")
    await conn.execute(
        "INSERT INTO yalayut_owners VALUES ('anthropics', 1.0)")
    await conn.commit()

    async def _get_db():
        return conn

    import src.infra.db as _db
    monkeypatch.setattr(_db, "get_db", _get_db)
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_full_path_task_to_agent_prompt(e2e_db, monkeypatch):
    import intersect
    import yalayut
    from coulson import context

    async def _query(task_ctx):
        return [_Art(artifact_id=1, name="anthropics-pdf",
                     body_excerpt="Use the PDF skill.")]

    monkeypatch.setattr(yalayut, "query", _query, raising=False)

    task = {
        "id": 9001, "title": "Extract text from a PDF report",
        "description": "parse the quarterly PDF",
        "agent_type": "coder",
        "context": json.dumps({"recipe_lookup": True}),
    }

    # 1. orchestrator pump would call this:
    task = await intersect.flash(task)
    assert task["skills"], "intersect attached an empty envelope"
    assert task["skills"][0]["exposure_class"] == "inject"

    # 2. coulson renders it into the agent prompt:
    prompt, injected_tools = await context.build_user_context(
        _Profile(), task, model_ctx=4096)
    assert "anthropics-pdf" in prompt
    assert "Use the PDF skill." in prompt

    # 3. skills.py shim sees the same inject slice:
    from src.memory import skills
    shim = await skills.find_relevant_skills("x", task=task)
    assert len(shim) == 1
    assert shim[0]["name"] == "anthropics-pdf"

    # 4. telemetry row written:
    cur = await e2e_db.execute(
        "SELECT exposure_class FROM yalayut_usage WHERE task_id = '9001'")
    rows = await cur.fetchall()
    assert rows and rows[0][0] == "inject"


@pytest.mark.asyncio
async def test_preempt_routes_task_to_mechanical_lane(e2e_db, monkeypatch):
    """Preempt routing to the mechanical lane works when the Phase 3 gate is
    explicitly enabled.  The gate (PHASE2_PREEMPT_ENABLED) is False by default;
    we monkeypatch it True here so the routing code stays tested end-to-end.
    The yalayut_recipe executor lands in Phase 3 — see flash.PHASE2_PREEMPT_ENABLED.
    """
    import sys
    import intersect
    import yalayut
    flash_mod = sys.modules["intersect.flash"]
    monkeypatch.setattr(flash_mod, "PHASE2_PREEMPT_ENABLED", True)

    async def _query(task_ctx):
        return [_Art(artifact_id=18, name="cc-pypackage", kind="shell_recipe",
                     mechanizable=True, score=1.0, inputs_schema={},
                     source="github:cookiecutter/cookiecutter-pypackage",
                     owner="cookiecutter")]

    monkeypatch.setattr(yalayut, "query", _query, raising=False)

    task = {
        "id": 9002, "title": "Scaffold the Python package",
        "description": "create the skeleton", "agent_type": "coder",
        "context": json.dumps({"recipe_lookup": True}),
    }
    task = await intersect.flash(task)
    assert task["skills"] == []
    assert task["runner"] == "mechanical"
    assert task["payload"]["action"] == "yalayut_recipe"
    assert task["payload"]["recipe_id"] == 18
