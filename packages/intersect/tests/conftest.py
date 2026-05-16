"""Shared intersect test fixtures."""
from __future__ import annotations

import json
import types

import pytest


class FakeArtifact:
    """Stand-in for yalayut's Artifact dataclass.

    Phase 1's Artifact carries: artifact_id, artifact_type, kind, name,
    source, owner, vet_tier, mechanizable, applies_to, score,
    inputs_schema, body_excerpt, env_status. We mirror the fields intersect reads.
    Rename map (addendum): id->artifact_id, vector_sim->score, body->body_excerpt.
    """

    def __init__(
        self, *, artifact_id=1, artifact_type="skill", kind="prompt_skill",
        name="anthropics-pdf", source="github:anthropics/skills@/skills",
        owner="anthropics", vet_tier=0, score=0.9, mechanizable=False,
        applies_to="execution", intent_keywords=None, inputs_schema=None,
        body_excerpt="PDF skill body text", env_status="ready",
    ):
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self.kind = kind
        self.name = name
        self.source = source
        self.owner = owner
        self.vet_tier = vet_tier
        self.score = score
        self.mechanizable = mechanizable
        self.applies_to = applies_to
        self.intent_keywords = intent_keywords or []
        self.inputs_schema = inputs_schema or {}
        self.body_excerpt = body_excerpt
        self.env_status = env_status


@pytest.fixture
def fake_artifact():
    return FakeArtifact


@pytest.fixture
def sample_task():
    """A plain workflow-step task dict as the orchestrator pump sees it."""
    return {
        "id": 4101,
        "title": "[3.2] Scaffold the Python package",
        "description": "Create the package skeleton with pyproject + src layout",
        "agent_type": "coder",
        "mission_id": 57,
        "context": json.dumps({
            "is_workflow_step": True,
            "workflow_step_id": "3.2",
            "recipe_lookup": True,
            "recipe_hint": "python package scaffold",
        }),
    }


@pytest.fixture
async def intersect_db(monkeypatch):
    """In-memory aiosqlite DB with the yalayut tables intersect reads/writes.

    Phase 1's migration owns the canonical schema; here we create only the
    columns Phase 2 touches so intersect tests do not depend on Phase 1
    code being importable.
    """
    import aiosqlite

    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(
        """
        CREATE TABLE yalayut_sources (
          source_id TEXT, trust_score REAL DEFAULT 0.3
        );
        CREATE TABLE yalayut_owners (
          owner_id TEXT, trust_score REAL DEFAULT 0.3
        );
        CREATE TABLE yalayut_usage (
          id INTEGER PRIMARY KEY, artifact_id INTEGER, task_id TEXT,
          exposure_class TEXT, bind_args_json TEXT, exposed BOOLEAN,
          called BOOLEAN, succeeded BOOLEAN, latency_ms INTEGER,
          conflict_loser BOOLEAN, would_have_used INTEGER,
          escape_reason TEXT, occurred_at TIMESTAMP
        );
        CREATE TABLE yalayut_bind_cache (
          id INTEGER PRIMARY KEY, manifest_id INTEGER,
          ctx_embedding BLOB, bound_args_json TEXT,
          hit_count INTEGER DEFAULT 0, created_at TIMESTAMP,
          last_used_at TIMESTAMP
        );
        """
    )
    await conn.commit()

    async def _get_db():
        return conn

    # intersect modules read the DB via src.infra.db.get_db; patch it.
    import src.infra.db as _db
    monkeypatch.setattr(_db, "get_db", _get_db)
    yield conn
    await conn.close()
