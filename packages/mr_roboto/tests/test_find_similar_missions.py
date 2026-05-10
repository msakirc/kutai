"""Tests for find_similar_missions — Z1 T6A (A7) idea fingerprint dedup."""
from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mr_roboto.find_similar_missions as fsm_module
from mr_roboto.find_similar_missions import (
    DEFAULT_SIMILARITY_THRESHOLD,
    _threshold,
    find_similar_missions,
    index_idea_fingerprint,
)


def _fake_chroma_collection(matches: list[dict] | None = None, count: int = 0):
    """Build a fake chroma collection that returns the given matches."""
    col = MagicMock()
    col.count.return_value = count

    def _query(query_embeddings, n_results, **kwargs):
        if not matches:
            return {"ids": [[]], "metadatas": [[]], "distances": [[]]}
        ids = [m["doc_id"] for m in matches]
        metas = [m.get("meta", {}) for m in matches]
        # `distance = 1 - similarity` for cosine.
        dists = [1.0 - m["similarity"] for m in matches]
        return {"ids": [ids], "metadatas": [metas], "distances": [dists]}

    col.query.side_effect = _query
    col.upsert = MagicMock()
    return col


def test_threshold_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("KUTAI_IDEA_DEDUP_THRESHOLD", raising=False)
    assert _threshold() == DEFAULT_SIMILARITY_THRESHOLD


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("KUTAI_IDEA_DEDUP_THRESHOLD", "0.7")
    assert _threshold() == 0.7


def test_threshold_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("KUTAI_IDEA_DEDUP_THRESHOLD", "not-a-float")
    assert _threshold() == DEFAULT_SIMILARITY_THRESHOLD


@pytest.mark.asyncio
async def test_no_idea_text_returns_ok_unchecked(tmp_path):
    res = await find_similar_missions(
        mission_id=1,
        idea_summary=None,
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert res["matches"] == []
    assert res["checked"] is False


@pytest.mark.asyncio
async def test_empty_collection_returns_ok(tmp_path):
    fake_col = _fake_chroma_collection(matches=None, count=0)
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=fake_col),
    ):
        res = await find_similar_missions(
            mission_id=1,
            idea_summary="A coffee subscription for night owls",
            workspace_path=str(tmp_path),
        )
    assert res["ok"] is True
    assert res["matches"] == []
    assert res["checked"] is True
    assert res["reason"] == "empty collection"


@pytest.mark.asyncio
async def test_match_above_threshold_returns_needs_review(tmp_path):
    fake_col = _fake_chroma_collection(
        matches=[
            {
                "doc_id": "mission_12",
                "similarity": 0.92,
                "meta": {
                    "mission_id": 12,
                    "title": "Coffee for night owls",
                    "final_status_note": "killed at phase 4: no demand",
                },
            },
        ],
        count=1,
    )
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=fake_col),
    ), patch(
        "mr_roboto.find_similar_missions._embed",
        new=AsyncMock(return_value=[0.1] * 768),
    ):
        res = await find_similar_missions(
            mission_id=42,
            idea_summary="Coffee subscription night",
            workspace_path=str(tmp_path),
        )
    assert res["ok"] is False
    assert len(res["matches"]) == 1
    assert res["matches"][0]["mission_id"] == 12
    assert res["matches"][0]["similarity"] >= 0.85
    # Report file emitted
    assert res["report_path"]
    assert os.path.isfile(res["report_path"])
    body = open(res["report_path"], encoding="utf-8").read()
    assert "Coffee for night owls" in body
    assert "killed at phase 4" in body


@pytest.mark.asyncio
async def test_match_below_threshold_returns_ok(tmp_path):
    fake_col = _fake_chroma_collection(
        matches=[
            {
                "doc_id": "mission_5",
                "similarity": 0.40,
                "meta": {
                    "mission_id": 5,
                    "title": "Pet rocks",
                },
            },
        ],
        count=1,
    )
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=fake_col),
    ), patch(
        "mr_roboto.find_similar_missions._embed",
        new=AsyncMock(return_value=[0.0] * 768),
    ):
        res = await find_similar_missions(
            mission_id=42,
            idea_summary="Coffee subscription night",
            workspace_path=str(tmp_path),
        )
    assert res["ok"] is True
    assert len(res["matches"]) == 1
    assert res["report_path"] is None


@pytest.mark.asyncio
async def test_self_mission_excluded_from_matches(tmp_path):
    """A mission must never match its own embedding."""
    fake_col = _fake_chroma_collection(
        matches=[
            {
                "doc_id": "mission_42",  # self
                "similarity": 0.99,
                "meta": {"mission_id": 42, "title": "Self"},
            },
            {
                "doc_id": "mission_12",
                "similarity": 0.50,
                "meta": {"mission_id": 12, "title": "Other"},
            },
        ],
        count=2,
    )
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=fake_col),
    ), patch(
        "mr_roboto.find_similar_missions._embed",
        new=AsyncMock(return_value=[0.0] * 768),
    ):
        res = await find_similar_missions(
            mission_id=42,
            idea_summary="hello",
            workspace_path=str(tmp_path),
        )
    # Self-row dropped — only mission_12 survives, below threshold so ok.
    assert res["ok"] is True
    assert all(m["mission_id"] != 42 for m in res["matches"])
    assert any(m["mission_id"] == 12 for m in res["matches"])


@pytest.mark.asyncio
async def test_threshold_param_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("KUTAI_IDEA_DEDUP_THRESHOLD", "0.50")
    fake_col = _fake_chroma_collection(
        matches=[{
            "doc_id": "mission_9",
            "similarity": 0.60,
            "meta": {"mission_id": 9, "title": "match"},
        }],
        count=1,
    )
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=fake_col),
    ), patch(
        "mr_roboto.find_similar_missions._embed",
        new=AsyncMock(return_value=[0.0] * 768),
    ):
        # With env=0.50, sim 0.60 breaches → not ok.
        res_env = await find_similar_missions(
            mission_id=1, idea_summary="x", workspace_path=str(tmp_path),
        )
        assert res_env["ok"] is False
        # Explicit threshold=0.99 → 0.60 below → ok.
        res_param = await find_similar_missions(
            mission_id=1,
            idea_summary="x",
            workspace_path=str(tmp_path),
            threshold=0.99,
        )
        assert res_param["ok"] is True


@pytest.mark.asyncio
async def test_uses_charter_text_when_no_idea_summary(tmp_path):
    charter_dir = tmp_path / ".charter"
    charter_dir.mkdir()
    (charter_dir / "product_charter.md").write_text(
        "# Charter\n\n## Brand Keywords\n- **Speed** — fast\n",
        encoding="utf-8",
    )
    fake_col = _fake_chroma_collection(matches=None, count=0)
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=fake_col),
    ):
        res = await find_similar_missions(
            mission_id=1, workspace_path=str(tmp_path),
        )
    assert res["checked"] is True


@pytest.mark.asyncio
async def test_collection_unavailable_returns_ok_unchecked(tmp_path):
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=None),
    ):
        res = await find_similar_missions(
            mission_id=1,
            idea_summary="hello",
            workspace_path=str(tmp_path),
        )
    assert res["ok"] is True
    assert res["checked"] is False


@pytest.mark.asyncio
async def test_index_idea_fingerprint_stores_doc(tmp_path):
    fake_col = _fake_chroma_collection()
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=fake_col),
    ), patch(
        "mr_roboto.find_similar_missions._embed",
        new=AsyncMock(return_value=[0.5] * 768),
    ):
        res = await index_idea_fingerprint(
            mission_id=42,
            idea_summary="My fresh idea",
            workspace_path=str(tmp_path),
            title="Fresh idea",
        )
    assert res["ok"] is True
    assert res["doc_id"] == "mission_42"
    assert fake_col.upsert.called
    kwargs = fake_col.upsert.call_args.kwargs
    assert kwargs["ids"] == ["mission_42"]
    assert kwargs["metadatas"][0]["mission_id"] == 42
    assert kwargs["metadatas"][0]["title"] == "Fresh idea"


@pytest.mark.asyncio
async def test_dispatch_through_run_completed(tmp_path):
    from mr_roboto import run
    fake_col = _fake_chroma_collection(matches=None, count=0)
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=fake_col),
    ):
        task = {
            "id": 1,
            "mission_id": 7,
            "payload": {
                "action": "find_similar_missions",
                "idea_summary": "Generic idea",
                "workspace_path": str(tmp_path),
            },
        }
        result = await run(task)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_dispatch_through_run_needs_review(tmp_path):
    from mr_roboto import run
    fake_col = _fake_chroma_collection(
        matches=[{
            "doc_id": "mission_3",
            "similarity": 0.95,
            "meta": {"mission_id": 3, "title": "Prior", "final_status_note": "abandoned"},
        }],
        count=1,
    )
    with patch(
        "mr_roboto.find_similar_missions._ensure_collection",
        new=AsyncMock(return_value=fake_col),
    ), patch(
        "mr_roboto.find_similar_missions._embed",
        new=AsyncMock(return_value=[0.1] * 768),
    ):
        task = {
            "id": 1,
            "mission_id": 7,
            "payload": {
                "action": "find_similar_missions",
                "idea_summary": "Hello world",
                "workspace_path": str(tmp_path),
            },
        }
        result = await run(task)
    assert result.status == "needs_review"
    assert "matches=1" in (result.error or "")
