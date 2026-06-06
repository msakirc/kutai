import asyncio
import json
import mr_roboto
from mr_roboto.prior_art_fetch import prior_art_fetch


def test_prior_art_fetch_reads_queries_writes_candidates(tmp_path, monkeypatch):
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps({
        "queries": ["habit tracker app"],
        "domain_keywords": ["habit tracker"],
        "ambition_tier": "private_beta",
    }), encoding="utf-8")
    cand_path = tmp_path / "candidates.json"

    async def fake_fetch(queries, domain_keywords=None, k=10,
                         ambition_tier="private_beta", **kw):
        assert queries == ["habit tracker app"]
        return {"candidates": [{"name": "Habitica",
                                "url": "https://habitica.com",
                                "status": "unknown", "sources": []}],
                "search_summary": {"queries_run": queries,
                                   "total_results_inspected": 1}}

    monkeypatch.setattr("src.research.prior_art.fetch_candidates", fake_fetch)

    res = asyncio.run(prior_art_fetch(
        queries_path=str(queries_path),
        candidates_path=str(cand_path)))
    assert res["ok"] is True
    assert res["candidate_count"] == 1
    written = json.loads(cand_path.read_text(encoding="utf-8"))
    assert written["candidates"][0]["url"] == "https://habitica.com"


def test_prior_art_fetch_via_dispatch(tmp_path, monkeypatch):
    qp = tmp_path / "q.json"; qp.write_text('{"queries":["x"]}', encoding="utf-8")
    cp = tmp_path / "c.json"

    async def fake_fetch(queries, **kw):
        return {"candidates": [], "search_summary": {"queries_run": queries}}
    monkeypatch.setattr("src.research.prior_art.fetch_candidates", fake_fetch)

    action = asyncio.run(mr_roboto.run({
        "id": 1, "mission_id": 80, "title": "t",
        "payload": {"action": "prior_art_fetch",
                    "queries_path": str(qp), "candidates_path": str(cp)},
    }))
    assert action.status == "completed"
