import json
import pytest


@pytest.mark.asyncio
async def test_dispatch_passes_when_artifact_contains_name(tmp_path, monkeypatch):
    import mr_roboto

    # Fake artifact store returning the pinned name as the JSON STRING the
    # engine actually stores (production truth).
    class _Store:
        async def retrieve(self, mid, name):
            assert name == "product_name"
            return '{"product_name": "FlowState"}'

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    # Make _resolve_path_list return our temp file verbatim (absolute path).
    art = tmp_path / "reverse_pitch.md"
    art.write_text("# Launch\nIntroducing FlowState.", encoding="utf-8")
    monkeypatch.setattr(mr_roboto, "_resolve_path_list", lambda paths: [str(art)])

    task = {
        "id": 1, "mission_id": 42,
        "payload": {
            "action": "verify_contains_product_name",
            "artifact_paths": [str(art)],
        },
    }
    res = await mr_roboto._run_dispatch(task)
    assert res.status == "completed"
    assert res.result["found"] is True


@pytest.mark.asyncio
async def test_dispatch_fails_when_artifact_missing_name(tmp_path, monkeypatch):
    import mr_roboto

    class _Store:
        async def retrieve(self, mid, name):
            return {"product_name": "FlowState"}

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    art = tmp_path / "reverse_pitch.md"
    art.write_text("# Launch\nIntroducing HabitTrack.", encoding="utf-8")
    monkeypatch.setattr(mr_roboto, "_resolve_path_list", lambda paths: [str(art)])

    task = {
        "id": 1, "mission_id": 42,
        "payload": {
            "action": "verify_contains_product_name",
            "artifact_paths": [str(art)],
        },
    }
    res = await mr_roboto._run_dispatch(task)
    assert res.status == "failed"
    assert "FlowState" in (res.error or "")


@pytest.mark.asyncio
async def test_dispatch_skips_when_no_name_pinned(tmp_path, monkeypatch):
    import mr_roboto

    class _Store:
        async def retrieve(self, mid, name):
            return None  # nothing pinned yet

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    art = tmp_path / "reverse_pitch.md"
    art.write_text("anything", encoding="utf-8")
    monkeypatch.setattr(mr_roboto, "_resolve_path_list", lambda paths: [str(art)])

    task = {
        "id": 1, "mission_id": 42,
        "payload": {
            "action": "verify_contains_product_name",
            "artifact_paths": [str(art)],
        },
    }
    res = await mr_roboto._run_dispatch(task)
    assert res.status == "completed"  # defensive skip, never hard-block


def test_verify_contains_product_name_is_full_reversibility():
    from mr_roboto.reversibility import get_reversibility
    assert get_reversibility("verify_contains_product_name") == "full"
