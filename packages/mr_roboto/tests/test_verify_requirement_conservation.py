"""Requirement-conservation checker — assert an assembled artifact carries
EVERY requirement id present in its upstream source artifact(s).

Root cause (m90): the `easy`/`medium`-tier assembly steps (3.9a traceability,
3.10a spec-part1, 3.10b final spec) silently compress a long requirement list
— handed 15 FRs they emit 11, dropping the newest ids. This deterministic
gate catches the drop at the producer and re-pends it (mirrors
verify_falsification_present), so the mission self-heals instead of halting.
"""
from mr_roboto.verify_requirement_conservation import (
    verify_requirement_conservation,
)


def test_all_ids_present_passes():
    res = verify_requirement_conservation(
        produced_text="## FR\nFR-001 ... FR-002 ... FR-003 ...",
        sources=[{"label": "functional_requirements",
                  "source_text": "FR-001 FR-002 FR-003",
                  "id_pattern": r"FR-\d+"}],
    )
    assert res["ok"] is True
    assert res["missing"] == []


def test_dropped_ids_fail_and_are_named():
    # Source defines FR-001..FR-015, produced carries only FR-001..FR-011.
    src = " ".join(f"FR-{i:03d}" for i in range(1, 16))
    prod = " ".join(f"FR-{i:03d}" for i in range(1, 12))
    res = verify_requirement_conservation(
        produced_text=prod,
        sources=[{"label": "functional_requirements",
                  "source_text": src, "id_pattern": r"FR-\d+"}],
    )
    assert res["ok"] is False
    miss = res["missing"][0]
    assert miss["label"] == "functional_requirements"
    assert miss["missing_ids"] == ["FR-012", "FR-013", "FR-014", "FR-015"]


def test_multi_source_each_pattern_checked():
    # FRs all carried, but a user story was dropped.
    res = verify_requirement_conservation(
        produced_text="FR-001 FR-002 US-001",
        sources=[
            {"label": "functional_requirements",
             "source_text": "FR-001 FR-002", "id_pattern": r"FR-\d+"},
            {"label": "user_stories_refined",
             "source_text": "US-001 US-002", "id_pattern": r"US-\d+"},
        ],
    )
    assert res["ok"] is False
    labels = {m["label"]: m["missing_ids"] for m in res["missing"]}
    assert "functional_requirements" not in labels  # FRs fully conserved
    assert labels["user_stories_refined"] == ["US-002"]


def test_extra_ids_in_produced_still_passes():
    # Conservation requires source ⊆ produced; extra produced ids are fine.
    res = verify_requirement_conservation(
        produced_text="FR-001 FR-002 FR-003 FR-004",
        sources=[{"label": "functional_requirements",
                  "source_text": "FR-001 FR-002", "id_pattern": r"FR-\d+"}],
    )
    assert res["ok"] is True


def test_pattern_does_not_cross_match():
    # An FR pattern must not be satisfied by a US id in the produced doc.
    res = verify_requirement_conservation(
        produced_text="US-001 US-002",  # no FRs at all
        sources=[{"label": "functional_requirements",
                  "source_text": "FR-001", "id_pattern": r"FR-\d+"}],
    )
    assert res["ok"] is False
    assert res["missing"][0]["missing_ids"] == ["FR-001"]


def test_empty_source_is_vacuous_pass_flagged():
    # A source with zero matching ids has nothing to conserve. Do NOT block
    # the producer (safe direction — the reviewer is the backstop); flag empty
    # so a wiring bug (wrong path → empty read) is observable.
    res = verify_requirement_conservation(
        produced_text="anything",
        sources=[{"label": "functional_requirements",
                  "source_text": "no ids here", "id_pattern": r"FR-\d+"}],
    )
    assert res["ok"] is True
    assert res["empty"] is True


def test_no_sources_is_vacuous_pass():
    res = verify_requirement_conservation(produced_text="x", sources=[])
    assert res["ok"] is True
    assert res["empty"] is True


# --- mr_roboto dispatch (reads produced + source artifacts off disk) ---

import pytest  # noqa: E402


@pytest.mark.asyncio
async def test_dispatch_fails_when_produced_drops_source_ids(tmp_path, monkeypatch):
    import mr_roboto
    src = tmp_path / "functional_requirements.md"
    src.write_text(" ".join(f"FR-{i:03d}" for i in range(1, 16)), encoding="utf-8")
    prod = tmp_path / "requirements_spec_part1.md"
    prod.write_text(" ".join(f"FR-{i:03d}" for i in range(1, 12)), encoding="utf-8")
    monkeypatch.setattr(mr_roboto, "_resolve_path_list",
                        lambda paths: [str(p) for p in (paths or [])])
    task = {"id": 1, "mission_id": 90, "payload": {
        "action": "verify_requirement_conservation",
        "produced_paths": [str(prod)],
        "sources": [{"label": "functional_requirements",
                     "source_paths": [str(src)], "id_pattern": r"FR-\d+"}],
    }}
    res = await mr_roboto._run_dispatch(task)
    assert res.status == "failed"
    assert "FR-012" in (res.error or "")


@pytest.mark.asyncio
async def test_dispatch_passes_when_all_conserved(tmp_path, monkeypatch):
    import mr_roboto
    ids = " ".join(f"FR-{i:03d}" for i in range(1, 16))
    src = tmp_path / "functional_requirements.md"
    src.write_text(ids, encoding="utf-8")
    prod = tmp_path / "requirements_spec_part1.md"
    prod.write_text("## Functional Requirements\n" + ids, encoding="utf-8")
    monkeypatch.setattr(mr_roboto, "_resolve_path_list",
                        lambda paths: [str(p) for p in (paths or [])])
    task = {"id": 1, "mission_id": 90, "payload": {
        "action": "verify_requirement_conservation",
        "produced_paths": [str(prod)],
        "sources": [{"label": "functional_requirements",
                     "source_paths": [str(src)], "id_pattern": r"FR-\d+"}],
    }}
    res = await mr_roboto._run_dispatch(task)
    assert res.status == "completed"
    assert res.result["ok"] is True


@pytest.mark.asyncio
async def test_dispatch_flags_vacuous_pass_when_paths_unresolved(tmp_path, monkeypatch):
    # Declared paths that resolve to nothing (typo / unmaterialized artifact /
    # unsubstituted {mission_id}) → no ids read → empty vacuous PASS. Safe
    # direction (no false re-pend) but it silently disables the gate — the
    # dispatch must FLAG it (wiring_suspect) so the foot-gun is observable
    # (mission_90: unsubstituted {mission_id} went unnoticed exactly this way).
    import mr_roboto
    monkeypatch.setattr(mr_roboto, "_resolve_path_list", lambda paths: [])
    task = {"id": 1, "mission_id": 90, "payload": {
        "action": "verify_requirement_conservation",
        "produced_paths": ["mission_90/requirements_spec.md"],
        "sources": [{"label": "functional_requirements",
                     "source_paths": ["mission_90/functional_requirements.md"],
                     "id_pattern": r"FR-\d+"}],
    }}
    res = await mr_roboto._run_dispatch(task)
    assert res.status == "completed"          # safe direction
    assert res.result.get("wiring_suspect") is True


def test_verify_requirement_conservation_is_full_reversibility():
    from mr_roboto.reversibility import get_reversibility
    assert get_reversibility("verify_requirement_conservation") == "full"
