"""P7 — reviewer regression fixtures.

Locks the structural contract of the five phase 0-6 reviewer steps:

    1.13  research_quality_review
    3.11  requirements_review
    4.16  architecture_review
    5.10  design_review
    6.6   project_plan_review

Each fixture is a JSON file under ``fixtures/v<schema_version>/<step_id>/``
and pairs:
  - ``input_artifacts``: shapes the reviewer would read off the blackboard,
  - ``expected_verdict``: the reviewer's structured response.

The test does NOT call any LLM. It uses a deterministic stub that returns
the fixture's ``expected_verdict`` and asserts:

  1. The stub's response parses cleanly through coulson's ``parse_action`` /
     direct JSON path (i.e. the reviewer prompt's output shape is well-formed).
  2. The verdict carries ``_schema_version`` matching the workflow step's
     declared expectation (the actual P7 contract).
  3. The verdict matches the structural schema declared in the workflow JSON
     (verdict/status enum from ``artifact_schema[<output>].fields``).
  4. The mechanical ``verify_schema_version`` action accepts the fixture
     when run against ``(input_artifacts ∪ {output: expected_verdict})``.

A future schema bump (``"1"`` → ``"2"``) MUST land alongside ``fixtures/v2/``
copies; otherwise this suite fails and blocks merge.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────────────
# Locate workflow + fixtures relative to repo root
# ────────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_FIXTURES_ROOT = _HERE / "fixtures"
_REPO_ROOT = _HERE.parent.parent.parent
_WORKFLOW = _REPO_ROOT / "src" / "workflows" / "i2p" / "i2p_v3.json"

REVIEWER_STEP_IDS = ("1.13", "3.11", "4.16", "5.10", "6.6")


def _load_workflow() -> dict:
    return json.loads(_WORKFLOW.read_text(encoding="utf-8"))


def _step_by_id(wf: dict, step_id: str) -> dict:
    for s in wf.get("steps", []):
        if s.get("id") == step_id:
            return s
    raise KeyError(step_id)


def _discover_fixtures() -> list[tuple[str, str, Path]]:
    """Return ``[(schema_version, step_id, fixture_path), ...]``.

    Skips fixtures whose stem looks like a Z1-Tier-2 ADR-shape companion
    (``good_adr_set``, ``bad_adr_set``, ``good_component_library``,
    ``bad_component_library``); those are loaded by
    ``tests/i2p/test_adr_shape.py`` and have a different shape than the
    reviewer-regression contract.
    """
    out: list[tuple[str, str, Path]] = []
    if not _FIXTURES_ROOT.is_dir():
        return out
    skip_stems = {
        "good_adr_set",
        "bad_adr_set",
        "good_component_library",
        "bad_component_library",
        # Z1 Tier 3 (T3A) — design-tokens / taste fixtures.
        "good_design_tokens",
        "bad_design_tokens",
        "good_taste_emphasis",
        "bad_taste_emphasis",
        # Z1 Tier 3 (T3B) — surfaces / user_flow / screen_inventory / shared_shell.
        "good_surfaces",
        "bad_surfaces",
        "good_user_flow",
        "bad_user_flow",
        "good_screen_inventory",
        "bad_screen_inventory",
        "good_shared_shell",
        "bad_shared_shell",
        # Z1 Tier 3 (T3C: C3+A10+C9+A11+C18) — per-screen plans / HTML prototypes / consistency.
        "good_screen_plan",
        "bad_screen_plan",
        "good_html_prototype",
        "bad_html_prototype",
        "good_screen_consistency",
        "bad_screen_consistency",
    }
    for vdir in sorted(_FIXTURES_ROOT.iterdir()):
        if not vdir.is_dir() or not vdir.name.startswith("v"):
            continue
        ver = vdir.name[1:]
        for step_dir in sorted(vdir.iterdir()):
            if not step_dir.is_dir():
                continue
            # dir name is step id with `.` → `_`
            step_id = step_dir.name.replace("_", ".")
            for fx in sorted(step_dir.glob("*.json")):
                if fx.stem in skip_stems:
                    continue
                out.append((ver, step_id, fx))
    return out


def _discover_adr_companion_fixtures() -> list[Path]:
    """Return Z1-Tier-2 ADR/component-library fixtures discovered alongside
    reviewer fixtures (loaded by tests/i2p/test_adr_shape.py)."""
    out: list[Path] = []
    if not _FIXTURES_ROOT.is_dir():
        return out
    for vdir in sorted(_FIXTURES_ROOT.iterdir()):
        if not vdir.is_dir():
            continue
        for step_dir in sorted(vdir.iterdir()):
            if not step_dir.is_dir():
                continue
            for stem in (
                "good_adr_set",
                "bad_adr_set",
                "good_component_library",
                "bad_component_library",
            ):
                p = step_dir / f"{stem}.json"
                if p.exists():
                    out.append(p)
    return out


_FIXTURES = _discover_fixtures()


# ────────────────────────────────────────────────────────────────────────────
# Suite-level invariants
# ────────────────────────────────────────────────────────────────────────────


def test_fixtures_cover_every_reviewer_step():
    """Each of the 5 reviewer steps has at least one good + one bad fixture."""
    by_step: dict[str, set[str]] = {sid: set() for sid in REVIEWER_STEP_IDS}
    for _ver, step_id, path in _FIXTURES:
        if step_id in by_step:
            by_step[step_id].add(path.stem)
    missing = {
        sid: list({"good", "bad"} - kinds) for sid, kinds in by_step.items() if {"good", "bad"} - kinds
    }
    assert not missing, f"reviewer steps missing fixtures: {missing}"


def test_fixture_count_meets_minimum():
    """Acceptance criterion: at least 5 reviewer steps × 2 fixtures = 10."""
    assert len(_FIXTURES) >= 10, f"fixture count {len(_FIXTURES)} below 10"


def test_z1_tier2_adr_companion_fixtures_present():
    """Z1 Tier 2 (P3+C7) — verify the four ADR/component-library
    companion fixtures are wired alongside reviewer 4.16."""
    companions = _discover_adr_companion_fixtures()
    stems = {p.stem for p in companions}
    expected = {
        "good_adr_set",
        "bad_adr_set",
        "good_component_library",
        "bad_component_library",
    }
    assert expected.issubset(stems), f"missing ADR companions: {expected - stems}"


# ────────────────────────────────────────────────────────────────────────────
# Stub LLM
# ────────────────────────────────────────────────────────────────────────────


class _StubLLM:
    """Deterministic LLM stand-in. Returns the fixture's expected_verdict
    serialized as a JSON ``final_answer`` envelope (the canonical shape that
    coulson's parser handles)."""

    def __init__(self, expected_verdict: dict):
        self.expected_verdict = expected_verdict
        self.calls = 0

    def respond(self) -> str:
        self.calls += 1
        return json.dumps({
            "action": "final_answer",
            "result": json.dumps(self.expected_verdict),
        })


# ────────────────────────────────────────────────────────────────────────────
# Per-fixture parameterized test
# ────────────────────────────────────────────────────────────────────────────


def _fixture_id(param) -> str:
    ver, step_id, path = param
    return f"v{ver}-{step_id}-{path.stem}"


@pytest.mark.parametrize("fixture", _FIXTURES, ids=_fixture_id)
def test_reviewer_fixture(fixture):
    """Lock the (artifact_name, _schema_version) reviewer contract.

    Verifies: stub-LLM response parses, carries _schema_version, matches
    the workflow step's structural schema, and the mechanical schema-version
    verifier accepts the artifacts.
    """
    from coulson.parsing import parse_action  # late import, suite-collected

    ver, step_id, path = fixture
    payload = json.loads(path.read_text(encoding="utf-8"))

    wf = _load_workflow()
    step = _step_by_id(wf, step_id)
    output_name = step["output_artifacts"][0]
    schema = step["artifact_schema"][output_name]

    # Invariant 1 — workflow declares _schema_version on this artifact.
    declared_version = schema.get("_schema_version")
    assert declared_version is not None, (
        f"step {step_id} artifact {output_name} missing _schema_version in workflow"
    )
    # Fixture version must match its directory.
    assert payload.get("schema_version") == declared_version == ver, (
        f"fixture {path.name} version mismatch: "
        f"fixture={payload.get('schema_version')} "
        f"workflow={declared_version} dir=v{ver}"
    )

    # Stub LLM returns the expected verdict.
    stub = _StubLLM(payload["expected_verdict"])
    raw = stub.respond()

    # Invariant 2 — coulson can parse the response cleanly.
    action = parse_action(raw)
    assert action is not None, f"parse_action returned None for {path.name}"
    assert action.get("action") == "final_answer", action
    body_text = action.get("result", "")
    assert body_text, "empty result body"
    body = json.loads(body_text)

    # Invariant 3 — verdict has expected `_schema_version`.
    assert body.get("_schema_version") == declared_version, (
        f"verdict missing/wrong _schema_version: "
        f"got={body.get('_schema_version')} expected={declared_version}"
    )

    # Invariant 4 — verdict's pass/fail field obeys the workflow enum when
    # the fixture is the `good` one. Bad fixtures are allowed to use any
    # value outside the enum (they represent reviewer rejections).
    fields = schema.get("fields") or {}
    # The verdict-bearing field is named `verdict` (1.13) or `status`
    # (3.11/4.16/5.10/6.6) — pick whichever is declared.
    verdict_field = next(
        (k for k in ("verdict", "status") if k in fields and "equals" in fields[k]),
        None,
    )
    if verdict_field is not None:
        allowed = list(fields[verdict_field].get("equals") or [])
        actual = body.get(verdict_field)
        # Stem convention: starts with "good" → pass-shape fixture; starts
        # with "bad" → reject-shape fixture. Anything else is unclassified
        # and skipped from the verdict-enum check (the schema-version path
        # below still runs).
        kind = "good" if path.stem.startswith("good") else (
            "bad" if path.stem.startswith("bad") else None
        )
        if kind == "good":
            assert actual in allowed, (
                f"good fixture {path.name} verdict {actual!r} not in allowed {allowed}"
            )
        elif kind == "bad":
            # Bad fixtures should NOT pretend to pass.
            assert actual not in allowed, (
                f"bad fixture {path.name} masquerades as pass with verdict {actual!r}"
            )

    # Invariant 5 — mechanical verify_schema_version accepts the bundle.
    from mr_roboto import run as mr_roboto_run

    expected_versions = {output_name: declared_version}
    artifacts = {output_name: body}
    # Also include the fixture's input artifacts: each declares its own
    # `_schema_version` (per-artifact-name N4 rule). We pick declared
    # version "1" for them — fresh fixtures all live at v1.
    for in_name, in_val in (payload.get("input_artifacts") or {}).items():
        artifacts[in_name] = in_val
        expected_versions[in_name] = "1"

    task = {
        "id": 0,
        "mission_id": 0,
        "payload": {
            "action": "verify_schema_version",
            "artifacts": artifacts,
            "expected_versions": expected_versions,
            "legacy_pre_p7": False,
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "completed", (
        f"verify_schema_version rejected fixture {path.name}: "
        f"error={result.error} result={result.result}"
    )
    assert result.result.get("ok") is True, result.result


# ────────────────────────────────────────────────────────────────────────────
# Mechanical action smoke tests (independent of fixtures)
# ────────────────────────────────────────────────────────────────────────────


def test_verify_schema_version_flags_missing_field():
    from mr_roboto.verify_schema_version import verify_schema_version

    res = verify_schema_version(
        artifacts={"foo": {"data": "x"}},
        expected_versions={"foo": "1"},
    )
    assert res["ok"] is False
    assert res["missing"] == ["foo"]


def test_verify_schema_version_flags_mismatch():
    from mr_roboto.verify_schema_version import verify_schema_version

    res = verify_schema_version(
        artifacts={"foo": {"_schema_version": "2", "data": "x"}},
        expected_versions={"foo": "1"},
    )
    assert res["ok"] is False
    assert res["mismatched"] == [{"name": "foo", "found": "2", "expected": "1"}]


def test_verify_schema_version_legacy_tolerates_missing():
    from mr_roboto.verify_schema_version import verify_schema_version

    res = verify_schema_version(
        artifacts={"foo": {"data": "x"}},
        expected_versions={"foo": "1"},
        legacy_pre_p7=True,
    )
    assert res["ok"] is True
    assert res["missing"] == []


def test_verify_schema_version_legacy_still_flags_mismatch():
    """Even legacy missions must not silently accept a wrong-version artifact."""
    from mr_roboto.verify_schema_version import verify_schema_version

    res = verify_schema_version(
        artifacts={"foo": {"_schema_version": "9", "data": "x"}},
        expected_versions={"foo": "1"},
        legacy_pre_p7=True,
    )
    assert res["ok"] is False
    assert res["mismatched"][0]["name"] == "foo"


def test_verify_schema_version_extracts_from_markdown_fence():
    from mr_roboto.verify_schema_version import verify_schema_version

    md = (
        "Here is the artifact:\n\n"
        "```json\n"
        '{"_schema_version": "1", "x": 1}\n'
        "```\n"
    )
    res = verify_schema_version(
        artifacts={"foo": md},
        expected_versions={"foo": "1"},
    )
    assert res["ok"] is True
