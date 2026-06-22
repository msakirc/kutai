import json
from pathlib import Path

WF = Path("src/workflows/i2p/i2p_v3.json")


def _steps():
    data = json.loads(WF.read_text(encoding="utf-8"))
    return data["steps"], {s["id"]: s for s in data["steps"]}


def test_naming_step_exists_and_is_object_no_produces():
    steps, by_id = _steps()
    s = by_id["0.0y"]
    assert s["agent"] == "analyst"
    assert s["depends_on"] == []
    assert s["input_artifacts"] == ["raw_idea", "strategic_context"]
    assert s["output_artifacts"] == ["product_name"]
    assert "produces" not in s
    assert s["artifact_schema"]["product_name"]["type"] == "object"
    assert "product_name" in s["artifact_schema"]["product_name"]["required_fields"]
    assert s.get("requires_grading") is not False
    assert "tools_hint" in s and "difficulty" in s


def test_0_0y_is_before_0_0z_in_array():
    steps, _ = _steps()
    ids = [s["id"] for s in steps]
    assert ids.index("0.0y") < ids.index("0.0z")


def test_0_0z_depends_on_naming_step():
    _, by_id = _steps()
    assert by_id["0.0z"]["depends_on"] == ["0.0y"]


def test_checks_declared_on_pitch_and_charter():
    _, by_id = _steps()
    for sid, art in (("0.0z", "reverse_pitch.md"), ("0.1", "product_charter.md")):
        kinds = [c["kind"] for c in by_id[sid].get("checks", [])]
        assert "verify_contains_product_name" in kinds
        chk = next(c for c in by_id[sid]["checks"]
                   if c["kind"] == "verify_contains_product_name")
        assert chk["payload"]["action"] == "verify_contains_product_name"
        assert any(art in p for p in chk["payload"]["artifact_paths"])
