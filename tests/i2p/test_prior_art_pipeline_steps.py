import json


def _steps():
    d = json.load(open("src/workflows/i2p/i2p_v3.json", encoding="utf-8"))
    out = {}
    def walk(o):
        if isinstance(o, dict):
            if o.get("id"):
                out[o["id"]] = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return out


def test_old_single_step_removed():
    s = _steps()
    assert "1.0" not in s
    assert {"1.0a", "1.0b", "1.0c"} <= set(s)


def test_pipeline_wiring():
    s = _steps()
    assert s["1.0a"]["depends_on"] == ["0.6"]
    assert s["1.0a"]["agent"] == "query_planner"
    assert s["1.0b"]["agent"] == "mechanical"
    assert s["1.0b"]["depends_on"] == ["1.0a"]
    assert s["1.0b"]["payload"]["action"] == "prior_art_fetch"
    assert s["1.0c"]["agent"] == "prior_art_synthesizer"
    assert s["1.0c"]["depends_on"] == ["1.0b"]
    assert "prior_art_report" in s["1.0c"]["output_artifacts"]
    assert s["1.0c"]["post_hooks"] == ["prior_art_min_coverage"]
    assert any("prior_art_report.json" in p for p in s["1.0c"]["produces"])
    assert "candidates_path" in s["1.0c"].get("context", {})


def test_json_still_valid():
    json.load(open("src/workflows/i2p/i2p_v3.json", encoding="utf-8"))
