"""Guard — the 1.13 research-quality rubric carries NO illustrative example
strings that a confabulating reviewer echoes back as observed findings.

Mission 90: the reviewer reported `headline promises 'free forever'` and
`placeholder text 'TODO: define boundaries'` as REAL findings — both were
verbatim echoes of the rubric's own parenthetical examples. Abstract criteria
replace them at the source (the verdict-verification pass is the robust fix;
this is the cheap complementary mitigation).
"""
import json
import re


def _step_1_13_instruction() -> str:
    d = json.load(open("src/workflows/i2p/i2p_v3.json", encoding="utf-8"))
    found = {}

    def walk(o):
        if isinstance(o, dict):
            if o.get("id"):
                found[o["id"]] = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(d)
    assert "1.13" in found, "step 1.13 missing from i2p_v3.json"
    return found["1.13"].get("instruction") or ""


def test_no_free_forever_example():
    assert "free forever" not in _step_1_13_instruction().lower()


def test_no_placeholder_token_list():
    instr = _step_1_13_instruction().lower()
    # The literal placeholder-token list ("TODO, TBD, <fill in>, lorem ipsum")
    # is what gets echoed as a fabricated 'TODO: define boundaries' finding.
    assert "lorem ipsum" not in instr
    assert "<fill in>" not in instr
    # standalone TBD token (not the 'tbd' inside 'JTBD'/Jobs-To-Be-Done)
    assert not re.search(r"\btbd\b", instr)
