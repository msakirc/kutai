"""Guard against internal workflow phase/lane codes leaking into a
PROPER-NOUN slot of an LLM-rendered instruction.

Root cause, mission 88 / task 524724 (2026-06-21): the `0.0a.draft`
instruction read "Specialise the intake questionnaire to THIS product
(Z1 B1)." "Z1 B1" is an internal phase·block code, but it sat directly
after the word "product" — the slot where a product NAME belongs. The
grader (cerebras/gpt-oss-120b) and the self_reflect model both read
"Z1 B1" as the product's required name and FAILed the writer's correct
output (which used the real product name "HabitTrack") as RELEVANT:NO
("wrong product name"). The step DLQ'd on the worker-attempt cap and
hard-blocked the whole mission at its first content step.

A phase code after an artifact/file name ("intake todo (Z1 B1)",
"reverse pitch (Z1 A1)") is harmless — those ran for dozens of missions.
The danger is ONLY a phase code in a slot a model reads as a proper noun.
This test forbids that specific shape; it does NOT ban traceability codes
elsewhere.
"""

import json
import pathlib
import re

I2P = pathlib.Path("src/workflows/i2p/i2p_v3.json")

# A two-token internal phase/lane code in parentheses, e.g. "(Z1 B1)".
_PHASE_CODE = r"\([A-Z]\d+ [A-Z]\d+\)"

# Proper-noun-priming nouns: when a phase code immediately follows one of
# these, a model reads the code as the entity's NAME and grades against it.
_PRIMING = r"(?:product|feature|app|service|tool|brand|company|platform)"

# "...product (Z1 B1)" — priming noun, whitespace, then the code. The
# optional article slot ("the"/"this"/"a") is already consumed by the
# noun match; we only need noun → code adjacency.
_LEAK_RE = re.compile(rf"{_PRIMING}\s+{_PHASE_CODE}", re.IGNORECASE)

# Fields that are rendered into the prompt an LLM (producer or grader) sees.
_LLM_FIELDS = ("instruction", "description", "done_when", "name")


def _iter_steps(node):
    if isinstance(node, dict):
        if "id" in node and any(k in node for k in _LLM_FIELDS):
            yield node
        for v in node.values():
            yield from _iter_steps(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_steps(v)


def test_no_phase_code_in_proper_noun_slot():
    data = json.loads(I2P.read_text(encoding="utf-8"))
    offenders = []
    for step in _iter_steps(data):
        sid = step.get("id", "?")
        for field in _LLM_FIELDS:
            val = step.get(field)
            if isinstance(val, str):
                for m in _LEAK_RE.finditer(val):
                    a = max(0, m.start() - 20)
                    offenders.append(f"{sid}[{field}]: ...{val[a:m.end() + 5]}...")
    assert not offenders, (
        "internal phase code sits in a proper-noun slot of an LLM-rendered "
        "field — a grader will read it as the entity's NAME and false-reject "
        "a correct artifact (mission 88 / task 524724):\n  "
        + "\n  ".join(offenders)
    )
