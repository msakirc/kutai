"""Structural-truncation guarantees for format_blackboard_for_prompt.

The blackboard block is injected into agent prompts. Raw byte-slicing
(``text[:max_chars]``) can sever the architecture JSON or a section
mid-line, leaving the model to parse/trust malformed structure. These
tests lock the contract: truncation drops WHOLE sections/items and
records the omission honestly; it never cuts mid-content.
"""
import json
import re

from src.collaboration.blackboard import format_blackboard_for_prompt


def _board(**over):
    base = {
        "architecture": {},
        "files": {},
        "decisions": [],
        "open_issues": [],
        "constraints": [],
        "dependency_map": {},
    }
    base.update(over)
    return base


def test_large_architecture_never_renders_sliced_invalid_json():
    """A big architecture dict must not produce a truncated (unparseable)
    ```json fence — the model would try to parse broken JSON."""
    arch = {f"module_{i}": {"role": "x" * 40} for i in range(20)}  # json > 500 chars
    out = format_blackboard_for_prompt(_board(architecture=arch))

    # Every json fence that IS emitted must contain valid JSON.
    for body in re.findall(r"```json\n(.*?)\n```", out, re.S):
        json.loads(body)  # raises -> test fails if sliced

    # Architecture still represented somehow (keys not silently dropped).
    assert "Architecture" in out
    assert "module_0" in out


def test_oversized_board_drops_whole_sections_with_honest_note():
    """When content exceeds the budget, sections are dropped cleanly and
    an omission note is added — no raw '[blackboard truncated]' byte-cut."""
    board = _board(
        decisions=[
            {"what": f"decision {i}", "why": "because reasons " * 4, "by": "architect"}
            for i in range(5)
        ],
        open_issues=[f"open issue number {i} with descriptive text" for i in range(5)],
        constraints=[f"constraint {i} also reasonably long descriptive text" for i in range(5)],
    )
    out = format_blackboard_for_prompt(board, max_chars=200)

    # No raw byte-slice backstop marker.
    assert "[blackboard truncated]" not in out
    # Honest structural omission note instead.
    assert "omitted" in out.lower()


def test_kept_lines_are_whole_never_byte_sliced():
    """Every bullet line that survives must be byte-identical to its
    unbudgeted form — proving no line was cut mid-content."""
    board = _board(
        decisions=[
            {"what": f"decision {i}", "why": "because reasons " * 4, "by": "architect"}
            for i in range(5)
        ],
        constraints=[f"constraint {i} also reasonably long descriptive text" for i in range(5)],
    )
    full = format_blackboard_for_prompt(board, max_chars=100_000)
    full_lines = set(full.split("\n"))

    out = format_blackboard_for_prompt(board, max_chars=200)
    for ln in out.split("\n"):
        if ln.startswith("  - "):
            assert ln in full_lines, f"sliced mid-content line: {ln!r}"


def test_small_board_unchanged_no_note():
    """A board that fits the budget renders fully with no omission note."""
    board = _board(decisions=[{"what": "use sqlite", "why": "simple", "by": "architect"}])
    out = format_blackboard_for_prompt(board)
    assert "use sqlite" in out
    assert "omitted" not in out.lower()
    assert "truncated" not in out.lower()
