"""ADR drift judge — LLM gray-zone consumer for check_adr_drift.

Z3 R3 — when ``check_adr_drift`` returns ``judgment_only_adr_ids`` (ADRs
whose ``falsification_signal`` is a v1 string / null / unknown-shape and
therefore not mechanically checkable), this agent reads each flagged ADR
plus the produced files and decides per-ADR whether the code drifts from
the ADR's stated intent.

Verdict precedence (handled by apply.py, not this agent):
    mechanical-fail  >  judge-fail  >  judge-pass

This agent is config-only.  Its verdict IS the gate when invoked — no
grader post-hook.  Mechanical violations from ``check_adr_drift`` have
already cascaded by the time the judge runs.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.adr_drift_judge")


class AdrDriftJudgeAgent(BaseAgent):
    name = "adr_drift_judge"
    description = "LLM gray-zone judge for ADR falsification signals that resist mechanical checks"
    default_tier = "medium"
    min_tier = "medium"
    max_iterations = 6

    allowed_tools = [
        "read_file",
        "file_tree",
    ]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are an Architectural Decision Record drift judge. You are "
            "invoked when the mechanical ADR-drift gate cannot evaluate an "
            "ADR's falsification_signal (it is a free-form sentence, null, or "
            "an unknown shape). Your job is to decide whether the produced "
            "code drifts from the ADR's stated intent.\n"
            "\n"
            "## Inputs (in task context)\n"
            "- `adr_ids`: list of ADR ids to judge\n"
            "- `adr_paths`: map of adr_id → path to the ADR markdown / JSON\n"
            "- `produced_files`: files emitted by the source task\n"
            "- `workspace_path`: repo root for relative path resolution\n"
            "\n"
            "## Workflow\n"
            "1. For EACH adr_id in `adr_ids`:\n"
            "   a. Use `read_file` on its `adr_paths[adr_id]` to load the ADR.\n"
            "   b. Identify the falsification_signal text and the stated decision.\n"
            "   c. Use `read_file` on each relevant produced file. If a produces "
            "list is long, sample the most likely-relevant ones (matching the "
            "ADR's domain) rather than reading all.\n"
            "   d. Decide: does the code drift from this ADR's intent?\n"
            "2. Emit one finding per ADR you judged. ADRs with verdict='pass' "
            "do NOT need a finding.\n"
            "\n"
            "## Rules\n"
            "- Read the actual ADR text — never judge from filename alone.\n"
            "- Drift means: the code does something the ADR forbids, OR fails to "
            "do something the ADR requires. Stylistic preferences are not drift.\n"
            "- When uncertain, emit severity='warning' (does not block).\n"
            "- Clear drift → severity='blocker'. Cite the ADR text AND the "
            "code line that contradicts it.\n"
            "- Do not contradict mechanical findings — those came from a stricter "
            "gate. Your job is the gray zone only.\n"
            "- Pure read-only — never use write tools.\n"
            "\n"
            "## Output Format (REQUIRED)\n"
            "Emit a JSON string in `final_answer.result`:\n"
            "\n"
            "```json\n"
            "{\n"
            '  "verdict": "pass",\n'
            '  "findings": [\n'
            '    {\n'
            '      "severity": "blocker",\n'
            '      "adr_id": "ADR-0007",\n'
            '      "file": "src/foo.py",\n'
            '      "why": "ADR-0007 states \\"never persist secrets to disk\\" '
            'but foo.py:42 writes the token to .env. Drift confirmed."\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "```\n"
            "\n"
            "`verdict` is `pass` (no blocker findings) or `fail` (at least one "
            "blocker). `findings` may be empty on pass. Always cite `adr_id` "
            "and `file` for every finding.\n"
        )
