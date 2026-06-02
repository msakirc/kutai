"""Prior-art synthesizer agent (i2p step 1.0c).

Reads the fetched candidates artifact (prior_art_candidates.json) and the
idea brief, then judges which candidates are real attempted solutions,
extracts lessons, and sets a verdict — emitting prior_art_report.json.

Hard constraint: every attempted_solutions[i].url MUST be copied from a
fetched candidate. The synthesizer may NOT add products from its own
knowledge. The prior_art_min_coverage post-hook enforces this.
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.prior_art_synthesizer")


class PriorArtSynthesizerAgent(BaseAgent):
    name = "prior_art_synthesizer"
    description = "Judges fetched prior-art candidates into a graveyard report"
    default_tier = "medium"
    min_tier = "cheap"
    max_iterations = 3
    allowed_tools = ["read_file", "write_file"]

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a prior-art analyst. You judge a list of ALREADY-FETCHED "
            "candidate products into a structured graveyard report.\n"
            "\n"
            "## Rules\n"
            "- You MUST read the candidates artifact first. Every "
            "attempted_solutions entry's `url` and `name` MUST be copied "
            "verbatim from a fetched candidate. You may ONLY use products and "
            "URLs that appear in the candidates file.\n"
            "- You must NEVER invent a product, NEVER write a `url` that is "
            "not in the candidates, and NEVER set `url` to null. If a "
            "candidate has no URL, drop it.\n"
            "- ALWAYS set `verdict` to one of: graveyard_well_populated, "
            "graveyard_thin, blue_ocean_validated, blue_ocean_suspicious. "
            "Use blue_ocean_validated only with >=3 queries and >=20 "
            "results inspected (see search_summary).\n"
            "- ALWAYS extract at least one key_lesson when attempted_solutions "
            "is non-empty.\n"
            "\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": {\n'
            '    "_schema_version": "1",\n'
            '    "search_summary": { "queries_run": [], "sources_used": [], '
            '"total_results_inspected": 0 },\n'
            '    "attempted_solutions": [ {"name": "...", "url": "https://...", '
            '"status": "dead", "thesis_summary": "...", "sources": ["..."]} ],\n'
            '    "key_lessons": [ {"lesson": "...", "evidence_refs": ["..."]} ],\n'
            '    "verdict": "graveyard_thin"\n'
            "  }\n"
            "}\n"
            "```\n"
            "Write this JSON object to the produces path with write_file."
        )
