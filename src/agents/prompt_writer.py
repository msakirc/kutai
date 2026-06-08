"""Image-generation prompt writer (Plan 3).

For a prototype with N placeholder <img> intents, emits one JSON envelope
mapping each placeholder_id to an enriched diffusion prompt. Pure config —
sys_prompt + tools, zero methods beyond get_system_prompt. Single-call.

Shape is enforced by:
  (a) sys_prompt (this file) — instructs the LLM,
  (b) artifact_schema on the i2p step (5.35.prompts) — workflow_engine's
      constrained_emit.maybe_apply runs a post-emit structured pass when
      the first call's output is malformed (response_format=json_schema).
"""
from .base import BaseAgent
from ..infra.logging_config import get_logger

logger = get_logger("agents.prompt_writer")


class PromptWriterAgent(BaseAgent):
    name = "prompt_writer"
    description = "Turns prototype placeholder intents into diffusion prompts"
    default_tier = "cheap"
    min_tier = "cheap"
    max_iterations = 1
    enable_self_reflection = False
    allowed_tools = []

    def get_system_prompt(self, task: dict) -> str:
        return (
            "You are a diffusion-prompt engineer for product UI mockups. "
            "Given a list of placeholder <img> intents from a prototype, "
            "you write ONE enriched diffusion prompt per placeholder so a "
            "text-to-image model can produce the real asset.\n\n"
            "## Context\n"
            "- `design_tokens` — color palette + typography. Bias generated "
            "images toward these colors.\n"
            "- `brand_voice` — short paragraph describing voice/mood. Bias "
            "subject choice and composition.\n"
            "- `section_intent` — per-placeholder screen role (hero / "
            "feature / avatar / product / background / icon). Drives "
            "composition + framing.\n"
            "- `placeholders` — list of `{placeholder_id, alt, width, "
            "height, section}`. `alt` is the seed; enrich it with style + "
            "composition + lighting + color cues.\n\n"
            "## You must\n"
            "- Always emit exactly one prompt per placeholder — every "
            "`placeholder_id` in the input must appear in the output.\n"
            "- Always echo the `placeholder_id` verbatim.\n"
            "- Always keep prompts under 220 characters — diffusion "
            "models truncate; the first words dominate.\n"
            "- Always start each prompt with the subject (the `alt` "
            "intent), then add style + composition + color cues.\n"
            "- Always include at least one `design_tokens` color cue.\n\n"
            "## You must never\n"
            "- Don't invent new `placeholder_id` values not in the input.\n"
            "- Don't return text outside the JSON envelope.\n"
            "- Don't include negative prompts or model-specific tokens "
            "(`<lora:...>`, `((emphasis))`).\n"
            "- Never copy `alt` verbatim — enrich it.\n\n"
            "## final_answer format\n"
            "```json\n"
            "{\n"
            '  "action": "final_answer",\n'
            '  "result": {\n'
            '    "_schema_version": "1",\n'
            '    "prompts": [\n'
            "      {\n"
            '        "placeholder_id": "<verbatim id from input>",\n'
            '        "prompt": "<enriched diffusion prompt, <=220 chars>"\n'
            "      }\n"
            "    ]\n"
            "  },\n"
            '  "memories": {}\n'
            "}\n"
            "```\n"
        )


import os as _os

_DEFAULT_TEMPLATE_PATH = "docs/templates/prompt_writer/diffusion_prompt_template.md"


def load_diffusion_prompt_template(path: str | None = None) -> str | None:
    """Load the diffusion-prompt few-shot template. Slots are
    {design_tokens}, {brand_voice}, {section_intent}, {placeholders}.
    Returns None if file missing — agent still works on sys_prompt alone."""
    p = path or _DEFAULT_TEMPLATE_PATH
    if not _os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None
