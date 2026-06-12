"""Image-generation prompt writer (Plan 3) — artifact schema + template loader.

The agent itself is now a Foundry data profile
(``packages/finch/src/finch/profiles/prompt_writer.yaml``); its system prompt
and config live there and are resolved via ``get_agent("prompt_writer")``.

This module retains the two pieces of LOGIC that other code imports:

  • ``PROMPT_WRITER_ARTIFACT_SCHEMA`` — the single source of truth for the
    emitted artifact shape ``{"_schema_version": "1", "prompts": [...]}``.
    Imported + reused by i2p step 5.35 and the mr_roboto P3-B enqueue spec
    (``packages/mr_roboto/src/mr_roboto/swap_placeholder_images.py``) rather
    than duplicating the dict inline.
  • ``load_diffusion_prompt_template`` — loads the optional few-shot template.

Shape is enforced by the profile's system_prompt (a) plus the constrained_emit
POSTHOOK child (b) — armed by beckman's ``determine_posthooks`` when it finds a
constrainable ``artifact_schema`` in the task context. Without that schema the
agent degrades gracefully: malformed JSON triggers normal retry.
"""
import pathlib as _pathlib

from ..infra.logging_config import get_logger

logger = get_logger("agents.prompt_writer")

# ---------------------------------------------------------------------------
# Canonical artifact schema — single source of truth.
#
# The agent emits:  {"_schema_version": "1", "prompts": [{placeholder_id, prompt}, ...]}
# Artifact name "diffusion_prompts" matches the i2p step 5.35 ``produces`` entry.
#
# Reuse this constant in:
#   • i2p step 5.35  ``artifact_schema`` field
#   • mr_roboto P3-B enqueue  ``artifact_schema`` kwarg
# ---------------------------------------------------------------------------
PROMPT_WRITER_ARTIFACT_SCHEMA: dict = {
    "diffusion_prompts": {
        "type": "object",
        "required_fields": ["prompts"],
        "_schema_version": "1",
    }
}


_DEFAULT_TEMPLATE_PATH = (
    _pathlib.Path(__file__).parent.parent.parent
    / "docs" / "templates" / "prompt_writer" / "diffusion_prompt_template.md"
)


def load_diffusion_prompt_template(path: str | None = None) -> str | None:
    """Load the diffusion-prompt few-shot template. Slots are
    {design_tokens}, {brand_voice}, {section_intent}, {placeholders}.
    Returns None if file missing — agent still works on sys_prompt alone.

    The default path is resolved relative to this module's location
    (``__file__``), not the process cwd, so the template loads correctly
    regardless of the working directory the caller uses at runtime.
    """
    p = _pathlib.Path(path) if path else _DEFAULT_TEMPLATE_PATH
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None
