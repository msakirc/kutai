"""Manifest synthesis — turn a fetched native artifact into a Manifest.

Phase 1 implements the PARSER path only (mechanical, no LLM): SKILL.md YAML
frontmatter -> Manifest. The recon confirms 100% of github_path artifacts
carry clean frontmatter, so the parser path covers every Phase 1 source.

LLM-fallback synthesis (for awesome-list bullets / freeform README) is
Phase 3 — it is NOT stubbed here; synthesize() simply does not handle those
sources because no Phase 1 adapter produces them. When Phase 3 adds the
awesome_list adapter it adds an `llm_synthesize()` branch keyed on a flag in
ArtifactRef.raw_meta. Phase 1's parser path is complete and self-tested.
"""
from __future__ import annotations

import re

from yalayut.contracts import ArtifactRef, Manifest
from yalayut.discovery.sources.github_path import parse_skill_md
from yalayut.manifest import canonical_name

# words to drop when mining keywords from a description
_STOP = {
    "use", "this", "skill", "whenever", "the", "user", "wants", "to", "do",
    "anything", "with", "a", "an", "and", "or", "of", "for", "when", "any",
    "files", "file", "that", "is", "are", "be", "guide", "creating", "create",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")


def _mine_keywords(text: str, limit: int = 8) -> list[str]:
    """Mechanically extract intent keywords from a description string."""
    seen: list[str] = []
    for w in _WORD.findall(text.lower()):
        if w in _STOP or w in seen:
            continue
        seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def _source_slug(source_id: str) -> str:
    """'github:anthropics/skills@/skills' -> 'anthropics'."""
    body = source_id.split("github:", 1)[-1]
    repo_part = body.split("@", 1)[0]
    return repo_part.split("/", 1)[0]


def synthesize(ref: ArtifactRef, raw_body: bytes) -> tuple[Manifest, str]:
    """Parser-path synthesis: frontmatter -> (Manifest, body string).

    The yalayut typed-recipe section (inputs_schema / invocation) is NEVER
    lifted from upstream — recon confirms no SKILL.md carries it. Synthesized
    prompt_skill artifacts get mechanizable=False; only hand-authored seed
    manifests declare invocation steps + mechanizable=True.
    """
    meta, body = parse_skill_md(raw_body)
    original = meta.get("name", ref.name)
    slug = ref.owner or _source_slug(ref.source_id)
    desc = meta.get("description", "")
    keywords = _mine_keywords(f"{original} {desc}")
    manifest = Manifest(
        name=canonical_name(slug, original),
        name_original=original,
        version="1.0.0",
        artifact_type="skill",
        kind="prompt_skill",
        source=ref.source_id,
        owner=ref.owner,
        license=meta.get("license"),
        mechanizable=False,
        model_hint=meta.get("model"),
        applies_to="execution",
        intent_keywords=keywords,
    )
    return manifest, body


async def llm_synthesize(raw_text: str, source_meta: dict) -> dict:
    """LLM-fallback manifest synthesis for unstructured sources
    (awesome-list bullets, freeform README). Routes a Sonnet call through
    ``husam.run`` — yalayut never imports LLMDispatcher directly.

    Returns a partial manifest dict: ``{intent_keywords, mechanizable,
    kind, install_cmd, auth_env}``. On any failure returns a conservative
    empty-ish manifest so the caller can still tier the artifact at T1/T2.
    """
    import json as _json

    from yazbunu import get_logger as _gl
    _log = _gl("yalayut.synthesize")

    from finch import build_messages
    _msgs = build_messages("yalayut_synth", {
        "name_original": source_meta.get("name_original", ""),
        "raw_text": raw_text[:2000],
    })
    # Original sends a single user message (no system) — preserve that structure.
    user_msg = _msgs[1]

    try:
        import husam
        resp = await husam.run(
            {
                "agent_type": "yalayut_synthesizer",
                "kind": "overhead",
                "context": {
                    "llm_call": {
                        "raw_dispatch": True,
                        "call_category": "overhead",
                        "messages": [user_msg],
                        "response_format": {"type": "json_object"},
                        "model_hint": "sonnet",
                    },
                },
            }
        )
        raw = resp.get("content")
        parsed = _json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception as e:  # noqa: BLE001 — synthesis must never crash cron
        _log.warning("llm_synthesize failed, using empty manifest: %s", e)
        parsed = {}

    return {
        "intent_keywords": list(parsed.get("intent_keywords") or []),
        "mechanizable": bool(parsed.get("mechanizable", False)),
        "kind": parsed.get("kind") or "prompt_skill",
        "install_cmd": parsed.get("install_cmd"),
        "auth_env": parsed.get("auth_env"),
    }
