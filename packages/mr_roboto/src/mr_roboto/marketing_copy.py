"""Z7 T6 A12 — marketing_copy: LLM-bound marketing copy generator.

Produces a structured ``marketing_copy.json`` artifact:

    {
        "hero": [
            {"headline": str, "subheadline": str, "cta": str},
            ...  (exactly 3 variants)
        ],
        "features": [
            {"name": str, "copy": str},
            ...  (one per spec feature)
        ],
        "pricing": [
            {"tier": str, "price_usd": float, "copy": str},
            ...  (one per pricing tier)
        ],
        "faq": [
            {"question": str, "answer": str},
            ...  (from spec or A8 faq artifact; may be empty)
        ]
    }

Artifact written to ``artifacts/marketing_copy/{mission_id}.json``.

Pipeline:
  1. Seed FAQ from A8 faq artifact (graceful degrade if absent).
  2. Dispatch LLM via beckman.enqueue (MAIN_WORK lane) with product spec + FAQ seed.
  3. Run A5 brand_voice_lint (graceful degrade if voice doc absent).
  4. Run A6 copy_compliance_review (graceful degrade on error).
  5. Emit founder_action "review marketing copy" with approve / regenerate-hero /
     regenerate-FAQ options.

Public API
----------
  run_marketing_copy(
      product_id, mission_id, product_spec,
      brand_voice_audience=None,
      faq_artifact_path=None,
      task_id=None,
  ) -> dict

Internal hooks (patched in tests)
----------------------------------
  enqueue(spec, **kwargs) -> dict        — beckman.enqueue wrapper
  _run_brand_voice_lint(...)  -> dict    — A5 lint wrapper
  _run_copy_compliance(...)   -> dict    — A6 compliance wrapper
  _emit_founder_action(...)   -> int     — founder_action create wrapper
  _load_brand_voice_doc(audience) -> str|None
  _load_faq_seed(faq_artifact_path, mission_id) -> list|None
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from yazbunu import get_logger
from src.ops.brand_voice import load_founder_voice

logger = get_logger("mr_roboto.marketing_copy")

# ─── Artifact output directory ────────────────────────────────────────────────
# Overridable via env var for tests.
_DEFAULT_ARTIFACTS_DIR = "data/artifacts/marketing_copy"


def _artifacts_dir() -> str:
    return os.environ.get("MARKETING_COPY_ARTIFACTS_DIR", _DEFAULT_ARTIFACTS_DIR)


# ─── LLM system prompt ───────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert product marketer. Given a product spec, generate structured "
    "marketing copy. You MUST respond with valid JSON only — no prose, no fences.\n\n"
    "Output schema:\n"
    "{\n"
    '  "hero": [\n'
    '    {"headline": str, "subheadline": str, "cta": str},\n'
    '    {"headline": str, "subheadline": str, "cta": str},\n'
    '    {"headline": str, "subheadline": str, "cta": str}\n'
    "  ],\n"
    '  "features": [{"name": str, "copy": str}, ...],\n'
    '  "pricing": [{"tier": str, "price_usd": number, "copy": str}, ...],\n'
    '  "faq": [{"question": str, "answer": str}, ...]\n'
    "}\n\n"
    "Rules:\n"
    "- Exactly 3 hero variants, each with distinct angle (benefit / pain-relief / social-proof).\n"
    "- One feature entry per item in features list.\n"
    "- One pricing entry per tier, with a compelling one-line value proposition.\n"
    "- FAQ: if seed entries are provided, reuse and expand them. Otherwise draft from the spec.\n"
    "- Write for the target audience. Keep copy clear, direct, benefit-focused.\n"
    "- NO markdown fences in output. Pure JSON only."
)


def _build_prompt(product_spec: dict, faq_seed: list | None) -> str:
    spec_json = json.dumps(product_spec, indent=2, ensure_ascii=False)
    faq_block = ""
    if faq_seed:
        faq_json = json.dumps(faq_seed, indent=2, ensure_ascii=False)
        faq_block = f"\n\nEXISTING FAQ ENTRIES (seed — include and expand):\n{faq_json}"

    return (
        f"PRODUCT SPEC:\n{spec_json}"
        f"{faq_block}\n\n"
        "Generate the marketing copy JSON following the schema above. "
        "Return only the JSON object, nothing else."
    )


# ─── Beckman enqueue wrapper ─────────────────────────────────────────────────


async def enqueue(spec: dict, **kwargs) -> dict:
    """Thin wrapper for test patching."""
    from general_beckman import enqueue as _enqueue
    return await _enqueue(spec, **kwargs)


# ─── FAQ seed loader ─────────────────────────────────────────────────────────


def _load_faq_seed(
    faq_artifact_path: str | None,
    mission_id: int | None,
    workspace_path: str | None = None,
) -> list | None:
    """Load FAQ entries from A8 artifact. Returns list or None (absent)."""
    candidates: list[str] = []

    if faq_artifact_path:
        candidates.append(faq_artifact_path)

    # Per-mission workspace (standard convention) takes precedence.
    if workspace_path:
        candidates.extend([
            f"{workspace_path}/faq_en.md",
            f"{workspace_path}/faq.md",
        ])

    # Legacy / orphan-call fallbacks.
    if mission_id is not None:
        candidates.extend([
            f"data/artifacts/faq_en.md",
            f"data/mission_{mission_id}/faq_en.md",
            f"data/mission_{mission_id}/faq.md",
        ])

    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if not raw:
                continue
            # Try parsing as JSON first (structured A8 output)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict) and "faq" in parsed:
                    return list(parsed["faq"])
            except json.JSONDecodeError:
                pass
            # Markdown Q&A format: lines starting with "**Q" or "## Q"
            entries = []
            lines = raw.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("**Q") or line.startswith("## "):
                    question = line.lstrip("#* ").strip()
                    answer_parts = []
                    i += 1
                    while i < len(lines) and not (
                        lines[i].strip().startswith("**Q") or
                        lines[i].strip().startswith("## ")
                    ):
                        a = lines[i].strip().lstrip("*A:").strip()
                        if a:
                            answer_parts.append(a)
                        i += 1
                    answer = " ".join(answer_parts).strip()
                    if question:
                        entries.append({"question": question, "answer": answer})
                else:
                    i += 1
            if entries:
                return entries
        except OSError:
            continue

    return None


# ─── Brand voice doc loader ───────────────────────────────────────────────────


def _load_brand_voice_doc(audience: str | None) -> str | None:
    """Load brand_voices/{audience}.md. Returns raw content or None."""
    if not audience:
        return None
    candidates = [
        f"docs/templates/brand_voices/{audience}.md",
        f"brand_voices/{audience}.md",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    return fh.read()
            except OSError:
                continue
    return None


# ─── A5 brand_voice_lint wrapper ─────────────────────────────────────────────


async def _run_brand_voice_lint(
    text: str,
    audience: str | None,
    task_id: int | None,
    mission_id: int | None,
) -> dict:
    """Invoke A5 brand_voice_lint on generated copy text.

    Degrades gracefully:
    - No audience → skip
    - No voice doc → skip
    - Internal error → skip with error note
    """
    if not audience:
        return {"status": "skip", "reason": "no brand_voice audience specified"}

    voice_doc = _load_brand_voice_doc(audience)
    if voice_doc is None:
        return {
            "status": "skip",
            "reason": f"brand_voice doc not found for audience={audience!r}",
        }

    try:
        from general_beckman.posthook_handlers.brand_voice_lint import handle as bvl_handle
        fake_task = {
            "id": task_id,
            "mission_id": mission_id,
            "context": {
                "brand_voice_audience": audience,
            },
        }
        fake_result = {"result": text, "text": text}
        return await bvl_handle(fake_task, fake_result)
    except Exception as exc:
        logger.warning(
            "marketing_copy: brand_voice_lint failed — skipping",
            audience=audience,
            error=str(exc),
        )
        return {
            "status": "skip",
            "reason": f"brand_voice_lint error: {exc}",
        }


# ─── A6 copy_compliance wrapper ──────────────────────────────────────────────


async def _run_copy_compliance(
    text: str,
    task_id: int | None,
    mission_id: int | None,
) -> dict:
    """Invoke A6 copy_compliance_review on generated copy text.

    Degrades gracefully on any error.
    """
    try:
        from general_beckman.posthook_handlers.copy_compliance_review import handle as ccr_handle
        fake_task = {
            "id": task_id,
            "mission_id": mission_id,
            "context": {},
        }
        fake_result = {"result": text, "copy_text": text}
        return await ccr_handle(fake_task, fake_result)
    except Exception as exc:
        logger.warning(
            "marketing_copy: copy_compliance_review failed — skipping",
            error=str(exc),
        )
        return {
            "status": "skip",
            "reason": f"copy_compliance_review error: {exc}",
        }


# ─── Founder action emitter ──────────────────────────────────────────────────


async def _emit_founder_action(
    mission_id: int,
    artifact_path: str,
    options: list[str],
    task_id: int | None,
) -> int | None:
    """Create founder_action card for marketing copy review.

    Returns the action id or None on failure.
    """
    try:
        from src.founder_actions import create as fa_create
        action = await fa_create(
            mission_id=mission_id,
            kind="generic",
            title="Review marketing copy",
            why=(
                "Marketing copy has been generated. Review the hero variants, "
                "feature copy, pricing copy, and FAQ entries before pasting into "
                "your site builder (Webflow / Framer / etc.)."
            ),
            instructions=[
                f"Artifact: {artifact_path}",
                "Options:",
            ] + [f"  - {opt}" for opt in options],
            blocking_task_id=task_id,
            notify_telegram=False,
            urgent=False,
        )
        return int(getattr(action, "id", 0) or 0) or None
    except Exception as exc:
        logger.warning("marketing_copy: founder_action emit failed: %s", exc)
        return None


# ─── Artifact serializer ─────────────────────────────────────────────────────


def _write_artifact(mission_id: int, artifact: dict, workspace_path: str | None = None) -> str:
    """Write artifact JSON to disk. Returns the file path.

    Precedence:
      1. workspace_path (per-mission, the standard convention) →
         ``{workspace_path}/marketing_copy.json``
      2. MARKETING_COPY_ARTIFACTS_DIR env override → ``{env}/{mission_id}.json``
      3. global default ``data/artifacts/marketing_copy/{mission_id}.json``
         (orphan calls without a mission workspace).
    """
    if workspace_path:
        out_dir = Path(workspace_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "marketing_copy.json"
    else:
        out_dir = Path(_artifacts_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{mission_id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, ensure_ascii=False)
    return str(path)


# ─── LLM result parser ───────────────────────────────────────────────────────


def _parse_llm_result(raw: Any) -> dict | None:
    """Extract marketing copy dict from LLM result payload."""
    if isinstance(raw, dict):
        # Direct structured result
        if "hero" in raw:
            return raw
        # Wrapped result
        for key in ("result", "answer", "content", "copy"):
            inner = raw.get(key)
            if isinstance(inner, dict) and "hero" in inner:
                return inner
            if isinstance(inner, str):
                inner = inner.strip()
                # Strip fences
                if inner.startswith("```"):
                    inner = inner.split("```", 2)[-1]
                    inner = inner.split("```", 1)[0].strip()
                    if inner.startswith("json"):
                        inner = inner[4:].strip()
                try:
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict) and "hero" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[-1]
            raw = raw.split("```", 1)[0].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "hero" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    return None


# ─── Main entry point ────────────────────────────────────────────────────────


async def run_marketing_copy(
    product_id: str,
    mission_id: int,
    product_spec: dict[str, Any],
    brand_voice_audience: str | None = None,
    faq_artifact_path: str | None = None,
    task_id: int | None = None,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    """Generate structured marketing copy for a product.

    Returns:
        {
            "status": "completed" | "error",
            "artifact": { hero, features, pricing, faq },
            "artifact_path": str,
            "lint_result": dict,
            "compliance_result": dict,
            "founder_action_id": int | None,
            "error": str  (only on status=="error")
        }
    """
    logger.info(
        "marketing_copy: starting",
        product_id=product_id,
        mission_id=mission_id,
        audience=brand_voice_audience or "(none)",
    )

    # ── Step 1: seed FAQ from A8 artifact ─────────────────────────────────
    faq_seed = _load_faq_seed(faq_artifact_path, mission_id, workspace_path)
    if faq_seed:
        logger.info(
            "marketing_copy: seeded %d FAQ entries from artifact",
            len(faq_seed),
        )
    else:
        logger.debug("marketing_copy: no FAQ seed found — LLM will draft from scratch")

    # ── Step 2: Dispatch LLM copy generation (MAIN_WORK lane) ─────────────
    prompt = _build_prompt(product_spec, faq_seed)
    # Prepend the founder's voice so generated hero/feature/pricing copy
    # reads in their voice, not generic corporate-speak. No-op when unfilled.
    _voice = load_founder_voice()
    if _voice:
        prompt = f"Brand voice — write all copy in this voice:\n{_voice[:800]}\n\n{prompt}"
    spec = {
        "title": f"marketing_copy:{product_id}:mission#{mission_id}",
        "description": prompt,
        "agent_type": "writer",
        "mission_id": mission_id,
        "context": {
            "product_id": product_id,
            "action_hint": "marketing_copy",
            "brand_voice_audience": brand_voice_audience or "",
            "llm_call": {
                "raw_dispatch": True,
                "call_category": "main_work",
                "task": "writer",
                "agent_type": "writer",
                "difficulty": 4,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "failures": [],
                "estimated_input_tokens": 1200,
                "estimated_output_tokens": 800,
            },
        },
    }

    try:
        enqueue_result = await enqueue(spec)
    except Exception as exc:
        logger.error("marketing_copy: LLM enqueue failed: %s", exc)
        return {"status": "error", "error": f"LLM enqueue failed: {exc}"}

    # ── Step 3: Parse LLM result ──────────────────────────────────────────
    raw_result = None
    if isinstance(enqueue_result, dict):
        raw_result = enqueue_result.get("result") or enqueue_result
    else:
        raw_result = getattr(enqueue_result, "raw", {}) or {}

    artifact = _parse_llm_result(raw_result)

    if artifact is None:
        logger.warning(
            "marketing_copy: LLM returned unparseable result — using empty scaffold",
        )
        artifact = {
            "hero": [
                {"headline": "", "subheadline": "", "cta": ""},
                {"headline": "", "subheadline": "", "cta": ""},
                {"headline": "", "subheadline": "", "cta": ""},
            ],
            "features": [],
            "pricing": [],
            "faq": faq_seed or [],
        }

    # Ensure hero has exactly 3 variants
    hero = list(artifact.get("hero") or [])
    while len(hero) < 3:
        hero.append({"headline": "", "subheadline": "", "cta": ""})
    artifact["hero"] = hero[:3]

    # Ensure faq is a list (may be None if LLM omitted it)
    if not isinstance(artifact.get("faq"), list):
        artifact["faq"] = faq_seed or []

    # ── Step 4: Write artifact to disk ────────────────────────────────────
    artifact_path = _write_artifact(mission_id, artifact, workspace_path)
    logger.info("marketing_copy: artifact written to %s", artifact_path)

    # ── Step 5: Flatten copy text for lint/compliance ─────────────────────
    all_copy_parts: list[str] = []
    for h in artifact["hero"]:
        if isinstance(h, dict):
            all_copy_parts.extend(v for v in h.values() if isinstance(v, str))
    for feat in artifact.get("features", []):
        if isinstance(feat, dict):
            all_copy_parts.append(feat.get("copy", ""))
    for p in artifact.get("pricing", []):
        if isinstance(p, dict):
            all_copy_parts.append(p.get("copy", ""))
    for faq_entry in artifact.get("faq", []):
        if isinstance(faq_entry, dict):
            all_copy_parts.append(faq_entry.get("answer", ""))
    copy_text = "\n".join(s for s in all_copy_parts if s)

    # ── Step 6: A5 brand_voice_lint ──────────────────────────────────────
    lint_result = await _run_brand_voice_lint(
        text=copy_text,
        audience=brand_voice_audience,
        task_id=task_id,
        mission_id=mission_id,
    )
    logger.info(
        "marketing_copy: brand_voice_lint status=%s",
        lint_result.get("status"),
    )

    # ── Step 7: A6 copy_compliance_review ────────────────────────────────
    try:
        compliance_result = await _run_copy_compliance(
            text=copy_text,
            task_id=task_id,
            mission_id=mission_id,
        )
    except Exception as exc:
        logger.warning("marketing_copy: copy_compliance raised: %s", exc)
        compliance_result = {"status": "error", "reason": str(exc)}

    logger.info(
        "marketing_copy: copy_compliance status=%s",
        compliance_result.get("status"),
    )

    # ── Step 8: Emit founder_action ───────────────────────────────────────
    founder_options = [
        "approve — copy is ready for Webflow / Framer",
        "regenerate-hero — regenerate all 3 hero variants",
        "regenerate-FAQ — regenerate FAQ section only",
    ]
    fa_id = await _emit_founder_action(
        mission_id=mission_id,
        artifact_path=artifact_path,
        options=founder_options,
        task_id=task_id,
    )

    return {
        "status": "completed",
        "artifact": artifact,
        "artifact_path": artifact_path,
        "lint_result": lint_result,
        "compliance_result": compliance_result,
        "founder_action_id": fa_id,
    }
