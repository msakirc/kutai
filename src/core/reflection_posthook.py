"""Post-hook prompt layer for the reflection + constrained-emit LLM children.

SP3b Task 5: ``self_reflect`` and ``constrained_emit`` are post-hook kinds that
spawn a raw_dispatch LLM child (on the husam worker) and REWRITE the source
task's result via the verdict path. This module is the prompt layer for those
two children — it lives next to ``src/core/grading.py`` and ``src/core/
code_review.py`` (the other post-hook prompt builders).

It owns:

* ``REFLECTION_BLOCKS`` / ``STACK_BLOCKS`` / ``LAYER_BLOCKS`` /
  ``build_reflection_prompt`` — the per-agent / stack / layer self-check
  checklists. Moved here from ``coulson/reflection.py`` (which now re-exports
  them for back-compat until Task 7 removes coulson's inline self_reflect
  caller).
* ``build_reflect_messages`` — the reviewer system+user message build for the
  reflection child (lifted from the inline ``self_reflect`` in coulson).
* ``build_emit_messages`` / ``schema_response_format`` / ``should_skip_emit`` —
  the constrained-emit prompt + response_format + skip-when-conforms predicate
  (lifted from ``src/workflows/engine/constrained_emit.py``).

These are pure builders — no DB, no LLM dispatch. The apply layer
(``_enqueue_posthook_llm_child``) builds a raw_dispatch child spec from them;
the continuation handlers (``posthook_continuations.py``) parse the child
response into a rewrite verdict.
"""
from __future__ import annotations

import json

from src.infra.logging_config import get_logger

logger = get_logger("core.reflection_posthook")


# ────────────────────────────────────────────────────────────────────────────
# Stack-aware prompt fragments — Z2 T4C
# ────────────────────────────────────────────────────────────────────────────

STACK_BLOCKS: dict[str, str] = {
    "fastapi": (
        "## Stack-specific reminders (fastapi)\n"
        "- Routes go in routers, not in `main.py`. Use `APIRouter` + `app.include_router`.\n"
        "- Inject DB sessions / services via `Depends(get_db)` — never import globals.\n"
        "- Pydantic v2: use `model_config = ConfigDict(...)`, NOT class `Config`. "
        "Validators are `@field_validator` + `@model_validator`.\n"
        "- Alembic: every migration has both `upgrade()` and `downgrade()`. "
        "Generate with `alembic revision --autogenerate`; always review the diff.\n"
        "- HTTP errors: raise `HTTPException(status_code=..., detail=...)` — don't return raw dicts for errors.\n"
        "- Async routes: `async def` + `await` for I/O; sync routes block the event loop."
    ),
    "nextjs": (
        "## Stack-specific reminders (nextjs)\n"
        "- App Router (default since 13+): pages live under `app/`, not `pages/`.\n"
        "- Server Components are the default — add `'use client'` only when you need\n"
        "  browser APIs, hooks, or event handlers.\n"
        "- Layouts: `app/layout.tsx` wraps every page. Nested layouts inherit.\n"
        "- Data fetching in Server Components: `async` functions, `fetch()` with caching\n"
        "  (`cache: 'force-cache'` / `no-store`). No `useEffect` for server data.\n"
        "- Route handlers: `app/api/.../route.ts` — export named `GET`, `POST`, etc.\n"
        "- Images: always `next/image`. Fonts: `next/font`. Links: `next/link`."
    ),
    "expo": (
        "## Stack-specific reminders (expo)\n"
        "- Use Expo Router (file-based routing under `app/`) for new projects.\n"
        "- Style with `StyleSheet.create` or NativeWind — no raw CSS.\n"
        "- Native modules: prefer Expo SDK equivalents before bare React Native modules.\n"
        "- Permissions: request at runtime via `expo-permissions` / module-specific APIs.\n"
        "- EAS Build: `eas build` for production — don't rely on `expo start` defaults.\n"
        "- Platform guards: `Platform.OS === 'ios'` for divergent behaviour; "
        "avoid `__DEV__`-only logic in production paths."
    ),
    "django": (
        "## Stack-specific reminders (django)\n"
        "- ORM: use `select_related` / `prefetch_related` to avoid N+1 queries.\n"
        "- Views: class-based views (CBVs) for standard CRUD; `@login_required` + `@permission_required`.\n"
        "- Migrations: always run `makemigrations` + `migrate`. Never edit generated migrations by hand.\n"
        "- Settings: keep secrets in env vars (`django-environ` or `os.environ`); never commit `.env`.\n"
        "- DRF: serializers validate input; use `serializer.validated_data`, not `request.data` directly.\n"
        "- Signals: prefer explicit service calls over `post_save` signals — signals hide data flow."
    ),
    "rails": (
        "## Stack-specific reminders (rails)\n"
        "- Migrations: `rails generate migration`, keep `up`/`down` or use `change`. "
        "Never remove columns without a deprecation migration first.\n"
        "- ActiveRecord: `includes` to avoid N+1. Use scopes for reusable query segments.\n"
        "- Strong params: always whitelist via `params.require(...).permit(...)` in controllers.\n"
        "- Services: fat models → extract service objects under `app/services/`.\n"
        "- Credentials: `rails credentials:edit` for secrets; never `ENV['KEY']` in db configs.\n"
        "- Tests: RSpec + FactoryBot; avoid fixtures for complex associations."
    ),
    "vite": (
        "## Stack-specific reminders (vite)\n"
        "- Config: `vite.config.ts` — use `defineConfig`. Plugins via `plugins: [...]`.\n"
        "- Aliases: `resolve.alias` for clean imports — `@/` → `src/`.\n"
        "- Env vars: prefix with `VITE_` to expose to the browser; others are server-only.\n"
        "- HMR: works out of the box; avoid side-effectful module-level code that breaks re-import.\n"
        "- Build: `vite build` outputs to `dist/`; set `base` for sub-path deployments.\n"
        "- CSS: PostCSS + Tailwind wired via `postcss.config.js`; no global CSS injection in components."
    ),
    "nestjs": (
        "## Stack-specific reminders (nestjs)\n"
        "- Architecture: Module → Controller → Service. Business logic lives in Services only.\n"
        "- DI: inject via constructor params + `@Injectable()`. Never instantiate services manually.\n"
        "- DTOs: validate with `class-validator` decorators + `ValidationPipe` globally registered.\n"
        "- Guards / Interceptors / Pipes: register globally in `app.useGlobalGuards` or per-controller.\n"
        "- TypeORM: `@Entity`, `@Column`, `@Repository`. Use `DataSource` for transactions.\n"
        "- Config: `@nestjs/config` + `.env` — never hard-code secrets. "
        "Use `ConfigService.get<string>('KEY')`."
    ),
}


# ────────────────────────────────────────────────────────────────────────────
# Layer-aware reminders — Z3 T4C
# ────────────────────────────────────────────────────────────────────────────

LAYER_BLOCKS: dict[str, str] = {
    "domain": (
        "## Layer reminder (domain)\n"
        "This file is core domain. No framework imports allowed "
        "(no fastapi, no flask, no nextjs, no SDKs). No I/O. Pure Python types only."
    ),
    "adapter": (
        "## Layer reminder (adapter)\n"
        "This file is an adapter. Translate between domain types and infra/SDK types only. "
        "Don't leak domain types past the boundary OR infra types into domain."
    ),
    "infra": (
        "## Layer reminder (infra)\n"
        "This file is infrastructure. May import SDKs, ORMs, frameworks. "
        "Don't import from domain.* — invert the dependency via adapter."
    ),
    "ui": (
        "## Layer reminder (ui)\n"
        "This file is UI. Pull data via hooks/services; no direct DB/HTTP from components."
    ),
    "test": "",
    "unknown": "",
}


REFLECTION_BLOCKS: dict[str, str] = {
    "coder": (
        "Self-check before final_answer:\n"
        "1. Did you RUN the code? Don't assume it works.\n"
        "2. Did TESTs pass (if a test suite exists)?\n"
        "3. Any TODO / pass / placeholder left in the code?\n"
        "4. Are all IMPORTs at the top of files and resolvable?\n"
        "If any 'no' — keep iterating, don't emit final_answer yet."
    ),
    "implementer": (
        "Self-check before final_answer:\n"
        "1. LINT clean (run `lint` tool)?\n"
        "2. `python -m py_compile <file>` — SYNTAX clean?\n"
        "3. Matches the SPEC / ARCHITECTURE.md interface exactly?\n"
        "4. Only your assigned file touched (no wandering)?\n"
        "If any 'no' — fix before final_answer."
    ),
    "fixer": (
        "Self-check before final_answer:\n"
        "1. Every FEEDBACK bullet addressed?\n"
        "2. TESTs run after edit, no new failures?\n"
        "3. Did not DELETE unrelated logic by accident?\n"
        "If any 'no' — fix before final_answer."
    ),
    "test_generator": (
        "Self-check before final_answer:\n"
        "1. Tests run? (Don't claim they pass without running.)\n"
        "2. COVERAGE — every public function + error path + boundary tested?\n"
        "3. No FLAKy waits / sleeps / time-based asserts?\n"
        "4. ASSERT messages helpful (not bare `assert x`)?\n"
        "If any 'no' — keep iterating."
    ),
    "oncall_agent": (
        "Self-check before final_answer:\n"
        "1. Did you READ the entire alert payload (not just severity)?\n"
        "2. Does a known incident PLAYBOOK match this alert? Use it.\n"
        "3. Is the proposed action in the WHITELIST "
        "(restart_service, rollback_to_last_green, scale_up, scale_down, "
        "drain_traffic, rotate_failed_key, archive_flake_test, "
        "escalate_to_founder)?\n"
        "4. Is the action in COOLDOWN? If yes — escalate instead, "
        "never retry blindly.\n"
        "5. Will the action be REVERSIBLE? If no and severity < critical, "
        "escalate to founder.\n"
        "6. Tier-3 (security) incidents always ESCALATE — never act directly.\n"
        "If any 'no' — re-evaluate before emitting final_answer."
    ),
    "support_tier1": (
        "Self-check before final_answer:\n"
        "1. Is every claim in your answer GROUNDED in a retrieved support_doc?\n"
        "2. Did you CITE each source by doc_id?\n"
        "3. Is `confidence` CALIBRATED — 1.0 only if docs answer verbatim, "
        "below 0.7 when inferring or docs are thin?\n"
        "4. Did you avoid PROMISING refunds, credits, or policy changes?\n"
        "5. If the user sounds ANGRY / URGENT, did you set confidence "
        "below 0.7 so the escalation path triggers?\n"
        "If any 'no' — fix before final_answer."
    ),
    "integration_reviewer": (
        "Self-check before final_answer:\n"
        "1. Did I check EVERY emitted file in the task, not just the ones "
        "I read first?\n"
        "2. Did I cross-reference SIGNATURES — caller argument names/order "
        "vs callee parameter names/order?\n"
        "3. Are ALL findings cited with both a file path AND a line number "
        "(or symbol name if line unknown)?\n"
        "4. For cross-file findings, did I cite BOTH file_a (caller/consumer) "
        "and file_b (callee/definition)?\n"
        "5. Did I check migration column names against model/schema field names?\n"
        "If any 'no' — read the missing files and update findings before "
        "emitting final_answer."
    ),
    "mention_monitor": (
        "Self-check before final_answer:\n"
        "1. Did you CHECK the score tier? (silent<4, digest 4-7, immediate>=7)\n"
        "2. Did you DEDUP — UNIQUE(source, source_id) + 24h canonical_url window?\n"
        "3. Is Twitter GATED? MENTION_TWITTER_ENABLED=1 required — never poll without it.\n"
        "4. Negative cluster >=3 in 1h → MUST trigger crisis_comms_draft (B6). Did it?\n"
        "5. Internal signal is a PROXY (tickets table) — never treat as authoritative.\n"
        "6. Never AUTO-RESPOND — score>=7 → founder_action only, never direct reply.\n"
        "If any 'no' — fix before final_answer."
    ),
}

_GENERIC_REFLECTION_BLOCK = (
    "Review your output for errors before final_answer. "
    "Did you actually do what was asked?"
)


def build_reflection_prompt(
    agent_name: str,
    iteration: int,
    stack: str | None = None,
    layer: str | None = None,
) -> str:
    """Return a role-specific self-check checklist for *agent_name*.

    Falls back to a generic prompt for agents without a dedicated checklist
    so that all currently enabled agents (researcher, writer, shopping_advisor,
    deal_analyst, product_researcher) continue to work unchanged.

    Parameters
    ----------
    stack:
        Optional stack identifier (e.g. ``"fastapi"``) or ``+``-joined
        multi-stack string (e.g. ``"fastapi+nextjs"``).  When set and the
        stack key exists in :data:`STACK_BLOCKS`, the relevant fragment is
        appended after the role block.  Multi-stack: each token is looked up
        independently; matched blocks are deduplicated and concatenated.
    layer:
        Optional layer tag from :func:`src.tools.inspect_layer.inspect_layer`
        (one of ``"domain"``, ``"adapter"``, ``"infra"``, ``"ui"``,
        ``"test"``, ``"unknown"``).  When set and :data:`LAYER_BLOCKS` has a
        non-empty entry for it, the layer block is appended after stack
        blocks (Z3 T4C).
    """
    block = REFLECTION_BLOCKS.get(agent_name, _GENERIC_REFLECTION_BLOCK)
    parts = [f"[iteration {iteration}] {block}"]

    if stack:
        seen: set[str] = set()
        for token in stack.split("+"):
            token = token.strip().lower()
            if token and token not in seen and token in STACK_BLOCKS:
                parts.append(STACK_BLOCKS[token])
                seen.add(token)

    if layer:
        layer_block = LAYER_BLOCKS.get(layer.strip().lower(), "")
        if layer_block:
            parts.append(layer_block)

    return "\n\n".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# Reflection child — reviewer message build (lifted from coulson self_reflect).
# ────────────────────────────────────────────────────────────────────────────

REFLECT_SYSTEM_BASE = (
    "You are a careful reviewer. Check this response "
    "for errors, omissions, or hallucinations. "
    "If the response is good, respond: "
    '{"verdict": "ok"}. '
    "If there are issues, respond: "
    '{"verdict": "fix", "issues": "description", '
    '"corrected_result": "the fixed version"}.'
)


def build_reflect_messages(
    task: dict, result: str, checklist: str | None = None,
) -> list[dict]:
    """Build the reviewer system+user messages for the reflection child.

    Mirrors the inline message build from ``coulson.reflection.self_reflect``
    (~314-333). ``checklist`` is the per-agent self-check block from
    :func:`build_reflection_prompt`, appended to the reviewer system message.
    """
    system_content = (
        f"{REFLECT_SYSTEM_BASE}\n\n{checklist}" if checklist else REFLECT_SYSTEM_BASE
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": (
            f"Task: {task.get('title', '')}\n"
            f"Description: {(task.get('description') or '')[:500]}\n\n"
            f"Response to review:\n{(result or '')[:3000]}"
        )},
    ]


# ────────────────────────────────────────────────────────────────────────────
# Constrained-emit child — response_format + message build + skip predicate.
# ────────────────────────────────────────────────────────────────────────────

# Cap draft to keep input token cost in line. 30000 so big multi-artifact
# drafts (form_specs + empty_error_state_specs) don't lose tail content before
# the emit even sees it (lifted from constrained_emit.py ~123).
_EMIT_DRAFT_CAP = 30000


def schema_response_format(artifact_schema: dict, step_id: str = "artifact"):
    """Build a json_schema response_format for *artifact_schema*.

    Returns the response_format dict, or ``None`` when the schema is
    unconstrainable (markdown / string) — caller skips the emit in that case.
    Lifted from constrained_emit.py ~70-82.
    """
    if not isinstance(artifact_schema, dict):
        return None
    from src.workflows.engine.json_schema_translator import build_response_format
    # JSON Schema 'name' must be alphanumeric+underscore; sanitize.
    safe_name = "step_" + "".join(
        c if c.isalnum() else "_" for c in str(step_id)
    )
    return build_response_format(artifact_schema, name=safe_name)


def should_skip_emit(draft: str, artifact_schema: dict) -> bool:
    """Skip the emit when the draft already parses as JSON with all required
    object/array artifact keys present.

    Re-emitting a conforming draft tends to COMPRESS rather than reshape — the
    model sees a long rich draft, gets a tight token budget, and trims content
    from tail fields to fit. The schema validator runs next and catches genuine
    shape gaps; the emit pass is only valuable when the draft is non-JSON or
    missing top-level keys. Lifted from constrained_emit.py ~92-116.
    """
    try:
        parsed = json.loads(draft)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False  # Draft isn't JSON — emit pass will reshape.
    if not isinstance(parsed, dict):
        return False
    need = [
        n for n, r in (artifact_schema or {}).items()
        if isinstance(r, dict) and r.get("type") in ("object", "array")
    ]
    return bool(need) and all(k in parsed for k in need)


def build_emit_messages(draft: str, response_format: dict) -> list[dict]:
    """Build the structured-emitter system+user messages for the emit child.

    Mirrors the inline prompt build from constrained_emit.py ~124-143.
    ``response_format`` is the dict returned by :func:`schema_response_format`.
    """
    schema_text = json.dumps(
        response_format["json_schema"]["schema"],
        ensure_ascii=False,
        indent=2,
    )
    draft_for_prompt = (draft or "")[:_EMIT_DRAFT_CAP]
    system = (
        "You are a structured-output emitter. Re-emit the artifact "
        "below as JSON conforming exactly to the provided schema. "
        "Do not add commentary. Do not wrap in envelopes. Output "
        "ONLY the JSON value.\n\n"
        "Rules:\n"
        "- Every required field must be present with a real value.\n"
        "- Do not invent fields not in the schema.\n"
        "- Preserve the draft's information; restructure into the "
        "schema, do not summarize away content."
    )
    user = (
        f"Schema:\n```json\n{schema_text}\n```\n\n"
        f"Draft to fix:\n```\n{draft_for_prompt}\n```\n\n"
        f"Emit the final JSON now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
