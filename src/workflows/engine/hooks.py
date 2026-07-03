"""Pre/post execution hooks for workflow steps in the orchestrator.

Handles artifact injection, output storage, conditional evaluation,
template expansion triggers, and CodingPipeline delegation detection.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from src.infra.logging_config import get_logger
from src.tools.workspace import get_mission_workspace
from .artifacts import ArtifactStore, format_artifacts_for_prompt, get_phase_summaries, CONTEXT_BUDGETS, _TIER_ORDER
from .conditions import evaluate_condition, resolve_group
from .policies import ReviewTracker
from .quality_gates import evaluate_gate, format_gate_result

logger = get_logger("workflows.engine.hooks")


def _output_hash(value) -> Optional[str]:
    """Stable short hash of a produced output, for the ledger ``out_hash``.

    Used by the repeat-detector (Phase 3) to spot identical re-attempts.
    Returns None for empty/non-string output so degenerate/empty paths
    carry a null hash (they store no comparable draft)."""
    import hashlib

    if not isinstance(value, str) or not value.strip():
        return None
    return hashlib.sha1(value.encode("utf-8", "replace")).hexdigest()[:16]


def append_rejection(
    ctx: dict,
    attempt,
    reason,
    out_hash=None,
) -> None:
    """Append a quality-rejection entry to ``ctx["_rejection_ledger"]``.

    The ledger is the compact, accumulated history of (approach,
    why-rejected) for a task — appended, never overwritten — so the retry
    prompt (coulson/context.py, T2) can show the worker every prior
    rejection reason and tell it to take a different path (spec C5).

    Shape per entry: ``{attempt:int, category:"quality", reason:str≤500,
    out_hash}``. ``reason`` is capped at 500 chars (M3/F5) so a long
    grader/schema error cannot bloat the ledger. Only QUALITY rejections
    call this; availability failures produce no judged output and append
    nothing (C2). Pure ctx mutation — the caller persists ``ctx``.
    """
    ctx.setdefault("_rejection_ledger", []).append(
        {
            "attempt": int(attempt),
            "category": "quality",
            "reason": str(reason)[:500],
            "out_hash": out_hash,
        }
    )


def build_summary_spec(text: str, artifact_name: str) -> dict:
    """Pure builder for the summarizer child (SP3). No mission_id/parent on the
    spec — those travel in continuation state. No input degenerate check (the
    output degeneracy check lives in the resume handler)."""
    import time as _time
    import uuid as _uuid

    truncated = text[:16000]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise summarizer. Produce a summary that "
                "preserves ALL key facts, decisions, and data points. "
                "Target: under 400 words. No filler."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Summarize this '{artifact_name}' artifact. Keep every "
                f"important fact, number, name, and decision:\n\n"
                f"{truncated}"
            ),
        },
    ]
    _suffix = f"{_time.monotonic_ns() % 1_000_000:06d}-{_uuid.uuid4().hex[:6]}"
    return {
        "title": f"summarizer:{artifact_name}:{_suffix}",
        "description": f"LLM summarization of artifact '{artifact_name}'",
        "agent_type": "summarizer",
        "kind": "overhead",
        "context": {"llm_call": {
            "raw_dispatch": True,
            "call_category": "overhead",
            "task": "summarizer",
            "agent_type": "summarizer",
            "difficulty": 2,
            "messages": messages,
            "failures": [],
            "prefer_speed": True,
            "prefer_local": True,
            "estimated_input_tokens": min(len(text) // 4, 4000),
            "estimated_output_tokens": 500,
        }},
    }


def _extract_json_string_field(text: str, key: str) -> str | None:
    """Recover a string field's value from possibly-truncated JSON.

    Walks char-by-char from the first ``"key": "`` match, honoring JSON
    escape rules. Returns the raw escaped-string body up to the first
    unescaped closing ``"`` — or, if the input is truncated mid-string,
    everything captured so far. Returns ``None`` only when the key isn't
    found at all.

    Why this exists: the older regex fallback required a downstream
    sentinel (``"memories"``, ``"subtasks"``, or a closing ``}``) and
    failed silently on LLM outputs that ran out of budget mid-string,
    leaving the raw broken envelope as the artifact (observed task 2890
    user_stories, 4132-char truncated triple-escaped output).
    """
    import re as _re
    m = _re.search(rf'"{key}"\s*:\s*"', text)
    if not m:
        return None
    i = m.end()
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            # Capture the escape pair verbatim — unescape happens after.
            if i + 1 < len(text):
                out.append(ch + text[i + 1])
                i += 2
                continue
            out.append(ch)
            break
        if ch == '"':
            return "".join(out)
        out.append(ch)
        i += 1
    # Truncated mid-string — return everything captured. The downstream
    # quality check / schema validator will reject if it's unsalvageable;
    # surfacing the partial body is strictly better than emitting the
    # raw `{"action":"final_answer","result":"...` envelope.
    return "".join(out)


def _unescape_json_string(s: str) -> str:
    """Reverse JSON string escapes. Tolerant of the triple-backslash mess
    Qwen3.5-9B produces on long nested arrays — falls back to manual
    pairwise replacement when strict json decoding fails."""
    try:
        return json.loads(f'"{s}"')
    except (json.JSONDecodeError, ValueError):
        return (
            s.replace("\\\\", "\\")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
        )


def canonicalize_for_retry(text: str, max_depth: int = 4) -> str:
    """Collapse multi-layer JSON escape compounding to canonical form.

    When ``_prev_output`` is fed back into a retry prompt, models like
    Qwen3.5-9B re-wrap it in their own ``final_answer`` envelope and
    re-escape inner quotes — every retry adds another escape layer, so
    by attempt 3-4 the output is unparseable triple-backslashed soup.

    Strategy: repeatedly try to parse ``text`` as JSON. If it parses to
    a list/dict, re-dump with ``ensure_ascii=False``. If it parses to a
    string (i.e. the input was a JSON-string holding more JSON), recurse
    on that string. Bounded by ``max_depth`` to keep pathological inputs
    from looping. Non-JSON text is returned unchanged.

    This is the structural counterpart to ``_unwrap_envelope``: unwrap
    strips the outer envelope, canonicalize collapses inner escape
    compounding so the next retry prompt sees clean JSON.
    """
    if not isinstance(text, str):
        return text
    current = text.strip()
    for _ in range(max_depth):
        if not current:
            return current
        first = current.lstrip()[:1]
        # Only attempt JSON parse if it actually looks like JSON. Plain
        # markdown / prose passes through untouched.
        if first not in ("[", "{", '"'):
            return current
        try:
            parsed = json.loads(current)
        except (json.JSONDecodeError, ValueError):
            return current
        if isinstance(parsed, str):
            # One escape layer peeled. If the inner string is itself JSON,
            # the loop continues; otherwise we hit the ``first not in``
            # guard next iteration.
            current = parsed.strip()
            continue
        if isinstance(parsed, (list, dict)):
            return json.dumps(parsed, ensure_ascii=False)
        return current
    return current


def _unwrap_envelope(text) -> str:
    """Strip JSON envelopes, model tokens, and degenerate repetition.

    Handles:
      - final_answer envelopes: {"action": "final_answer", "result": "..."}
      - tool_call envelopes: {"action": "tool_call", "tool": "write_file",
        "args": {"content": "..."}}
      - Model-specific tokens: <|function_call|>, <|function_result|>
      - Markdown code fences
      - Degenerate repetition (same section repeated 3+ times)

    ``text`` may be a list/dict when the agent emitted a structured
    artifact directly (e.g. step 1.10 competitor_research with
    ``artifact_schema.competitors.type == "array"`` and the agent put
    the JSON array straight into ``result.result``). Such inputs are
    serialized to JSON text first so the downstream string operations
    don't blow up with ``'list' object has no attribute 'strip'``
    (mission 57 task 4581 workflow_advance crashed here 2026-04-27).
    Anything else non-string is coerced via ``str()``; ``None`` and
    empty string short-circuit to ``""``.
    """
    import re as _re

    if text is None:
        return ""
    if not isinstance(text, str):
        if isinstance(text, (list, dict)):
            try:
                text = json.dumps(text, ensure_ascii=False)
            except (TypeError, ValueError):
                text = str(text)
        else:
            text = str(text)

    # ── Strip model-specific tokens ──
    stripped = text.strip()
    for token in ("<|function_call|>", "<|function_result|>",
                  "<|im_start|>", "<|im_end|>"):
        stripped = stripped.replace(token, "")
    stripped = stripped.strip()

    # ── Strip markdown code fences ──
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        stripped = stripped.rsplit("```", 1)[0].strip()

    # ── Try JSON parse for known envelope types ──
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            # CallResult / structured_emit envelope — the HaLLederiz Kadir
            # response dict {"content": "<artifact>", "model": ...,
            # "usage": ..., "task": "structured_emit", ...} where the real
            # array/object artifact is DOUBLE-ENCODED as a JSON string under
            # "content". Peel it and recurse (content may itself be a bare
            # array/object, or — rarely — a nested final_answer). Guard
            # tightly: require a str content AND >=2 CallResult-only sibling
            # markers AND no result/action keys, so a legit artifact that
            # merely has a "content" field is not mistaken for an envelope.
            # (mission_79 2026-05-30: every array/object artifact persisted
            # as this envelope → validators counted ~0 items / missing field.)
            _CR_MARKERS = ("model", "model_name", "usage", "cost", "latency",
                           "ran_on", "provider", "is_local", "capability_score")
            if (
                "result" not in obj
                and "action" not in obj
                and isinstance(obj.get("content"), str)
                and sum(1 for k in _CR_MARKERS if k in obj) >= 2
            ):
                return _unwrap_envelope(obj["content"])
            # final_answer envelope
            if "result" in obj:
                val = obj["result"]
                stripped = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
            # write_file tool_call envelope — extract the file content
            elif obj.get("action") == "tool_call" and obj.get("tool") == "write_file":
                args = obj.get("args", {})
                if isinstance(args, dict) and "content" in args:
                    val = args["content"]
                    stripped = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass

    # ── Fallback for broken / truncated JSON ──
    # The structural extractor walks the string char-by-char respecting
    # JSON escape rules and returns the partial body when truncated.
    # Replaces the older regex pair which silently failed on outputs
    # without a closing-marker sentinel.
    if '"result"' in stripped and '"final_answer"' in stripped:
        body = _extract_json_string_field(stripped, "result")
        if body is not None:
            stripped = _unescape_json_string(body)

    if '"write_file"' in stripped and '"content"' in stripped:
        body = _extract_json_string_field(stripped, "content")
        if body is not None:
            stripped = _unescape_json_string(body)

    return stripped


def _single_produces(produces) -> bool:
    """True when the step declares exactly one ``.md``/``.json`` produces path.

    Shared by ``materialize_produces`` (which canonicalizes only the single
    case) and ``post_execute_workflow_step`` (which promotes that canonical to
    the step's ``result``) so the two predicates can never drift.
    """
    if not isinstance(produces, list):
        return False
    return len([e for e in produces
                if isinstance(e, str) and e.endswith((".md", ".json"))]) == 1


async def materialize_produces(ctx: dict, task: dict, result, output_value):
    """Sole writer of declared ``produces`` paths.

    For each single declared ``.md`` / ``.json`` produces path, gather the
    on-disk content (whatever the agent's write_file left) and ``output_value``
    as competing candidates, pick the schema-best via ``select_canonical``,
    stamp ``mission_id`` front-matter idempotently, and write the canonical
    path last. Returns the canonical content as the new ``output_value`` (when
    a single path is declared) so the in-memory schema gate validates exactly
    what landed on disk. Fully fail-soft — never raises, always leaves a file.
    """
    mission_id = task.get("mission_id") or ctx.get("mission_id")
    produces = ctx.get("produces") or []
    if not (output_value and mission_id) or not isinstance(produces, list):
        return output_value

    # Mechanical siblings (workflow_advance, git_commit, ...) inherit ctx but
    # do not emit artifacts — mirror the schema gate's _is_producer guard.
    executor = (task.get("executor") or ctx.get("executor") or "")
    agent_type = (task.get("agent_type") or ctx.get("agent_type") or "")
    if executor == "mechanical" or agent_type == "mechanical":
        return output_value

    from coulson.grounding import select_canonical, stamp_front_matter
    # Read WORKSPACE_DIR dynamically from its source module so test
    # monkeypatching of ``src.tools.workspace.WORKSPACE_DIR`` takes effect.
    import src.tools.workspace as _ws

    schema = ctx.get("artifact_schema") or {}

    def _schema_ok(c: str) -> bool:
        try:
            return bool(validate_artifact_schema(c, schema)[0])
        except Exception:
            return False

    single = _single_produces(produces)
    # When the step carries an artifact_schema, _apply_auto_strip removes the
    # write tools (unless _allow_write_tools) — the agent CANNOT have written
    # disk this run, so any on-disk file is a STALE artifact from a prior failed
    # attempt. The fresh output_value (this run's final_answer) must outrank it,
    # even when the stale file is itself schema-valid (task 524364: a gate-failed
    # 'dead' report kept being resurrected over the corrected 'active' result).
    #
    # CRITICAL: this predicate MUST match _apply_auto_strip. Write tools are
    # stripped ONLY for structured-only schemas WITHOUT a free-form (.md) produces
    # path — markdown/string schemas AND markdown produces (even under an
    # object/array schema) KEEP write tools. A markdown step's disk file is
    # therefore the agent's FRESH write (not stale) and must outrank the
    # narration-prone final_answer. Treating any non-empty schema as write-stripped
    # was a predicate drift that flipped markdown order to [output_value, disk] and
    # let a narration clobber the agent's clean file (task 567379 [0.6a.draft]
    # non_goals_draft). Keying off schema type ALONE (ignoring the .md produces
    # form) was the SAME drift for object-schema markdown steps (m90 5.0c user_flow
    # object schema + user_flow.md produces → strip → analyst narrated → clobber).
    from coulson import _write_tools_redundant
    write_stripped = (
        isinstance(schema, dict) and bool(schema)
        and _write_tools_redundant(schema, produces)
        and not ctx.get("_allow_write_tools")
    )
    canonical_out = output_value
    for entry in produces:
        if not (isinstance(entry, str) and entry.endswith((".md", ".json"))):
            continue
        abs_path = entry if os.path.isabs(entry) else os.path.join(_ws.WORKSPACE_DIR, entry)
        disk = None
        try:
            with open(abs_path, encoding="utf-8") as fh:
                disk = fh.read()
        except OSError:
            disk = None
        # Priority is CONDITIONAL (see write_stripped above):
        #   • write NOT stripped → the agent's fresh on-disk write outranks
        #     output_value; a rich valid file is preserved (intake #73), only a
        #     disk file that FAILS the schema yields to the result (mission 81).
        #   • write stripped (schema'd step) → disk is necessarily a STALE prior
        #     attempt, so the fresh output_value outranks it (task 524364).
        #
        # output_value is the step's SINGLE result, so it can only stand in for
        # the produces path of a single-artifact step. For a multi-produces step
        # (agent-writes-each: ADR decision+register, screen_inventory+shared_shell)
        # output_value belongs to one logical artifact and must never overwrite a
        # sibling file — so it is dropped from the candidate list and each file is
        # materialized from its own on-disk content (Cut #2, spec 2026-06-07).
        if not single:
            candidates = [disk]
        elif write_stripped:
            candidates = [output_value, disk]   # fresh result outranks stale disk
        else:
            candidates = [disk, output_value]   # agent's fresh write outranks result
        chosen = select_canonical(candidates, _schema_ok)
        if not isinstance(chosen, str):
            continue
        kind = "json" if entry.endswith(".json") else "md"
        chosen = stamp_front_matter(chosen, int(mission_id), kind)
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(chosen)
            logger.info(
                f"[Workflow Hook] materialize_produces -> {abs_path} "
                f"({len(chosen)} chars)"
            )
        except OSError as e:
            logger.debug(f"[Workflow Hook] materialize write failed {abs_path}: {e}")
            continue
        if single:
            canonical_out = chosen
    return canonical_out


def resolve_produces_artifact(source: dict, source_ctx: dict):
    """Single source of truth for a produces-step's artifact at GRADE time.

    The artifact a step produces is the CANONICAL form ``materialize_produces``
    wrote to disk — written UNCONDITIONALLY for every declared single produces
    path — NOT ``tasks.result``. On a schema'd step ``_apply_auto_strip`` removes
    ``write_file``, so the agent cannot write disk and instead NARRATES its
    final_answer ("Wrote X.md containing …"); ``tasks.result`` then holds that
    narration, which carries no artifact body. The grade chain (deterministic
    schema gate + LLM grader, which both read ``source['result']``) must judge
    the disk artifact — exactly as ``verify_charter_shape`` and downstream
    consumers (``coulson/context.py``) already do — instead of false-rejecting a
    correct on-disk artifact (task 567373 [0.1] product_charter: disk had all 5
    ``##`` sections, result was a 570-char narration → "missing all 5 sections"
    → degenerate-repeat DLQ).

    Returns the fence-unwrapped disk content for a SINGLE-``produces`` step, or
    ``None`` when the step is not single-produces or the disk file is absent /
    empty (caller keeps ``source['result']``). Pull, not push: every reader
    resolves the one materialized source rather than relying on a fragile
    canonical→``tasks.result`` back-write surviving every persistence path.
    """
    produces = source_ctx.get("produces") or []
    if not _single_produces(produces):
        return None
    mission_id = source.get("mission_id") or source_ctx.get("mission_id")
    import src.tools.workspace as _ws
    from coulson.grounding import unwrap_fenced_artifact
    for entry in produces:
        if not (isinstance(entry, str) and entry.endswith((".md", ".json"))):
            continue
        rel = entry.replace("{mission_id}", str(mission_id)) if mission_id is not None else entry
        abs_path = rel if os.path.isabs(rel) else os.path.join(_ws.WORKSPACE_DIR, rel)
        try:
            with open(abs_path, encoding="utf-8") as fh:
                disk = fh.read()
        except OSError:
            return None
        if not (isinstance(disk, str) and disk.strip()):
            return None
        u = unwrap_fenced_artifact(disk)
        return u if (isinstance(u, str) and u.strip()) else disk
    return None


def _top_level_required_field_names(rule: dict) -> list[str]:
    """Return required (non-optional) top-level field names from an object rule.

    Used by the text fallback for small LLMs that produce structured prose
    instead of JSON. Returns empty list for non-object rules. Accepts
    either canonical (``fields``) or legacy (``required_fields``) form —
    delegates to the dialect normalizer.
    """
    if not isinstance(rule, dict) or rule.get("type") != "object":
        return []
    from src.workflows.engine.schema_dialect import _normalize_rule
    rule = _normalize_rule(rule)
    fields = rule.get("fields") or {}
    return [
        fname for fname, frule in fields.items()
        if isinstance(frule, dict) and not frule.get("optional")
    ]


def _extract_artifact_value(output_value, artifact_name: str, rtype: str):
    r"""Parse output and extract the value for ``artifact_name``.

    Returns the extracted value (dict/list/str) or None if parse failed.
    Tries direct JSON, artifact-name-keyed wrapper, single matching value
    of correct type, then ``\`\`\`json ... \`\`\`\`` code block.
    """
    if isinstance(output_value, (dict, list)):
        data = output_value
    else:
        try:
            data = json.loads(output_value) if isinstance(output_value, str) else output_value
        except (json.JSONDecodeError, TypeError):
            # Code-block extraction
            import re as _re2
            text = str(output_value)
            blk = _re2.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
            if blk:
                try:
                    data = json.loads(blk.group(1))
                except (json.JSONDecodeError, TypeError):
                    return None
            else:
                return None

    # Direct shape match
    expected_pytype = (
        dict if rtype == "object"
        else list if rtype == "array"
        else None
    )
    if expected_pytype is not None and isinstance(data, expected_pytype):
        # If it's a dict-typed artifact and the dict happens to wrap the
        # named artifact (``{"openapi_spec": {...}}``), unwrap.
        if rtype == "object" and isinstance(data, dict) and artifact_name in data:
            inner = data[artifact_name]
            if isinstance(inner, dict):
                return inner
        return data

    # Artifact-keyed wrapper
    if isinstance(data, dict):
        if artifact_name in data:
            return data[artifact_name]
        # Single value of correct type
        if expected_pytype is not None:
            matches = [v for v in data.values() if isinstance(v, expected_pytype)]
            if len(matches) == 1:
                return matches[0]
    return None


def _is_empty_required_value(val) -> bool:
    """Decide if a required-field value is a placeholder rather than real
    content. Used by validate_artifact_schema to guard against constrained-
    decoding passes that satisfy "field present" with empty containers
    (mission 46 task 2964: ``[{feature_id: {}, feature_name: {}, ...}]`` —
    schema validator passed because all fields existed, but every value
    was an empty object).

    Empty placeholders considered missing:
      * ``None``
      * empty string / whitespace-only string
      * empty dict ``{}``
      * empty list ``[]``

    Real content (boolean ``False``, number ``0``, non-empty string, etc.)
    is NOT empty even though it is falsy in Python.
    """
    if val is None:
        return True
    if isinstance(val, str):
        return not val.strip()
    if isinstance(val, dict):
        return len(val) == 0
    if isinstance(val, list):
        return len(val) == 0
    return False


# ── Lazy-true evidence patterns ─────────────────────────────────────────
#
# Default evidence tokens by verification flag name. The post-execute hook
# checks the task's audit_log (since this attempt's started_at) for tool
# executions whose ``target`` or ``details`` contain at least one token. If
# the flag is true but no token matched, the agent lied — fail with a
# didactic message and route to retry. Per-schema ``evidence_for`` map
# overrides this default.
_DEFAULT_EVIDENCE_TOKENS: dict[str, list[str]] = {
    "health_check_verified": [
        "curl", "wget", "http://", "https://", "requests.get",
        "fetch", "axios", ".get(",
    ],
    "health_check_passed": [
        "curl", "wget", "http://", "https://", "fetch", "axios",
    ],
    "dev_server_verified": [
        "curl", "wget", "http://", "fetch", "playwright", "puppeteer",
        "axios", "browser",
    ],
    "dependencies_installed": [
        "pip install", "pip3 install", "pipenv install", "poetry add",
        "poetry install", "npm install", "npm i ", "npm ci",
        "yarn add", "yarn install", "pnpm install", "pnpm add",
        "go get", "go mod", "cargo add", "cargo build",
        # KutAI-internal tools that wrap install commands.
        "verify_deps", "scaffold",
    ],
    "connection_verified": [
        "psql", "mysql", "mongo", "redis-cli", "sqlite3",
        "SELECT", "ping", "connect(", ".connect", "create_engine",
    ],
    "smoke_tests_passed": [
        "pytest", "jest", "mocha", "playwright", "vitest", "go test",
        "cargo test", "rspec", "phpunit",
    ],
    "all_passed": [
        "pytest", "jest", "mocha", "playwright", "vitest", "go test",
        "cargo test", "test", "check",
    ],
    "all_resolved": [
        "patch", "fix", "edit", "write_file", "modify",
    ],
    "data_seeded": [
        "INSERT", "seed", "fixture", "load_data", "seeders",
        "psql", "mysql", "mongo",
    ],
    "headers_configured": [
        "helmet", "headers", "X-Frame", "Content-Security-Policy",
        "Strict-Transport", "write_file",
    ],
    "patches_applied": [
        "patch", "apt", "yum", "apk", "upgrade", "update", "pip install",
        "npm install", "fix",
    ],
    "fixes_applied": [
        "edit", "write_file", "patch", "fix", "git commit",
    ],
    "optimizations_applied": [
        "edit", "write_file", "patch", "optimize", "git commit",
    ],
    "sprint_id_completed": [
        "git commit", "git tag", "git push", "merge",
    ],
    "features_completed": [
        "git commit", "git tag", "git push", "merge",
    ],
}


def _truthy_flag_fields(rule: dict) -> list[str]:
    """Return field names whose rule requires ``equals: true`` (canonical or
    legacy ``must_be_true`` form). Walks one level — verification flags are
    top-level booleans on the artifact object."""
    if not isinstance(rule, dict):
        return []
    must_true = list(rule.get("must_be_true") or [])
    fields = rule.get("fields") or {}
    for fname, frule in fields.items():
        if (
            isinstance(frule, dict)
            and frule.get("type") == "boolean"
            and frule.get("equals") is True
        ):
            must_true.append(fname)
    return list(dict.fromkeys(must_true))  # dedupe, keep order


async def _check_truthy_evidence(
    task: dict, output_value: str, schema: dict
) -> Optional[str]:
    """For each verification flag set to ``true`` in the artifact, confirm
    that the agent actually ran a matching command during this attempt.

    Returns an error message (didactic) if a flag is unsupported by audit
    evidence, or None if every truthy flag has evidence.

    Only inspects audit_log entries newer than ``task.started_at`` so that
    evidence from earlier attempts doesn't whitewash a lazy retry.
    """
    if not schema or not output_value:
        return None
    task_id = task.get("id")
    if not task_id:
        return None

    # Parse output once.
    try:
        if isinstance(output_value, str):
            obj = json.loads(_unwrap_envelope(output_value))
        else:
            obj = output_value
    except (json.JSONDecodeError, TypeError):
        return None  # validator already caught structural problems

    # Walk each artifact rule. ``schema`` may have artifact_name -> rule.
    for art_name, rule in schema.items():
        if not isinstance(rule, dict):
            continue
        # Locate the artifact value in the output. If the output IS the
        # artifact (e.g. the agent returned the dict directly), use it.
        if isinstance(obj, dict) and art_name in obj and isinstance(obj[art_name], dict):
            art_value = obj[art_name]
        elif isinstance(obj, dict):
            art_value = obj
        else:
            continue

        flag_fields = _truthy_flag_fields(rule)
        if not flag_fields:
            continue

        # Evidence map: schema override beats defaults.
        per_schema = rule.get("evidence_for") or {}

        # Filter to flags that the agent claimed are True.
        claimed_true = [
            f for f in flag_fields
            if isinstance(art_value.get(f), bool) and art_value[f] is True
        ]
        if not claimed_true:
            continue

        # Pull audit events for this task.
        try:
            from ...infra.audit import get_audit_log
        except ImportError:
            return None
        entries = await get_audit_log(task_id=task_id, limit=500)
        # Filter by attempt window: events newer than started_at.
        started_at = task.get("started_at") or ""
        if started_at:
            entries = [e for e in entries if (e.get("timestamp") or "") >= started_at]
        # Build a single haystack of target+details for substring scan.
        haystack = "\n".join(
            f"{e.get('target') or ''} {e.get('details') or ''}".lower()
            for e in entries
        )

        unsupported: list[str] = []
        for f in claimed_true:
            tokens = per_schema.get(f) or _DEFAULT_EVIDENCE_TOKENS.get(f)
            if not tokens:
                continue  # no evidence rule for this field — skip
            tokens_lower = [t.lower() for t in tokens]
            if not any(t in haystack for t in tokens_lower):
                unsupported.append(f)

        if unsupported:
            sample_tokens = {
                f: (per_schema.get(f) or _DEFAULT_EVIDENCE_TOKENS.get(f) or [])[:3]
                for f in unsupported
            }
            return (
                f"'{art_name}' lazy-true detected for {unsupported}: claimed "
                f"true but audit_log for this attempt shows NO matching "
                f"command. Verification flags require ACTUAL work, not just "
                f"flipping the bool. Examples of expected evidence: "
                f"{sample_tokens}. Run the verification command via the "
                f"shell tool, then emit true. If the check genuinely fails, "
                f"emit false and report the blocker — this step has not "
                f"completed."
            )

    return None


async def resolve_dynamic_constraints(
    schema: dict,
    mission_id: int | str | None,
) -> dict:
    """Resolve dynamic constraints inside a schema against upstream artifacts.

    Currently supports:

    - ``min_items_from``: ``{"artifact": "<name>"[, "path": "a.b"][, "floor": int]}``
      — replaces ``min_items`` with the upstream artifact's item count
      (parsed JSON array, optionally drilled into via dot-separated
      ``path``). Failures degrade to ``floor`` (default 1) so a missing
      upstream doesn't silently pass tiny backlogs.

    Returns a deep copy of ``schema`` with resolved literal constraints. Safe
    to call with ``mission_id=None`` — the function becomes a no-op (the
    floor takes over).
    """
    import json as _json
    import copy as _copy

    if not isinstance(schema, dict) or not schema:
        return schema

    resolved = _copy.deepcopy(schema)

    store = None
    if mission_id is not None:
        try:
            store = get_artifact_store()
        except Exception:
            store = None

    async def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "array" and "min_items_from" in node:
                spec = node.get("min_items_from") or {}
                upstream_name = spec.get("artifact") if isinstance(spec, dict) else None
                inner_path = spec.get("path") if isinstance(spec, dict) else None
                floor = int(spec.get("floor", 1)) if isinstance(spec, dict) else 1
                resolved_count: int | None = None
                if upstream_name and store is not None and mission_id is not None:
                    try:
                        raw = await store.retrieve(mission_id, upstream_name)
                        if raw:
                            try:
                                parsed = _json.loads(raw)
                                # Drill into a dot-path (e.g. "mvp_feature_list")
                                # so the schema author can point at a list nested
                                # in an upstream object artifact.
                                if isinstance(inner_path, str) and inner_path:
                                    cursor: Any = parsed
                                    for part in inner_path.split("."):
                                        if isinstance(cursor, dict):
                                            cursor = cursor.get(part)
                                        else:
                                            cursor = None
                                            break
                                    parsed = cursor
                                if isinstance(parsed, list):
                                    resolved_count = len(parsed)
                            except (ValueError, TypeError):
                                pass
                    except Exception as e:
                        logger.debug(
                            f"resolve_dynamic_constraints: lookup of "
                            f"{upstream_name!r} failed: {e}"
                        )
                effective = max(resolved_count or 0, floor)
                # Take the larger of any explicit min_items already on the rule
                # and the upstream-derived count. Authors who wrote both want
                # the stricter of the two.
                existing = int(node.get("min_items") or 0)
                node["min_items"] = max(effective, existing)
                # Drop the source key so downstream consumers (translator,
                # checklist) see only the canonical literal.
                node.pop("min_items_from", None)
            for v in node.values():
                await _walk(v)
        elif isinstance(node, list):
            for v in node:
                await _walk(v)

    await _walk(resolved)
    return resolved


def validate_artifact_schema(
    output_value: str, schema: dict, inputs: dict | None = None,
    *, produces_markdown: bool = False,
) -> tuple[bool, str]:
    """Validate an artifact output against its schema definition.

    Returns (is_valid, error_message). error_message is empty if valid.

    ``inputs`` (optional ``{artifact_name: parsed_value}``) anchors the
    dialect's ``empty_ok_when_input_empty`` per-field exemption to upstream
    input artifacts. Omit it and validation behaves exactly as before.

    ``produces_markdown`` — True when the validated artifact is a markdown
    file on disk (the step's produces is ``*.md``). For an ``object``/``array``
    schema the structured value can never be extracted from markdown, so the
    prose text-fallback degenerates into a literal substring search for the
    field NAMES — meaningless for prose (false-reject AND false-pass; mission-90
    567452 [5.0c] user_flow searched for the literal ``mermaid_per_surface`` in
    a flow doc). Every such step carries a ``verify_*_shape`` mechanical check
    that is the authoritative validator, so we DEFER to it and skip the
    fallback. The structured (data-extracted) dialect path is unaffected — a
    real JSON object is still validated strictly.
    """
    if not schema:
        return True, ""

    # Unwrap final_answer JSON envelope if present — agents sometimes
    # wrap their results in {"action": "final_answer", "result": "..."}
    # Uses _unwrap_envelope which handles malformed JSON (unescaped quotes).
    if isinstance(output_value, str):
        output_value = _unwrap_envelope(output_value)

    from src.workflows.engine.schema_dialect import validate_value as _dialect_validate

    for artifact_name, rules in schema.items():
        if not isinstance(rules, dict):
            continue  # skip non-artifact entries like max_output_chars
        schema_type = rules.get("type", "string")

        if schema_type in ("object", "array"):
            data = _extract_artifact_value(output_value, artifact_name, schema_type)
            if data is not None:
                err = _dialect_validate(rules, data, path=artifact_name, inputs=inputs)
                if err:
                    return False, f"Schema validation: {err}"
                continue  # passed dialect check

            # Could not extract a structured value. For a markdown produces this
            # is expected (markdown is not JSON); the prose text-fallback below
            # would substring-match field NAMES against prose — pure noise. Defer
            # to the step's verify_*_shape check (mission-90 567452 [5.0c]).
            if produces_markdown:
                continue

            # Parse failed — fall back for small LLMs that emit prose.
            if schema_type == "object":
                required = _top_level_required_field_names(rules)
                # Honor the empty-scope exemption EXACTLY as the JSON-parsed
                # dialect path does (line above): a field marked
                # empty_ok_when_input_empty whose upstream input is empty is
                # legitimately absent and must NOT be flagged "missing content
                # about" (task 567396 [1.11a] compliance_overlay: analyst emitted
                # PROSE for an empty-scope overlay, jurisdictions=[]; the prose
                # fallback ignored the exemption the JSON path honors → DLQ).
                if required:
                    from src.workflows.engine.schema_dialect import (
                        _empty_exemption_granted, _normalize_rule,
                    )
                    _nfields = _normalize_rule(rules).get("fields") or {}
                    required = [
                        f for f in required
                        if not _empty_exemption_granted(_nfields.get(f, {}), inputs)
                    ]
                if required:
                    import re as _re_obj
                    def _norm(s):
                        s = s.lower().replace("'", "").replace("’", "")
                        return _re_obj.sub(r"[‐-―−\-_]", " ", s)
                    text_norm = _norm(str(output_value))
                    _QUALIFIER_SUFFIXES = {
                        "level", "rate", "count", "status", "type",
                        "value", "score", "index", "ratio",
                    }
                    missing = []
                    for f in required:
                        words = _norm(f).split()
                        core = [w for w in words if w not in _QUALIFIER_SUFFIXES]
                        if not core:
                            core = words
                        if not all(w in text_norm for w in core):
                            missing.append(f)
                    if missing:
                        return False, f"'{artifact_name}' missing content about: {missing}"
                continue

            # Array text fallback — count list items.
            min_items = int(rules.get("min_items") or 0)
            if min_items > 0:
                import re as _re
                text = str(output_value)
                items = _re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]|\-|\*)\s+\S', text)
                if len(items) < min_items:
                    table_rows = [
                        line for line in text.split("\n")
                        if line.strip().startswith("|")
                        and not _re.match(r'^\s*\|[\s\-:]+\|', line)
                        and not _re.match(r'^\s*\|\s*:?-', line)
                    ]
                    if len(table_rows) > 1:
                        items = table_rows[1:]
                if len(items) < min_items and text.lstrip().startswith("["):
                    json_items = _re.findall(r'\{[^{}]{20,}\}', text)
                    if len(json_items) >= min_items:
                        items = json_items
                if len(items) < min_items:
                    return False, f"'{artifact_name}' has ~{len(items)} list items, need >= {min_items}"
            continue

        if schema_type == "string":
            min_length = rules.get("min_length", 1)
            if not output_value or len(str(output_value).strip()) < min_length:
                return False, f"'{artifact_name}' is too short (min {min_length} chars)"

        elif schema_type == "markdown":
            required_sections = rules.get("required_sections", [])
            text = str(output_value)
            text_lower = text.lower()
            # Normalize numbered headers: "## 1. Vision" → "## Vision"
            import re as _re
            text_normalized = _re.sub(
                r'^(#{1,4})\s*\d+[\.\)]\s*',
                r'\1 ',
                text_lower,
                flags=_re.MULTILINE,
            )
            # Check for actual markdown headers (## Section or ### Section),
            # not just substring mentions like "Vision (streamlining...)".
            # The header MUST start a line (MULTILINE ^) — an inline / backticked
            # prose mention such as "a `# Non-goals` body section" must NOT count
            # (task 567379: a narration that *described* writing the doc passed
            # the old `f"# {s}" in text` substring check and validated as a real
            # artifact, clobbering the writer's clean on-disk file).
            missing = []
            for s in required_sections:
                s_lower = s.lower()
                s_esc = _re.escape(s_lower)
                # Accept (all line-anchored): ## Vision / ### Vision / # Vision,
                # **Vision**, and a bare/setext title line "Vision" / "Vision\n---".
                has_header = bool(
                    _re.search(rf'^\s*#{{1,4}}\s*{s_esc}\b', text_normalized, _re.MULTILINE)
                    or _re.search(rf'^\s*\*\*{s_esc}\*\*', text_normalized, _re.MULTILINE)
                    or _re.search(rf'^\s*{s_esc}\s*$', text_normalized, _re.MULTILINE)
                )
                if not has_header:
                    missing.append(s)
            if missing:
                return False, f"'{artifact_name}' missing sections: {missing}"

    return True, ""

# ── Disguised failure detection ────────────────────────────────────────────

# Phrases that indicate the agent failed but wrapped it in final_answer
_FAILURE_PHRASES = [
    # Direct failure statements
    "cannot be completed",
    "cannot be performed",
    "cannot proceed",
    "could not be performed",
    "could not be completed",
    "could not produce a final answer",
    "unable to complete",
    "cannot complete",
    "cannot execute",
    "cannot run",
    # Tool/environment failures
    "shell tool is blocked",
    "shell tool is disabled",
    "shell tool is completely",
    "shell tool was blocked",
    "shell tool has failed",
    "shell is blocked",
    "tool is temporarily disabled",
    "tool is blocked",
    "tool is disabled",
    "tool block",
    "tool failed",
    "tool is completely",
    "tool is non-functional",
    "tools are available",  # "no alternative tools are available"
    # Status keywords
    "execution blocked",
    "status: blocked",
    "status: failed",
    "cannot proceed until",
    "blocked - cannot",
    "critical blocker",
    "no test execution possible",
    "cannot be generated without",
    "not run - shell",
    "not run -",
    "workspace appears empty",
    "workspace is inaccessible",
    "inaccessible due to",
    # Verification failures
    "verification status: failed",
    "verification failed",
    "task failed",
    "failed / incomplete",
    "creation failed",
    "suite creation failed",
    # Missing/incomplete outputs
    "documentation is incomplete",
    "no artifact",
    "artifact cannot be generated",
    "not yet created",
    "placeholder text",
]

# Output starts with raw JSON tool_call — agent never produced final_answer
_RAW_TOOL_CALL_PREFIXES = [
    '{"action":"tool_call"',
    '{"action": "tool_call"',
    '{"action":"toolcall"',
    '{"action": "toolcall"',
]

# Phrases that look like failure but are legitimate analysis
_FALSE_POSITIVE_PHRASES = [
    "competitor",       # "competitor failed to..." is analysis, not failure
    "risk if wrong",    # risk assessment
    "failure mode",     # design docs discussing failure modes
    "error handling",   # architecture discussing error handling
    "error states",     # UX discussing error states
    "empty states",     # UX design for empty states
]

# Hard failure indicators — these override false positives. Each must
# be specific enough that a positive sentence form ("the shell tool is
# working fine") doesn't trip detection. Bare ``"shell tool"`` was
# removed 2026-04-27 because it was matching legit references to the
# tool in design docs and post-mortems.
_HARD_FAILURE_PHRASES = [
    "shell tool is blocked",
    "shell tool is disabled",
    "shell tool was blocked",
    "tool is blocked",
    "tool is disabled",
    "cannot proceed until",
    "execution blocked",
    "_ctx is not defined",
    "not in allowlist",
]


_NEGATION_PRECEDERS = (
    "not ", "no ", "n't ", "non-", "without ", "never ",
    "isn't ", "wasn't ", "aren't ", "weren't ",
)


_SENTENCE_BOUNDARY_CHARS = ".!?\n;"


def _phrase_matches_unnegated(text_lower: str, phrase: str) -> bool:
    """Return True iff ``phrase`` appears in ``text_lower`` AT LEAST ONCE
    without a negation token earlier in the SAME sentence.

    Mission 57 task 4373 surfaced the false positive: a problem-statement
    artifact contained ``"is not a critical blocker"`` and the substring
    ``"critical blocker"`` (in ``_FAILURE_PHRASES``) tripped detection
    five retries in a row. The output was structurally fine — the
    negated phrase was descriptive content, not a failure report.

    Heuristic: for each occurrence, walk back to the nearest sentence
    boundary (``. ! ? \\n ;``) or 80 chars, whichever is closer. Check
    that window for a negation token. If ALL occurrences are negated,
    treat as no match. Otherwise at least one positive use exists ->
    match. Sentence-bounded so a negation in an EARLIER sentence
    doesn't mask a positive use later in the text.
    """
    if phrase not in text_lower:
        return False
    start = 0
    while True:
        idx = text_lower.find(phrase, start)
        if idx == -1:
            return False  # all checked occurrences were negated
        # Find the start of the current sentence.
        win_start = max(0, idx - 80)
        for j in range(idx - 1, win_start - 1, -1):
            if text_lower[j] in _SENTENCE_BOUNDARY_CHARS:
                win_start = j + 1
                break
        window = text_lower[win_start:idx]
        if not any(neg in window for neg in _NEGATION_PRECEDERS):
            return True
        start = idx + 1


def _is_disguised_failure(output: str) -> bool:
    """Detect if a 'completed' result is actually a failure report."""
    if not output or len(output) < 10:
        return False

    stripped = output.strip()

    # Raw tool_call JSON as the result — agent never produced an answer
    for prefix in _RAW_TOOL_CALL_PREFIXES:
        if prefix in stripped[:100]:
            return True

    lower = output.lower()

    # Hard failure indicators — always a failure regardless of context.
    # Still negation-aware: ``"the shell tool was working"`` shouldn't
    # be flagged just because ``"shell tool"`` is hard-list.
    if any(_phrase_matches_unnegated(lower, p) for p in _HARD_FAILURE_PHRASES):
        return True

    # Check for false positives — legitimate analysis about failures
    has_false_positive = any(fp in lower for fp in _FALSE_POSITIVE_PHRASES)

    # Count failure indicators (negation-aware).
    hits = sum(
        1 for phrase in _FAILURE_PHRASES
        if _phrase_matches_unnegated(lower, phrase)
    )

    # 2+ failure phrases = almost certainly a disguised failure
    if hits >= 2:
        return True

    # 1 failure phrase without false positive context
    if hits >= 1 and not has_false_positive:
        return True

    return False



def _structural_summary(text: str, target: int = 1500) -> str:
    """Fallback summary without LLM — keep headings + first lines."""
    if len(text) <= target:
        return text
    lines = text.split("\n")
    out: list[str] = []
    total = 0
    for line in lines:
        s = line.strip()
        if s.startswith("#") or (out and not out[-1].startswith("#") and not s):
            out.append(s)
            total += len(s) + 1
        elif s and (not out or out[-1].startswith("#")):
            out.append(s[:200])
            total += min(len(s), 200) + 1
        if total >= target:
            break
    return "\n".join(out) if out else text[:target]


# ── Module-level singleton ─────────────────────────────────────────────────

_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Return the module-level ArtifactStore singleton (lazy init)."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore(use_db=True)
    return _artifact_store


# ── Helper functions ───────────────────────────────────────────────────────


async def should_skip_workflow_step(task: dict) -> tuple[bool, str]:
    """Evaluate the step's ``skip_when_expr`` against loaded artifacts.

    Returns ``(True, reason)`` when the step should short-circuit as
    skipped, else ``(False, "")``. Safe default is to NOT skip on any
    parse or lookup error — misconfigured expressions should not silently
    suppress work. Current supported expression shape:

        <artifact_name>.<dot.path> != '<literal>'
        <artifact_name>.<dot.path> == '<literal>'

    Runtime-loaded — the artifact store is consulted each time the task
    is dispatched so skip conditions reflect post-completion state of
    dependencies, not whatever was in flight when the task was expanded.
    """
    ctx = _parse_context(task)
    mission_id = ctx.get("mission_id") or task.get("mission_id")
    if mission_id is None:
        return False, ""

    expr = ctx.get("skip_when_expr")
    if not expr or not isinstance(expr, str):
        return False, ""

    import re as _re
    m = _re.match(
        r"\s*([A-Za-z_][\w]*)((?:\.[A-Za-z_][\w]*)+)\s*(==|!=)\s*'([^']*)'\s*$",
        expr,
    )
    if not m:
        logger.warning(
            f"[Workflow Hook] skip_when_expr {expr!r} unparseable — not skipping"
        )
        return False, ""

    artifact_name = m.group(1)
    path = m.group(2).lstrip(".").split(".")
    op = m.group(3)
    literal = m.group(4)

    try:
        store = get_artifact_store()
        raw_artifact = await store.retrieve(mission_id, artifact_name)
    except Exception as exc:
        logger.debug(
            f"[Workflow Hook] skip_when lookup failed for {artifact_name}: {exc}"
        )
        return False, ""

    if raw_artifact is None:
        # Artifact not produced yet — can't evaluate; let step run.
        return False, ""

    try:
        data = json.loads(raw_artifact) if isinstance(raw_artifact, str) else raw_artifact
    except (json.JSONDecodeError, TypeError):
        return False, ""

    current = data
    for segment in path:
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            current = None
            break

    matched = (op == "==" and current == literal) or (op == "!=" and current != literal)
    if matched:
        return True, f"{artifact_name}.{'.'.join(path)} {op} {literal!r}"
    return False, ""


def _scan_empty_exemption_markers(node: Any, out: set | None = None) -> set:
    """Collect every ``empty_ok_when_input_empty`` marker string in a schema."""
    if out is None:
        out = set()
    if isinstance(node, dict):
        m = node.get("empty_ok_when_input_empty")
        if isinstance(m, str) and m:
            out.add(m)
        for v in node.values():
            _scan_empty_exemption_markers(v, out)
    elif isinstance(node, list):
        for it in node:
            _scan_empty_exemption_markers(it, out)
    return out


def collect_empty_exemption_inputs(schema: dict, mission_id: int) -> dict | None:
    """Load upstream input artifacts referenced by ``empty_ok_when_input_empty``.

    Returns ``{artifact_name: parsed_value}`` for the dialect's conditional-
    empty exemption, or ``None`` when the schema declares no marker (no file
    IO in that case — the common path). Each artifact is read from its
    produced file ``<mission_workspace>/<artifact_name>.json`` — authored by
    the UPSTREAM task, so the exemption can't be self-granted by a lazy
    producer. Unreadable/missing files are simply omitted (the gate then
    finds no proof and rejects the empty value).
    """
    markers = _scan_empty_exemption_markers(schema)
    if not markers or mission_id is None:
        return None
    # First dotted segment is the artifact name; reject any that could escape
    # the mission workspace (markers are static config, but cheap to harden).
    names = {m.split(".", 1)[0] for m in markers}
    names = {n for n in names if n and "/" not in n and "\\" not in n and ".." not in n}
    if not names:
        return None
    inputs: dict = {}
    try:
        ws = get_mission_workspace(mission_id)
    except Exception:  # noqa: BLE001 — never let the loader break the gate
        return None
    for name in names:
        try:
            with open(os.path.join(ws, f"{name}.json"), encoding="utf-8") as f:
                inputs[name] = json.load(f)
        except Exception:  # noqa: BLE001 — missing/unreadable → no proof
            continue
    return inputs or None


def is_workflow_step(context: dict) -> bool:
    """Check whether the task context marks this as a workflow step."""
    return bool(context.get("is_workflow_step"))


def extract_output_artifact_names(context: dict) -> list[str]:
    """Get output_artifacts list from context, defaulting to empty."""
    return context.get("output_artifacts", [])


def _parse_context(task: dict) -> dict:
    """Parse task context, handling both dict and JSON string forms."""
    ctx = task.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    if not isinstance(ctx, dict):
        ctx = {}
    return ctx


# ──────────────────────────────────────────────────────────────────
# Removed 2026-04-27 (handoff item D, follow-up to commit f9507bb):
# `enrich_task_description` and `pre_execute_workflow_step` had ZERO
# production callers since the Task 13 orchestrator trim (~2026-04-
# 20). Live prompt build runs through `BaseAgent._build_context`
# (deps + chain context + missing-artifact NOTE + schema retry hint).
# Phase-summary injection that the dead path provided is NOT replic-
# ated elsewhere — accepted loss per architectural pick A. If a fut-
# ure mission shows steps starved for prior-phase summary context,
# port `_generate_phase_summary` calls into `_build_context`.
# ──────────────────────────────────────────────────────────────────



_review_tracker = ReviewTracker()


_POST_EXECUTE_SLOW_MS = 250


async def post_execute_workflow_step(task: dict, result: dict) -> None:
    """Post-hook: store output artifacts, evaluate conditional groups,
    trigger template expansion, and track review cycles.

    If the task is not a workflow step, returns immediately.

    Instrumented with a wall-clock timer that logs a WARNING whenever
    a single invocation exceeds ``_POST_EXECUTE_SLOW_MS``. Pump-tick
    starvation (Telegram "Query is too old" bursts on 2026-04-24) was
    traced to synchronous CPU-heavy work inside this hook — schema
    validation regex on 20-30k-char outputs, ``dogru_mu_samet.assess``,
    JSON round-trips, ``_unwrap_envelope`` on malformed long strings.
    The warning surfaces hot invocations without speculatively moving
    things into ``run_in_executor``; once profile data confirms a
    specific offender, it can be migrated.
    """
    import time as _time
    _t0 = _time.perf_counter()
    try:
        return await _post_execute_workflow_step_impl(task, result)
    finally:
        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        if _elapsed_ms > _POST_EXECUTE_SLOW_MS:
            ctx_for_log = _parse_context(task)
            output_len = 0
            try:
                ov = result.get("result") if isinstance(result, dict) else None
                if isinstance(ov, str):
                    output_len = len(ov)
            except Exception:
                pass
            logger.warning(
                f"[Workflow Hook] post_execute slow: "
                f"{_elapsed_ms:.0f}ms "
                f"task_id={task.get('id','?')} "
                f"step={ctx_for_log.get('workflow_step_id','?')} "
                f"output_chars={output_len}"
            )


async def _live_artifact_schema(mission_id, step_id: str):
    """Return *step_id*'s ``artifact_schema`` from the LIVE workflow JSON.

    ``ctx.artifact_schema`` is frozen into the task at expander time. When the
    workflow JSON is edited (e.g. go_no_go_decision.recommendation gains
    ``equals_lenient``), an already-expanded producer task keeps the stale
    snapshot — and ``advance()`` re-reads that producer row, so a ``/dlq retry``
    of the workflow_advance task re-validates against the old schema and DLQs a
    now-valid artifact forever (mission #81, 2026-06-04).

    Mirrors coulson._refresh_workflow_step_config's ``get_step`` lookup, so
    template-expanded feature steps (whose art_prefix'd ids are not in
    ``wf.steps``) return None here and keep their prefixed snapshot. Best-effort:
    returns None on any failure so the caller falls back to the frozen schema.
    """
    if not (mission_id and step_id):
        return None
    try:
        from src.infra.db import get_db
        from src.workflows.engine.loader import load_workflow
        wf_name = "i2p_v3"
        try:
            _db = await get_db()
            _cur = await _db.execute(
                "SELECT context FROM missions WHERE id = ?", (mission_id,)
            )
            _row = await _cur.fetchone()
            await _cur.close()
            if _row and _row[0]:
                _mctx = json.loads(_row[0])
                if isinstance(_mctx, str):
                    _mctx = json.loads(_mctx)
                if isinstance(_mctx, dict):
                    wf_name = _mctx.get("workflow_name") or wf_name
        except Exception:
            pass
        _wf = load_workflow(wf_name)
        _step = _wf.get_step(step_id)
        if _step and isinstance(_step.get("artifact_schema"), dict):
            return _step["artifact_schema"]
    except Exception:
        return None
    return None


async def _post_execute_workflow_step_impl(task: dict, result: dict) -> None:
    ctx = _parse_context(task)
    if not is_workflow_step(ctx):
        return

    mission_id = ctx.get("mission_id") or task.get("mission_id")
    output_names = extract_output_artifact_names(ctx)
    step_id = ctx.get("workflow_step_id", "")

    if not mission_id:
        return

    # Re-sync artifact_schema from the live workflow JSON before the schema
    # gate below. Without this, a workflow_advance retry validates the produced
    # artifact against the producer task's frozen schema snapshot — workflow
    # edits never reach in-flight missions on retry (mission #81 #291858).
    _live_schema = await _live_artifact_schema(mission_id, step_id)
    if _live_schema:
        ctx["artifact_schema"] = _live_schema

    store = get_artifact_store()

    if not output_names:
        # No output artifacts — skip artifact storage but still run
        # phase completion check below.
        workflow_phase = ctx.get("workflow_phase")
        if workflow_phase:
            await _check_phase_completion(mission_id, workflow_phase)
        return
    output_value = result.get("result", "")

    # ── Unwrap envelope from the agent result ──
    output_value = _unwrap_envelope(output_value)

    # ── Recover artifact content from workspace files ──
    # If the agent wrote output to files (via write_file), the result text
    # may be a short summary.  Collect ALL matching workspace files and
    # combine them with the result — this ensures schema validation sees
    # the full content even when multiple output artifacts exist.
    if mission_id and output_names:
        try:
            import os
            from ...tools.workspace import WORKSPACE_DIR
            artifact_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
            # Build candidate paths from BOTH the legacy `<output>.<ext>`
            # convention AND the step's declared `produces` list (which
            # may include subdirs like `.charter/`, `.intake/`). Without
            # the produces-derived candidates, a step that writes to
            # `.charter/product_charter.md` is never found by the
            # `mission_<id>/product_charter.md` probe, and result-only
            # validation runs against whatever the agent emitted on retry
            # (production 2026-05-14 mission 69 step 0.1: full charter on
            # disk at .charter/ subdir, result was a short blurb, schema
            # validation said all 5 sections missing).
            _produces = ctx.get("produces") or []
            file_parts = []
            _seen_paths: set[str] = set()
            for entry in _produces:
                if isinstance(entry, str) and entry:
                    rel = entry
                    abs_p = rel if os.path.isabs(rel) else os.path.join(WORKSPACE_DIR, rel)
                    if abs_p in _seen_paths:
                        continue
                    _seen_paths.add(abs_p)
                    if os.path.isfile(abs_p):
                        try:
                            with open(abs_p, "r", encoding="utf-8") as f:
                                fc = f.read()
                            fc = _unwrap_envelope(fc)
                            if len(fc) > 100:
                                file_parts.append(fc)
                                logger.info(
                                    f"[Workflow Hook] Found produces "
                                    f"'{rel}' ({len(fc)} chars)"
                                )
                        except OSError:
                            pass
            for name in output_names:
                for ext in (".json", ".md", ".txt"):
                    fpath = os.path.join(artifact_dir, f"{name}{ext}")
                    if fpath in _seen_paths:
                        continue
                    _seen_paths.add(fpath)
                    if os.path.isfile(fpath):
                        with open(fpath, "r", encoding="utf-8") as f:
                            file_content = f.read()
                        file_content = _unwrap_envelope(file_content)
                        from dogru_mu_samet import assess as cq_assess, salvage as cq_salvage
                        cq = cq_assess(file_content)
                        if cq.is_degenerate:
                            cleaned = cq_salvage(file_content)
                            if cleaned:
                                logger.info(
                                    f"[Workflow Hook] Salvaged degenerate "
                                    f"workspace file '{name}{ext}' "
                                    f"({len(file_content)} -> {len(cleaned)} chars)"
                                )
                                try:
                                    with open(fpath, "w", encoding="utf-8") as wf:
                                        wf.write(cleaned)
                                except OSError:
                                    pass
                                file_parts.append(cleaned)
                            else:
                                logger.warning(
                                    f"[Workflow Hook] Deleting unsalvageable "
                                    f"workspace file '{name}{ext}' "
                                    f"({len(file_content)} chars, {cq.summary})"
                                )
                                try:
                                    os.remove(fpath)
                                except OSError:
                                    pass
                            break
                        if len(file_content) > 100:
                            file_parts.append(file_content)
                            logger.info(
                                f"[Workflow Hook] Found artifact '{name}' "
                                f"in workspace ({len(file_content)} chars)"
                            )
                        break
            if file_parts:
                # Prefer the agent's current result over stale workspace
                # files from earlier failed attempts. Previously the code
                # picked whichever was LONGER, which let a 26k-char
                # wrapped-and-truncated file from a prior DLQ'd attempt
                # overwrite the current attempt's clean 6k JSON (observed
                # on mission 46 task 2888 2026-04-24: retry produced
                # valid MoSCoW but the artifact file was the prior
                # truncated wrapper, poisoning downstream 2889).
                # Rule: only adopt the workspace file when the current
                # result is empty OR fails to json-parse AND the file
                # DOES parse cleanly.
                best_file = max(file_parts, key=len)

                def _parses_ok(s: str) -> bool:
                    try:
                        json.loads(s)
                        return True
                    except (json.JSONDecodeError, TypeError):
                        return False

                current_ok = bool(output_value) and _parses_ok(output_value)
                if not current_ok and _parses_ok(best_file):
                    output_value = best_file
                else:
                    # Markdown schemas — JSON-parse gate is meaningless.
                    # Adopt the workspace file when it is materially longer
                    # than the result AND contains required-section markers
                    # the result is missing. Keeps the JSON-truncation guard
                    # above for JSON schemas while letting markdown
                    # validation see the on-disk artifact.
                    _schema = ctx.get("artifact_schema") or {}
                    _markdown_schema = None
                    if isinstance(_schema, dict):
                        for _v in _schema.values():
                            if isinstance(_v, dict) and _v.get("type") == "markdown":
                                _markdown_schema = _v
                                break
                    if _markdown_schema is not None:
                        _req_sections = _markdown_schema.get("required_sections") or []
                        def _has_all_sections(text: str) -> bool:
                            tl = (text or "").lower()
                            for s in _req_sections:
                                sl = s.lower()
                                if not (
                                    f"# {sl}" in tl
                                    or f"**{sl}**" in tl
                                    or f"\n{sl}\n" in tl
                                ):
                                    return False
                            return bool(_req_sections)
                        if (
                            len(best_file) > max(500, len(output_value or "") * 2)
                            and _has_all_sections(best_file)
                            and not _has_all_sections(output_value or "")
                        ):
                            logger.info(
                                "[Workflow Hook] Adopted workspace markdown artifact "
                                f"({len(best_file)} chars) over short result "
                                f"({len(output_value or '')} chars) for schema validation"
                            )
                            output_value = best_file
        except Exception as e:
            logger.debug(f"[Workflow Hook] Workspace artifact recovery failed: {e}")

    # ── Detect fake completions ──
    # Small LLMs wrap failure reports in final_answer. Detect and reject.
    if output_value and _is_disguised_failure(output_value):
        result["status"] = "failed"
        result["error"] = "Agent reported completion but output indicates failure"
        result["error_category"] = "quality"  # deterministic — retry immediately
        logger.warning(
            f"[Workflow Hook] Step '{step_id}' detected as disguised failure — "
            f"overriding to failed for retry"
        )
        return  # Don't store garbage artifacts

    # ── Final quality gate before storing ──
    # Degeneracy detection (repetition / low-entropy / size_exceeded) targets LLM
    # output. Deterministic dispatchers (shopping_pipeline_v2) emit structured
    # state artifacts — candidate lists carrying review snippets — that
    # legitimately exceed the LLM size ceiling and cannot be "degenerate". Running
    # the check on them false-rejects (mission #84: groups_state 64915 > 20000).
    _DETERMINISTIC_DISPATCHERS = {"shopping_pipeline_v2"}
    _pe_agent = (task.get("agent_type") or ctx.get("agent_type") or "")
    if output_value and _pe_agent not in _DETERMINISTIC_DISPATCHERS:
        from dogru_mu_samet import assess as cq_assess
        _artifact_schema = ctx.get("artifact_schema", {})
        _step_max = _artifact_schema.get("max_output_chars", 20_000)
        cq = cq_assess(output_value, max_size=_step_max)
        if cq.is_degenerate:
            result["status"] = "failed"
            result["error"] = f"Degenerate content rejected: {cq.summary}"
            result["error_category"] = "quality"  # deterministic — retry immediately
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' output rejected: "
                f"{cq.summary} ({len(output_value)} chars)"
            )
            # Rejection ledger (T1/M3): this branch returns BEFORE the
            # post_execute ctx persist (~1820), so an in-memory append
            # would be LOST. Persist the ledger explicitly here. Degenerate
            # output stores no artifact -> ledger-only, no durable draft
            # (spec F3). Stamped for the attempt about to run.
            try:
                from general_beckman import update_task as _update_task
                _deg_attempt = int(task.get("worker_attempts", 0)) + 1
                append_rejection(
                    ctx, _deg_attempt,
                    f"degenerate: {cq.summary}",
                    _output_hash(output_value),
                )
                ctx["_schema_error_for_attempt"] = _deg_attempt
                await _update_task(task.get("id"), context=json.dumps(ctx))
            except Exception as _e:
                logger.debug(
                    f"[Workflow Hook] degenerate ledger persist skipped: {_e}"
                )
            return

    # Multi-artifact envelope split: when LLM emits {art1: ..., art2: ...}
    # for a multi-output step, store per-key value not the whole envelope.
    parsed_envelope = None
    if len(output_names) > 1 and output_value:
        try:
            cand = json.loads(output_value)
            if isinstance(cand, dict) and all(n in cand for n in output_names):
                parsed_envelope = cand
        except (json.JSONDecodeError, TypeError):
            pass

    def _slot_value(name: str) -> str:
        if parsed_envelope is not None:
            v = parsed_envelope[name]
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return output_value

    for name in output_names:
        sv = _slot_value(name)
        await store.store(mission_id, name, sv)
        logger.info(
            f"[Workflow Hook] Post-execute: stored artifact '{name}' "
            f"for mission {mission_id} ({len(sv)} chars)"
        )

    # ── Auto-summarize large artifacts ──
    # Structural summary is stored immediately (fast, no LLM).
    # LLM upgrade is queued for the orchestrator to process between cycles.
    _SUMMARY_THRESHOLD = 3000
    _MIN_SUMMARY_LEN = 50
    if output_value:
        for name in output_names:
            sv = _slot_value(name)
            if len(sv) <= _SUMMARY_THRESHOLD:
                continue
            summary = _structural_summary(sv)
            if summary and len(summary) >= _MIN_SUMMARY_LEN:
                summary_name = f"{name}_summary"
                await store.store(mission_id, summary_name, summary)
                logger.info(
                    f"[Workflow Hook] Structural summary '{name}' -> '{summary_name}' "
                    f"({len(sv)} -> {len(summary)} chars)"
                )
            # LLM-upgrade summary is scheduled by Beckman as a post-hook
            # task after grade passes (packages/general_beckman/apply.py::
            # _apply_posthook_verdict) — no need to queue here.

    # ── Materialize the declared `produces` paths (sole writer) ──
    # One deterministic pass: pick the schema-best of {on-disk write,
    # output_value} (fence-unwrapped), stamp mission_id, write the canonical
    # path, and return that content so the schema gate below validates exactly
    # what is on disk. Replaces the old fill-missing block (+ _produces_file_is_stale)
    # and the coulson auto-persist/canonicalize blocks.
    if output_value and mission_id:
        try:
            _pre_mat = output_value
            output_value = await materialize_produces(ctx, task, result, output_value)
            # ── Single source of truth for a single-`produces` step ──
            # materialize_produces canonicalizes the on-disk file (schema-best
            # of {disk write, result}, fence-unwrapped, mission_id-stamped). For
            # a single-produces step that canonical IS the step's artifact — so
            # propagate it to EVERY place the artifact is consumed, not just the
            # disk file the engine gate validates:
            #   • result["result"]  -> route_result -> tasks.result, read by the
            #     grade post-hook gate, the LLM grader, constrained_emit /
            #     self_reflect rewriters, DLQ inspection.
            #   • the artifact store -> read by downstream steps' input_artifacts
            #     (coulson/context.py _store.retrieve); store.store above ran on
            #     the PRE-materialize value (possibly a "Wrote X.md" narration).
            # This collapses the narration-vs-canonical divergence at its source
            # instead of patching each reader. Only when materialize actually
            # changed the value (canonical != raw result); multi-produces leaves
            # output_value unchanged and mechanical siblings emit no artifact —
            # both skipped. The agent's raw narration is preserved for
            # continuation handlers via the pre-hook _agent_result_snapshot.
            _exec = (task.get("executor") or ctx.get("executor") or "")
            _atype = (task.get("agent_type") or ctx.get("agent_type") or "")
            if (
                _single_produces(ctx.get("produces"))
                and _exec != "mechanical" and _atype != "mechanical"
                and isinstance(output_value, str) and output_value.strip()
                and output_value != _pre_mat
            ):
                if isinstance(result, dict):
                    result["result"] = output_value
                for _name in output_names:
                    await store.store(mission_id, _name, output_value)
                    if len(output_value) > _SUMMARY_THRESHOLD:
                        _summ = _structural_summary(output_value)
                        if _summ and len(_summ) >= _MIN_SUMMARY_LEN:
                            await store.store(mission_id, f"{_name}_summary", _summ)
        except Exception as _e:
            logger.debug(f"[Workflow Hook] materialize_produces skipped: {_e}")

    # Legacy fallback — steps with NO declared `produces` still persist their
    # artifact as `<name>.md` at the mission root (downstream summaries /
    # consumers may read it). Steps WITH `produces` are owned by
    # materialize_produces above and skip this.
    if output_value and mission_id and not (ctx.get("produces") or []):
        try:
            import src.tools.workspace as _ws
            artifact_dir = os.path.join(_ws.WORKSPACE_DIR, f"mission_{mission_id}")
            os.makedirs(artifact_dir, exist_ok=True)
            for name in output_names:
                file_path = os.path.join(artifact_dir, f"{name}.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(output_value)
                logger.debug(f"[Workflow Hook] Wrote artifact to {file_path}")
        except Exception as e:
            logger.debug(f"[Workflow Hook] Could not write artifact to disk: {e}")

    # ── Validate artifact schema ──
    # Gate is on `artifact_schema` AND "this task is the LLM-driven worker
    # that's supposed to produce the artifact". Mechanical sibling tasks
    # (workflow_advance, git_commit, clarify, etc.) inherit artifact_schema
    # from their parent step's ctx but DON'T produce artifact content —
    # they do bookkeeping. The pre-existing gate `if artifact_schema and
    # output_value` skipped them by accident (empty output → false). My
    # earlier "empty output must fail" tightening (commit 2d6d82c) broke
    # them by removing that incidental skip; mission 46 tasks 3797 + 3804
    # DLQ'd as mechanical workflow_advance with "schema requires
    # backend_design_compilation / technical_design_document" even though
    # their LLM siblings (2923, 2924) had already produced the artifacts.
    #
    # Right gate: validate only when this task IS the artifact producer
    # (no executor, agent isn't mechanical). SP3: grader/artifact_summarizer
    # agent classes removed — post-hook grading/summarize now run as
    # continuation handlers, not as tasks in the queue.
    # For producer tasks, an empty output is still a real failure (not
    # the silent bypass that let task 2921 ghost-complete on attempt 2).
    artifact_schema = ctx.get("artifact_schema")
    _agent_type = (task.get("agent_type") or ctx.get("agent_type") or "")
    _executor = (task.get("executor") or ctx.get("executor") or "")
    _is_producer = (
        _executor != "mechanical"
        and _agent_type not in ("mechanical",)
    )
    # Skip schema validation when the agent emitted a clarify action on a
    # ``triggers_clarification`` step. Such steps produce a HUMAN
    # QUESTION, not a structured artifact — the question text lives in
    # ``result.question`` (or ``result.clarification``), not in
    # ``result.result`` which would normally hold the artifact body. The
    # ``triggers_clarification`` override below at line ~1140 then routes
    # to needs_clarification correctly. Without this skip, mission 57
    # step 0.5 (human_clarification_request) burned 5 retries because
    # the schema demanded ``array, min_items: 3`` but output_value was
    # empty (clarify text in the wrong field).
    _is_clarify_action = bool(
        ctx.get("triggers_clarification")
        and (
            (isinstance(result, dict) and result.get("status") == "needs_clarification")
            or (isinstance(result, dict) and (result.get("question") or "").strip())
            or (isinstance(result, dict) and (result.get("clarification") or "").strip())
        )
    )
    # Skip schema validation when upstream already declared failure — the
    # real cause (availability, timeout, ModelCallFailed) is already in
    # result["error"]. Overwriting with "empty output schema validation"
    # masks it and spams Telegram with misleading messages while real
    # availability errors silently roll up.
    _upstream_failed = (
        isinstance(result, dict)
        and (result.get("status") == "failed"
             or result.get("error_category") == "availability")
    )
    if artifact_schema and _is_producer and not _is_clarify_action and not _upstream_failed:
        if not output_value or not str(output_value).strip():
            is_valid = False
            error_msg = (
                "empty output — the agent returned no artifact content for "
                f"a step that requires schema {list(artifact_schema.keys())}"
            )
        else:
            # Resolve any dynamic constraints (e.g. min_items_from) against
            # upstream artifacts before validating. No-op for static schemas.
            try:
                _eff_schema = await resolve_dynamic_constraints(
                    artifact_schema,
                    task.get("mission_id") or ctx.get("mission_id"),
                )
            except Exception as _e:
                logger.debug(f"[Workflow Hook] dynamic-constraint resolution skipped: {_e}")
                _eff_schema = artifact_schema
            # Anchor any empty_ok_when_input_empty exemption to the REAL
            # upstream input artifacts (no-op when the schema carries no
            # marker — the common case, zero file IO). This is the gate that
            # DLQ'd a legitimately-empty compliance_overlay (mission 87 task
            # 524377): when the upstream fingerprint has no jurisdictions,
            # zero required documents is the correct answer.
            _gate_inputs = None
            try:
                _gate_inputs = collect_empty_exemption_inputs(
                    _eff_schema, task.get("mission_id") or ctx.get("mission_id")
                )
            except Exception as _e:
                logger.debug(f"[Workflow Hook] empty-exemption inputs skipped: {_e}")
            # A markdown produces (every produced path is *.md) carries an
            # object/array schema only as a hint for its verify_*_shape check —
            # the schema gate's prose text-fallback must not literal-match field
            # names against markdown (mission-90 567452). Conservative: only when
            # ALL produces are .md, so a mixed .md+.json step keeps full checks.
            _produces = ctx.get("produces") or []
            _produces_markdown = bool(_produces) and all(
                str(p).endswith(".md") for p in _produces
            )
            is_valid, error_msg = validate_artifact_schema(
                output_value, _eff_schema, inputs=_gate_inputs,
                produces_markdown=_produces_markdown,
            )
        if is_valid and os.environ.get("LAZY_TRUE_EVIDENCE_CHECK") == "1":
            # Lazy-true detection: agent claimed a verification flag is true
            # but the audit_log shows no actual verification command ran.
            # Mission 57 task 4458 (2026-04-30): emitted
            # ``health_check_verified: true`` with zero curl in audit_log
            # because constrained decoding had been forcing the bool. Even
            # without const, small models still flip on retry pressure.
            # Gated by env var while the token lists get validated against
            # real successful missions — false positives would stall
            # legitimate work. Flip the flag to observe behavior, then
            # decide whether to make it default-on.
            try:
                evidence_err = await _check_truthy_evidence(
                    task, output_value, artifact_schema
                )
                if evidence_err:
                    is_valid = False
                    error_msg = evidence_err
            except Exception as e:
                logger.debug(f"[Workflow Hook] Evidence check skipped: {e}")
        if not is_valid:
            attempts = task.get("worker_attempts", 0)
            logger.warning(
                f"[Workflow Hook] Artifact schema validation failed for step "
                f"'{step_id}': {error_msg}. Attempt {attempts}"
            )
            # Store the error AND the previous output in context so the
            # enriched prompt on next attempt shows what went wrong and what
            # was already produced — preventing the agent from burning
            # iterations re-reading files it already wrote.
            try:
                from general_beckman import update_task
                new_ctx = dict(ctx)
                new_ctx["_schema_error"] = error_msg
                # Canonicalize before storing — collapses any escape
                # layers Qwen / Mistral / etc. emit so the NEXT retry's
                # prompt shows clean JSON, not soup that the model will
                # re-escape into deeper compounding.
                # fallback only — artifact-backed continuation reads full draft (T3)
                new_ctx["_prev_output"] = canonicalize_for_retry(
                    output_value
                )[:6000]
                # Rejection ledger (T1): record this quality reason so the
                # next prompt shows the accumulated history of what was
                # rejected (spec C5). Stamped for the attempt about to run.
                append_rejection(
                    new_ctx, int(attempts) + 1, f"schema: {error_msg}",
                    _output_hash(output_value),
                )
                # Stamp for the NEXT attempt — _retry_or_dlq increments
                # worker_attempts before re-queuing. The reader gates on
                # match against the live worker_attempts.
                new_ctx["_schema_error_for_attempt"] = int(attempts) + 1
                await update_task(
                    task.get("id"),
                    context=json.dumps(new_ctx),
                )
            except Exception as e:
                logger.debug(f"[Workflow Hook] Could not update task context: {e}")
            # Signal failure — the unified retry/DLQ path in the orchestrator
            # decides whether to retry, delay, or give up.
            # error_category="quality": schema validation is deterministic —
            # same input + same model reproduces the failure. beckman's
            # decide_retry fires quality retries IMMEDIATELY (no backoff
            # ladder) because wall-clock waiting changes nothing; only the
            # retry-hint checklist / model swap / agent refresh help.
            # Without the explicit tag the failure defaulted to a non-quality
            # category and hit the availability backoff ladder (production
            # 2026-05-15 mission 70 #44529: schema-fail retry showed
            # eta=34s instead of firing immediately).
            result["status"] = "failed"
            result["error"] = f"Schema validation: {error_msg}"
            result["error_category"] = "quality"

    # ── Force needs_clarification for human-gate steps ──
    # Steps with triggers_clarification=true bypass LLM's clarify action.
    # Only fires ONCE — if clarification_history already has answers,
    # the human already responded and the step should complete normally.
    #
    # The agent's clarify question may live in ``result.question``,
    # ``result.clarification``, or ``result.result`` depending on which
    # parser path produced it. Pick the first non-empty source so the
    # override doesn't silently skip when the agent picked the wrong
    # field name (mission 57 task 4376 — clarify text was in
    # ``result.question`` while ``output_value`` was empty).
    _clarify_text = output_value
    if not _clarify_text and isinstance(result, dict):
        _clarify_text = (
            (result.get("question") or "").strip()
            or (result.get("clarification") or "").strip()
        )
    if (ctx.get("triggers_clarification")
            and _clarify_text
            and not ctx.get("clarification_history")):
        from dogru_mu_samet import assess as cq_assess
        _clar_cq = cq_assess(_clarify_text)
        if _clar_cq.is_degenerate:
            result["status"] = "failed"
            result["error"] = (
                f"Clarification question was degenerate ({_clar_cq.summary}), "
                f"retrying instead of sending garbled text to human"
            )
            result["error_category"] = "quality"  # deterministic — retry immediately
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' clarification rejected: "
                f"{_clar_cq.summary}"
            )
            return
        result["status"] = "needs_clarification"
        result["clarification"] = _clarify_text
        logger.info(
            f"[Workflow Hook] Step '{step_id}' triggers_clarification — "
            f"overriding result status to needs_clarification"
        )

    # ── Store clarification_answers artifact when human-gate step completes ──
    # Second run (after user answered): clarification_history exists, step completes.
    # Store user_clarification as the clarification_answers artifact so downstream
    # steps (e.g. idea_brief_compilation) can consume it.
    if (ctx.get("triggers_clarification")
            and ctx.get("clarification_history")):
        user_clarification = ctx.get("user_clarification", "")
        if user_clarification and mission_id:
            await store.store(mission_id, "clarification_answers", user_clarification)
            # Also write to disk
            try:
                from ...tools.workspace import WORKSPACE_DIR
                import os
                artifact_dir = os.path.join(WORKSPACE_DIR, f"mission_{mission_id}")
                os.makedirs(artifact_dir, exist_ok=True)
                file_path = os.path.join(artifact_dir, "clarification_answers.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(user_clarification)
            except Exception:
                pass
            logger.info(
                f"[Workflow Hook] Step '{step_id}' completed with clarification — "
                f"stored 'clarification_answers' artifact ({len(user_clarification)} chars)"
            )

    # ── Check conditional group triggers ──
    await _check_conditional_triggers(mission_id, output_names, store)

    # ── Check template expansion trigger ──
    if "implementation_backlog" in output_names:
        await _trigger_template_expansion(mission_id, output_value)

    # ── Track review status ──
    status = result.get("status", "completed")
    if status in ("needs_review", "failed"):
        action = _review_tracker.record_failure(step_id)
        if action == "escalate":
            logger.warning(
                f"[Workflow Hook] Step '{step_id}' exceeded max review "
                f"cycles — escalating to needs_clarification"
            )

    # ── Check phase completion for checkpoint/resume support ──
    workflow_phase = ctx.get("workflow_phase")
    if mission_id and workflow_phase:
        await _check_phase_completion(mission_id, workflow_phase)


async def _resolve_workflow_name_from_mission(mission_id: int) -> str:
    """Best-effort workflow name from `missions.context.workflow_name` ("" if
    unavailable). Lets the phase-completion writer seed the real name instead of
    "" on the first checkpoint write."""
    try:
        from ...infra.db import get_mission
        mission = await get_mission(mission_id)
        if not mission:
            return ""
        m_ctx = mission.get("context") or "{}"
        if isinstance(m_ctx, str):
            m_ctx = json.loads(m_ctx) if m_ctx else {}
        return str((m_ctx or {}).get("workflow_name") or "")
    except Exception:
        return ""


async def _check_phase_completion(mission_id: int, phase_id: str) -> bool:
    """Detect when all tasks in a workflow phase are done and checkpoint it.

    Returns True if the phase is complete, False otherwise.
    """
    try:
        from ...infra.db import get_tasks_for_mission, get_workflow_checkpoint, upsert_workflow_checkpoint
    except ImportError as exc:
        logger.debug(f"[Workflow Hook] Phase completion check skipped (import): {exc}")
        return False

    try:
        tasks = await get_tasks_for_mission(mission_id)
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not fetch tasks for mission {mission_id}: {exc}")
        return False

    terminal_states = {"completed", "skipped", "cancelled"}
    phase_tasks = []
    for t in tasks:
        ctx = _parse_context(t)
        if ctx.get("workflow_phase") == phase_id:
            phase_tasks.append(t)

    if not phase_tasks:
        return False

    all_done = all(t.get("status") in terminal_states for t in phase_tasks)
    if not all_done:
        return False

    # Phase complete — update checkpoint
    workflow_name = ""  # hoisted: read by _evaluate_phase_gate even if the
    # checkpoint read below throws (otherwise NameError at gate eval).
    try:
        checkpoint = await get_workflow_checkpoint(mission_id)
        completed = checkpoint["completed_phases"] if checkpoint else []
        workflow_name = checkpoint["workflow_name"] if checkpoint else ""
        # On the FIRST write there is no prior checkpoint, so workflow_name was
        # persisted as "" — leaving the table useless to the reviewer-failure
        # loader (Class C, 2026-06-21). Seed the real name from mission context.
        if not workflow_name:
            workflow_name = await _resolve_workflow_name_from_mission(mission_id)

        if phase_id not in completed:
            completed.append(phase_id)

        await upsert_workflow_checkpoint(
            mission_id=mission_id,
            workflow_name=workflow_name,
            current_phase=phase_id,
            completed_phases=completed,
        )
        logger.info(
            f"[Workflow Hook] Phase '{phase_id}' complete for mission {mission_id} "
            f"({len(phase_tasks)} tasks). Checkpoint updated."
        )
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Could not update checkpoint: {exc}")

    # Generate a summary artifact for the completed phase
    await _generate_phase_summary(mission_id, phase_id, phase_tasks)

    # ── Evaluate quality gate ──
    await _evaluate_phase_gate(mission_id, phase_id, workflow_name)

    return True


async def _evaluate_phase_gate(
    mission_id: int, phase_id: str, workflow_name: str = ""
) -> None:
    """Evaluate the quality gate for a completed phase and store the result.

    Loads the gate definition from the workflow JSON if available,
    falling back to hardcoded gates (v1/v2 compatibility).
    """
    store = get_artifact_store()
    try:
        phase_num = phase_id.replace("phase_", "")

        # Try to load gate definition from workflow JSON metadata
        gate_def = None
        if workflow_name:
            try:
                from .loader import load_workflow
                wf = load_workflow(workflow_name)
                gates = wf.metadata.get("quality_gates", {})
                gate_def = gates.get(phase_id)
            except Exception as exc:
                logger.debug(
                    f"[Workflow Hook] Could not load workflow gate "
                    f"for {phase_id}: {exc}"
                )

        passed, details = await evaluate_gate(
            mission_id, phase_id, store, gate_def=gate_def
        )

        # Store gate result as artifact
        result_text = format_gate_result(phase_id, passed, details)
        await store.store(mission_id, f"phase_{phase_num}_gate_result", result_text)

        if details:  # Only log if there was actually a gate
            if passed:
                logger.info(
                    f"[Workflow Hook] Quality gate for '{phase_id}' PASSED "
                    f"(mission {mission_id})"
                )
            else:
                logger.warning(
                    f"[Workflow Hook] Quality gate for '{phase_id}' FAILED "
                    f"(mission {mission_id}): {result_text}"
                )
    except Exception as exc:
        logger.debug(f"[Workflow Hook] Quality gate evaluation failed: {exc}")


async def _generate_phase_summary(
    mission_id: int, phase_id: str, phase_tasks: list[dict]
) -> None:
    """Build a structured summary from a completed phase's output artifacts.

    The summary is stored as ``phase_{N}_summary`` in the artifact store so
    that subsequent phases can receive it as context.
    """
    from .status import PHASE_NAMES

    store = get_artifact_store()

    # Collect output artifact names from all phase tasks
    output_names: list[str] = []
    for t in phase_tasks:
        ctx = _parse_context(t)
        output_names.extend(ctx.get("output_artifacts", []))

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in output_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    # Fetch artifact contents
    artifact_contents = await store.collect(mission_id, unique_names)

    # Build summary text
    phase_name = PHASE_NAMES.get(phase_id, phase_id)
    # Extract phase number for the artifact key
    try:
        phase_num = phase_id.split("_", 1)[1]
    except IndexError:
        phase_num = phase_id

    names_with_content = [
        n for n in unique_names if artifact_contents.get(n)
    ]
    artifact_count = len(names_with_content)

    lines: list[str] = [
        f"## Phase {phase_num}: {phase_name} — Summary",
        f"**Key outputs:** {', '.join(names_with_content) if names_with_content else 'none'}",
        f"**Artifacts produced:** {artifact_count}",
        "",
    ]

    for name in names_with_content:
        content = artifact_contents[name] or ""
        excerpt = content[:200]
        if len(content) > 200:
            excerpt += "..."
        lines.append(f"### {name}\n{excerpt}")
        lines.append("")

    summary_text = "\n".join(lines).rstrip()

    from dogru_mu_samet import assess as cq_assess
    _phase_cq = cq_assess(summary_text)
    if _phase_cq.is_degenerate:
        logger.warning(
            f"[Workflow Hook] Phase summary degenerate ({_phase_cq.summary}), "
            f"using structural fallback"
        )
        summary_text = _structural_summary(summary_text)

    summary_artifact_name = f"phase_{phase_num}_summary"
    await store.store(mission_id, summary_artifact_name, summary_text)
    logger.info(
        f"[Workflow Hook] Generated summary for '{phase_id}' "
        f"({artifact_count} artifacts) -> '{summary_artifact_name}'"
    )


async def _check_conditional_triggers(
    mission_id: int, output_names: list[str], store: ArtifactStore
) -> None:
    """Evaluate conditional groups when their trigger artifact is produced."""
    try:
        from .loader import load_workflow

        # Try loading the workflow used by this mission
        workflow_name = "i2p_v3"  # fallback
        try:
            from ...infra.db import get_mission
            mission = await get_mission(mission_id)
            if mission:
                m_ctx = mission.get("context", "{}")
                if isinstance(m_ctx, str):
                    m_ctx = json.loads(m_ctx)
                workflow_name = m_ctx.get("workflow_name", "i2p_v3")
        except Exception:
            pass
        wf = load_workflow(workflow_name)
    except Exception:
        logger.debug("[Workflow Hook] Could not load workflow for conditional eval")
        return

    for group in wf.conditional_groups:
        condition_artifact = group.get("condition_artifact", "")
        if condition_artifact not in output_names:
            continue

        artifact_value = await store.retrieve(mission_id, condition_artifact)
        if artifact_value is None:
            continue

        condition_check = group.get("condition_check", "")
        result_bool = evaluate_condition(condition_check, artifact_value)
        included, excluded = resolve_group(group, artifact_value)

        logger.info(
            f"[Workflow Hook] Conditional group '{group.get('group_id')}': "
            f"condition={result_bool}, include={len(included)}, "
            f"exclude={len(excluded)} steps"
        )

        # Update task statuses in DB for excluded steps
        if excluded:
            try:
                from general_beckman import update_task_by_context_field, propagate_skips

                for step in excluded:
                    logger.warning(
                        "TASK SKIPPED (conditional group exclusion)",
                        step=step,
                        group=group.get("group_id"),
                        mission_id=mission_id,
                    )
                    await update_task_by_context_field(
                        mission_id=mission_id,
                        field="workflow_step_id",
                        value=step,
                        status="skipped",
                        error=f"excluded by conditional group '{group.get('group_id')}'",
                    )
                # Cascade skips to downstream dependents
                skipped_count = await propagate_skips(mission_id)
                if skipped_count:
                    logger.info(
                        f"[Workflow Hook] Cascaded skip to {skipped_count} dependent tasks"
                    )
            except (ImportError, Exception) as e:
                logger.debug(
                    f"[Workflow Hook] Could not skip excluded steps: {e}"
                )


async def _trigger_template_expansion(mission_id: int, backlog_text: str) -> None:
    """Expand feature_implementation_template for each feature in backlog.

    Respects ``depends_on_features`` from the backlog: the first task of a
    dependent feature won't start until the last task of its prerequisite
    feature completes.  After all features are expanded, inserts a
    cross-feature integration test step.

    Bounded by ``MAX_FEATURES_PER_MISSION`` (config). Mission 57 ran 56
    features off a 1-entry backlog because nothing here capped or deduped
    the iteration. The schema validator at [8.0] now blocks bad backlogs
    upstream; this is defence-in-depth for the path between approval and
    expansion.
    """
    import json as _json
    from src.app.config import MAX_FEATURES_PER_MISSION

    try:
        features = _json.loads(backlog_text)
        if not isinstance(features, list):
            logger.warning(
                f"[Workflow Hook] implementation_backlog for mission #{mission_id} "
                f"is not a list (got {type(features).__name__}); skipping expansion"
            )
            return
    except (ValueError, TypeError):
        logger.warning(
            f"[Workflow Hook] Could not parse implementation_backlog as JSON "
            f"for mission #{mission_id}; skipping expansion"
        )
        return

    if not features:
        logger.warning(
            f"[Workflow Hook] implementation_backlog for mission #{mission_id} "
            f"is empty — halting feature expansion (no features to build)"
        )
        return

    # ── Defensive normalize: dedup by feature_id, keep first occurrence ──
    # Schema validator at [8.0] enforces unique_by upstream and the F-NNN
    # pattern; this layer is defence-in-depth. We dedup but do NOT reject
    # by id-pattern — legacy missions and tests use other id shapes, and
    # the expander is the wrong place to reintroduce a stricter contract.
    seen_fids: set[str] = set()
    deduped: list[dict] = []
    skipped_dup = 0
    skipped_no_id = 0
    for feature in features:
        if not isinstance(feature, dict):
            continue
        fid = feature.get("id") or feature.get("feature_id")
        if not isinstance(fid, str) or not fid.strip():
            skipped_no_id += 1
            continue
        if fid in seen_fids:
            skipped_dup += 1
            continue
        seen_fids.add(fid)
        deduped.append(feature)

    if skipped_dup or skipped_no_id:
        logger.warning(
            f"[Workflow Hook] mission #{mission_id} backlog: skipped "
            f"{skipped_dup} duplicate fid(s) and {skipped_no_id} missing-id entry/entries"
        )

    # ── Hard cap ──
    if len(deduped) > MAX_FEATURES_PER_MISSION:
        logger.warning(
            f"[Workflow Hook] mission #{mission_id} backlog has {len(deduped)} "
            f"features; capping to MAX_FEATURES_PER_MISSION="
            f"{MAX_FEATURES_PER_MISSION}. Drop count: "
            f"{len(deduped) - MAX_FEATURES_PER_MISSION}"
        )
        deduped = deduped[:MAX_FEATURES_PER_MISSION]

    if not deduped:
        logger.warning(
            f"[Workflow Hook] mission #{mission_id}: no usable features after "
            f"dedup/validation — halting expansion"
        )
        return

    features = deduped

    try:
        from .loader import load_workflow
        from .expander import (
            expand_template, expand_steps_to_tasks,
            expand_steps_with_multifile,
        )
        from general_beckman import add_task as insert_task, update_task

        # Try the workflow used by this mission, fall back to i2p_v3
        workflow_name = "i2p_v3"
        try:
            from ...infra.db import get_mission
            mission = await get_mission(mission_id)
            if mission:
                m_ctx = mission.get("context", "{}")
                if isinstance(m_ctx, str):
                    m_ctx = _json.loads(m_ctx)
                workflow_name = m_ctx.get("workflow_name", "i2p_v3")
        except Exception:
            pass

        wf = load_workflow(workflow_name)
        template = wf.get_template("feature_implementation_template")
        if not template:
            logger.warning("[Workflow Hook] feature_implementation_template not found")
            return

        # Z5 T1 — resolve the mission's target platform so the feature
        # template expands the web (feat.7-10) or Expo (feat.7m-10m)
        # frontend variant. Read from the platform_requirements artifact;
        # default to 'web' when the artifact is absent or lacks the field
        # (legacy missions / non-i2p workflows are unaffected).
        target_platform = "web"
        try:
            _store = ArtifactStore(use_db=True)
            _pr_raw = await _store.retrieve(mission_id, "platform_requirements")
            if _pr_raw:
                _pr = _json.loads(_pr_raw) if isinstance(_pr_raw, str) else _pr_raw
                if isinstance(_pr, dict):
                    _tp = _pr.get("target_platform")
                    if isinstance(_tp, str) and _tp.strip().lower() in (
                        "web", "mobile", "both"
                    ):
                        target_platform = _tp.strip().lower()
        except Exception as _exc:
            logger.debug(
                f"[Workflow Hook] could not resolve target_platform for "
                f"mission #{mission_id}: {_exc!r} — defaulting to 'web'"
            )
        logger.info(
            f"[Workflow Hook] mission #{mission_id} feature-template "
            f"target_platform={target_platform!r}"
        )

        # Track feature_id → (first_task_id, last_task_id) for cross-feature deps
        feature_task_range: dict[str, tuple[int, int]] = {}

        # Idempotency check (handoff item P): mission 46 phase 8 had
        # two waves of feat tasks (4015-4023 + 4040-onward) when
        # workflow_advance fired twice for the parent step (or when a
        # reset re-triggered the expansion). Each feature's expanded
        # tasks share a ``[8.{fid}.<step>]`` title prefix, so a single
        # SQL pre-check tells us which fids are already expanded for
        # this mission. We skip those fids on this call.
        already_expanded: set[str] = set()
        try:
            from ...infra.db import get_db
            _db = await get_db()
            _cur = await _db.execute(
                """SELECT title FROM tasks
                    WHERE mission_id = ?
                      AND title LIKE '[8.%]%'""",
                (mission_id,),
            )
            for _row in await _cur.fetchall():
                _t = _row[0] if isinstance(_row, tuple) else _row["title"]
                if not _t or not _t.startswith("[8."):
                    continue
                # Title is "[8.<fid>.<stepname>] ..." — extract fid.
                _close_bracket = _t.find("]")
                if _close_bracket <= 3:
                    continue
                _step_id = _t[1:_close_bracket]  # e.g. "8.feat1.write_code"
                _parts = _step_id.split(".", 2)
                if len(_parts) >= 2:
                    already_expanded.add(_parts[1])
            await _cur.close()
        except Exception as _exc:
            logger.debug(
                f"[Workflow Hook] feature-template idempotency check "
                f"failed: {_exc!r} — proceeding without skip"
            )

        for feature in features:
            if not isinstance(feature, dict):
                continue
            fid = feature.get("id", feature.get("feature_id", "unknown"))
            fname = feature.get("name", feature.get("feature_name", "Unnamed"))

            if fid in already_expanded:
                logger.info(
                    f"[Workflow Hook] feature '{fid}' already expanded "
                    f"for mission #{mission_id} — skipping (handoff P)"
                )
                continue

            expanded = expand_template(
                template,
                params={
                    "feature_id": fid,
                    "feature_name": fname,
                    "target_platform": target_platform,
                },
                prefix=f"8.{fid}.",
            )

            # Z3 P2 (2026-05-18 sweep): use multifile-aware expander so
            # feature template expansion picks up multifile targets +
            # founder /density dials.
            tasks = await expand_steps_with_multifile(
                expanded, mission_id=mission_id, initial_context={},
            )

            # Batch insert with rollback on failure
            inserted_ids = []
            try:
                for t in tasks:
                    t.pop("depends_on_steps", None)
                    task_id = await insert_task(**t)
                    inserted_ids.append(task_id)
            except Exception as insert_err:
                # Rollback: cancel partially inserted tasks
                for tid in inserted_ids:
                    try:
                        await update_task(tid, status="cancelled")
                    except Exception:
                        pass
                logger.error(
                    f"[Workflow Hook] Partial expansion rollback for '{fid}': {insert_err}"
                )
                continue  # Skip this feature, try next one

            if inserted_ids:
                feature_task_range[fid] = (inserted_ids[0], inserted_ids[-1])

            logger.info(
                f"[Workflow Hook] Expanded template for feature '{fid}' "
                f"({len(expanded)} steps \u2192 {len(tasks)} tasks)"
            )

        # ── Wire cross-feature dependencies ──
        # If feature B depends_on_features: ["A"], then B's first task
        # should wait until A's last task completes.
        for feature in features:
            if not isinstance(feature, dict):
                continue
            fid = feature.get("id", feature.get("feature_id", "unknown"))
            dep_features = feature.get("depends_on_features", [])
            if not dep_features or fid not in feature_task_range:
                continue

            first_task_id = feature_task_range[fid][0]
            prerequisite_task_ids = []
            for dep_fid in dep_features:
                if dep_fid in feature_task_range:
                    prerequisite_task_ids.append(feature_task_range[dep_fid][1])

            if prerequisite_task_ids:
                try:
                    dep_json = _json.dumps(prerequisite_task_ids)
                    await update_task(first_task_id, depends_on=dep_json)
                    logger.info(
                        f"[Workflow Hook] Feature '{fid}' first task #{first_task_id} "
                        f"depends on tasks {prerequisite_task_ids} (cross-feature)"
                    )
                except Exception as dep_err:
                    logger.debug(
                        f"[Workflow Hook] Could not set cross-feature deps: {dep_err}"
                    )

        # ── Insert cross-feature integration test step ──
        # Runs after ALL features are done — tests interactions between features
        if len(feature_task_range) >= 2:
            all_last_tasks = [last for _, last in feature_task_range.values()]
            feature_names = []
            for f in features:
                if isinstance(f, dict):
                    feature_names.append(
                        f.get("name", f.get("feature_name", f.get("id", "?")))
                    )
            try:
                integration_task_id = await insert_task(
                    title="[8.integration] Cross-feature integration tests",
                    description=(
                        f"Test interactions between all implemented features: "
                        f"{', '.join(feature_names)}. "
                        f"Verify: shared data flows correctly between features, "
                        f"auth/permissions work across feature boundaries, "
                        f"navigation between features works, "
                        f"no conflicts in shared resources (DB, API routes, state). "
                        f"Run the full test suite and report results."
                    ),
                    mission_id=mission_id,
                    agent_type="test_generator",
                    tier="auto",
                    priority=7,
                    depends_on=all_last_tasks,
                    context={
                        "workflow_step_id": "8.integration",
                        "workflow_phase": "phase_8",
                        "is_workflow_step": True,
                        "difficulty": 6,
                        "tools_hint": ["shell", "read_file", "write_file", "coverage",
                                       "query_codebase", "codebase_map"],
                        "output_artifacts": ["integration_test_results"],
                    },
                )
                logger.info(
                    f"[Workflow Hook] Created cross-feature integration test "
                    f"task #{integration_task_id} (depends on {len(all_last_tasks)} features)"
                )
            except Exception as integ_err:
                logger.debug(
                    f"[Workflow Hook] Could not create integration test task: {integ_err}"
                )

    except (ImportError, Exception) as e:
        logger.debug(f"[Workflow Hook] Template expansion failed: {e}")
