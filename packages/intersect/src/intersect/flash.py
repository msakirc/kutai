"""flash(task) — the intersect entry point.

Invoked once per task by the orchestrator pump, before dispatch. Flow
(spec "The intersect" section):

  1. candidates = yalayut.query(task_ctx)
  2. score each: confidence = score × source_trust × owner_trust
                              × hint_bonus
  3. budget caps: api ≤3/step, mcp tools ≤3/server ≤6/step
  4. exposure_class per candidate from (tier × kind × confidence)
  5. static-bind args for preempt + parametric inject
  6. preempt  → route task to mechanical lane (runner=mechanical,
                payload.action=yalayut_recipe)
     others   → attach task["skills"] = list[dict] envelope
  7. emit yalayut_usage telemetry

Errors anywhere → graceful degrade: task["skills"] = [], return task.
Never imports LLMDispatcher (Phase 2 has no LLM-bind).
"""
from __future__ import annotations

# Phase gate: preempt routes a yalayut shell_recipe to the mechanical lane via
# the yalayut_recipe mr_roboto executor. Enabled in Phase 3 — the executor now
# exists (packages/mr_roboto/src/mr_roboto/executors/yalayut_recipe.py) and
# yalayut.run_recipe is wired end-to-end. Kept as a flag so the preempt path
# can be disabled without reverting code if a regression surfaces; when False,
# preempt-classified artifacts are downgraded to inject instead.
PHASE2_PREEMPT_ENABLED: bool = True

import json

from src.infra.logging_config import get_logger

from intersect import binding, budget, exposure, scoring, telemetry

logger = get_logger("intersect.flash")


def _parse_context(task: dict) -> dict:
    """Best-effort parse of task['context'] into a dict."""
    ctx = task.get("context")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    return ctx if isinstance(ctx, dict) else {}


def _build_task_ctx(task: dict, ctx: dict) -> dict:
    """Assemble the binding context — the dict bind_from paths walk.

    Exposes ``task`` at both the real seed convention (``task.payload.*``)
    and the legacy convention (``task.parent_mission.payload.*``) so all
    manifests resolve regardless of which convention their bind_from uses.

    Seed convention (cc-pypackage, cc-django, cc-data-science):
        bind_from: [task.payload.project_name, task.title]
    Legacy convention (kept for backward compat):
        bind_from: [task.parent_mission.payload.project_name]

    The mission/task payload dict is expected at ``ctx["payload"]``, which
    the expander now injects for every workflow step (Cause 2 fix in
    src/workflows/engine/expander.py).
    """
    payload = ctx.get("payload") or {}
    return {
        "task": {
            "id": task.get("id"),
            "title": task.get("title", ""),
            "description": task.get("description", ""),
            "mission_id": task.get("mission_id"),
            # Real seed convention: task.payload.*
            "payload": payload,
            # Legacy convention: task.parent_mission.payload.*
            "parent_mission": {"payload": payload},
            "context": ctx,
        },
    }


async def _trust(table: str, id_col: str, ident: str | None) -> float:
    """Look up a trust score from yalayut_sources / yalayut_owners.

    Defaults to 1.0 when the row is absent — a missing trust row must
    not silently zero out every confidence (that would suppress all
    matches). Phase 1 seeds trusted sources/owners; an unseeded ident
    is treated as neutral here and capped by the tier classifier.
    """
    if not ident:
        return 1.0
    try:
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            f"SELECT trust_score FROM {table} WHERE {id_col} = ?", (ident,),
        )
        row = await cur.fetchone()
        await cur.close()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as exc:
        logger.debug("trust lookup failed (%s): %s", table, exc)
    return 1.0


def _slot_key(artifact) -> str:
    """Conflict-resolution slot key — same kind competes for one slot.

    agent_config skills compete with each other; prompt_skills do not
    (multiple prose hints stack fine). Recipe kinds compete per kind.
    """
    kind = getattr(artifact, "kind", None)
    if kind in ("agent_config",):
        return f"slot:{kind}"
    return f"id:{getattr(artifact, 'artifact_id', None)}"  # unique → no collision


async def _fire_miss_signal(task: dict, ctx: dict) -> None:
    """Record a proactive demand miss when the catalog returns nothing.

    ``planning_miss`` when the step declared a ``recipe_hint`` (the planner
    expected catalog help and got none); ``step_entry_miss`` otherwise.
    Best-effort — a signal failure must never disturb dispatch.
    """
    try:
        import yalayut
        title = (task.get("title") or "").strip()
        if not title:
            return
        sig_type = "planning_miss" if ctx.get("recipe_hint") else "step_entry_miss"
        keywords = [w for w in (title + " "
                    + (task.get("description") or "")).split() if len(w) > 2]
        await yalayut.record_demand_signal(
            source_step_pattern=f"{sig_type}:{title[:40]}",
            intent_keywords=keywords[:12],
            signal_type=sig_type,
            confidence=0.3,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("demand miss signal skipped: %s", exc)


async def flash(task: dict) -> dict:
    """Match skills, attach the task['skills'] envelope, route preempt.

    Always returns the task with a ``skills`` key (possibly empty).
    """
    task.setdefault("skills", [])
    try:
        ctx = _parse_context(task)

        # recipe_lookup gate — design/architecture/debug steps opt out.
        if ctx.get("recipe_lookup") is False:
            return task

        import yalayut
        task_ctx = _build_task_ctx(task, ctx)
        candidates = await yalayut.query(task)
        if not candidates:
            await _fire_miss_signal(task, ctx)
            return task

        recipe_hint = ctx.get("recipe_hint")

        # ── score + classify each candidate ──
        scored: list[tuple[object, float]] = []
        for art in candidates:
            # env-gated artifacts (missing auth) are skipped silently.
            if getattr(art, "env_status", "ready") != "ready":
                continue
            src_trust = await _trust(
                "yalayut_sources", "source_id", getattr(art, "source", None))
            own_trust = await _trust(
                "yalayut_owners", "owner_id", getattr(art, "owner", None))
            hint_bonus = scoring.compute_hint_bonus(art, recipe_hint)
            conf = scoring.score_artifact(
                art, source_trust=src_trust, owner_trust=own_trust,
                hint_bonus=hint_bonus,
            )
            scored.append((art, conf))

        # ── conflict resolution — keep highest-score per slot ──
        best_per_slot: dict[str, tuple[object, float]] = {}
        conflict_losers: list[dict] = []
        for art, conf in sorted(scored, key=lambda p: p[1], reverse=True):
            key = _slot_key(art)
            if key in best_per_slot:
                conflict_losers.append({
                    "artifact_id": getattr(art, "artifact_id", None),
                    "exposure_class": "inject",
                })
                continue
            best_per_slot[key] = (art, conf)

        # ── build applications ──
        applications: list[dict] = []
        preempt_app: dict | None = None
        for art, conf in best_per_slot.values():
            klass = exposure.classify(art, confidence=conf)
            if klass == "quarantine":
                continue

            bound, complete = binding.static_bind(art, task_ctx)
            # parametric + incomplete → consult bind cache.
            if not complete and getattr(art, "inputs_schema", None):
                cached = await binding.lookup_bind_cache(art, task_ctx)
                if cached:
                    bound, complete = cached, True
            # newly-completed static bind → seed the cache.
            elif complete and bound and getattr(art, "inputs_schema", None):
                await binding.write_bind_cache(art, task_ctx, bound)

            if klass == "preempt":
                # preempt with unbound required fields downgrades to inject.
                if not complete:
                    klass = "inject"
                elif not PHASE2_PREEMPT_ENABLED:
                    # Phase gate: yalayut_recipe executor is Phase 3 scope.
                    # Downgrade to inject so the artifact's body still surfaces
                    # in the agent's context rather than routing to a mechanical
                    # lane that cannot handle it (mr_roboto has no yalayut_recipe
                    # action yet).  Flip PHASE2_PREEMPT_ENABLED in Phase 3.
                    klass = "inject"
                else:
                    preempt_app = {
                        "artifact_id": getattr(art, "artifact_id", None),
                        "exposure_class": "preempt",
                        "bind_args": bound,
                    }
                    # First preempt wins — recipe owns the whole task.
                    break

            render = exposure.render_variant(
                art, bound_args=bound if complete else None)
            applications.append({
                "artifact_id": getattr(art, "artifact_id", None),
                "name": getattr(art, "name", ""),
                "artifact_type": getattr(art, "artifact_type", "skill"),
                "exposure_class": klass,
                "applies_to": "execution",          # Phase 2: execution only
                "render": render,
                "mcp_server": getattr(art, "name", None)
                if getattr(art, "artifact_type", "") == "mcp" else None,
                "payload": {
                    "body": getattr(art, "body_excerpt", ""),
                    "kind": getattr(art, "kind", None),
                    "bound_args": bound if complete else None,
                },
                "confidence": conf,
                "bind_args": bound if complete else None,
            })

        # ── preempt path — route to mechanical lane, no envelope ──
        if preempt_app is not None:
            task["runner"] = "mechanical"
            task["payload"] = {
                "action": "yalayut_recipe",
                "recipe_id": preempt_app["artifact_id"],
                "args": preempt_app["bind_args"],
            }
            task["skills"] = []
            await telemetry.record_usage(
                task_id=str(task.get("id", "")),
                exposed=[preempt_app],
                conflict_losers=conflict_losers,
            )
            return task

        # ── budget caps ──
        kept, dropped = budget.apply_caps(applications)

        # ── attach envelope (strip internal-only keys) ──
        envelope: list[dict] = []
        for app in kept:
            envelope.append({
                "artifact_id": app["artifact_id"],
                "name": app["name"],
                "exposure_class": app["exposure_class"],
                "applies_to": app["applies_to"],
                "render": app["render"],
                "payload": app["payload"],
                "confidence": app["confidence"],
            })
        task["skills"] = envelope

        await telemetry.record_usage(
            task_id=str(task.get("id", "")),
            exposed=[
                {"artifact_id": a["artifact_id"],
                 "exposure_class": a["exposure_class"],
                 "bind_args": a.get("bind_args")}
                for a in kept
            ],
            conflict_losers=conflict_losers + [
                {"artifact_id": d["artifact_id"],
                 "exposure_class": d["exposure_class"]}
                for d in dropped
            ],
        )
        return task

    except Exception as exc:
        logger.warning("intersect.flash degraded for task %s: %s",
                        task.get("id"), exc)
        task["skills"] = []
        return task
