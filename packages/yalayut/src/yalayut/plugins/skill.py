"""SkillPlugin — DiscoveryPlugin + AccessPlugin for artifact_type 'skill'.

Implements both protocols: parse_manifest/vet_checks (discovery side) and
to_application/bind_args/execute (access side). Phase 1 ships the skill
plugin; api.py and mcp.py plugins are Phase 3+ (their schema tables exist
from Task 1, but no api/mcp adapter feeds them in Phase 1).
"""
from __future__ import annotations

from pathlib import Path

from yalayut.contracts import (
    Issue, IndexRow, Manifest, Result, SkillApplication, TaskContext,
)

_BODY_CAP = 50 * 1024
_HINT_CAP = 5 * 1024


class SkillPlugin:
    """Skill artifact plugin."""

    artifact_type = "skill"

    # ── DiscoveryPlugin side ────────────────────────────────────────────
    def parse_manifest(self, raw: bytes, source_meta: dict) -> Manifest:
        """Parse a fetched SKILL.md into a Manifest (delegates to synthesis)."""
        from yalayut.contracts import ArtifactRef
        from yalayut.discovery.synthesize import synthesize
        ref = ArtifactRef(
            source_id=source_meta.get("source_id", ""),
            name=source_meta.get("name", ""),
            fetch_url=source_meta.get("fetch_url", ""),
            owner=source_meta.get("owner"),
        )
        manifest, _body = synthesize(ref, raw)
        return manifest

    def vet_checks(
        self, manifest: Manifest, body_path: Path
    ) -> list[Issue]:
        """Skill-specific structural checks (complements gate-zero auto_checks).

        The 9 cross-cutting gate-zero checks live in vetting/auto_checks.py
        and run for every artifact_type. This method adds checks that need
        skill-type knowledge: body presence, body size, kind validity.
        """
        issues: list[Issue] = []
        if not body_path or not body_path.exists():
            issues.append(Issue("body_present", 3, "no body file"))
            return issues
        size = body_path.stat().st_size
        cap = _HINT_CAP if manifest.kind == "internal_hint" else _BODY_CAP
        if size > cap:
            issues.append(Issue(
                "body_size", 2, f"{size}B exceeds {cap}B cap"
            ))
        if manifest.kind not in {
            "internal_hint", "prompt_skill", "shell_recipe", "procedure",
            "agent_config",
        }:
            issues.append(Issue(
                "kind_valid", 3, f"unknown skill kind {manifest.kind!r}"
            ))
        if manifest.kind == "shell_recipe" and not manifest.invocation.get(
            "steps"
        ):
            issues.append(Issue(
                "recipe_steps", 2, "shell_recipe with no invocation.steps"
            ))
        return issues

    # ── AccessPlugin side ───────────────────────────────────────────────
    def to_application(
        self, row: IndexRow, task_ctx: TaskContext
    ) -> SkillApplication:
        """Build the structured SkillApplication for one matched skill.

        Exposure class is decided by Phase 3's intersect; here we return the
        stored ceiling so the object is coherent if read before intersect
        exists. render defaults to 'prose'; prebind is chosen by intersect
        when args bind. payload carries the body excerpt for the consumer.
        """
        exposure = row.exposure_class or "inject"
        if exposure not in {"inject", "tool", "preempt"}:
            exposure = "inject"
        return SkillApplication(
            artifact_id=row.id,
            name=row.name,
            exposure_class=exposure,
            applies_to=row.applies_to or "execution",
            render="prose",
            payload={
                "kind": row.kind,
                "body_excerpt": row.body_excerpt,
                "model_hint": row.model_hint,
            },
            confidence=0.0,
        )

    def bind_args(
        self, row: IndexRow, task_ctx: TaskContext
    ) -> dict | None:
        """Static bind for parametric recipes. prompt_skill/agent_config have
        no inputs_schema -> None. Only shell_recipe/procedure carry one, and
        the schema lives in the on-disk manifest.yaml — Phase 1 loads it from
        manifest_path when present and resolves bind_from paths against
        task_ctx. Returns the bound dict, or None if not parametric / unbound.
        """
        if row.kind not in {"shell_recipe", "procedure"}:
            return None
        if not row.manifest_path:
            return None
        import yaml
        try:
            raw = yaml.safe_load(Path(row.manifest_path).read_text())
        except (OSError, yaml.YAMLError):
            return None
        schema = (raw or {}).get("inputs_schema", {})
        if not schema:
            return None
        bound: dict = {}
        for field_name, spec in schema.items():
            value = None
            for path in spec.get("bind_from", []):
                value = _resolve_path(path, task_ctx)
                if value is not None:
                    break
            if value is None and "default" in spec:
                value = spec["default"]
            if value is None:
                return None          # incomplete static bind -> caller falls
            bound[field_name] = value  # back to prose inject (spec cost ladder)
        return bound

    def execute(
        self, row: IndexRow, task_ctx: TaskContext, inputs: dict
    ) -> Result:
        """Run a mechanizable shell_recipe. Non-mechanizable skills refuse.

        Phase 1 wires the refusal + the invocation-step shell loop. The actual
        mr_roboto preempt routing (deciding to CALL execute) is Phase 3; this
        body is the executor mr_roboto's preempt path invokes via run_recipe.
        """
        if not row.mechanizable:
            return Result(ok=False, detail="artifact not mechanizable")
        if row.kind not in {"shell_recipe", "procedure"}:
            return Result(ok=False, detail=f"kind {row.kind!r} not executable")
        if not row.manifest_path:
            return Result(ok=False, detail="no manifest_path for recipe")
        import subprocess
        import yaml
        try:
            raw = yaml.safe_load(Path(row.manifest_path).read_text())
        except (OSError, yaml.YAMLError) as e:
            return Result(ok=False, detail=f"manifest unreadable: {e}")
        steps = (raw or {}).get("invocation", {}).get("steps", [])
        if not steps:
            return Result(ok=False, detail="recipe has no steps")
        run_log: list[str] = []
        for step in steps:
            cmd = step.get("cmd", "")
            if not cmd:
                continue
            try:
                proc = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired:
                return Result(
                    ok=False, detail=f"step timed out: {cmd}",
                    data={"log": run_log},
                )
            run_log.append(f"$ {cmd}\n{proc.stdout}{proc.stderr}")
            if proc.returncode != 0:
                return Result(
                    ok=False, detail=f"step failed ({proc.returncode}): {cmd}",
                    data={"log": run_log},
                )
        return Result(
            ok=True, detail="recipe complete",
            artifacts=list(raw.get("artifacts", [])),
            data={"log": run_log},
        )


def _resolve_path(path: str, task_ctx: TaskContext) -> object | None:
    """Resolve a dotted bind_from path like 'task.title' or
    'task.parent_mission.payload.project_name' against a TaskContext."""
    parts = path.split(".")
    if parts and parts[0] == "task":
        parts = parts[1:]
    cur: object = {
        "title": task_ctx.title,
        "description": task_ctx.description,
        "agent_type": task_ctx.agent_type,
        "payload": task_ctx.payload,
        "parent_mission": {"payload": task_ctx.payload},
    }
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur if cur != "" else None
