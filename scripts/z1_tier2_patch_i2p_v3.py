"""One-shot patcher for i2p_v3.json — Z1 Tier 2 (P3+C7+A8).

Idempotent: safe to re-run. Mutates the workflow in place to:
  1. Extend phase-4 ADR-emitting steps (4.1, 4.2, 4.4, 4.6, 4.8, 4.9,
     4.10, 4.14) to universal-shape ADR with cost-curve on stack steps.
  2. Insert 4.2a `component_library_decision` step (C7).
  3. Insert mechanical `<step_id>.verify` sibling steps that call
     ``verify_adr_shape`` (and ``verify_cost_curve_present`` on stack steps).
  4. Insert single mechanical `4.14.verify_register` step that calls
     ``verify_adr_register``.
  5. Update 4.16 reviewer ``instruction`` with the new ADR/cost/c-lib checks.
  6. Add ``skip_when: mission.legacy_pre_adr == '1'`` to all new/changed
     ADR steps + the new mechanical sibling steps.

Run: ``python scripts/z1_tier2_patch_i2p_v3.py``
"""
from __future__ import annotations

import json
from pathlib import Path

WF = Path("src/workflows/i2p/i2p_v3.json")

# ADR-emitting LLM steps — id → (decision_domain, decision_label, is_stack)
ADR_STEPS: dict[str, tuple[str, str, bool]] = {
    "4.1": ("architecture_pattern", "architecture_pattern_decision", False),
    "4.2": ("tech_stack", "tech_stack_decision", True),
    "4.4": ("database", "database_schema_decision", True),
    "4.6": ("auth", "auth_design_decision", True),
    "4.8": ("third_party_services", "third_party_selections_decision", True),
    "4.9": ("infrastructure", "infrastructure_designs_decision", True),
    "4.10": ("communication", "communication_designs_decision", True),
}

# Stack ADRs require cost-curve (A8).
STACK_STEPS = {sid for sid, (_, _, is_stack) in ADR_STEPS.items() if is_stack}

UNIVERSAL_ADR_INSTRUCTION_SUFFIX = (
    "\n\nZ1 Tier 2 (P3+C7+A8): emit your decision as a universal-shape ADR "
    "JSON file at `mission_{mission_id}/.adr/<adr_id>.json` AND append a "
    "row to `mission_{mission_id}/.adr/register.md` (one line per ADR: "
    "`- ADR-YYYY-MM-DD-NNN — <title> — status:<status>`). The ADR JSON "
    "MUST contain ALL of these top-level fields:\n"
    "- `_schema_version`: \"1\"\n"
    "- `adr_id`: e.g. `ADR-{date}-{seq3}` (date = today YYYY-MM-DD, seq3 = "
    "zero-padded 3-digit sequence per mission)\n"
    "- `title`: one-line decision name\n"
    "- `status`: one of `proposed | accepted | deprecated | superseded`\n"
    "- `context`: problem/situation paragraph\n"
    "- `decision`: chosen option, prose\n"
    "- `consequences`: trade-offs incurred (positive + negative)\n"
    "- `options_considered`: list of >=2 options. EACH option has `id` "
    "(e.g. OPT-A), `name`, `rationale_for`, `rationale_against`, "
    "`tech_maturity_score` (0-10 heuristic), `novelty_benefit` (str), "
    "`reversal_cost` (low|medium|high)\n"
    "- `chosen_option_id`: MUST resolve to one of the option ids above\n"
    "- `falsification_signal`: when would we revisit this? measurable "
    "(e.g. \"if active SKU > 200 by week 4\")\n"
    "- `reversal_cost`: low|medium|high (overall ADR-level)\n"
    "- `supersedes_adr_id`: null or a prior `ADR-…` id\n"
    "\nThe mechanical post-hook `verify_adr_shape` will reject the ADR if "
    "any field is missing, `chosen_option_id` is orphan, `falsification_signal` "
    "is empty, or `reversal_cost` is outside the allowed enum."
)

STACK_COST_CURVE_SUFFIX = (
    "\n\nA8 — STACK ADR cost curve: because this is a stack-related decision, "
    "EVERY option in `options_considered` MUST also include "
    "`monthly_cost_curve: {at_mvp: \"$X\", at_1k_users: \"$Y\", "
    "at_100k_users: \"$Z\"}` AND the ADR top level MUST include "
    "`cost_at_target_users_usd` (number — picked from the curve based on z0's "
    "`expected_user_scale`) and `cost_mitigation_plan` (string or null when "
    "the curve fits z0's `cost_ceiling_monthly_usd`). The mechanical post-hook "
    "`verify_cost_curve_present` rejects missing curves; the reviewer at 4.16 "
    "rejects when `cost_at_target_users_usd > cost_ceiling_monthly_usd` AND "
    "`cost_mitigation_plan` is null."
)

REVIEWER_416_SUFFIX = (
    "\n\nZ1 Tier 2 (P3+C7+A8) — ADR-aware checks:\n"
    "- For each ADR JSON under `mission_{mission_id}/.adr/`: the universal "
    "shape (adr_id, title, status, context, decision, consequences, "
    "options_considered ≥2, chosen_option_id, falsification_signal, "
    "reversal_cost, supersedes_adr_id, _schema_version) is fully populated.\n"
    "- `chosen_option_id` resolves into `options_considered` (no orphan picks).\n"
    "- `falsification_signal` is non-empty AND measurable (a concrete "
    "threshold or signal — not just \"if it doesn't work\").\n"
    "- `reversal_cost` is `low | medium | high`.\n"
    "- Each ADR cites which charter `solution.id` it serves (charter "
    "`Solutions We Own`).\n"
    "- For STACK ADRs (4.2, 4.4, 4.6, 4.8, 4.9, 4.10): every option has a "
    "populated `monthly_cost_curve`; the ADR has `cost_at_target_users_usd` "
    "(number); reject if `cost_at_target_users_usd > "
    "z0.cost_ceiling_monthly_usd` AND `cost_mitigation_plan` is null.\n"
    "- The component-library ADR from step 4.2a is present and references "
    "the tech-stack ADR from 4.2 via `supersedes_adr_id` or an explicit "
    "`stack_adr_ref` field.\n"
    "- Reject the architecture_review_result with `status: fail` if any of "
    "the above checks miss; list each failure under `issues[]`."
)


def _verify_step_payload(
    step_id: str, decision_domain: str, label: str, is_stack: bool
) -> dict:
    """Build the mechanical verify-shape sibling step JSON."""
    payload = {
        "action": "verify_adr_shape",
        "adr_paths": [f"mission_{{mission_id}}/.adr/{label}.json"],
        "expected_schema_version": "1",
    }
    return {
        "id": f"{step_id}.verify",
        "phase": "phase_4",
        "name": f"{label}_shape_check",
        "agent": "mechanical",
        "executor": "mechanical",
        "difficulty": "easy",
        "tools_hint": [],
        "depends_on": [step_id],
        "skip_when": "mission.legacy_pre_adr == '1'",
        "input_artifacts": [label],
        "output_artifacts": [f"{label}_shape_result"],
        "payload": payload,
        "instruction": (
            f"Mechanical post-check for step {step_id} ({decision_domain}): "
            "asserts the universal-shape ADR fields (P3) are present + "
            "well-formed. Failure forces the LLM step to regenerate."
        ),
        "done_when": "verify_adr_shape returns ok=true.",
        "context": {},
    }


def _cost_curve_step_payload(step_id: str, label: str) -> dict:
    """Build the mechanical cost-curve verify sibling for stack steps."""
    return {
        "id": f"{step_id}.verify_cost_curve",
        "phase": "phase_4",
        "name": f"{label}_cost_curve_check",
        "agent": "mechanical",
        "executor": "mechanical",
        "difficulty": "easy",
        "tools_hint": [],
        "depends_on": [f"{step_id}.verify"],
        "skip_when": "mission.legacy_pre_adr == '1'",
        "input_artifacts": [label],
        "output_artifacts": [f"{label}_cost_curve_result"],
        "payload": {
            "action": "verify_cost_curve_present",
            "adr_paths": [f"mission_{{mission_id}}/.adr/{label}.json"],
        },
        "instruction": (
            f"Mechanical post-check (A8) for stack step {step_id}: every "
            "option in the ADR has a populated monthly_cost_curve and the "
            "ADR carries cost_at_target_users_usd + cost_mitigation_plan."
        ),
        "done_when": "verify_cost_curve_present returns ok=true.",
        "context": {},
    }


def _component_library_step() -> dict:
    """C7 — component-library ADR analyst step (4.2a)."""
    return {
        "id": "4.2a",
        "phase": "phase_4",
        "name": "component_library_decision",
        "agent": "analyst",
        "difficulty": "medium",
        "tools_hint": ["smart_search", "web_search", "read_file", "write_file"],
        "depends_on": ["4.2"],
        "skip_when": "mission.legacy_pre_adr == '1'",
        "may_need_clarification": False,
        "input_artifacts": ["product_charter", "tech_stack_decision"],
        "output_artifacts": ["component_library_decision"],
        "produces": [
            "mission_{mission_id}/.adr/component_library_decision.json"
        ],
        "instruction": (
            "Z1 Tier 2 (C7) — pick the component library for the frontend "
            "stack chosen in 4.2. Read the product_charter (Solutions We "
            "Own + Brand Keywords) and the tech_stack_decision ADR. "
            "Evaluate at least 3 candidates appropriate to the stack "
            "(e.g. for React: shadcn/ui, MUI, Mantine, Chakra, Ant Design, "
            "or custom). For each option include rationale_for / "
            "rationale_against / tech_maturity_score / novelty_benefit / "
            "reversal_cost / monthly_cost_curve (most are free OSS — "
            "express commercial-license gotchas in the cost field).\n\n"
            "Emit as a universal-shape ADR JSON at "
            "`mission_{mission_id}/.adr/component_library_decision.json` "
            "AND append the row to `mission_{mission_id}/.adr/register.md`. "
            "The ADR MUST include a `stack_adr_ref` field pointing at the "
            "tech_stack_decision ADR's `adr_id` so the reviewer can confirm "
            "the linkage. All universal-shape rules from steps 4.1/4.2 "
            "apply (see verify_adr_shape mechanical check). Phase 5 design "
            "steps will read this ADR by id."
            + UNIVERSAL_ADR_INSTRUCTION_SUFFIX
        ),
        "done_when": (
            "component_library_decision.json exists with universal ADR "
            "shape + stack_adr_ref + cost_curve per option."
        ),
        "artifact_schema": {
            "component_library_decision": {
                "type": "object",
                "required_fields": [
                    "adr_id",
                    "title",
                    "status",
                    "context",
                    "decision",
                    "consequences",
                    "options_considered",
                    "chosen_option_id",
                    "falsification_signal",
                    "reversal_cost",
                    "supersedes_adr_id",
                    "stack_adr_ref",
                ],
                "_schema_version": "1",
            }
        },
        "context": {"estimated_output_tokens": 4000},
    }


def patch():
    wf = json.loads(WF.read_text(encoding="utf-8"))
    steps = wf["steps"]
    # Build id → index map.
    id_to_idx = {s.get("id"): i for i, s in enumerate(steps)}

    # 1. Mutate ADR-emitting LLM steps.
    for step_id, (domain, label, is_stack) in ADR_STEPS.items():
        idx = id_to_idx.get(step_id)
        if idx is None:
            print(f"WARN: step {step_id} not found, skipping")
            continue
        step = steps[idx]

        # Append ADR-shape suffix to instruction (idempotent guard).
        instr = step.get("instruction", "")
        if "Z1 Tier 2 (P3+C7+A8)" not in instr:
            step["instruction"] = instr + UNIVERSAL_ADR_INSTRUCTION_SUFFIX
        if is_stack and "A8 — STACK ADR cost curve" not in step["instruction"]:
            step["instruction"] += STACK_COST_CURVE_SUFFIX

        # Add label to output_artifacts (alongside whatever's there).
        outs = step.setdefault("output_artifacts", [])
        if label not in outs:
            outs.append(label)

        # Add produces path.
        produces = step.setdefault("produces", [])
        adr_path = f"mission_{{mission_id}}/.adr/{label}.json"
        if adr_path not in produces:
            produces.append(adr_path)
        register_path = "mission_{mission_id}/.adr/register.md"
        if register_path not in produces:
            produces.append(register_path)

        # Extend artifact_schema with the ADR shape under `label`.
        sch = step.setdefault("artifact_schema", {})
        adr_required = [
            "adr_id",
            "title",
            "status",
            "context",
            "decision",
            "consequences",
            "options_considered",
            "chosen_option_id",
            "falsification_signal",
            "reversal_cost",
            "supersedes_adr_id",
        ]
        sch[label] = {
            "type": "object",
            "required_fields": adr_required,
            "_schema_version": "1",
        }

        # Add skip_when (no-op if already present).
        if step.get("skip_when") is None:
            step["skip_when"] = "mission.legacy_pre_adr == '1'"

    # 2. Mutate 4.14 (already ADR-shaped) — extend its fields.
    idx14 = id_to_idx.get("4.14")
    if idx14 is not None:
        s14 = steps[idx14]
        if "Z1 Tier 2 (P3+C7+A8)" not in s14.get("instruction", ""):
            s14["instruction"] = (
                "Write Architecture Decision Records (ADRs) for every major "
                "decision: architecture pattern, frontend/backend framework, "
                "database, hosting, auth approach, each vendor, and the "
                "component library. Emit the universal-shape ADR fields "
                "(see suffix below). 4.14 also produces an `adr_register` "
                "summarizing every ADR collected from upstream steps.\n\n"
                "Read each ADR JSON file under `mission_{mission_id}/.adr/` "
                "produced by steps 4.1/4.2/4.2a/4.4/4.6/4.8/4.9/4.10. The "
                "register-only artifact `adr_register` is a list of "
                "{adr_id, title, status, decision_domain, summary, links_to, "
                "supersedes} per ADR plus a `completeness_check` block "
                "naming required_domains and any missing ones.\n"
                + UNIVERSAL_ADR_INSTRUCTION_SUFFIX
            )
        # Bump artifact_schema for adrs item_fields to universal shape.
        sch14 = s14.setdefault("artifact_schema", {})
        sch14["adrs"] = {
            "type": "array",
            "min_items": 3,
            "item_fields": [
                "adr_id",
                "title",
                "status",
                "context",
                "decision",
                "consequences",
                "options_considered",
                "chosen_option_id",
                "falsification_signal",
                "reversal_cost",
                "supersedes_adr_id",
            ],
            "_schema_version": "1",
        }
        # Add register output if not present.
        outs14 = s14.setdefault("output_artifacts", [])
        if "adr_register" not in outs14:
            outs14.append("adr_register")
        produces14 = s14.setdefault("produces", [])
        register_path = "mission_{mission_id}/.adr/register.md"
        if register_path not in produces14:
            produces14.append(register_path)
        if s14.get("skip_when") is None:
            s14["skip_when"] = "mission.legacy_pre_adr == '1'"

    # 3. Update 4.16 reviewer instruction.
    idx16 = id_to_idx.get("4.16")
    if idx16 is not None:
        s16 = steps[idx16]
        if "Z1 Tier 2 (P3+C7+A8)" not in s16.get("instruction", ""):
            s16["instruction"] = s16.get("instruction", "") + REVIEWER_416_SUFFIX

    # 4. Inject 4.2a + verify steps. Idempotent: skip if id already exists.
    new_step_ids: list[str] = []

    def _has_id(step_id: str) -> bool:
        return any(s.get("id") == step_id for s in steps)

    if not _has_id("4.2a"):
        new_step_ids.append("4.2a")
        steps.append(_component_library_step())

    # Verify-shape sibling for each ADR-emitting step (including 4.2a).
    verify_targets = list(ADR_STEPS.items())
    # 4.2a uses its own label.
    verify_targets.append(
        ("4.2a", ("component_library", "component_library_decision", False))
    )
    # 4.14 verify-register handled separately.
    for step_id, (domain, label, is_stack) in verify_targets:
        vid = f"{step_id}.verify"
        if not _has_id(vid):
            new_step_ids.append(vid)
            steps.append(_verify_step_payload(step_id, domain, label, is_stack))
        if is_stack:
            cid = f"{step_id}.verify_cost_curve"
            if not _has_id(cid):
                new_step_ids.append(cid)
                steps.append(_cost_curve_step_payload(step_id, label))

    # 4.14 register verify.
    register_vid = "4.14.verify_register"
    if not _has_id(register_vid):
        new_step_ids.append(register_vid)
        steps.append({
            "id": register_vid,
            "phase": "phase_4",
            "name": "adr_register_consistency_check",
            "agent": "mechanical",
            "executor": "mechanical",
            "difficulty": "easy",
            "tools_hint": [],
            "depends_on": ["4.14"],
            "skip_when": "mission.legacy_pre_adr == '1'",
            "input_artifacts": ["adrs", "adr_register"],
            "output_artifacts": ["adr_register_consistency_result"],
            "payload": {
                "action": "verify_adr_register",
                "register_path": "mission_{mission_id}/.adr/register.md",
                "adr_dir": "mission_{mission_id}/.adr",
            },
            "instruction": (
                "Mechanical post-check (P3): every ADR id in register.md "
                "resolves to a JSON file under mission_{mission_id}/.adr/, "
                "and there are no orphan ADR JSONs. Failure forces 4.14 to "
                "regenerate the register."
            ),
            "done_when": "verify_adr_register returns ok=true.",
            "context": {},
        })

    WF.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Patched {WF}. New step ids: {new_step_ids}")
    print(
        f"Mutated ADR steps: {sorted(ADR_STEPS.keys())} + 4.14 + 4.16 reviewer"
    )


if __name__ == "__main__":
    patch()
