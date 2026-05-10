"""One-shot script: insert Z1 T3B steps (5.0b/5.0c/5.0d + verify siblings)
into i2p_v3.json and extend the 5.10 reviewer instruction.

Idempotent: re-running this will not create duplicate steps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WF = Path(__file__).resolve().parents[1] / "src" / "workflows" / "i2p" / "i2p_v3.json"


def _surfaces_step() -> dict:
    return {
        "id": "5.0b",
        "phase": "phase_5",
        "name": "surfaces_lock",
        "agent": "mechanical",
        "difficulty": "easy",
        "tools_hint": [],
        "depends_on": ["2.11b"],
        "may_need_clarification": True,
        "input_artifacts": ["prd_final_summary"],
        "output_artifacts": ["surfaces_config"],
        "instruction": (
            "Ask founder which surfaces this product targets. Reply keyboard "
            "options: 'mobile only', 'web only', 'mobile + web', "
            "'mobile + web + desktop', 'mobile + web + admin'. Persist the "
            "decision to mission_{mission_id}/.charter/surfaces.json with "
            "schema {_schema_version, mission_id, surfaces, primary_surface, "
            "founder_confirmed_at}. primary_surface defaults to first item; "
            "founder may override."
        ),
        "done_when": "surfaces_config exists with founder confirmation.",
        "produces": ["mission_{mission_id}/.charter/surfaces.json"],
        "executor": "clarify",
        "payload": {
            "action": "clarify",
            "kind": "surface_choice",
            "options": [
                "mobile only",
                "web only",
                "mobile + web",
                "mobile + web + desktop",
                "mobile + web + admin",
            ],
            "persist_to": "mission_{mission_id}/.charter/surfaces.json",
        },
        "context": {},
    }


def _surfaces_verify_step() -> dict:
    return {
        "id": "5.0b.verify",
        "phase": "phase_5",
        "name": "surfaces_verify",
        "agent": "mechanical",
        "difficulty": "easy",
        "tools_hint": [],
        "depends_on": ["5.0b"],
        "may_need_clarification": False,
        "input_artifacts": ["surfaces_config"],
        "output_artifacts": ["surfaces_verify_result"],
        "instruction": (
            "Mechanical verify of surfaces.json shape. Asserts non-empty "
            "surfaces array, valid surface tokens, primary_surface in list."
        ),
        "done_when": "surfaces_verify_result.ok == true",
        "executor": "verify_surfaces_shape",
        "payload": {
            "action": "verify_surfaces_shape",
            "path": ".charter/surfaces.json",
        },
        "context": {},
    }


def _user_flow_step() -> dict:
    return {
        "id": "5.0c",
        "phase": "phase_5",
        "name": "user_flow_lock",
        "agent": "analyst",
        "difficulty": "medium",
        "tools_hint": [],
        "depends_on": ["5.0b.verify"],
        "may_need_clarification": False,
        "input_artifacts": [
            "surfaces_config",
            "product_charter",
            "prd_final_summary",
        ],
        "output_artifacts": ["user_flow"],
        "instruction": (
            "Read surfaces.json + product_charter (especially Solutions and "
            "persona references). Emit mission_{mission_id}/.flow/user_flow.md "
            "with YAML frontmatter (_schema_version='1', mission_id, "
            "surfaces) and one ```mermaid graph TD``` block per declared "
            "surface. Each node must follow the pattern: "
            "Name[\"Screen Name<br/>/route\"]. Match the paraflow shape: "
            "per-surface H2 sections with persona name in parentheses; "
            "Mobile may split into Authentication/Onboarding + Main App."
        ),
        "done_when": "user_flow.md exists with one mermaid block per surface.",
        "produces": ["mission_{mission_id}/.flow/user_flow.md"],
        "artifact_schema": {
            "user_flow": {
                "type": "object",
                "required_fields": ["surfaces", "mermaid_per_surface"],
            }
        },
        "context": {"estimated_output_tokens": 4000},
    }


def _user_flow_verify_step() -> dict:
    return {
        "id": "5.0c.verify",
        "phase": "phase_5",
        "name": "user_flow_verify",
        "agent": "mechanical",
        "difficulty": "easy",
        "tools_hint": [],
        "depends_on": ["5.0c"],
        "may_need_clarification": False,
        "input_artifacts": ["user_flow"],
        "output_artifacts": ["user_flow_verify_result"],
        "instruction": (
            "Mechanical verify of user_flow.md shape. Asserts YAML "
            "frontmatter and at least one ```mermaid block per declared "
            "surface."
        ),
        "done_when": "user_flow_verify_result.ok == true",
        "executor": "verify_user_flow_shape",
        "payload": {
            "action": "verify_user_flow_shape",
            "path": ".flow/user_flow.md",
        },
        "context": {},
    }


def _screen_inventory_step() -> dict:
    return {
        "id": "5.0d",
        "phase": "phase_5",
        "name": "screen_inventory_and_shell",
        "agent": "analyst",
        "difficulty": "medium",
        "tools_hint": [],
        "depends_on": ["5.0c.verify"],
        "may_need_clarification": False,
        "input_artifacts": ["user_flow", "surfaces_config"],
        "output_artifacts": ["screen_inventory", "shared_shell"],
        "instruction": (
            "Read user_flow.md. Extract every unique screen by parsing "
            "mermaid node labels (Name[\"Screen Name<br/>/route\"]). "
            "Emit two artifacts:\n"
            "1. mission_{mission_id}/.flow/screen_inventory.md — YAML "
            "frontmatter with total_screens, chunk_size=4 (B8 cohesion "
            "ceiling), chunks (list of lists, each <= chunk_size). Body "
            "groups screens by surface (## Mobile / ## Web / etc.). Each "
            "screen line: '- Screen Name (`/route`)'.\n"
            "2. mission_{mission_id}/.flow/shared_shell.md — YAML "
            "frontmatter with shared_components dict + applicable_to_surfaces. "
            "Body has H2 sections for at minimum: Header, EmptyState, "
            "ErrorState, LoadingState (TabBar/Footer optional but "
            "recommended). Each section describes the invariant pattern."
        ),
        "done_when": "screen_inventory and shared_shell both exist.",
        "produces": [
            "mission_{mission_id}/.flow/screen_inventory.md",
            "mission_{mission_id}/.flow/shared_shell.md",
        ],
        "artifact_schema": {
            "screen_inventory": {
                "type": "object",
                "required_fields": ["total_screens", "chunk_size", "chunks"],
            },
            "shared_shell": {
                "type": "object",
                "required_fields": ["shared_components", "applicable_to_surfaces"],
            },
        },
        "context": {"estimated_output_tokens": 6000},
    }


def _screen_inventory_verify_step() -> dict:
    return {
        "id": "5.0d.verify",
        "phase": "phase_5",
        "name": "screen_inventory_and_shell_verify",
        "agent": "mechanical",
        "difficulty": "easy",
        "tools_hint": [],
        "depends_on": ["5.0d"],
        "may_need_clarification": False,
        "input_artifacts": ["screen_inventory", "shared_shell"],
        "output_artifacts": ["screen_inventory_verify_result"],
        "instruction": (
            "Mechanical verify of screen_inventory.md (chunks math, every "
            "screen has route) AND shared_shell.md (header + empty + error "
            "+ loading present)."
        ),
        "done_when": "both verifies pass",
        "executor": "verify_screen_inventory_shape",
        "payload": {
            "action": "verify_screen_inventory_shape",
            "path": ".flow/screen_inventory.md",
            "follow_up": {
                "action": "verify_shared_shell_shape",
                "path": ".flow/shared_shell.md",
            },
        },
        "context": {},
    }


def _shared_shell_verify_step() -> dict:
    return {
        "id": "5.0d.verify_shell",
        "phase": "phase_5",
        "name": "shared_shell_verify",
        "agent": "mechanical",
        "difficulty": "easy",
        "tools_hint": [],
        "depends_on": ["5.0d"],
        "may_need_clarification": False,
        "input_artifacts": ["shared_shell"],
        "output_artifacts": ["shared_shell_verify_result"],
        "instruction": (
            "Mechanical verify of shared_shell.md shape (frontmatter + "
            "header/empty/error/loading sections present)."
        ),
        "done_when": "shared_shell_verify_result.ok == true",
        "executor": "verify_shared_shell_shape",
        "payload": {
            "action": "verify_shared_shell_shape",
            "path": ".flow/shared_shell.md",
        },
        "context": {},
    }


REVIEW_510_EXTENSION = (
    " Additionally, verify Z1 T3B structural artifacts: "
    "(a) .charter/surfaces.json exists with primary_surface declared and "
    "in surfaces list; "
    "(b) .flow/user_flow.md has one mermaid diagram per declared surface "
    "(count match); "
    "(c) .flow/screen_inventory.md total_screens equals sum of chunks and "
    "every screen has a route; "
    "(d) .flow/shared_shell.md declares the minimum 4 shells "
    "(header / empty_state / error_state / loading_state); "
    "(e) per-screen plans referenced in inventory exist on disk."
)


def main() -> int:
    raw = WF.read_text(encoding="utf-8")
    data = json.loads(raw)
    steps = data["steps"]

    existing = {s.get("id") for s in steps}
    new_steps = [
        _surfaces_step(),
        _surfaces_verify_step(),
        _user_flow_step(),
        _user_flow_verify_step(),
        _screen_inventory_step(),
        _screen_inventory_verify_step(),
        _shared_shell_verify_step(),
    ]
    insert_at = next(i for i, s in enumerate(steps) if s.get("id") == "5.1")
    added = 0
    for ns in new_steps:
        if ns["id"] in existing:
            continue
        steps.insert(insert_at + added, ns)
        added += 1
    if added:
        print(f"inserted {added} step(s)")
    else:
        print("steps already present, skipping")

    # Extend 5.10 reviewer instruction (idempotent: marker check).
    for s in steps:
        if s.get("id") == "5.10":
            instr = s.get("instruction", "")
            if "Z1 T3B" not in instr:
                s["instruction"] = instr.rstrip() + REVIEW_510_EXTENSION
                print("extended 5.10 reviewer instruction")
            else:
                print("5.10 reviewer instruction already extended")
            break

    WF.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
