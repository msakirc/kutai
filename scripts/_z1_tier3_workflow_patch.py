"""One-shot patcher: gate legacy phase 5 + inject Tier 3 per-screen plan
and HTML prototype steps + extend reviewer 5.10. Idempotent — safe to
re-run; the `_z1_tier3_patched` marker on the workflow short-circuits
subsequent runs."""
from __future__ import annotations

import json
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / "src" / "workflows" / "i2p" / "i2p_v3.json"
LEGACY_GATE = "mission.legacy_pre_per_screen_plans == '0'"
NEW_GATE = "mission.legacy_pre_per_screen_plans == '1'"


def main() -> int:
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    if wf.get("_z1_tier3_patched"):
        print("already patched — skipping")
        return 0

    # 1) Gate every existing 5.x as legacy (skip in non-legacy missions).
    legacy_ids: list[str] = []
    for s in wf["steps"]:
        sid = s.get("id", "")
        if sid.startswith("5.") and sid not in {"5.0", "5.0a", "5.0b", "5.0c", "5.0d"}:
            existing = s.get("skip_when")
            if existing:
                s["skip_when"] = "(" + existing + ") or (" + LEGACY_GATE + ")"
            else:
                s["skip_when"] = LEGACY_GATE
            legacy_ids.append(sid)
    print("gated legacy phase 5 IDs:", legacy_ids)

    # 2) Find insertion index — right before 6.1.
    insert_idx = next(i for i, s in enumerate(wf["steps"]) if s.get("id") == "6.1")

    # 3) New step group definitions.
    plan_chunk_a_instr = (
        "Z1 Tier 3 (C3+A10+C14): generate paraflow-shape per-screen plans for the "
        "FIRST chunk of `screen_inventory.chunks` (B8 cohesion ceiling: 3-5 screens). "
        "For EVERY screen in this chunk, write a Markdown file to "
        "`mission_{mission_id}/.screens/<slug>/screen_plan.md` with this STRICT shape:\n\n"
        "1. YAML frontmatter delimited by `---` carrying:\n"
        "   - `_schema_version: \"1\"`\n"
        "   - `mission_id: <id>`\n"
        "   - `screen_id: <slug>`\n"
        "   - `route: \"/...\"`\n"
        "   - `surface: mobile|web|desktop` (from surfaces.md)\n"
        "   - `inherits_shell: [...]` (subset of components in shared_shell.md; "
        "same list across the whole chunk unless you also emit "
        "`<!-- inherits_shell_override: <reason> -->` in the body)\n"
        "2. `# <ScreenName>` H1 + one-line description paragraph (what this screen "
        "lets the user accomplish).\n"
        "3. At least one `## <Section>` H2 beyond `## States` describing the "
        "screen's content blocks (e.g. `## Search Bar`, `## Featured Content`, "
        "`## Quick Actions`, `## TabBar`). Bullet lists per section.\n"
        "4. `## States` H2 with EXACTLY four H3 sub-sections in order: "
        "`### Default`, `### Empty`, `### Loading`, `### Error`. Each describes "
        "the visual state with bullet details (skeleton hints for Loading; retry "
        "CTA for Error; hero illustration + CTA for Empty).\n\n"
        "OUTPUT FORMAT: emit ONE Markdown file per screen via `write_file` to "
        "the path above. Your final_answer payload is `per_screen_plans_chunk_a` "
        "JSON object listing the produced paths: `{\"_schema_version\": \"1\", "
        "\"chunk\": \"a\", \"plans\": [{\"screen_id\": \"home\", \"path\": "
        "\"mission_{mission_id}/.screens/home/screen_plan.md\"}, ...]}`. The "
        "mechanical post-hook `verify_screen_plan_shape` rejects each plan if "
        "the frontmatter / description / section / States contract is not met."
    )
    plan_chunk_b_instr = (
        "Z1 Tier 3 (C3+A10+C14): generate paraflow-shape per-screen plans for the "
        "SECOND chunk of `screen_inventory.chunks`. Same shape contract as 5.20a. "
        "Write each plan to `mission_{mission_id}/.screens/<slug>/screen_plan.md`. "
        "Inherit shared_shell decisions from chunk_a unless you have a "
        "screen-specific reason — emit `<!-- inherits_shell_override: <reason> -->` "
        "in the body when diverging. Final_answer is `per_screen_plans_chunk_b` "
        "JSON object listing produced paths. If `screen_inventory.chunks` has "
        "only one chunk, return `{\"_schema_version\": \"1\", \"chunk\": \"b\", "
        "\"plans\": []}`."
    )
    html_chunk_a_instr = (
        "Z1 Tier 3 (C9+A11): for every screen plan in `per_screen_plans_chunk_a`, "
        "generate a paraflow-shape mobile HTML prototype at "
        "`mission_{mission_id}/.web/<slug>.html`.\n\n"
        "STRICT contract:\n"
        "1. `<!DOCTYPE html>` at the top.\n"
        "2. Mobile viewport 390x844 — body must carry Tailwind `w-[390px] "
        "min-h-[844px]` (or inline `width:390px; min-height:844px`).\n"
        "3. Tailwind via CDN: `<script src=\"https://cdn.tailwindcss.com\"></script>`.\n"
        "4. Iconify-Lucide via CDN: "
        "`<script src=\"https://cdn.jsdelivr.net/npm/iconify-icon@2/dist/iconify-icon.min.js\"></script>` "
        "and reference icons as `<iconify-icon icon=\"lucide:home\"></iconify-icon>`.\n"
        "5. Colors: ONLY use values present in `design_tokens.json` (read at "
        "gen time and inline as hex/rgb literals). NEVER invent new color values.\n"
        "6. Images: PLACEHOLDER ONLY — every `<img>` src must use "
        "`https://placehold.co/<W>x<H>/<bg>/<fg>?text=<intent>`. Z2 `gorsel_ustasi` "
        "swaps to real images later. Each `<img>` MUST carry a descriptive `alt=\"...\"` "
        "attribute that describes what the real image should show — this becomes "
        "the prompt for image gen.\n"
        "7. Apply `shared_shell.md` components consistently — the same header / "
        "nav / tab-bar markup across every screen in the chunk.\n\n"
        "OUTPUT FORMAT: emit ONE HTML file per screen via `write_file`. Your "
        "final_answer is `html_prototypes_chunk_a` JSON object listing produced "
        "paths: `{\"_schema_version\": \"1\", \"chunk\": \"a\", \"prototypes\": "
        "[{\"screen_id\": \"home\", \"path\": \"mission_{mission_id}/.web/home.html\"}, ...]}`. "
        "Mechanical post-hook `verify_html_prototype_shape` rejects any HTML "
        "missing the contract above."
    )
    html_chunk_b_instr = (
        "Z1 Tier 3 (C9+A11): generate HTML prototypes for the SECOND chunk's "
        "screens. Same contract as 5.30a (DOCTYPE / 390x844 / Tailwind / Iconify "
        "/ design_tokens-only colors / placehold.co images with descriptive alt "
        "/ shared_shell). Match chunk_a's stylistic decisions. If "
        "`per_screen_plans_chunk_b.plans` is empty, return "
        "`{\"_schema_version\": \"1\", \"chunk\": \"b\", \"prototypes\": []}`."
    )

    new_steps = [
        {
            "id": "5.20a",
            "phase": "phase_5",
            "name": "generate_per_screen_plans_chunk_a",
            "agent": "analyst",
            "difficulty": "medium",
            "tools_hint": ["read_file", "write_file"],
            "depends_on": ["5.0a", "5.0c", "5.0d"],
            "may_need_clarification": False,
            "input_artifacts": [
                "screen_inventory", "shared_shell", "design_tokens",
                "taste_emphasis", "user_flow", "surfaces",
            ],
            "output_artifacts": ["per_screen_plans_chunk_a"],
            "context": {"estimated_output_tokens": 9000},
            "instruction": plan_chunk_a_instr,
            "done_when": (
                "per_screen_plans_chunk_a JSON lists at least 1 produced plan "
                "path AND every listed file exists on disk."
            ),
            "artifact_schema": {
                "per_screen_plans_chunk_a": {
                    "type": "object",
                    "required_fields": ["chunk", "plans"],
                    "_schema_version": "1",
                }
            },
            "produces": ["mission_{mission_id}/.screens/"],
            "skip_when": NEW_GATE,
        },
        {
            "id": "5.20a.verify_shape",
            "phase": "phase_5",
            "name": "verify_screen_plan_shape_chunk_a",
            "agent": "mechanical",
            "depends_on": ["5.20a"],
            "executor": "verify_screen_plan_shape",
            "payload": {"action": "verify_screen_plan_shape"},
            "context": {"estimated_output_tokens": 0},
            "instruction": (
                "Mechanical sibling — assert paraflow shape on every plan "
                "produced by 5.20a. Caller wires `plan_paths` from the 5.20a "
                "artifact's `plans[].path` list."
            ),
            "skip_when": NEW_GATE,
        },
        {
            "id": "5.20b",
            "phase": "phase_5",
            "name": "generate_per_screen_plans_chunk_b",
            "agent": "analyst",
            "difficulty": "medium",
            "tools_hint": ["read_file", "write_file"],
            "depends_on": ["5.20a"],
            "may_need_clarification": False,
            "input_artifacts": [
                "screen_inventory", "shared_shell", "design_tokens",
                "taste_emphasis", "user_flow", "surfaces",
                "per_screen_plans_chunk_a",
            ],
            "output_artifacts": ["per_screen_plans_chunk_b"],
            "context": {"estimated_output_tokens": 9000},
            "instruction": plan_chunk_b_instr,
            "done_when": (
                "per_screen_plans_chunk_b is a JSON object with `chunk` + "
                "`plans` keys."
            ),
            "artifact_schema": {
                "per_screen_plans_chunk_b": {
                    "type": "object",
                    "required_fields": ["chunk", "plans"],
                    "_schema_version": "1",
                }
            },
            "produces": ["mission_{mission_id}/.screens/"],
            "skip_when": NEW_GATE,
        },
        {
            "id": "5.20b.verify_shape",
            "phase": "phase_5",
            "name": "verify_screen_plan_shape_chunk_b",
            "agent": "mechanical",
            "depends_on": ["5.20b"],
            "executor": "verify_screen_plan_shape",
            "payload": {"action": "verify_screen_plan_shape"},
            "context": {"estimated_output_tokens": 0},
            "instruction": (
                "Mechanical sibling — assert paraflow shape on every plan "
                "produced by 5.20b."
            ),
            "skip_when": NEW_GATE,
        },
        {
            "id": "5.20.verify_consistency",
            "phase": "phase_5",
            "name": "verify_screen_plan_consistency",
            "agent": "mechanical",
            "depends_on": ["5.20a.verify_shape", "5.20b.verify_shape"],
            "executor": "verify_screen_consistency",
            "payload": {"action": "verify_screen_consistency"},
            "context": {"estimated_output_tokens": 0},
            "instruction": (
                "Mechanical sibling — cross-screen `inherits_shell` "
                "consistency check across both chunks (B8/C18). Caller wires "
                "`screen_plan_paths` (union of both chunks' plans) and "
                "`shared_shell_components` from shared_shell.md."
            ),
            "skip_when": NEW_GATE,
        },
        {
            "id": "5.30a",
            "phase": "phase_5",
            "name": "generate_html_prototypes_chunk_a",
            "agent": "analyst",
            "difficulty": "hard",
            "tools_hint": ["read_file", "write_file"],
            "depends_on": ["5.20.verify_consistency"],
            "may_need_clarification": False,
            "input_artifacts": [
                "per_screen_plans_chunk_a", "design_tokens", "taste_emphasis",
                "shared_shell", "surfaces",
            ],
            "output_artifacts": ["html_prototypes_chunk_a"],
            "context": {"estimated_output_tokens": 14000},
            "instruction": html_chunk_a_instr,
            "done_when": (
                "html_prototypes_chunk_a JSON lists at least 1 produced HTML "
                "path AND every listed file exists on disk."
            ),
            "artifact_schema": {
                "html_prototypes_chunk_a": {
                    "type": "object",
                    "required_fields": ["chunk", "prototypes"],
                    "_schema_version": "1",
                }
            },
            "produces": ["mission_{mission_id}/.web/"],
            "skip_when": NEW_GATE,
        },
        {
            "id": "5.30a.verify_shape",
            "phase": "phase_5",
            "name": "verify_html_prototype_shape_chunk_a",
            "agent": "mechanical",
            "depends_on": ["5.30a"],
            "executor": "verify_html_prototype_shape",
            "payload": {"action": "verify_html_prototype_shape"},
            "context": {"estimated_output_tokens": 0},
            "instruction": (
                "Mechanical sibling — assert paraflow shape on every HTML "
                "produced by 5.30a. Caller wires `html_paths` and "
                "`design_tokens` (parsed JSON) from the artifact store."
            ),
            "skip_when": NEW_GATE,
        },
        {
            "id": "5.30b",
            "phase": "phase_5",
            "name": "generate_html_prototypes_chunk_b",
            "agent": "analyst",
            "difficulty": "hard",
            "tools_hint": ["read_file", "write_file"],
            "depends_on": ["5.30a", "5.20b"],
            "may_need_clarification": False,
            "input_artifacts": [
                "per_screen_plans_chunk_b", "design_tokens", "taste_emphasis",
                "shared_shell", "surfaces", "html_prototypes_chunk_a",
            ],
            "output_artifacts": ["html_prototypes_chunk_b"],
            "context": {"estimated_output_tokens": 14000},
            "instruction": html_chunk_b_instr,
            "done_when": (
                "html_prototypes_chunk_b is a JSON object with `chunk` + "
                "`prototypes` keys."
            ),
            "artifact_schema": {
                "html_prototypes_chunk_b": {
                    "type": "object",
                    "required_fields": ["chunk", "prototypes"],
                    "_schema_version": "1",
                }
            },
            "produces": ["mission_{mission_id}/.web/"],
            "skip_when": NEW_GATE,
        },
        {
            "id": "5.30b.verify_shape",
            "phase": "phase_5",
            "name": "verify_html_prototype_shape_chunk_b",
            "agent": "mechanical",
            "depends_on": ["5.30b"],
            "executor": "verify_html_prototype_shape",
            "payload": {"action": "verify_html_prototype_shape"},
            "context": {"estimated_output_tokens": 0},
            "instruction": (
                "Mechanical sibling — assert paraflow shape on every HTML "
                "produced by 5.30b."
            ),
            "skip_when": NEW_GATE,
        },
    ]

    wf["steps"][insert_idx:insert_idx] = new_steps

    # 4) Extend reviewer 5.10 instruction with new criteria.
    extra = (
        "\n\nZ1 Tier 3 (C3+A10+C9+A11+C14+C18) — per-screen plan + HTML "
        "prototype review (applies when legacy_pre_per_screen_plans is 0):\n"
        "- REJECT (status=fail) when any screen in `screen_inventory.md` lacks "
        "a corresponding `mission_{mission_id}/.screens/<slug>/screen_plan.md` "
        "AND `mission_{mission_id}/.web/<slug>.html` on disk.\n"
        "- REJECT when a screen plan does not declare ALL FOUR `## States` H3 "
        "sub-sections (Default / Empty / Loading / Error).\n"
        "- REJECT when an HTML prototype omits `<!DOCTYPE html>`, the 390x844 "
        "viewport, the Tailwind script tag, an Iconify script tag, or any "
        "`<img>` lacks a non-empty descriptive `alt`.\n"
        "- REJECT when an HTML uses a color value not present in "
        "`design_tokens.json`.\n"
        "- REJECT when chunk's screens declare inconsistent `inherits_shell` "
        "without an explicit `<!-- inherits_shell_override: <reason> -->` "
        "comment.\n"
        "Surface the offending screen_id + path + concrete failure in the "
        "issue list."
    )
    for s in wf["steps"]:
        if s.get("id") == "5.10":
            s["instruction"] = (s.get("instruction") or "") + extra

    wf["_z1_tier3_patched"] = True
    WORKFLOW.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
    print("workflow updated; total step count:", len(wf["steps"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
