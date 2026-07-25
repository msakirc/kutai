"""Scaffold-then-fill: mechanical frontmatter from the inventory, model body.

The deepest root for m90 5.20a/5.20b drift: the screen's mechanical fields
(screen_id / route / mission_id) are DECLARED in screen_inventory.md — they are
not a model choice. So the engine materializes one file per inventory-chunk
screen with authoritative frontmatter, grafting in the model's authored BODY
where it produced one. Invented screens (slug not in the chunk) are dropped.
This makes drift + routeless/renamed frontmatter structurally impossible.
"""
from __future__ import annotations

from mr_roboto.scaffold_screen_plans import (
    build_screen_plan_files,
    _slugify,
)
from mr_roboto.verify_screen_plan_shape import verify_screen_plan_shape
from mr_roboto.verify_screen_plans_match_inventory import (
    verify_screen_plans_match_inventory,
)


_INVENTORY = (
    "---\n"
    "total_screens: 8\n"
    "chunk_size: 4\n"
    "chunks:\n"
    "  - - Landing Page (`/`)\n"
    "    - Sign Up (`/signup`)\n"
    "    - Login (`/login`)\n"
    "    - Forgot Password (`/forgot-password`)\n"
    "  - - Onboarding (`/onboarding`)\n"
    "    - Dashboard (`/dashboard`)\n"
    "    - Habit List (`/habits`)\n"
    "    - Habit Detail (`/habits/:id`)\n"
    "mission_id: 90\n"
    "---\n"
)


def test_slugify():
    assert _slugify("Landing Page") == "landing-page"
    assert _slugify("Habit Detail") == "habit-detail"
    assert _slugify("Progress/Stats") == "progress-stats"


def test_builds_one_file_per_chunk_screen_with_authoritative_frontmatter():
    out = build_screen_plan_files(
        inventory_text=_INVENTORY, chunk_index=0, mission_id="90",
        surface="web", model_files=[],
    )
    targets = out["targets"]
    assert len(targets) == 4
    routes = {t["route"] for t in targets}
    assert routes == {"/", "/signup", "/login", "/forgot-password"}
    landing = next(t for t in targets if t["route"] == "/")
    assert landing["path"] == "mission_90/.screens/landing-page/screen_plan.md"
    fm = landing["content"].split("---", 2)[1]
    assert "mission_id: 90" in fm
    assert "route: \"/\"" in fm
    assert "screen_id: landing-page" in fm


def test_scaffold_only_files_pass_shape_gate():
    out = build_screen_plan_files(
        inventory_text=_INVENTORY, chunk_index=0, mission_id="90",
        surface="web", model_files=[],
    )
    for t in out["targets"]:
        res = verify_screen_plan_shape(plan_text=t["content"])
        assert res["ok"] is True, (t["path"], res.get("problems"))


def test_duplicate_slug_for_valid_route_is_dropped():
    """m90 open-item #4: a leftover dir at a NON-canonical slug for a VALID
    inventory route (`habits` vs canonical `habit-list`; `errands` vs
    `errands-list`) survived prior runs. The invented check required slug AND
    route to be foreign, so a duplicate slug for a valid route slipped through,
    persisted across retries, and polluted the recursive shape glob
    (verify_screen_plan_shape saw stale files missing _schema_version -> DLQ).
    The authoritative set is EXACTLY the canonical chunk slugs; any other dir is
    removable."""
    dup = {"path": "mission_90/.screens/habits/screen_plan.md",
           "text": "---\nroute: /habits\n---\n# Habits\nstale body\n"}
    out = build_screen_plan_files(
        inventory_text=_INVENTORY, chunk_index=1, mission_id="90",
        surface="web", model_files=[dup])
    slugs = {t["path"].split("/.screens/")[1].split("/")[0] for t in out["targets"]}
    assert "habit-list" in slugs                 # canonical target created
    assert "habits" not in slugs                 # duplicate slug not a target
    assert "mission_90/.screens/habits/screen_plan.md" in out["invented"]  # flagged for removal


def test_scaffold_dir_passes_correspondence_gate():
    out = build_screen_plan_files(
        inventory_text=_INVENTORY, chunk_index=0, mission_id="90",
        surface="web", model_files=[],
    )
    res = verify_screen_plans_match_inventory(
        plan_texts=[t["content"] for t in out["targets"]],
        inventory_text=_INVENTORY, chunk_index=0,
    )
    assert res["ok"] is True, res


def test_model_body_is_grafted_onto_authoritative_frontmatter():
    # Model wrote a plan for /login but with a WRONG route + rich body.
    model = (
        "---\nscreen_id: sign_in\nroute: /auth/login\nmission_id: 90\n"
        "surface: web\ninherits_shell: [\"Header\"]\n---\n\n"
        "# Sign In\n\nThe login screen authenticates returning users.\n\n"
        "## Form\n- email\n- password\n\n"
        "## States\n\n### Default\nform\n\n### Empty\nx\n\n"
        "### Loading\nspinner\n\n### Error\nbad creds\n"
    )
    out = build_screen_plan_files(
        inventory_text=_INVENTORY, chunk_index=0, mission_id="90",
        surface="web",
        model_files=[{"path": "mission_90/.screens/login/screen_plan.md",
                      "text": model}],
    )
    login = next(t for t in out["targets"] if t["route"] == "/login")
    # authoritative frontmatter (route corrected from the inventory)
    fm = login["content"].split("---", 2)[1]
    assert "route: \"/login\"" in fm
    assert "screen_id: login" in fm
    assert "/auth/login" not in fm
    # model's body preserved
    assert "authenticates returning users" in login["content"]
    assert "## Form" in login["content"]


def test_invented_model_file_is_dropped():
    model = (
        "---\nscreen_id: leaderboard\nroute: /social\nmission_id: 90\n"
        "surface: web\ninherits_shell: []\n---\n\n# Social\n\nx\n"
    )
    out = build_screen_plan_files(
        inventory_text=_INVENTORY, chunk_index=0, mission_id="90",
        surface="web",
        model_files=[{"path": "mission_90/.screens/social/screen_plan.md",
                      "text": model}],
    )
    assert "mission_90/.screens/social/screen_plan.md" in out["invented"]
    # and it is NOT among the target routes
    assert "/social" not in {t["route"] for t in out["targets"]}


def test_chunk1_is_cumulative_keeps_chunk0_files():
    """chunk steps write to ONE shared .screens/ dir, and the correspondence
    gate is cumulative — so materializing chunk 1 must PRODUCE chunks 0∪1 (keep
    the chunk-0 plans 5.20a authored), not drop them as invented."""
    chunk0 = [
        {"path": "mission_90/.screens/landing-page/screen_plan.md",
         "text": "---\nscreen_id: landing-page\nroute: \"/\"\nmission_id: 90\n"
                 "surface: web\ninherits_shell: []\n---\n\n# Landing\n\nHome.\n\n"
                 "## Hero\n- x\n\n## States\n\n### Default\nx\n\n### Empty\nx\n\n"
                 "### Loading\nx\n\n### Error\nx\n"},
    ]
    out = build_screen_plan_files(
        inventory_text=_INVENTORY, chunk_index=1, mission_id="90",
        surface="web", model_files=chunk0,
    )
    routes = {t["route"] for t in out["targets"]}
    assert routes == {"/", "/signup", "/login", "/forgot-password",
                      "/onboarding", "/dashboard", "/habits", "/habits/:id"}
    res = verify_screen_plans_match_inventory(
        plan_texts=[t["content"] for t in out["targets"]],
        inventory_text=_INVENTORY, chunk_index=1,
    )
    assert res["ok"] is True, res
    # the chunk-0 landing plan is NOT dropped
    assert "mission_90/.screens/landing-page/screen_plan.md" not in out["invented"]


def test_final_set_is_faithful_even_from_all_wrong_model_output():
    """The exact m90 5.20b failure: model produced routeless / invented files.
    The materialized set is still exactly the inventory chunk with valid
    frontmatter."""
    wrong = [
        {"path": "mission_90/.screens/errands/screen_plan.md",
         "text": "---\nscreen_id: errands\nmission_id: 90\nsurface: web\n"
                 "inherits_shell: []\n---\n\n# Errands\n\nx\n"},  # no route
        {"path": "mission_90/.screens/dashboard/screen_plan.md",
         "text": "---\nscreen_id: dashboard\nmission_id: 90\nsurface: web\n"
                 "inherits_shell: []\n---\n\n# Dashboard\n\nx\n"},
    ]
    out = build_screen_plan_files(
        inventory_text=_INVENTORY, chunk_index=0, mission_id="90",
        surface="web", model_files=wrong,
    )
    res = verify_screen_plans_match_inventory(
        plan_texts=[t["content"] for t in out["targets"]],
        inventory_text=_INVENTORY, chunk_index=0,
    )
    assert res["ok"] is True, res
    for t in out["targets"]:
        assert verify_screen_plan_shape(plan_text=t["content"])["ok"] is True
