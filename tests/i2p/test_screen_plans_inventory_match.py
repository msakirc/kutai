"""Screen-plan ⇄ inventory correspondence gate (mechanical).

m90 5.20a/5.20b invented their own screen set (Dashboard/Habit Editor/Habit
Tracker/…) instead of generating plans for `screen_inventory.chunks[N]`
(Landing/Sign Up/Login/Forgot Password). The shape gate validated FORM but never
CORRESPONDENCE, so invented + missing screens passed silently. This gate keys on
ROUTE (the stable contract — models rename screen_ids but routes are fixed) and
asserts the produced plans are exactly the inventory chunk's screens.
"""
from __future__ import annotations

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
    "---\n\n## Web\n- ...\n"
)


def _plan(route: str, screen_id: str = "x") -> str:
    return (
        f"---\n_schema_version: \"1\"\nmission_id: 90\nscreen_id: {screen_id}\n"
        f"route: {route}\nsurface: web\ninherits_shell: [\"Header\"]\n---\n\n# X\n"
    )


def test_faithful_chunk0_passes():
    plans = [_plan("/"), _plan("/signup"), _plan("/login"), _plan("/forgot-password")]
    res = verify_screen_plans_match_inventory(
        plan_texts=plans, inventory_text=_INVENTORY, chunk_index=0
    )
    assert res["ok"] is True, res
    assert res["missing"] == [] and res["extra"] == []


def test_invented_screen_flagged_as_extra():
    plans = [_plan("/"), _plan("/signup"), _plan("/login"), _plan("/made-up-errands")]
    res = verify_screen_plans_match_inventory(
        plan_texts=plans, inventory_text=_INVENTORY, chunk_index=0
    )
    assert res["ok"] is False
    assert "/made-up-errands" in res["extra"]
    assert "/forgot-password" in res["missing"]


def test_screen_in_wrong_chunk_is_extra():
    """/dashboard is a real inventory screen but belongs to chunk 1 — producing
    it in chunk 0 is a misplacement, flagged as extra (not silently accepted)."""
    plans = [_plan("/"), _plan("/signup"), _plan("/login"), _plan("/dashboard")]
    res = verify_screen_plans_match_inventory(
        plan_texts=plans, inventory_text=_INVENTORY, chunk_index=0
    )
    assert res["ok"] is False
    assert "/dashboard" in res["extra"]


def test_route_keyed_not_screen_id_keyed():
    """A renamed screen_id with the CORRECT route passes — routes are the contract."""
    plans = [
        _plan("/", "home_page"),
        _plan("/signup", "registration"),
        _plan("/login", "sign_in"),
        _plan("/forgot-password", "pw_reset"),
    ]
    res = verify_screen_plans_match_inventory(
        plan_texts=plans, inventory_text=_INVENTORY, chunk_index=0
    )
    assert res["ok"] is True, res


def test_missing_screen_flagged():
    plans = [_plan("/"), _plan("/signup"), _plan("/login")]  # dropped /forgot-password
    res = verify_screen_plans_match_inventory(
        plan_texts=plans, inventory_text=_INVENTORY, chunk_index=0
    )
    assert res["ok"] is False
    assert "/forgot-password" in res["missing"]


def test_cumulative_chunk1_expects_chunks_0_and_1():
    plans = [
        _plan("/"), _plan("/signup"), _plan("/login"), _plan("/forgot-password"),
        _plan("/onboarding"), _plan("/dashboard"), _plan("/habits"), _plan("/habits/:id"),
    ]
    res = verify_screen_plans_match_inventory(
        plan_texts=plans, inventory_text=_INVENTORY, chunk_index=1, cumulative=True
    )
    assert res["ok"] is True, res


def test_cumulative_chunk1_missing_a_chunk0_screen_fails():
    plans = [  # forgot /forgot-password from chunk 0
        _plan("/"), _plan("/signup"), _plan("/login"),
        _plan("/onboarding"), _plan("/dashboard"), _plan("/habits"), _plan("/habits/:id"),
    ]
    res = verify_screen_plans_match_inventory(
        plan_texts=plans, inventory_text=_INVENTORY, chunk_index=1, cumulative=True
    )
    assert res["ok"] is False
    assert "/forgot-password" in res["missing"]


def test_route_normalization_trailing_slash_and_quotes():
    plans = [_plan('"/"'), _plan("/signup/"), _plan("/login"), _plan("/forgot-password")]
    res = verify_screen_plans_match_inventory(
        plan_texts=plans, inventory_text=_INVENTORY, chunk_index=0
    )
    assert res["ok"] is True, res


def test_unparseable_inventory_is_vacuous_safe_pass():
    res = verify_screen_plans_match_inventory(
        plan_texts=[_plan("/whatever")], inventory_text="no frontmatter here",
        chunk_index=0,
    )
    assert res["ok"] is True
    assert res["empty"] is True


def test_m90_scenario_all_wrong():
    """The exact m90 5.20a failure: chunk 0 is auth screens, model produced
    dashboard/habit-editor/habit-tracker/onboarding."""
    plans = [_plan("/dashboard"), _plan("/habits/:id/edit"),
             _plan("/habit-tracker"), _plan("/onboarding")]
    res = verify_screen_plans_match_inventory(
        plan_texts=plans, inventory_text=_INVENTORY, chunk_index=0
    )
    assert res["ok"] is False
    assert set(res["missing"]) == {"/", "/signup", "/login", "/forgot-password"}
    assert "/habit-tracker" in res["extra"]
