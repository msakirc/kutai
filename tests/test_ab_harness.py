# tests/test_ab_harness.py
"""Z9 T5D/T5E — A/B experiment harness + Stripe pricing A/B.

Covers:
  - assign_variant: creates control + treatment rows, wires the PostHog
    feature-flag, honours the insufficient-N guard (<100 DAU → no split)
    and the /experiment_disable opt-out.
  - A/B result evaluation: Bayesian winner pick from the two arm metrics
    (src/growth/ab_result.py reusing verdict_stats).
  - retire_variant: winner/loser status transitions + posthog flag flip.
  - /experiment_ship, /experiment_rollback, /experiment_disable.
  - pricing typed confirmation: a bare /confirm is rejected; the full
    `/confirm pricing <amount> <interval> <window>` echo is accepted.

The whole A/B pipeline is mechanical — these tests assert no
LLMDispatcher.request call is made.

Each test runs on a fresh temp-file SQLite DB with a fresh event loop
(project convention — see test_hypothesis_verdict.py).
"""
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _fresh_db():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    import src.infra.db as db_mod

    db_mod.DB_PATH = db_path
    db_mod._db_connection = None
    os.environ["DB_PATH"] = db_path
    await db_mod.init_db()
    return db_mod, db_path


async def _close_db(db_mod):
    try:
        conn = getattr(db_mod, "_db_connection", None)
        if conn is not None:
            await conn.close()
        db_mod._db_connection = None
    except Exception:
        pass


def _vendor_env(action, dau=250, ok=True):
    """Fake vendor_call envelope for posthog actions."""
    if action == "get_active_users":
        return {"ok": ok, "result": {"result": [
            {"label": "Active users", "data": [dau], "count": dau}]}}
    if action == "create_feature_flag":
        return {"ok": ok, "result": {"id": 7001, "key": "exp_flag"}}
    if action == "update_feature_flag":
        return {"ok": ok, "result": {"id": 7001}}
    return {"ok": ok, "result": {}}


def _patch_vendor_call(target_module, dau=250):
    """Patch vendor_call.run in an executor module to a mock posthog."""
    async def _fake(sub):
        action = (((sub or {}).get("context") or {})
                  .get("post_hook") or {}).get("action", "")
        return _vendor_env(action, dau=dau)
    return patch.object(target_module, "vendor_call_run", _fake) \
        if hasattr(target_module, "vendor_call_run") else None


# ---------------------------------------------------------------------------
# A/B result evaluation — pure Bayesian winner pick
# ---------------------------------------------------------------------------

def test_ab_evaluate_treatment_wins():
    """A clear treatment lift over control → treatment winner, confident."""
    from src.growth.ab_result import evaluate_ab

    ab = evaluate_ab(control_metric=100.0, treatment_metric=130.0,
                     rel_sigma=0.05)
    assert ab.winner == "treatment"
    assert ab.confident is True
    assert ab.p_treatment_better >= 0.95
    assert ab.relative_lift > 0


def test_ab_evaluate_control_wins():
    """Treatment underperforming control → control winner."""
    from src.growth.ab_result import evaluate_ab

    ab = evaluate_ab(control_metric=100.0, treatment_metric=70.0,
                     rel_sigma=0.05)
    assert ab.winner == "control"
    assert ab.confident is True
    assert ab.p_control_better >= 0.95


def test_ab_evaluate_inconclusive():
    """A tiny difference relative to noise → inconclusive, not confident."""
    from src.growth.ab_result import evaluate_ab

    ab = evaluate_ab(control_metric=100.0, treatment_metric=101.0,
                     rel_sigma=0.05)
    assert ab.winner == "inconclusive"
    assert ab.confident is False


# ---------------------------------------------------------------------------
# assign_variant — control + treatment, insufficient-N guard, opt-out
# ---------------------------------------------------------------------------

def test_assign_variant_creates_control_and_treatment():
    """assign_variant with healthy DAU creates two variant rows + flag."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            hyp = await db_mod.insert_hypothesis(
                mid, "checkout", {"metric": "conv", "direction": "up",
                                  "magnitude": 0.1}, 1209600, "checkout|conv")
            from mr_roboto.executors import assign_variant as av

            task = {"id": 1, "mission_id": mid,
                    "payload": {"action": "assign_variant",
                                "feature": "checkout"}}
            with patch.object(av, "_daily_active_users",
                              AsyncMock(return_value=(True, 250))), \
                 patch.object(av, "_wire_posthog_flag",
                              AsyncMock(return_value={"ok": True,
                                                      "flag_id": 7001})):
                res = await av.run(task)

            assert res["ok"] is True
            assert res["split"] is True
            variants = await db_mod.get_variants(mission_id=mid)
            names = sorted(v["variant_name"] for v in variants)
            assert names == ["control", "treatment"]
            assert all(v["status"] == "active" for v in variants)
            assert res["hypothesis_id"] == hyp
        finally:
            await _close_db(db_mod)
    run_async(_t())


def test_assign_variant_insufficient_n_skips_split():
    """DAU < 100 → no split, ab_skipped_low_n event, hypothesis intact."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            await db_mod.insert_hypothesis(
                mid, "feat", {"metric": "act", "direction": "up",
                              "magnitude": 0.1}, 604800, "feat|act")
            from mr_roboto.executors import assign_variant as av

            task = {"id": 1, "mission_id": mid,
                    "payload": {"action": "assign_variant",
                                "feature": "feat"}}
            # Low DAU via mocked posthog get_active_users.
            with patch.object(av, "_daily_active_users",
                              AsyncMock(return_value=(True, 42))):
                res = await av.run(task)

            assert res["ok"] is True
            assert res["split"] is False
            assert res["reason"] == "insufficient_n"
            assert res["daily_active"] == 42
            # No variant rows.
            assert await db_mod.get_variants(mission_id=mid) == []
            # ab_skipped_low_n growth event present.
            evts = await db_mod.get_growth_events(
                mission_id=mid, kind="ab_skipped_low_n")
            assert len(evts) == 1
            assert evts[0]["properties"]["daily_active"] == 42
        finally:
            await _close_db(db_mod)
    run_async(_t())


def test_assign_variant_insufficient_n_via_mock_posthog():
    """End-to-end: real mock-mode posthog get_active_users → guard works.

    The posthog mock config returns 250 DAU, so the split proceeds; the
    guard is exercised by overriding the mock to a low count.
    """
    async def _t():
        db_mod, _ = await _fresh_db()
        os.environ["KUTAI_ENV"] = "test"  # mock mode on
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            from mr_roboto.executors import assign_variant as av

            task = {"id": 1, "mission_id": mid,
                    "payload": {"action": "assign_variant",
                                "feature": "feat"}}
            # Force a low-DAU mock_response so the guard trips.
            from src.integrations.registry import get_integration_registry
            reg = get_integration_registry()
            reg._mock_responses.setdefault("posthog", {})["get_active_users"] = {
                "result": [{"label": "Active users", "data": [12],
                            "count": 12}]
            }
            res = await av.run(task)
            assert res["ok"] is True
            assert res["split"] is False
            assert res["reason"] == "insufficient_n"
        finally:
            # restore healthy mock
            try:
                from src.integrations.registry import (
                    get_integration_registry,
                )
                reg = get_integration_registry()
                reg._mock_responses.get("posthog", {})["get_active_users"] = {
                    "result": [{"label": "Active users",
                                "data": [240, 248, 252, 250],
                                "count": 250}]
                }
            except Exception:
                pass
            await _close_db(db_mod)
    run_async(_t())


def test_assign_variant_respects_opt_out():
    """mission.context['use_ab']=False → no split (ab_skipped_disabled)."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(
                title="m", description="d", context={"use_ab": False})
            from mr_roboto.executors import assign_variant as av

            task = {"id": 1, "mission_id": mid,
                    "payload": {"action": "assign_variant",
                                "feature": "feat"}}
            res = await av.run(task)
            assert res["ok"] is True
            assert res["split"] is False
            assert res["reason"] == "ab_disabled"
            assert await db_mod.get_variants(mission_id=mid) == []
        finally:
            await _close_db(db_mod)
    run_async(_t())


# ---------------------------------------------------------------------------
# A/B result evaluation hooked into the verdict flow
# ---------------------------------------------------------------------------

def test_verdict_flow_evaluates_ab_winner():
    """record_verdict on a mission with variants writes an ab_result event."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            hyp_id = await db_mod.insert_hypothesis(
                mid, "checkout", {"metric": "conv", "direction": "up",
                                  "magnitude": 0.1}, 604800, "checkout|conv")
            await db_mod.insert_variant(mid, hyp_id, "control", "{}")
            await db_mod.insert_variant(mid, hyp_id, "treatment", "{}")

            from mr_roboto.executors import record_verdict as rv

            # Patch the prediction metric pull + inject per-arm metrics.
            async def _fake_posthog(task, metric):
                return {"ok": True, "series": [100, 110, 122]}

            task = {"id": 2, "mission_id": mid,
                    "payload": {"action": "record_verdict",
                                "hypothesis_id": hyp_id,
                                "variant_metrics": {"control": 100.0,
                                                    "treatment": 135.0}}}
            with patch.object(rv, "_posthog_metric", _fake_posthog):
                res = await rv.run(task)

            assert res["ok"] is True
            ab = res["ab_result"]
            assert ab is not None
            assert ab["winner"] == "treatment"
            assert ab["confident"] is True
            evts = await db_mod.get_growth_events(
                mission_id=mid, kind="ab_result")
            assert len(evts) == 1
            assert evts[0]["properties"]["winner"] == "treatment"
        finally:
            await _close_db(db_mod)
    run_async(_t())


def test_verdict_flow_no_variants_no_ab_result():
    """record_verdict without variants → ab_result None, no event."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            hyp_id = await db_mod.insert_hypothesis(
                mid, "feat", {"metric": "act", "direction": "up",
                              "magnitude": 0.1}, 604800, "feat|act")
            from mr_roboto.executors import record_verdict as rv

            async def _fake_posthog(task, metric):
                return {"ok": True, "series": [100, 110]}

            task = {"id": 2, "mission_id": mid,
                    "payload": {"action": "record_verdict",
                                "hypothesis_id": hyp_id}}
            with patch.object(rv, "_posthog_metric", _fake_posthog):
                res = await rv.run(task)
            assert res["ab_result"] is None
            assert await db_mod.get_growth_events(
                mission_id=mid, kind="ab_result") == []
        finally:
            await _close_db(db_mod)
    run_async(_t())


# ---------------------------------------------------------------------------
# retire_variant — winner/loser status + flag flip
# ---------------------------------------------------------------------------

def test_retire_variant_ship_marks_winner_and_loser():
    """retire_variant ship → treatment winner, control loser, flag flipped."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            import json
            rule = json.dumps({"kind": "feature",
                               "posthog_flag_key": "exp_flag",
                               "posthog_flag_id": 7001})
            await db_mod.insert_variant(mid, None, "control", rule)
            await db_mod.insert_variant(mid, None, "treatment", rule)

            from mr_roboto.executors import retire_variant as rt

            flips = []

            async def _fake_flip(task, flag_id, flag_key, rollout):
                flips.append((flag_key, rollout))
                return True

            task = {"id": 3, "mission_id": mid,
                    "payload": {"action": "retire_variant",
                                "mission_id": mid, "winner": "treatment",
                                "decision": "ship"}}
            with patch.object(rt, "_flip_posthog_flag", _fake_flip):
                res = await rt.run(task)

            assert res["ok"] is True
            assert res["retired"] == 2
            variants = await db_mod.get_variants(mission_id=mid)
            byname = {v["variant_name"]: v for v in variants}
            assert byname["treatment"]["status"] == "winner"
            assert byname["control"]["status"] == "loser"
            assert byname["treatment"]["retired_at"] is not None
            # Winner flag → 100, loser → 0.
            assert ("exp_flag", 100) in flips
            assert ("exp_flag", 0) in flips
        finally:
            await _close_db(db_mod)
    run_async(_t())


def test_retire_variant_rollback_crowns_control():
    """retire_variant rollback → control winner regardless of request."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            await db_mod.insert_variant(mid, None, "control", "{}")
            await db_mod.insert_variant(mid, None, "treatment", "{}")
            from mr_roboto.executors import retire_variant as rt

            task = {"id": 3, "mission_id": mid,
                    "payload": {"action": "retire_variant",
                                "mission_id": mid, "winner": "treatment",
                                "decision": "rollback"}}
            with patch.object(rt, "_flip_posthog_flag",
                              AsyncMock(return_value=True)):
                res = await rt.run(task)
            assert res["winner"] == "control"
            variants = await db_mod.get_variants(mission_id=mid)
            byname = {v["variant_name"]: v for v in variants}
            assert byname["control"]["status"] == "winner"
            assert byname["treatment"]["status"] == "loser"
        finally:
            await _close_db(db_mod)
    run_async(_t())


def test_retire_variant_idempotent_no_active():
    """retire_variant on an already-retired experiment → no-op pass."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            vid = await db_mod.insert_variant(mid, None, "control", "{}")
            await db_mod.update_variant_status(vid, "winner")
            from mr_roboto.executors import retire_variant as rt

            task = {"id": 3, "mission_id": mid,
                    "payload": {"action": "retire_variant",
                                "mission_id": mid}}
            res = await rt.run(task)
            assert res["ok"] is True
            assert res["retired"] == 0
        finally:
            await _close_db(db_mod)
    run_async(_t())


# ---------------------------------------------------------------------------
# Telegram — /experiment_ship, /experiment_rollback, /experiment_disable
# ---------------------------------------------------------------------------

def _make_bot():
    """Build a TelegramInterface without running __init__ network setup."""
    from src.app.telegram_bot import TelegramInterface

    bot = TelegramInterface.__new__(TelegramInterface)
    bot._pending_action = {}
    return bot


def _fake_update(text="", chat_id=1):
    upd = MagicMock()
    upd.effective_chat.id = chat_id
    upd.message.text = text
    return upd


def _fake_context(args):
    ctx = MagicMock()
    ctx.args = args
    return ctx


def test_cmd_experiment_disable_sets_flag():
    """/experiment_disable flips mission.context['use_ab'] false."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            bot = _make_bot()
            replies = []
            bot._reply = AsyncMock(
                side_effect=lambda u, t, **k: replies.append(t))
            await bot.cmd_experiment_disable(
                _fake_update(), _fake_context([str(mid)]))
            row = await db_mod.get_mission(mid)
            import json
            ctx = json.loads(row["context"]) if isinstance(
                row["context"], str) else row["context"]
            assert ctx["use_ab"] is False
            assert any("opted out" in r for r in replies)
        finally:
            await _close_db(db_mod)
    run_async(_t())


def test_cmd_experiment_ship_runs_retire_variant():
    """/experiment_ship calls retire_variant and reports the result."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            await db_mod.insert_variant(mid, None, "control", "{}")
            await db_mod.insert_variant(mid, None, "treatment", "{}")
            bot = _make_bot()
            replies = []
            bot._reply = AsyncMock(
                side_effect=lambda u, t, **k: replies.append(t))
            from mr_roboto.executors import retire_variant as rt
            with patch.object(rt, "_flip_posthog_flag",
                              AsyncMock(return_value=True)):
                await bot.cmd_experiment_ship(
                    _fake_update(), _fake_context([str(mid)]))
            variants = await db_mod.get_variants(mission_id=mid)
            statuses = sorted(v["status"] for v in variants)
            assert statuses == ["loser", "winner"]
            assert any("shipped to 100" in r for r in replies)
        finally:
            await _close_db(db_mod)
    run_async(_t())


def test_cmd_experiment_rollback_crowns_control():
    """/experiment_rollback retires variants with control as winner."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            await db_mod.insert_variant(mid, None, "control", "{}")
            await db_mod.insert_variant(mid, None, "treatment", "{}")
            bot = _make_bot()
            replies = []
            bot._reply = AsyncMock(
                side_effect=lambda u, t, **k: replies.append(t))
            from mr_roboto.executors import retire_variant as rt
            with patch.object(rt, "_flip_posthog_flag",
                              AsyncMock(return_value=True)):
                await bot.cmd_experiment_rollback(
                    _fake_update(), _fake_context([str(mid)]))
            variants = await db_mod.get_variants(mission_id=mid)
            byname = {v["variant_name"]: v["status"] for v in variants}
            assert byname["control"] == "winner"
            assert byname["treatment"] == "loser"
            assert any("rolled back" in r for r in replies)
        finally:
            await _close_db(db_mod)
    run_async(_t())


# ---------------------------------------------------------------------------
# T5E — pricing typed confirmation (bare /confirm rejected)
# ---------------------------------------------------------------------------

def test_pricing_bare_confirm_rejected():
    """A bare /confirm with no params is rejected — typed echo required."""
    async def _t():
        bot = _make_bot()
        replies = []
        bot._reply = AsyncMock(side_effect=lambda u, t, **k: replies.append(t))
        await bot.cmd_confirm(_fake_update(), _fake_context([]))
        assert any("Typed confirmation required" in r for r in replies)
    run_async(_t())


def test_pricing_incomplete_confirm_rejected():
    """`/confirm pricing 19.99` without interval+window is rejected."""
    async def _t():
        bot = _make_bot()
        replies = []
        bot._reply = AsyncMock(side_effect=lambda u, t, **k: replies.append(t))
        await bot.cmd_confirm(
            _fake_update(), _fake_context(["pricing", "19.99"]))
        assert any("Incomplete" in r for r in replies)
    run_async(_t())


def test_pricing_full_params_confirm_accepted():
    """Full-params `/confirm pricing 19.99 month 14d` matching pending → ok."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            bot = _make_bot()
            replies = []
            bot._reply = AsyncMock(
                side_effect=lambda u, t, **k: replies.append(t))
            # Stage a pending pricing experiment matching the typed echo.
            bot._pending_action[1] = {
                "command": "_pricing_confirm",
                "params": {
                    "mission_id": mid, "amount": 19.99,
                    "interval": "month", "window": "14d",
                    "control_price_id": "price_ctrl",
                    "treatment_price_id": "price_treat",
                },
            }
            from mr_roboto.executors import assign_variant as av
            with patch.object(av, "_daily_active_users",
                              AsyncMock(return_value=(True, 300))):
                await bot.cmd_confirm(
                    _fake_update(),
                    _fake_context(["pricing", "19.99", "month", "14d"]))
            assert any("Pricing A/B confirmed" in r for r in replies)
            # Pricing variants created.
            variants = await db_mod.get_variants(mission_id=mid)
            assert sorted(v["variant_name"] for v in variants) == \
                ["control", "treatment"]
            # Pending action consumed.
            assert 1 not in bot._pending_action
        finally:
            await _close_db(db_mod)
    run_async(_t())


def test_pricing_confirm_param_mismatch_rejected():
    """Typed params not matching the pending experiment are rejected."""
    async def _t():
        bot = _make_bot()
        replies = []
        bot._reply = AsyncMock(side_effect=lambda u, t, **k: replies.append(t))
        bot._pending_action[1] = {
            "command": "_pricing_confirm",
            "params": {"mission_id": 5, "amount": 29.99,
                       "interval": "month", "window": "14d"},
        }
        await bot.cmd_confirm(
            _fake_update(),
            _fake_context(["pricing", "19.99", "month", "14d"]))
        assert any("do not match" in r for r in replies)
        # Pending action NOT consumed on mismatch.
        assert 1 in bot._pending_action
    run_async(_t())


# ---------------------------------------------------------------------------
# Architecture contract — the A/B pipeline is mechanical (no LLM)
# ---------------------------------------------------------------------------

def test_ab_pipeline_makes_no_llm_call():
    """assign_variant + retire_variant never call LLMDispatcher.request."""
    async def _t():
        db_mod, _ = await _fresh_db()
        try:
            mid = await db_mod.add_mission(title="m", description="d")
            from mr_roboto.executors import assign_variant as av
            from mr_roboto.executors import retire_variant as rt

            with patch(
                "src.core.llm_dispatcher.LLMDispatcher.request"
            ) as mock_req:
                with patch.object(av, "_daily_active_users",
                                  AsyncMock(return_value=(True, 200))), \
                     patch.object(av, "_wire_posthog_flag",
                                  AsyncMock(return_value={"ok": True,
                                                          "flag_id": 1})):
                    await av.run({"id": 1, "mission_id": mid,
                                  "payload": {"action": "assign_variant",
                                              "feature": "f"}})
                with patch.object(rt, "_flip_posthog_flag",
                                  AsyncMock(return_value=True)):
                    await rt.run({"id": 2, "mission_id": mid,
                                  "payload": {"action": "retire_variant",
                                              "mission_id": mid}})
                assert mock_req.call_count == 0
        finally:
            await _close_db(db_mod)
    run_async(_t())
