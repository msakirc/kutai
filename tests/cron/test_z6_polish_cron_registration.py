"""Z6 polish (P9) — cron registration sweep.

Verifies that every Z6 cron declared in ``cron_seed.py`` has a matching
``_executor`` handler dispatched in ``mr_roboto/__init__.py`` (and the
seed entry is actually loaded into the boot path).

Two failure modes this guards against:
  * Seed entry exists → executor missing → cron fires every interval,
    DLQ fills with ``unknown mechanical action`` rows.
  * Executor exists → seed entry missing → silently dead cron, nobody
    ever invokes it.
"""
from __future__ import annotations

from pathlib import Path


Z6_CRON_TITLES = {
    "stripe_dispute_check",
    "stripe_revenue_digest",
    "tax_export_ledger",
    "compliance_template_staleness",
    "credential_rotation_reminder",
}


def _mr_roboto_init_text() -> str:
    path = (
        Path(__file__).resolve().parents[2]
        / "packages" / "mr_roboto" / "src" / "mr_roboto" / "__init__.py"
    )
    return path.read_text(encoding="utf-8")


def test_z6_crons_present_in_seed():
    from general_beckman.cron_seed import INTERNAL_CADENCES
    seeded_titles = {c["title"] for c in INTERNAL_CADENCES}
    missing = Z6_CRON_TITLES - seeded_titles
    assert not missing, (
        f"Z6 crons missing from cron_seed.INTERNAL_CADENCES: "
        f"{sorted(missing)}"
    )


def test_z6_cron_executors_dispatched_in_mr_roboto():
    """Each Z6 cron's ``_executor`` value must be referenced by a
    ``if action == "<name>"`` branch in mr_roboto's dispatch."""
    from general_beckman.cron_seed import INTERNAL_CADENCES
    init_text = _mr_roboto_init_text()
    for cadence in INTERNAL_CADENCES:
        title = cadence["title"]
        if title not in Z6_CRON_TITLES:
            continue
        executor = cadence["payload"].get("_executor")
        assert executor, f"Z6 cron {title} missing _executor in payload"
        # The dispatch site uses ``if action == "<executor>":``. We do a
        # cheap substring check on the canonical string literal.
        needle = f'action == "{executor}"'
        assert needle in init_text, (
            f"Z6 cron {title} executor {executor!r} not dispatched in "
            f"mr_roboto/__init__.py (looked for {needle!r})"
        )


def test_z6_cron_executor_modules_importable():
    """Every Z6 cron handler lives at
    ``mr_roboto.executors.<name>`` — sanity-check importability."""
    import importlib
    modules = [
        "mr_roboto.executors.stripe_dispute_check",
        "mr_roboto.executors.stripe_revenue_digest",
        "mr_roboto.executors.tax_export_ledger",
        "mr_roboto.executors.compliance_template_staleness",
        "mr_roboto.executors.credential_rotation_reminder",
    ]
    for mod in modules:
        importlib.import_module(mod)


def test_seed_is_consumed_at_cron_boot_path():
    """cron_seed.seed_internal_cadences must be imported and called from
    the beckman cron startup module — otherwise the seed is dead code."""
    cron_path = (
        Path(__file__).resolve().parents[2]
        / "packages" / "general_beckman" / "src" / "general_beckman"
        / "cron.py"
    )
    text = cron_path.read_text(encoding="utf-8")
    assert "seed_internal_cadences" in text, (
        "general_beckman/cron.py must import seed_internal_cadences"
    )
    assert "await seed_internal_cadences()" in text, (
        "general_beckman/cron.py must await seed_internal_cadences() at boot"
    )


def test_no_orphan_z6_executor_files():
    """Inverse check — every mr_roboto.executors module that looks Z6-
    related must have a matching seed entry (so we don't have dead
    handlers nobody calls)."""
    from general_beckman.cron_seed import INTERNAL_CADENCES
    seeded_executors = {
        c["payload"].get("_executor")
        for c in INTERNAL_CADENCES
        if c["payload"].get("_executor")
    }
    executors_dir = (
        Path(__file__).resolve().parents[2]
        / "packages" / "mr_roboto" / "src" / "mr_roboto" / "executors"
    )
    expected_z6_executors = {
        "stripe_dispute_check",
        "stripe_revenue_digest",
        "tax_export_ledger",
        "compliance_template_staleness",
        "credential_rotation_reminder",
    }
    for name in expected_z6_executors:
        f = executors_dir / f"{name}.py"
        assert f.exists(), f"missing executor module {f}"
        assert name in seeded_executors, (
            f"executor {name}.py exists but no cron_seed entry references it"
        )
