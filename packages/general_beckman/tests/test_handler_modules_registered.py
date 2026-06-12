"""The two SP5 CPS migrations must be in _HANDLER_MODULES so their resume
handlers exist after a restart (else continuation rows stay pending forever)."""
from general_beckman.continuations import (
    _HANDLER_MODULES,
    _HANDLERS,
    register_startup_handlers,
)


def test_sp5_modules_listed():
    assert "src.core.task_classifier" in _HANDLER_MODULES
    assert "src.app.jobs.investor_bullets" in _HANDLER_MODULES
    # Image-gen Plan 3 swap chain (CPS-migrated 2026-06-11).
    assert "mr_roboto.swap_placeholder_images" in _HANDLER_MODULES


def test_sp5_handlers_register_on_startup():
    register_startup_handlers()
    assert "task_classifier.classify.resume" in _HANDLERS
    assert "investor_bullets.hypothesis.resume" in _HANDLERS
    assert "investor_bullets.hypothesis.resume_err" in _HANDLERS
    assert "mr_roboto.swap_images.prompts_done" in _HANDLERS
    assert "mr_roboto.swap_images.prompts_err" in _HANDLERS
    assert "mr_roboto.swap_images.image_done" in _HANDLERS
    assert "mr_roboto.swap_images.image_err" in _HANDLERS
