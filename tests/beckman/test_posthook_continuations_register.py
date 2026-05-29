"""SP3 Task 4 - posthook continuation handlers register."""
def test_posthook_handlers_registered():
    from general_beckman import posthook_continuations as pc
    from general_beckman.continuations import _HANDLERS
    pc.register_continuations()
    for name in (
        "posthook.grade.resume", "posthook.grade.resume_err",
        "posthook.code_review.resume", "posthook.code_review.resume_err",
        "posthook.summary.resume", "posthook.summary.resume_err",
    ):
        assert name in _HANDLERS, f"{name} not registered"

def test_module_in_handler_modules_static_list():
    from general_beckman.continuations import _HANDLER_MODULES
    assert "general_beckman.posthook_continuations" in _HANDLER_MODULES
