def test_comms_handlers_registered():
    import general_beckman.continuations as C
    import mr_roboto.executors.comms_continuations  # noqa: F401 — triggers register
    for name in (
        "comms.crisis_holding.resume", "comms.crisis_holding.resume_err",
        "comms.incident_update.resume", "comms.incident_update.resume_err",
        "comms.press_kit.resume", "comms.press_kit.resume_err",
    ):
        assert name in C._HANDLERS, f"{name} not registered"


def test_comms_module_in_handler_modules():
    import general_beckman.continuations as C
    assert "mr_roboto.executors.comms_continuations" in C._HANDLER_MODULES
