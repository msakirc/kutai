def test_husam_imports_and_exposes_run():
    import husam
    assert callable(husam.run)

def test_husam_does_not_import_coulson():
    # Purity: the non-agentic worker must never depend on the react worker.
    import sys, husam  # noqa: F401
    assert not any(m == "coulson" or m.startswith("coulson.") for m in sys.modules), \
        "husam import pulled in coulson"
