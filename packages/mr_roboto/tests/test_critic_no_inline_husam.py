import pathlib


def test_gated_executors_have_no_inline_husam_or_produce_verdict():
    src = pathlib.Path("packages/mr_roboto/src/mr_roboto/__init__.py").read_text(encoding="utf-8")
    assert "produce_verdict" not in src
    assert "husam.run" not in src  # gated executors never call husam directly


def test_critic_gate_module_has_no_orchestrator():
    src = pathlib.Path("packages/mr_roboto/src/mr_roboto/critic_gate.py").read_text(encoding="utf-8")
    assert "async def produce_verdict" not in src
    assert "async def critic_gate(" not in src
    assert "import husam" not in src
