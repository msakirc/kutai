"""SP3 Task 10 - deleted post-hook agents are gone + unrouted."""
def test_agents_not_in_registry():
    from src.agents import AGENT_REGISTRY
    for name in ("grader", "code_reviewer", "artifact_summarizer"):
        assert name not in AGENT_REGISTRY, f"{name} still in AGENT_REGISTRY"

def test_classifier_does_not_route_to_deleted_agents():
    import src.core.task_classifier as tc
    with open(tc.__file__, encoding="utf-8") as f:
        src = f.read()
    for name in ("grader", "code_reviewer", "artifact_summarizer"):
        assert f'"{name}"' not in src, f"{name} still referenced in classifier"

def test_agents_import_clean():
    import importlib
    import src.agents
    importlib.reload(src.agents)  # must not raise
