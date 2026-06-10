"""Task 17: i2p workflow agent_types are a subset of PROFILE_REGISTRY ∪ {mechanical}.

mechanical = mr_roboto executor, not an agent profile.
oncall_agent = class carve-out that stays in src/agents but IS in AGENT_REGISTRY
               via the merged dict; i2p does not use it directly so it need not
               be listed here, but it's harmless to include in the allowset.
"""
import json
import pathlib

from finch import PROFILE_REGISTRY

_WORKTREE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _i2p_agent_types() -> set:
    p = _WORKTREE_ROOT / "src" / "workflows" / "i2p" / "i2p_v3.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    types: set = set()

    def walk(o):
        if isinstance(o, dict):
            if "agent" in o and isinstance(o["agent"], str):
                types.add(o["agent"])
            if "agent_type" in o and isinstance(o["agent_type"], str):
                types.add(o["agent_type"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    return types


def test_i2p_agent_types_in_registry():
    """Every agent_type in i2p_v3.json must resolve to a real profile (not just
    fall through to the executor default). Allowed set = Foundry registry keys
    ∪ {"mechanical"} (mr_roboto) ∪ {"oncall_agent"} (class carve-out)."""
    keys = set(PROFILE_REGISTRY) | {"mechanical", "oncall_agent"}
    i2p_types = _i2p_agent_types()
    missing = {t for t in i2p_types if t not in keys}
    assert not missing, (
        f"i2p agent_types not found in registry (would silently fall back to "
        f"executor): {missing}"
    )
