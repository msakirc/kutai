"""Z1 Tier 7B (C21) — Paraflow goldens regression test.

Skips by default. Set ``KUTAI_TEST_MISSION_WORKSPACE`` to the absolute
path of a completed (or in-progress) mission workspace to run.

CI note (do NOT modify CI yaml from this file): run nightly with
``KUTAI_TEST_MISSION_WORKSPACE`` pointing at a synthetic mission_57-style
fixture. The asserted verdict is ``!= paraflow_gap`` — par + partial both
pass; only a wholesale missing bundle fails.

Optional env:
- ``KUTAI_TEST_PARAFLOW_ARCHETYPE`` (default: ``truthrate``)
"""
from __future__ import annotations

import os
import pytest

from c21_paraflow_diff import diff_bundle


_WORKSPACE_ENV = "KUTAI_TEST_MISSION_WORKSPACE"
_ARCHETYPE_ENV = "KUTAI_TEST_PARAFLOW_ARCHETYPE"


@pytest.mark.skipif(
    not os.environ.get(_WORKSPACE_ENV),
    reason=f"set {_WORKSPACE_ENV} to enable",
)
def test_mission_workspace_meets_paraflow_par_or_partial():
    """Mission workspace must not be a wholesale paraflow_gap."""
    workspace = os.environ[_WORKSPACE_ENV]
    archetype = os.environ.get(_ARCHETYPE_ENV, "truthrate")
    res = diff_bundle(workspace, archetype)
    assert res["verdict"] != "paraflow_gap", (
        f"mission workspace {workspace} is paraflow_gap "
        f"(score={res['score']}, gaps={res['gaps']})"
    )
