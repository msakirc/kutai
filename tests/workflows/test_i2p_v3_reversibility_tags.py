import json
import pathlib

I2P = pathlib.Path("src/workflows/i2p/i2p_v3.json")

# Hard-locked dangerous step IDs — audited 2026-05-05.
# These steps perform truly irreversible actions: cloud provisioning (billing starts),
# DNS propagation, production DB migrations, live payment activation, public deploys,
# public announcements, mass email sends, and app store submissions.
#
# Each is tagged reversibility="irreversible" (the most-severe tier of the
# canonical full/partial/irreversible taxonomy — see safety_guard.tags) plus
# locked=true. The pairing matters: safety_guard.executor_hook gates a step
# on the founder ONLY when its resolved tag is IRREVERSIBLE, and `locked`
# makes the static tag authoritative (no runtime downgrade). A locked step
# tagged merely "full"/"partial" would NOT trip the founder-wait gate.
LOCKED_IRREVERSIBLE_IDS = {
    "13.1",  # production_infrastructure — cloud resources provisioned, billing starts
    "13.2",  # network_and_security_infra — DNS records propagate globally
    "13.6",  # production_data_setup — DB migrations run against production DB
    "14.3",  # execute_launch — payment live mode ON, full prod deploy
    "14.5",  # marketing_site_publish — site goes public on production
    "14.6",  # launch_announcements — public posts on Product Hunt / social media
    "14.7",  # launch_email — email blast sent to subscriber list
    # 14.8 (app_store_submission) is the metadata-DRAFTING parent (read_only,
    # reversibility=full). The actual irreversible binary uploads are the
    # mechanical sub-steps below — that is where the founder-wait lock belongs.
    "14.8.submit",       # app_store_submit_binary — TestFlight ingest (iOS)
    "14.8.submit_play",  # play_internal_submit_binary — Play internal (Android)
}


def _walk(node, found, target_ids):
    if isinstance(node, dict):
        sid = node.get("id")
        if sid in target_ids:
            found[sid] = node
        for v in node.values():
            _walk(v, found, target_ids)
    elif isinstance(node, list):
        for v in node:
            _walk(v, found, target_ids)


def test_dangerous_steps_are_locked_irreversible():
    if not LOCKED_IRREVERSIBLE_IDS:
        import pytest
        pytest.skip("no dangerous steps tagged in this workflow yet")
    data = json.loads(I2P.read_text(encoding="utf-8"))
    found = {}
    _walk(data, found, LOCKED_IRREVERSIBLE_IDS)
    missing = LOCKED_IRREVERSIBLE_IDS - found.keys()
    assert not missing, f"missing dangerous step ids in workflow: {missing}"
    for sid, step in found.items():
        assert step.get("reversibility") == "irreversible", (
            f"{sid} reversibility != irreversible"
        )
        assert step.get("locked") is True, f"{sid} locked != true"
