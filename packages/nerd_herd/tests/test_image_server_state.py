"""Tests for nerd_herd image_server_resident + image_server_vram_mb (Task 5)."""
import nerd_herd


def test_default_snapshot_has_image_server_fields():
    nh = nerd_herd.NerdHerd()
    snap = nh.snapshot()
    assert hasattr(snap, "image_server_resident")
    assert hasattr(snap, "image_server_vram_mb")
    assert snap.image_server_resident is False
    assert snap.image_server_vram_mb == 0


def test_push_image_server_state_flips_snapshot():
    nh = nerd_herd.NerdHerd()
    nh.push_image_server_state(resident=True, vram_mb=4500)
    snap = nh.snapshot()
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb == 4500
    nh.push_image_server_state(resident=False, vram_mb=0)
    snap = nh.snapshot()
    assert snap.image_server_resident is False


def test_module_level_record_image_server_state():
    """Singleton path. record_image_server_state writes through to the
    singleton's push_image_server_state, observable via the singleton's
    snapshot()."""
    import nerd_herd as nh_mod
    # Drive the singleton
    nh_mod.record_image_server_state(resident=True, vram_mb=4500)
    # Read back via the singleton directly
    snap = nh_mod._get_singleton().snapshot()
    assert hasattr(snap, "image_server_resident")
    assert hasattr(snap, "image_server_vram_mb")
    assert snap.image_server_resident is True
    assert snap.image_server_vram_mb == 4500
