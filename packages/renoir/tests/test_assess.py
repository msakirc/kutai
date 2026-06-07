import io
from PIL import Image
import renoir
from renoir import assess, ImageVerdict


def _png(color=(120, 80, 200), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_image_ok():
    v = assess(_png())
    assert isinstance(v, ImageVerdict) and v.ok is True


def test_not_an_image_rejected():
    v = assess(b"<html>rate limited</html>")
    assert v.ok is False and v.reason == "not_an_image"


def test_empty_rejected():
    v = assess(b"")
    assert v.ok is False and v.reason == "empty"


def test_all_one_color_rejected():
    v = assess(_png(color=(0, 0, 0)))
    assert v.ok is False and v.reason == "blank"


def test_too_small_rejected():
    v = assess(_png(size=(4, 4)))
    assert v.ok is False and v.reason == "too_small"


# --- FIX 1: decompression-bomb cap ---

def test_normal_image_within_default_cap_ok():
    """64x64 = 4096 px — well under the 50 MP cap, must pass."""
    v = assess(_png(size=(64, 64)))
    assert v.ok is True


def test_oversized_image_rejected_before_load(monkeypatch):
    """Monkeypatch cap to 100 px; 64x64=4096 > 100 → should be rejected as too_large."""
    monkeypatch.setattr(renoir, "_MAX_PIXELS", 100)
    v = assess(_png(size=(64, 64)))
    assert v.ok is False
    assert v.reason == "too_large"
