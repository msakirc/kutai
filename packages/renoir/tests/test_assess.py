import io
from PIL import Image
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
