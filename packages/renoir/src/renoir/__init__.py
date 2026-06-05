"""renoir — image quality judge. Parallel to dogru_mu_samet (text)."""
from __future__ import annotations

import io
from dataclasses import dataclass

_MIN_DIM = 16


@dataclass(frozen=True)
class ImageVerdict:
    ok: bool
    reason: str = ""


def assess(data: bytes) -> ImageVerdict:
    """Validate generated image bytes. Catches the free-provider failure mode
    of HTTP 200 with a non-image / blank / tiny body."""
    if not data:
        return ImageVerdict(False, "empty")
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return ImageVerdict(False, "not_an_image")

    w, h = img.size
    if w < _MIN_DIM or h < _MIN_DIM:
        return ImageVerdict(False, "too_small")

    try:
        rgb = img.convert("RGB")
        extrema = rgb.getextrema()
        # Blank: every pixel is identical AND all three channels share the same
        # value (achromatic — a shade of black/white/grey with no hue).
        # A valid solid-colour image (e.g. purple) has distinct channel values
        # and is NOT considered blank.
        if all(lo == hi for lo, hi in extrema):
            r_val, g_val, b_val = extrema[0][0], extrema[1][0], extrema[2][0]
            if r_val == g_val == b_val:
                return ImageVerdict(False, "blank")
    except Exception:
        pass

    return ImageVerdict(True, "")
