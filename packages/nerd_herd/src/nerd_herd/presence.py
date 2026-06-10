"""User-presence collector — input-idle time + foreground-fullscreen.

Windows-only signals via ctypes; degrades to "away, not fullscreen" on any
other platform or any error. NEVER raises — presence sensing must not break
the snapshot hot path.
"""
from __future__ import annotations

import sys

from yazbunu import get_logger

logger = get_logger("nerd_herd.presence")

AWAY_IDLE_S = 1e9  # sentinel "no user" idle value used on failure/off-Windows


def _idle_seconds_from_ticks(*, now_ms: int, last_ms: int) -> float:
    """Idle seconds from GetTickCount values, correct across the 32-bit wrap.

    GetTickCount is an unsigned 32-bit ms counter that wraps every ~49.7
    days. Modular subtraction in 32-bit space gives the true elapsed ms
    regardless of wrap; no negative/garbage values.
    """
    delta_ms = (now_ms - last_ms) & 0xFFFFFFFF
    return delta_ms / 1000.0


class PresenceCollector:
    name = "presence"

    def __init__(self) -> None:
        self._is_windows = sys.platform == "win32"

    def collect(self) -> dict:
        """Return {'user_idle_s': float, 'foreground_fullscreen': bool}.

        Cheap (~1-2ms): two WinAPI calls. Safe to call per-snapshot.
        """
        try:
            idle = float(self._idle_seconds_impl())
        except Exception as e:
            logger.debug("presence idle probe failed", error=str(e))
            idle = AWAY_IDLE_S
        try:
            full = bool(self._fullscreen_impl())
        except Exception as e:
            logger.debug("presence fullscreen probe failed", error=str(e))
            full = False
        return {"user_idle_s": idle, "foreground_fullscreen": full}

    def _idle_seconds_impl(self) -> float:
        if not self._is_windows:
            return AWAY_IDLE_S
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        kernel32.GetTickCount.restype = wintypes.DWORD
        if not user32.GetLastInputInfo(ctypes.byref(info)):
            raise OSError("GetLastInputInfo failed")
        return _idle_seconds_from_ticks(
            now_ms=kernel32.GetTickCount(), last_ms=info.dwTime
        )

    def _fullscreen_impl(self) -> bool:
        if not self._is_windows:
            return False
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetDesktopWindow.restype = wintypes.HWND
        user32.GetShellWindow.restype = wintypes.HWND
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        if hwnd in (user32.GetDesktopWindow(), user32.GetShellWindow()):
            return False
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        win_w = rect.right - rect.left
        win_h = rect.bottom - rect.top
        return win_w >= screen_w and win_h >= screen_h
