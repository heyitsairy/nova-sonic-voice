"""Audio device utilities for Nova Sonic Voice Agent."""

from __future__ import annotations

import pyaudio


def _detect_c920_device() -> int | None:
    """Find the C920 pyaudio input device index."""
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if "C920" in str(info.get("name", "")) and info["maxInputChannels"] > 0:
                return i
    finally:
        p.terminate()
    return None


def detect_mic() -> int | None:
    """Detect the best available microphone device index.

    Prefers C920, falls back to any available input device.
    """
    c920 = _detect_c920_device()
    if c920 is not None:
        return c920

    # Fall back to default
    p = pyaudio.PyAudio()
    try:
        default_info = p.get_default_input_device_info()
        return int(default_info["index"])
    except (OSError, KeyError):
        return None
    finally:
        p.terminate()
