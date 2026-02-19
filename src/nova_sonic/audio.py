"""Audio device utilities for Nova Sonic Voice Agent.

Provides mic detection and audio format conversion utilities
for bridging between different audio pipelines (Discord, local mic, etc.).

Audio formats:
    Discord:    48kHz / 16-bit / stereo PCM (s16le)
    Nova input: 16kHz / 16-bit / mono LPCM
    Nova output: 24kHz / 16-bit / mono LPCM
"""

from __future__ import annotations

import logging
import struct
from typing import Iterator

import pyaudio

logger = logging.getLogger(__name__)

# --- Constants ---

# Discord voice audio format
DISCORD_SAMPLE_RATE = 48000
DISCORD_CHANNELS = 2
DISCORD_SAMPLE_WIDTH = 2  # 16-bit

# Nova Sonic audio formats
NOVA_INPUT_SAMPLE_RATE = 16000
NOVA_OUTPUT_SAMPLE_RATE = 24000
NOVA_CHANNELS = 1
NOVA_SAMPLE_WIDTH = 2  # 16-bit


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


# --- Audio Format Conversion ---


def stereo_to_mono(pcm_data: bytes) -> bytes:
    """Convert stereo 16-bit PCM to mono by averaging channels.

    Takes interleaved stereo samples (L, R, L, R, ...) and produces
    mono samples by averaging each pair. Input and output are both
    16-bit signed little-endian PCM.

    Args:
        pcm_data: Stereo 16-bit PCM bytes (interleaved L/R samples).

    Returns:
        Mono 16-bit PCM bytes (half the input length).
    """
    if len(pcm_data) % 4 != 0:
        # Each stereo frame = 4 bytes (2 bytes L + 2 bytes R)
        # Trim to nearest frame boundary
        pcm_data = pcm_data[:len(pcm_data) - (len(pcm_data) % 4)]

    if not pcm_data:
        return b""

    n_frames = len(pcm_data) // 4
    stereo = struct.unpack(f"<{n_frames * 2}h", pcm_data)

    mono_samples = []
    for i in range(0, len(stereo), 2):
        avg = (stereo[i] + stereo[i + 1]) // 2
        mono_samples.append(avg)

    return struct.pack(f"<{len(mono_samples)}h", *mono_samples)


def downsample_linear(pcm_data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Downsample mono 16-bit PCM using linear interpolation.

    Simple but fast. For voice audio, the quality difference vs. a proper
    resampler (like libsamplerate) is negligible. Avoids ffmpeg subprocess
    overhead for real-time streaming.

    Args:
        pcm_data: Mono 16-bit PCM bytes at src_rate Hz.
        src_rate: Source sample rate (e.g., 48000).
        dst_rate: Target sample rate (e.g., 16000).

    Returns:
        Mono 16-bit PCM bytes at dst_rate Hz.
    """
    if src_rate == dst_rate:
        return pcm_data

    if not pcm_data:
        return b""

    n_samples = len(pcm_data) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_data)

    ratio = src_rate / dst_rate
    out_len = int(n_samples / ratio)

    if out_len == 0:
        return b""

    output = []
    for i in range(out_len):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx

        if idx + 1 < n_samples:
            val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        else:
            val = float(samples[idx])

        # Clamp to int16 range
        val = max(-32768, min(32767, int(val)))
        output.append(val)

    return struct.pack(f"<{len(output)}h", *output)


def upsample_linear(pcm_data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Upsample mono 16-bit PCM using linear interpolation.

    Used to convert Nova's 24kHz output to Discord's 48kHz.
    Linear interpolation is sufficient for voice audio playback.

    Args:
        pcm_data: Mono 16-bit PCM bytes at src_rate Hz.
        src_rate: Source sample rate (e.g., 24000).
        dst_rate: Target sample rate (e.g., 48000).

    Returns:
        Mono 16-bit PCM bytes at dst_rate Hz.
    """
    if src_rate == dst_rate:
        return pcm_data

    if not pcm_data:
        return b""

    n_samples = len(pcm_data) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_data)

    ratio = src_rate / dst_rate  # < 1 for upsampling
    out_len = int(n_samples / ratio)

    if out_len == 0:
        return b""

    output = []
    for i in range(out_len):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx

        if idx + 1 < n_samples:
            val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
        else:
            val = float(samples[idx])

        val = max(-32768, min(32767, int(val)))
        output.append(val)

    return struct.pack(f"<{len(output)}h", *output)


def mono_to_stereo(pcm_data: bytes) -> bytes:
    """Convert mono 16-bit PCM to stereo by duplicating each sample.

    Discord expects stereo audio for playback. This duplicates each
    mono sample to both left and right channels.

    Args:
        pcm_data: Mono 16-bit PCM bytes.

    Returns:
        Stereo 16-bit PCM bytes (double the input length).
    """
    if not pcm_data:
        return b""

    n_samples = len(pcm_data) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_data)

    stereo = []
    for s in samples:
        stereo.append(s)  # Left
        stereo.append(s)  # Right

    return struct.pack(f"<{len(stereo)}h", *stereo)


def discord_to_nova(pcm_data: bytes) -> bytes:
    """Convert Discord audio to Nova Sonic input format.

    Pipeline: 48kHz stereo → mono → 16kHz mono

    Args:
        pcm_data: Discord audio (48kHz/16-bit/stereo PCM).

    Returns:
        Nova input audio (16kHz/16-bit/mono PCM).
    """
    mono = stereo_to_mono(pcm_data)
    return downsample_linear(mono, DISCORD_SAMPLE_RATE, NOVA_INPUT_SAMPLE_RATE)


def nova_to_discord(pcm_data: bytes) -> bytes:
    """Convert Nova Sonic output to Discord playback format.

    Pipeline: 24kHz mono → 48kHz mono → 48kHz stereo

    Args:
        pcm_data: Nova output audio (24kHz/16-bit/mono PCM).

    Returns:
        Discord audio (48kHz/16-bit/stereo PCM).
    """
    upsampled = upsample_linear(pcm_data, NOVA_OUTPUT_SAMPLE_RATE, DISCORD_SAMPLE_RATE)
    return mono_to_stereo(upsampled)


def chunk_audio(pcm_data: bytes, chunk_size_bytes: int) -> Iterator[bytes]:
    """Split PCM audio into fixed-size chunks for streaming.

    The last chunk may be smaller than chunk_size_bytes.

    Args:
        pcm_data: Raw PCM audio bytes.
        chunk_size_bytes: Size of each chunk in bytes.

    Yields:
        Chunks of PCM audio bytes.
    """
    for offset in range(0, len(pcm_data), chunk_size_bytes):
        yield pcm_data[offset:offset + chunk_size_bytes]
