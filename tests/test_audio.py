"""Tests for audio device detection and format conversion utilities."""

import struct
from unittest.mock import MagicMock, patch

from nova_sonic.audio import (
    chunk_audio,
    discord_to_nova,
    downsample_linear,
    mono_to_stereo,
    nova_to_discord,
    stereo_to_mono,
    upsample_linear,
)


class TestDetectMic:
    @patch("nova_sonic.audio.pyaudio.PyAudio")
    def test_detect_c920(self, mock_pyaudio_class):
        mock_pa = MagicMock()
        mock_pyaudio_class.return_value = mock_pa
        mock_pa.get_device_count.return_value = 3
        mock_pa.get_device_info_by_index.side_effect = [
            {"name": "HDA Intel PCH", "maxInputChannels": 2},
            {"name": "sysdefault", "maxInputChannels": 0},
            {"name": "HD Pro Webcam C920: USB Audio (hw:2,0)", "maxInputChannels": 2},
        ]

        from nova_sonic.audio import _detect_c920_device
        result = _detect_c920_device()
        assert result == 2

    @patch("nova_sonic.audio.pyaudio.PyAudio")
    def test_detect_no_c920(self, mock_pyaudio_class):
        mock_pa = MagicMock()
        mock_pyaudio_class.return_value = mock_pa
        mock_pa.get_device_count.return_value = 1
        mock_pa.get_device_info_by_index.side_effect = [
            {"name": "Default Input", "maxInputChannels": 1},
        ]

        from nova_sonic.audio import _detect_c920_device
        result = _detect_c920_device()
        assert result is None

    @patch("nova_sonic.audio._detect_c920_device", return_value=None)
    @patch("nova_sonic.audio.pyaudio.PyAudio")
    def test_detect_mic_fallback(self, mock_pyaudio_class, mock_c920):
        mock_pa = MagicMock()
        mock_pyaudio_class.return_value = mock_pa
        mock_pa.get_default_input_device_info.return_value = {"index": 5}

        from nova_sonic.audio import detect_mic
        result = detect_mic()
        assert result == 5

    @patch("nova_sonic.audio._detect_c920_device", return_value=7)
    def test_detect_mic_prefers_c920(self, mock_c920):
        from nova_sonic.audio import detect_mic
        result = detect_mic()
        assert result == 7


# --- Stereo/Mono Conversion Tests ---


class TestStereoToMono:
    def test_basic_conversion(self):
        """Average of L and R channels."""
        # Two stereo frames: (100, 200), (300, 400)
        stereo = struct.pack("<4h", 100, 200, 300, 400)
        mono = stereo_to_mono(stereo)
        samples = struct.unpack(f"<{len(mono) // 2}h", mono)
        assert samples == (150, 350)

    def test_negative_values(self):
        """Handles negative samples correctly."""
        stereo = struct.pack("<2h", -1000, 1000)
        mono = stereo_to_mono(stereo)
        samples = struct.unpack("<1h", mono)
        assert samples == (0,)

    def test_empty_input(self):
        assert stereo_to_mono(b"") == b""

    def test_trims_incomplete_frame(self):
        """Non-frame-aligned input gets trimmed."""
        # 6 bytes = 1 complete frame (4 bytes) + 2 extra bytes
        stereo = struct.pack("<2h", 100, 200) + b"\x00\x00"
        mono = stereo_to_mono(stereo)
        samples = struct.unpack("<1h", mono)
        assert samples == (150,)

    def test_output_is_half_length(self):
        """Mono output is exactly half the stereo input length."""
        stereo = struct.pack("<10h", *range(10))
        mono = stereo_to_mono(stereo)
        assert len(mono) == len(stereo) // 2

    def test_extreme_values(self):
        """Handles int16 boundary values."""
        stereo = struct.pack("<2h", 32767, -32768)
        mono = stereo_to_mono(stereo)
        samples = struct.unpack("<1h", mono)
        # (32767 + (-32768)) // 2 = -1 // 2 = -1 (Python floor division)
        assert samples[0] == -1


class TestMonoToStereo:
    def test_basic_conversion(self):
        """Each mono sample duplicated to both channels."""
        mono = struct.pack("<2h", 100, 200)
        stereo = mono_to_stereo(mono)
        samples = struct.unpack(f"<{len(stereo) // 2}h", stereo)
        assert samples == (100, 100, 200, 200)

    def test_empty_input(self):
        assert mono_to_stereo(b"") == b""

    def test_output_is_double_length(self):
        mono = struct.pack("<5h", *range(5))
        stereo = mono_to_stereo(mono)
        assert len(stereo) == len(mono) * 2

    def test_roundtrip(self):
        """stereo_to_mono(mono_to_stereo(x)) == x for identical channels."""
        original = struct.pack("<3h", 100, -200, 300)
        roundtrip = stereo_to_mono(mono_to_stereo(original))
        assert roundtrip == original


# --- Resampling Tests ---


class TestDownsampleLinear:
    def test_identity_same_rate(self):
        """No-op when src == dst."""
        data = struct.pack("<4h", 10, 20, 30, 40)
        assert downsample_linear(data, 48000, 48000) == data

    def test_3x_downsample(self):
        """48kHz -> 16kHz = 3x reduction."""
        # 9 samples at 48kHz -> 3 samples at 16kHz
        samples = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        data = struct.pack(f"<{len(samples)}h", *samples)
        result = downsample_linear(data, 48000, 16000)
        out_samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert len(out_samples) == 3

    def test_2x_downsample(self):
        """48kHz -> 24kHz = 2x reduction."""
        samples = [0, 1000, 2000, 3000, 4000, 5000]
        data = struct.pack(f"<{len(samples)}h", *samples)
        result = downsample_linear(data, 48000, 24000)
        out_samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert len(out_samples) == 3

    def test_empty_input(self):
        assert downsample_linear(b"", 48000, 16000) == b""

    def test_output_values_are_interpolated(self):
        """Output values should be interpolated, not just picked."""
        # Linear ramp: 0, 1000, 2000 at 2x rate -> interpolated at 1x
        samples = [0, 1000, 2000, 3000]
        data = struct.pack(f"<{len(samples)}h", *samples)
        result = downsample_linear(data, 2, 1)
        out_samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert len(out_samples) == 2
        # First sample at position 0.0 -> value 0
        assert out_samples[0] == 0
        # Second sample at position 2.0 -> value 2000
        assert out_samples[1] == 2000


class TestUpsampleLinear:
    def test_identity_same_rate(self):
        data = struct.pack("<4h", 10, 20, 30, 40)
        assert upsample_linear(data, 24000, 24000) == data

    def test_2x_upsample(self):
        """24kHz -> 48kHz = 2x expansion."""
        samples = [0, 1000, 2000]
        data = struct.pack(f"<{len(samples)}h", *samples)
        result = upsample_linear(data, 24000, 48000)
        out_samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert len(out_samples) == 6

    def test_interpolated_values(self):
        """Upsampled values should interpolate between source samples."""
        # Two samples: 0 and 1000, upsampled 2x -> 0, 500, 1000, ...
        samples = [0, 1000]
        data = struct.pack(f"<{len(samples)}h", *samples)
        result = upsample_linear(data, 1, 2)
        out_samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert len(out_samples) == 4
        assert out_samples[0] == 0
        assert out_samples[1] == 500  # Midpoint interpolation

    def test_empty_input(self):
        assert upsample_linear(b"", 24000, 48000) == b""


# --- End-to-End Conversion Tests ---


class TestDiscordToNova:
    def test_converts_format(self):
        """48kHz stereo -> 16kHz mono (3x downsample + stereo to mono)."""
        # 6 stereo frames at 48kHz = 24 bytes
        stereo = struct.pack("<12h", *range(12))
        result = discord_to_nova(stereo)
        # 6 frames / 2 (stereo->mono) = 6 mono samples
        # 6 / 3 (48k->16k) = 2 output samples
        out_samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert len(out_samples) == 2

    def test_empty_input(self):
        assert discord_to_nova(b"") == b""


class TestNovaToDiscord:
    def test_converts_format(self):
        """24kHz mono -> 48kHz stereo (2x upsample + mono to stereo)."""
        # 3 mono samples at 24kHz = 6 bytes
        mono = struct.pack("<3h", 100, 200, 300)
        result = nova_to_discord(mono)
        # 3 * 2 (24k->48k) = 6 upsampled mono
        # 6 * 2 (mono->stereo) = 12 stereo samples = 24 bytes
        out_samples = struct.unpack(f"<{len(result) // 2}h", result)
        assert len(out_samples) == 12

    def test_stereo_pairs_are_equal(self):
        """Each stereo pair should have identical L/R values."""
        mono = struct.pack("<2h", 500, 1000)
        result = nova_to_discord(mono)
        out_samples = struct.unpack(f"<{len(result) // 2}h", result)
        # Check every pair is equal
        for i in range(0, len(out_samples), 2):
            assert out_samples[i] == out_samples[i + 1]

    def test_empty_input(self):
        assert nova_to_discord(b"") == b""


# --- Chunking Tests ---


class TestChunkAudio:
    def test_exact_chunks(self):
        """Data that divides evenly into chunks."""
        data = b"\x00" * 100
        chunks = list(chunk_audio(data, 25))
        assert len(chunks) == 4
        assert all(len(c) == 25 for c in chunks)

    def test_last_chunk_smaller(self):
        """Last chunk is smaller when data doesn't divide evenly."""
        data = b"\x00" * 100
        chunks = list(chunk_audio(data, 30))
        assert len(chunks) == 4
        assert len(chunks[-1]) == 10

    def test_single_chunk(self):
        data = b"\x00" * 10
        chunks = list(chunk_audio(data, 100))
        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_empty_data(self):
        chunks = list(chunk_audio(b"", 100))
        assert chunks == []

    def test_preserves_content(self):
        """All chunks concatenated equals original data."""
        data = bytes(range(256)) * 4
        chunks = list(chunk_audio(data, 100))
        assert b"".join(chunks) == data
