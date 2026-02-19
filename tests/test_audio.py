"""Tests for audio device detection utilities."""

from unittest.mock import MagicMock, patch


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
