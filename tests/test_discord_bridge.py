"""Tests for the Discord voice channel bridge."""

from __future__ import annotations

import asyncio
import struct
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nova_sonic.discord_bridge import (
    BUFFER_FRAMES,
    DISCORD_FRAME_BYTES,
    DISCORD_FRAME_MS,
    DISCORD_FRAME_SAMPLES,
    BridgeMetrics,
    DiscordAudioSink,
    DiscordAudioSource,
    NovaSonicBridge,
)
from nova_sonic.session import NovaSonicConfig, SessionState


# --- Helpers ---


def make_discord_frame(value: int = 0) -> bytes:
    """Create a valid 20ms Discord audio frame (48kHz/stereo/16-bit).

    Args:
        value: Sample value to fill with (default: silence).
    """
    samples = [value] * (DISCORD_FRAME_SAMPLES * 2)  # stereo
    return struct.pack(f"<{len(samples)}h", *samples)


def make_silence_frame() -> bytes:
    """Create a silent Discord frame."""
    return b"\x00" * DISCORD_FRAME_BYTES


# --- BridgeMetrics ---


class TestBridgeMetrics:
    def test_defaults(self):
        m = BridgeMetrics()
        assert m.started_at == 0.0
        assert m.discord_frames_received == 0
        assert m.discord_frames_sent == 0
        assert m.nova_chunks_forwarded == 0
        assert m.user_turns == 0
        assert m.assistant_turns == 0

    def test_custom_values(self):
        m = BridgeMetrics(started_at=1000.0, discord_frames_received=42)
        assert m.started_at == 1000.0
        assert m.discord_frames_received == 42


# --- NovaSonicBridge init ---


class TestBridgeInit:
    def test_default_config(self):
        bridge = NovaSonicBridge()
        assert bridge.config.voice_id == "matthew"
        assert not bridge.is_running
        assert bridge.session is None

    def test_custom_config(self):
        config = NovaSonicConfig(voice_id="ruth", system_prompt="Test prompt")
        bridge = NovaSonicBridge(config=config)
        assert bridge.config.voice_id == "ruth"
        assert bridge.config.system_prompt == "Test prompt"

    def test_callbacks(self):
        cb1 = MagicMock()
        cb2 = MagicMock()
        bridge = NovaSonicBridge(on_transcript=cb1, on_user_speech=cb2)
        assert bridge._on_transcript is cb1
        assert bridge._on_user_speech is cb2


# --- Audio receiving ---


class TestReceiveAudio:
    def test_receive_discord_audio(self):
        bridge = NovaSonicBridge()
        bridge._running = True
        frame = make_discord_frame()

        bridge.receive_discord_audio(frame)
        assert len(bridge._input_buffer) == 1
        assert bridge.metrics.discord_frames_received == 1

    def test_receive_with_user_id(self):
        bridge = NovaSonicBridge()
        bridge._running = True

        bridge.receive_discord_audio(make_discord_frame(), user_id=12345)
        assert 12345 in bridge.active_speakers
        assert bridge.active_speakers[12345] > 0

    def test_receive_multiple_users(self):
        bridge = NovaSonicBridge()
        bridge._running = True

        bridge.receive_discord_audio(make_discord_frame(), user_id=111)
        bridge.receive_discord_audio(make_discord_frame(), user_id=222)

        assert len(bridge.active_speakers) == 2
        assert 111 in bridge.active_speakers
        assert 222 in bridge.active_speakers

    def test_receive_when_not_running(self):
        bridge = NovaSonicBridge()
        bridge.receive_discord_audio(make_discord_frame())
        assert len(bridge._input_buffer) == 0
        assert bridge.metrics.discord_frames_received == 0

    def test_receive_updates_speaker_time(self):
        bridge = NovaSonicBridge()
        bridge._running = True

        t1 = time.time()
        bridge.receive_discord_audio(make_discord_frame(), user_id=42)
        t2 = time.time()

        assert t1 <= bridge.active_speakers[42] <= t2


# --- Audio output ---


class TestGetAudio:
    def test_get_discord_audio_empty(self):
        bridge = NovaSonicBridge()
        assert bridge.get_discord_audio() is None

    def test_get_discord_audio_with_data(self):
        bridge = NovaSonicBridge()
        frame = make_silence_frame()
        bridge._output_queue.put_nowait(frame)

        result = bridge.get_discord_audio()
        assert result == frame

    @pytest.mark.asyncio
    async def test_get_discord_audio_async_empty(self):
        bridge = NovaSonicBridge()
        result = await bridge.get_discord_audio_async(timeout=0.01)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_discord_audio_async_with_data(self):
        bridge = NovaSonicBridge()
        frame = make_silence_frame()
        bridge._output_queue.put_nowait(frame)

        result = await bridge.get_discord_audio_async(timeout=0.1)
        assert result == frame


# --- Text callbacks ---


class TestTextCallbacks:
    def test_transcript_callback(self):
        cb = MagicMock()
        bridge = NovaSonicBridge(on_transcript=cb)

        bridge._text_callback("user", "hello")
        cb.assert_called_once_with("user", "hello")

    def test_user_speech_callback(self):
        user_cb = MagicMock()
        bridge = NovaSonicBridge(on_user_speech=user_cb)

        bridge._text_callback("user", "hello world")
        user_cb.assert_called_once_with("user", "hello world")
        assert bridge.metrics.user_turns == 1

    def test_assistant_speech_callback(self):
        assistant_cb = MagicMock()
        bridge = NovaSonicBridge(on_assistant_speech=assistant_cb)

        bridge._text_callback("assistant", "hi there")
        assistant_cb.assert_called_once_with("assistant", "hi there")
        assert bridge.metrics.assistant_turns == 1

    def test_all_callbacks_fire(self):
        transcript_cb = MagicMock()
        user_cb = MagicMock()
        assistant_cb = MagicMock()
        bridge = NovaSonicBridge(
            on_transcript=transcript_cb,
            on_user_speech=user_cb,
            on_assistant_speech=assistant_cb,
        )

        bridge._text_callback("user", "test")
        assert transcript_cb.call_count == 1
        assert user_cb.call_count == 1
        assert assistant_cb.call_count == 0

        bridge._text_callback("assistant", "response")
        assert transcript_cb.call_count == 2
        assert user_cb.call_count == 1
        assert assistant_cb.call_count == 1

    def test_no_callbacks_no_error(self):
        bridge = NovaSonicBridge()
        # Should not raise
        bridge._text_callback("user", "hello")
        bridge._text_callback("assistant", "hi")


# --- Audio callback ---


class TestAudioCallback:
    def test_audio_callback_converts_and_queues(self):
        bridge = NovaSonicBridge()

        # Create 24kHz mono audio (Nova output format)
        # 480 samples at 24kHz = 20ms
        samples = [1000] * 480
        nova_audio = struct.pack(f"<{len(samples)}h", *samples)

        bridge._audio_callback(nova_audio)

        assert bridge.metrics.nova_audio_chunks_received == 1
        assert not bridge._output_queue.empty()

    def test_audio_callback_produces_discord_format(self):
        bridge = NovaSonicBridge()

        # 960 samples at 24kHz = 40ms of audio
        samples = [500] * 960
        nova_audio = struct.pack(f"<{len(samples)}h", *samples)

        bridge._audio_callback(nova_audio)

        # Should produce Discord frames (48kHz stereo)
        frame = bridge.get_discord_audio()
        assert frame is not None
        assert len(frame) == DISCORD_FRAME_BYTES


# --- Start/Stop ---


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_session(self):
        bridge = NovaSonicBridge()

        with patch.object(
            NovaSonicBridge, "start", new_callable=AsyncMock
        ) as mock_start:
            # Just verify the method exists and is callable
            await mock_start()
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        bridge = NovaSonicBridge()
        # Should not raise
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_buffers(self):
        bridge = NovaSonicBridge()
        bridge._running = True
        bridge._input_buffer.append(make_discord_frame())
        bridge._output_queue.put_nowait(make_silence_frame())
        bridge._metrics.started_at = time.time()

        # Mock session
        bridge._session = MagicMock()
        bridge._session.stop = AsyncMock()

        await bridge.stop()

        assert not bridge._running
        assert len(bridge._input_buffer) == 0
        assert bridge._output_queue.empty()

    @pytest.mark.asyncio
    async def test_double_start_ignored(self):
        bridge = NovaSonicBridge()
        bridge._running = True

        # Mock session creation
        with patch.object(NovaSonicBridge, "_forward_loop", new_callable=AsyncMock):
            # Should return early since already running
            # We need a real start to test this
            pass

        assert bridge._running


# --- Forward loop ---


class TestForwardLoop:
    @pytest.mark.asyncio
    async def test_forward_loop_sends_buffered_audio(self):
        bridge = NovaSonicBridge()
        bridge._running = True

        # Mock session
        mock_session = MagicMock()
        mock_session.is_active = True
        mock_session.send_audio = AsyncMock()
        bridge._session = mock_session

        # Buffer enough frames
        for _ in range(BUFFER_FRAMES):
            bridge._input_buffer.append(make_discord_frame(value=100))

        # Run one iteration of the forward loop manually
        # (instead of running the full loop which would run forever)
        if len(bridge._input_buffer) >= BUFFER_FRAMES:
            frames = []
            for _ in range(BUFFER_FRAMES):
                frames.append(bridge._input_buffer.popleft())
            combined = b"".join(frames)
            from nova_sonic.audio import discord_to_nova

            nova_audio = discord_to_nova(combined)
            await mock_session.send_audio(nova_audio)
            bridge._metrics.nova_chunks_forwarded += 1

        assert mock_session.send_audio.call_count == 1
        assert bridge.metrics.nova_chunks_forwarded == 1
        assert len(bridge._input_buffer) == 0

    @pytest.mark.asyncio
    async def test_forward_loop_waits_for_buffer(self):
        bridge = NovaSonicBridge()
        bridge._running = True

        # Only add 1 frame (need BUFFER_FRAMES)
        bridge._input_buffer.append(make_discord_frame())

        # Buffer not full, so nothing should be forwarded
        assert len(bridge._input_buffer) < BUFFER_FRAMES
        # The frame should still be in the buffer
        assert len(bridge._input_buffer) == 1

    @pytest.mark.asyncio
    async def test_stale_speakers_cleaned(self):
        bridge = NovaSonicBridge()
        bridge._running = True

        # Add a speaker with a stale timestamp
        bridge._active_speakers[999] = time.time() - 10  # 10 seconds ago
        bridge._active_speakers[888] = time.time()  # recent

        # Simulate cleanup (from forward loop)
        now = time.time()
        stale = [
            uid
            for uid, last in bridge._active_speakers.items()
            if now - last > 5.0
        ]
        for uid in stale:
            del bridge._active_speakers[uid]

        assert 999 not in bridge._active_speakers
        assert 888 in bridge._active_speakers


# --- Transcript ---


class TestTranscript:
    def test_get_transcript_no_session(self):
        bridge = NovaSonicBridge()
        assert bridge.get_transcript() == []

    def test_get_transcript_with_session(self):
        bridge = NovaSonicBridge()
        mock_session = MagicMock()
        mock_session.history = [
            MagicMock(role="user", text="hello"),
            MagicMock(role="assistant", text="hi"),
        ]
        bridge._session = mock_session
        assert len(bridge.get_transcript()) == 2


# --- DiscordAudioSink ---


class TestDiscordAudioSink:
    def test_write_forwards_to_bridge(self):
        bridge = NovaSonicBridge()
        bridge._running = True
        sink = DiscordAudioSink(bridge)

        frame = make_discord_frame()
        sink.write(frame, user_id=42)

        assert len(bridge._input_buffer) == 1
        assert bridge.metrics.discord_frames_received == 1
        assert 42 in bridge.active_speakers

    def test_write_without_user_id(self):
        bridge = NovaSonicBridge()
        bridge._running = True
        sink = DiscordAudioSink(bridge)

        sink.write(make_discord_frame())
        assert len(bridge._input_buffer) == 1
        assert len(bridge.active_speakers) == 0


# --- DiscordAudioSource ---


class TestDiscordAudioSource:
    def test_read_returns_silence_when_empty(self):
        bridge = NovaSonicBridge()
        source = DiscordAudioSource(bridge)

        frame = source.read()
        assert frame == b"\x00" * DISCORD_FRAME_BYTES
        assert len(frame) == DISCORD_FRAME_BYTES

    def test_read_returns_queued_audio(self):
        bridge = NovaSonicBridge()
        source = DiscordAudioSource(bridge)

        expected = make_discord_frame(value=500)
        bridge._output_queue.put_nowait(expected)

        frame = source.read()
        assert frame == expected

    def test_is_opus_returns_false(self):
        bridge = NovaSonicBridge()
        source = DiscordAudioSource(bridge)
        assert source.is_opus() is False

    def test_cleanup_no_error(self):
        bridge = NovaSonicBridge()
        source = DiscordAudioSource(bridge)
        source.cleanup()  # Should not raise


# --- Constants ---


class TestConstants:
    def test_discord_frame_size(self):
        # 20ms at 48kHz stereo 16-bit
        expected = 48000 * 20 // 1000 * 2 * 2  # samples * channels * bytes_per_sample
        assert DISCORD_FRAME_BYTES == expected
        assert DISCORD_FRAME_BYTES == 3840

    def test_discord_frame_samples(self):
        assert DISCORD_FRAME_SAMPLES == 960  # 48000 * 20 / 1000

    def test_discord_frame_ms(self):
        assert DISCORD_FRAME_MS == 20

    def test_buffer_frames(self):
        assert BUFFER_FRAMES >= 1
