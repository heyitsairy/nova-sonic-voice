"""Tests for the Nova Sonic session module."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from nova_sonic.session import (
    DEFAULT_REGION,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VOICE_ID,
    INPUT_SAMPLE_RATE,
    MAX_HISTORY_TURNS,
    MODEL_ID,
    OUTPUT_SAMPLE_RATE,
    ConversationTurn,
    NovaSonicConfig,
    NovaSonicSession,
    SessionMetrics,
    SessionState,
)


class TestNovaSonicConfig:
    def test_defaults(self):
        config = NovaSonicConfig()
        assert config.region == DEFAULT_REGION
        assert config.model_id == MODEL_ID
        assert config.voice_id == DEFAULT_VOICE_ID
        assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
        assert config.max_tokens == 1024
        assert config.top_p == 0.9
        assert config.temperature == 0.7
        assert config.input_device_index is None

    def test_custom_config(self):
        config = NovaSonicConfig(
            region="us-west-2",
            voice_id="ruth",
            system_prompt="Custom prompt",
            max_tokens=512,
            input_device_index=3,
        )
        assert config.region == "us-west-2"
        assert config.voice_id == "ruth"
        assert config.system_prompt == "Custom prompt"
        assert config.max_tokens == 512
        assert config.input_device_index == 3


class TestSessionMetrics:
    def test_defaults(self):
        metrics = SessionMetrics()
        assert metrics.session_start_time == 0.0
        assert metrics.events_received == 0
        assert metrics.audio_chunks_sent == 0
        assert metrics.audio_chunks_received == 0
        assert metrics.turns_completed == 0
        assert metrics.reconnections == 0


class TestConversationTurn:
    def test_creation(self):
        turn = ConversationTurn(role="user", text="hello", timestamp=1234.0)
        assert turn.role == "user"
        assert turn.text == "hello"
        assert turn.timestamp == 1234.0

    def test_default_timestamp(self):
        turn = ConversationTurn(role="assistant", text="hi")
        assert turn.timestamp == 0.0


class TestSessionState:
    def test_initial_state(self):
        session = NovaSonicSession()
        assert session.state == SessionState.IDLE
        assert not session.is_active

    def test_state_enum_values(self):
        assert SessionState.IDLE.value == "idle"
        assert SessionState.CONNECTING.value == "connecting"
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.RECONNECTING.value == "reconnecting"
        assert SessionState.CLOSING.value == "closing"
        assert SessionState.CLOSED.value == "closed"


class TestEventProtocol:
    """Test that events are formatted correctly for Nova Sonic."""

    def test_session_start_event(self):
        config = NovaSonicConfig(max_tokens=512, top_p=0.8, temperature=0.5)
        event = {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": config.max_tokens,
                        "topP": config.top_p,
                        "temperature": config.temperature,
                    }
                }
            }
        }
        data = json.loads(json.dumps(event))
        assert data["event"]["sessionStart"]["inferenceConfiguration"]["maxTokens"] == 512
        assert data["event"]["sessionStart"]["inferenceConfiguration"]["topP"] == 0.8

    def test_prompt_start_event(self):
        prompt_name = "test-prompt"
        event = {
            "event": {
                "promptStart": {
                    "promptName": prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                }
            }
        }
        data = json.loads(json.dumps(event))
        assert data["event"]["promptStart"]["promptName"] == prompt_name
        assert data["event"]["promptStart"]["audioOutputConfiguration"]["sampleRateHertz"] == 24000

    def test_audio_input_event(self):
        pcm_data = b"\x00\x01\x02\x03" * 256
        b64 = base64.b64encode(pcm_data).decode("utf-8")
        event = {
            "event": {
                "audioInput": {
                    "promptName": "p1",
                    "contentName": "a1",
                    "content": b64,
                }
            }
        }
        data = json.loads(json.dumps(event))
        decoded = base64.b64decode(data["event"]["audioInput"]["content"])
        assert decoded == pcm_data


class TestResponseParsing:
    def test_parse_text_output(self):
        response = {"event": {"textOutput": {"content": "Hello!"}}}
        assert response["event"]["textOutput"]["content"] == "Hello!"

    def test_parse_audio_output(self):
        audio_bytes = b"\x00\x01" * 100
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        response = {"event": {"audioOutput": {"content": b64}}}
        decoded = base64.b64decode(response["event"]["audioOutput"]["content"])
        assert decoded == audio_bytes

    def test_parse_content_start_speculative(self):
        response = {
            "event": {
                "contentStart": {
                    "role": "ASSISTANT",
                    "additionalModelFields": json.dumps({"generationStage": "SPECULATIVE"}),
                }
            }
        }
        fields = json.loads(response["event"]["contentStart"]["additionalModelFields"])
        assert fields["generationStage"] == "SPECULATIVE"

    def test_parse_usage_event(self):
        response = {"event": {"usageEvent": {"inputTokens": 42, "outputTokens": 15}}}
        assert response["event"]["usageEvent"]["inputTokens"] == 42

    def test_parse_completion_end(self):
        response = {"event": {"completionEnd": {}}}
        assert "completionEnd" in response["event"]


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_stop_when_idle(self):
        session = NovaSonicSession()
        await session.stop()
        assert session.state == SessionState.IDLE

    @pytest.mark.asyncio
    async def test_stop_when_already_closed(self):
        session = NovaSonicSession()
        session._state = SessionState.CLOSED
        await session.stop()
        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_send_audio_when_not_active(self):
        session = NovaSonicSession()
        await session.send_audio(b"\x00" * 100)
        assert session.metrics.audio_chunks_sent == 0

    def test_metrics_increment(self):
        metrics = SessionMetrics()
        metrics.audio_chunks_sent += 1
        metrics.audio_chunks_received += 5
        metrics.events_received += 10
        metrics.turns_completed += 2
        assert metrics.audio_chunks_sent == 1
        assert metrics.audio_chunks_received == 5
        assert metrics.events_received == 10
        assert metrics.turns_completed == 2


class TestConversationHistory:
    """Test conversation history tracking and session continuation."""

    def test_empty_history_initially(self):
        session = NovaSonicSession()
        assert session.history == []

    def test_history_returns_copy(self):
        session = NovaSonicSession()
        session._history.append(ConversationTurn(role="user", text="hi"))
        h = session.history
        h.append(ConversationTurn(role="assistant", text="extra"))
        assert len(session.history) == 1  # Original unchanged

    def test_flush_user_turn(self):
        session = NovaSonicSession()
        session._current_user_text = "Hello there"
        session._flush_current_turn()
        assert len(session._history) == 1
        assert session._history[0].role == "user"
        assert session._history[0].text == "Hello there"
        assert session._current_user_text == ""

    def test_flush_assistant_turn(self):
        session = NovaSonicSession()
        session._current_assistant_text = "Hi! How can I help?"
        session._flush_current_turn()
        assert len(session._history) == 1
        assert session._history[0].role == "assistant"
        assert session._history[0].text == "Hi! How can I help?"
        assert session._current_assistant_text == ""

    def test_flush_both_turns(self):
        session = NovaSonicSession()
        session._current_user_text = "Hello"
        session._current_assistant_text = "Hi there"
        session._flush_current_turn()
        assert len(session._history) == 2
        assert session._history[0].role == "user"
        assert session._history[1].role == "assistant"

    def test_flush_ignores_empty_text(self):
        session = NovaSonicSession()
        session._current_user_text = ""
        session._current_assistant_text = "  "
        session._flush_current_turn()
        assert len(session._history) == 0

    def test_flush_trims_whitespace(self):
        session = NovaSonicSession()
        session._current_user_text = "  hello  "
        session._flush_current_turn()
        assert session._history[0].text == "hello"

    def test_history_trimmed_to_max(self):
        session = NovaSonicSession()
        # Fill with more than MAX_HISTORY_TURNS * 2
        for i in range(MAX_HISTORY_TURNS * 3):
            session._history.append(ConversationTurn(role="user", text=f"msg {i}"))
        session._current_user_text = "overflow"
        session._flush_current_turn()
        assert len(session._history) <= MAX_HISTORY_TURNS * 2

    def test_continuation_prompt_no_history(self):
        session = NovaSonicSession()
        prompt = session._build_continuation_prompt()
        assert prompt == session.config.system_prompt

    def test_continuation_prompt_with_history(self):
        session = NovaSonicSession()
        session._history.append(ConversationTurn(role="user", text="What is Python?"))
        session._history.append(ConversationTurn(
            role="assistant",
            text="Python is a programming language.",
        ))
        prompt = session._build_continuation_prompt()
        assert "What is Python?" in prompt
        assert "Python is a programming language." in prompt
        assert "Session continuation" in prompt
        assert session.config.system_prompt in prompt

    def test_continuation_prompt_limits_to_recent(self):
        session = NovaSonicSession()
        for i in range(MAX_HISTORY_TURNS + 5):
            session._history.append(ConversationTurn(role="user", text=f"msg {i}"))
        prompt = session._build_continuation_prompt()
        # Should only include the last MAX_HISTORY_TURNS entries
        assert f"msg {MAX_HISTORY_TURNS + 4}" in prompt
        assert "msg 0" not in prompt


class TestCallbacks:
    def test_on_text_callback_stored(self):
        cb = MagicMock()
        session = NovaSonicSession(on_text=cb)
        assert session._on_text is cb

    def test_on_audio_callback_stored(self):
        cb = MagicMock()
        session = NovaSonicSession(on_audio=cb)
        assert session._on_audio is cb

    def test_on_reconnect_callback_stored(self):
        cb = MagicMock()
        session = NovaSonicSession(on_reconnect=cb)
        assert session._on_reconnect is cb

    @pytest.mark.asyncio
    async def test_start_rejects_wrong_state(self):
        session = NovaSonicSession()
        session._state = SessionState.ACTIVE
        await session.start()
        # Should stay in ACTIVE, not attempt to reconnect
        assert session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_reconnect_rejects_wrong_state(self):
        session = NovaSonicSession()
        # IDLE state cannot reconnect
        await session.reconnect()
        assert session.state == SessionState.IDLE


class TestReconnectEvent:
    def test_reconnect_event_exists(self):
        session = NovaSonicSession()
        assert isinstance(session.reconnect_event, asyncio.Event)
        assert not session.reconnect_event.is_set()
