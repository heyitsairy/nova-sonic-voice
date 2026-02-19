"""Tests for the Nova Sonic voice agent."""

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from nova_sonic.agent import NovaSonicVoiceAgent
from nova_sonic.session import ConversationTurn, NovaSonicConfig, NovaSonicSession


class TestNovaSonicVoiceAgent:
    def test_initial_state(self):
        agent = NovaSonicVoiceAgent()
        assert not agent.is_running
        assert agent.session is None
        assert agent.elapsed_seconds == 0.0

    def test_custom_config(self):
        config = NovaSonicConfig(voice_id="ruth")
        agent = NovaSonicVoiceAgent(config=config)
        assert agent.config.voice_id == "ruth"

    def test_output_device_index(self):
        agent = NovaSonicVoiceAgent(output_device_index=5)
        assert agent._output_device_index == 5

    def test_callbacks_stored(self):
        user_cb = MagicMock()
        assistant_cb = MagicMock()
        agent = NovaSonicVoiceAgent(
            on_user_text=user_cb,
            on_assistant_text=assistant_cb,
        )
        assert agent._on_user_text is user_cb
        assert agent._on_assistant_text is assistant_cb

    def test_text_callback_routing(self):
        user_cb = MagicMock()
        assistant_cb = MagicMock()
        agent = NovaSonicVoiceAgent(
            on_user_text=user_cb,
            on_assistant_text=assistant_cb,
        )
        agent._text_callback("user", "hello")
        user_cb.assert_called_once_with("user", "hello")
        assistant_cb.assert_not_called()

        agent._text_callback("assistant", "hi there")
        assistant_cb.assert_called_once_with("assistant", "hi there")

    def test_text_callback_no_callback_set(self):
        agent = NovaSonicVoiceAgent()
        # Should not raise
        agent._text_callback("user", "hello")
        agent._text_callback("assistant", "hi")

    def test_reconnect_callback(self):
        agent = NovaSonicVoiceAgent()
        # Should not raise
        agent._reconnect_callback()

    def test_get_transcript_no_session(self):
        agent = NovaSonicVoiceAgent()
        assert agent.get_transcript() == []

    def test_get_transcript_with_session(self):
        agent = NovaSonicVoiceAgent()
        session = NovaSonicSession()
        session._history.append(ConversationTurn(role="user", text="hello"))
        session._history.append(ConversationTurn(role="assistant", text="hi"))
        agent._session = session
        transcript = agent.get_transcript()
        assert len(transcript) == 2
        assert transcript[0].role == "user"
        assert transcript[1].role == "assistant"
