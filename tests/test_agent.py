"""Tests for the Nova Sonic voice agent."""

from unittest.mock import MagicMock, patch

from nova_sonic.agent import NovaSonicVoiceAgent
from nova_sonic.session import NovaSonicConfig


class TestNovaSonicVoiceAgent:
    def test_initial_state(self):
        agent = NovaSonicVoiceAgent()
        assert not agent.is_running

    def test_custom_config(self):
        config = NovaSonicConfig(voice_id="ruth")
        agent = NovaSonicVoiceAgent(config=config)
        assert agent.config.voice_id == "ruth"

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
