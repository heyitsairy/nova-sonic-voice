"""Tests for the Nova-calls-Airy orchestrator."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nova_sonic.orchestrator import (
    AIRY_TAG_CLOSE,
    AIRY_TAG_OPEN,
    AIRY_TAG_PATTERN,
    AiryOrchestrator,
    DispatchResult,
    OrchestratorConfig,
    build_nova_system_prompt,
)
from nova_sonic.session import ConversationTurn, NovaSonicSession


# --- Fixtures ---


@pytest.fixture
def mock_session():
    """Create a mock NovaSonicSession."""
    session = MagicMock(spec=NovaSonicSession)
    session._on_text = None
    session._history = []
    session.reconnect = AsyncMock()
    session.is_active = True
    return session


@pytest.fixture
def config():
    return OrchestratorConfig(
        webhook_url="https://discord.com/api/webhooks/123/abc",
        bot_token="fake-bot-token",
        channel_id="999888777",
        dispatch_timeout=5.0,
    )


@pytest.fixture
def orchestrator(mock_session, config):
    return AiryOrchestrator(
        session=mock_session,
        config=config,
    )


def _mock_http_response(status=200, json_data=None, text_data=""):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text_data)
    # Make it work as async context manager
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


# --- Tag Pattern Tests ---


class TestTagPattern:
    """Test the regex pattern for <airy> tags."""

    def test_simple_tag(self):
        text = "<airy>search memory</airy>"
        match = AIRY_TAG_PATTERN.search(text)
        assert match is not None
        assert match.group(1) == "search memory"

    def test_tag_with_surrounding_text(self):
        text = "Let me think about that. <airy>look up weather</airy> One moment."
        match = AIRY_TAG_PATTERN.search(text)
        assert match is not None
        assert match.group(1) == "look up weather"

    def test_multiline_tag(self):
        text = "<airy>search for\nrecent conversations</airy>"
        match = AIRY_TAG_PATTERN.search(text)
        assert match is not None
        assert match.group(1) == "search for\nrecent conversations"

    def test_multiple_tags(self):
        text = "<airy>first</airy> then <airy>second</airy>"
        matches = list(AIRY_TAG_PATTERN.finditer(text))
        assert len(matches) == 2
        assert matches[0].group(1) == "first"
        assert matches[1].group(1) == "second"

    def test_no_tag(self):
        text = "Just a normal response with no tags."
        match = AIRY_TAG_PATTERN.search(text)
        assert match is None

    def test_empty_tag(self):
        text = "<airy></airy>"
        match = AIRY_TAG_PATTERN.search(text)
        assert match is not None
        assert match.group(1) == ""


# --- Text Interception Tests ---


class TestTextInterception:
    """Test the orchestrator's text callback interception."""

    def test_user_text_passes_through(self, orchestrator):
        """User text should always pass through without interception."""
        received = []
        orchestrator._on_text_passthrough = lambda role, text: received.append((role, text))

        orchestrator._intercept_text("user", "Hello there")
        assert ("user", "Hello there") in received

    def test_plain_assistant_text_passes_through(self, orchestrator):
        """Assistant text without tags passes through."""
        received = []
        orchestrator._on_text_passthrough = lambda role, text: received.append((role, text))

        orchestrator._intercept_text("assistant", "Hello! How can I help?")
        assert ("assistant", "Hello! How can I help?") in received

    def test_complete_tag_dispatches(self, orchestrator):
        """A complete <airy> tag should trigger dispatch."""
        with patch.object(orchestrator, "_dispatch_to_airy") as mock_dispatch:
            orchestrator._intercept_text(
                "assistant",
                "Let me check. <airy>search memory for yesterday</airy>",
            )
            mock_dispatch.assert_called_once_with("search memory for yesterday")

    def test_text_before_tag_passes_through(self, orchestrator):
        """Text before an <airy> tag should pass through."""
        received = []
        orchestrator._on_text_passthrough = lambda role, text: received.append((role, text))

        with patch.object(orchestrator, "_dispatch_to_airy"):
            orchestrator._intercept_text(
                "assistant",
                "Let me check. <airy>search</airy>",
            )
        assert any("Let me check." in text for _, text in received)

    def test_text_after_tag_passes_through(self, orchestrator):
        """Text after a closed <airy> tag should pass through."""
        received = []
        orchestrator._on_text_passthrough = lambda role, text: received.append((role, text))

        with patch.object(orchestrator, "_dispatch_to_airy"):
            orchestrator._intercept_text(
                "assistant",
                "<airy>search</airy> One moment please.",
            )
        assert any("One moment please." in text for _, text in received)

    def test_partial_tag_accumulates(self, orchestrator):
        """A tag split across chunks should accumulate."""
        with patch.object(orchestrator, "_dispatch_to_airy") as mock_dispatch:
            # First chunk opens the tag
            orchestrator._intercept_text("assistant", "Hmm. <airy>search memory")
            assert orchestrator._in_tag is True
            mock_dispatch.assert_not_called()

            # Second chunk closes it
            orchestrator._intercept_text("assistant", " for recent topics</airy>")
            assert orchestrator._in_tag is False
            mock_dispatch.assert_called_once_with("search memory for recent topics")

    def test_partial_tag_across_three_chunks(self, orchestrator):
        """Tag content spread across three chunks."""
        with patch.object(orchestrator, "_dispatch_to_airy") as mock_dispatch:
            orchestrator._intercept_text("assistant", "<airy>first ")
            assert orchestrator._in_tag is True

            orchestrator._intercept_text("assistant", "second ")
            assert orchestrator._in_tag is True

            orchestrator._intercept_text("assistant", "third</airy>")
            assert orchestrator._in_tag is False
            mock_dispatch.assert_called_once_with("first second third")

    def test_empty_tag_not_dispatched(self, orchestrator):
        """An empty <airy></airy> tag should not dispatch."""
        with patch.object(orchestrator, "_dispatch_to_airy") as mock_dispatch:
            orchestrator._intercept_text("assistant", "<airy></airy>")
            mock_dispatch.assert_not_called()

    def test_reset_clears_state(self, orchestrator):
        """Reset should clear accumulated tag state."""
        orchestrator._in_tag = True
        orchestrator._tag_buffer = "partial content"
        orchestrator.reset()
        assert orchestrator._in_tag is False
        assert orchestrator._tag_buffer == ""


# --- Discord Dispatch Tests ---


class TestDiscordDispatch:
    """Test dispatching prompts to Airy via Discord."""

    @pytest.mark.asyncio
    async def test_successful_dispatch(self, orchestrator, mock_session):
        """Successful Discord dispatch should inject response and reconnect."""
        # Mock HTTP session
        mock_http = AsyncMock()

        # Mock webhook post
        webhook_resp = _mock_http_response(200, {"id": "msg_123"})
        # Mock thread poll (Airy's reply found on first check)
        thread_resp = _mock_http_response(200, [
            {"author": {"id": "111222333"}, "content": "The weather is sunny."}
        ])

        mock_http.get = MagicMock(return_value=thread_resp)
        mock_http.post = MagicMock(return_value=webhook_resp)
        mock_http.closed = False

        orchestrator._http = mock_http
        orchestrator._bot_user_id = "111222333"  # Skip resolution

        await orchestrator._post_and_poll("what is the weather")

        # Should have added two history entries
        assert len(mock_session._history) == 2
        assert "what is the weather" in mock_session._history[0].text
        assert "The weather is sunny." in mock_session._history[1].text

        # Should have triggered reconnect
        mock_session.reconnect.assert_awaited_once()

        # Should have recorded the dispatch
        assert len(orchestrator._dispatches) == 1
        assert orchestrator._dispatches[0].success is True
        assert orchestrator._dispatches[0].response == "The weather is sunny."

    @pytest.mark.asyncio
    async def test_webhook_post_failure(self, orchestrator, mock_session):
        """Failed webhook post should be recorded as a failed dispatch."""
        mock_http = AsyncMock()
        webhook_resp = _mock_http_response(500, text_data="Internal Server Error")
        mock_http.post = MagicMock(return_value=webhook_resp)
        mock_http.closed = False

        orchestrator._http = mock_http
        orchestrator._bot_user_id = "111222333"

        await orchestrator._post_and_poll("test prompt")

        assert len(orchestrator._dispatches) == 1
        assert orchestrator._dispatches[0].success is False
        assert "Webhook post failed" in orchestrator._dispatches[0].error
        mock_session.reconnect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_timeout(self, orchestrator, mock_session):
        """Timeout should be recorded as a failed dispatch."""
        orchestrator._config.dispatch_timeout = 0.2
        orchestrator._config.poll_interval = 0.1

        mock_http = AsyncMock()
        webhook_resp = _mock_http_response(200, {"id": "msg_123"})
        # Thread returns empty (no reply) and channel returns empty
        empty_resp = _mock_http_response(200, [])
        mock_http.post = MagicMock(return_value=webhook_resp)
        mock_http.get = MagicMock(return_value=empty_resp)
        mock_http.closed = False

        orchestrator._http = mock_http
        orchestrator._bot_user_id = "111222333"

        await orchestrator._post_and_poll("slow question")

        assert len(orchestrator._dispatches) == 1
        assert orchestrator._dispatches[0].success is False
        assert "No reply" in orchestrator._dispatches[0].error
        mock_session.reconnect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_response_callback_fires(self, orchestrator, mock_session):
        """The on_airy_response callback should fire after dispatch."""
        callback_results = []
        orchestrator._on_airy_response = lambda r: callback_results.append(r)

        mock_http = AsyncMock()
        webhook_resp = _mock_http_response(200, {"id": "msg_123"})
        thread_resp = _mock_http_response(200, [
            {"author": {"id": "111222333"}, "content": "Result"}
        ])
        mock_http.post = MagicMock(return_value=webhook_resp)
        mock_http.get = MagicMock(return_value=thread_resp)
        mock_http.closed = False

        orchestrator._http = mock_http
        orchestrator._bot_user_id = "111222333"

        await orchestrator._post_and_poll("test")

        assert len(callback_results) == 1
        assert callback_results[0].success is True
        assert callback_results[0].response == "Result"

    @pytest.mark.asyncio
    async def test_channel_fallback(self, orchestrator, mock_session):
        """If no thread exists, should check channel messages."""
        mock_http = AsyncMock()
        webhook_resp = _mock_http_response(200, {"id": "msg_123"})
        # Thread 404 (no thread exists)
        thread_resp = _mock_http_response(404)
        # Channel has the reply
        channel_resp = _mock_http_response(200, [
            {"author": {"id": "111222333"}, "content": "Found in channel"}
        ])
        mock_http.post = MagicMock(return_value=webhook_resp)
        mock_http.get = MagicMock(side_effect=[thread_resp, channel_resp])
        mock_http.closed = False

        orchestrator._http = mock_http
        orchestrator._bot_user_id = "111222333"

        await orchestrator._post_and_poll("find this")

        assert len(orchestrator._dispatches) == 1
        assert orchestrator._dispatches[0].success is True
        assert orchestrator._dispatches[0].response == "Found in channel"

    @pytest.mark.asyncio
    async def test_dispatch_not_configured(self, mock_session):
        """Dispatch without config should fail immediately."""
        orch = AiryOrchestrator(
            session=mock_session,
            config=OrchestratorConfig(
                webhook_url="",
                bot_token="",
                channel_id="",
            ),
        )
        orch._dispatch_to_airy("test prompt")

        assert len(orch._dispatches) == 1
        assert orch._dispatches[0].success is False
        assert "config incomplete" in orch._dispatches[0].error


# --- Webhook Post Tests ---


class TestWebhookPost:
    """Test posting to Discord webhook."""

    @pytest.mark.asyncio
    async def test_webhook_post_content(self, orchestrator):
        """Webhook post should include the prompt and Nova Voice username."""
        mock_http = AsyncMock()
        webhook_resp = _mock_http_response(200, {"id": "msg_456"})
        mock_http.post = MagicMock(return_value=webhook_resp)
        mock_http.closed = False
        orchestrator._http = mock_http

        result = await orchestrator._post_webhook(mock_http, "search for weather")

        assert result == "msg_456"
        # Verify the webhook was called with the right payload
        mock_http.post.assert_called_once()
        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["content"] == "search for weather"
        assert payload["username"] == "Nova Voice"


# --- Response Injection Tests ---


class TestResponseInjection:
    """Test injecting Airy's response back into Nova."""

    @pytest.mark.asyncio
    async def test_injection_adds_history(self, orchestrator, mock_session):
        """Injection should add user and assistant turns to history."""
        await orchestrator._inject_response("what time is it", "It's 5 PM")

        assert len(mock_session._history) == 2
        assert mock_session._history[0].role == "user"
        assert "[Airy was asked: what time is it]" in mock_session._history[0].text
        assert mock_session._history[1].role == "assistant"
        assert "[Airy responded: It's 5 PM]" in mock_session._history[1].text

    @pytest.mark.asyncio
    async def test_injection_triggers_reconnect(self, orchestrator, mock_session):
        """Injection should force a session reconnect."""
        await orchestrator._inject_response("test", "result")
        mock_session.reconnect.assert_awaited_once()


# --- Config Tests ---


class TestConfig:
    """Test OrchestratorConfig."""

    def test_env_var_fallback(self):
        """Config should fall back to environment variables."""
        with patch.dict("os.environ", {
            "NOVA_WEBHOOK_URL": "https://discord.com/api/webhooks/test",
            "DISCORD_TOKEN": "env-token",
            "NOVA_CHANNEL_ID": "env-channel",
        }):
            config = OrchestratorConfig()
            assert config.webhook_url == "https://discord.com/api/webhooks/test"
            assert config.bot_token == "env-token"
            assert config.channel_id == "env-channel"
            assert config.is_configured is True

    def test_explicit_values_override_env(self):
        """Explicit constructor values should override env vars."""
        with patch.dict("os.environ", {
            "NOVA_WEBHOOK_URL": "from-env",
        }):
            config = OrchestratorConfig(
                webhook_url="explicit",
                bot_token="tok",
                channel_id="ch",
            )
            assert config.webhook_url == "explicit"

    def test_not_configured_when_missing(self):
        """is_configured should be False when any value is missing."""
        with patch.dict("os.environ", {}, clear=True):
            config = OrchestratorConfig(
                webhook_url="",
                bot_token="",
                channel_id="",
            )
            assert config.is_configured is False


# --- System Prompt Tests ---


class TestSystemPrompt:
    """Test the Nova system prompt builder."""

    def test_default_prompt(self):
        prompt = build_nova_system_prompt()
        assert "<airy>" in prompt
        assert "</airy>" in prompt
        assert "search memory" in prompt

    def test_custom_personality(self):
        prompt = build_nova_system_prompt(base_personality="You are a pirate.")
        assert "You are a pirate." in prompt
        assert "<airy>" in prompt

    def test_custom_capabilities(self):
        prompt = build_nova_system_prompt(
            available_capabilities=["fly a plane", "cook pasta"]
        )
        assert "fly a plane" in prompt
        assert "cook pasta" in prompt

    def test_prompt_includes_rules(self):
        prompt = build_nova_system_prompt()
        assert "simple greetings" in prompt.lower() or "casual chat" in prompt.lower()
        assert "naturally" in prompt.lower()


# --- Integration Tests ---


class TestOrchestratorWiring:
    """Test that the orchestrator wires into the session correctly."""

    def test_wires_into_session_on_text(self, mock_session, config):
        """Creating an orchestrator should replace the session's on_text."""
        original = MagicMock()
        mock_session._on_text = original

        orch = AiryOrchestrator(session=mock_session, config=config)

        # The session's callback should now be the interceptor
        assert mock_session._on_text == orch._intercept_text
        # The original should be preserved
        assert orch._original_on_text == original

    def test_stop_cancels_active_dispatch(self, orchestrator):
        """Stop should cancel any in-flight dispatch."""
        mock_task = MagicMock()
        mock_task.done.return_value = False
        orchestrator._active_dispatch = mock_task

        orchestrator.stop()
        mock_task.cancel.assert_called_once()

    def test_dispatch_count(self, orchestrator):
        """Dispatch count tracks completed dispatches."""
        assert orchestrator.dispatch_count == 0
        orchestrator._dispatches.append(
            DispatchResult(prompt="test", response="ok", latency_ms=100, success=True)
        )
        assert orchestrator.dispatch_count == 1


# --- Bot User ID Resolution Tests ---


class TestBotUserResolution:
    """Test resolving the bot's user ID from the token."""

    @pytest.mark.asyncio
    async def test_resolves_from_api(self, orchestrator):
        """Should resolve bot user ID from /users/@me."""
        mock_http = AsyncMock()
        user_resp = _mock_http_response(200, {"id": "bot_12345"})
        mock_http.get = MagicMock(return_value=user_resp)
        mock_http.closed = False
        orchestrator._http = mock_http

        user_id = await orchestrator._resolve_bot_user_id()
        assert user_id == "bot_12345"
        assert orchestrator._bot_user_id == "bot_12345"

    @pytest.mark.asyncio
    async def test_caches_user_id(self, orchestrator):
        """Should cache the bot user ID after first resolution."""
        orchestrator._bot_user_id = "cached_id"
        result = await orchestrator._resolve_bot_user_id()
        assert result == "cached_id"
