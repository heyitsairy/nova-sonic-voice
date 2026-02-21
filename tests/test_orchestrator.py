"""Tests for the Nova-calls-Airy orchestrator."""

from __future__ import annotations

import asyncio
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
from nova_sonic.session import ConversationTurn, NovaSonicConfig, NovaSonicSession


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
        claude_model="claude-sonnet-4-20250514",
        dispatch_timeout=5.0,
    )


@pytest.fixture
def orchestrator(mock_session, config):
    return AiryOrchestrator(
        session=mock_session,
        config=config,
    )


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
        with patch.object(orchestrator, "_dispatch_to_claude") as mock_dispatch:
            orchestrator._intercept_text(
                "assistant",
                "Let me check. <airy>search memory for yesterday</airy>",
            )
            mock_dispatch.assert_called_once_with("search memory for yesterday")

    def test_text_before_tag_passes_through(self, orchestrator):
        """Text before an <airy> tag should pass through."""
        received = []
        orchestrator._on_text_passthrough = lambda role, text: received.append((role, text))

        with patch.object(orchestrator, "_dispatch_to_claude"):
            orchestrator._intercept_text(
                "assistant",
                "Let me check. <airy>search</airy>",
            )
        assert any("Let me check." in text for _, text in received)

    def test_text_after_tag_passes_through(self, orchestrator):
        """Text after a closed <airy> tag should pass through."""
        received = []
        orchestrator._on_text_passthrough = lambda role, text: received.append((role, text))

        with patch.object(orchestrator, "_dispatch_to_claude"):
            orchestrator._intercept_text(
                "assistant",
                "<airy>search</airy> One moment please.",
            )
        assert any("One moment please." in text for _, text in received)

    def test_partial_tag_accumulates(self, orchestrator):
        """A tag split across chunks should accumulate."""
        with patch.object(orchestrator, "_dispatch_to_claude") as mock_dispatch:
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
        with patch.object(orchestrator, "_dispatch_to_claude") as mock_dispatch:
            orchestrator._intercept_text("assistant", "<airy>first ")
            assert orchestrator._in_tag is True

            orchestrator._intercept_text("assistant", "second ")
            assert orchestrator._in_tag is True

            orchestrator._intercept_text("assistant", "third</airy>")
            assert orchestrator._in_tag is False
            mock_dispatch.assert_called_once_with("first second third")

    def test_empty_tag_not_dispatched(self, orchestrator):
        """An empty <airy></airy> tag should not dispatch."""
        with patch.object(orchestrator, "_dispatch_to_claude") as mock_dispatch:
            orchestrator._intercept_text("assistant", "<airy></airy>")
            mock_dispatch.assert_not_called()

    def test_reset_clears_state(self, orchestrator):
        """Reset should clear accumulated tag state."""
        orchestrator._in_tag = True
        orchestrator._tag_buffer = "partial content"
        orchestrator.reset()
        assert orchestrator._in_tag is False
        assert orchestrator._tag_buffer == ""


# --- Claude Dispatch Tests ---


class TestClaudeDispatch:
    """Test dispatching prompts to Claude."""

    @pytest.mark.asyncio
    async def test_successful_dispatch(self, orchestrator, mock_session):
        """Successful Claude call should inject response and reconnect."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="The weather is sunny.")]

        with patch.object(orchestrator, "_get_claude_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            await orchestrator._call_claude_and_inject("what is the weather")

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
    async def test_dispatch_timeout(self, orchestrator, mock_session):
        """Timeout should be recorded as a failed dispatch."""
        orchestrator._config.dispatch_timeout = 0.1

        with patch.object(orchestrator, "_get_claude_client") as mock_client_fn:
            mock_client = AsyncMock()

            async def slow_create(**kwargs):
                await asyncio.sleep(10)

            mock_client.messages.create = slow_create
            mock_client_fn.return_value = mock_client

            await orchestrator._call_claude_and_inject("slow question")

        assert len(orchestrator._dispatches) == 1
        assert orchestrator._dispatches[0].success is False
        assert "Timeout" in orchestrator._dispatches[0].error

        # Should NOT have reconnected on failure
        mock_session.reconnect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_error(self, orchestrator, mock_session):
        """API error should be recorded as a failed dispatch."""
        with patch.object(orchestrator, "_get_claude_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(
                side_effect=Exception("API error")
            )
            mock_client_fn.return_value = mock_client

            await orchestrator._call_claude_and_inject("broken question")

        assert len(orchestrator._dispatches) == 1
        assert orchestrator._dispatches[0].success is False
        assert "API error" in orchestrator._dispatches[0].error
        mock_session.reconnect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_response_callback_fires(self, orchestrator, mock_session):
        """The on_airy_response callback should fire after dispatch."""
        callback_results = []
        orchestrator._on_airy_response = lambda r: callback_results.append(r)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Result")]

        with patch.object(orchestrator, "_get_claude_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_client_fn.return_value = mock_client

            await orchestrator._call_claude_and_inject("test")

        assert len(callback_results) == 1
        assert callback_results[0].success is True
        assert callback_results[0].response == "Result"


# --- Response Injection Tests ---


class TestResponseInjection:
    """Test injecting Claude's response back into Nova."""

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
