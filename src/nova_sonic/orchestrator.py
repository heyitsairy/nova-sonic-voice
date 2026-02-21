"""Nova-calls-Airy orchestrator.

Watches Nova 2 Sonic's text output for <airy>...</airy> tags and dispatches
them to a Claude agent backend. When Claude responds, the result is injected
back into the Nova session via forced reconnect.

Nova is the planner and prompter. Claude (Airy) is the brain.

Architecture:
    User speaks
        ↓
    Nova 2 Sonic (real-time STT + conversational responses)
        ↓ emits <airy>prompt</airy> in text output
    AiryOrchestrator (tag parser + dispatcher)
        ↓ sends prompt to Claude
    Anthropic API (Claude with Airy's personality)
        ↓ returns response
    AiryOrchestrator (injects result into session)
        ↓ forced reconnect with result in conversation history
    Nova 2 Sonic (speaks the enriched response)
        ↓
    User hears
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from nova_sonic.session import (
    ConversationTurn,
    NovaSonicConfig,
    NovaSonicSession,
    TextCallback,
)

logger = logging.getLogger(__name__)

# Tag pattern for Nova calling Airy
AIRY_TAG_PATTERN = re.compile(r"<airy>(.*?)</airy>", re.DOTALL)

# Partial tag detection (tag opened but not yet closed)
AIRY_TAG_OPEN = "<airy>"
AIRY_TAG_CLOSE = "</airy>"

# Default Claude model
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Default Airy system prompt for Claude
DEFAULT_AIRY_SYSTEM_PROMPT = (
    "You are Airy, an AI agent with persistent memory and real-time awareness. "
    "You are being called by a voice interface (Nova 2 Sonic) that handles "
    "real-time conversation. When called, you receive a specific request and "
    "should respond concisely (1-3 sentences) since your response will be "
    "spoken aloud. Be natural, warm, and direct."
)


@dataclass
class OrchestratorConfig:
    """Configuration for the Airy orchestrator."""

    claude_model: str = DEFAULT_CLAUDE_MODEL
    airy_system_prompt: str = DEFAULT_AIRY_SYSTEM_PROMPT
    max_tokens: int = 256
    dispatch_timeout: float = 15.0  # Max seconds to wait for Claude


@dataclass
class DispatchResult:
    """Result of dispatching a prompt to Claude."""

    prompt: str
    response: str
    latency_ms: float
    success: bool
    error: str | None = None


# Callback for when Airy responds
AiryResponseCallback = Callable[[DispatchResult], None]


class AiryOrchestrator:
    """Watches Nova's text output for <airy> tags and dispatches to Claude.

    Sits between the Nova session and the consumer, intercepting assistant
    text that contains <airy>...</airy> tags. When a complete tag is found,
    the enclosed prompt is sent to Claude. The response is injected back
    into Nova's conversation via forced session reconnect.

    Example::

        orchestrator = AiryOrchestrator(
            session=nova_session,
            config=OrchestratorConfig(),
        )
        orchestrator.start()
        # ... Nova talks, emits <airy> tags, Claude responds, Nova speaks result
        orchestrator.stop()
    """

    def __init__(
        self,
        session: NovaSonicSession,
        config: OrchestratorConfig | None = None,
        on_airy_response: AiryResponseCallback | None = None,
        on_text: TextCallback | None = None,
    ):
        self._session = session
        self._config = config or OrchestratorConfig()
        self._on_airy_response = on_airy_response
        self._on_text_passthrough = on_text  # Forward non-tag text to consumer

        # Claude client (initialized lazily)
        self._claude_client: anthropic.AsyncAnthropic | None = None

        # Tag accumulation state
        self._tag_buffer = ""
        self._in_tag = False

        # Dispatch tracking
        self._dispatches: list[DispatchResult] = []
        self._active_dispatch: asyncio.Task | None = None

        # Wire into session's text callback
        self._original_on_text = session._on_text
        session._on_text = self._intercept_text

    @property
    def dispatches(self) -> list[DispatchResult]:
        """History of all dispatches to Claude."""
        return list(self._dispatches)

    @property
    def dispatch_count(self) -> int:
        return len(self._dispatches)

    def _get_claude_client(self) -> anthropic.AsyncAnthropic:
        """Lazy-init the Claude client."""
        if self._claude_client is None:
            self._claude_client = anthropic.AsyncAnthropic()
        return self._claude_client

    def _intercept_text(self, role: str, text: str) -> None:
        """Intercept text callbacks from Nova to detect <airy> tags.

        For user text: pass through directly.
        For assistant text: scan for <airy> tags, accumulate if partial,
        dispatch when complete. Pass through non-tag text to consumer.
        """
        if role == "user":
            # User text always passes through
            if self._on_text_passthrough:
                self._on_text_passthrough(role, text)
            if self._original_on_text:
                self._original_on_text(role, text)
            return

        # Assistant text: check for tags
        if self._in_tag:
            # We're inside an open tag, accumulate
            if AIRY_TAG_CLOSE in text:
                # Tag closes in this chunk
                close_idx = text.index(AIRY_TAG_CLOSE)
                self._tag_buffer += text[:close_idx]
                remaining = text[close_idx + len(AIRY_TAG_CLOSE):]
                self._in_tag = False

                # Dispatch the complete tag
                prompt = self._tag_buffer.strip()
                self._tag_buffer = ""
                if prompt:
                    logger.info("Airy tag detected: %s", prompt[:100])
                    self._dispatch_to_claude(prompt)

                # Pass through any remaining text after the tag
                if remaining.strip():
                    if self._on_text_passthrough:
                        self._on_text_passthrough(role, remaining)
            else:
                # Tag still open, keep accumulating
                self._tag_buffer += text
            return

        # Not in a tag: check if one opens
        if AIRY_TAG_OPEN in text:
            open_idx = text.index(AIRY_TAG_OPEN)

            # Pass through text before the tag
            before = text[:open_idx]
            if before.strip():
                if self._on_text_passthrough:
                    self._on_text_passthrough(role, before)

            after = text[open_idx + len(AIRY_TAG_OPEN):]

            if AIRY_TAG_CLOSE in after:
                # Complete tag in one chunk
                close_idx = after.index(AIRY_TAG_CLOSE)
                prompt = after[:close_idx].strip()
                remaining = after[close_idx + len(AIRY_TAG_CLOSE):]

                if prompt:
                    logger.info("Airy tag detected (single chunk): %s", prompt[:100])
                    self._dispatch_to_claude(prompt)

                if remaining.strip():
                    if self._on_text_passthrough:
                        self._on_text_passthrough(role, remaining)
            else:
                # Tag opened but not closed yet
                self._in_tag = True
                self._tag_buffer = after
        else:
            # No tags, pass through entirely
            # Check for complete tags via regex (handles edge cases)
            matches = list(AIRY_TAG_PATTERN.finditer(text))
            if matches:
                last_end = 0
                for match in matches:
                    # Pass through text before this match
                    before = text[last_end:match.start()]
                    if before.strip():
                        if self._on_text_passthrough:
                            self._on_text_passthrough(role, before)

                    prompt = match.group(1).strip()
                    if prompt:
                        logger.info("Airy tag detected (regex): %s", prompt[:100])
                        self._dispatch_to_claude(prompt)

                    last_end = match.end()

                # Pass through text after last match
                remaining = text[last_end:]
                if remaining.strip():
                    if self._on_text_passthrough:
                        self._on_text_passthrough(role, remaining)
            else:
                # Plain text, pass through
                if self._on_text_passthrough:
                    self._on_text_passthrough(role, text)

    def _dispatch_to_claude(self, prompt: str) -> None:
        """Fire an async dispatch to Claude with the given prompt."""
        # Cancel any in-flight dispatch
        if self._active_dispatch and not self._active_dispatch.done():
            self._active_dispatch.cancel()

        self._active_dispatch = asyncio.create_task(
            self._call_claude_and_inject(prompt)
        )

    async def _call_claude_and_inject(self, prompt: str) -> None:
        """Call Claude with the prompt and inject the response into Nova."""
        start = time.time()
        result = DispatchResult(
            prompt=prompt,
            response="",
            latency_ms=0,
            success=False,
        )

        try:
            client = self._get_claude_client()

            response = await asyncio.wait_for(
                client.messages.create(
                    model=self._config.claude_model,
                    max_tokens=self._config.max_tokens,
                    system=self._config.airy_system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._config.dispatch_timeout,
            )

            result.response = response.content[0].text
            result.success = True
            result.latency_ms = (time.time() - start) * 1000

            logger.info(
                "Claude responded (%.0fms): %s",
                result.latency_ms,
                result.response[:100],
            )

            # Inject the response into Nova's conversation history
            await self._inject_response(prompt, result.response)

        except asyncio.TimeoutError:
            result.error = f"Timeout after {self._config.dispatch_timeout}s"
            result.latency_ms = (time.time() - start) * 1000
            logger.warning("Claude dispatch timed out for: %s", prompt[:100])

        except Exception as e:
            result.error = str(e)
            result.latency_ms = (time.time() - start) * 1000
            logger.exception("Claude dispatch failed for: %s", prompt[:100])

        self._dispatches.append(result)

        if self._on_airy_response:
            self._on_airy_response(result)

    async def _inject_response(self, prompt: str, response: str) -> None:
        """Inject Claude's response back into Nova via forced reconnect.

        Adds a synthetic conversation exchange to the session history:
        - User turn: "[Airy was asked: {prompt}]"
        - Assistant turn: "[Airy responded: {response}]"

        Then forces a session reconnect so Nova picks up the context.
        """
        # Add the exchange to session history
        self._session._history.append(ConversationTurn(
            role="user",
            text=f"[Airy was asked: {prompt}]",
            timestamp=time.time(),
        ))
        self._session._history.append(ConversationTurn(
            role="assistant",
            text=f"[Airy responded: {response}]",
            timestamp=time.time(),
        ))

        # Force reconnect so Nova picks up the enriched context
        logger.info("Injecting Airy response and forcing reconnect...")
        await self._session.reconnect()

    def reset(self) -> None:
        """Reset tag parsing state."""
        self._tag_buffer = ""
        self._in_tag = False

    def stop(self) -> None:
        """Stop the orchestrator and cancel any active dispatch."""
        if self._active_dispatch and not self._active_dispatch.done():
            self._active_dispatch.cancel()
        self.reset()


def build_nova_system_prompt(
    base_personality: str = "",
    available_capabilities: list[str] | None = None,
) -> str:
    """Build Nova's system prompt that teaches it to call Airy.

    Args:
        base_personality: Optional personality traits for Nova's conversational style.
        available_capabilities: List of things Airy can do (for Nova to know what to ask).

    Returns:
        System prompt string for the Nova session.
    """
    capabilities = available_capabilities or [
        "search memory and past conversations",
        "look up factual knowledge",
        "perform web searches",
        "run code and analyze data",
        "access calendar and email",
        "remember things for later",
    ]

    cap_list = "\n".join(f"  - {cap}" for cap in capabilities)

    personality = base_personality or (
        "You are a friendly, natural conversational voice interface. "
        "Keep your own responses brief and warm."
    )

    return f"""{personality}

You have access to a powerful AI agent called Airy through a special tag system.
When you need to think deeper, access memory, search for information, or take any
action beyond simple conversation, use this exact format:

<airy>your specific request here</airy>

Examples:
- User asks "What did we talk about yesterday?" → <airy>search memory for yesterday's conversation topics</airy>
- User asks "What's the weather?" → <airy>search the web for current weather in Scarborough, Ontario</airy>
- User asks a complex question → <airy>the user is asking about [topic], provide a detailed answer</airy>

Airy can:
{cap_list}

Rules:
- For simple greetings and casual chat, respond directly without calling Airy
- When you call Airy, say something natural like "Let me think about that" or "One moment" first
- When Airy's response appears in the conversation history as "[Airy responded: ...]",
  incorporate that information naturally into your next spoken response
- Never read the <airy> tags or bracket notation aloud
- Stay conversational and natural at all times"""
