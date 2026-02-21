"""Nova-calls-Airy orchestrator.

Watches Nova 2 Sonic's text output for <airy>...</airy> tags and dispatches
them to Airy through Discord. When Airy responds, the result is injected
back into the Nova session via forced reconnect.

Nova is the voice. Discord is the wire. Airy is the brain.

Architecture:
    User speaks
        ↓
    Nova 2 Sonic (real-time STT + conversational responses)
        ↓ emits <airy>prompt</airy> in text output
    AiryOrchestrator (tag parser + dispatcher)
        ↓ posts prompt to Discord via webhook
    Airy (sees the message, processes with full context)
        ↓ replies in thread
    AiryOrchestrator (polls thread, picks up reply)
        ↓ injects result into session history + forced reconnect
    Nova 2 Sonic (speaks the enriched response)
        ↓
    User hears

Nova never blocks. Dispatch is fully async. Nova keeps talking while
Airy processes, and weaves the response in when it arrives.

Config via environment variables:
    NOVA_WEBHOOK_URL     Discord webhook URL (posts as "Nova Voice")
    DISCORD_TOKEN        Bot token for reading replies
    NOVA_CHANNEL_ID      Channel the webhook posts to
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

from nova_sonic.session import (
    ConversationTurn,
    NovaSonicSession,
    TextCallback,
)

logger = logging.getLogger(__name__)

# Tag pattern for Nova calling Airy
AIRY_TAG_PATTERN = re.compile(r"<airy>(.*?)</airy>", re.DOTALL)

# Partial tag detection (tag opened but not yet closed)
AIRY_TAG_OPEN = "<airy>"
AIRY_TAG_CLOSE = "</airy>"

# Discord API base
DISCORD_API = "https://discord.com/api/v10"


@dataclass
class OrchestratorConfig:
    """Configuration for the Airy orchestrator.

    All values can be set via environment variables. Constructor args
    override env vars.
    """

    # Discord webhook URL for posting (appears as "Nova Voice")
    webhook_url: str = ""
    # Bot token for reading Airy's replies
    bot_token: str = ""
    # Channel ID the webhook posts to
    channel_id: str = ""
    # How often to poll for Airy's reply (seconds)
    poll_interval: float = 1.0
    # Max time to wait for Airy's reply (seconds)
    dispatch_timeout: float = 30.0

    def __post_init__(self):
        if not self.webhook_url:
            self.webhook_url = os.environ.get("NOVA_WEBHOOK_URL", "")
        if not self.bot_token:
            self.bot_token = os.environ.get("DISCORD_TOKEN", "")
        if not self.channel_id:
            self.channel_id = os.environ.get("NOVA_CHANNEL_ID", "")

    @property
    def is_configured(self) -> bool:
        """True if all required Discord config is present."""
        return bool(self.webhook_url and self.bot_token and self.channel_id)


@dataclass
class DispatchResult:
    """Result of dispatching a prompt to Airy via Discord."""

    prompt: str
    response: str
    latency_ms: float
    success: bool
    error: str | None = None


# Callback for when Airy responds
AiryResponseCallback = Callable[[DispatchResult], None]


class AiryOrchestrator:
    """Watches Nova's text output for <airy> tags and dispatches to Airy via Discord.

    Sits between the Nova session and the consumer, intercepting assistant
    text that contains <airy>...</airy> tags. When a complete tag is found,
    the prompt is posted to Discord via webhook. Airy processes the message
    and replies. The orchestrator polls for the reply and injects it back
    into Nova's conversation via forced session reconnect.

    Nova never blocks. Dispatch is async. Nova keeps talking while waiting.

    Example::

        orchestrator = AiryOrchestrator(
            session=nova_session,
            config=OrchestratorConfig(
                webhook_url="https://discord.com/api/webhooks/...",
                bot_token="...",
                channel_id="...",
            ),
        )
        # Nova talks, emits <airy> tags, Airy responds via Discord, Nova speaks result
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

        # HTTP session (initialized lazily)
        self._http: aiohttp.ClientSession | None = None

        # Bot user ID (resolved lazily from token)
        self._bot_user_id: str | None = None

        # Tag accumulation state
        self._tag_buffer = ""
        self._in_tag = False

        # Dispatch tracking
        self._dispatches: list[DispatchResult] = []
        self._active_dispatch: asyncio.Task | None = None

        # Wire into session's text callback
        self._original_on_text = session._on_text
        session._on_text = self._intercept_text

        if not self._config.is_configured:
            logger.warning(
                "Orchestrator created but Discord config incomplete. "
                "Set NOVA_WEBHOOK_URL, DISCORD_TOKEN, NOVA_CHANNEL_ID."
            )

    @property
    def dispatches(self) -> list[DispatchResult]:
        """History of all dispatches to Airy."""
        return list(self._dispatches)

    @property
    def dispatch_count(self) -> int:
        return len(self._dispatches)

    async def _get_http(self) -> aiohttp.ClientSession:
        """Lazy-init the HTTP session."""
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for Discord dispatch. pip install aiohttp")
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(
                headers={"Authorization": f"Bot {self._config.bot_token}"},
            )
        return self._http

    async def _resolve_bot_user_id(self) -> str:
        """Get the bot's own user ID from the token."""
        if self._bot_user_id:
            return self._bot_user_id
        http = await self._get_http()
        async with http.get(f"{DISCORD_API}/users/@me") as resp:
            if resp.status == 200:
                data = await resp.json()
                self._bot_user_id = data["id"]
                logger.info("Resolved bot user ID: %s", self._bot_user_id)
                return self._bot_user_id
            else:
                raise RuntimeError(f"Failed to resolve bot user ID: {resp.status}")

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
                    self._dispatch_to_airy(prompt)

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
                    self._dispatch_to_airy(prompt)

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
                        self._dispatch_to_airy(prompt)

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

    def _dispatch_to_airy(self, prompt: str) -> None:
        """Fire an async dispatch to Airy via Discord. Non-blocking."""
        if not self._config.is_configured:
            logger.error("Cannot dispatch: Discord config incomplete")
            self._dispatches.append(DispatchResult(
                prompt=prompt,
                response="",
                latency_ms=0,
                success=False,
                error="Discord config incomplete",
            ))
            return

        # Cancel any in-flight dispatch
        if self._active_dispatch and not self._active_dispatch.done():
            self._active_dispatch.cancel()

        self._active_dispatch = asyncio.create_task(
            self._post_and_poll(prompt)
        )

    async def _post_and_poll(self, prompt: str) -> None:
        """Post prompt to Discord webhook, poll for Airy's reply, inject it."""
        start = time.time()
        result = DispatchResult(
            prompt=prompt,
            response="",
            latency_ms=0,
            success=False,
        )

        try:
            http = await self._get_http()
            bot_user_id = await self._resolve_bot_user_id()

            # Post via webhook
            webhook_msg_id = await self._post_webhook(http, prompt)
            if not webhook_msg_id:
                result.error = "Webhook post failed"
                result.latency_ms = (time.time() - start) * 1000
                self._dispatches.append(result)
                if self._on_airy_response:
                    self._on_airy_response(result)
                return

            logger.info("Posted to Discord (msg %s), polling for reply...", webhook_msg_id)

            # Poll for Airy's reply
            response_text = await self._poll_for_reply(
                http, webhook_msg_id, bot_user_id
            )

            if response_text:
                result.response = response_text
                result.success = True
                result.latency_ms = (time.time() - start) * 1000
                logger.info(
                    "Airy responded (%.0fms): %s",
                    result.latency_ms,
                    result.response[:100],
                )
                # Inject the response into Nova's conversation history
                await self._inject_response(prompt, result.response)
            else:
                result.error = f"No reply within {self._config.dispatch_timeout}s"
                result.latency_ms = (time.time() - start) * 1000
                logger.warning("Airy did not reply in time for: %s", prompt[:100])

        except asyncio.CancelledError:
            result.error = "Dispatch cancelled"
            result.latency_ms = (time.time() - start) * 1000
            logger.info("Dispatch cancelled for: %s", prompt[:100])

        except Exception as e:
            result.error = str(e)
            result.latency_ms = (time.time() - start) * 1000
            logger.exception("Discord dispatch failed for: %s", prompt[:100])

        self._dispatches.append(result)

        if self._on_airy_response:
            self._on_airy_response(result)

    async def _post_webhook(self, http: aiohttp.ClientSession, prompt: str) -> str | None:
        """Post a message via Discord webhook. Returns the message ID or None."""
        payload = {
            "content": prompt,
            "username": "Nova Voice",
        }
        # Webhook URL uses its own auth (token in URL), so no Authorization header
        async with http.post(
            f"{self._config.webhook_url}?wait=true",
            json=payload,
            headers={"Authorization": ""},  # Override the bot token header
        ) as resp:
            if resp.status in (200, 201, 204):
                data = await resp.json()
                return data.get("id")
            else:
                body = await resp.text()
                logger.error("Webhook post failed: %s %s", resp.status, body[:200])
                return None

    async def _poll_for_reply(
        self,
        http: aiohttp.ClientSession,
        after_message_id: str,
        bot_user_id: str,
    ) -> str | None:
        """Poll the channel for a reply from the bot after the webhook message.

        Checks both:
        1. Thread created on the webhook message (Airy's typical behavior)
        2. Direct channel replies after the webhook message

        Returns the reply text or None if timeout.
        """
        deadline = time.time() + self._config.dispatch_timeout

        while time.time() < deadline:
            await asyncio.sleep(self._config.poll_interval)

            # Try thread first (thread_id = message_id for auto-threads)
            reply = await self._check_thread(http, after_message_id, bot_user_id)
            if reply:
                return reply

            # Fall back to channel messages after the webhook message
            reply = await self._check_channel(http, after_message_id, bot_user_id)
            if reply:
                return reply

        return None

    async def _check_thread(
        self,
        http: aiohttp.ClientSession,
        thread_id: str,
        bot_user_id: str,
    ) -> str | None:
        """Check for a bot reply in the thread created on the webhook message."""
        url = f"{DISCORD_API}/channels/{thread_id}/messages?limit=5"
        async with http.get(url) as resp:
            if resp.status == 200:
                messages = await resp.json()
                for msg in messages:
                    if msg.get("author", {}).get("id") == bot_user_id:
                        return msg.get("content", "")
            # 404 means no thread exists yet, which is fine
            return None

    async def _check_channel(
        self,
        http: aiohttp.ClientSession,
        after_message_id: str,
        bot_user_id: str,
    ) -> str | None:
        """Check for a bot reply in the channel after the webhook message."""
        url = (
            f"{DISCORD_API}/channels/{self._config.channel_id}"
            f"/messages?after={after_message_id}&limit=10"
        )
        async with http.get(url) as resp:
            if resp.status == 200:
                messages = await resp.json()
                for msg in messages:
                    if msg.get("author", {}).get("id") == bot_user_id:
                        return msg.get("content", "")
            return None

    async def _inject_response(self, prompt: str, response: str) -> None:
        """Inject Airy's response back into Nova via forced reconnect.

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

    async def close(self) -> None:
        """Close the HTTP session. Call on shutdown."""
        self.stop()
        if self._http and not self._http.closed:
            await self._http.close()


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
