"""Standalone conversation CLI for Nova Sonic Voice Agent.

Wires the C920 mic → Nova 2 Sonic → BT speaker for real-time
voice conversations through Breeze's hardware.

Usage:
    AWS_ACCESS_KEY_ID=$(pass aws/bedrock/access-key-id) \\
    AWS_SECRET_ACCESS_KEY=$(pass aws/bedrock/secret-access-key) \\
    AWS_DEFAULT_REGION=us-east-1 \\
    python3 -m nova_sonic.cli

    # With Airy orchestrator (Nova calls Airy through Discord):
    NOVA_WEBHOOK_URL=... DISCORD_TOKEN=... NOVA_CHANNEL_ID=... \\
    python3 -m nova_sonic.cli --airy

Options:
    --voice VOICE_ID    Voice to use (default: matthew)
    --system PROMPT     Custom system prompt
    --duration SECS     Max conversation duration in seconds (0=unlimited)
    --airy              Enable Airy orchestrator (Nova calls Airy via Discord)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time

from nova_sonic.agent import NovaSonicVoiceAgent
from nova_sonic.audio import detect_mic
from nova_sonic.orchestrator import AiryOrchestrator, OrchestratorConfig, build_nova_system_prompt
from nova_sonic.session import NovaSonicConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nova_sonic.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time voice conversation with Nova 2 Sonic"
    )
    parser.add_argument(
        "--voice",
        default="matthew",
        help="Voice ID for Nova Sonic output (default: matthew)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt (default: friendly assistant)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Max conversation duration in seconds (0=unlimited, Ctrl+C to stop)",
    )
    parser.add_argument(
        "--airy",
        action="store_true",
        help="Enable Airy orchestrator (Nova calls Airy via Discord). "
        "Requires NOVA_WEBHOOK_URL, DISCORD_TOKEN, NOVA_CHANNEL_ID env vars.",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="Discord webhook URL (overrides NOVA_WEBHOOK_URL env var)",
    )
    parser.add_argument(
        "--channel-id",
        default=None,
        help="Discord channel ID (overrides NOVA_CHANNEL_ID env var)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


class ConversationDisplay:
    """Real-time conversation display in the terminal."""

    def __init__(self):
        self._current_role: str | None = None
        self._turn_count = 0
        self._start_time = 0.0

    def start(self, airy_enabled: bool = False) -> None:
        self._start_time = time.time()
        print()
        print("=" * 60)
        print("  Nova 2 Sonic Voice Conversation")
        if airy_enabled:
            print("  [Airy orchestrator enabled via Discord]")
        print("=" * 60)
        print("  Speak naturally. Press Ctrl+C to stop.")
        print("  Session auto-continues past the 8-min limit.")
        print("=" * 60)
        print()

    def on_user_text(self, role: str, text: str) -> None:
        if self._current_role != "user":
            self._current_role = "user"
            elapsed = time.time() - self._start_time
            print(f"\n  [{elapsed:5.0f}s] You: ", end="", flush=True)
        print(text, end="", flush=True)

    def on_assistant_text(self, role: str, text: str) -> None:
        if self._current_role != "assistant":
            self._current_role = "assistant"
            self._turn_count += 1
            elapsed = time.time() - self._start_time
            print(f"\n  [{elapsed:5.0f}s] Nova: ", end="", flush=True)
        print(text, end="", flush=True)

    def on_airy_dispatch(self, result) -> None:
        """Show when Nova calls Airy and gets a response."""
        elapsed = time.time() - self._start_time
        if result.success:
            print(f"\n  [{elapsed:5.0f}s] [Airy] asked: {result.prompt[:80]}")
            print(f"  [{elapsed:5.0f}s] [Airy] responded ({result.latency_ms:.0f}ms): {result.response[:100]}")
        else:
            print(f"\n  [{elapsed:5.0f}s] [Airy] failed: {result.error}")
        self._current_role = None

    def on_reconnect(self) -> None:
        elapsed = time.time() - self._start_time
        print(f"\n  [{elapsed:5.0f}s] [session renewed, conversation continues]")
        self._current_role = None

    def summary(self, agent: NovaSonicVoiceAgent, orchestrator=None) -> None:
        elapsed = time.time() - self._start_time
        session = agent.session
        metrics = session.metrics if session else None
        history = agent.get_transcript()

        print("\n")
        print("=" * 60)
        print("  Conversation Summary")
        print("=" * 60)
        print(f"  Duration:     {elapsed:.0f}s")
        if metrics:
            print(f"  Turns:        {metrics.turns_completed}")
            print(f"  Reconnects:   {metrics.reconnections}")
            print(f"  Audio sent:   {metrics.audio_chunks_sent} chunks")
            print(f"  Audio recv:   {metrics.audio_chunks_received} chunks")
            print(f"  Events:       {metrics.events_received}")
        if orchestrator:
            dispatches = orchestrator.dispatches
            succeeded = sum(1 for d in dispatches if d.success)
            failed = len(dispatches) - succeeded
            avg_latency = (
                sum(d.latency_ms for d in dispatches if d.success) / succeeded
                if succeeded
                else 0
            )
            print(f"  Airy calls:   {len(dispatches)} ({succeeded} ok, {failed} failed)")
            if succeeded:
                print(f"  Avg latency:  {avg_latency:.0f}ms")
        if history:
            print(f"  Transcript:   {len(history)} entries")
            print()
            print("  --- Transcript ---")
            for turn in history:
                label = "You" if turn.role == "user" else "Nova"
                # Truncate long entries
                text = turn.text if len(turn.text) <= 120 else turn.text[:117] + "..."
                print(f"  {label}: {text}")
        print("=" * 60)


async def run(args: argparse.Namespace) -> None:
    """Run the conversation."""
    display = ConversationDisplay()

    # Detect mic
    mic_index = detect_mic()
    if mic_index is not None:
        logger.info("Mic detected at device index %d", mic_index)
    else:
        logger.warning("No mic detected, using system default")

    # Build config
    config = NovaSonicConfig(
        voice_id=args.voice,
        input_device_index=mic_index,
    )

    # If Airy orchestrator is enabled, use the Nova-calls-Airy system prompt
    if args.airy:
        config.system_prompt = build_nova_system_prompt(
            base_personality=args.system or "",
        )
        logger.info("Airy orchestrator enabled (Discord dispatch)")
    elif args.system:
        config.system_prompt = args.system

    # Create agent
    agent = NovaSonicVoiceAgent(
        config=config,
        on_user_text=display.on_user_text,
        on_assistant_text=display.on_assistant_text,
    )

    # Handle Ctrl+C
    stop_event = asyncio.Event()

    def signal_handler():
        print("\n\n  Stopping...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Start conversation
    display.start(airy_enabled=args.airy)
    await agent.start()

    # Wire up Airy orchestrator after session exists
    orchestrator = None
    if args.airy and agent.session:
        orch_config = OrchestratorConfig(
            dispatch_timeout=30.0,
        )
        # CLI args override env vars
        if args.webhook_url:
            orch_config.webhook_url = args.webhook_url
        if args.channel_id:
            orch_config.channel_id = args.channel_id

        if not orch_config.is_configured:
            logger.error(
                "Airy orchestrator requested but Discord config incomplete. "
                "Set NOVA_WEBHOOK_URL, DISCORD_TOKEN, NOVA_CHANNEL_ID."
            )
        else:
            orchestrator = AiryOrchestrator(
                session=agent.session,
                config=orch_config,
                on_airy_response=display.on_airy_dispatch,
                on_text=display.on_assistant_text,
            )
            logger.info("Airy orchestrator wired into session")

    # Wire up reconnect display
    if agent.session:
        original_on_reconnect = agent.session._on_reconnect

        def reconnect_with_display() -> None:
            display.on_reconnect()
            if original_on_reconnect:
                original_on_reconnect()

        agent.session._on_reconnect = reconnect_with_display

    # Wait for stop signal or duration limit
    if args.duration > 0:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=args.duration)
        except asyncio.TimeoutError:
            print(f"\n\n  Duration limit reached ({args.duration}s)")
    else:
        await stop_event.wait()

    # Shutdown
    if orchestrator:
        await orchestrator.close()
        logger.info(
            "Airy orchestrator stopped (%d dispatches)",
            orchestrator.dispatch_count,
        )
    await agent.stop()
    display.summary(agent, orchestrator=orchestrator)


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
