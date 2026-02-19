"""Standalone conversation CLI for Nova Sonic Voice Agent.

Wires the C920 mic → Nova 2 Sonic → BT speaker for real-time
voice conversations through Breeze's hardware.

Usage:
    AWS_ACCESS_KEY_ID=$(pass aws/bedrock/access-key-id) \\
    AWS_SECRET_ACCESS_KEY=$(pass aws/bedrock/secret-access-key) \\
    AWS_DEFAULT_REGION=us-east-1 \\
    python3 -m nova_sonic.cli

Options:
    --voice VOICE_ID    Voice to use (default: matthew)
    --system PROMPT     Custom system prompt
    --duration SECS     Max conversation duration in seconds (0=unlimited)
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

    def start(self) -> None:
        self._start_time = time.time()
        print()
        print("=" * 60)
        print("  Nova 2 Sonic Voice Conversation")
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

    def on_reconnect(self) -> None:
        elapsed = time.time() - self._start_time
        print(f"\n  [{elapsed:5.0f}s] [session renewed, conversation continues]")
        self._current_role = None

    def summary(self, agent: NovaSonicVoiceAgent) -> None:
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
    if args.system:
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
    display.start()
    await agent.start()

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
    await agent.stop()
    display.summary(agent)


def main() -> None:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
