"""Discord voice channel bridge for Nova Sonic.

Bridges Discord voice channel audio to/from a Nova Sonic session.
Handles format conversion (48kHz stereo ↔ 16kHz mono), real-time
transcription posting, and session lifecycle management.

Architecture:
    Discord Voice Channel
        ↓ (48kHz/stereo PCM)
    NovaSonicAudioSink
        ↓ discord_to_nova()
    NovaSonicVoiceAgent.send_audio()
        ↓ (AWS Bedrock WebSocket)
    Nova 2 Sonic Model
        ↓ (24kHz/mono PCM + text)
    NovaSonicAudioSource
        ↓ nova_to_discord()
    Discord Voice Channel
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from nova_sonic.audio import (
    DISCORD_CHANNELS,
    DISCORD_SAMPLE_RATE,
    NOVA_INPUT_SAMPLE_RATE,
    NOVA_OUTPUT_SAMPLE_RATE,
    discord_to_nova,
    nova_to_discord,
)
from nova_sonic.session import (
    ConversationTurn,
    NovaSonicConfig,
    NovaSonicSession,
    SessionState,
)

logger = logging.getLogger(__name__)

# Discord audio frame parameters
# discord.py sends 20ms frames at 48kHz stereo (16-bit)
DISCORD_FRAME_MS = 20
DISCORD_FRAME_SAMPLES = DISCORD_SAMPLE_RATE * DISCORD_FRAME_MS // 1000  # 960
DISCORD_FRAME_BYTES = DISCORD_FRAME_SAMPLES * DISCORD_CHANNELS * 2  # 3840

# How much audio to buffer before sending to Nova (in ms)
# Smaller = lower latency, larger = fewer API calls
BUFFER_MS = 60  # 3 Discord frames
BUFFER_FRAMES = BUFFER_MS // DISCORD_FRAME_MS


# Callback type for transcript updates
TranscriptCallback = Callable[[str, str], None]  # (role, text)


@dataclass
class BridgeMetrics:
    """Track bridge performance."""

    started_at: float = 0.0
    discord_frames_received: int = 0
    discord_frames_sent: int = 0
    nova_chunks_forwarded: int = 0
    nova_audio_chunks_received: int = 0
    user_turns: int = 0
    assistant_turns: int = 0


class NovaSonicBridge:
    """Bridges a Discord voice channel to a Nova Sonic conversation.

    Usage with discord.py::

        bridge = NovaSonicBridge(
            config=NovaSonicConfig(voice_id="ruth"),
            on_transcript=lambda role, text: channel.send(f"{role}: {text}"),
        )

        # When joining a voice channel:
        voice_client = await channel.connect()
        await bridge.start()

        # Feed Discord audio into the bridge:
        # (called from a discord.py AudioSink or voice receive handler)
        bridge.receive_discord_audio(pcm_data, user_id)

        # Get audio to send back to Discord:
        # (called from a discord.py AudioSource)
        audio = bridge.get_discord_audio()

        # When leaving:
        await bridge.stop()

    The bridge does NOT manage the Discord voice client directly.
    It provides audio I/O methods that integrate with discord.py's
    voice infrastructure.
    """

    def __init__(
        self,
        config: NovaSonicConfig | None = None,
        on_transcript: TranscriptCallback | None = None,
        on_user_speech: TranscriptCallback | None = None,
        on_assistant_speech: TranscriptCallback | None = None,
        on_turn_complete: Callable[[ConversationTurn], None] | None = None,
    ):
        """Initialize the bridge.

        Args:
            config: Nova Sonic configuration (voice, system prompt, etc.).
            on_transcript: Called for all transcript events (user and assistant).
            on_user_speech: Called when user speech is transcribed.
            on_assistant_speech: Called when assistant responds.
            on_turn_complete: Called when a full turn (user or assistant) completes.
        """
        self.config = config or NovaSonicConfig()
        self._on_transcript = on_transcript
        self._on_user_speech = on_user_speech
        self._on_assistant_speech = on_assistant_speech
        self._on_turn_complete = on_turn_complete

        # Session
        self._session: NovaSonicSession | None = None
        self._running = False
        self._metrics = BridgeMetrics()

        # Audio buffers
        self._input_buffer: deque[bytes] = deque()  # Discord frames waiting to be sent
        self._output_queue: asyncio.Queue[bytes] = asyncio.Queue()  # Nova audio for Discord

        # Tasks
        self._forward_task: asyncio.Task | None = None
        self._output_task: asyncio.Task | None = None

        # Track speaking users (for multi-user support)
        self._active_speakers: dict[int, float] = {}  # user_id -> last_audio_time

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def metrics(self) -> BridgeMetrics:
        return self._metrics

    @property
    def session(self) -> NovaSonicSession | None:
        return self._session

    @property
    def active_speakers(self) -> dict[int, float]:
        """Map of user_id -> last audio timestamp for active speakers."""
        return dict(self._active_speakers)

    def _text_callback(self, role: str, text: str) -> None:
        """Handle text events from Nova."""
        if self._on_transcript:
            self._on_transcript(role, text)

        if role == "user":
            self._metrics.user_turns += 1
            if self._on_user_speech:
                self._on_user_speech(role, text)
        elif role == "assistant":
            self._metrics.assistant_turns += 1
            if self._on_assistant_speech:
                self._on_assistant_speech(role, text)

    def _audio_callback(self, audio_bytes: bytes) -> None:
        """Handle audio output from Nova (24kHz mono).

        Converts to Discord format and queues for playback.
        """
        discord_audio = nova_to_discord(audio_bytes)
        self._metrics.nova_audio_chunks_received += 1

        # Split into Discord-sized frames (20ms each)
        offset = 0
        while offset + DISCORD_FRAME_BYTES <= len(discord_audio):
            frame = discord_audio[offset : offset + DISCORD_FRAME_BYTES]
            try:
                self._output_queue.put_nowait(frame)
                self._metrics.discord_frames_sent += 1
            except asyncio.QueueFull:
                logger.debug("Output queue full, dropping frame")
            offset += DISCORD_FRAME_BYTES

        # Handle remaining partial frame (pad with silence)
        if offset < len(discord_audio):
            remaining = discord_audio[offset:]
            padded = remaining + b"\x00" * (DISCORD_FRAME_BYTES - len(remaining))
            try:
                self._output_queue.put_nowait(padded)
                self._metrics.discord_frames_sent += 1
            except asyncio.QueueFull:
                pass

    def _reconnect_callback(self) -> None:
        """Handle session reconnection."""
        logger.info("Nova session reconnected, bridge continuing")

    async def start(self) -> None:
        """Start the bridge and open a Nova Sonic session."""
        if self._running:
            return

        self._running = True
        self._metrics = BridgeMetrics(started_at=time.time())

        self._session = NovaSonicSession(
            config=self.config,
            on_text=self._text_callback,
            on_audio=self._audio_callback,
            on_reconnect=self._reconnect_callback,
        )
        await self._session.start()

        # Start the input forwarding task
        self._forward_task = asyncio.create_task(self._forward_loop())

        logger.info("Nova Sonic bridge started")

    async def stop(self) -> None:
        """Stop the bridge and close the Nova session."""
        if not self._running:
            return

        self._running = False

        # Cancel tasks
        if self._forward_task and not self._forward_task.done():
            self._forward_task.cancel()
            try:
                await asyncio.wait_for(self._forward_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Stop session
        if self._session:
            await self._session.stop()

        # Clear buffers
        self._input_buffer.clear()
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        elapsed = time.time() - self._metrics.started_at if self._metrics.started_at else 0
        logger.info(
            "Nova Sonic bridge stopped (elapsed=%.0fs, in_frames=%d, out_frames=%d, "
            "user_turns=%d, assistant_turns=%d)",
            elapsed,
            self._metrics.discord_frames_received,
            self._metrics.discord_frames_sent,
            self._metrics.user_turns,
            self._metrics.assistant_turns,
        )

    def receive_discord_audio(self, pcm_data: bytes, user_id: int | None = None) -> None:
        """Receive a Discord audio frame for forwarding to Nova.

        Called by the voice receive handler when audio arrives from
        a user in the voice channel. Audio should be 48kHz/16-bit/stereo PCM.

        Args:
            pcm_data: Raw PCM audio from Discord (48kHz/stereo).
            user_id: Optional Discord user ID of the speaker.
        """
        if not self._running:
            return

        self._metrics.discord_frames_received += 1

        if user_id is not None:
            self._active_speakers[user_id] = time.time()

        self._input_buffer.append(pcm_data)

    def get_discord_audio(self) -> bytes | None:
        """Get the next audio frame to send to Discord.

        Returns a 20ms frame of 48kHz/stereo PCM audio, or None if
        no audio is available. Called by the discord.py AudioSource.

        Returns:
            20ms PCM frame (3840 bytes) or None.
        """
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_discord_audio_async(self, timeout: float = 0.02) -> bytes | None:
        """Async version of get_discord_audio with timeout.

        Args:
            timeout: Max wait time in seconds (default: 20ms, one Discord frame).

        Returns:
            20ms PCM frame or None if timeout.
        """
        try:
            return await asyncio.wait_for(self._output_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def get_transcript(self) -> list[ConversationTurn]:
        """Get the full conversation transcript."""
        if self._session:
            return self._session.history
        return []

    async def _forward_loop(self) -> None:
        """Forward buffered Discord audio to Nova at regular intervals.

        Buffers multiple Discord frames (default: 60ms / 3 frames) before
        converting and sending to Nova. This reduces the per-chunk overhead
        of format conversion while keeping latency low.
        """
        try:
            while self._running:
                if len(self._input_buffer) >= BUFFER_FRAMES:
                    # Collect frames
                    frames = []
                    for _ in range(min(BUFFER_FRAMES, len(self._input_buffer))):
                        frames.append(self._input_buffer.popleft())

                    # Concatenate and convert
                    combined = b"".join(frames)
                    nova_audio = discord_to_nova(combined)

                    # Send to Nova
                    if self._session and self._session.is_active:
                        await self._session.send_audio(nova_audio)
                        self._metrics.nova_chunks_forwarded += 1

                # Clean up stale speakers (no audio for > 5 seconds)
                now = time.time()
                stale = [
                    uid
                    for uid, last in self._active_speakers.items()
                    if now - last > 5.0
                ]
                for uid in stale:
                    del self._active_speakers[uid]

                await asyncio.sleep(0.01)  # 10ms polling

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Forward loop error")


class DiscordAudioSink:
    """A convenience wrapper that acts as a discord.py compatible audio sink.

    Receives audio from Discord voice and forwards it to a NovaSonicBridge.
    Works with discord.py's VoiceClient.listen() API (if available) or
    can be used manually in a voice receive callback.

    Example::

        bridge = NovaSonicBridge(config=config)
        sink = DiscordAudioSink(bridge)

        # In your bot's voice receive handler:
        def on_voice_receive(user, data):
            sink.write(data.pcm, user.id)
    """

    def __init__(self, bridge: NovaSonicBridge):
        self.bridge = bridge

    def write(self, pcm_data: bytes, user_id: int | None = None) -> None:
        """Write PCM audio from Discord to the bridge.

        Args:
            pcm_data: 48kHz/16-bit/stereo PCM audio.
            user_id: Discord user ID of the speaker.
        """
        self.bridge.receive_discord_audio(pcm_data, user_id)


class DiscordAudioSource:
    """A convenience wrapper that acts as a discord.py compatible audio source.

    Reads audio from a NovaSonicBridge and provides it as PCM frames
    for Discord playback.

    Example::

        bridge = NovaSonicBridge(config=config)
        source = DiscordAudioSource(bridge)

        # In your bot's audio send loop:
        frame = source.read()
        if frame:
            voice_client.send_audio_packet(frame)
    """

    def __init__(self, bridge: NovaSonicBridge):
        self.bridge = bridge

    def read(self) -> bytes:
        """Read the next 20ms PCM frame for Discord.

        Returns silence if no audio is available (to maintain
        continuous audio stream for Discord).

        Returns:
            3840 bytes of 48kHz/16-bit/stereo PCM.
        """
        frame = self.bridge.get_discord_audio()
        if frame is not None:
            return frame
        # Return silence to keep the audio stream alive
        return b"\x00" * DISCORD_FRAME_BYTES

    def is_opus(self) -> bool:
        """Tell discord.py this is raw PCM, not Opus."""
        return False

    def cleanup(self) -> None:
        """Called when the audio source is no longer needed."""
        pass
