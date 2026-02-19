"""High-level voice agent that orchestrates mic, playback, and Nova sessions.

This is the main entry point for voice conversations.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pyaudio

from nova_sonic.audio import _detect_c920_device
from nova_sonic.session import (
    CHANNELS,
    CHUNK_SIZE,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    ConversationTurn,
    NovaSonicConfig,
    NovaSonicSession,
    TextCallback,
)

logger = logging.getLogger(__name__)

PYAUDIO_FORMAT = pyaudio.paInt16


class NovaSonicVoiceAgent:
    """Orchestrates mic capture, Nova Sonic session, and audio playback.

    Handles session continuation automatically when the 8-min limit fires.
    Mic capture pauses briefly during reconnect and resumes seamlessly.

    Example::

        agent = NovaSonicVoiceAgent(
            config=NovaSonicConfig(voice_id="ruth"),
            on_assistant_text=lambda role, text: print(f"Assistant: {text}"),
        )
        await agent.start()
        # ... speaks through mic, hears response through speaker ...
        await agent.stop()
    """

    def __init__(
        self,
        config: NovaSonicConfig | None = None,
        on_user_text: TextCallback | None = None,
        on_assistant_text: TextCallback | None = None,
        output_device_index: int | None = None,
    ):
        self.config = config or NovaSonicConfig()
        self._on_user_text = on_user_text
        self._on_assistant_text = on_assistant_text
        self._output_device_index = output_device_index

        self._session: NovaSonicSession | None = None
        self._capture_task: asyncio.Task | None = None
        self._playback_task: asyncio.Task | None = None
        self._running = False
        self._start_time = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def session(self) -> NovaSonicSession | None:
        return self._session

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time == 0.0:
            return 0.0
        return time.time() - self._start_time

    def _text_callback(self, role: str, text: str) -> None:
        """Route text events to the appropriate callback."""
        if role == "user" and self._on_user_text:
            self._on_user_text(role, text)
        elif role == "assistant" and self._on_assistant_text:
            self._on_assistant_text(role, text)

    def _reconnect_callback(self) -> None:
        """Called when a session reconnect completes."""
        logger.info("Session reconnected (conversation continues)")

    async def start(self) -> None:
        """Start the voice agent: open session, begin mic capture and playback."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()

        self._session = NovaSonicSession(
            config=self.config,
            on_text=self._text_callback,
            on_reconnect=self._reconnect_callback,
        )
        await self._session.start()

        self._capture_task = asyncio.create_task(self._capture_loop())
        self._playback_task = asyncio.create_task(self._playback_loop())

        logger.info("Voice agent started")

    async def stop(self) -> None:
        """Stop the voice agent."""
        if not self._running:
            return

        self._running = False

        for task in [self._capture_task, self._playback_task]:
            if task and not task.done():
                task.cancel()

        tasks = [t for t in [self._capture_task, self._playback_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self._session:
            await self._session.stop()

        logger.info(
            "Voice agent stopped (elapsed=%.0fs, turns=%d, reconnects=%d)",
            self.elapsed_seconds,
            self._session.metrics.turns_completed if self._session else 0,
            self._session.metrics.reconnections if self._session else 0,
        )

    async def send_audio(self, pcm_data: bytes) -> None:
        """Send raw PCM audio directly (for external integrations)."""
        if self._session and self._session.is_active:
            await self._session.send_audio(pcm_data)

    def get_transcript(self) -> list[ConversationTurn]:
        """Get the full conversation transcript."""
        if self._session:
            return self._session.history
        return []

    async def _capture_loop(self) -> None:
        """Capture audio from mic and stream to Nova.

        Handles reconnects gracefully by waiting for the session to be active
        again before resuming audio capture.
        """
        device_index = self.config.input_device_index
        if device_index is None:
            device_index = _detect_c920_device()

        p = pyaudio.PyAudio()
        stream = p.open(
            format=PYAUDIO_FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE,
        )

        logger.info("Mic capture started (device=%s)", device_index)

        try:
            while self._running:
                # Wait for session to be active (handles reconnect gaps)
                if not self._session or not self._session.is_active:
                    # Drain mic buffer during reconnect to avoid stale audio
                    stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    await asyncio.sleep(0.05)
                    continue

                audio_data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: stream.read(CHUNK_SIZE, exception_on_overflow=False),
                )
                await self._session.send_audio(audio_data)
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Mic capture error")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info("Mic capture stopped")

    async def _playback_loop(self) -> None:
        """Play audio responses from Nova through the output device.

        Continues across session reconnects since the audio queue persists.
        """
        p = pyaudio.PyAudio()

        # Use specified output device or default
        open_kwargs = {
            "format": PYAUDIO_FORMAT,
            "channels": CHANNELS,
            "rate": OUTPUT_SAMPLE_RATE,
            "output": True,
            "frames_per_buffer": CHUNK_SIZE,
        }
        if self._output_device_index is not None:
            open_kwargs["output_device_index"] = self._output_device_index

        stream = p.open(**open_kwargs)

        logger.info("Audio playback started (device=%s)", self._output_device_index or "default")

        try:
            while self._running:
                if not self._session:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    audio_data = await asyncio.wait_for(
                        self._session.audio_output_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                await asyncio.get_event_loop().run_in_executor(
                    None, stream.write, audio_data
                )

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Audio playback error")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info("Audio playback stopped")
