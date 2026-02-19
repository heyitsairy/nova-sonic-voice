"""High-level voice agent that orchestrates mic, playback, and Nova sessions.

This is the main entry point for voice conversations.
"""

from __future__ import annotations

import asyncio
import logging

import pyaudio

from nova_sonic.audio import _detect_c920_device
from nova_sonic.session import (
    CHANNELS,
    CHUNK_SIZE,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    NovaSonicConfig,
    NovaSonicSession,
    TextCallback,
)

logger = logging.getLogger(__name__)

PYAUDIO_FORMAT = pyaudio.paInt16


class NovaSonicVoiceAgent:
    """Orchestrates mic capture, Nova Sonic session, and audio playback.

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
    ):
        self.config = config or NovaSonicConfig()
        self._on_user_text = on_user_text
        self._on_assistant_text = on_assistant_text

        self._session: NovaSonicSession | None = None
        self._capture_task: asyncio.Task | None = None
        self._playback_task: asyncio.Task | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def _text_callback(self, role: str, text: str) -> None:
        """Route text events to the appropriate callback."""
        if role == "user" and self._on_user_text:
            self._on_user_text(role, text)
        elif role == "assistant" and self._on_assistant_text:
            self._on_assistant_text(role, text)

    async def start(self) -> None:
        """Start the voice agent: open session, begin mic capture and playback."""
        if self._running:
            return

        self._running = True

        self._session = NovaSonicSession(
            config=self.config,
            on_text=self._text_callback,
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

        logger.info("Voice agent stopped")

    async def send_audio(self, pcm_data: bytes) -> None:
        """Send raw PCM audio directly (for external integrations)."""
        if self._session and self._session.is_active:
            await self._session.send_audio(pcm_data)

    async def _capture_loop(self) -> None:
        """Capture audio from mic and stream to Nova."""
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
            while self._running and self._session and self._session.is_active:
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
        """Play audio responses from Nova through the default output."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=PYAUDIO_FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        logger.info("Audio playback started")

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
