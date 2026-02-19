"""Nova 2 Sonic bidirectional streaming session.

Manages the WebSocket connection, event protocol, and response routing
for a single Nova Sonic conversation.

Audio formats:
    Input:  16kHz / 16-bit / mono LPCM (base64 encoded)
    Output: 24kHz / 16-bit / mono LPCM (base64 encoded)

Session lifecycle:
    sessionStart → promptStart → contentStart(SYSTEM) → textInput →
    contentEnd → contentStart(AUDIO, interactive) → audioInput chunks →
    ... responses flow back ... → contentEnd → promptEnd → sessionEnd
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme,
)
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from smithy_aws_core.identity import EnvironmentCredentialsResolver

logger = logging.getLogger(__name__)

# --- Constants ---

MODEL_ID = "amazon.nova-2-sonic-v1:0"
DEFAULT_REGION = "us-east-1"

# Audio configuration
INPUT_SAMPLE_RATE = 16000    # Nova expects 16kHz mono
OUTPUT_SAMPLE_RATE = 24000   # Nova outputs 24kHz mono
CHANNELS = 1
SAMPLE_WIDTH = 2             # 16-bit PCM
CHUNK_SIZE = 1024            # Samples per chunk

# Session limits
SESSION_TIMEOUT_SECONDS = 7 * 60  # Reconnect before 8-min limit
SILENCE_AFTER_SPEECH_SECONDS = 2  # Silence to send after speech for turn detection

# Conversation history for session continuation
MAX_HISTORY_TURNS = 10  # Keep last N exchanges for context replay

# Default voice
DEFAULT_VOICE_ID = "matthew"

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly, helpful voice assistant. Keep responses short and natural, "
    "1 to 3 sentences unless the topic demands more."
)


class SessionState(Enum):
    """State machine for a Nova Sonic session."""
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class NovaSonicConfig:
    """Configuration for a Nova Sonic session."""
    region: str = DEFAULT_REGION
    model_id: str = MODEL_ID
    voice_id: str = DEFAULT_VOICE_ID
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 1024
    top_p: float = 0.9
    temperature: float = 0.7
    input_device_index: int | None = None  # None = auto-detect


@dataclass
class SessionMetrics:
    """Track session performance metrics."""
    session_start_time: float = 0.0
    events_received: int = 0
    audio_chunks_sent: int = 0
    audio_chunks_received: int = 0
    turns_completed: int = 0
    reconnections: int = 0


@dataclass
class ConversationTurn:
    """A single turn in the conversation history."""
    role: str       # "user" or "assistant"
    text: str       # Transcript text
    timestamp: float = 0.0


# Callback types
TextCallback = Callable[[str, str], None]  # (role, text) -> None
AudioCallback = Callable[[bytes], None]    # (audio_bytes) -> None
ReconnectCallback = Callable[[], None]     # () -> None (called on reconnect)


class NovaSonicSession:
    """A single Nova Sonic bidirectional streaming session.

    Manages the WebSocket connection, event protocol, and audio I/O.
    Supports automatic session continuation at the 8-minute boundary
    by tracking conversation history and replaying it on reconnect.

    Example::

        session = NovaSonicSession(config=NovaSonicConfig(voice_id="ruth"))
        await session.start()
        await session.send_audio(pcm_chunk)  # 16kHz/16bit/mono
        # ... audio responses arrive in session.audio_output_queue
        await session.stop()
    """

    def __init__(
        self,
        config: NovaSonicConfig | None = None,
        on_text: TextCallback | None = None,
        on_audio: AudioCallback | None = None,
        on_reconnect: ReconnectCallback | None = None,
    ):
        self.config = config or NovaSonicConfig()
        self._on_text = on_text
        self._on_audio = on_audio
        self._on_reconnect = on_reconnect

        # Connection state
        self._client: BedrockRuntimeClient | None = None
        self._stream = None
        self._state = SessionState.IDLE
        self._metrics = SessionMetrics()

        # Protocol identifiers (new UUIDs per session)
        self._prompt_name = ""
        self._system_content_name = ""
        self._audio_content_name = ""

        # Audio queues
        self._audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Conversation history for session continuation
        self._history: list[ConversationTurn] = []
        self._current_user_text = ""
        self._current_assistant_text = ""

        # Task handles
        self._response_task: asyncio.Task | None = None
        self._session_timer_task: asyncio.Task | None = None

        # Session continuation control
        self._should_reconnect = False
        self._reconnect_event = asyncio.Event()

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == SessionState.ACTIVE

    @property
    def metrics(self) -> SessionMetrics:
        return self._metrics

    @property
    def audio_output_queue(self) -> asyncio.Queue[bytes]:
        """Queue of audio bytes to play. Consumers can drain this."""
        return self._audio_output_queue

    @property
    def history(self) -> list[ConversationTurn]:
        """Conversation history across all session segments."""
        return list(self._history)

    @property
    def reconnect_event(self) -> asyncio.Event:
        """Set when a reconnect is triggered. Cleared after reconnect completes."""
        return self._reconnect_event

    def _init_client(self) -> None:
        """Initialize the Bedrock client with SigV4 auth."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.config.region}.amazonaws.com",
            region=self.config.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
        )
        self._client = BedrockRuntimeClient(config=config)
        logger.info("Bedrock client initialized (region=%s)", self.config.region)

    def _build_continuation_prompt(self) -> str:
        """Build a system prompt that includes conversation history for continuity."""
        if not self._history:
            return self.config.system_prompt

        # Build a conversation summary from recent turns
        recent = self._history[-MAX_HISTORY_TURNS:]
        history_lines = []
        for turn in recent:
            label = "User" if turn.role == "user" else "Assistant"
            history_lines.append(f"{label}: {turn.text}")

        history_text = "\n".join(history_lines)

        return (
            f"{self.config.system_prompt}\n\n"
            f"[Session continuation] The conversation so far:\n{history_text}\n\n"
            f"Continue the conversation naturally from where it left off. "
            f"Do not repeat previous responses or acknowledge the reconnection."
        )

    async def _send_event(self, event_dict: dict) -> None:
        """Send a JSON event to the stream."""
        event_json = json.dumps(event_dict)
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self._stream.input_stream.send(event)

    async def _open_stream(self, system_prompt: str) -> None:
        """Open a new stream and send the setup events."""
        # Fresh UUIDs for this session segment
        self._prompt_name = str(uuid.uuid4())
        self._system_content_name = str(uuid.uuid4())
        self._audio_content_name = str(uuid.uuid4())

        logger.info("Opening bidirectional stream...")
        self._stream = await self._client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.config.model_id)
        )

        # 1. Session start
        await self._send_event({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": self.config.max_tokens,
                        "topP": self.config.top_p,
                        "temperature": self.config.temperature,
                    }
                }
            }
        })

        # 2. Prompt start with audio output config
        await self._send_event({
            "event": {
                "promptStart": {
                    "promptName": self._prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": self.config.voice_id,
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                }
            }
        })

        # 3. System prompt (with history on reconnect)
        await self._send_event({
            "event": {
                "contentStart": {
                    "promptName": self._prompt_name,
                    "contentName": self._system_content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        })
        await self._send_event({
            "event": {
                "textInput": {
                    "promptName": self._prompt_name,
                    "contentName": self._system_content_name,
                    "content": system_prompt,
                }
            }
        })
        await self._send_event({
            "event": {
                "contentEnd": {
                    "promptName": self._prompt_name,
                    "contentName": self._system_content_name,
                }
            }
        })

        # 4. Open interactive audio input (stays open for continuous streaming)
        await self._send_event({
            "event": {
                "contentStart": {
                    "promptName": self._prompt_name,
                    "contentName": self._audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": INPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        })

        logger.info("Stream opened (prompt=%s)", self._prompt_name[:8])

    async def start(self) -> None:
        """Open the stream and set up the session."""
        if self._state not in (SessionState.IDLE, SessionState.CLOSED):
            logger.warning("Cannot start session in state %s", self._state.value)
            return

        self._state = SessionState.CONNECTING

        if not self._client:
            self._init_client()

        await self._open_stream(self.config.system_prompt)

        self._state = SessionState.ACTIVE
        self._metrics.session_start_time = time.time()

        # Start response processor
        self._response_task = asyncio.create_task(self._process_responses())

        # Start session timer for 8-min reconnect
        self._session_timer_task = asyncio.create_task(self._session_timer())

        logger.info("Session started")

    async def reconnect(self) -> None:
        """Reconnect with conversation context for session continuation.

        Closes the current stream and opens a new one with conversation
        history injected into the system prompt, so the model can continue
        the conversation naturally.
        """
        if self._state not in (SessionState.ACTIVE, SessionState.RECONNECTING):
            logger.warning("Cannot reconnect in state %s", self._state.value)
            return

        self._state = SessionState.RECONNECTING
        logger.info(
            "Reconnecting (turns=%d, history=%d entries)...",
            self._metrics.turns_completed,
            len(self._history),
        )

        # Flush any partial turn text into history
        self._flush_current_turn()

        # Cancel old tasks
        for task in [self._response_task, self._session_timer_task]:
            if task and not task.done():
                task.cancel()

        if self._response_task:
            try:
                await asyncio.wait_for(self._response_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Close old stream gracefully
        try:
            await self._send_event({
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                    }
                }
            })
            await self._send_event({
                "event": {"promptEnd": {"promptName": self._prompt_name}}
            })
            await self._send_event({
                "event": {"sessionEnd": {}}
            })
            await self._stream.input_stream.close()
        except Exception:
            logger.debug("Error closing old stream during reconnect", exc_info=True)

        # Open new stream with conversation context
        continuation_prompt = self._build_continuation_prompt()
        await self._open_stream(continuation_prompt)

        self._state = SessionState.ACTIVE
        self._metrics.session_start_time = time.time()
        self._metrics.reconnections += 1

        # Start new response processor and timer
        self._response_task = asyncio.create_task(self._process_responses())
        self._session_timer_task = asyncio.create_task(self._session_timer())

        # Notify consumers
        self._reconnect_event.set()
        self._reconnect_event.clear()

        if self._on_reconnect:
            self._on_reconnect()

        logger.info(
            "Reconnected (segment #%d, history=%d turns)",
            self._metrics.reconnections + 1,
            len(self._history),
        )

    async def send_audio(self, pcm_data: bytes) -> None:
        """Send a chunk of 16kHz/16bit/mono PCM audio to Nova."""
        if not self.is_active:
            return

        b64 = base64.b64encode(pcm_data).decode("utf-8")
        await self._send_event({
            "event": {
                "audioInput": {
                    "promptName": self._prompt_name,
                    "contentName": self._audio_content_name,
                    "content": b64,
                }
            }
        })
        self._metrics.audio_chunks_sent += 1

    async def stop(self) -> None:
        """Cleanly close the session."""
        if self._state in (SessionState.IDLE, SessionState.CLOSING, SessionState.CLOSED):
            return

        self._should_reconnect = False
        self._state = SessionState.CLOSING
        logger.info("Closing session...")

        # Flush any partial turn
        self._flush_current_turn()

        # Cancel tasks
        for task in [self._response_task, self._session_timer_task]:
            if task and not task.done():
                task.cancel()

        if self._response_task:
            try:
                await asyncio.wait_for(self._response_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Send close events
        try:
            await self._send_event({
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                    }
                }
            })
            await self._send_event({
                "event": {"promptEnd": {"promptName": self._prompt_name}}
            })
            await self._send_event({
                "event": {"sessionEnd": {}}
            })
            await self._stream.input_stream.close()
        except Exception:
            logger.debug("Error sending close events", exc_info=True)

        self._state = SessionState.CLOSED
        logger.info(
            "Session closed (events=%d, sent=%d, received=%d, turns=%d, reconnects=%d)",
            self._metrics.events_received,
            self._metrics.audio_chunks_sent,
            self._metrics.audio_chunks_received,
            self._metrics.turns_completed,
            self._metrics.reconnections,
        )

    def _flush_current_turn(self) -> None:
        """Save any accumulated partial turn text to history."""
        if self._current_user_text.strip():
            self._history.append(ConversationTurn(
                role="user",
                text=self._current_user_text.strip(),
                timestamp=time.time(),
            ))
            self._current_user_text = ""

        if self._current_assistant_text.strip():
            self._history.append(ConversationTurn(
                role="assistant",
                text=self._current_assistant_text.strip(),
                timestamp=time.time(),
            ))
            self._current_assistant_text = ""

        # Trim history to max length
        if len(self._history) > MAX_HISTORY_TURNS * 2:
            self._history = self._history[-MAX_HISTORY_TURNS * 2:]

    async def _process_responses(self) -> None:
        """Listen for and route events from Nova Sonic."""
        role: str | None = None
        display_text = False

        try:
            while self.is_active:
                output = await self._stream.await_output()
                result = await output[1].receive()

                if not (result.value and result.value.bytes_):
                    continue

                data = json.loads(result.value.bytes_.decode("utf-8"))
                self._metrics.events_received += 1

                if "event" not in data:
                    continue

                event = data["event"]

                if "contentStart" in event:
                    cs = event["contentStart"]
                    role = cs.get("role")
                    additional = cs.get("additionalModelFields")
                    if additional:
                        fields = json.loads(additional) if isinstance(additional, str) else additional
                        display_text = fields.get("generationStage") == "SPECULATIVE"
                    else:
                        display_text = False

                elif "textOutput" in event:
                    text = event["textOutput"]["content"]
                    if role == "ASSISTANT" and display_text:
                        logger.info("Assistant: %s", text[:200])
                        self._current_assistant_text += text
                        if self._on_text:
                            self._on_text("assistant", text)
                    elif role == "USER":
                        logger.info("User: %s", text[:200])
                        self._current_user_text += text
                        if self._on_text:
                            self._on_text("user", text)

                elif "audioOutput" in event:
                    audio_b64 = event["audioOutput"]["content"]
                    audio_bytes = base64.b64decode(audio_b64)
                    self._metrics.audio_chunks_received += 1
                    await self._audio_output_queue.put(audio_bytes)
                    if self._on_audio:
                        self._on_audio(audio_bytes)

                elif "completionEnd" in event:
                    self._metrics.turns_completed += 1
                    # Flush the completed turn to history
                    self._flush_current_turn()
                    logger.info("Turn %d complete", self._metrics.turns_completed)

                elif "usageEvent" in event:
                    pass  # Periodic, don't log

                # Other events (completionStart, contentEnd) are silent

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if "closed" not in str(e).lower():
                logger.exception("Response processing error")

    async def _session_timer(self) -> None:
        """Monitor session duration and trigger reconnect before 8-min limit."""
        try:
            await asyncio.sleep(SESSION_TIMEOUT_SECONDS)
            if self.is_active:
                logger.info(
                    "Session timeout approaching (%.0fs elapsed), reconnecting...",
                    time.time() - self._metrics.session_start_time,
                )
                self._should_reconnect = True
                await self.reconnect()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Session timer error during reconnect")
