"""Minimal test: real-time bidirectional voice with Nova 2 Sonic.

Streams audio from the C920 mic to Nova Sonic via Bedrock WebSocket,
receives audio + text responses, plays them through the default output.

Press Ctrl+C to stop.

Usage:
    AWS_ACCESS_KEY_ID=$(pass aws/bedrock/access-key-id) \
    AWS_SECRET_ACCESS_KEY=$(pass aws/bedrock/secret-access-key) \
    AWS_DEFAULT_REGION=us-east-1 \
    python3 tests/manual/test_nova_sonic.py
"""

import asyncio
import base64
import json
import subprocess
import sys
import uuid

import pyaudio
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

# --- Constants ---
MODEL_ID = "amazon.nova-2-sonic-v1:0"
REGION = "us-east-1"

# Audio formats
INPUT_SAMPLE_RATE = 16000   # Nova expects 16kHz mono
OUTPUT_SAMPLE_RATE = 24000  # Nova outputs 24kHz mono
CHANNELS = 1
SAMPLE_WIDTH = 2            # 16-bit
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024           # Samples per chunk


def detect_c920_device() -> int | None:
    """Find the C920 pyaudio input device index."""
    p = pyaudio.PyAudio()
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if "C920" in str(info.get("name", "")) and info["maxInputChannels"] > 0:
                return i
    finally:
        p.terminate()
    return None


class NovaSonicTest:
    """Real-time Nova 2 Sonic bidirectional stream test.

    Follows the official AWS sample pattern exactly.
    """

    def __init__(self):
        self.client: BedrockRuntimeClient | None = None
        self.stream = None
        self.response = None
        self.is_active = False
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.role: str | None = None
        self.display_assistant_text = False
        self.events_received = 0
        self.text_responses: list[str] = []
        self.got_audio = False

    def _init_client(self):
        """Initialize the Bedrock client with proper SigV4 auth."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{REGION}.amazonaws.com",
            region=REGION,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="bedrock")},
        )
        self.client = BedrockRuntimeClient(config=config)
        print("✅ Bedrock client initialized (SigV4 auth)")

    async def send_event(self, event_json: str):
        """Send a raw JSON string event to the stream."""
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self.stream.input_stream.send(event)

    async def start_session(self):
        """Start session following the exact official AWS sample pattern."""
        if not self.client:
            self._init_client()

        print("📡 Opening bidirectional stream...")
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID)
        )
        self.is_active = True
        print("✅ Stream opened")

        # 1. Session start
        await self.send_event(json.dumps({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7,
                    }
                }
            }
        }))
        print("   Sent sessionStart")

        # 2. Prompt start with audio output config
        await self.send_event(json.dumps({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain",
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                }
            }
        }))
        print("   Sent promptStart")

        # 3. System prompt (interactive: true, matching official sample)
        await self.send_event(json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "SYSTEM",
                    "textInputConfiguration": {
                        "mediaType": "text/plain",
                    },
                }
            }
        }))

        system_prompt = (
            "You are a friendly assistant. The user and you will engage in a spoken dialog "
            "exchanging the transcripts of a natural real-time conversation. Keep your responses "
            "short, generally two or three sentences for chatty scenarios."
        )
        await self.send_event(json.dumps({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "content": system_prompt,
                }
            }
        }))

        await self.send_event(json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                }
            }
        }))
        print("   Sent system prompt")

        # Start response processing
        self.response = asyncio.create_task(self._process_responses())

    async def start_audio_input(self):
        """Open the interactive audio input content block."""
        await self.send_event(json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
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
        }))
        print("   Started interactive audio input")

    async def send_audio_chunk(self, audio_bytes: bytes):
        """Send a single audio chunk."""
        if not self.is_active:
            return
        blob = base64.b64encode(audio_bytes)
        await self.send_event(json.dumps({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": blob.decode("utf-8"),
                }
            }
        }))

    async def end_audio_input(self):
        """Close the audio input content block."""
        await self.send_event(json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                }
            }
        }))

    async def capture_audio(self, device_index: int | None = None):
        """Capture audio from mic and send to Nova Sonic in real time."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE,
        )

        print(f"🎤 Mic streaming started (16kHz, device={device_index})")
        print("   Speak now! Press Ctrl+C to stop.\n")

        await self.start_audio_input()

        try:
            while self.is_active:
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\n❌ Mic error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("   Mic stopped")
            await self.end_audio_input()

    async def _process_responses(self):
        """Process events from Nova Sonic."""
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()

                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode("utf-8")
                    json_data = json.loads(response_data)
                    self.events_received += 1

                    if "event" in json_data:
                        event = json_data["event"]

                        if "contentStart" in event:
                            cs = event["contentStart"]
                            self.role = cs.get("role")
                            additional = cs.get("additionalModelFields")
                            if additional:
                                fields = json.loads(additional) if isinstance(additional, str) else additional
                                self.display_assistant_text = fields.get("generationStage") == "SPECULATIVE"
                            else:
                                self.display_assistant_text = False

                        elif "textOutput" in event:
                            text = event["textOutput"]["content"]
                            if self.role == "ASSISTANT" and self.display_assistant_text:
                                print(f"\n   🤖 Assistant: {text}")
                                self.text_responses.append(text)
                            elif self.role == "USER":
                                print(f"\n   👤 You: {text}")

                        elif "audioOutput" in event:
                            audio_b64 = event["audioOutput"]["content"]
                            audio_bytes = base64.b64decode(audio_b64)
                            await self.audio_queue.put(audio_bytes)
                            if not self.got_audio:
                                print("\n   🔊 Receiving audio response...")
                                self.got_audio = True

                        elif "contentEnd" in event:
                            pass

                        elif "usageEvent" in event:
                            usage = event["usageEvent"]
                            tokens_in = usage.get("inputTokens", 0)
                            tokens_out = usage.get("outputTokens", 0)
                            if tokens_out > 0:
                                print(f"\n   [usage] in={tokens_in}, out={tokens_out}")

                        elif "completionEnd" in event:
                            print("\n   [turn complete]")

                        else:
                            print(f"\n   [event] {list(event.keys())}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if "closed" not in str(e).lower():
                print(f"\n❌ Response error: {e}")

    async def play_audio(self):
        """Play audio responses through the default output."""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        print("🔈 Audio playback ready (24kHz)")

        try:
            while self.is_active:
                audio_data = await self.audio_queue.get()
                for i in range(0, len(audio_data), CHUNK_SIZE):
                    if not self.is_active:
                        break
                    end = min(i + CHUNK_SIZE, len(audio_data))
                    chunk = audio_data[i:end]
                    await asyncio.get_event_loop().run_in_executor(
                        None, stream.write, chunk
                    )
                    await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"\n❌ Playback error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("   Playback stopped")

    async def end_session(self):
        """Clean shutdown."""
        if not self.is_active:
            return

        try:
            await self.send_event(json.dumps({
                "event": {"promptEnd": {"promptName": self.prompt_name}}
            }))
            await self.send_event(json.dumps({
                "event": {"sessionEnd": {}}
            }))
            await self.stream.input_stream.close()
        except Exception:
            pass
        self.is_active = False
        print("   Session ended")


async def main():
    print("=" * 60)
    print("Nova 2 Sonic Real-Time Voice Test")
    print("=" * 60)

    # Detect C920 mic
    device_index = detect_c920_device()
    if device_index is not None:
        print(f"✅ C920 mic found at pyaudio device {device_index}")
    else:
        print("⚠️  C920 not found, using default input device")

    # Create client and start session
    nova = NovaSonicTest()
    await nova.start_session()

    # Start playback and capture
    playback_task = asyncio.create_task(nova.play_audio())
    capture_task = asyncio.create_task(nova.capture_audio(device_index))

    # Wait for Ctrl+C
    try:
        while nova.is_active:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    # Shutdown
    nova.is_active = False

    for task in [capture_task, playback_task]:
        if not task.done():
            task.cancel()
    await asyncio.gather(capture_task, playback_task, return_exceptions=True)

    if nova.response and not nova.response.done():
        nova.response.cancel()

    await nova.end_session()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Events received: {nova.events_received}")
    print(f"Text responses: {len(nova.text_responses)}")
    print(f"Got audio response: {nova.got_audio}")
    if nova.text_responses:
        print(f"Text: {' '.join(nova.text_responses)}")
    success = nova.got_audio or len(nova.text_responses) > 0
    print(f"Result: {'✅ PASS' if success else '⚠️  CONNECTED (no speech detected in mic input)'}")


if __name__ == "__main__":
    asyncio.run(main())
