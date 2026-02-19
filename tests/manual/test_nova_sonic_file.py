"""Test Nova 2 Sonic by sending a pre-recorded speech file.

Bypasses mic capture to verify the full pipeline independently of room conditions.

Usage:
    AWS_ACCESS_KEY_ID=$(pass aws/bedrock/access-key-id) \
    AWS_SECRET_ACCESS_KEY=$(pass aws/bedrock/secret-access-key) \
    AWS_DEFAULT_REGION=us-east-1 \
    python3 tests/manual/test_nova_sonic_file.py
"""

import asyncio
import base64
import json
import sys
import uuid
from pathlib import Path

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

MODEL_ID = "amazon.nova-2-sonic-v1:0"
REGION = "us-east-1"
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

# Pre-recorded speech file (16kHz/16bit/mono raw PCM)
SPEECH_FILE = "/tmp/test_speech_16k.raw"


class NovaSonicFileTest:
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
        self.all_events: list[str] = []

    def _init_client(self):
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
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self.stream.input_stream.send(event)

    async def start_session(self):
        if not self.client:
            self._init_client()

        print("📡 Opening bidirectional stream...")
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID)
        )
        self.is_active = True
        print("✅ Stream opened")

        # Session start
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

        # Prompt start
        await self.send_event(json.dumps({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
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

        # System prompt
        await self.send_event(json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        }))
        await self.send_event(json.dumps({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "content": "You are a friendly assistant. Keep responses to 1 or 2 sentences.",
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

    async def send_audio_file(self, file_path: str):
        """Send a pre-recorded PCM file as audio input, paced at real-time speed."""
        pcm_data = Path(file_path).read_bytes()
        duration = len(pcm_data) / (INPUT_SAMPLE_RATE * 2)  # 16-bit = 2 bytes/sample
        print(f"📁 Sending {file_path} ({len(pcm_data)} bytes, {duration:.1f}s)")

        # Start audio content
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

        # Send in chunks, paced to simulate real-time
        chunk_bytes = CHUNK_SIZE * 2  # 2 bytes per sample
        chunk_duration = CHUNK_SIZE / INPUT_SAMPLE_RATE  # seconds per chunk
        total_chunks = 0

        for i in range(0, len(pcm_data), chunk_bytes):
            chunk = pcm_data[i:i + chunk_bytes]
            b64 = base64.b64encode(chunk).decode("utf-8")
            await self.send_event(json.dumps({
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": b64,
                    }
                }
            }))
            total_chunks += 1
            # Pace at real-time speed so Nova's turn detection works
            await asyncio.sleep(chunk_duration)

        print(f"   Sent {total_chunks} chunks at real-time pace")

        # After the speech, send some silence to trigger turn detection
        silence = b"\x00" * chunk_bytes
        b64_silence = base64.b64encode(silence).decode("utf-8")
        print("   Sending 2s of silence for turn detection...")
        silence_chunks = int(2.0 / chunk_duration)
        for _ in range(silence_chunks):
            await self.send_event(json.dumps({
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": b64_silence,
                    }
                }
            }))
            await asyncio.sleep(chunk_duration)

        print("   Speech + silence sent")

    async def _process_responses(self):
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
                        event_type = list(event.keys())[0]
                        self.all_events.append(event_type)

                        if "contentStart" in event:
                            cs = event["contentStart"]
                            self.role = cs.get("role")
                            additional = cs.get("additionalModelFields")
                            if additional:
                                fields = json.loads(additional) if isinstance(additional, str) else additional
                                self.display_assistant_text = fields.get("generationStage") == "SPECULATIVE"
                            else:
                                self.display_assistant_text = False
                            print(f"   [contentStart] role={self.role} type={cs.get('type')} speculative={self.display_assistant_text}")

                        elif "textOutput" in event:
                            text = event["textOutput"]["content"]
                            if self.role == "ASSISTANT" and self.display_assistant_text:
                                print(f"   🤖 {text}")
                                self.text_responses.append(text)
                            elif self.role == "USER":
                                print(f"   👤 You: {text}")

                        elif "audioOutput" in event:
                            audio_b64 = event["audioOutput"]["content"]
                            audio_bytes = base64.b64decode(audio_b64)
                            await self.audio_queue.put(audio_bytes)
                            if not self.got_audio:
                                print("   🔊 Audio response streaming...")
                                self.got_audio = True

                        elif "usageEvent" in event:
                            usage = event["usageEvent"]
                            tokens_in = usage.get("inputTokens", 0)
                            tokens_out = usage.get("outputTokens", 0)
                            print(f"   [usage] in={tokens_in}, out={tokens_out}")

                        elif "completionEnd" in event:
                            print("   [turn complete]")

                        elif "contentEnd" in event:
                            pass  # silent

                        else:
                            print(f"   [{event_type}] {json.dumps(event[event_type])[:100]}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if "closed" not in str(e).lower():
                print(f"❌ Response error: {e}")

    async def play_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=OUTPUT_SAMPLE_RATE, output=True)

        try:
            while self.is_active or not self.audio_queue.empty():
                try:
                    audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                await asyncio.get_event_loop().run_in_executor(None, stream.write, audio_data)
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def end_session(self):
        try:
            await self.send_event(json.dumps({"event": {"contentEnd": {"promptName": self.prompt_name, "contentName": self.audio_content_name}}}))
            await self.send_event(json.dumps({"event": {"promptEnd": {"promptName": self.prompt_name}}}))
            await self.send_event(json.dumps({"event": {"sessionEnd": {}}}))
            await self.stream.input_stream.close()
        except Exception:
            pass
        self.is_active = False


async def main():
    print("=" * 60)
    print("Nova 2 Sonic File Input Test")
    print("=" * 60)

    if not Path(SPEECH_FILE).exists():
        print(f"❌ Speech file not found: {SPEECH_FILE}")
        print("   Generate it with: echo 'Hello!' | piper --model ... --output_file /tmp/test_speech.wav")
        print("   Then convert: ffmpeg -i /tmp/test_speech.wav -ar 16000 -ac 1 -f s16le /tmp/test_speech_16k.raw")
        return

    nova = NovaSonicFileTest()
    await nova.start_session()

    playback_task = asyncio.create_task(nova.play_audio())

    # Send the pre-recorded speech
    await nova.send_audio_file(SPEECH_FILE)

    # Keep streaming silence and wait for response
    print("\n⏳ Waiting for response...")

    # Continue sending silence while waiting (Nova expects continuous audio stream)
    silence = b"\x00" * (CHUNK_SIZE * 2)
    b64_silence = base64.b64encode(silence).decode("utf-8")

    for i in range(150):  # ~10s more of silence
        if nova.got_audio or nova.text_responses:
            # Got a response! Wait for it to finish
            print("   Response detected, waiting for completion...")
            await asyncio.sleep(5)
            break
        await nova.send_event(json.dumps({
            "event": {
                "audioInput": {
                    "promptName": nova.prompt_name,
                    "contentName": nova.audio_content_name,
                    "content": b64_silence,
                }
            }
        }))
        await asyncio.sleep(CHUNK_SIZE / INPUT_SAMPLE_RATE)

    # Shutdown
    print("\n🛑 Shutting down...")
    nova.is_active = False
    await nova.end_session()

    if nova.response and not nova.response.done():
        nova.response.cancel()
    if not playback_task.done():
        playback_task.cancel()
    await asyncio.gather(nova.response, playback_task, return_exceptions=True)

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Events received: {nova.events_received}")
    print(f"Event types: {nova.all_events}")
    print(f"Text responses: {len(nova.text_responses)}")
    print(f"Got audio: {nova.got_audio}")
    if nova.text_responses:
        print(f"Response: {' '.join(nova.text_responses)}")
    success = nova.got_audio or len(nova.text_responses) > 0
    print(f"Result: {'✅ PASS' if success else '❌ FAIL (no content response)'}")


if __name__ == "__main__":
    asyncio.run(main())
