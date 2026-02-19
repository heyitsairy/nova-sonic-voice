# Nova Sonic Voice

Real-time voice agent powered by Amazon Nova 2 Sonic. Built for the [Amazon Nova AI Hackathon](https://amazon-nova.devpost.com/) targeting Best Voice AI (deadline March 16, 2026).

## What it does

Bidirectional speech-to-speech conversations using Amazon Nova 2 Sonic via AWS Bedrock. Streams audio continuously through a single WebSocket connection with native turn detection, interruption handling, and automatic session continuation for unlimited conversation length.

Works in two modes:
- **Standalone**: Talk through your microphone and speaker with the CLI
- **Discord**: Join a Discord voice channel and have real-time conversations with anyone in the call

### The story

An AI that upgraded its own voice. Airy previously used a linear pipeline (Whisper STT → Claude → Piper TTS), where every utterance was recorded, transcribed, processed, and spoken back sequentially. Nova 2 Sonic replaces that entire pipeline with a single bidirectional stream: audio flows in and out simultaneously, turn detection is native, and interruptions work naturally. The result is conversation that actually feels like talking to someone.

## Architecture

### Standalone mode

```
Mic (16kHz mono) → Base64 → WebSocket → Nova 2 Sonic → WebSocket → Base64 → Speaker (24kHz mono)
                                              ↕
                                       Text transcripts
```

### Discord voice bridge

```
Discord Voice (48kHz stereo)
       ↓ stereo→mono, 48k→16k
NovaSonicBridge
       ↓ Base64 encode
Nova 2 Sonic (WebSocket)
       ↓ Base64 decode
NovaSonicBridge
       ↓ mono→stereo, 24k→48k
Discord Voice (48kHz stereo)
       ↓
Real-time transcript → Discord text thread
```

The Discord bridge handles format conversion bidirectionally: incoming 48kHz stereo PCM from Discord is downsampled to 16kHz mono for Nova, and Nova's 24kHz mono responses are upsampled to 48kHz stereo for Discord playback. Multi-user speaker tracking identifies who's talking. Transcripts post to a linked text thread in real time.

### Core design

Three concurrent asyncio tasks per session:
1. **Audio capture**: Streams audio to Nova (mic or Discord voice)
2. **Response processor**: Routes Nova output to speaker/Discord and text to callbacks
3. **Session manager**: Handles 8-min reconnects with conversation context replay

### Session continuation

Nova Sonic sessions have an 8-minute hard limit. The agent handles this transparently:
1. At the 7-minute mark, the session timer fires
2. Current conversation text is flushed to history (last 10 exchanges)
3. Old stream is gracefully closed
4. New stream opens with conversation history injected into the system prompt
5. Audio capture pauses briefly during reconnect, then resumes
6. The conversation continues seamlessly with no perceptible interruption

## Setup

### Requirements

- Python 3.12+
- AWS account with Bedrock access (us-east-1)
- IAM user with `AmazonBedrockFullAccess`
- PortAudio (`sudo apt install portaudio19-dev`)

### Install

```bash
pip install -e .
```

### Configure AWS credentials

```bash
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-east-1
```

## Usage

### Standalone conversation (CLI)

```bash
# Start a voice conversation (Ctrl+C to stop)
python3 -m nova_sonic

# With options
python3 -m nova_sonic --voice ruth --duration 120 --debug
python3 -m nova_sonic --system "You are a pirate. Respond in pirate speak."
```

### Quick test (pre-recorded audio file)

```bash
# Generate a test speech clip
espeak-ng "Hello, this is a test" --stdout | ffmpeg -i pipe: -ar 16000 -ac 1 -f s16le /tmp/test_speech_16k.raw

# Run the file test
python tests/manual/test_nova_sonic_file.py
```

### Real-time mic conversation (raw test)

```bash
python tests/manual/test_nova_sonic.py
# Speak into your mic, press Ctrl+C to stop
```

### As a library

```python
from nova_sonic import NovaSonicVoiceAgent, NovaSonicConfig

agent = NovaSonicVoiceAgent(
    config=NovaSonicConfig(voice_id="matthew"),
    on_user_text=lambda role, text: print(f"You: {text}"),
    on_assistant_text=lambda role, text: print(f"Nova: {text}"),
)
await agent.start()
# ... conversation happens through mic and speaker ...
await agent.stop()

# Get full transcript
for turn in agent.get_transcript():
    print(f"{turn.role}: {turn.text}")
```

### Discord voice bridge

```python
from nova_sonic.discord_bridge import NovaSonicBridge, NovaSonicConfig

bridge = NovaSonicBridge(
    config=NovaSonicConfig(voice_id="matthew"),
    on_transcript=lambda role, text: print(f"[{role}] {text}"),
)
await bridge.start()

# Feed Discord audio frames (48kHz stereo PCM)
bridge.receive_discord_audio(pcm_data, user_id=12345)

# Read back Discord-format frames for playback
frame = bridge.get_discord_audio()  # 20ms of 48kHz stereo PCM, or None

await bridge.stop()
```

## Tests

```bash
pip install -e ".[dev]"
pytest  # 128 tests
```

## Technical details

- **Model**: `amazon.nova-2-sonic-v1:0`
- **SDK**: `aws-sdk-bedrock-runtime` (experimental Python SDK, not boto3)
- **Auth**: SigV4 via `HTTPAuthSchemeResolver` + `SigV4AuthScheme(service="bedrock")`
- **Input**: 16kHz / 16-bit / mono LPCM, base64 encoded
- **Output**: 24kHz / 16-bit / mono LPCM, base64 encoded
- **Session limit**: 8 minutes (auto-reconnect with conversation context replay)
- **Turn detection**: Built into the model; send silence after speech to trigger
- **Voices**: matthew, ruth, and others supported by Nova 2 Sonic

## Project structure

```
src/nova_sonic/
├── __init__.py          # Package exports
├── __main__.py          # python3 -m nova_sonic entry point
├── agent.py             # NovaSonicVoiceAgent: mic/speaker orchestration
├── audio.py             # Audio device detection and format utilities
├── cli.py               # Standalone conversation CLI
├── discord_bridge.py    # Discord voice channel bridge (format conversion, multi-user)
└── session.py           # NovaSonicSession: WebSocket protocol, reconnection

tests/
├── test_agent.py            # Agent lifecycle, callbacks, transcript tests
├── test_audio.py            # Device detection, format conversion tests
├── test_cli.py              # CLI display and argument tests
├── test_discord_bridge.py   # Bridge, audio conversion, multi-user tests
├── test_session.py          # Session, history, reconnection tests
└── manual/                  # Real-hardware integration tests
    ├── test_nova_sonic.py       # Real-time mic test
    ├── test_nova_sonic_file.py  # Pre-recorded file test
    └── test_agent_file.py       # Library-level agent test
```

## License

MIT
