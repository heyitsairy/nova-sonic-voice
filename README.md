# Nova Sonic Voice

Real-time voice agent powered by Amazon Nova 2 Sonic. Built for the [Amazon Nova AI Hackathon](https://amazon-nova.devpost.com/) (deadline March 16, 2026).

## What it does

Bidirectional speech-to-speech conversations using Amazon Nova 2 Sonic via AWS Bedrock. Streams audio from a microphone to Nova and plays back responses in real time, with native turn detection and interruption handling.

### The story

An AI that upgraded its own voice. Previously used Whisper STT + Piper TTS (record → transcribe → generate → play). Nova 2 Sonic replaces that entire pipeline with a single bidirectional stream for fluid, natural conversation.

## Architecture

```
Mic (16kHz mono) → Base64 encode → WebSocket → Nova 2 Sonic → WebSocket → Base64 decode → Speaker (24kHz mono)
                                                     ↓
                                              Text transcripts
```

Three concurrent asyncio tasks per session:
1. **Mic capture**: Streams 16kHz/16bit/mono LPCM audio to Nova
2. **Response processor**: Routes audio to speaker, text to callbacks
3. **Session manager**: Handles 8-min reconnects with conversation context replay

### Session continuation

Nova Sonic sessions have an 8-minute hard limit. The agent handles this automatically:
1. At the 7-minute mark, the session timer fires
2. Current conversation text is flushed to history
3. Old stream is gracefully closed
4. New stream opens with the conversation history injected into the system prompt
5. Mic capture pauses briefly during reconnect, then resumes
6. The conversation continues seamlessly

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

## Tests

```bash
pip install -e ".[dev]"
pytest  # 59 tests
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
├── __init__.py       # Package exports
├── __main__.py       # python3 -m nova_sonic entry point
├── agent.py          # NovaSonicVoiceAgent: orchestrates mic, session, playback
├── audio.py          # Audio device detection (C920 mic, fallback)
├── cli.py            # Standalone conversation CLI
└── session.py        # NovaSonicSession: WebSocket protocol, reconnection

tests/
├── test_agent.py     # Agent unit tests
├── test_audio.py     # Audio detection tests
├── test_cli.py       # CLI display and argument tests
├── test_session.py   # Session, history, reconnection tests
└── manual/           # Real-hardware integration tests
    ├── test_nova_sonic.py       # Real-time mic test
    ├── test_nova_sonic_file.py  # Pre-recorded file test
    └── test_agent_file.py       # Library-level agent test
```

## License

MIT
