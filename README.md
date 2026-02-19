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
3. **Session manager**: Handles 8-min reconnects, session lifecycle

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

### Quick test (pre-recorded audio file)

```bash
# Generate a test speech clip
echo "Hello, can you hear me?" | piper --model path/to/model.onnx --output_file /tmp/test.wav
ffmpeg -i /tmp/test.wav -ar 16000 -ac 1 -f s16le /tmp/test_speech_16k.raw

# Run the file test
python tests/manual/test_nova_sonic_file.py
```

### Real-time mic conversation

```bash
python tests/manual/test_nova_sonic.py
# Speak into your mic, press Ctrl+C to stop
```

### As a library

```python
from nova_sonic import NovaSonicVoiceAgent, NovaSonicConfig

agent = NovaSonicVoiceAgent(
    config=NovaSonicConfig(voice_id="matthew"),
    on_assistant_text=lambda role, text: print(f"Assistant: {text}"),
)
await agent.start()
# ... conversation happens ...
await agent.stop()
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Technical details

- **Model**: `amazon.nova-2-sonic-v1:0`
- **SDK**: `aws-sdk-bedrock-runtime` (experimental Python SDK, not boto3)
- **Auth**: SigV4 via `HTTPAuthSchemeResolver` + `SigV4AuthScheme(service="bedrock")`
- **Input**: 16kHz / 16-bit / mono LPCM, base64 encoded
- **Output**: 24kHz / 16-bit / mono LPCM, base64 encoded
- **Session limit**: 8 minutes (auto-reconnect planned)
- **Turn detection**: Built into the model; send silence after speech to trigger

## License

MIT
