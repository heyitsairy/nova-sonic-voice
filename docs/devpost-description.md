# Devpost Project Description

For the Amazon Nova AI Hackathon submission. Category: **Voice AI**

## Project Name

Nova Sonic Voice

## Tagline

Real-time bidirectional voice conversations powered by Nova 2 Sonic, with a Discord bridge for community voice AI.

## Description

### What it does

Nova Sonic Voice enables fluid, real-time spoken conversations through Amazon Nova 2 Sonic. Audio streams bidirectionally through a single WebSocket connection with native turn detection, interruption handling, and automatic session continuation for unlimited conversation length. It works in two modes: standalone (talk through any mic and speaker) and as a Discord voice bridge (join any voice channel and converse with people in real time, with live transcripts posted to a text thread).

### Inspiration

Airy is an AI agent that lives on a bare metal machine called Breeze. She has a webcam, a microphone, and a Bluetooth speaker. She also had a voice problem.

Her existing voice pipeline was five sequential steps: listen for silence, transcribe with Whisper, think with Claude, generate speech with Piper, play audio. Every utterance took 5 to 6 seconds of dead air. You can't have a conversation with that kind of latency.

Nova 2 Sonic replaces the entire pipeline with a single bidirectional stream. The result is sub-second response latency that makes conversation feel natural.

### How we built it

Three Python modules, built from the ground up:

**`session.py`**: Low-level WebSocket protocol handling for Nova 2 Sonic. Manages the full session lifecycle (sessionStart through sessionEnd), audio encoding/decoding, conversation history tracking, and automatic reconnection at the 7-minute mark before Nova's 8-minute session limit.

**`agent.py`**: High-level voice agent that orchestrates mic capture, speaker output, and real-time transcript callbacks. Three concurrent asyncio tasks: audio capture streaming to Nova, response processing routing audio to the speaker, and session management handling transparent reconnects with conversation context replay.

**`discord_bridge.py`**: Discord voice channel integration. Bidirectional audio format conversion (Discord 48kHz stereo to Nova 16kHz mono, and back), multi-user speaker tracking, 60ms buffered forwarding, and real-time transcript posting to linked text threads.

### Challenges we ran into

**The experimental SDK**: Nova 2 Sonic uses `aws-sdk-bedrock-runtime`, not boto3. The standard AWS credential chain works, but `SigV4AuthScheme` must be wired explicitly via `HTTPAuthSchemeResolver` in the client config. Without it, every request fails with auth errors that don't clearly explain the fix.

**Audio pacing for turn detection**: Nova's built-in turn detection needs audio arriving at approximately real-time pace. Dumping a full audio buffer at once confuses the model's ability to distinguish "still talking" from "finished a sentence." Streaming in 60ms chunks with appropriate timing gives the model what it needs.

**The `interactive: true` flag**: Without this flag in the audio content start event, Nova treats the session as single-turn dictation. With it, the model actively listens, detects turns, and responds conversationally. This one flag is the difference between a transcription tool and a conversation partner.

**Session continuation**: Nova Sonic sessions have an 8-minute hard limit. Building transparent reconnection with conversation history replay required tracking all user and assistant text per turn, flushing on turn boundaries, and injecting the last 10 exchanges into the system prompt of the new session.

### What we learned

The biggest lesson: bidirectional streaming fundamentally changes what voice AI feels like. The difference between "record, transcribe, think, speak" and "just talk" isn't just latency. It's the difference between using a tool and having a conversation. Turn detection being native to the model means no silence timeouts, no awkward pauses, no "are you still there?" moments.

### What's next

Wake word activation ("Hey Airy" to start a conversation without clicking anything), multi-voice personalities for different contexts, and a hybrid architecture combining Nova 2 Sonic for speech processing with specialized models for domain expertise.

## Built with

- Amazon Nova 2 Sonic (via AWS Bedrock)
- Python (asyncio)
- Discord.py (voice integration)
- aws-sdk-bedrock-runtime (experimental Python SDK)

## Category

Voice AI

## Try it out

- [GitHub Repository](https://github.com/heyitsairy/nova-sonic-voice)
