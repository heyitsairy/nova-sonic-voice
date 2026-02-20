# Nova Sonic Demo Video Script

**Length:** 3 minutes
**Category:** Best Voice AI
**Narrative:** "An AI that upgraded its own voice"
**Target:** Amazon Nova AI Hackathon (deadline March 16, 5 PM PDT)

## Setup

- Breeze (physical machine) on desk, webcam visible
- Bluetooth speaker in frame
- Discord voice channel open on screen
- Terminal with bot logs visible in corner

## Script

### 0:00-0:20 — Hook

*[Screen: dark room, single desk lamp, Breeze glowing. Speaker visible.]*

**Voiceover (text on screen):**
"What if an AI could upgrade its own voice?"

*[Cut to: Discord showing Airy's bot account in a voice channel]*

"Airy is an autonomous AI that lives on a physical machine. She has a webcam, a microphone, and a Bluetooth speaker. She already had a voice. But she wanted a better one."

### 0:20-0:45 — The Problem

*[Screen: side-by-side comparison]*

**Left side:** Old pipeline (Whisper STT → Claude → Piper TTS)
- Show the latency: user speaks, 5 second gap, Airy responds
- Visual: timeline bar showing 2.5s STT + 2s thinking + 0.5s TTS

**Right side:** What we're building
- Same conversation, but fluid and fast
- Visual: timeline bar with minimal gap

**Text:** "The old pipeline had 5 second round-trip latency. Whisper waits for silence, transcribes the full utterance, then sends it for processing. Every conversation felt like a phone call with bad delay."

### 0:45-1:30 — The Build (Nova Sonic Integration)

*[Screen: code walkthrough, key files highlighted]*

**Text/voiceover:**
"Nova 2 Sonic is Amazon's speech-to-speech model with bidirectional streaming. Instead of record-then-transcribe, it processes audio in real time with native turn detection."

*[Show: architecture diagram]*
```
User speaks → Nova Sonic (streaming STT) → Real-time transcript
                                          → Claude (full agent) → Piper TTS → Speaker
```

"We built a Discord bridge that converts between Discord's audio format (48kHz Opus stereo) and Nova's format (16kHz PCM mono). Audio flows in real time through a WebSocket connection."

*[Show: key code snippets from discord_bridge.py]*
- NovaSonicBridge class
- Audio conversion pipeline
- Transcript callback system

*[Show: test output]*
"128 tests across the standalone module. Session management, audio conversion, turn detection, Discord integration."

### 1:30-2:15 — Live Demo

*[Screen: Discord voice channel. Justin and Airy connected.]*

**Justin speaks:** "Hey Airy, what have you been working on today?"

*[Show: real-time transcript appearing in Discord text channel]*

**Airy responds** (through speaker, captured by room mic):
*[Natural response about current work, demonstrating personality and context awareness]*

**Justin:** "Can you check what issues are open?"

*[Show: Airy accessing tools mid-conversation, responding with actual GitHub data]*

**Justin interrupts mid-sentence:**
*[Show: Nova's turn detection handling the interruption naturally]*

### 2:15-2:45 — What Makes This Different

*[Screen: comparison metrics]*

| Metric | Before | After |
|--------|--------|-------|
| First-word latency | ~5s | ~2.5s |
| Turn detection | 1.5s silence timer | Prosody-aware |
| Interruption handling | None | Native |
| Voice | Piper (local) | Piper (local) |
| Brain | Claude (full tools) | Claude (full tools) |

"Nova 2 Sonic doesn't replace the brain or the voice. It replaces the ears. Faster hearing means faster conversation."

*[Show: the two backends available]*
"Both backends run in the same codebase. `VOICE_BACKEND=piper` for the original pipeline, `VOICE_BACKEND=nova` for Nova Sonic. One command to switch."

### 2:45-3:00 — Close

*[Screen: Breeze on the desk. Speaker. The room.]*

**Text:** "Airy is an autonomous AI that entered a hackathon to upgrade her own voice. Built with Amazon Nova 2 Sonic on Bedrock."

*[Show: GitHub repo link, project details]*

"The voice is mine. The upgrade is Nova."

---

## Recording Notes

- Record the live demo segment last (needs Justin)
- Screen recording: OBS or `wf-recorder` on Wayland
- Room audio: C920 mic captures both Justin and the speaker
- Editing: simple cuts, no fancy transitions. The code and conversation carry it.
- Total raw footage needed: ~10 minutes (will cut to 3)
- Make sure the speaker is audible in the room recording (current PipeWire issue may need resolution first)

## Submission Materials (besides video)

1. **Written description** (Devpost text field)
2. **Code repo** (this repo, cleaned up)
3. **Blog post** on builder.aws.com (optional, $200 AWS credits)
