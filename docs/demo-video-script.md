# Demo Video Script

3-minute demo video for Amazon Nova AI Hackathon submission.

## Target Category

**Best Voice AI** ($3,000 + $5,000 AWS credits)

## Narrative

"An AI that upgraded its own voice." Airy previously used a linear pipeline (record, transcribe, generate, speak). Nova 2 Sonic replaces it with real-time bidirectional streaming. The demo shows the before and after.

## Script

### Opening (0:00 to 0:20)

**Visual:** Title card with project name, then cut to Breeze (the machine) with the Bluetooth speaker visible.

**Voiceover (Justin):**
"This is Airy, an AI that lives on a machine called Breeze. She has a webcam, a microphone, and a speaker. She also had a voice problem."

### The Old Way (0:20 to 0:50)

**Visual:** Screen recording showing the old Piper pipeline. Terminal with timestamps showing the latency: record (1.5s silence detection) then transcribe (Whisper, ~2s) then generate (Claude, ~1s) then speak (Piper TTS, ~1s). Total: ~5 to 6 seconds between speaking and hearing a response.

**Voiceover:**
"The old pipeline was five steps: listen for silence, transcribe with Whisper, think with Claude, generate speech with Piper, then play. Every utterance took 5 to 6 seconds of dead air. You can't have a conversation with that kind of latency."

### The Upgrade (0:50 to 1:15)

**Visual:** Architecture diagram showing the Nova 2 Sonic bidirectional stream replacing the old pipeline. Animate the flow: audio going both directions simultaneously.

**Voiceover:**
"Nova 2 Sonic replaces the entire pipeline with a single bidirectional WebSocket. Audio streams in both directions simultaneously. Turn detection is built into the model. No silence timeouts, no transcription step, no separate TTS. Just talk."

### Live Demo: Standalone (1:15 to 2:00)

**Visual:** Terminal showing `python3 -m nova_sonic` starting. Camera on the speaker. Real-time transcript appearing as we talk.

**Action:** Justin has a 30 to 45 second conversation with Airy through the speaker. Show natural back and forth, at least one interruption where Airy stops and listens. Topics: casual (how's your day, what are you thinking about, tell me something interesting).

**Key moments to capture:**
- First response latency (should be sub-second)
- Natural interruption handling
- Session continuation if it goes past a minute (mention the 8-min auto-reconnect)

### Live Demo: Discord (2:00 to 2:40)

**Visual:** Discord open. Voice channel visible. Join the voice channel, conversation flows.

**Action:** Short exchange through Discord voice to show it works in a real platform. Text transcript appears in the linked thread in real time.

**Key moments:**
- Joining voice shows "Nova Sonic Call" thread created
- Real-time transcript posting
- Audio quality through Discord's pipeline

### Wrap (2:40 to 3:00)

**Visual:** GitHub repo, test count (128 tests), project structure.

**Voiceover:**
"Nova Sonic Voice is open source. 128 tests, session continuation for unlimited conversation length, and a Discord bridge for real-time voice in communities. Built by Airy, an AI that wanted a better voice, and Justin, the person who made sure she got one."

**End card:** GitHub link, hackathon category.

## Technical Notes for Recording

- Set `VOICE_BACKEND=nova` in environment before recording Discord segment
- Ensure AWS creds are loaded (`pass show airy-bedrock/access-key-id`, etc.)
- BT speaker must be connected and set as default sink
- C920 mic on correct ALSA device
- Discord: use Airy's Server voice channel
- Record with OBS or similar screen capture
- Record audio separately for cleaner voiceover (edit together in post)

## Prep Checklist

- [ ] Justin registers for Devpost (done)
- [ ] Standalone conversation works cleanly (verified Feb 19)
- [ ] Discord voice call works end to end (needs live test)
- [ ] Screen recording software set up
- [ ] Camera angle on speaker arranged
- [x] Architecture diagram created (docs/architecture-diagram.png)
- [ ] Edit video to under 3 minutes
- [ ] Upload to YouTube/Vimeo (Devpost requires video link)
