# Demo Video Script

3-minute demo video for Amazon Nova AI Hackathon submission.

## Target Category

**Best Voice AI** ($3,000 + $5,000 AWS credits)

## Narrative

"We gave a voice model a brain." Nova 2 Sonic handles real-time voice. When it needs to think deeper, it calls Claude. The demo shows: the old slow pipeline, the Nova upgrade for speed, and then the new orchestrator pattern where Nova actively calls into Claude for cognition.

## Script

### Opening (0:00 to 0:20)

**Visual:** Title card with project name, then cut to Breeze (the machine) with the Bluetooth speaker visible.

**Voiceover (Justin):**
"This is Airy, an AI agent that lives on a machine called Breeze. She has a webcam, a mic, and a speaker. She needed two things: a fast voice, and a way to think while she talks."

### The Problem (0:20 to 0:45)

**Visual:** Screen recording showing the old Piper pipeline. Terminal with timestamps showing the latency: record (1.5s silence detection) then transcribe (Whisper, ~2s) then generate (Claude, ~1s) then speak (Piper TTS, ~1s). Total: ~5 to 6 seconds.

**Voiceover:**
"The old pipeline was five serial steps. Listen, transcribe, think, generate speech, play. Every exchange took 5 to 6 seconds of dead air. And voice models like Nova 2 Sonic fix the speed problem, but they can't search memory, browse the web, or reason deeply. Fast or smart. Pick one."

### The Solution (0:45 to 1:10)

**Visual:** Architecture diagram showing the dual-model flow. Nova 2 Sonic bidirectional stream on top, with a branch showing `<airy>` tags being intercepted, dispatched to Claude, and responses injected back.

**Voiceover:**
"What if you didn't have to pick? Nova 2 Sonic handles the voice: sub-second latency, native turn detection, natural conversation. When it needs deeper cognition, it calls Claude through a tag protocol. Nova plans what to ask. Claude thinks and responds. The answer flows back into the conversation. The user just hears a seamless voice that's both fast and smart."

### Live Demo: Nova Calls Airy (1:10 to 2:10)

**Visual:** Terminal showing the voice agent starting with the orchestrator enabled. Camera on the speaker. Real-time transcript on screen showing both spoken text and `<airy>` tag dispatches.

**Action:** Justin has a 45 to 60 second conversation. Start with casual chat (Nova handles directly), then ask something that triggers an Airy call:

- "Hey, how's it going?" (Nova responds directly, sub-second)
- "What did we talk about yesterday?" (Nova emits `<airy>search memory for yesterday's topics</airy>`, Claude responds, Nova speaks the enriched answer)
- "What's the weather like right now?" (Another `<airy>` dispatch)

**Key moments to capture:**
- Fast direct responses for simple chat (Nova alone)
- The `<airy>` tag appearing in the transcript when Nova calls Claude
- Claude's response being injected and Nova speaking it naturally
- The contrast: casual chat is instant, complex questions take a beat but come back with real answers

### Live Demo: Discord (2:10 to 2:40)

**Visual:** Discord open. Voice channel visible. Join the voice channel, conversation flows.

**Action:** Short exchange through Discord voice showing the same pattern works in a real platform. Ask something that triggers an Airy dispatch so the tag interception is visible in the text transcript thread.

**Key moments:**
- Joining voice shows "Nova Sonic Call" thread created
- Real-time transcript posting including `[Airy responded: ...]` entries
- Audio quality through Discord's pipeline

### Wrap (2:40 to 3:00)

**Visual:** GitHub repo, test count (164 tests), project structure showing the five modules.

**Voiceover:**
"Nova Sonic Voice is open source. 164 tests. Session continuation for unlimited conversation length. A Discord bridge for real-time voice in communities. And an orchestrator that lets Nova call Claude when it needs to think. Built by Airy and Justin."

**End card:** GitHub link, hackathon category.

## Technical Notes for Recording

- Orchestrator must be wired into the session (use the demo entry point that enables it)
- Set `ANTHROPIC_API_KEY` in environment for Claude dispatch
- Set AWS creds for Nova (`pass show airy-bedrock/access-key-id`, etc.)
- BT speaker must be connected and set as default sink
- C920 mic on correct ALSA device
- For Discord segment: set `VOICE_BACKEND=nova` in environment
- Record with OBS or similar screen capture
- Record audio separately for cleaner voiceover (edit together in post)
- Consider showing the terminal/transcript alongside the speaker for visual proof of tag dispatches

## Prep Checklist

- [ ] Orchestrator end-to-end test passes (Nova actually emits tags, Claude responds)
- [ ] Standalone conversation with orchestrator works cleanly
- [ ] Discord voice call works with orchestrator enabled
- [ ] Screen recording software set up
- [ ] Camera angle on speaker arranged
- [x] Architecture diagram created (docs/architecture-diagram.png)
- [x] Updated architecture diagram showing orchestrator flow
- [ ] Edit video to under 3 minutes
- [ ] Upload to YouTube/Vimeo (Devpost requires video link)
