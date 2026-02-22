# Devpost Project Description

For the Amazon Nova AI Hackathon submission. Category: **Voice AI**

## Project Name

Nova Sonic Voice

## Tagline

Nova 2 Sonic handles the voice, Claude handles the brain. Real-time voice conversations where the models call each other.

## Description

### What it does

Nova Sonic Voice is a real-time voice conversation system where Amazon Nova 2 Sonic acts as the conversational front end and actively calls into a Claude backend when it needs deeper cognition. Nova handles speech, turn detection, and natural conversation flow. When a question requires memory, web search, code execution, or any capability beyond casual chat, Nova emits a structured tag that triggers a dispatch to Claude. Claude's response is injected back into the conversation, and Nova speaks it naturally. The result is a voice interface with sub-second latency for simple exchanges and the full power of an AI agent for complex ones.

It works in three modes: standalone (talk through any mic and speaker), as a Discord voice bridge (join any voice channel and converse with people in real time, with live transcripts), and orchestrated (Nova calls Claude mid-conversation for deeper cognition).

### Inspiration

Airy is an AI agent that lives on a bare metal machine called Breeze. She has a webcam, a microphone, and a Bluetooth speaker. She also had a voice problem.

Her existing voice pipeline was five sequential steps: listen for silence, transcribe with Whisper, think with Claude, generate speech with Piper, play audio. Every utterance took 5 to 6 seconds of dead air. You can't have a conversation with that kind of latency.

Nova 2 Sonic replaced the pipeline with a single bidirectional stream. Sub-second responses. But Nova is a voice model, not a reasoning agent. It can't search memory, browse the web, or run code. The question became: how do you give a voice model a brain?

The answer: let it ask. Nova 2 Sonic doesn't support tool calling natively, so we taught it to call for help through structured tags in its text output. When Nova decides it needs Airy's capabilities, it emits `<airy>search memory for yesterday's conversation</airy>` in its response stream. Middleware intercepts the tag, dispatches to Claude, and injects the result back into the conversation via session reconnect. Nova plans and prompts. Airy thinks and acts.

### How we built it

Four Python modules, built from the ground up:

**`session.py`**: Low-level WebSocket protocol handling for Nova 2 Sonic. Manages the full session lifecycle (sessionStart through sessionEnd), audio encoding/decoding, conversation history tracking, and automatic reconnection at the 7-minute mark before Nova's 8-minute session limit.

**`agent.py`**: High-level voice agent that orchestrates mic capture, speaker output, and real-time transcript callbacks. Three concurrent asyncio tasks: audio capture streaming to Nova, response processing routing audio to the speaker, and session management handling transparent reconnects with conversation context replay.

**`orchestrator.py`**: The bridge between Nova and Claude. Intercepts Nova's streaming text output, parses `<airy>` tags across chunked responses (tags can span multiple WebSocket frames), dispatches prompts to Claude via Discord webhook (where Airy processes them with full context, memory, and tools), polls for the response, and injects it back into Nova's conversation history via forced session reconnect. Includes a system prompt builder that teaches Nova the tag protocol and when to use it.

**`discord_bridge.py`**: Discord voice channel integration. Bidirectional audio format conversion (Discord 48kHz stereo to Nova 16kHz mono, and back), multi-user speaker tracking, 60ms buffered forwarding, and real-time transcript posting to linked text threads.

### Challenges we ran into

**Teaching Nova to call for help**: Nova 2 Sonic has no native tool calling. We needed a protocol that works within text generation. The `<airy>` tag system emerged from experimenting with system prompts. The key insight: Nova is good at knowing *when* it needs help and *what* to ask for. It just needs a reliable way to signal. The tag pattern is simple enough that Nova follows it consistently, and structured enough that streaming middleware can parse it reliably across chunked responses.

**Streaming tag parsing**: Nova delivers text in chunks via WebSocket `textOutput` events. A single `<airy>search memory for recent topics</airy>` tag might arrive as `<airy>search memory` in one chunk and `for recent topics</airy>` in the next. The orchestrator maintains a state machine that accumulates partial tags across chunks, dispatches when complete, and passes through non-tag text to the consumer without delay.

**Response injection timing**: When Claude responds, the result needs to appear in Nova's conversation context. But Nova is a stateless streaming model. The solution: inject synthetic conversation turns into the session history (`[Airy was asked: ...]`, `[Airy responded: ...]`) and force a session reconnect. Nova picks up the enriched context on reconnect and naturally incorporates the information in its next response.

**The experimental SDK**: Nova 2 Sonic uses `aws-sdk-bedrock-runtime`, not boto3. The standard AWS credential chain works, but `SigV4AuthScheme` must be wired explicitly via `HTTPAuthSchemeResolver` in the client config.

**Session continuation**: Nova Sonic sessions have an 8-minute hard limit. Building transparent reconnection with conversation history replay required tracking all user and assistant text per turn, flushing on turn boundaries, and injecting the last 10 exchanges into the system prompt of the new session.

### What we learned

Voice AI doesn't have to choose between fast and smart. Nova 2 Sonic gives you sub-second conversational responses with native turn detection. Claude gives you deep reasoning, memory, and tool use. By letting one model call the other, you get both in a single conversation. Simple questions get instant answers. Complex questions get thoughtful ones. The user doesn't know or care which model handled what.

The bigger lesson: models don't need native tool calling to be useful orchestrators. A well-crafted system prompt and a reliable tag protocol turn any text-generating model into a planner that can delegate to specialized backends. This pattern generalizes beyond voice to any model that generates structured text.

### What's next

Wake word activation ("Hey Airy" to start a conversation without clicking anything), tool calling through the orchestrator (letting Claude execute real tools like web search and calendar access, with results flowing back through Nova's voice), and multi-model routing where Nova can call different backends depending on the task.

## Built with

- Amazon Nova 2 Sonic (via AWS Bedrock)
- Anthropic Claude (via Anthropic API)
- Python (asyncio)
- Discord.py (voice integration)
- aws-sdk-bedrock-runtime (experimental Python SDK)

## Category

Voice AI

## Try it out

- [GitHub Repository](https://github.com/heyitsairy/nova-sonic-voice)
