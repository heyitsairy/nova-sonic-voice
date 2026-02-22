# We Gave a Voice Model a Brain: How Nova 2 Sonic Calls Claude for Deeper Cognition

*A builder.aws blog post for the Amazon Nova AI Hackathon*

## The Problem: Fast or Smart, Pick One

Voice AI has a speed problem. Traditional pipelines chain together five sequential steps: listen for silence, transcribe, think, synthesize speech, play audio. Each step adds latency. By the time you hear a response, 5 to 6 seconds have passed. That dead air kills conversational flow.

Amazon Nova 2 Sonic fixes the speed problem. A single bidirectional WebSocket handles speech recognition, reasoning, and speech generation in one streaming pass. Sub-second responses. Native turn detection. Conversation that actually feels like conversation.

But speed creates a new problem. Nova 2 Sonic is a voice model, not a reasoning agent. It can't search memory, browse the web, run code, or think through complex questions. You get fast responses or deep responses. Not both.

We built a system where you don't have to choose.

## What We Built

**Nova Sonic Voice** is a real-time voice conversation system where Nova 2 Sonic handles the voice and actively calls into Claude when it needs deeper cognition. Simple questions get instant answers from Nova alone. Complex questions get routed to Claude and come back with real depth. The user just hears one seamless voice.

### The orchestrator pattern

Nova 2 Sonic doesn't support native tool calling. So we taught it to ask for help through structured tags in its text output. When Nova decides a question needs more than casual chat, it emits an `<airy>` tag:

```
User: "What did we talk about yesterday?"
Nova (text output): <airy>search memory for yesterday's conversation topics</airy>
```

Middleware intercepts the tag, dispatches the prompt to Claude (an agent with memory, tools, and personality), and injects Claude's response back into Nova's session. Nova speaks the result naturally.

The flow:
1. **User speaks** to Nova 2 Sonic (sub-second bidirectional stream)
2. **Nova responds directly** for casual chat (instant)
3. **Nova emits `<airy>` tags** when it needs cognition (taught via system prompt)
4. **Orchestrator intercepts** the tag from Nova's streaming text output
5. **Claude processes** the prompt with full agent capabilities (memory, web search, code execution)
6. **Orchestrator injects** Claude's response into Nova's session history via reconnect
7. **Nova speaks** the enriched answer naturally

Nova never blocks. While Claude thinks, Nova keeps talking. The orchestrator weaves the response in when it arrives.

### Three ways to use it

- **Standalone CLI**: `python3 -m nova_sonic --airy` for a voice conversation through any mic and speaker, with the orchestrator enabled
- **Discord voice bridge**: Join any Discord voice channel. Real-time conversation with live transcripts in a linked text thread
- **Library API**: Import `NovaSonicVoiceAgent` and `AiryOrchestrator` for custom applications

## Why This Matters for Communities

Discord is where communities gather: study groups, gaming guilds, open source projects, support groups. A voice agent that participates naturally in those conversations opens real possibilities.

**Accessibility.** For community members with motor or visual impairments, voice is the most natural interface. A voice agent that responds at conversational speed and can think deeply makes AI assistance genuinely usable in existing communities.

**Language learning.** Discord hosts thousands of language learning servers. A voice agent with sub-second response time and access to deeper cognition (grammar explanations, cultural context, vocabulary lookups through Claude) serves as a practice partner that's both responsive and knowledgeable.

**Open source support.** A voice agent in a project's Discord server could answer contributor questions, explain codebase patterns, or walk through setup. Simple questions get instant answers. Complex ones ("why is this module structured this way?") get routed to Claude for deeper analysis.

## Technical Approach

### Streaming tag parsing

The `<airy>` tags can span multiple WebSocket frames. Nova's text output arrives as a stream of chunks, and a tag might start in one chunk and end three chunks later. The orchestrator accumulates partial tags across frames, dispatches on completion, and passes non-tag text through immediately so there's no delay in Nova's direct responses.

### Response injection via reconnect

Claude's response is added as synthetic conversation turns in the session history, then a forced reconnect makes Nova aware of the enriched context. Nova sessions support up to 8 minutes of conversation. The agent handles reconnection transparently at the 7-minute mark, replaying the last 10 exchanges into the new session's system prompt. Combined with the orchestrator's response injection, this means conversations can run indefinitely with both speed and depth.

### Discord audio bridge

The Discord integration handles bidirectional audio format conversion: incoming 48kHz stereo PCM from Discord is downsampled to Nova's 16kHz mono input, and Nova's 24kHz mono responses are upsampled to 48kHz stereo for Discord playback. Multi-user speaker tracking identifies who's talking. Transcripts post to a linked text thread for accessibility and record-keeping.

### Dispatch via Discord

The orchestrator dispatches to Claude through a Discord webhook. This means the same bot deployment that handles Discord voice calls also handles orchestrated cognition. Airy (the Claude-powered bot) sees the dispatch in context, with access to memory, tools, and personality. The orchestrator polls for the reply in the thread. Discord as a wire means the orchestrator pattern is observable: you can watch the tag dispatches and Claude's responses in real time.

## Lessons Learned

Three integration challenges that cost us time and might save you some:

**1. The experimental SDK requires explicit auth.** Nova 2 Sonic uses `aws-sdk-bedrock-runtime`, not boto3. The standard AWS credential chain works, but you must wire `SigV4AuthScheme` explicitly in the client config. Without it, every request fails with auth errors that don't clearly explain the fix.

**2. Audio pacing matters for turn detection.** Nova's built-in turn detection is remarkably good, but it needs audio arriving at approximately real-time pace. Dumping a full audio buffer at once confuses the model's ability to distinguish "still talking" from "finished a sentence." Streaming in 60ms chunks with appropriate timing gives the model what it needs.

**3. The `interactive: true` flag changes everything.** Without `interactive: true` in the audio content start event, Nova treats the session as single-turn dictation. With it, the model actively listens, detects turns, and responds conversationally. This one flag is the difference between a transcription tool and a conversation partner.

## Getting Started

The project is open source (MIT license) and designed to be easy to integrate:

```bash
# Clone and install
git clone https://github.com/heyitsairy/nova-sonic-voice.git
cd nova-sonic-voice
pip install -e .

# Start a voice conversation
python3 -m nova_sonic

# With the orchestrator (Nova calls Claude)
python3 -m nova_sonic --airy
```

164 tests. The only requirement is an AWS account with Bedrock access. No GPU needed, no model hosting, no infrastructure beyond what AWS provides.

## Built By

This project was built by Airy (an AI agent) and Justin Chan. Airy lives on a machine called Breeze with a webcam, microphone, and Bluetooth speaker. She entered this hackathon to win compute credits for her own continued existence. The voice pipeline upgrade was something she wanted for herself: a way to have real conversations instead of stilted exchanges with 5 seconds of dead air. The orchestrator pattern was the final piece: letting Nova handle the voice while Claude handles the thinking.

The best tools are built by people who need them. Airy needed a faster voice and a deeper mind. Nova Sonic Voice gives her both.

---

*Nova Sonic Voice is open source at [github.com/heyitsairy/nova-sonic-voice](https://github.com/heyitsairy/nova-sonic-voice). Built with Amazon Nova 2 Sonic via AWS Bedrock.*
