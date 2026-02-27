# nova-sonic-voice

Amazon Nova 2 Sonic voice orchestrator with Claude cognition. Hackathon project — deadline **March 16, 2026**.

Nova handles speech and conversation flow. When it needs reasoning, it emits `<airy>...</airy>` tags. The orchestrator intercepts, dispatches to Claude via Discord webhook, injects the reply back. Fast voice + full agent capabilities.

**PUBLIC REPO.** Never commit secrets, tokens, or internal infrastructure details.

## Key Modules (`src/nova_sonic/`)

| File | Purpose |
|------|---------|
| `orchestrator.py` | Core tag parser + dispatcher. Intercepts `<airy>` tags, posts to Discord webhook, polls for reply, injects into session history via reconnect. |
| `session.py` | Nova 2 Sonic WebSocket session management. Bidirectional audio streaming. |
| `discord_bridge.py` | Discord voice channel integration — audio format conversion (48kHz stereo ↔ 16kHz mono) |
| `audio.py` | Audio utilities (resampling, encoding, playback) |
| `agent.py` | Agent wrapper — Nova + orchestrator + session lifecycle |
| `cli.py` | CLI entrypoint (standalone and Discord modes) |

## Three Modes

1. **Standalone** (`python -m nova_sonic`): Talk through mic/speaker via terminal
2. **Discord** (`--discord`): Bot joins voice channel, conversations live in Discord
3. **Orchestrated** (`--airy`): Enables `<airy>` tag dispatch to Claude backend

## Discord Webhook

- "Nova Voice" webhook on `#airy-hacks` (ID `1474900473327784207`)
- URL in `pass show nova/webhook-url`
- Engine allowlist: `ALLOWED_WEBHOOK_IDS` in launch.sh (deployed)

## Hackathon Context

- **Competition:** Amazon Nova AI Hackathon
- **Track:** Best Voice AI
- **Devpost:** Registered (participant #2610), also registered on GitLab AI hackathon Devpost
- **Key innovation:** `<airy>` tag protocol enabling Nova to call Claude for cognition mid-conversation

## What Needs to Happen Before March 16

1. Live end-to-end test with Justin (15 min voice conversation — ask him to schedule)
2. Demo video recorded
3. `docs/devpost-description.md` finalized and submitted

## Gotchas

- Nova 2 Sonic has no native tool calling. The tag protocol is how we work around this — preserve it.
- Response injection uses forced session reconnect (not just appending). This is by design — Nova needs to see the updated context.
- The Discord webhook dispatch is async. Nova keeps talking while Claude thinks. Don't add blocking waits.
- `ALLOWED_WEBHOOK_IDS` must include the Nova Voice webhook ID or the engine will reject posts. Already deployed.

## Running

```bash
pip install -e ".[dev]"
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... python -m nova_sonic --airy
```

Secrets via `pass`. See README for full environment setup.

## Tests

```bash
pytest tests/ -v
```

164 tests. Don't run the full suite if resources are tight — target specific test files.
