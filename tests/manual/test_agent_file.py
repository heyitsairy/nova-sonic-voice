"""Test the NovaSonicVoiceAgent with a pre-recorded speech file.

Verifies the library-level agent (not the raw manual test) works end-to-end
with real AWS credentials and actual audio.

Usage:
    AWS_ACCESS_KEY_ID=$(pass aws/bedrock/access-key-id) \\
    AWS_SECRET_ACCESS_KEY=$(pass aws/bedrock/secret-access-key) \\
    AWS_DEFAULT_REGION=us-east-1 \\
    python3 tests/manual/test_agent_file.py
"""

import asyncio
import time
from pathlib import Path

from nova_sonic import NovaSonicVoiceAgent, NovaSonicConfig
from nova_sonic.session import INPUT_SAMPLE_RATE, CHUNK_SIZE

SPEECH_FILE = "/tmp/test_speech_16k.raw"


async def main():
    print("=" * 60)
    print("  NovaSonicVoiceAgent File Test")
    print("=" * 60)

    if not Path(SPEECH_FILE).exists():
        print(f"Missing {SPEECH_FILE}")
        print("Generate: espeak-ng 'Hello' --stdout | ffmpeg -i pipe: -ar 16000 -ac 1 -f s16le /tmp/test_speech_16k.raw")
        return

    transcripts: list[tuple[str, str]] = []
    got_audio = False

    def on_user(role, text):
        transcripts.append(("user", text))
        print(f"  You: {text}")

    def on_assistant(role, text):
        transcripts.append(("assistant", text))
        print(f"  Nova: {text}")

    config = NovaSonicConfig(voice_id="matthew")

    agent = NovaSonicVoiceAgent(
        config=config,
        on_user_text=on_user,
        on_assistant_text=on_assistant,
    )

    # Start agent WITHOUT mic capture (we'll send audio manually)
    agent._running = True
    agent._start_time = time.time()
    from nova_sonic.session import NovaSonicSession
    agent._session = NovaSonicSession(
        config=config,
        on_text=agent._text_callback,
        on_reconnect=agent._reconnect_callback,
    )
    await agent._session.start()

    print("\n  Session started. Sending pre-recorded speech...\n")

    # Read and send file
    pcm_data = Path(SPEECH_FILE).read_bytes()
    chunk_bytes = CHUNK_SIZE * 2  # 16-bit = 2 bytes per sample
    chunk_duration = CHUNK_SIZE / INPUT_SAMPLE_RATE

    for i in range(0, len(pcm_data), chunk_bytes):
        chunk = pcm_data[i:i + chunk_bytes]
        await agent._session.send_audio(chunk)
        await asyncio.sleep(chunk_duration)

    # Send silence for turn detection
    silence = b"\x00" * chunk_bytes
    for _ in range(int(3.0 / chunk_duration)):
        await agent._session.send_audio(silence)
        await asyncio.sleep(chunk_duration)

    # Wait for response
    print("\n  Waiting for response...")
    for _ in range(100):
        if transcripts:
            await asyncio.sleep(3)
            break
        await asyncio.sleep(0.1)

    # Drain audio queue to verify audio was received
    audio_chunks = 0
    while not agent._session.audio_output_queue.empty():
        await agent._session.audio_output_queue.get()
        audio_chunks += 1

    # Stop
    await agent._session.stop()
    agent._running = False

    # Summary
    history = agent.get_transcript()
    print(f"\n  Audio chunks received: {audio_chunks}")
    print(f"  Transcript entries: {len(transcripts)}")
    print(f"  History entries: {len(history)}")
    print(f"  Metrics: {agent._session.metrics}")

    success = len(transcripts) > 0 or audio_chunks > 0
    print(f"\n  Result: {'PASS' if success else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
