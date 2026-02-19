"""Nova Sonic Voice Agent.

Real-time speech-to-speech conversations using Amazon Nova 2 Sonic.
"""

from nova_sonic.session import (
    ConversationTurn,
    NovaSonicConfig,
    NovaSonicSession,
    SessionMetrics,
    SessionState,
)
from nova_sonic.agent import NovaSonicVoiceAgent
from nova_sonic.audio import detect_mic
from nova_sonic.discord_bridge import (
    DiscordAudioSink,
    DiscordAudioSource,
    NovaSonicBridge,
)

__all__ = [
    "ConversationTurn",
    "DiscordAudioSink",
    "DiscordAudioSource",
    "NovaSonicConfig",
    "NovaSonicBridge",
    "NovaSonicSession",
    "NovaSonicVoiceAgent",
    "SessionMetrics",
    "SessionState",
    "detect_mic",
]
