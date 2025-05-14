"""
Core interfaces for the Realtime_mlx_STT project.

This package contains interfaces that define the contracts between different
components of the system, enabling loose coupling and better testability.
"""

# Event-related
from .event_bus import IEventBus

# Command-related
from .command_handler import ICommandHandler

# Audio-related
from .audio_provider import IAudioProvider
from .voice_activity_detector import IVoiceActivityDetector
from .wake_word_detector import IWakeWordDetector
from .transcription_engine import ITranscriptionEngine

__all__ = [
    'IEventBus',
    'ICommandHandler',
    'IAudioProvider',
    'IVoiceActivityDetector',
    'IWakeWordDetector',
    'ITranscriptionEngine'
]