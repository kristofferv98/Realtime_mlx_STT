"""
Core module for the Realtime_mlx_STT project.

This module provides the core infrastructure components for the vertical slice architecture,
including events, commands, and interfaces for the system's components.
"""

# Import progress bar control early to ensure tqdm is disabled project-wide
from src.Infrastructure.ProgressBar import ProgressBarManager

# First import interfaces to avoid circular imports
from .Common.Interfaces.event_bus import IEventBus
from .Common.Interfaces.command_handler import ICommandHandler
from .Common.Interfaces.audio_provider import IAudioProvider
from .Common.Interfaces.voice_activity_detector import IVoiceActivityDetector
from .Common.Interfaces.wake_word_detector import IWakeWordDetector
from .Common.Interfaces.transcription_engine import ITranscriptionEngine

# Then import implementations
from .Events.event import Event
from .Events.event_bus import EventBus
from .Commands.command import Command
from .Commands.command_dispatcher import CommandDispatcher

__all__ = [
    # Events
    'Event',
    'EventBus',
    'IEventBus',
    
    # Commands
    'Command',
    'CommandDispatcher',
    'ICommandHandler',
    
    # Feature interfaces
    'IAudioProvider',
    'IVoiceActivityDetector',
    'IWakeWordDetector',
    'ITranscriptionEngine'
]