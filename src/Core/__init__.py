"""
Core module for the Realtime_mlx_STT project.

This module provides the core infrastructure components for the vertical slice architecture,
including events, commands, and interfaces for the system's components.
"""

# Re-export core components
from .Events import Event, EventBus
from .Commands import Command, CommandDispatcher
from .Common.Interfaces import (
    IEventBus,
    ICommandHandler,
    IAudioProvider,
    IVoiceActivityDetector,
    IWakeWordDetector,
    ITranscriptionEngine
)

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