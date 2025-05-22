"""
Common interfaces and abstractions for the Core module.

This package contains the interface definitions that enable
loose coupling between different components of the system.
"""

from .Interfaces import (
    IEventBus,
    ICommandHandler,
    IAudioProvider,
    IVoiceActivityDetector,
    IWakeWordDetector,
    ITranscriptionEngine
)

__all__ = [
    'IEventBus',
    'ICommandHandler',
    'IAudioProvider',
    'IVoiceActivityDetector',
    'IWakeWordDetector',
    'ITranscriptionEngine'
]