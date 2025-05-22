"""
Voice Activity Detection feature for Realtime_mlx_STT.

This package provides voice activity detection capabilities using multiple
detection algorithms including WebRTC VAD, Silero VAD, and a combined approach.

Main components:
- VadModule: Main facade for VAD functionality
- Detectors: WebRTC, Silero, and Combined VAD implementations
- Commands: Configuration and control commands
- Events: Speech and silence detection events
"""

from .VadModule import VadModule
from .Events.SpeechDetectedEvent import SpeechDetectedEvent
from .Events.SilenceDetectedEvent import SilenceDetectedEvent

__all__ = [
    'VadModule',
    'SpeechDetectedEvent',
    'SilenceDetectedEvent'
]