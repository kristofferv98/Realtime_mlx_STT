"""Voice Activity Detection Event Classes.

This module exports event classes for the VoiceActivityDetection feature.
"""

from .SpeechDetectedEvent import SpeechDetectedEvent
from .SilenceDetectedEvent import SilenceDetectedEvent

__all__ = [
    'SpeechDetectedEvent',
    'SilenceDetectedEvent'
]