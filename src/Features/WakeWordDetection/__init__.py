"""
WakeWordDetection feature for the Realtime_mlx_STT project.

This feature provides wake word detection capabilities, allowing the system
to remain in a low-resource "listening" mode until a specific keyword or phrase
is detected, at which point it will activate full speech transcription.
"""

from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule
from src.Features.WakeWordDetection.Models.WakeWordConfig import WakeWordConfig
from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent

__version__ = "0.1.0"