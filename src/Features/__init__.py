"""
Features module for Realtime_mlx_STT.

This package contains all feature implementations following the vertical slice architecture.
Each feature is self-contained with its own Commands, Events, Handlers, and Models.

Available features:
- AudioCapture: Audio input management from microphones and files
- Transcription: Speech-to-text conversion using MLX Whisper or OpenAI
- VoiceActivityDetection: Speech detection and segmentation
- WakeWordDetection: Keyword spotting for activation
"""

# Import module facades for easy access
from .AudioCapture.AudioCaptureModule import AudioCaptureModule
from .Transcription.TranscriptionModule import TranscriptionModule
from .VoiceActivityDetection.VadModule import VadModule
from .WakeWordDetection.WakeWordModule import WakeWordModule

__all__ = [
    'AudioCaptureModule',
    'TranscriptionModule',
    'VadModule',
    'WakeWordModule'
]