"""
Transcription Feature package.

This package contains components for speech-to-text transcription using
MLX-optimized Whisper large-v3-turbo models on Apple Silicon.
"""

from src.Features.Transcription.TranscriptionModule import TranscriptionModule

# Import subpackages for easier access
from src.Features.Transcription.Models import TranscriptionConfig, TranscriptionResult, TranscriptionSession
from src.Features.Transcription.Commands import (
    TranscribeAudioCommand, ConfigureTranscriptionCommand,
    StartTranscriptionSessionCommand, StopTranscriptionSessionCommand
)
from src.Features.Transcription.Events import (
    TranscriptionStartedEvent, TranscriptionUpdatedEvent, TranscriptionErrorEvent
)

__all__ = [
    'TranscriptionModule',
    
    # Models
    'TranscriptionConfig',
    'TranscriptionResult',
    'TranscriptionSession',
    
    # Commands
    'TranscribeAudioCommand',
    'ConfigureTranscriptionCommand',
    'StartTranscriptionSessionCommand',
    'StopTranscriptionSessionCommand',
    
    # Events
    'TranscriptionStartedEvent',
    'TranscriptionUpdatedEvent',
    'TranscriptionErrorEvent'
]