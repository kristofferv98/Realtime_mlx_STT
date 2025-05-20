"""
Transcription Engines package.

This package contains implementations of various transcription engines
that convert audio to text.
"""

from src.Features.Transcription.Engines.DirectMlxWhisperEngine import DirectMlxWhisperEngine
from src.Features.Transcription.Engines.DirectTranscriptionManager import DirectTranscriptionManager
from src.Features.Transcription.Engines.OpenAITranscriptionEngine import OpenAITranscriptionEngine

# For backward compatibility
MlxWhisperEngine = DirectMlxWhisperEngine
TranscriptionProcessManager = DirectTranscriptionManager

__all__ = [
    'DirectMlxWhisperEngine',
    'DirectTranscriptionManager',
    'OpenAITranscriptionEngine',
    # Legacy names for backward compatibility
    'MlxWhisperEngine',
    'TranscriptionProcessManager'
]