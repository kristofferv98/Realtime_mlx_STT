"""
Realtime_mlx_STT Library

A real-time speech-to-text transcription library optimized for Apple Silicon,
combining the real-time audio processing capabilities of RealtimeSTT with
the high-performance Whisper large-v3-turbo model optimized through MLX.

This library provides low-latency, high-accuracy transcription for macOS applications.
"""

__version__ = "0.1.0"

# Core components
from .audio_input import AudioInput
from .audio_recorder import AudioToTextRecorder

# MLX optimized components
from .mlx_transcriber import MLXTranscriber

# Expose main classes at the package level
__all__ = [
    'AudioInput',
    'AudioToTextRecorder',
    'MLXTranscriber',
]