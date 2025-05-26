"""
Realtime MLX STT - High-performance speech-to-text for Apple Silicon.

A powerful speech-to-text library optimized for Apple Silicon (M1/M2/M3) Macs,
providing real-time transcription with multiple API levels.

## Quick Start

Client API (Recommended):
    ```python
    from realtime_mlx_stt import STTClient
    
    # Create client
    client = STTClient()
    
    # Transcribe for 10 seconds
    for result in client.transcribe(duration=10):
        print(result.text)
    
    # Or use streaming
    with client.stream() as stream:
        for result in stream:
            print(result.text)
    ```

With API Keys:
    ```python
    from realtime_mlx_stt import create_client
    
    client = create_client(
        openai_api_key="sk-...",
        porcupine_api_key="..."
    )
    
    # Use OpenAI
    for result in client.transcribe(engine="openai"):
        print(result.text)
    
    # Wake word mode
    client.start_wake_word("jarvis")
    ```

Legacy APIs (still supported):
    ```python
    # Simple API
    from realtime_mlx_stt import Transcriber
    transcriber = Transcriber()
    
    # Session API
    from realtime_mlx_stt import TranscriptionSession
    session = TranscriptionSession()
    ```

## Features
- Multiple transcription engines (MLX Whisper, OpenAI)
- Voice Activity Detection (VAD) with WebRTC and Silero
- Wake word detection (Porcupine)
- Real-time streaming transcription
- Type-safe configuration
- Multiple API levels for different use cases

## Requirements
- Apple Silicon Mac (M1/M2/M3)
- Python 3.9+
- macOS 11.0+
"""

__version__ = "0.1.0"
__author__ = "Kristoffer Vatnehol"
__email__ = "kristoffer.vatnehol@gmail.com"

# Main API
from .transcriber import Transcriber
from .session import TranscriptionSession, SessionState
from .client import STTClient, create_client

# Configuration classes
from .config import (
    ModelConfig,
    VADConfig,
    WakeWordConfig,
    TranscriberConfig,
    TranscriptionEngine as ConfigEngine,
    VADDetectorType,
    WakeWordDetector
)

# Types and enums
from .types import (
    AudioSource,
    TranscriptionEngine,
    VADMode,
    TranscriptionResult,
    AudioDevice,
    TranscriptionCallback,
    AudioCallback,
    ErrorCallback
)

# Utility functions
from .utils import (
    list_audio_devices,
    setup_minimal_logging,
    format_duration,
    validate_language_code
)

# Wake word support (optional)
try:
    from .wake_word import WakeWordTranscriber
    __all__ = [
        # Main classes
        'STTClient',
        'create_client',
        'Transcriber',
        'TranscriptionSession',
        'SessionState',
        'WakeWordTranscriber',
        
        # Configuration classes
        'ModelConfig',
        'VADConfig',
        'WakeWordConfig',
        'TranscriberConfig',
        'VADDetectorType',
        'WakeWordDetector',
        
        # Types
        'AudioSource',
        'TranscriptionEngine', 
        'VADMode',
        'TranscriptionResult',
        'AudioDevice',
        
        # Callbacks
        'TranscriptionCallback',
        'AudioCallback',
        'ErrorCallback',
        
        # Utils
        'list_audio_devices',
        'setup_minimal_logging',
        'format_duration',
        'validate_language_code'
    ]
except ImportError:
    # Wake word not available
    __all__ = [
        # Main classes
        'STTClient',
        'create_client',
        'Transcriber',
        'TranscriptionSession',
        'SessionState',
        
        # Configuration classes
        'ModelConfig',
        'VADConfig',
        'WakeWordConfig',
        'TranscriberConfig',
        'VADDetectorType',
        'WakeWordDetector',
        
        # Types
        'AudioSource',
        'TranscriptionEngine', 
        'VADMode',
        'TranscriptionResult',
        'AudioDevice',
        
        # Callbacks
        'TranscriptionCallback',
        'AudioCallback',
        'ErrorCallback',
        
        # Utils
        'list_audio_devices',
        'setup_minimal_logging',
        'format_duration',
        'validate_language_code'
    ]