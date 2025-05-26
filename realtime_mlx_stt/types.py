"""
User-facing types and enums for the high-level API.
"""

from enum import Enum
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass


class AudioSource(Enum):
    """Audio input source options."""
    MICROPHONE = "microphone"
    FILE = "file"
    STREAM = "stream"


class TranscriptionEngine(Enum):
    """Available transcription engines."""
    MLX_WHISPER = "mlx_whisper"
    OPENAI = "openai"


class VADMode(Enum):
    """Voice Activity Detection modes."""
    SILERO = "silero"
    WEBRTC = "webrtc"
    COMBINED = "combined"
    DISABLED = "disabled"


@dataclass
class TranscriptionResult:
    """Result from a transcription operation."""
    text: str
    confidence: float
    timestamp: float
    duration: Optional[float] = None
    language: Optional[str] = None


@dataclass
class AudioDevice:
    """Audio device information."""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False


# Type aliases for callbacks
TranscriptionCallback = Callable[[TranscriptionResult], None]
AudioCallback = Callable[[bytes], None]
ErrorCallback = Callable[[Exception], None]