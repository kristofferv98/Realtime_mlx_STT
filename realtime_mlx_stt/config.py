"""
Configuration classes for the high-level API.

These classes provide type-safe configuration options that mirror
the server's configuration structure.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class TranscriptionEngine(Enum):
    """Available transcription engines."""
    MLX_WHISPER = "mlx_whisper"
    OPENAI = "openai"


class VADDetectorType(Enum):
    """Voice Activity Detection types."""
    WEBRTC = "webrtc"
    SILERO = "silero"
    COMBINED = "combined"


class WakeWordDetector(Enum):
    """Wake word detection engines."""
    PORCUPINE = "porcupine"


@dataclass
class ModelConfig:
    """Configuration for transcription model."""
    engine: TranscriptionEngine = TranscriptionEngine.MLX_WHISPER
    model: str = "whisper-large-v3-turbo"
    language: Optional[str] = None  # None for auto-detect
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "engine": self.engine.value if isinstance(self.engine, Enum) else self.engine,
            "model": self.model,
            "language": self.language
        }


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    enabled: bool = True
    detector_type: VADDetectorType = VADDetectorType.COMBINED
    sensitivity: float = 0.6
    min_speech_duration: float = 0.25
    min_silence_duration: float = 0.1
    window_size: int = 5
    
    # Advanced parameters
    webrtc_aggressiveness: Optional[int] = None  # 0-3
    silero_threshold: Optional[float] = None  # 0.1-0.9
    frame_duration_ms: Optional[int] = None  # 10, 20, or 30
    
    def __post_init__(self):
        """Validate parameters."""
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ValueError(f"Sensitivity must be between 0.0 and 1.0, got {self.sensitivity}")
        
        if self.webrtc_aggressiveness is not None:
            if not 0 <= self.webrtc_aggressiveness <= 3:
                raise ValueError(f"WebRTC aggressiveness must be 0-3, got {self.webrtc_aggressiveness}")
        
        if self.silero_threshold is not None:
            if not 0.1 <= self.silero_threshold <= 0.9:
                raise ValueError(f"Silero threshold must be 0.1-0.9, got {self.silero_threshold}")
        
        if self.frame_duration_ms is not None:
            if self.frame_duration_ms not in [10, 20, 30]:
                raise ValueError(f"Frame duration must be 10, 20, or 30ms, got {self.frame_duration_ms}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        config = {
            "enabled": self.enabled,
            "detector_type": self.detector_type.value if isinstance(self.detector_type, Enum) else self.detector_type,
            "sensitivity": self.sensitivity,
            "min_speech_duration": self.min_speech_duration,
            "min_silence_duration": self.min_silence_duration,
            "window_size": self.window_size
        }
        
        # Add advanced parameters if set
        parameters = {}
        if self.webrtc_aggressiveness is not None:
            parameters["webrtc_aggressiveness"] = self.webrtc_aggressiveness
        if self.silero_threshold is not None:
            parameters["silero_threshold"] = self.silero_threshold
        if self.frame_duration_ms is not None:
            parameters["frame_duration_ms"] = self.frame_duration_ms
        
        if parameters:
            config["parameters"] = parameters
        
        return config


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    enabled: bool = True
    words: List[str] = field(default_factory=lambda: ["jarvis"])
    sensitivity: float = 0.7
    timeout: int = 30  # seconds
    detector: WakeWordDetector = WakeWordDetector.PORCUPINE
    access_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate parameters."""
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ValueError(f"Sensitivity must be between 0.0 and 1.0, got {self.sensitivity}")
        
        if self.timeout < 5:
            raise ValueError(f"Timeout must be at least 5 seconds, got {self.timeout}")
        
        # Validate wake words
        supported_words = {
            "jarvis", "alexa", "computer", "hey google", 
            "ok google", "hey siri", "porcupine"
        }
        for word in self.words:
            if word.lower() not in supported_words:
                raise ValueError(
                    f"Unsupported wake word '{word}'. "
                    f"Supported words: {', '.join(sorted(supported_words))}"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "enabled": self.enabled,
            "detector": self.detector.value if isinstance(self.detector, Enum) else self.detector,
            "words": [w.lower() for w in self.words],
            "sensitivity": self.sensitivity,
            "timeout": self.timeout
        }


@dataclass
class TranscriberConfig:
    """Complete configuration for Transcriber."""
    model: ModelConfig = field(default_factory=ModelConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    wake_word: Optional[WakeWordConfig] = None
    
    # Callbacks
    on_transcription: Optional[Any] = None  # Callable[[TranscriptionResult], None]
    on_error: Optional[Any] = None  # Callable[[Exception], None]
    on_wake_word: Optional[Any] = None  # Callable[[str, float], None]
    
    # Other settings
    device_index: Optional[int] = None
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        config = {
            "transcription": self.model.to_dict(),
            "vad": self.vad.to_dict()
        }
        
        if self.wake_word:
            config["wake_word"] = self.wake_word.to_dict()
        
        return config