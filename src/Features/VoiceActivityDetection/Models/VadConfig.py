"""
VadConfig model for Voice Activity Detection configuration.

This module defines the configuration settings for VAD detectors.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class VadConfig:
    """
    Configuration for Voice Activity Detection.
    
    Attributes:
        detector_type: Type of VAD detector ('webrtc', 'silero', 'combined')
        aggressiveness: WebRTC VAD aggressiveness level (0-3)
        threshold: Detection threshold for ML-based VADs (0.0-1.0)
        sample_rate: Audio sample rate in Hz
        frame_duration_ms: Frame duration for processing
        pre_speech_buffer_duration: Duration of audio to keep before speech (seconds)
        min_speech_duration: Minimum duration to consider as speech (seconds)
        max_speech_duration: Maximum duration for a single speech segment (seconds)
        min_silence_duration: Minimum silence duration to end speech (seconds)
        options: Additional detector-specific options
    """
    detector_type: str = 'combined'
    aggressiveness: int = 2
    threshold: float = 0.5
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    pre_speech_buffer_duration: float = 0.5
    min_speech_duration: float = 0.2
    max_speech_duration: Optional[float] = None
    min_silence_duration: float = 0.5
    options: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate detector type
        valid_detectors = {'webrtc', 'silero', 'combined'}
        if self.detector_type not in valid_detectors:
            raise ValueError(f"Invalid detector_type: {self.detector_type}. Must be one of {valid_detectors}")
        
        # Validate aggressiveness
        if not 0 <= self.aggressiveness <= 3:
            raise ValueError(f"Aggressiveness must be between 0 and 3, got {self.aggressiveness}")
        
        # Validate threshold
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")
        
        # Validate sample rate
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")
        
        # Validate frame duration
        if self.frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms, got {self.frame_duration_ms}")
        
        # Initialize options if None
        if self.options is None:
            self.options = {}