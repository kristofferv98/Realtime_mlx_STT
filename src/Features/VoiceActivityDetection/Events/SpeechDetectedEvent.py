"""
SpeechDetectedEvent for voice activity detection.

This event is published when speech is detected in the audio stream.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Union
import time

from src.Core.Events.event import Event


@dataclass
class SpeechDetectedEvent(Event):
    """
    Event published when speech is detected in the audio stream.
    
    This event signals the beginning of a speech segment and includes information about
    the detection, such as timestamp, confidence level, and a reference to the
    audio data where speech was first detected.
    
    Attributes:
        confidence: Confidence level of the speech detection (0.0-1.0)
        audio_timestamp: Timestamp when the audio was captured
        detector_type: Type of VAD detector that detected the speech
        audio_reference: Reference to the audio chunk containing the speech
        speech_id: Unique identifier for this speech segment
    """
    
    confidence: float
    audio_timestamp: float = field(default_factory=time.time)
    detector_type: str = "unknown"
    audio_reference: Optional[Any] = None
    speech_id: str = field(default_factory=lambda: str(time.time_ns()))
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate confidence level
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between a value of 0.0 and 1.0, got {self.confidence}")