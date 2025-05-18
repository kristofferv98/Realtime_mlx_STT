"""
SpeechDetectedEvent for voice activity detection.

This event is published when speech is detected in the audio stream.
"""

from typing import Optional, Any, Union
import time
import uuid
from datetime import datetime

from src.Core.Events.event import Event


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
    
    def __init__(self, 
                confidence: float,
                audio_timestamp: float = None,
                detector_type: str = "unknown",
                audio_reference: Optional[Any] = None, 
                speech_id: str = None,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """Initialize the event with the required parameters."""
        super().__init__(id=id, timestamp=timestamp, name=name)
        
        # Required parameters
        self.confidence = confidence
        
        # Parameters with defaults
        self.audio_timestamp = audio_timestamp if audio_timestamp is not None else time.time()
        self.detector_type = detector_type
        self.audio_reference = audio_reference
        self.speech_id = speech_id if speech_id is not None else str(time.time_ns())
        
        # Validate confidence level
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between a value of 0.0 and 1.0, got {self.confidence}")