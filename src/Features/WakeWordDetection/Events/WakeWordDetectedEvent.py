"""
WakeWordDetectedEvent for wake word detection.

This event is published when a wake word is detected in the audio.
"""

from datetime import datetime
from typing import Optional, Any

from src.Core.Events.event import Event


class WakeWordDetectedEvent(Event):
    """
    Event published when a wake word is detected.
    
    This event is published when the system detects a wake word in the audio.
    
    Attributes:
        wake_word: The detected wake word
        confidence: Confidence score of the detection (0.0-1.0)
        audio_timestamp: Timestamp of the audio where detection occurred
        detector_type: The type of wake word detector that made the detection
        audio_reference: Reference to audio data (may be used for debugging)
    """
    
    def __init__(self, 
                wake_word: str,
                confidence: float,
                audio_timestamp: float,
                detector_type: str,
                audio_reference: Any = None,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """Initialize the event with the required parameters."""
        super().__init__(id=id, timestamp=timestamp, name=name)
        self.wake_word = wake_word
        self.confidence = confidence
        self.audio_timestamp = audio_timestamp
        self.detector_type = detector_type
        self.audio_reference = audio_reference