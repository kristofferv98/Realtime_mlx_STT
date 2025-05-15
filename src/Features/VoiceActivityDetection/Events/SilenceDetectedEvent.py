"""
SilenceDetectedEvent for voice activity detection.

This event is published when silence is detected following a speech segment.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Union
import time

from src.Core.Events.event import Event


@dataclass
class SilenceDetectedEvent(Event):
    """
    Event published when silence is detected after a speech segment.
    
    This event signals the end of a speech segment and includes information about
    the speech that was detected, such as its duration and a reference to the
    complete audio segment.
    
    Attributes:
        speech_duration: Duration of the speech segment in seconds
        audio_timestamp: Timestamp of the last audio chunk in the speech segment
        speech_start_time: When the speech segment began
        speech_end_time: When the speech segment ended
        audio_reference: Reference to the complete audio segment
        speech_id: Unique identifier matching the corresponding SpeechDetectedEvent
    """
    
    speech_duration: float
    audio_timestamp: float = field(default_factory=time.time)
    speech_start_time: float = 0.0
    speech_end_time: float = field(default_factory=time.time)
    audio_reference: Optional[Any] = None
    speech_id: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Make sure the speech duration is positive
        if self.speech_duration < 0:
            raise ValueError(f"Speech duration cannot be negative, got {self.speech_duration}")
        
        # Set speech end time if not provided
        if self.speech_end_time == 0.0:
            self.speech_end_time = self.audio_timestamp
            
        # Calculate speech start time if not provided
        if self.speech_start_time == 0.0:
            self.speech_start_time = self.speech_end_time - self.speech_duration