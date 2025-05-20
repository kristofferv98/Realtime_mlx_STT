"""
WakeWordDetectionStartedEvent for wake word detection.

This event is published when wake word detection starts.
"""

from datetime import datetime
from typing import Optional, List

from src.Core.Events.event import Event


class WakeWordDetectionStartedEvent(Event):
    """
    Event published when wake word detection starts.
    
    This event is published when the system begins listening for wake words.
    
    Attributes:
        detector_type: The type of wake word detector being used
        wake_words: List of wake words being detected
    """
    
    def __init__(self, 
                detector_type: str,
                wake_words: List[str],
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """Initialize the event with the required parameters."""
        super().__init__(id=id, timestamp=timestamp, name=name)
        self.detector_type = detector_type
        self.wake_words = wake_words