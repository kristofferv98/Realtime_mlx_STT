"""
WakeWordDetectionStoppedEvent for wake word detection.

This event is published when wake word detection stops.
"""

from datetime import datetime
from typing import Optional

from src.Core.Events.event import Event


class WakeWordDetectionStoppedEvent(Event):
    """
    Event published when wake word detection stops.
    
    This event is published when the system stops listening for wake words.
    
    Attributes:
        reason: Reason for stopping detection (e.g., "user_requested", "timeout", "error")
        detector_type: The type of wake word detector that was being used
    """
    
    def __init__(self, 
                reason: str,
                detector_type: str,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """Initialize the event with the required parameters."""
        super().__init__(id=id, timestamp=timestamp, name=name)
        self.reason = reason
        self.detector_type = detector_type