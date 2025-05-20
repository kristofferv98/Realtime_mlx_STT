"""
WakeWordTimeoutEvent for wake word detection.

This event is published when no speech is detected after a wake word within the timeout period.
"""

from datetime import datetime
from typing import Optional

from src.Core.Events.event import Event


class WakeWordTimeoutEvent(Event):
    """
    Event published when no speech is detected after a wake word.
    
    This event is published when the system detects a wake word but
    no speech follows within the timeout period.
    
    Attributes:
        wake_word: The detected wake word that timed out
        timeout_duration: The duration of the timeout in seconds
    """
    
    def __init__(self, 
                wake_word: str,
                timeout_duration: float,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """Initialize the event with the required parameters."""
        super().__init__(id=id, timestamp=timestamp, name=name)
        self.wake_word = wake_word
        self.timeout_duration = timeout_duration