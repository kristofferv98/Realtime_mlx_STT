"""
RecordingStateChangedEvent for notifying about recording state changes.

This event is published when the recording state changes, such as 
when recording starts, stops, or encounters an error.
"""

from typing import Optional, Dict, Any
import uuid
from datetime import datetime
from enum import Enum, auto

from src.Core.Events.event import Event


class RecordingState(Enum):
    """Enumeration of possible recording states."""
    INITIALIZED = auto()
    STARTING = auto()
    RECORDING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class RecordingStateChangedEvent(Event):
    """
    Event published when the recording state changes.
    
    This event contains information about the previous and current recording
    states, along with any relevant metadata.
    
    Args:
        previous_state: The previous recording state
        current_state: The current recording state
        device_id: The ID of the device being used for recording
        error_message: Optional error message if the state is ERROR
        metadata: Additional metadata about the recording
    """
    
    def __init__(self,
                previous_state: RecordingState,
                current_state: RecordingState,
                device_id: int,
                error_message: Optional[str] = None,
                metadata: Dict[str, Any] = None,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """Initialize the event with the required parameters."""
        super().__init__(id=id, timestamp=timestamp, name=name)
        
        # Required parameters
        self.previous_state = previous_state
        self.current_state = current_state
        self.device_id = device_id
        
        # Optional parameters
        self.error_message = error_message
        self.metadata = metadata if metadata is not None else {}