"""
RecordingStateChangedEvent for notifying about recording state changes.

This event is published when the recording state changes, such as 
when recording starts, stops, or encounters an error.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
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


@dataclass
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
    
    # The previous recording state
    previous_state: RecordingState
    
    # The current recording state
    current_state: RecordingState
    
    # ID of the device being used for recording
    device_id: int
    
    # Optional error message if state is ERROR
    error_message: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize empty metadata dict if None."""
        super().__post_init__()
        if self.metadata is None:
            self.metadata = {}