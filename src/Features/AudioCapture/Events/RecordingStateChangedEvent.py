from dataclasses import dataclass
from typing import Optional, Literal
from src.Core.Events.event import Event


@dataclass
class RecordingStateChangedEvent(Event):
    """Event emitted when the recording state changes."""
    
    # The new recording state
    state: Literal["started", "stopped", "paused", "error", "device_changed"]
    
    # Optional message with additional information
    message: Optional[str] = None
    
    # Device ID if relevant
    device_id: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = "RecordingStateChangedEvent"