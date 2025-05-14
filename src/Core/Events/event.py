"""
Base Event class for the event-driven architecture.

This module provides the core Event class that all specific events should inherit from.
Events are used for communication between components in a loosely coupled manner.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Event:
    """
    Base class for all events in the system.
    
    All events have a unique ID, timestamp, and optional name.
    Specific event types should inherit from this class and add their own attributes.
    """
    
    # Auto-generated fields with default values
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    name: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Set the name to the class name if not provided."""
        if self.name is None:
            self.name = self.__class__.__name__