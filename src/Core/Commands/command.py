"""
Base Command class for the command-mediator pattern.

This module provides the core Command class that all specific commands should inherit from.
Commands represent actions or requests in the system and are handled by command handlers.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Command:
    """
    Base class for all commands in the system.
    
    All commands have a unique ID, timestamp, and optional name.
    Specific command types should inherit from this class and add their own attributes.
    """
    
    # Auto-generated fields with default values
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    name: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Set the name to the class name if not provided."""
        if self.name is None:
            self.name = self.__class__.__name__