"""
StartWakeWordDetectionCommand for wake word detection.

This command initiates the wake word detection process.
"""

from dataclasses import dataclass
from typing import Optional

from src.Core.Commands.command import Command


@dataclass
class StartWakeWordDetectionCommand(Command):
    """
    Command to start wake word detection.
    
    This command initiates listening for wake words using the
    configured wake word detector.
    
    Attributes:
        detector_type: Optional detector type to use (overrides configured value)
    """
    
    # Optional parameters to override config
    detector_type: Optional[str] = None