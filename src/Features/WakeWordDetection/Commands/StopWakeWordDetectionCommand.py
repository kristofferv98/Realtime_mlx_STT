"""
StopWakeWordDetectionCommand for wake word detection.

This command terminates the wake word detection process.
"""

from dataclasses import dataclass

from src.Core.Commands.command import Command


@dataclass
class StopWakeWordDetectionCommand(Command):
    """
    Command to stop wake word detection.
    
    This command stops listening for wake words and cleans up
    associated resources.
    """
    pass