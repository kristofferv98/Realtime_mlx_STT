"""
ListDevicesCommand for retrieving audio device information.

This command requests a list of available audio input devices.
"""

from dataclasses import dataclass
from src.Core.Commands.command import Command


@dataclass
class ListDevicesCommand(Command):
    """
    Command to request a list of available audio input devices.
    
    When handled, this command will return a list of DeviceInfo objects
    representing all available audio input devices.
    """
    
    # No additional parameters needed for this command
    pass