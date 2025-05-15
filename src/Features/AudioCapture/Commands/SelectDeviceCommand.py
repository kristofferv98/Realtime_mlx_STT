"""
SelectDeviceCommand for choosing an audio input device.

This command requests that a specific audio device be selected for recording.
"""

from dataclasses import dataclass
from typing import Optional
from src.Core.Commands.command import Command


@dataclass
class SelectDeviceCommand(Command):
    """
    Command to select a specific audio device for recording.
    
    When handled, this command will configure the audio provider to use
    the specified device for future recordings.
    
    Args:
        device_id: The ID of the device to select, or None to use the default device
    """
    
    # The ID of the device to select (None means use default device)
    device_id: Optional[int] = None