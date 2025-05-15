"""
StartRecordingCommand for initiating audio recording.

This command requests that audio recording be started with
specified parameters.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.Core.Commands.command import Command


@dataclass
class StartRecordingCommand(Command):
    """
    Command to start audio recording with the configured device.
    
    When handled, this command will initiate audio capture with the
    specified parameters.
    
    Args:
        sample_rate: The desired sample rate in Hz (default: 16000)
        chunk_size: The desired chunk size in samples (default: 512)
        channels: The number of audio channels (default: 1 for mono)
        audio_format: The audio format identifier (default: 'int16')
        device_id: Optional device ID to use (overrides previously selected device)
        options: Additional provider-specific recording options
    """
    
    # Recording parameters
    sample_rate: int = 16000
    chunk_size: int = 512
    channels: int = 1
    audio_format: str = 'int16'
    
    # Optional device override
    device_id: Optional[int] = None
    
    # Additional provider-specific options
    options: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize empty options dict if None."""
        super().__post_init__()
        if self.options is None:
            self.options = {}