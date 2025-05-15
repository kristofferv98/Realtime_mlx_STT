from dataclasses import dataclass
from typing import Optional
from src.Core.Commands.command import Command


@dataclass
class StartRecordingCommand(Command):
    """Command to start audio recording."""
    
    # Device ID to use (None for default)
    device_id: Optional[int] = None
    
    # Desired sample rate in Hz
    sample_rate: int = 16000
    
    # Desired chunk size in samples
    chunk_size: int = 512
    
    # Number of channels (1 for mono, 2 for stereo)
    channels: int = 1
    
    # Format of the audio data (e.g., 'int16', 'float32')
    format: str = 'int16'
    
    # Whether to resample to the target_samplerate if device doesn't match
    resample_if_needed: bool = True
    
    # Enable debug logging
    debug_mode: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = "StartRecordingCommand"