"""
DetectWakeWordCommand for wake word detection.

This command processes an audio chunk to detect wake words.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.Core.Commands.command import Command
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk


@dataclass
class DetectWakeWordCommand(Command):
    """
    Command to detect wake words in an audio chunk.
    
    This command processes a single audio chunk to detect wake words.
    
    Attributes:
        detector_type: Optional detector type to use (overrides configured value)
        return_confidence: Whether to return confidence score
        audio_chunk: The audio chunk to process
    """
    
    # Optional detector type override
    detector_type: Optional[str] = None
    
    # Whether to return confidence score
    return_confidence: bool = False
    
    # Audio chunk to process 
    # Using field(default=None) to make it a parameter with a default
    # while still making it required at runtime through post-init validation
    audio_chunk: AudioChunk = field(default=None)
    
    def __post_init__(self):
        """Validate required fields."""
        super().__post_init__()
        if self.audio_chunk is None:
            raise ValueError("audio_chunk is required and cannot be None")