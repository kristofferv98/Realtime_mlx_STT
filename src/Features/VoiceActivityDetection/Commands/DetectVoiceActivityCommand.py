"""
DetectVoiceActivityCommand for voice activity detection.

This command triggers the processing of an audio chunk to detect speech.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from src.Core.Commands.command import Command
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk


@dataclass
class DetectVoiceActivityCommand(Command):
    """
    Command to process an audio chunk and detect voice activity.
    
    This command is used to analyze audio data and determine if it contains speech.
    It includes the audio chunk to analyze and optional parameters for the detection.
    
    Attributes:
        audio_chunk: AudioChunk object containing the audio data to analyze
        sensitivity: Optional adjustment to detection sensitivity (0.0-1.0)
        detector_type: Optional specifier for which VAD detector to use
        return_confidence: Whether to return confidence score along with detection result
        options: Additional detector-specific options
    """
    
    # Since Command already has default fields, we need to make all fields here have defaults too,
    # even the "required" ones. We'll enforce requirements in __post_init__
    audio_chunk: AudioChunk = field(default=None)
    
    # Optional arguments with defaults
    sensitivity: Optional[float] = None
    detector_type: Optional[str] = None
    return_confidence: bool = False
    options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required fields and call parent init."""
        super().__post_init__()
        
        # Enforce required fields
        if self.audio_chunk is None:
            raise ValueError("audio_chunk is required for DetectVoiceActivityCommand")