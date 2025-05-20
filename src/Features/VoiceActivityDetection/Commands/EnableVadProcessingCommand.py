"""
EnableVadProcessingCommand for voice activity detection.

This command enables active processing of audio chunks for voice activity detection.
"""

from dataclasses import dataclass

from src.Core.Commands.command import Command


@dataclass
class EnableVadProcessingCommand(Command):
    """
    Command to enable active processing of audio chunks for voice activity detection.
    
    This command instructs the voice activity detection system to start processing
    audio chunks for speech detection. When enabled, the VAD system will actively
    apply detection algorithms to each incoming audio chunk, which consumes CPU
    resources but allows for speech detection.
    
    This is typically used when:
    1. A wake word has been detected and the system should listen for speech
    2. Direct speech transcription is needed without wake word activation
    3. VAD processing was previously disabled and needs to be reactivated
    
    The command takes no additional parameters beyond the base Command class.
    """
    
    def __post_init__(self):
        """Validate fields and call parent init."""
        super().__post_init__()