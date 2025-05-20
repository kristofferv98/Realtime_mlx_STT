"""
DisableVadProcessingCommand for voice activity detection.

This command disables active processing of audio chunks for voice activity detection.
"""

from dataclasses import dataclass

from src.Core.Commands.command import Command


@dataclass
class DisableVadProcessingCommand(Command):
    """
    Command to disable active processing of audio chunks for voice activity detection.
    
    This command instructs the voice activity detection system to stop processing
    audio chunks for speech detection. When disabled, the VAD system will still
    receive audio chunks but will skip applying detection algorithms, significantly
    reducing CPU usage.
    
    This is typically used when:
    1. The system is waiting for a wake word and doesn't need speech detection
    2. Speech processing has completed and the system returns to idle state
    3. The system needs to conserve resources during inactive periods
    
    The command takes no additional parameters beyond the base Command class.
    """
    
    def __post_init__(self):
        """Validate fields and call parent init."""
        super().__post_init__()