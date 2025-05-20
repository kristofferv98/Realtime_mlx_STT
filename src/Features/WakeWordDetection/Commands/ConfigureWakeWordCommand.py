"""
ConfigureWakeWordCommand for wake word detection.

This command configures the behavior of the wake word detection system.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.Core.Commands.command import Command
from src.Features.WakeWordDetection.Models.WakeWordConfig import WakeWordConfig


@dataclass
class ConfigureWakeWordCommand(Command):
    """
    Command to configure the wake word detection system.
    
    This command sets parameters for the wake word detection system,
    including detector type, wake words, and sensitivities.
    
    Attributes:
        config: Configuration model for wake word detection
        detector_type: Optional override for detector type
        wake_words: Optional list of wake words to detect
        sensitivities: Optional sensitivities for each wake word
        access_key: Optional Porcupine access key
    """
    
    # Configuration parameters
    config: WakeWordConfig = field(default_factory=WakeWordConfig)
    
    # Optional direct parameters (for convenience)
    detector_type: Optional[str] = None
    wake_words: Optional[List[str]] = None
    sensitivities: Optional[List[float]] = None
    access_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate and merge parameters into config."""
        super().__post_init__()
        
        # Override config with direct parameters if provided
        if self.detector_type:
            self.config.detector_type = self.detector_type
        if self.wake_words:
            self.config.wake_words = self.wake_words
        if self.sensitivities:
            self.config.sensitivities = self.sensitivities
        if self.access_key:
            self.config.access_key = self.access_key