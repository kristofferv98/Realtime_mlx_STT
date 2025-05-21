"""
WakeWordConfig model for configuring wake word detection.

This module provides the configuration model for wake word detection systems.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    
    # Detector selection
    detector_type: str = "porcupine"  # Options: "porcupine", "custom"
    
    # Wake word settings
    wake_words: List[str] = field(default_factory=lambda: ["porcupine"])
    sensitivities: List[float] = None  # Defaults to [0.5] * len(wake_words)
    
    # Porcupine-specific settings
    access_key: Optional[str] = None  # Will check PORCUPINE_ACCESS_KEY env var if None
    keyword_paths: List[str] = field(default_factory=list)  # Custom keyword model files
    
    # Detection behavior
    activation_delay: float = 0.0  # Seconds to wait for VAD before switching to wake word mode
    speech_timeout: float = 5.0  # Seconds to wait for speech after wake word detection
    buffer_duration: float = 0.1  # Seconds of audio to buffer before/including wake word
    exclude_pre_wake_word_audio: bool = True  # If True, VAD pre-buffer is cleared after wake word
    
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with defaults and validate."""
        # Default for sensitivities
        if self.sensitivities is None:
            self.sensitivities = [0.5] * len(self.wake_words)
            
        # Validate detector type
        if self.detector_type not in ["porcupine", "custom"]:
            raise ValueError(f"Invalid detector_type: {self.detector_type}")
            
        # Validate sensitivities
        if len(self.sensitivities) != len(self.wake_words):
            self.sensitivities = [self.sensitivities[0] if self.sensitivities else 0.5] * len(self.wake_words)
            
        # Validate sensitivities range
        for i, s in enumerate(self.sensitivities):
            if not 0.0 <= s <= 1.0:
                raise ValueError(f"Sensitivity at index {i} must be between 0.0 and 1.0, got {s}")