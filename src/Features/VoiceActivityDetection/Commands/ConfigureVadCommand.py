"""
ConfigureVadCommand for voice activity detection.

This command configures the behavior of the voice activity detection system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from src.Core.Commands.command import Command


@dataclass
class ConfigureVadCommand(Command):
    """
    Command to configure the voice activity detection system.
    
    This command allows setting parameters for the VAD system's operation,
    including detector type, sensitivity, and advanced configuration options.
    
    Attributes:
        detector_type: Type of VAD detector to use ('webrtc', 'silero', 'combined')
        sensitivity: General sensitivity setting (0.0-1.0, higher = more sensitive)
        window_size: Number of frames to consider for detection decisions
        min_speech_duration: Minimum speech segment duration in seconds
        speech_pad_ms: Additional padding in ms to add before/after speech
        pre_speech_buffer_size: Number of audio chunks to keep in pre-speech buffer
                              Larger values capture more audio before speech detection
                              Recommended: 64 (equivalent to ~2 seconds at 32ms/chunk)
        parameters: Additional detector-specific parameters
    """
    
    # Since Command already has default fields, we need to make all fields here have defaults too
    detector_type: str = field(default=None)
    sensitivity: float = 0.5
    window_size: int = 5
    min_speech_duration: float = 0.25
    speech_pad_ms: int = 100
    pre_speech_buffer_size: int = 64  # ~2 seconds at 32ms/chunk
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate fields and call parent init."""
        super().__post_init__()
        
        # Enforce required fields
        if self.detector_type is None:
            raise ValueError("detector_type is required for ConfigureVadCommand")
        
        # Validation checks
        if self.detector_type not in ['webrtc', 'silero', 'combined']:
            raise ValueError(f"Invalid detector_type: {self.detector_type}. "
                            "Must be one of: 'webrtc', 'silero', 'combined'")
        
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ValueError(f"Sensitivity must be between 0.0 and 1.0, got {self.sensitivity}")
        
        if self.window_size < 1:
            raise ValueError(f"Window size must be at least 1, got {self.window_size}")
        
        if self.min_speech_duration < 0:
            raise ValueError(f"Minimum speech duration cannot be negative, got {self.min_speech_duration}")
            
        if self.pre_speech_buffer_size < 1:
            raise ValueError(f"Pre-speech buffer size must be at least 1, got {self.pre_speech_buffer_size}")
        # Remove direct logging from model class - will be handled in the command handler
        
    def map_to_detector_config(self) -> Dict[str, Any]:
        """
        Map this command's parameters to detector-specific configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary for the specified detector
        """
        config = {}
        
        # Common parameters
        config['sample_rate'] = self.parameters.get('sample_rate', 16000)
        
        # Always include pre-speech buffer size in configuration
        config['pre_speech_buffer_size'] = self.pre_speech_buffer_size
        
        # Parameters based on detector type
        if self.detector_type == 'webrtc':
            config['aggressiveness'] = int(self.sensitivity * 3)  # 0-3 scale
            config['speech_threshold'] = 0.5 + (self.sensitivity * 0.3)  # 0.5-0.8 scale
            config['history_size'] = self.window_size
            
        elif self.detector_type == 'silero':
            config['threshold'] = 0.3 + (self.sensitivity * 0.5)  # 0.3-0.8 scale
            config['min_speech_duration_ms'] = int(self.min_speech_duration * 1000)
            config['min_silence_duration_ms'] = self.speech_pad_ms
            
        elif self.detector_type == 'combined':
            # Check if individual thresholds are provided in parameters
            if 'webrtc_aggressiveness' in self.parameters:
                config['webrtc_aggressiveness'] = self.parameters['webrtc_aggressiveness']
            else:
                config['webrtc_aggressiveness'] = int(self.sensitivity * 3)  # 0-3 scale
            
            if 'silero_threshold' in self.parameters:
                config['silero_threshold'] = self.parameters['silero_threshold']
            else:
                config['silero_threshold'] = 0.3 + (self.sensitivity * 0.5)  # 0.3-0.8 scale
            
            if 'webrtc_threshold' in self.parameters:
                config['webrtc_threshold'] = self.parameters['webrtc_threshold']
            else:
                config['webrtc_threshold'] = 0.6  # Fixed threshold for more sensitive detection
            
            config['speech_confirmation_frames'] = max(1, int(self.window_size / 2))
            # Reduce from 3 to 2 frames for faster detection
            if self.parameters.get('faster_detection', True) and config['speech_confirmation_frames'] > 2:
                config['speech_confirmation_frames'] = 2
            config['silence_confirmation_frames'] = self.window_size
            
        # Handle common frame-related parameters from the parameters dict
        if 'frame_duration_ms' in self.parameters:
            config['frame_duration_ms'] = self.parameters['frame_duration_ms']
        
        if self.detector_type == 'combined':
            if 'speech_confirmation_frames' in self.parameters:
                config['speech_confirmation_frames'] = self.parameters['speech_confirmation_frames']
                
            if 'silence_confirmation_frames' in self.parameters:
                config['silence_confirmation_frames'] = self.parameters['silence_confirmation_frames']
                
            if 'speech_buffer_size' in self.parameters:
                config['speech_buffer_size'] = self.parameters['speech_buffer_size']
        
        # Include any additional parameters from the parameters dict
        config.update(self.parameters)
        
        return config