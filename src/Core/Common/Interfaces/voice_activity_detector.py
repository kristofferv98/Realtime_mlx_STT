"""
Voice Activity Detector interface.

This module defines the IVoiceActivityDetector interface that abstracts the speech
detection functionality in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class IVoiceActivityDetector(ABC):
    """
    Interface for voice activity detectors that identify speech in audio data.
    
    Voice activity detectors analyze audio data to determine whether it contains
    speech. Different implementations may use different algorithms and have
    different performance characteristics in terms of accuracy, latency, and
    resource usage.
    """
    
    @abstractmethod
    def setup(self) -> bool:
        """
        Initialize the voice activity detector and prepare it for use.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def detect(self, audio_data: bytes, sample_rate: Optional[int] = None) -> bool:
        """
        Detect if the provided audio data contains speech.
        
        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Sample rate of the audio data (optional, can use default)
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        pass
    
    @abstractmethod
    def detect_with_confidence(self, audio_data: bytes, 
                              sample_rate: Optional[int] = None) -> Tuple[bool, float]:
        """
        Detect if the provided audio data contains speech and return confidence level.
        
        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Sample rate of the audio data (optional, can use default)
            
        Returns:
            Tuple[bool, float]: (speech_detected, confidence_score)
        """
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the voice activity detector with the provided parameters.
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            bool: True if configuration was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state of the voice activity detector.
        
        This is useful when starting a new detection session to clear any
        accumulated state.
        """
        pass
    
    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration of the voice activity detector.
        
        Returns:
            Dict[str, Any]: Dictionary of current configuration parameters
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the voice activity detector implementation.
        
        Returns:
            str: Name of the detector
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources used by the voice activity detector.
        """
        pass