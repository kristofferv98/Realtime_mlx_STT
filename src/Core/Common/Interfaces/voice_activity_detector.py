"""
Voice Activity Detector interface.

This module defines the IVoiceActivityDetector interface that abstracts the voice activity
detection functionality in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class IVoiceActivityDetector(ABC):
    """
    Interface for voice activity detection components that identify speech in audio.
    
    Implementations might include WebRTC VAD, Silero VAD, or combined approaches.
    """
    
    @abstractmethod
    def setup(self) -> bool:
        """
        Initialize the VAD component.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Determine if an audio chunk contains speech.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state of the VAD component.
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the VAD component.
        
        Returns:
            Dict[str, Any]: Configuration parameters
        """
        pass
    
    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> bool:
        """
        Update the configuration of the VAD component.
        
        Args:
            config: Configuration parameters to update
            
        Returns:
            bool: True if configuration was successfully updated
        """
        pass