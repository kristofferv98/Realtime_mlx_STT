"""
Wake Word Detector interface.

This module defines the IWakeWordDetector interface that abstracts the wake word
detection functionality in the system.
"""

from abc import ABC, abstractmethod
from typing import List


class IWakeWordDetector(ABC):
    """
    Interface for wake word detection components.
    
    Implementations might include Porcupine, OpenWakeWord, or custom detectors.
    """
    
    @abstractmethod
    def setup(self, wake_words: List[str], sensitivities: List[float]) -> bool:
        """
        Initialize the wake word detector with specified wake words.
        
        Args:
            wake_words: List of wake word names or paths to models
            sensitivities: List of sensitivity values for each wake word
            
        Returns:
            bool: True if setup was successful
        """
        pass
    
    @abstractmethod
    def process(self, audio_chunk: bytes) -> int:
        """
        Process an audio chunk to detect wake words.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            int: Index of detected wake word or -1 if none detected
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Get the expected sample rate for the detector.
        
        Returns:
            int: Expected sample rate in Hz
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources used by the wake word detector.
        """
        pass