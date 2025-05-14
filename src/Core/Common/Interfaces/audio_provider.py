"""
Audio Provider interface.

This module defines the IAudioProvider interface that abstracts the audio capture
functionality in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class IAudioProvider(ABC):
    """
    Interface for audio input providers that capture audio data from various sources.
    
    Implementations might include microphone input, file input, or network streams.
    """
    
    @abstractmethod
    def setup(self) -> bool:
        """
        Initialize the audio provider and prepare it for recording.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """
        Start the audio capture process.
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the audio capture process.
        
        Returns:
            bool: True if successfully stopped, False otherwise
        """
        pass
    
    @abstractmethod
    def read_chunk(self) -> bytes:
        """
        Read a single chunk of audio data.
        
        Returns:
            bytes: Raw audio data as bytes
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Get the sample rate of the audio data.
        
        Returns:
            int: Sample rate in Hz
        """
        pass
    
    @abstractmethod
    def get_chunk_size(self) -> int:
        """
        Get the size of each audio chunk in samples.
        
        Returns:
            int: Chunk size in samples
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources used by the audio provider.
        """
        pass
    
    @abstractmethod
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio input devices.
        
        Returns:
            List[Dict[str, Any]]: List of device information dictionaries
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the audio provider is currently running.
        
        Returns:
            bool: True if the provider is running
        """
        pass