"""
Transcription Engine interface.

This module defines the ITranscriptionEngine interface that abstracts the speech-to-text
functionality in the system.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class ITranscriptionEngine(ABC):
    """
    Interface for transcription engines that convert audio to text.
    
    Implementations might include MLX-optimized Whisper, other local models,
    or remote API-based transcription services.
    """
    
    @abstractmethod
    def start(self) -> bool:
        """
        Initialize and start the transcription engine.
        
        Returns:
            bool: True if the engine started successfully
        """
        pass
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> None:
        """
        Request transcription of complete audio segment.
        
        Args:
            audio: Audio data as numpy array (float32, -1.0 to 1.0 range)
        """
        pass
    
    @abstractmethod
    def add_audio_chunk(self, audio_chunk: np.ndarray, is_last: bool = False) -> None:
        """
        Add an audio chunk for streaming transcription.
        
        Args:
            audio_chunk: Audio data chunk as numpy array
            is_last: Whether this is the last chunk in the stream
        """
        pass
    
    @abstractmethod
    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get the transcription result (blocking with timeout).
        
        Args:
            timeout: Maximum time to wait for a result in seconds
            
        Returns:
            Optional[Dict[str, Any]]: Transcription result or None if not available
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources used by the transcription engine.
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the transcription engine is currently running.
        
        Returns:
            bool: True if the engine is running
        """
        pass