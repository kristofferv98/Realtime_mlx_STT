from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    
    # Raw audio bytes (typically PCM data)
    raw_data: bytes
    
    # Sample rate in Hz
    sample_rate: int
    
    # Number of audio channels (1 for mono, 2 for stereo)
    channels: int
    
    # Format of the audio data (e.g., int16, float32)
    format: str
    
    # Optional numpy array representation, lazily created when needed
    _numpy_data: Optional[np.ndarray] = None
    
    # Timestamp when this chunk was captured
    timestamp: float = 0.0
    
    # Sequence number for ordering chunks
    sequence_number: int = 0
    
    @property
    def numpy_data(self) -> np.ndarray:
        """
        Get the audio data as a numpy array.
        Lazily converts from bytes if needed.
        
        Returns:
            np.ndarray: Audio data as a numpy array
        """
        if self._numpy_data is None:
            if self.format == 'int16':
                self._numpy_data = np.frombuffer(self.raw_data, dtype=np.int16)
            elif self.format == 'float32':
                self._numpy_data = np.frombuffer(self.raw_data, dtype=np.float32)
            # Add more formats as needed
        
        return self._numpy_data
    
    def to_float32(self) -> np.ndarray:
        """
        Convert the audio data to float32 format normalized to [-1.0, 1.0].
        
        Returns:
            np.ndarray: Normalized float32 audio data
        """
        if self.format == 'int16':
            return self.numpy_data.astype(np.float32) / 32768.0
        elif self.format == 'float32':
            return self.numpy_data
        
        # Default conversion
        return self.numpy_data.astype(np.float32)