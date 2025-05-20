from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Dict, Any


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
    
    # Timestamp when this chunk was captured
    timestamp: float = 0.0
    
    # Sequence number for ordering chunks
    sequence_number: int = 0
    
    # Optional numpy array representation, lazily created when needed
    _numpy_data: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    
    # Normalized float32 data representation, cached to avoid redundant conversions
    _normalized_data: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    
    # Cache for derived data to avoid redundant calculations
    _cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        """
        Initialize computed fields after object creation.
        This ensures we only calculate things once.
        """
        # Pre-calculate duration for faster access
        self._cache['duration'] = None
    
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
        This is cached to avoid redundant conversions.
        
        Returns:
            np.ndarray: Normalized float32 audio data
        """
        # Return cached normalized data if available
        if self._normalized_data is not None:
            return self._normalized_data
            
        # Create and cache normalized data
        if self.format == 'int16':
            self._normalized_data = self.numpy_data.astype(np.float32) / 32768.0
        elif self.format == 'float32':
            self._normalized_data = self.numpy_data
        else:
            # Default conversion
            self._normalized_data = self.numpy_data.astype(np.float32)
            
            # Normalize if not in [-1.0, 1.0] range
            max_val = np.max(np.abs(self._normalized_data))
            if max_val > 0 and max_val > 1.0:
                self._normalized_data = self._normalized_data / max_val
        
        return self._normalized_data
        
    def get_duration(self) -> float:
        """
        Calculate the duration of this audio chunk in seconds.
        Value is cached for fast repeated access.
        
        Returns:
            float: Duration in seconds
        """
        # Return cached duration if available
        if self._cache['duration'] is not None:
            return self._cache['duration']
            
        # Calculate based on the number of samples and sample rate
        num_samples = len(self.numpy_data)
        if self.channels > 1:
            # If stereo or multi-channel, adjust the sample count
            num_samples = num_samples // self.channels
            
        # Duration = samples / sample_rate
        self._cache['duration'] = num_samples / self.sample_rate
        return self._cache['duration']