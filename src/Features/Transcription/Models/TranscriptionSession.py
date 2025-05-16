"""
TranscriptionSession model.

This module defines the TranscriptionSession class that tracks the state
of an ongoing transcription operation.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from src.Features.Transcription.Models.TranscriptionConfig import TranscriptionConfig
from src.Features.Transcription.Models.TranscriptionResult import TranscriptionResult


@dataclass
class TranscriptionSession:
    """
    Represents an ongoing transcription session.
    
    This class tracks the state of a transcription operation, including
    buffered audio, configuration, and results.
    """
    # Session identification
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Timing information
    start_time: float = field(default_factory=time.time)
    last_activity_time: float = field(default_factory=time.time)
    
    # Configuration
    config: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    
    # State tracking
    is_active: bool = True
    language: Optional[str] = None
    detected_language_confidence: float = 0.0
    
    # Results
    current_text: str = ""
    results: List[TranscriptionResult] = field(default_factory=list)
    
    # Audio storage
    _audio_chunks: List[np.ndarray] = field(default_factory=list)
    _total_audio_duration_ms: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Add an audio chunk to the session.
        
        Args:
            audio_chunk: Audio data as numpy array
        """
        self._audio_chunks.append(audio_chunk)
        
        # Update duration (assuming 16kHz sample rate)
        chunk_duration_ms = (len(audio_chunk) / 16000) * 1000
        self._total_audio_duration_ms += chunk_duration_ms
        
        # Update activity timestamp
        self.last_activity_time = time.time()
    
    def get_combined_audio(self) -> np.ndarray:
        """
        Get all audio chunks combined into a single array.
        
        Returns:
            np.ndarray: Combined audio data
        """
        if not self._audio_chunks:
            return np.array([], dtype=np.float32)
        
        return np.concatenate(self._audio_chunks)
    
    def get_latest_audio(self, duration_ms: Optional[float] = None) -> np.ndarray:
        """
        Get the most recent audio up to the specified duration.
        
        Args:
            duration_ms: Maximum duration in milliseconds (or None for all audio)
            
        Returns:
            np.ndarray: Latest audio data
        """
        if not self._audio_chunks:
            return np.array([], dtype=np.float32)
        
        if duration_ms is None or duration_ms >= self._total_audio_duration_ms:
            return self.get_combined_audio()
        
        # Calculate how many samples we need (at 16kHz)
        samples_needed = int((duration_ms / 1000) * 16000)
        
        # Work backwards from the most recent chunks
        result = []
        samples_collected = 0
        
        for chunk in reversed(self._audio_chunks):
            result.insert(0, chunk)
            samples_collected += len(chunk)
            
            if samples_collected >= samples_needed:
                break
        
        combined = np.concatenate(result)
        
        # Trim to exact sample count if we have extra
        if len(combined) > samples_needed:
            combined = combined[-samples_needed:]
            
        return combined
    
    def clear_audio_buffer(self) -> None:
        """Clear the stored audio buffer."""
        self._audio_chunks = []
        self._total_audio_duration_ms = 0.0
    
    def add_result(self, result: TranscriptionResult) -> None:
        """
        Add a transcription result to the session.
        
        Args:
            result: The transcription result to add
        """
        self.results.append(result)
        
        # Update current text if this is a final result
        if result.is_final:
            self.current_text = result.text
            
        # Update detected language if available
        if result.language and not self.language:
            self.language = result.language
            self.detected_language_confidence = result.confidence
        
        # Update activity timestamp
        self.last_activity_time = time.time()
    
    def close(self) -> None:
        """Mark the session as inactive."""
        self.is_active = False
    
    @property
    def duration_ms(self) -> float:
        """Get the total duration of audio processed in this session in milliseconds."""
        return self._total_audio_duration_ms
    
    @property
    def idle_time(self) -> float:
        """Get the time in seconds since the last activity."""
        return time.time() - self.last_activity_time
    
    @property
    def audio_sample_count(self) -> int:
        """Get the total number of audio samples in the session."""
        return int((self._total_audio_duration_ms / 1000) * 16000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the session to a dictionary representation."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "last_activity_time": self.last_activity_time,
            "is_active": self.is_active,
            "language": self.language,
            "detected_language_confidence": self.detected_language_confidence,
            "current_text": self.current_text,
            "duration_ms": self.duration_ms,
            "audio_sample_count": self.audio_sample_count,
            "config": self.config.to_dict(),
            "result_count": len(self.results)
        }