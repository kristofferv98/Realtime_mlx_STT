"""
TranscriptionResult model.

This module defines the TranscriptionResult class that represents the output
of a transcription operation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed text with timing information."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """
    Represents the result of a transcription operation.
    
    This class encapsulates the text output along with metadata such as
    confidence scores, language detection, timing information, and processing stats.
    """
    text: str
    is_final: bool
    session_id: str
    timestamp: float
    language: Optional[str] = None
    confidence: float = 1.0
    segments: List[TranscriptionSegment] = field(default_factory=list)
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_segments(self) -> bool:
        """Check if the result contains segment information."""
        return len(self.segments) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary representation."""
        result = {
            "text": self.text,
            "is_final": self.is_final,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }
        
        if self.language:
            result["language"] = self.language
            
        if self.processing_time is not None:
            result["processing_time"] = self.processing_time
            
        if self.has_segments:
            result["segments"] = [
                {
                    "text": segment.text,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "confidence": segment.confidence
                }
                for segment in self.segments
            ]
            
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result