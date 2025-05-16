"""
TranscriptionUpdatedEvent for notifying when transcription text is updated.

This event is published when new transcription text is available,
either as a partial result during streaming or a final result.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from src.Core.Events.event import Event


class TranscriptionUpdatedEvent(Event):
    """
    Event published when transcription text is updated.
    
    This event contains the latest transcription text along with
    metadata about the transcription process.
    
    Args:
        session_id: Identifier for the transcription session
        text: The transcribed text
        is_final: Whether this is a final result
        confidence: Confidence score (0.0-1.0)
        language: Detected language code
        audio_timestamp: Timestamp of the audio being transcribed (ms)
        processing_time: Time taken to process the audio (ms)
        segments: Optional list of time-aligned text segments
        metadata: Additional transcription metadata
        id: Optional event ID
        timestamp: Optional event timestamp
        name: Optional event name
    """
    
    def __init__(self, 
                session_id: str,
                text: str,
                is_final: bool,
                confidence: float = 1.0,
                language: Optional[str] = None,
                audio_timestamp: float = 0.0,
                processing_time: Optional[float] = None,
                segments: Optional[List[Dict[str, Any]]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """
        Initialize the event.
        
        Args:
            session_id: Transcription session ID
            text: Transcribed text
            is_final: Whether this is a final result
            confidence: Confidence score
            language: Detected language code
            audio_timestamp: Audio timestamp in milliseconds
            processing_time: Processing time in milliseconds
            segments: List of time-aligned segments
            metadata: Additional metadata
            id: Optional event ID
            timestamp: Optional event timestamp
            name: Optional event name
        """
        # Initialize the base class
        super().__init__(id=id, timestamp=timestamp, name=name)
        
        # Initialize our fields
        self.session_id = session_id
        self.text = text
        self.is_final = is_final
        self.confidence = confidence
        self.language = language
        self.audio_timestamp = audio_timestamp
        self.processing_time = processing_time
        self.segments = segments or []
        self.metadata = metadata or {}