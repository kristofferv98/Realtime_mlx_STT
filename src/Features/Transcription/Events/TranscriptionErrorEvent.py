"""
TranscriptionErrorEvent for notifying when transcription encounters an error.

This event is published when an error occurs during transcription,
providing detailed information about the failure.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from src.Core.Events.event import Event


class TranscriptionErrorEvent(Event):
    """
    Event published when transcription encounters an error.
    
    This event contains information about errors that occur during
    transcription, such as engine failures or processing issues.
    
    Args:
        session_id: Identifier for the transcription session
        error_message: Human-readable error message
        error_type: Type of error that occurred
        audio_timestamp: Timestamp of the audio being processed when the error occurred
        details: Additional error details or context
        recovery_attempted: Whether error recovery was attempted
        recovery_successful: Whether error recovery was successful
        id: Optional event ID
        timestamp: Optional event timestamp
        name: Optional event name
    """
    
    def __init__(self, 
                session_id: str,
                error_message: str,
                error_type: str,
                audio_timestamp: float = 0.0,
                details: Optional[Dict[str, Any]] = None,
                recovery_attempted: bool = False,
                recovery_successful: bool = False,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """
        Initialize the event.
        
        Args:
            session_id: Transcription session ID
            error_message: Error message
            error_type: Error type
            audio_timestamp: Audio timestamp in milliseconds
            details: Additional error details
            recovery_attempted: Whether recovery was attempted
            recovery_successful: Whether recovery was successful
            id: Optional event ID
            timestamp: Optional event timestamp
            name: Optional event name
        """
        # Initialize the base class
        super().__init__(id=id, timestamp=timestamp, name=name)
        
        # Initialize our fields
        self.session_id = session_id
        self.error_message = error_message
        self.error_type = error_type
        self.audio_timestamp = audio_timestamp
        self.details = details or {}
        self.recovery_attempted = recovery_attempted
        self.recovery_successful = recovery_successful