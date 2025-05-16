"""
TranscriptionStartedEvent for notifying when transcription begins.

This event is published when a new transcription session starts or
a new segment of speech is detected for transcription.
"""

from datetime import datetime
from src.Core.Events.event import Event


class TranscriptionStartedEvent(Event):
    """
    Event published when transcription begins.
    
    This event indicates that the system has started processing audio
    for transcription, either as a new session or a new speech segment.
    
    Args:
        session_id: Identifier for the transcription session
        language: Language code (e.g., 'en', 'fr') or None if auto-detecting
        audio_timestamp: Timestamp of the audio being transcribed (ms)
        id: Optional event ID
        timestamp: Optional event timestamp
        name: Optional event name
    """
    
    def __init__(self, 
                session_id: str,
                language: str = None,
                audio_timestamp: float = 0.0,
                id: str = None,
                timestamp: datetime = None,
                name: str = None):
        """
        Initialize the event.
        
        Args:
            session_id: Transcription session ID
            language: Language code
            audio_timestamp: Audio timestamp in milliseconds
            id: Optional event ID
            timestamp: Optional event timestamp
            name: Optional event name
        """
        # Initialize the base class
        super().__init__(id=id, timestamp=timestamp, name=name)
        
        # Initialize our fields
        self.session_id = session_id
        self.language = language
        self.audio_timestamp = audio_timestamp