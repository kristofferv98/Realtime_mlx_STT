"""
StartTranscriptionSessionCommand for initiating a new transcription session.

This command creates a new transcription session with the specified parameters,
allowing for tracking of state across multiple audio chunks.
"""

import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from src.Core.Commands.command import Command


class StartTranscriptionSessionCommand(Command):
    """
    Command to start a new transcription session.
    
    When handled, this command will create a new session for managing
    an ongoing transcription operation with consistent settings.
    
    Args:
        session_id: Unique identifier for the session (auto-generated if None)
        language: Optional language code (e.g., 'en', 'fr') or None for auto-detection
        streaming: Whether to use streaming mode
        config: Additional session configuration parameters
    """
    
    def __init__(self,
                session_id: Optional[str] = None,
                language: Optional[str] = None,
                streaming: bool = True,
                config: Optional[Dict[str, Any]] = None,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """
        Initialize the command.
        
        Args:
            session_id: Session ID (auto-generated if None)
            language: Language code
            streaming: Streaming mode flag
            config: Additional configuration
            id: Optional command ID
            timestamp: Optional command timestamp
            name: Optional command name
        """
        # Initialize the base class
        super().__init__(id=id, timestamp=timestamp, name=name)
        
        # Initialize our fields
        self.session_id = session_id or str(uuid.uuid4())
        self.language = language
        self.streaming = streaming
        self.config = config or {}