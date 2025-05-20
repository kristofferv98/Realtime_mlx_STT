"""
TranscribeAudioCommand for requesting transcription of audio data.

This command requests that an audio chunk be transcribed, either as part
of a streaming session or as a complete audio segment.
"""

from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
import uuid

from src.Core.Commands.command import Command


# We're using a traditional class approach rather than a dataclass
# due to issues with required and optional parameters in inheritance
class TranscribeAudioCommand(Command):
    """
    Command to transcribe an audio chunk.
    
    When handled, this command will process the provided audio data
    and generate a text transcription.
    
    Args:
        audio_chunk: Audio data as numpy array (float32, -1.0 to 1.0 range)
        session_id: Identifier for the transcription session
        is_first_chunk: Whether this is the first chunk in a session
        is_last_chunk: Whether this is the final chunk in a session
        language: Optional language code (e.g., 'en', 'fr') or None for auto-detection
        timestamp_ms: Timestamp of the audio in milliseconds
        options: Additional transcription options
    """
    
    def __init__(self, 
                audio_chunk: Any,
                session_id: str,
                is_first_chunk: bool = False,
                is_last_chunk: bool = False,
                timestamp_ms: float = 0.0,
                language: Optional[str] = None,
                options: Optional[Dict[str, Any]] = None,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """
        Initialize the command.
        
        Args:
            audio_chunk: Audio data as either numpy array or file path (str)
            session_id: Session identifier
            is_first_chunk: Whether this is the first chunk
            is_last_chunk: Whether this is the last chunk
            timestamp_ms: Audio timestamp in milliseconds
            language: Language code
            options: Additional options
            id: Optional command ID
            timestamp: Optional command timestamp
            name: Optional command name
        """
        # Initialize the base class
        super().__init__(id=id, timestamp=timestamp, name=name)
        
        # Initialize our fields
        self.audio_chunk = audio_chunk
        self.session_id = session_id
        self.is_first_chunk = is_first_chunk
        self.is_last_chunk = is_last_chunk
        self.timestamp_ms = timestamp_ms
        self.language = language
        self.options = options or {}
        
        # Flag to indicate if audio_chunk is a file path
        self.is_file_path = isinstance(self.audio_chunk, str)
        
        # Validate audio data if it's a numpy array
        if not self.is_file_path and not isinstance(self.audio_chunk, np.ndarray):
            raise TypeError("audio_chunk must be a numpy ndarray or a file path (str)")
        
        # Ensure audio is float32 in [-1.0, 1.0] range if it's a numpy array
        if not self.is_file_path and hasattr(self.audio_chunk, 'dtype') and self.audio_chunk.dtype != np.float32:
            self.audio_chunk = self.audio_chunk.astype(np.float32)
        
        # Normalize if not already in [-1, 1] range and it's a numpy array
        if not self.is_file_path and isinstance(self.audio_chunk, np.ndarray):
            max_val = np.max(np.abs(self.audio_chunk))
            if max_val > 0 and max_val > 1.0:
                self.audio_chunk = self.audio_chunk / max_val