"""
StopTranscriptionSessionCommand for finalizing a transcription session.

This command ends a transcription session and optionally processes
any remaining audio in the buffer.
"""

from typing import Optional
from datetime import datetime

from src.Core.Commands.command import Command


class StopTranscriptionSessionCommand(Command):
    """
    Command to stop an ongoing transcription session.
    
    When handled, this command will finalize a transcription session and
    clean up any resources associated with it.
    
    Args:
        session_id: Identifier of the session to stop
        flush_remaining_audio: Whether to process any remaining audio in the buffer
        save_results: Whether to save the transcription results
        output_path: Path to save the results to (if save_results is True)
    """
    
    def __init__(self,
                session_id: str,
                flush_remaining_audio: bool = True,
                save_results: bool = False,
                output_path: Optional[str] = None,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """
        Initialize the command.
        
        Args:
            session_id: Session ID
            flush_remaining_audio: Whether to process remaining audio
            save_results: Whether to save results
            output_path: Path to save results
            id: Optional command ID
            timestamp: Optional command timestamp
            name: Optional command name
        """
        # Initialize the base class
        super().__init__(id=id, timestamp=timestamp, name=name)
        
        # Initialize our fields
        self.session_id = session_id
        self.flush_remaining_audio = flush_remaining_audio
        self.save_results = save_results
        self.output_path = output_path