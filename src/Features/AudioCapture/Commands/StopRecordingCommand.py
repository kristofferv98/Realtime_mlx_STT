"""
StopRecordingCommand for ending audio recording.

This command requests that the current audio recording be stopped.
"""

from dataclasses import dataclass
from typing import Optional
from src.Core.Commands.command import Command


@dataclass
class StopRecordingCommand(Command):
    """
    Command to stop the current audio recording.
    
    When handled, this command will stop any active audio recording and
    clean up resources as needed.
    
    Args:
        flush_buffer: Whether to flush the audio buffer after stopping
        save_recording: Whether to save the recorded audio
        output_path: Optional path to save the recording to, if save_recording is True
    """
    
    # Whether to flush the audio buffer after stopping
    flush_buffer: bool = False
    
    # Whether to save the recorded audio
    save_recording: bool = False
    
    # Path to save the recording to, if save_recording is True
    output_path: Optional[str] = None