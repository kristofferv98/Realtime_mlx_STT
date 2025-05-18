"""
AudioChunkCapturedEvent for notifying when new audio data is available.

This event is published when a new chunk of audio data has been captured
from an audio input device.
"""

import uuid
from datetime import datetime
from typing import Optional

from src.Core.Events.event import Event
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk


class AudioChunkCapturedEvent(Event):
    """
    Event published when a new chunk of audio data is captured.
    
    This event contains the captured audio data along with metadata about
    the capture such as the device ID, source, and sequence number for ordering.
    
    Args:
        audio_chunk: The captured audio data with its metadata
        source_id: A string identifier for the audio source (e.g., "microphone", "file")
        device_id: The ID of the device that captured the audio
        provider_name: The name of the audio provider (e.g., "PyAudioInputProvider")
    """
    
    def __init__(self, 
                audio_chunk: AudioChunk,
                source_id: str,
                device_id: int,
                provider_name: str,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """Initialize the event with the required parameters."""
        super().__init__(id=id, timestamp=timestamp, name=name)
        self.audio_chunk = audio_chunk
        self.source_id = source_id
        self.device_id = device_id
        self.provider_name = provider_name