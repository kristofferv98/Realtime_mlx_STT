"""
AudioChunkCapturedEvent for notifying when new audio data is available.

This event is published when a new chunk of audio data has been captured
from an audio input device.
"""

from dataclasses import dataclass
from src.Core.Events.event import Event
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk


@dataclass
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
    
    # The captured audio data
    audio_chunk: AudioChunk
    
    # Source identifier (e.g., "microphone", "file")
    source_id: str
    
    # ID of the device that captured the audio
    device_id: int
    
    # Name of the audio provider (e.g., "PyAudioInputProvider")
    provider_name: str