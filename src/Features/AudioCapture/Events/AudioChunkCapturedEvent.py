from dataclasses import dataclass
from src.Core.Events.event import Event
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk


@dataclass
class AudioChunkCapturedEvent(Event):
    """Event emitted when an audio chunk is captured."""
    
    # The captured audio chunk
    audio_chunk: AudioChunk
    
    # Whether this is the last chunk in a sequence
    is_last: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = "AudioChunkCapturedEvent"