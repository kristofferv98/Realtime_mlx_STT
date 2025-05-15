from dataclasses import dataclass
from src.Core.Commands.command import Command


@dataclass
class StopRecordingCommand(Command):
    """Command to stop audio recording."""
    
    # Whether to flush any buffered audio
    flush: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = "StopRecordingCommand"