from dataclasses import dataclass
from src.Core.Commands.command import Command


@dataclass
class ListDevicesCommand(Command):
    """Command to list available audio input devices."""
    
    # Whether to refresh the device list
    refresh: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = "ListDevicesCommand"