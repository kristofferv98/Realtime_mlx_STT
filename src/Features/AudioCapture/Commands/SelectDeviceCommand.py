from dataclasses import dataclass
from src.Core.Commands.command import Command


@dataclass
class SelectDeviceCommand(Command):
    """Command to select an audio input device."""
    
    # Device ID to select
    device_id: int
    
    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = "SelectDeviceCommand"