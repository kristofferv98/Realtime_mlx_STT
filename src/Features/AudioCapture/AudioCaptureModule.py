from typing import Any, Dict, List

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import IEventBus
from src.Features.AudioCapture.Commands.StartRecordingCommand import StartRecordingCommand
from src.Features.AudioCapture.Commands.StopRecordingCommand import StopRecordingCommand
from src.Features.AudioCapture.Commands.SelectDeviceCommand import SelectDeviceCommand
from src.Features.AudioCapture.Commands.ListDevicesCommand import ListDevicesCommand
from src.Features.AudioCapture.Handlers.AudioCommandHandler import AudioCommandHandler

class AudioCaptureModule:
    """
    Module that provides audio capture functionality.
    
    This module encapsulates all audio capture related components and
    registers them with the system.
    """
    
    @staticmethod
    def register(command_dispatcher: CommandDispatcher, event_bus: IEventBus) -> None:
        """
        Register all audio capture components with the system.
        
        Args:
            command_dispatcher: Command dispatcher to register handlers with
            event_bus: Event bus for publishing and subscribing to events
        """
        # Create the audio command handler
        audio_handler = AudioCommandHandler(event_bus)
        
        # Register the handler with the command dispatcher
        command_dispatcher.register_handler(StartRecordingCommand, audio_handler)
        command_dispatcher.register_handler(StopRecordingCommand, audio_handler)
        command_dispatcher.register_handler(SelectDeviceCommand, audio_handler)
        command_dispatcher.register_handler(ListDevicesCommand, audio_handler)

    @staticmethod
    def list_devices(command_dispatcher: CommandDispatcher) -> List[Dict[str, Any]]:
        """
        Helper method to list available audio devices.
        
        Args:
            command_dispatcher: Command dispatcher to dispatch the command
            
        Returns:
            List[Dict[str, Any]]: List of available audio devices
        """
        command = ListDevicesCommand()
        command_dispatcher.dispatch(command)
        return getattr(command, 'result', [])