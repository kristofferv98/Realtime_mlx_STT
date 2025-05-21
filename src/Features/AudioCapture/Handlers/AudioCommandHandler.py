"""
AudioCommandHandler for processing audio capture commands.

This handler implements the ICommandHandler interface to process audio-related commands
like listing devices, selecting devices, and controlling recording.
"""

from typing import Any, Dict, List, Optional, Union, Type, cast

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

from src.Core.Common.Interfaces.command_handler import ICommandHandler
from src.Core.Commands.command import Command
from src.Core.Events.event_bus import IEventBus
from src.Features.AudioCapture.Commands.ListDevicesCommand import ListDevicesCommand
from src.Features.AudioCapture.Commands.SelectDeviceCommand import SelectDeviceCommand
from src.Features.AudioCapture.Commands.StartRecordingCommand import StartRecordingCommand
from src.Features.AudioCapture.Commands.StopRecordingCommand import StopRecordingCommand
from src.Features.AudioCapture.Models.DeviceInfo import DeviceInfo
from src.Core.Common.Interfaces.audio_provider import IAudioProvider
from src.Features.AudioCapture.Events.RecordingStateChangedEvent import RecordingState


class AudioCommandHandler(ICommandHandler[Any]):
    """
    Handler for audio capture commands.
    
    This handler processes commands related to audio input devices and recording control.
    It delegates to the appropriate audio provider for implementation.
    """
    
    def __init__(self, 
                 event_bus: IEventBus,
                 providers: Dict[str, IAudioProvider],
                 active_provider: str = "microphone"):
        """
        Initialize the audio command handler.
        
        Args:
            event_bus: Event bus for publishing events
            providers: Dictionary of available audio providers (e.g., microphone, file)
            active_provider: Key of the provider to use by default
        """
        self.logger = LoggingModule.get_logger(__name__)
        self.event_bus = event_bus
        self.providers = providers
        self.active_provider = active_provider
        
        if self.active_provider not in self.providers:
            available_providers = ", ".join(self.providers.keys())
            self.logger.warning(f"Active provider '{active_provider}' not found. "
                               f"Available providers: {available_providers}")
            
            # Set active_provider to the first available provider if not found
            if self.providers:
                self.active_provider = next(iter(self.providers))
                self.logger.info(f"Using '{self.active_provider}' as active provider instead")
    
    def handle(self, command: Command) -> Any:
        """
        Handle an audio command and produce a result.
        
        Args:
            command: The command to handle
            
        Returns:
            The result of the command execution (type depends on command)
            
        Raises:
            TypeError: If the command is not of the expected type
            ValueError: If the command contains invalid data
            Exception: If an error occurs during command execution
        """
        if isinstance(command, ListDevicesCommand):
            return self._handle_list_devices(command)
        elif isinstance(command, SelectDeviceCommand):
            return self._handle_select_device(command)
        elif isinstance(command, StartRecordingCommand):
            return self._handle_start_recording(command)
        elif isinstance(command, StopRecordingCommand):
            return self._handle_stop_recording(command)
        else:
            raise TypeError(f"Unsupported command type: {type(command).__name__}")
    
    def can_handle(self, command: Command) -> bool:
        """
        Check if this handler can handle the given command.
        
        Args:
            command: The command to check
            
        Returns:
            bool: True if this handler can handle the command, False otherwise
        """
        return isinstance(command, (
            ListDevicesCommand,
            SelectDeviceCommand,
            StartRecordingCommand,
            StopRecordingCommand
        ))
    
    def _get_provider(self, provider_key: Optional[str] = None) -> IAudioProvider:
        """
        Get the specified provider or the active provider.
        
        Args:
            provider_key: Key of the provider to get, or None for active provider
            
        Returns:
            IAudioProvider: The requested audio provider
            
        Raises:
            ValueError: If the provider key is not found
        """
        key = provider_key or self.active_provider
        
        if key not in self.providers:
            available_providers = ", ".join(self.providers.keys())
            raise ValueError(f"Provider '{key}' not found. Available providers: {available_providers}")
        
        return self.providers[key]
    
    def _handle_list_devices(self, command: ListDevicesCommand) -> List[Dict[str, Any]]:
        """
        Handle a ListDevicesCommand by listing available audio devices.
        
        Args:
            command: The ListDevicesCommand
            
        Returns:
            List[Dict[str, Any]]: List of device information dictionaries
        """
        self.logger.info("Handling ListDevicesCommand")
        
        all_devices = []
        
        # Collect devices from all providers
        for provider_key, provider in self.providers.items():
            try:
                devices = provider.list_devices()
                
                # Add provider information to each device
                for device in devices:
                    device['provider'] = provider_key
                
                all_devices.extend(devices)
                
            except Exception as e:
                self.logger.error(f"Error listing devices from provider '{provider_key}': {e}")
        
        return all_devices
    
    def _handle_select_device(self, command: SelectDeviceCommand) -> bool:
        """
        Handle a SelectDeviceCommand by selecting an audio device.
        
        Args:
            command: The SelectDeviceCommand
            
        Returns:
            bool: True if device was successfully selected
        """
        device_id = command.device_id
        self.logger.info(f"Handling SelectDeviceCommand for device_id={device_id}")
        
        # If device_id is None, use default device
        if device_id is None:
            self.logger.info("No device ID specified, using default device")
            return True
        
        # Find the provider for this device
        for provider_key, provider in self.providers.items():
            devices = provider.list_devices()
            
            # Check if this device belongs to this provider
            if any(device['device_id'] == device_id for device in devices):
                # Set this as the active provider
                self.active_provider = provider_key
                self.logger.info(f"Selected device {device_id} from provider '{provider_key}'")
                return True
        
        self.logger.warning(f"Device with ID {device_id} not found in any provider")
        return False
    
    def _handle_start_recording(self, command: StartRecordingCommand) -> bool:
        """
        Handle a StartRecordingCommand by starting audio recording.
        
        Args:
            command: The StartRecordingCommand
            
        Returns:
            bool: True if recording was successfully started
        """
        self.logger.info("Handling StartRecordingCommand")
        
        # Select provider based on device_id if specified
        if command.device_id is not None:
            self._handle_select_device(SelectDeviceCommand(device_id=command.device_id))
        
        provider = self._get_provider()
        
        # Make sure provider is set up
        if not provider.setup():
            self.logger.error("Failed to set up audio provider")
            return False
        
        # Start recording
        return provider.start()
    
    def _handle_stop_recording(self, command: StopRecordingCommand) -> bool:
        """
        Handle a StopRecordingCommand by stopping audio recording.
        
        Args:
            command: The StopRecordingCommand
            
        Returns:
            bool: True if recording was successfully stopped
        """
        self.logger.info("Handling StopRecordingCommand")
        
        provider = self._get_provider()
        
        # Stop recording
        result = provider.stop()
        
        # Handle additional options
        if result and command.flush_buffer:
            self.logger.info("Flushing audio buffer")
            # This assumes the provider has a flush_buffer method, which isn't in the interface
            # If needed, this could be implemented in provider-specific ways
        
        if result and command.save_recording and command.output_path:
            self.logger.info(f"Saving recording to {command.output_path}")
            # This assumes the provider has a save_recording method, which isn't in the interface
            # If needed, this could be implemented in provider-specific ways
        
        return result