"""
AudioCaptureModule for registering and providing access to audio capture functionality.

This module serves as the main entry point for the AudioCapture feature,
handling registration of commands, providers, and event handlers.
"""

from typing import Dict, List, Any, Optional, Callable

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import IEventBus
from src.Core.Common.Interfaces.audio_provider import IAudioProvider
from src.Features.AudioCapture.Commands.ListDevicesCommand import ListDevicesCommand
from src.Features.AudioCapture.Commands.SelectDeviceCommand import SelectDeviceCommand
from src.Features.AudioCapture.Commands.StartRecordingCommand import StartRecordingCommand
from src.Features.AudioCapture.Commands.StopRecordingCommand import StopRecordingCommand
from src.Features.AudioCapture.Events.AudioChunkCapturedEvent import AudioChunkCapturedEvent
from src.Features.AudioCapture.Events.RecordingStateChangedEvent import RecordingStateChangedEvent, RecordingState
from src.Features.AudioCapture.Handlers.AudioCommandHandler import AudioCommandHandler
from src.Features.AudioCapture.Providers.PyAudioInputProvider import PyAudioInputProvider
from src.Features.AudioCapture.Providers.FileAudioProvider import FileAudioProvider
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk


class AudioCaptureModule:
    """
    Module for audio capture functionality.
    
    This class provides registration and convenience methods for the AudioCapture feature.
    It serves as a facade for the underlying components (providers, handlers, etc.).
    """
    
    @staticmethod
    def register(command_dispatcher: CommandDispatcher,
                event_bus: IEventBus,
                use_microphone: bool = True,
                use_file: bool = True,
                file_path: Optional[str] = None,
                default_sample_rate: int = 16000,
                default_chunk_size: int = 512) -> AudioCommandHandler:
        """
        Register the AudioCapture feature with the system.
        
        This method:
        1. Creates the audio providers
        2. Registers the command handler
        3. Sets up any necessary subscriptions
        
        Args:
            command_dispatcher: The command dispatcher to register handlers with
            event_bus: The event bus for publishing/subscribing to events
            use_microphone: Whether to create a microphone provider
            use_file: Whether to create a file provider
            file_path: Path to audio file (for file provider)
            default_sample_rate: Default sample rate to use (Hz)
            default_chunk_size: Default chunk size to use (samples)
            
        Returns:
            AudioCommandHandler: The registered command handler
        """
        logger = LoggingModule.get_logger(__name__)
        logger.info("Registering AudioCapture feature")
        
        # Create providers
        providers: Dict[str, IAudioProvider] = {}
        
        if use_microphone:
            logger.info("Creating microphone provider")
            mic_provider = PyAudioInputProvider(
                event_bus=event_bus,
                sample_rate=default_sample_rate,
                chunk_size=default_chunk_size
            )
            providers["microphone"] = mic_provider
        
        if use_file:
            logger.info("Creating file provider")
            file_provider = FileAudioProvider(
                event_bus=event_bus,
                file_path=file_path,
                target_sample_rate=default_sample_rate,
                chunk_size=default_chunk_size
            )
            providers["file"] = file_provider
        
        # Determine default provider
        active_provider = "microphone" if use_microphone else "file" if use_file else None
        
        if not active_provider:
            logger.warning("No audio providers created")
            if not providers:
                raise ValueError("At least one provider (microphone or file) must be enabled")
            active_provider = next(iter(providers))
        
        # Create command handler
        handler = AudioCommandHandler(
            event_bus=event_bus,
            providers=providers,
            active_provider=active_provider
        )
        
        # Register with command dispatcher
        command_dispatcher.register_handler(ListDevicesCommand, handler)
        command_dispatcher.register_handler(SelectDeviceCommand, handler)
        command_dispatcher.register_handler(StartRecordingCommand, handler)
        command_dispatcher.register_handler(StopRecordingCommand, handler)
        
        # Initialize providers
        for provider_name, provider in providers.items():
            try:
                if provider.setup():
                    logger.info(f"Successfully set up {provider_name} provider")
                else:
                    logger.warning(f"Failed to set up {provider_name} provider")
            except Exception as e:
                logger.error(f"Error setting up {provider_name} provider: {e}")
        
        return handler
    
    @staticmethod
    def list_devices(command_dispatcher: CommandDispatcher) -> List[Dict[str, Any]]:
        """
        List all available audio input devices.
        
        Args:
            command_dispatcher: The command dispatcher to use
            
        Returns:
            List[Dict[str, Any]]: List of device information dictionaries
        """
        command = ListDevicesCommand()
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def select_device(command_dispatcher: CommandDispatcher, 
                     device_id: Optional[int] = None) -> bool:
        """
        Select an audio device for recording.
        
        Args:
            command_dispatcher: The command dispatcher to use
            device_id: The ID of the device to select, or None for default
            
        Returns:
            bool: True if device was successfully selected
        """
        command = SelectDeviceCommand(device_id=device_id)
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def start_recording(command_dispatcher: CommandDispatcher,
                       sample_rate: int = 16000,
                       chunk_size: int = 512,
                       device_id: Optional[int] = None) -> bool:
        """
        Start audio recording.
        
        Args:
            command_dispatcher: The command dispatcher to use
            sample_rate: The desired sample rate in Hz
            chunk_size: The desired chunk size in samples
            device_id: Optional device ID to use
            
        Returns:
            bool: True if recording was successfully started
        """
        command = StartRecordingCommand(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            device_id=device_id
        )
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def stop_recording(command_dispatcher: CommandDispatcher,
                      save_recording: bool = False,
                      output_path: Optional[str] = None) -> bool:
        """
        Stop audio recording.
        
        Args:
            command_dispatcher: The command dispatcher to use
            save_recording: Whether to save the recorded audio
            output_path: Path to save the recording to
            
        Returns:
            bool: True if recording was successfully stopped
        """
        command = StopRecordingCommand(
            save_recording=save_recording,
            output_path=output_path
        )
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def on_audio_chunk_captured(event_bus: IEventBus, 
                               handler: Callable[[AudioChunk], None]) -> None:
        """
        Subscribe to audio chunk captured events.
        
        Args:
            event_bus: The event bus to subscribe to
            handler: Function to call when an audio chunk is captured
        """
        event_bus.subscribe(AudioChunkCapturedEvent, 
                           lambda event: handler(event.audio_chunk))
    
    @staticmethod
    def on_recording_state_changed(event_bus: IEventBus,
                                  handler: Callable[[RecordingState, RecordingState], None]) -> None:
        """
        Subscribe to recording state changed events.
        
        Args:
            event_bus: The event bus to subscribe to
            handler: Function to call when recording state changes
        """
        event_bus.subscribe(RecordingStateChangedEvent,
                           lambda event: handler(event.previous_state, event.current_state))