import logging
import threading
import time
from typing import Dict, List, Optional, Any, Type, cast

from src.Core.Commands.command import Command
from src.Core.Common.Interfaces.audio_provider import IAudioProvider
from src.Core.Common.Interfaces.command_handler import ICommandHandler
from src.Core.Events.event_bus import IEventBus

from src.Features.AudioCapture.Commands.StartRecordingCommand import StartRecordingCommand
from src.Features.AudioCapture.Commands.StopRecordingCommand import StopRecordingCommand
from src.Features.AudioCapture.Commands.SelectDeviceCommand import SelectDeviceCommand
from src.Features.AudioCapture.Commands.ListDevicesCommand import ListDevicesCommand

from src.Features.AudioCapture.Events.AudioChunkCapturedEvent import AudioChunkCapturedEvent
from src.Features.AudioCapture.Events.RecordingStateChangedEvent import RecordingStateChangedEvent

from src.Features.AudioCapture.Models.AudioChunk import AudioChunk
from src.Features.AudioCapture.Models.DeviceInfo import DeviceInfo

from src.Features.AudioCapture.Providers.PyAudioInputProvider import PyAudioInputProvider

logger = logging.getLogger(__name__)

class AudioCommandHandler(ICommandHandler):
    """
    Handles audio-related commands in the system.
    """
    
    def __init__(self, event_bus: IEventBus):
        """
        Initialize the audio command handler.
        
        Args:
            event_bus: Event bus for publishing events
        """
        self.event_bus = event_bus
        self.audio_provider: Optional[IAudioProvider] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.sequence_counter = 0
        
    def can_handle(self, command: Command) -> bool:
        """
        Determine if this handler can process the given command.
        
        Args:
            command: Command to check
            
        Returns:
            bool: True if this handler can process the command
        """
        return isinstance(command, (
            StartRecordingCommand, 
            StopRecordingCommand,
            SelectDeviceCommand,
            ListDevicesCommand
        ))
    
    def handle(self, command: Command) -> bool:
        """
        Process the given command.
        
        Args:
            command: Command to process
            
        Returns:
            bool: True if the command was processed successfully
        """
        if isinstance(command, StartRecordingCommand):
            return self._handle_start_recording(command)
        elif isinstance(command, StopRecordingCommand):
            return self._handle_stop_recording(command)
        elif isinstance(command, SelectDeviceCommand):
            return self._handle_select_device(command)
        elif isinstance(command, ListDevicesCommand):
            return self._handle_list_devices(command)
        
        return False
    
    def _handle_start_recording(self, command: StartRecordingCommand) -> bool:
        """
        Handle the StartRecordingCommand.
        
        Args:
            command: The command to process
            
        Returns:
            bool: True if handled successfully
        """
        # Stop any existing recording
        if self.recording_thread and self.recording_thread.is_alive():
            self._handle_stop_recording(StopRecordingCommand())
        
        # Create audio provider if needed
        if self.audio_provider is None:
            self.audio_provider = PyAudioInputProvider(
                input_device_index=command.device_id,
                debug_mode=command.debug_mode,
                target_samplerate=command.sample_rate,
                chunk_size=command.chunk_size,
                channels=command.channels,
            )
        
        # Set up the audio provider
        if not self.audio_provider.setup():
            logger.error("Failed to set up audio provider")
            self.event_bus.publish(RecordingStateChangedEvent(
                state="error",
                message="Failed to set up audio provider"
            ))
            return False
        
        # Start the audio provider
        if not self.audio_provider.start():
            logger.error("Failed to start audio provider")
            self.event_bus.publish(RecordingStateChangedEvent(
                state="error",
                message="Failed to start audio provider"
            ))
            return False
        
        # Reset the stop event
        self.stop_event.clear()
        
        # Start the recording thread
        self.recording_thread = threading.Thread(
            target=self._recording_loop,
            daemon=True
        )
        self.recording_thread.start()
        
        # Publish state changed event
        self.event_bus.publish(RecordingStateChangedEvent(
            state="started",
            device_id=command.device_id
        ))
        
        return True
    
    def _handle_stop_recording(self, command: StopRecordingCommand) -> bool:
        """
        Handle the StopRecordingCommand.
        
        Args:
            command: The command to process
            
        Returns:
            bool: True if handled successfully
        """
        # Signal the recording thread to stop
        self.stop_event.set()
        
        # Stop the audio provider
        if self.audio_provider:
            self.audio_provider.stop()
        
        # Wait for the recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
            if self.recording_thread.is_alive():
                logger.warning("Recording thread did not terminate gracefully")
        
        # Reset the thread
        self.recording_thread = None
        
        # Publish state changed event
        self.event_bus.publish(RecordingStateChangedEvent(
            state="stopped"
        ))
        
        return True
    
    def _handle_select_device(self, command: SelectDeviceCommand) -> bool:
        """
        Handle the SelectDeviceCommand.
        
        Args:
            command: The command to process
            
        Returns:
            bool: True if handled successfully
        """
        # Stop any existing recording
        if self.recording_thread and self.recording_thread.is_alive():
            self._handle_stop_recording(StopRecordingCommand())
        
        # Clean up existing provider
        if self.audio_provider:
            self.audio_provider.cleanup()
        
        # Create a new provider with the selected device
        self.audio_provider = PyAudioInputProvider(input_device_index=command.device_id)
        
        # Publish state changed event
        self.event_bus.publish(RecordingStateChangedEvent(
            state="device_changed",
            device_id=command.device_id
        ))
        
        return True
    
    def _handle_list_devices(self, command: ListDevicesCommand) -> bool:
        """
        Handle the ListDevicesCommand.
        
        Args:
            command: The command to process
            
        Returns:
            bool: True if handled successfully
        """
        # Create a temporary provider if needed
        temp_provider = self.audio_provider or PyAudioInputProvider()
        
        # Get the device list
        device_list = temp_provider.list_devices()
        
        # Convert to DeviceInfo objects
        devices = []
        for device_dict in device_list:
            device = DeviceInfo(
                id=device_dict['id'],
                name=device_dict['name'],
                max_input_channels=device_dict['max_input_channels'],
                default_sample_rate=device_dict['default_sample_rate'],
                supported_sample_rates=device_dict['supported_sample_rates'],
                is_default=device_dict['is_default'],
                extra_info=device_dict.get('original_info')
            )
            devices.append(device)
        
        # If we created a temporary provider, clean it up
        if self.audio_provider is None:
            temp_provider.cleanup()
        
        # Store the devices in the result
        setattr(command, 'result', devices)
        
        return True
    
    def _recording_loop(self):
        """Recording thread that captures audio chunks and publishes events."""
        logger.debug("Recording loop started")
        
        try:
            # Main recording loop
            while not self.stop_event.is_set() and self.audio_provider and self.audio_provider.is_running():
                # Read a chunk of audio data
                raw_data = self.audio_provider.read_chunk()
                
                if not raw_data:
                    # Empty chunk, possibly end of file or error
                    logger.debug("Received empty audio chunk")
                    time.sleep(0.01)  # Avoid busy waiting
                    continue
                
                # Create an AudioChunk
                chunk = AudioChunk(
                    raw_data=raw_data,
                    sample_rate=self.audio_provider.get_sample_rate(),
                    channels=1,  # Assuming mono
                    format='int16',  # Assuming 16-bit PCM
                    timestamp=time.time(),
                    sequence_number=self.sequence_counter
                )
                self.sequence_counter += 1
                
                # Publish the audio chunk event
                self.event_bus.publish(AudioChunkCapturedEvent(
                    audio_chunk=chunk
                ))
                
                # Small sleep to avoid overwhelming the system
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Error in recording loop: {e}", exc_info=True)
            self.event_bus.publish(RecordingStateChangedEvent(
                state="error",
                message=f"Recording error: {str(e)}"
            ))
        finally:
            logger.debug("Recording loop ended")