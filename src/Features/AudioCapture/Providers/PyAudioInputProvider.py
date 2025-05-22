"""
PyAudioInputProvider for capturing audio from microphones using PyAudio.

This provider implements the IAudioProvider interface using PyAudio to
capture audio from system microphones.
"""

import threading
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pyaudio
from scipy import signal

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

from src.Core.Common.Interfaces.audio_provider import IAudioProvider
from src.Core.Events.event_bus import IEventBus
from src.Features.AudioCapture.Models.DeviceInfo import DeviceInfo
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk
from src.Features.AudioCapture.Events.AudioChunkCapturedEvent import AudioChunkCapturedEvent
from src.Features.AudioCapture.Events.RecordingStateChangedEvent import (
    RecordingStateChangedEvent, RecordingState
)


class PyAudioInputProvider(IAudioProvider):
    """
    PyAudio implementation of IAudioProvider for microphone input.
    
    This provider uses PyAudio to access system microphones and capture
    audio data. It supports listing devices, configuring recording parameters,
    and streaming audio data in chunks.
    """
    
    def __init__(self, 
                 event_bus: IEventBus, 
                 device_id: Optional[int] = None,
                 sample_rate: int = 16000,
                 chunk_size: int = 512,
                 channels: int = 1,
                 audio_format: int = pyaudio.paInt16,
                 debug_mode: bool = False):
        """
        Initialize the PyAudio input provider.
        
        Args:
            event_bus: Event bus to publish events to
            device_id: Device ID to use (None for default device)
            sample_rate: Sample rate in Hz (default: 16000)
            chunk_size: Chunk size in samples (default: 512)
            channels: Number of audio channels (default: 1 for mono)
            audio_format: PyAudio format (default: paInt16)
            debug_mode: Whether to print debug information
        """
        self.logger = LoggingModule.get_logger(__name__)
        self.event_bus = event_bus
        self.device_id = device_id
        self.target_sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_format = audio_format
        self.debug_mode = debug_mode
        
        # PyAudio objects
        self.audio_interface = None
        self.stream = None
        
        # Recording state tracking
        self.is_recording = False
        self.recording_thread = None
        self.stop_recording_event = threading.Event()
        self.sequence_number = 0
        self.device_sample_rate = None
        self.current_state = RecordingState.INITIALIZED
        
    def setup(self) -> bool:
        """
        Initialize the PyAudio interface and prepare for recording.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            self.logger.info("Setting up PyAudio input provider")
            self.audio_interface = pyaudio.PyAudio()
            
            # Get the actual device to use (default if None)
            actual_device_index = self.device_id
            if actual_device_index is None:
                actual_device_index = self.audio_interface.get_default_input_device_info()['index']
                self.logger.info(f"Using default input device: {actual_device_index}")
            
            self.device_id = actual_device_index
            
            # Determine best sample rate for device
            self.device_sample_rate = self._get_best_sample_rate(actual_device_index)
            
            if self.debug_mode:
                self.logger.debug(f"Setting up audio on device {self.device_id} with sample rate {self.device_sample_rate}")
            
            # Publish state change event
            self._publish_state_change(
                RecordingState.INITIALIZED, 
                RecordingState.INITIALIZED
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing PyAudio: {e}")
            if self.audio_interface:
                self.audio_interface.terminate()
                self.audio_interface = None
            
            # Publish error state
            self._publish_state_change(
                RecordingState.INITIALIZED,
                RecordingState.ERROR,
                error_message=str(e)
            )
            
            return False
    
    def start(self) -> bool:
        """
        Start the audio capture process.
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return False
        
        if not self.audio_interface:
            self.logger.error("PyAudio not initialized. Call setup() first.")
            return False
        
        try:
            # Update state
            self._publish_state_change(
                self.current_state,
                RecordingState.STARTING
            )
            
            # Open audio stream
            self.stream = self.audio_interface.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.device_sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_id,
            )
            
            # Reset sequence number and start recording thread
            self.sequence_number = 0
            self.stop_recording_event.clear()
            self.is_recording = True
            
            self.recording_thread = threading.Thread(
                target=self._recording_worker,
                daemon=False  # Non-daemon for proper cleanup
            )
            self.recording_thread.start()
            
            # Update state
            self._publish_state_change(
                RecordingState.STARTING,
                RecordingState.RECORDING
            )
            
            self.logger.info(f"Started recording on device {self.device_id} at {self.device_sample_rate} Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting recording: {e}")
            
            # Clean up if stream was created
            if self.stream:
                self.stream.close()
                self.stream = None
            
            # Update state
            self._publish_state_change(
                self.current_state,
                RecordingState.ERROR,
                error_message=str(e)
            )
            
            return False
    
    def stop(self) -> bool:
        """
        Stop the audio capture process.
        
        Returns:
            bool: True if successfully stopped, False otherwise
        """
        if not self.is_recording:
            self.logger.warning("No recording in progress")
            return False
        
        try:
            # Update state
            self._publish_state_change(
                self.current_state,
                RecordingState.STOPPING
            )
            
            # Signal recording thread to stop and wait for it
            self.stop_recording_event.set()
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)
                if self.recording_thread.is_alive():
                    self.logger.error("Recording thread did not stop in time, forcing termination")
                    # Force stop by setting is_recording to False
                    self.is_recording = False
                    # Try join once more with shorter timeout
                    self.recording_thread.join(timeout=0.5)
            
            # Close the stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            self.is_recording = False
            
            # Update state
            self._publish_state_change(
                RecordingState.STOPPING,
                RecordingState.STOPPED
            )
            
            self.logger.info("Stopped recording")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping recording: {e}")
            
            # Force stop recording state regardless of error
            self.is_recording = False
            if self.stream:
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            
            # Update state
            self._publish_state_change(
                self.current_state,
                RecordingState.ERROR,
                error_message=str(e)
            )
            
            return False
    
    def read_chunk(self) -> bytes:
        """
        Read a single chunk of audio data.
        
        Returns:
            bytes: Raw audio data as bytes
        """
        if not self.is_recording or not self.stream:
            self.logger.warning("Cannot read chunk: no active recording")
            return b''
        
        try:
            return self.stream.read(self.chunk_size, exception_on_overflow=False)
        except Exception as e:
            self.logger.error(f"Error reading audio chunk: {e}")
            return b''
    
    def get_sample_rate(self) -> int:
        """
        Get the sample rate of the audio data.
        
        Returns:
            int: Sample rate in Hz
        """
        return self.target_sample_rate
    
    def get_chunk_size(self) -> int:
        """
        Get the size of each audio chunk in samples.
        
        Returns:
            int: Chunk size in samples
        """
        return self.chunk_size
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the audio provider.
        """
        self.logger.info("Cleaning up PyAudio resources")
        
        # Ensure recording is stopped
        if self.is_recording:
            self.stop()
        
        # Ensure thread is terminated
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_recording_event.set()
            self.is_recording = False
            self.recording_thread.join(timeout=1.0)
            if self.recording_thread.is_alive():
                self.logger.error("Failed to cleanly stop recording thread during cleanup")
        
        # Cleanup PyAudio resources
        if self.audio_interface:
            try:
                self.audio_interface.terminate()
            except Exception as e:
                self.logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self.audio_interface = None
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio input devices.
        
        Returns:
            List[Dict[str, Any]]: List of device information dictionaries
        """
        device_infos = []
        
        try:
            if not self.audio_interface:
                self.audio_interface = pyaudio.PyAudio()
            
            # Get default device info
            try:
                default_device_info = self.audio_interface.get_default_input_device_info()
                default_device_index = default_device_info['index']
            except IOError:
                default_device_index = -1
            
            # Iterate through all devices
            device_count = self.audio_interface.get_device_count()
            for i in range(device_count):
                try:
                    device_info = self.audio_interface.get_device_info_by_index(i)
                    max_input_channels = device_info.get('maxInputChannels', 0)
                    
                    # Only include devices with input capabilities
                    if max_input_channels > 0:
                        supported_rates = self._get_supported_sample_rates(i)
                        
                        # Create DeviceInfo object
                        is_default = (i == default_device_index)
                        device = DeviceInfo.from_pyaudio_device_info(
                            device_id=i,
                            device_info=device_info,
                            supported_rates=supported_rates,
                            is_default=is_default
                        )
                        
                        # Convert to dictionary for return
                        device_dict = {
                            'device_id': device.device_id,
                            'name': device.name,
                            'max_input_channels': device.max_input_channels,
                            'default_sample_rate': device.default_sample_rate,
                            'supported_sample_rates': device.supported_sample_rates,
                            'is_default': device.is_default
                        }
                        
                        device_infos.append(device_dict)
                except Exception as e:
                    self.logger.warning(f"Error getting info for device {i}: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error listing devices: {e}")
        
        return device_infos
    
    def is_running(self) -> bool:
        """
        Check if the audio provider is currently running.
        
        Returns:
            bool: True if the provider is recording
        """
        return self.is_recording
    
    def _get_supported_sample_rates(self, device_index: int) -> List[int]:
        """
        Test which standard sample rates are supported by the device.
        
        Args:
            device_index: The device index to check
            
        Returns:
            List[int]: List of supported sample rates
        """
        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        supported_rates = []
        
        try:
            device_info = self.audio_interface.get_device_info_by_index(device_index)
            max_channels = device_info.get('maxInputChannels', 1)
            
            for rate in standard_rates:
                try:
                    if self.audio_interface.is_format_supported(
                        rate,
                        input_device=device_index,
                        input_channels=max_channels if max_channels > 0 else 1,
                        input_format=self.audio_format
                    ):
                        supported_rates.append(rate)
                except Exception:
                    continue
        except Exception as e:
            self.logger.warning(f"Error checking supported rates for device {device_index}: {e}")
        
        return supported_rates
    
    def _get_best_sample_rate(self, device_index: int) -> int:
        """
        Determine the best sample rate for the device that's closest to the target.
        
        Args:
            device_index: The device index to check
            
        Returns:
            int: The best sample rate to use
        """
        try:
            device_info = self.audio_interface.get_device_info_by_index(device_index)
            supported_rates = self._get_supported_sample_rates(device_index)
            
            # If target rate is supported, use it
            if self.target_sample_rate in supported_rates:
                return self.target_sample_rate
            
            # Otherwise find the highest supported rate
            if supported_rates:
                return max(supported_rates)
            
            # Fall back to device default if no supported rates found
            return int(device_info.get('defaultSampleRate', 44100))
            
        except Exception as e:
            self.logger.warning(f"Error determining sample rate: {e}")
            return 44100  # Safe fallback
    
    def _resample_audio(self, audio_data: bytes) -> Tuple[bytes, np.ndarray]:
        """
        Resample audio data from device rate to target rate if needed.
        
        Args:
            audio_data: Raw audio data as bytes
            
        Returns:
            Tuple[bytes, np.ndarray]: The resampled audio as bytes and numpy array
        """
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # If device and target rates match, no resampling needed
        if self.device_sample_rate == self.target_sample_rate:
            return audio_data, audio_np
        
        # Calculate target number of samples
        target_num_samples = int(len(audio_np) * self.target_sample_rate / self.device_sample_rate)
        
        # Apply anti-aliasing filter if downsampling
        if self.target_sample_rate < self.device_sample_rate:
            # Calculate Nyquist frequency and normalized cutoff
            nyquist = self.device_sample_rate / 2.0
            cutoff = self.target_sample_rate / 2.0
            normal_cutoff = cutoff / nyquist
            
            # Design and apply Butterworth filter
            b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)
            filtered_audio = signal.filtfilt(b, a, audio_np)
            
            # Resample
            resampled = signal.resample_poly(filtered_audio, self.target_sample_rate, 
                                             self.device_sample_rate)
        else:
            # Upsampling (no need for anti-aliasing filter)
            resampled = signal.resample_poly(audio_np, self.target_sample_rate, 
                                            self.device_sample_rate)
        
        # Convert back to int16
        resampled_int16 = resampled.astype(np.int16)
        
        return resampled_int16.tobytes(), resampled_int16
    
    def _recording_worker(self):
        """
        Worker thread that continuously reads audio data and publishes events.
        """
        self.logger.info("Recording worker started")
        
        while self.is_recording and not self.stop_recording_event.is_set():
            try:
                # Read raw audio chunk
                raw_data = self.read_chunk()
                
                if not raw_data:
                    time.sleep(0.01)  # Avoid busy-waiting if read fails
                    continue
                
                # Resample if needed
                resampled_data, audio_array = self._resample_audio(raw_data)
                
                # Create timestamp for this chunk
                timestamp = time.time()
                
                # Increment sequence number
                self.sequence_number += 1
                
                # Create AudioChunk object
                audio_chunk = AudioChunk(
                    raw_data=resampled_data,
                    sample_rate=self.target_sample_rate,
                    channels=self.channels,
                    format='int16',
                    timestamp=timestamp,
                    sequence_number=self.sequence_number
                )
                
                # Publish event with the audio chunk
                event = AudioChunkCapturedEvent(
                    audio_chunk=audio_chunk,
                    source_id='microphone',
                    device_id=self.device_id,
                    provider_name='PyAudioInputProvider'
                )
                
                self.event_bus.publish(event)
                
            except Exception as e:
                self.logger.error(f"Error in recording worker: {e}")
                
                # Check if we should still be recording after the error
                if not self.is_recording or self.stop_recording_event.is_set():
                    break
                
                # Sleep briefly to avoid rapid failure loops
                time.sleep(0.1)
        
        self.logger.info("Recording worker stopped")
    
    def _publish_state_change(self, previous_state: RecordingState, 
                             current_state: RecordingState, 
                             error_message: Optional[str] = None):
        """
        Publish a state change event.
        
        Args:
            previous_state: The previous recording state
            current_state: The new recording state
            error_message: Optional error message if state is ERROR
        """
        self.current_state = current_state
        
        event = RecordingStateChangedEvent(
            previous_state=previous_state,
            current_state=current_state,
            device_id=self.device_id if self.device_id is not None else -1,
            error_message=error_message,
            metadata={
                'sample_rate': self.target_sample_rate,
                'chunk_size': self.chunk_size,
                'channels': self.channels,
                'device_sample_rate': self.device_sample_rate
            }
        )
        
        self.event_bus.publish(event)