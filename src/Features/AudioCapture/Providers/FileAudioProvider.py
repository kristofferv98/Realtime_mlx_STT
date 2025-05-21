"""
FileAudioProvider for playing audio from files.

This provider implements the IAudioProvider interface to stream audio
from file sources instead of microphones.
"""

import os
import threading
import time
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from scipy import signal

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

from src.Core.Common.Interfaces.audio_provider import IAudioProvider
from src.Core.Events.event_bus import IEventBus
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk
from src.Features.AudioCapture.Events.AudioChunkCapturedEvent import AudioChunkCapturedEvent
from src.Features.AudioCapture.Events.RecordingStateChangedEvent import (
    RecordingStateChangedEvent, RecordingState
)


class FileAudioProvider(IAudioProvider):
    """
    File-based implementation of IAudioProvider for audio file input.
    
    This provider reads audio data from files instead of microphones,
    which is useful for testing or processing pre-recorded audio.
    It supports common audio formats like WAV, MP3, FLAC, etc. via the
    soundfile library.
    """
    
    def __init__(self, 
                 event_bus: IEventBus,
                 file_path: Optional[str] = None,
                 target_sample_rate: int = 16000,
                 chunk_size: int = 512,
                 chunk_duration_ms: int = 30,
                 playback_speed: float = 1.0,
                 loop: bool = False,
                 debug_mode: bool = False):
        """
        Initialize the file audio provider.
        
        Args:
            event_bus: Event bus to publish events to
            file_path: Path to audio file to stream (can be set later)
            target_sample_rate: Sample rate in Hz to use for output
            chunk_size: Chunk size in samples (overridden by chunk_duration_ms if provided)
            chunk_duration_ms: Duration of each chunk in milliseconds
            playback_speed: Speed multiplier for playback (1.0=normal, 0.5=half speed, 2.0=double)
            loop: Whether to loop the file when it reaches the end
            debug_mode: Whether to print debug information
        """
        self.logger = LoggingModule.get_logger(__name__)
        self.event_bus = event_bus
        self.file_path = file_path
        self.target_sample_rate = target_sample_rate
        self.chunk_size = chunk_size
        self.chunk_duration_ms = chunk_duration_ms
        self.playback_speed = playback_speed
        self.loop = loop
        self.debug_mode = debug_mode
        
        # File data and state
        self.audio_data = None
        self.file_sample_rate = None
        self.file_channels = None
        self.position = 0
        self.device_id = 0  # Use 0 as a placeholder for file device
        
        # Compute the chunk size based on duration if provided
        if chunk_duration_ms > 0:
            self.chunk_size = int(target_sample_rate * (chunk_duration_ms / 1000.0))
            if self.debug_mode:
                self.logger.debug(f"Calculated chunk size: {self.chunk_size} samples "
                                 f"({chunk_duration_ms}ms at {target_sample_rate}Hz)")
        
        # Recording state tracking
        self.is_recording = False
        self.playback_thread = None
        self.stop_recording_event = threading.Event()
        self.sequence_number = 0
        self.current_state = RecordingState.INITIALIZED
    
    def setup(self) -> bool:
        """
        Initialize the file provider and load the audio file.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            self.logger.info("Setting up file audio provider")
            
            if not self.file_path:
                self.logger.warning("No file path provided. Setup successful but no file loaded.")
                self._publish_state_change(
                    RecordingState.INITIALIZED,
                    RecordingState.INITIALIZED
                )
                return True
            
            return self.load_file(self.file_path)
            
        except Exception as e:
            self.logger.error(f"Error setting up file provider: {e}")
            
            # Publish error state
            self._publish_state_change(
                RecordingState.INITIALIZED,
                RecordingState.ERROR,
                error_message=str(e)
            )
            
            return False
    
    def load_file(self, file_path: str) -> bool:
        """
        Load an audio file for streaming.
        
        Args:
            file_path: Path to the audio file to load
            
        Returns:
            bool: True if file was loaded successfully, False otherwise
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File does not exist: {file_path}")
            return False
        
        try:
            self.logger.info(f"Loading audio file: {file_path}")
            
            # Load audio file
            audio_data, file_sample_rate = sf.read(file_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                self.logger.info(f"Converting {audio_data.shape[1]} channels to mono")
                audio_data = audio_data.mean(axis=1)
            
            # Resample if needed
            if file_sample_rate != self.target_sample_rate:
                self.logger.info(f"Resampling audio from {file_sample_rate}Hz to {self.target_sample_rate}Hz")
                audio_data = self._resample_audio(audio_data, file_sample_rate, self.target_sample_rate)
            
            # Store file info
            self.file_path = file_path
            self.audio_data = audio_data
            self.file_sample_rate = file_sample_rate
            self.file_channels = 1  # We convert to mono above
            self.position = 0
            
            # Publish state change event
            self._publish_state_change(
                self.current_state,
                RecordingState.INITIALIZED,
                metadata={
                    'file_path': file_path,
                    'duration_seconds': len(audio_data) / self.target_sample_rate,
                    'original_sample_rate': file_sample_rate
                }
            )
            
            self.logger.info(f"File loaded successfully: {os.path.basename(file_path)} "
                            f"({len(audio_data) / self.target_sample_rate:.2f} seconds)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            
            # Publish error state
            self._publish_state_change(
                self.current_state,
                RecordingState.ERROR,
                error_message=f"Error loading file: {e}"
            )
            
            return False
    
    def start(self) -> bool:
        """
        Start playing audio from the file.
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        if self.is_recording:
            self.logger.warning("Playback already in progress")
            return False
        
        if self.audio_data is None:
            self.logger.error("No audio file loaded. Call setup() or load_file() first.")
            return False
        
        try:
            # Update state
            self._publish_state_change(
                self.current_state,
                RecordingState.STARTING
            )
            
            # Reset position and start playback thread
            self.position = 0
            self.sequence_number = 0
            self.stop_recording_event.clear()
            self.is_recording = True
            
            self.playback_thread = threading.Thread(
                target=self._playback_worker,
                daemon=True
            )
            self.playback_thread.start()
            
            # Update state
            self._publish_state_change(
                RecordingState.STARTING,
                RecordingState.RECORDING
            )
            
            self.logger.info(f"Started playback of file: {os.path.basename(self.file_path)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting playback: {e}")
            
            # Reset state
            self.is_recording = False
            
            # Update state
            self._publish_state_change(
                self.current_state,
                RecordingState.ERROR,
                error_message=str(e)
            )
            
            return False
    
    def stop(self) -> bool:
        """
        Stop playing audio from the file.
        
        Returns:
            bool: True if successfully stopped, False otherwise
        """
        if not self.is_recording:
            self.logger.warning("No playback in progress")
            return False
        
        try:
            # Update state
            self._publish_state_change(
                self.current_state,
                RecordingState.STOPPING
            )
            
            # Signal playback thread to stop and wait for it
            self.stop_recording_event.set()
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=2.0)
                if self.playback_thread.is_alive():
                    self.logger.warning("Playback thread did not stop in time")
            
            self.is_recording = False
            
            # Update state
            self._publish_state_change(
                RecordingState.STOPPING,
                RecordingState.STOPPED
            )
            
            self.logger.info("Stopped playback")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping playback: {e}")
            
            # Force stop playback state regardless of error
            self.is_recording = False
            
            # Update state
            self._publish_state_change(
                self.current_state,
                RecordingState.ERROR,
                error_message=str(e)
            )
            
            return False
    
    def read_chunk(self) -> bytes:
        """
        Read a single chunk of audio data from the file.
        
        Returns:
            bytes: Raw audio data as bytes
        """
        if not self.is_recording or self.audio_data is None:
            self.logger.warning("Cannot read chunk: no active playback or no file loaded")
            return b''
        
        try:
            # Calculate end position
            end_position = min(self.position + self.chunk_size, len(self.audio_data))
            
            # Get the chunk of audio data
            chunk = self.audio_data[self.position:end_position]
            
            # Move position
            self.position = end_position
            
            # Check for end of file
            if self.position >= len(self.audio_data):
                if self.loop:
                    self.logger.info("End of file reached, looping")
                    self.position = 0
                else:
                    self.logger.info("End of file reached")
                    # Signal to stop if not looping
                    self.stop_recording_event.set()
            
            # Convert to int16 bytes
            return (chunk * 32767).astype(np.int16).tobytes()
            
        except Exception as e:
            self.logger.error(f"Error reading chunk: {e}")
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
        self.logger.info("Cleaning up file audio provider")
        
        if self.is_recording:
            self.stop()
        
        # Clear file data
        self.audio_data = None
        self.file_sample_rate = None
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio file 'devices' (just shows the current file).
        
        Returns:
            List[Dict[str, Any]]: List with a single device representing the file
        """
        if not self.file_path:
            return []
        
        return [{
            'device_id': 0,
            'name': f"File: {os.path.basename(self.file_path)}",
            'max_input_channels': 1,
            'default_sample_rate': self.target_sample_rate,
            'supported_sample_rates': [self.target_sample_rate],
            'is_default': True,
            'file_path': self.file_path,
            'duration_seconds': 0 if self.audio_data is None else len(self.audio_data) / self.target_sample_rate
        }]
    
    def is_running(self) -> bool:
        """
        Check if the audio provider is currently running.
        
        Returns:
            bool: True if the provider is playing audio
        """
        return self.is_recording
    
    def set_position(self, position_seconds: float) -> bool:
        """
        Set the current playback position.
        
        Args:
            position_seconds: Position in seconds
            
        Returns:
            bool: True if position was set successfully
        """
        if self.audio_data is None:
            self.logger.error("Cannot set position: no file loaded")
            return False
        
        try:
            # Convert seconds to samples
            position_samples = int(position_seconds * self.target_sample_rate)
            
            # Clamp to valid range
            position_samples = max(0, min(position_samples, len(self.audio_data) - 1))
            
            # Set position
            self.position = position_samples
            
            self.logger.info(f"Set position to {position_seconds:.2f} seconds "
                            f"({position_samples} samples)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting position: {e}")
            return False
    
    def get_duration(self) -> float:
        """
        Get the duration of the loaded audio file.
        
        Returns:
            float: Duration in seconds, or 0 if no file is loaded
        """
        if self.audio_data is None:
            return 0.0
        
        return len(self.audio_data) / self.target_sample_rate
    
    def _resample_audio(self, audio_data: np.ndarray, 
                       original_rate: int, 
                       target_rate: int) -> np.ndarray:
        """
        Resample audio data to the target sample rate.
        
        Args:
            audio_data: Audio data as numpy array
            original_rate: Original sample rate in Hz
            target_rate: Target sample rate in Hz
            
        Returns:
            np.ndarray: Resampled audio data
        """
        # Apply anti-aliasing filter if downsampling
        if target_rate < original_rate:
            # Calculate Nyquist frequency and normalized cutoff
            nyquist = original_rate / 2.0
            cutoff = target_rate / 2.0
            normal_cutoff = cutoff / nyquist
            
            # Design and apply Butterworth filter
            b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # Resample
            resampled = signal.resample_poly(filtered_audio, target_rate, original_rate)
        else:
            # Upsampling (no need for anti-aliasing filter)
            resampled = signal.resample_poly(audio_data, target_rate, original_rate)
        
        return resampled
    
    def _playback_worker(self):
        """
        Worker thread that plays back audio data from the file.
        """
        self.logger.info("Playback worker started")
        
        # Calculate sleep time between chunks based on playback speed
        chunk_duration_sec = self.chunk_size / self.target_sample_rate
        sleep_time = chunk_duration_sec / self.playback_speed
        
        if self.debug_mode:
            self.logger.debug(f"Chunk duration: {chunk_duration_sec:.3f}s, "
                             f"Sleep time: {sleep_time:.3f}s (speed: {self.playback_speed}x)")
        
        while self.is_recording and not self.stop_recording_event.is_set():
            try:
                # Read audio chunk
                raw_data = self.read_chunk()
                
                if not raw_data:
                    # End of file and not looping, or error
                    if self.position >= len(self.audio_data) and not self.loop:
                        self.logger.info("End of file reached and not looping, stopping playback")
                        self.is_recording = False
                        
                        # Publish state change
                        self._publish_state_change(
                            RecordingState.RECORDING,
                            RecordingState.STOPPED
                        )
                        
                        break
                    
                    time.sleep(0.01)  # Avoid busy waiting
                    continue
                
                # Create timestamp
                timestamp = time.time()
                
                # Increment sequence number
                self.sequence_number += 1
                
                # Create AudioChunk object
                audio_chunk = AudioChunk(
                    raw_data=raw_data,
                    sample_rate=self.target_sample_rate,
                    channels=1,
                    format='int16',
                    timestamp=timestamp,
                    sequence_number=self.sequence_number
                )
                
                # Publish event with the audio chunk
                event = AudioChunkCapturedEvent(
                    audio_chunk=audio_chunk,
                    source_id='file',
                    device_id=self.device_id,
                    provider_name='FileAudioProvider'
                )
                
                self.event_bus.publish(event)
                
                # Sleep to simulate real-time playback
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in playback worker: {e}")
                
                # Check if we should still be playing after the error
                if not self.is_recording or self.stop_recording_event.is_set():
                    break
                
                # Sleep briefly to avoid rapid failure loops
                time.sleep(0.1)
        
        self.logger.info("Playback worker stopped")
    
    def _publish_state_change(self, previous_state: RecordingState, 
                             current_state: RecordingState, 
                             error_message: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """
        Publish a state change event.
        
        Args:
            previous_state: The previous recording state
            current_state: The new recording state
            error_message: Optional error message if state is ERROR
            metadata: Additional metadata to include in the event
        """
        self.current_state = current_state
        
        # Merge provided metadata with default metadata
        event_metadata = {
            'sample_rate': self.target_sample_rate,
            'chunk_size': self.chunk_size,
            'playback_speed': self.playback_speed,
            'loop': self.loop
        }
        
        if metadata:
            event_metadata.update(metadata)
        
        event = RecordingStateChangedEvent(
            previous_state=previous_state,
            current_state=current_state,
            device_id=self.device_id,
            error_message=error_message,
            metadata=event_metadata
        )
        
        self.event_bus.publish(event)