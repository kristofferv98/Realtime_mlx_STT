import numpy as np
import threading
import time
import logging
import wave
import os
from typing import Dict, Any, List, Optional

from src.Core.Common.Interfaces.audio_provider import IAudioProvider

logger = logging.getLogger(__name__)

class FileAudioProvider(IAudioProvider):
    """
    Implementation of IAudioProvider that reads audio from a file.
    """
    
    def __init__(
            self,
            file_path: str,
            target_samplerate: int = 16000,
            chunk_size: int = 512,
            channels: int = 1,
            playback_speed: float = 1.0,
            loop: bool = False,
            debug_mode: bool = False,
        ):
        """
        Initialize the file audio provider.
        
        Args:
            file_path: Path to the audio file
            target_samplerate: Desired sample rate in Hz
            chunk_size: Desired chunk size in samples
            channels: Number of audio channels
            playback_speed: Playback speed multiplier
            loop: Whether to loop the file
            debug_mode: Enable debug logging
        """
        self.file_path = file_path
        self.target_samplerate = target_samplerate
        self.chunk_size = chunk_size
        self.channels = channels
        self.playback_speed = playback_speed
        self.loop = loop
        self.debug_mode = debug_mode
        
        self.file_sample_rate = None
        self.file_channels = None
        self.wave_file = None
        self._is_running = False
        self._lock = threading.RLock()
        self._position = 0
        self._audio_data = None
        
    def setup(self) -> bool:
        """
        Initialize the file reader.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        with self._lock:
            try:
                if not os.path.exists(self.file_path):
                    logger.error(f"File not found: {self.file_path}")
                    return False
                
                # Open the file and read its properties
                self.wave_file = wave.open(self.file_path, 'rb')
                self.file_sample_rate = self.wave_file.getframerate()
                self.file_channels = self.wave_file.getnchannels()
                
                # Read the entire file into memory
                frames = self.wave_file.readframes(self.wave_file.getnframes())
                self._audio_data = np.frombuffer(frames, dtype=np.int16)
                
                # Convert to mono if necessary
                if self.file_channels > 1 and self.channels == 1:
                    self._audio_data = self._audio_data.reshape(-1, self.file_channels)[:, 0]
                
                # Resample if necessary
                if self.file_sample_rate != self.target_samplerate:
                    target_length = int(len(self._audio_data) * self.target_samplerate / self.file_sample_rate)
                    from scipy import signal
                    self._audio_data = signal.resample(self._audio_data, target_length).astype(np.int16)
                
                # Reset position
                self._position = 0
                
                if self.debug_mode:
                    logger.debug(f"Loaded audio file {self.file_path}")
                    logger.debug(f"Original sample rate: {self.file_sample_rate}, channels: {self.file_channels}")
                    logger.debug(f"Target sample rate: {self.target_samplerate}, channels: {self.channels}")
                    logger.debug(f"Audio data length: {len(self._audio_data)} samples")
                
                return True
                
            except Exception as e:
                logger.error(f"Error initializing file audio provider: {e}")
                if self.wave_file:
                    self.wave_file.close()
                    self.wave_file = None
                return False

    def start(self) -> bool:
        """
        Start the audio playback.
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        with self._lock:
            if self._audio_data is None:
                success = self.setup()
                if not success:
                    return False
                
            self._is_running = True
            return True

    def stop(self) -> bool:
        """
        Stop the audio playback.
        
        Returns:
            bool: True if successfully stopped, False otherwise
        """
        with self._lock:
            self._is_running = False
            return True

    def read_chunk(self) -> bytes:
        """
        Read a chunk of audio data from the file.
        
        Returns:
            bytes: Raw audio data as bytes
        """
        if not self._is_running or self._audio_data is None:
            # Return empty audio chunk if not running
            return b'\x00' * (self.chunk_size * 2)  # 2 bytes per sample for int16
        
        with self._lock:
            # Calculate chunk size considering playback speed
            effective_chunk_size = int(self.chunk_size * self.playback_speed)
            
            # Get the chunk
            end_pos = self._position + effective_chunk_size
            
            if end_pos > len(self._audio_data):
                # End of file reached
                if self.loop:
                    # Handle looping
                    chunk = np.zeros(self.chunk_size, dtype=np.int16)
                    remaining = len(self._audio_data) - self._position
                    
                    if remaining > 0:
                        chunk[:remaining] = self._audio_data[self._position:]
                    
                    needed = self.chunk_size - remaining
                    if needed > 0:
                        chunk[remaining:] = self._audio_data[:needed]
                    
                    self._position = needed % len(self._audio_data)
                else:
                    # Return partial chunk and then zeros
                    chunk = np.zeros(self.chunk_size, dtype=np.int16)
                    remaining = len(self._audio_data) - self._position
                    
                    if remaining > 0:
                        chunk[:remaining] = self._audio_data[self._position:]
                    
                    self._position = len(self._audio_data)  # Stay at the end
            else:
                # Normal case: get the full chunk
                chunk = self._audio_data[self._position:end_pos]
                
                # Pad if needed to ensure consistent chunk size
                if len(chunk) < self.chunk_size:
                    padded = np.zeros(self.chunk_size, dtype=np.int16)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                
                self._position = end_pos
            
            # Add a small delay to simulate real-time behavior
            time.sleep(self.chunk_size / self.target_samplerate / self.playback_speed)
            
            return chunk.tobytes()

    def get_sample_rate(self) -> int:
        """
        Get the sample rate of the audio data.
        
        Returns:
            int: Sample rate in Hz
        """
        return self.target_samplerate

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
        with self._lock:
            self._is_running = False
            if self.wave_file:
                self.wave_file.close()
                self.wave_file = None

    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio input devices.
        For file provider, this returns a single "device" representing the file.
        
        Returns:
            List[Dict[str, Any]]: List of device information dictionaries
        """
        return [{
            'id': 0,
            'name': f"File: {os.path.basename(self.file_path)}",
            'max_input_channels': self.channels,
            'default_sample_rate': self.target_samplerate,
            'supported_sample_rates': [self.target_samplerate],
            'is_default': True,
            'file_path': self.file_path
        }]

    def is_running(self) -> bool:
        """
        Check if the audio provider is currently running.
        
        Returns:
            bool: True if the provider is running
        """
        return self._is_running