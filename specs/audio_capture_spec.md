# Audio Capture Feature Slice Implementation Plan

## Overview

This document outlines the implementation plan for refactoring the audio capture functionality of RealtimeSTT_mlx into a vertical slice architecture. The Audio Capture feature slice will be responsible for capturing audio from various sources (primarily microphone input), preprocessing it, and making it available to other components through events.

## Current Implementation Analysis

The current implementation in `RealtimeSTT/audio_input.py` provides:

1. **Device Management**: Lists and selects audio input devices
2. **Stream Initialization**: Creates and configures PyAudio streams
3. **Audio Reading**: Reads chunks of audio data from the selected device
4. **Preprocessing**: Resamples audio data to the required sample rate
5. **Filtering**: Applies low-pass filtering for downsampling
6. **Cleanup**: Properly releases audio resources

The current implementation is tightly coupled with:
- PyAudio for audio capture
- Specific sample rate and chunk size parameters
- Direct usage in `audio_recorder.py` through function calls

## Feature Slice Goals

The Audio Capture feature slice will:

1. **Decouple** audio capture from other system components
2. **Encapsulate** device selection and management
3. **Standardize** audio data format for consumption by other components
4. **Support** different audio input sources (microphone, file, etc.)
5. **Emit** events when audio data is available, allowing other components to react
6. **Provide** commands for controlling audio capture

## Implementation Details

### 1. Directory Structure

```
src/
├── Features/
│   ├── AudioCapture/
│   │   ├── Commands/
│   │   │   ├── StartRecordingCommand.py
│   │   │   ├── StopRecordingCommand.py
│   │   │   ├── SelectDeviceCommand.py
│   │   │   └── ListDevicesCommand.py
│   │   ├── Events/
│   │   │   ├── AudioChunkCapturedEvent.py
│   │   │   └── RecordingStateChangedEvent.py
│   │   ├── Handlers/
│   │   │   └── AudioCommandHandler.py
│   │   ├── Models/
│   │   │   ├── AudioChunk.py
│   │   │   └── DeviceInfo.py
│   │   ├── Providers/
│   │   │   ├── PyAudioInputProvider.py
│   │   │   └── FileAudioProvider.py
│   │   └── AudioCaptureModule.py
```

### 2. Components

#### 2.1 Models

**AudioChunk.py**
```python
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    
    # Raw audio bytes (typically PCM data)
    raw_data: bytes
    
    # Sample rate in Hz
    sample_rate: int
    
    # Number of audio channels (1 for mono, 2 for stereo)
    channels: int
    
    # Format of the audio data (e.g., int16, float32)
    format: str
    
    # Optional numpy array representation, lazily created when needed
    _numpy_data: Optional[np.ndarray] = None
    
    # Timestamp when this chunk was captured
    timestamp: float = 0.0
    
    # Sequence number for ordering chunks
    sequence_number: int = 0
    
    @property
    def numpy_data(self) -> np.ndarray:
        """
        Get the audio data as a numpy array.
        Lazily converts from bytes if needed.
        
        Returns:
            np.ndarray: Audio data as a numpy array
        """
        if self._numpy_data is None:
            if self.format == 'int16':
                self._numpy_data = np.frombuffer(self.raw_data, dtype=np.int16)
            elif self.format == 'float32':
                self._numpy_data = np.frombuffer(self.raw_data, dtype=np.float32)
            # Add more formats as needed
        
        return self._numpy_data
    
    def to_float32(self) -> np.ndarray:
        """
        Convert the audio data to float32 format normalized to [-1.0, 1.0].
        
        Returns:
            np.ndarray: Normalized float32 audio data
        """
        if self.format == 'int16':
            return self.numpy_data.astype(np.float32) / 32768.0
        elif self.format == 'float32':
            return self.numpy_data
        
        # Default conversion
        return self.numpy_data.astype(np.float32)
```

**DeviceInfo.py**
```python
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DeviceInfo:
    """Information about an audio input device."""
    
    # Device identifier
    id: int
    
    # Device name
    name: str
    
    # Maximum input channels
    max_input_channels: int
    
    # Default sample rate
    default_sample_rate: int
    
    # Supported sample rates
    supported_sample_rates: List[int]
    
    # Is this the default device?
    is_default: bool = False
    
    # Additional device-specific information
    extra_info: Optional[dict] = None
```

#### 2.2 Commands

**StartRecordingCommand.py**
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from src.Core.Commands.command import Command


@dataclass
class StartRecordingCommand(Command):
    """Command to start audio recording."""
    
    # Device ID to use (None for default)
    device_id: Optional[int] = None
    
    # Desired sample rate in Hz
    sample_rate: int = 16000
    
    # Desired chunk size in samples
    chunk_size: int = 512
    
    # Number of channels (1 for mono, 2 for stereo)
    channels: int = 1
    
    # Format of the audio data (e.g., 'int16', 'float32')
    format: str = 'int16'
    
    # Whether to resample to the target_samplerate if device doesn't match
    resample_if_needed: bool = True
    
    # Enable debug logging
    debug_mode: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = "StartRecordingCommand"
```

**StopRecordingCommand.py**
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
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
```

**SelectDeviceCommand.py**
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
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
```

**ListDevicesCommand.py**
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
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
```

#### 2.3 Events

**AudioChunkCapturedEvent.py**
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from src.Core.Events.event import Event
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk


@dataclass
class AudioChunkCapturedEvent(Event):
    """Event emitted when an audio chunk is captured."""
    
    # The captured audio chunk
    audio_chunk: AudioChunk
    
    # Whether this is the last chunk in a sequence
    is_last: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = "AudioChunkCapturedEvent"
```

**RecordingStateChangedEvent.py**
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from src.Core.Events.event import Event


@dataclass
class RecordingStateChangedEvent(Event):
    """Event emitted when the recording state changes."""
    
    # The new recording state
    state: Literal["started", "stopped", "paused", "error", "device_changed"]
    
    # Optional message with additional information
    message: Optional[str] = None
    
    # Device ID if relevant
    device_id: Optional[int] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.name is None:
            self.name = "RecordingStateChangedEvent"
```

#### 2.4 Providers

**PyAudioInputProvider.py**
```python
import pyaudio
import numpy as np
import threading
import time
import logging
from typing import Dict, Any, List, Optional
from colorama import init, Fore, Style
from scipy.signal import butter, filtfilt, resample_poly

from src.Core.Common.Interfaces.audio_provider import IAudioProvider
from src.Features.AudioCapture.Models.DeviceInfo import DeviceInfo

logger = logging.getLogger(__name__)

class PyAudioInputProvider(IAudioProvider):
    """
    Implementation of IAudioProvider using PyAudio for microphone input.
    This adapts the original AudioInput class to the new architecture.
    """
    
    def __init__(
            self,
            input_device_index: Optional[int] = None,
            debug_mode: bool = False,
            target_samplerate: int = 16000,
            chunk_size: int = 512,
            audio_format: int = pyaudio.paInt16,
            channels: int = 1,
            resample_to_target: bool = True,
        ):
        """
        Initialize the PyAudio input provider.
        
        Args:
            input_device_index: Optional device index, None for default
            debug_mode: Enable debug logging
            target_samplerate: Desired sample rate in Hz
            chunk_size: Desired chunk size in samples
            audio_format: PyAudio format constant
            channels: Number of audio channels
            resample_to_target: Whether to resample if device doesn't match target
        """
        self.input_device_index = input_device_index
        self.debug_mode = debug_mode
        self.audio_interface = None
        self.stream = None
        self.device_sample_rate = None
        self.target_samplerate = target_samplerate
        self.chunk_size = chunk_size
        self.audio_format = audio_format
        self.channels = channels
        self.resample_to_target = resample_to_target
        self._is_running = False
        self._lock = threading.RLock()
        
    def get_supported_sample_rates(self, device_index):
        """Test which standard sample rates are supported by the specified device."""
        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        supported_rates = []

        device_info = self.audio_interface.get_device_info_by_index(device_index)
        max_channels = device_info.get('maxInputChannels')

        for rate in standard_rates:
            try:
                if self.audio_interface.is_format_supported(
                    rate,
                    input_device=device_index,
                    input_channels=max_channels,
                    input_format=self.audio_format,
                ):
                    supported_rates.append(rate)
            except:
                continue
        return supported_rates

    def _get_best_sample_rate(self, actual_device_index, desired_rate):
        """Determines the best available sample rate for the device."""
        try:
            device_info = self.audio_interface.get_device_info_by_index(actual_device_index)
            supported_rates = self.get_supported_sample_rates(actual_device_index)

            if desired_rate in supported_rates:
                return desired_rate

            return max(supported_rates)
            
            # The following code is commented out as in the original implementation
            # lower_rates = [r for r in supported_rates if r <= desired_rate]
            # if lower_rates:
            #     return max(lower_rates)
            # higher_rates = [r for r in supported_rates if r > desired_rate]
            # if higher_rates:
            #     return min(higher_rates)

            return int(device_info.get('defaultSampleRate', 44100))

        except Exception as e:
            logging.warning(f"Error determining sample rate: {e}")
            return 44100  # Safe fallback

    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List all available audio input devices with supported sample rates.
        
        Returns:
            List[Dict[str, Any]]: List of device information dictionaries
        """
        devices = []
        try:
            init()  # Initialize colorama
            temp_audio_interface = pyaudio.PyAudio() if self.audio_interface is None else self.audio_interface
            device_count = temp_audio_interface.get_device_count()

            if self.debug_mode:
                print(f"Available audio input devices:")
            
            for i in range(device_count):
                device_info = temp_audio_interface.get_device_info_by_index(i)
                device_name = device_info.get('name')
                max_input_channels = device_info.get('maxInputChannels', 0)

                if max_input_channels > 0:  # Only consider devices with input capabilities
                    supported_rates = self.get_supported_sample_rates(i) if self.audio_interface else []
                    
                    if self.debug_mode:
                        print(f"{Fore.LIGHTGREEN_EX}Device {Style.RESET_ALL}{i}{Fore.LIGHTGREEN_EX}: {device_name}{Style.RESET_ALL}")
                        rates_formatted = ", ".join([f"{Fore.CYAN}{rate}{Style.RESET_ALL}" for rate in supported_rates])
                        print(f"  {Fore.YELLOW}Supported sample rates: {rates_formatted}{Style.RESET_ALL}")
                    
                    # Create standardized device info dictionary
                    device = {
                        'id': i,
                        'name': device_name,
                        'max_input_channels': max_input_channels,
                        'default_sample_rate': int(device_info.get('defaultSampleRate', 44100)),
                        'supported_sample_rates': supported_rates,
                        'is_default': i == temp_audio_interface.get_default_input_device_info()['index'],
                        'original_info': device_info
                    }
                    devices.append(device)

            # Clean up if we created a temporary interface
            if self.audio_interface is None and temp_audio_interface:
                temp_audio_interface.terminate()
                
            return devices

        except Exception as e:
            logger.error(f"Error listing devices: {e}")
            return []

    def setup(self) -> bool:
        """
        Initialize audio interface and open stream.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        with self._lock:
            try:
                self.audio_interface = pyaudio.PyAudio()

                if self.debug_mode:
                    logger.debug(f"Input device index: {self.input_device_index}")
                
                actual_device_index = (self.input_device_index if self.input_device_index is not None 
                                    else self.audio_interface.get_default_input_device_info()['index'])
                
                if self.debug_mode:
                    logger.debug(f"Actual selected device index: {actual_device_index}")
                
                self.input_device_index = actual_device_index
                self.device_sample_rate = self._get_best_sample_rate(actual_device_index, self.target_samplerate)

                if self.debug_mode:
                    logger.debug(f"Setting up audio on device {self.input_device_index} with sample rate {self.device_sample_rate}")

                try:
                    self.stream = self.audio_interface.open(
                        format=self.audio_format,
                        channels=self.channels,
                        rate=self.device_sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size,
                        input_device_index=self.input_device_index,
                    )
                    if self.debug_mode:
                        logger.debug(f"Audio recording initialized successfully at {self.device_sample_rate} Hz")
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize audio stream at {self.device_sample_rate} Hz: {e}")
                    return False

            except Exception as e:
                logger.error(f"Error initializing audio recording: {e}")
                if self.audio_interface:
                    self.audio_interface.terminate()
                return False

    def start(self) -> bool:
        """
        Start the audio capture process.
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        with self._lock:
            if self.stream is None:
                return self.setup()
            self._is_running = True
            return True

    def stop(self) -> bool:
        """
        Stop the audio capture process.
        
        Returns:
            bool: True if successfully stopped, False otherwise
        """
        with self._lock:
            self._is_running = False
            return True

    def read_chunk(self) -> bytes:
        """
        Read a single chunk of audio data.
        
        Returns:
            bytes: Raw audio data as bytes
        """
        if not self._is_running or self.stream is None:
            # Return empty audio chunk if not running
            return b'\x00' * (self.chunk_size * 2)  # 2 bytes per sample for int16
        
        # Read from stream
        raw_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
        
        # Resample if necessary
        if self.resample_to_target and self.device_sample_rate != self.target_samplerate:
            audio_np = np.frombuffer(raw_data, dtype=np.int16)
            
            # Apply filtering for downsampling
            if self.target_samplerate < self.device_sample_rate:
                audio_np = self.lowpass_filter(
                    audio_np, 
                    self.target_samplerate / 2, 
                    self.device_sample_rate
                )
            
            # Resample to target rate
            resampled = self.resample_audio(
                audio_np,
                self.target_samplerate,
                self.device_sample_rate
            )
            
            # Convert back to bytes
            return resampled.astype(np.int16).tobytes()
            
        return raw_data

    def get_sample_rate(self) -> int:
        """
        Get the sample rate of the audio data.
        
        Returns:
            int: Sample rate in Hz
        """
        return self.target_samplerate if self.resample_to_target else self.device_sample_rate

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
            try:
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                if self.audio_interface:
                    self.audio_interface.terminate()
                    self.audio_interface = None
                self._is_running = False
            except Exception as e:
                logger.error(f"Error cleaning up audio resources: {e}")

    def is_running(self) -> bool:
        """
        Check if the audio provider is currently running.
        
        Returns:
            bool: True if the provider is running
        """
        return self._is_running

    def lowpass_filter(self, signal, cutoff_freq, sample_rate):
        """
        Apply a low-pass Butterworth filter to prevent aliasing in the signal.

        Args:
            signal (np.ndarray): Input audio signal to filter
            cutoff_freq (float): Cutoff frequency in Hz
            sample_rate (float): Sampling rate of the input signal in Hz

        Returns:
            np.ndarray: Filtered audio signal
        """
        nyquist_rate = sample_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist_rate
        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def resample_audio(self, pcm_data, target_sample_rate, original_sample_rate):
        """
        Filter and resample audio data to a target sample rate.

        Args:
            pcm_data (np.ndarray): Input audio data
            target_sample_rate (int): Desired output sample rate in Hz
            original_sample_rate (int): Original sample rate of input in Hz

        Returns:
            np.ndarray: Resampled audio data
        """
        if target_sample_rate < original_sample_rate:
            # Downsampling with low-pass filter
            pcm_filtered = self.lowpass_filter(pcm_data, target_sample_rate / 2, original_sample_rate)
            resampled = resample_poly(pcm_filtered, target_sample_rate, original_sample_rate)
        else:
            # Upsampling without low-pass filter
            resampled = resample_poly(pcm_data, target_sample_rate, original_sample_rate)
        return resampled
```

**FileAudioProvider.py**
```python
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
```

#### 2.5 Handlers

**AudioCommandHandler.py**
```python
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Type, cast

from src.Core.Commands.command import Command
from src.Core.Commands.command_dispatcher import CommandDispatcher
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
```

#### 2.6 Module

**AudioCaptureModule.py**
```python
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
```

## Implementation Strategy

### Phase 1: Base Structure

1. Create all the directories and empty files for the feature slice
2. Implement the models first (AudioChunk, DeviceInfo)
3. Implement the commands (StartRecordingCommand, StopRecordingCommand, etc.)
4. Implement the events (AudioChunkCapturedEvent, RecordingStateChangedEvent)

### Phase 2: Providers

1. Implement the PyAudioInputProvider adapting the original AudioInput class
2. Implement the FileAudioProvider for testing and file input

### Phase 3: Command Handler

1. Implement the AudioCommandHandler
2. Create the AudioCaptureModule that registers everything

### Phase 4: Testing and Integration

1. Create unit tests for the components
2. Create an integration test that uses the feature slice
3. Connect this feature slice to existing audio recorder functionality as a first step

## Benefits

1. **Testability**: Each component can be tested in isolation
2. **Flexibility**: Easy to add new audio providers or change existing ones
3. **Maintainability**: All audio-related code is in one feature slice
4. **Extensibility**: New audio features can be added without modifying existing code
5. **Decoupling**: Other components interact through events and commands, not direct calls

## Considerations and Challenges

1. **Performance**: Ensure that audio processing remains efficient despite the additional abstraction
2. **Thread Safety**: Careful coordination between threads is needed for audio capture
3. **Resource Management**: Proper cleanup of audio resources to prevent leaks
4. **Compatibility**: Ensure compatibility with the original audio recorder during transition

## Next Steps After Implementation

1. Connect this feature slice to the Voice Activity Detection feature
2. Update the main application facade to use the new architecture
3. Begin refactoring the next feature slice (Voice Activity Detection)