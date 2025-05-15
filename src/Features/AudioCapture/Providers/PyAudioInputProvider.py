import pyaudio
import numpy as np
import threading
import time
import logging
from typing import Dict, Any, List, Optional
from colorama import init, Fore, Style
from scipy.signal import butter, filtfilt, resample_poly

from src.Core.Common.Interfaces.audio_provider import IAudioProvider

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