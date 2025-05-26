"""
Utility functions for the high-level API.
"""

import os
import logging
from typing import List, Optional
import pyaudio

from .types import AudioDevice


def setup_minimal_logging(level: int = logging.ERROR):
    """
    Configure minimal logging for non-verbose mode.
    
    Args:
        level: Logging level to set
    """
    # Set root logger
    logging.getLogger().setLevel(level)
    
    # Suppress specific noisy loggers
    for logger_name in ['src', 'realtimestt', 'transformers', 'torch']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Disable progress bars
    os.environ['TQDM_DISABLE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')


def list_audio_devices() -> List[AudioDevice]:
    """
    List all available audio input devices.
    
    Returns:
        List of AudioDevice objects
    """
    devices = []
    p = pyaudio.PyAudio()
    
    try:
        default_input = p.get_default_input_device_info()
        default_index = default_input['index']
        
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                
                # Only include input devices
                if info['maxInputChannels'] > 0:
                    device = AudioDevice(
                        index=i,
                        name=info['name'],
                        channels=info['maxInputChannels'],
                        sample_rate=int(info['defaultSampleRate']),
                        is_default=(i == default_index)
                    )
                    devices.append(device)
            except Exception:
                # Skip devices that can't be accessed
                pass
    finally:
        p.terminate()
    
    return devices


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "5.2s", "1m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def validate_language_code(language: str) -> bool:
    """
    Validate if a language code is supported.
    
    Args:
        language: Language code to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Common language codes supported by Whisper
    supported_languages = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
        'ar', 'hi', 'tr', 'pl', 'nl', 'sv', 'fi', 'da', 'no', 'el',
        'he', 'id', 'ms', 'th', 'vi', 'cs', 'ro', 'hu', 'uk', 'bg'
    }
    
    return language.lower() in supported_languages


def get_audio_format_info(sample_rate: int = 16000, channels: int = 1, 
                         sample_width: int = 2) -> dict:
    """
    Get audio format information dictionary.
    
    Args:
        sample_rate: Sample rate in Hz
        channels: Number of channels
        sample_width: Sample width in bytes
        
    Returns:
        Dictionary with audio format information
    """
    return {
        'sample_rate': sample_rate,
        'channels': channels,
        'sample_width': sample_width,
        'format': 'PCM',
        'bits_per_sample': sample_width * 8,
        'bytes_per_second': sample_rate * channels * sample_width
    }