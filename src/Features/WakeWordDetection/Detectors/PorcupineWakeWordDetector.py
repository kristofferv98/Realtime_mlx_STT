"""
PorcupineWakeWordDetector implementation for wake word detection.

This module provides an implementation of the IWakeWordDetector interface
using the Picovoice Porcupine wake word detection engine.
"""

import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

try:
    import pvporcupine
except ImportError:
    pvporcupine = None

from src.Core.Common.Interfaces.wake_word_detector import IWakeWordDetector


class PorcupineWakeWordDetector(IWakeWordDetector):
    """
    Porcupine-based wake word detector.
    
    This implementation uses the Picovoice Porcupine SDK for wake word
    detection, which offers high-accuracy, low-resource wake word detection.
    """
    
    def __init__(self,
                 access_key: Optional[str] = None,
                 keywords: List[str] = None,
                 keyword_paths: List[str] = None,
                 sensitivities: List[float] = None):
        """
        Initialize the Porcupine wake word detector.
        
        Args:
            access_key: Porcupine access key (optional, can use env var)
            keywords: List of wake word names to use (built-in words)
            keyword_paths: List of paths to custom keyword model files (.ppn)
            sensitivities: List of sensitivities for each wake word
        """
        self.logger = LoggingModule.get_logger(__name__)
        self.porcupine = None
        self.access_key = access_key
        self.keywords = keywords or ["porcupine"]
        self.keyword_paths = keyword_paths
        self.sensitivity = 0.5
        self.sensitivities = sensitivities
        
        if self.sensitivities is None:
            self.sensitivities = [self.sensitivity] * len(self.keywords if not keyword_paths else keyword_paths)
            
        self.wake_words = []  # Will be populated during setup
        
        # Check if Porcupine is available
        if pvporcupine is None:
            self.logger.error("Porcupine package not found. Please install with 'pip install pvporcupine'")
    
    def setup(self, wake_words: List[str] = None, sensitivities: List[float] = None) -> bool:
        """
        Initialize the Porcupine detector with specified wake words.
        
        Args:
            wake_words: List of wake word names or paths to models
            sensitivities: List of sensitivity values for each wake word
            
        Returns:
            bool: True if setup was successful
        """
        # Handle both interface pattern and extended functionality
        if wake_words:
            self.keywords = wake_words
        if sensitivities:
            self.sensitivities = sensitivities
            
        # Ensure sensitivities match wake words
        if len(self.sensitivities) != len(self.keywords if not self.keyword_paths else self.keyword_paths):
            self.sensitivities = [self.sensitivity] * len(self.keywords if not self.keyword_paths else self.keyword_paths)
        
        try:
            if pvporcupine is None:
                raise ImportError("Porcupine package not found")
                
            # Get access key from config or environment
            access_key = self._get_access_key()
            if not access_key:
                self.logger.error("Porcupine access key is required")
                return False
                
            # Choose between built-in and custom keywords
            if self.keyword_paths:
                self.logger.info(f"Using custom keyword paths: {self.keyword_paths}")
                self.porcupine = pvporcupine.create(
                    access_key=access_key,
                    keyword_paths=self.keyword_paths,
                    sensitivities=self.sensitivities
                )
                self.wake_words = [os.path.basename(path).replace('.ppn', '') 
                                  for path in self.keyword_paths]
            else:
                self.logger.info(f"Using built-in keywords: {self.keywords}")
                self.porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=self.keywords,
                    sensitivities=self.sensitivities
                )
                self.wake_words = self.keywords
                
            self.logger.info(f"Initialized Porcupine with {len(self.wake_words)} wake words: {self.wake_words}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Porcupine: {e}")
            return False
    
    def process(self, audio_chunk: bytes) -> int:
        """
        Process an audio chunk to detect wake words.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            int: Index of detected wake word or -1 if none detected
        """
        if not self.porcupine:
            if not self.setup():
                return -1
        
        try:
            # Process audio with Porcupine
            pcm = self._prepare_audio(audio_chunk)
            if pcm is None:
                return -1
            
            return self.porcupine.process(pcm)
        except Exception as e:
            self.logger.error(f"Error in wake word detection: {e}")
            return -1
    
    def detect(self, audio_data: bytes, sample_rate: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Detect if the provided audio data contains a wake word.
        
        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Sample rate of the audio data (optional, uses default if None)
            
        Returns:
            Tuple[bool, Optional[str]]: (detected, wake_word_name)
        """
        index = self.process(audio_data)
        if index >= 0 and index < len(self.wake_words):
            return True, self.wake_words[index]
        return False, None
        
    def detect_with_confidence(self, audio_data: bytes, 
                               sample_rate: Optional[int] = None) -> Tuple[bool, float, Optional[str]]:
        """
        Detect if the provided audio data contains a wake word and return confidence.
        
        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Sample rate of the audio data (optional, uses default if None)
            
        Returns:
            Tuple[bool, float, Optional[str]]: (detected, confidence, wake_word_name)
        """
        index = self.process(audio_data)
        if index >= 0 and index < len(self.wake_words):
            # Since Porcupine doesn't return confidence, use sensitivity as confidence
            confidence = self.sensitivities[index]
            return True, confidence, self.wake_words[index]
        return False, 0.0, None
        
    def _prepare_audio(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Ensure audio is in correct format for Porcupine.
        
        Args:
            audio_data: Raw audio data as bytes
            
        Returns:
            np.ndarray: Audio data as int16 numpy array with correct length
        """
        try:
            # Convert to 16-bit PCM if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                return None
                
            # Ensure matching frame length
            required_length = self.porcupine.frame_length
            if len(audio_array) < required_length:
                self.logger.debug(f"Audio frame too short: {len(audio_array)} < {required_length}")
                # Pad with zeros if needed
                audio_array = np.pad(audio_array, (0, required_length - len(audio_array)))
            elif len(audio_array) > required_length:
                # Truncate to required length
                audio_array = audio_array[:required_length]
                
            return audio_array
        except Exception as e:
            self.logger.error(f"Error preparing audio: {e}")
            return None
            
    def _get_access_key(self) -> str:
        """
        Get Porcupine access key from instance or environment.
        
        Returns:
            str: Porcupine access key
            
        Raises:
            ValueError: If access key is not provided and not in environment
        """
        # Try to get access key from instance, then environment
        if self.access_key:
            return self.access_key
            
        # Try environment variable
        access_key = os.environ.get("PORCUPINE_ACCESS_KEY")
        if access_key:
            return access_key
            
        raise ValueError("Porcupine access key not provided and PORCUPINE_ACCESS_KEY environment variable not set")
    
    def get_sample_rate(self) -> int:
        """
        Get the expected sample rate for the detector.
        
        Returns:
            int: Expected sample rate in Hz
        """
        if not self.porcupine:
            if not self.setup():
                # Default Porcupine sample rate
                return 16000
        return self.porcupine.sample_rate
    
    def get_required_audio_format(self) -> Dict[str, Any]:
        """
        Get the required audio format for this detector.
        
        Returns:
            Dict with required format parameters like:
            {
                'sample_rate': 16000,  # Hz
                'frame_length': 512,   # samples
                'bit_depth': 16,       # bits
                'channels': 1          # mono
            }
        """
        if not self.porcupine:
            if not self.setup():
                # Default Porcupine format
                return {
                    'sample_rate': 16000,
                    'frame_length': 512,
                    'bit_depth': 16,
                    'channels': 1
                }
        
        return {
            'sample_rate': self.porcupine.sample_rate,
            'frame_length': self.porcupine.frame_length,
            'bit_depth': 16,
            'channels': 1
        }
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the wake word detector with the provided parameters.
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            bool: True if configuration was successful, False otherwise
        """
        try:
            # Extract configuration parameters
            if 'keywords' in config:
                self.keywords = config['keywords']
            
            if 'keyword_paths' in config:
                self.keyword_paths = config['keyword_paths']
            
            if 'sensitivities' in config:
                self.sensitivities = config['sensitivities']
            elif 'sensitivity' in config:
                self.sensitivity = config['sensitivity']
                self.sensitivities = [self.sensitivity] * len(self.keywords if not self.keyword_paths else self.keyword_paths)
            
            if 'access_key' in config:
                self.access_key = config['access_key']
            
            # Reinitialize with new configuration
            return self.setup()
        except Exception as e:
            self.logger.error(f"Error configuring Porcupine detector: {e}")
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration of the wake word detector.
        
        Returns:
            Dict[str, Any]: Dictionary of current configuration parameters
        """
        return {
            'detector_type': 'porcupine',
            'wake_words': self.wake_words,
            'keywords': self.keywords,
            'keyword_paths': self.keyword_paths,
            'sensitivities': self.sensitivities,
            'sample_rate': self.get_sample_rate(),
            'frame_length': self.porcupine.frame_length if self.porcupine else None
        }
    
    def get_name(self) -> str:
        """
        Get the name of the wake word detector implementation.
        
        Returns:
            str: Name of the detector
        """
        return "Porcupine Wake Word Detector"
    
    def reset(self) -> None:
        """
        Reset the internal state of the wake word detector.
        """
        # Porcupine is stateless, so nothing to reset
        pass
    
    def cleanup(self) -> None:
        """
        Release resources used by the wake word detector.
        """
        if self.porcupine:
            try:
                self.porcupine.delete()
            except Exception as e:
                self.logger.error(f"Error cleaning up Porcupine: {e}")
            finally:
                self.porcupine = None