"""
WebRtcVadDetector implementation of IVoiceActivityDetector.

This module provides an implementation of voice activity detection using the 
WebRTC VAD algorithm, which is fast and lightweight.
"""

import struct
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import webrtcvad

from src.Core.Common.Interfaces.voice_activity_detector import IVoiceActivityDetector
from src.Infrastructure.Logging import LoggingModule


class WebRtcVadDetector(IVoiceActivityDetector):
    """
    WebRTC-based voice activity detector.
    
    This implementation uses the WebRTC VAD algorithm, which is optimized for
    real-time applications. It is fast and has low computational requirements,
    making it suitable for use on resource-constrained devices.
    
    The detector supports different aggressiveness modes (0-3) to balance
    between false positives and false negatives.
    """
    
    def __init__(self, 
                 aggressiveness: int = 3,
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30,
                 history_size: int = 5,
                 speech_threshold: float = 0.6):
        """
        Initialize the WebRTC VAD detector.
        
        Args:
            aggressiveness: VAD aggressiveness mode (0-3, where 3 is most aggressive)
            sample_rate: Audio sample rate in Hz (must be 8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in milliseconds (must be 10, 20, or 30)
            history_size: Number of frames to keep in history for smoothing
            speech_threshold: Fraction of frames needed to classify as speech
        """
        self.logger = LoggingModule.get_logger(__name__)
        
        # Validate sample rate
        valid_sample_rates = [8000, 16000, 32000, 48000]
        if sample_rate not in valid_sample_rates:
            raise ValueError(f"Sample rate must be one of {valid_sample_rates}, got {sample_rate}")
        
        # Validate frame duration
        valid_durations = [10, 20, 30]
        if frame_duration_ms not in valid_durations:
            raise ValueError(f"Frame duration must be one of {valid_durations}, got {frame_duration_ms}")
        
        # Validate aggressiveness
        if not 0 <= aggressiveness <= 3:
            raise ValueError(f"Aggressiveness must be between 0 and 3, got {aggressiveness}")
        
        self.vad = None
        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.history_size = history_size
        self.speech_threshold = speech_threshold
        self.history = []
    
    def setup(self) -> bool:
        """
        Initialize the WebRTC VAD model.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            self.vad = webrtcvad.Vad()
            self.vad.set_mode(self.aggressiveness)
            self.history = []
            self.logger.info(f"Initialized WebRTC VAD with aggressiveness {self.aggressiveness}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize WebRTC VAD: {e}")
            return False
    
    def detect(self, audio_data: bytes, sample_rate: Optional[int] = None) -> bool:
        """
        Detect if the provided audio data contains speech.
        
        Args:
            audio_data: Raw audio data as bytes (must be 16-bit PCM)
            sample_rate: Sample rate of the audio data (optional, uses default if None)
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        is_speech, _ = self.detect_with_confidence(audio_data, sample_rate)
        return is_speech
    
    def detect_with_confidence(self, audio_data: bytes, 
                              sample_rate: Optional[int] = None) -> Tuple[bool, float]:
        """
        Detect if the provided audio data contains speech and return confidence level.
        
        Args:
            audio_data: Raw audio data as bytes (must be 16-bit PCM)
            sample_rate: Sample rate of the audio data (optional, uses default if None)
            
        Returns:
            Tuple[bool, float]: (speech_detected, confidence_score)
        """
        if self.vad is None:
            if not self.setup():
                return False, 0.0
        
        # Use provided sample rate or default
        rate = sample_rate if sample_rate is not None else self.sample_rate
        
        # Validate sample rate
        valid_sample_rates = [8000, 16000, 32000, 48000]
        if rate not in valid_sample_rates:
            self.logger.warning(f"Invalid sample rate {rate}, using {self.sample_rate}")
            rate = self.sample_rate
        
        # Ensure audio data is correctly sized for WebRTC VAD
        expected_frame_size = int(rate * self.frame_duration_ms / 1000) * 2  # 16-bit = 2 bytes per sample
        
        # If audio chunk doesn't match expected size, handle appropriately
        if len(audio_data) != expected_frame_size:
            self.logger.debug(f"Audio chunk size {len(audio_data)} doesn't match expected size {expected_frame_size}")
            
            # For now, just use what we can if the data is larger
            if len(audio_data) > expected_frame_size:
                audio_data = audio_data[:expected_frame_size]
            else:
                # If data is too small, pad with zeros or skip
                return False, 0.0
        
        try:
            # Detect speech in the current frame
            is_speech = self.vad.is_speech(audio_data, rate)
            
            # Update history
            self.history.append(is_speech)
            if len(self.history) > self.history_size:
                self.history.pop(0)
            
            # Calculate confidence based on history
            if not self.history:
                return False, 0.0
            
            confidence = sum(self.history) / len(self.history)
            is_speech_final = confidence >= self.speech_threshold
            
            return is_speech_final, confidence
            
        except Exception as e:
            self.logger.error(f"Error in speech detection: {e}")
            return False, 0.0
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the voice activity detector with the provided parameters.
        
        Args:
            config: Dictionary of configuration parameters, including:
                - aggressiveness: VAD aggressiveness mode (0-3)
                - sample_rate: Audio sample rate in Hz
                - frame_duration_ms: Frame duration in milliseconds
                - history_size: Number of frames to keep in history
                - speech_threshold: Threshold for speech classification
            
        Returns:
            bool: True if configuration was successful, False otherwise
        """
        try:
            # Handle aggressiveness
            if 'aggressiveness' in config:
                aggressiveness = config['aggressiveness']
                if not 0 <= aggressiveness <= 3:
                    self.logger.warning(f"Invalid aggressiveness {aggressiveness}, must be 0-3")
                else:
                    self.aggressiveness = aggressiveness
                    if self.vad:
                        self.vad.set_mode(self.aggressiveness)
            
            # Handle sample rate
            if 'sample_rate' in config:
                sample_rate = config['sample_rate']
                valid_sample_rates = [8000, 16000, 32000, 48000]
                if sample_rate not in valid_sample_rates:
                    self.logger.warning(f"Invalid sample rate {sample_rate}, must be one of {valid_sample_rates}")
                else:
                    self.sample_rate = sample_rate
                    self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
            
            # Handle frame duration
            if 'frame_duration_ms' in config:
                frame_duration_ms = config['frame_duration_ms']
                valid_durations = [10, 20, 30]
                if frame_duration_ms not in valid_durations:
                    self.logger.warning(f"Invalid frame duration {frame_duration_ms}, must be one of {valid_durations}")
                else:
                    self.frame_duration_ms = frame_duration_ms
                    self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
            
            # Handle history size
            if 'history_size' in config:
                history_size = config['history_size']
                if history_size < 1:
                    self.logger.warning(f"Invalid history size {history_size}, must be >= 1")
                else:
                    self.history_size = history_size
                    # Trim history if needed
                    if len(self.history) > self.history_size:
                        self.history = self.history[-self.history_size:]
            
            # Handle speech threshold
            if 'speech_threshold' in config:
                speech_threshold = config['speech_threshold']
                if not 0 <= speech_threshold <= 1:
                    self.logger.warning(f"Invalid speech threshold {speech_threshold}, must be between 0 and 1")
                else:
                    self.speech_threshold = speech_threshold
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring WebRTC VAD: {e}")
            return False
    
    def reset(self) -> None:
        """
        Reset the internal state of the voice activity detector.
        """
        self.history = []
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration of the voice activity detector.
        
        Returns:
            Dict[str, Any]: Dictionary of current configuration parameters
        """
        return {
            'aggressiveness': self.aggressiveness,
            'sample_rate': self.sample_rate,
            'frame_duration_ms': self.frame_duration_ms,
            'history_size': self.history_size,
            'speech_threshold': self.speech_threshold,
            'detector_type': self.get_name()
        }
    
    def get_name(self) -> str:
        """
        Get the name of the voice activity detector implementation.
        
        Returns:
            str: Name of the detector
        """
        return "WebRTC VAD"
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the voice activity detector.
        """
        self.vad = None
        self.history = []