"""
CombinedVadDetector implementation of IVoiceActivityDetector.

This module provides a combined approach to voice activity detection by 
employing a two-stage system: fast detection with WebRTC VAD followed by 
more accurate verification with Silero VAD.
"""

import time
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Union, TYPE_CHECKING

import numpy as np

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

if TYPE_CHECKING:
    from src.Core.Common.Interfaces.voice_activity_detector import IVoiceActivityDetector
else:
    from src.Core.Common.Interfaces.voice_activity_detector import IVoiceActivityDetector
from src.Features.VoiceActivityDetection.Detectors.WebRtcVadDetector import WebRtcVadDetector
from src.Features.VoiceActivityDetection.Detectors.SileroVadDetector import SileroVadDetector


class DetectionState(Enum):
    """State machine states for voice activity detection"""
    SILENCE = 0
    POTENTIAL_SPEECH = 1
    SPEECH = 2
    POTENTIAL_SILENCE = 3


class CombinedVadDetector(IVoiceActivityDetector):
    """
    Combined voice activity detector using both WebRTC and Silero VAD.
    
    This implementation uses a two-stage approach for optimal performance:
    1. WebRTC VAD for fast initial detection (low computational cost)
    2. Silero VAD for verification of detected speech (higher accuracy)
    
    The detector maintains a state machine to smooth transitions between
    speech and silence, reducing false positives and improving stability.
    """
    
    def __init__(self, 
                 webrtc_aggressiveness: int = 2,
                 silero_threshold: float = 0.6,  # Increased from 0.4 for more conservative detection
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30,
                 speech_confirmation_frames: int = 2,  # Reduced from 3 for faster detection
                 silence_confirmation_frames: int = 30,  # Increased to 30 to allow for longer pauses in extended speech
                 speech_buffer_size: int = 100,  # Increased to 100 for better tracking of long speech segments
                 webrtc_threshold: float = 0.6,  # Decreased from 0.8 for more sensitive detection
                 use_silero_confirmation: bool = True):
        """
        Initialize the Combined VAD detector.
        
        Args:
            webrtc_aggressiveness: WebRTC VAD aggressiveness (0-3)
            silero_threshold: Silero speech probability threshold (0.0-1.0)
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration for WebRTC VAD
            speech_confirmation_frames: Number of frames required to confirm speech
            silence_confirmation_frames: Number of frames required to confirm silence
            speech_buffer_size: Size of buffer for tracking speech frames
            webrtc_threshold: Threshold for speech in WebRTC history buffer
            use_silero_confirmation: Whether to use Silero for confirmation
        """
        self.logger = LoggingModule.get_logger(__name__)
        # Removed direct log level setting - now controlled by LoggingModule configuration
        
        # Core parameters
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.speech_confirmation_frames = speech_confirmation_frames
        self.silence_confirmation_frames = silence_confirmation_frames
        self.speech_buffer_size = speech_buffer_size
        self.use_silero_confirmation = use_silero_confirmation
        
        # Initialize detectors
        self.webrtc_detector = WebRtcVadDetector(
            aggressiveness=webrtc_aggressiveness,
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            history_size=speech_buffer_size,
            speech_threshold=webrtc_threshold
        )
        
        # Calculate window size ensuring minimum of 512 samples for Silero
        window_size_samples = int(sample_rate * frame_duration_ms / 1000)
        if window_size_samples < 512:
            self.logger.warning(f"Calculated window size {window_size_samples} is too small for Silero VAD (minimum 512), adjusting to 512")
            window_size_samples = 512
        
        self.silero_detector = SileroVadDetector(
            threshold=silero_threshold,
            sample_rate=sample_rate,
            window_size_samples=window_size_samples
        )
        
        # State tracking
        self.state = DetectionState.SILENCE
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.speech_start_time = 0.0
        self.current_speech_duration = 0.0
        
        # Statistics for debugging and analysis
        self.stats = {
            'webrtc_positives': 0,
            'silero_confirmations': 0,
            'silero_rejections': 0,
            'state_transitions': 0,
            'speech_segments': 0,
            'avg_speech_duration': 0.0,
            'total_speech_duration': 0.0
        }
    
    def setup(self) -> bool:
        """
        Initialize the VAD detectors.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            webrtc_success = self.webrtc_detector.setup()
            silero_success = self.silero_detector.setup()
            
            if not webrtc_success:
                self.logger.error("Failed to initialize WebRTC VAD detector")
                return False
            
            if self.use_silero_confirmation and not silero_success:
                self.logger.error("Failed to initialize Silero VAD detector")
                return False
            
            self.reset()
            self.logger.info("Initialized Combined VAD detector")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Combined VAD detector: {e}")
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
        
        This implements a state machine approach for stable speech detection:
        - SILENCE → POTENTIAL_SPEECH: When WebRTC detects initial speech
        - POTENTIAL_SPEECH → SPEECH: When enough consecutive speech frames are detected
        - SPEECH → POTENTIAL_SILENCE: When silence is initially detected
        - POTENTIAL_SILENCE → SILENCE: When enough consecutive silence frames are detected
        
        Silero VAD is used to confirm WebRTC detections when transitioning to SPEECH state.
        
        Args:
            audio_data: Raw audio data as bytes (must be 16-bit PCM)
            sample_rate: Sample rate of the audio data (optional, uses default if None)
            
        Returns:
            Tuple[bool, float]: (speech_detected, confidence_score)
        """
        # Use provided sample rate or default
        rate = sample_rate if sample_rate is not None else self.sample_rate
        
        try:
            # Always run WebRTC VAD (fast)
            webrtc_speech, webrtc_confidence = self.webrtc_detector.detect_with_confidence(audio_data, rate)
            
            # Store current timestamp for duration calculations
            current_time = time.time()
            
            # Update state machine based on current state and detection results
            if self.state == DetectionState.SILENCE:
                if webrtc_speech:
                    # Potential transition to speech
                    self.stats['webrtc_positives'] += 1
                    self.state = DetectionState.POTENTIAL_SPEECH
                    self.consecutive_speech_frames = 1
                    self.speech_start_time = current_time
                    self._log_state_transition("SILENCE", "POTENTIAL_SPEECH")
                    self.logger.debug(f"WebRTC detected initial speech with confidence {webrtc_confidence:.2f}")
                    return False, webrtc_confidence
                else:
                    # Stay in silence
                    return False, webrtc_confidence
                    
            elif self.state == DetectionState.POTENTIAL_SPEECH:
                if webrtc_speech:
                    # More evidence of speech
                    self.consecutive_speech_frames += 1
                    
                    # Check if we have enough evidence to confirm speech
                    if self.consecutive_speech_frames >= self.speech_confirmation_frames:
                        # If Silero confirmation is enabled, use it to verify
                        if self.use_silero_confirmation:
                            silero_speech, silero_confidence = self.silero_detector.detect_with_confidence(audio_data, rate)
                            
                            if silero_speech:
                                # Both detectors agree: this is speech
                                self.stats['silero_confirmations'] += 1
                                self.state = DetectionState.SPEECH
                                self.consecutive_silence_frames = 0
                                self.stats['speech_segments'] += 1
                                self._log_state_transition("POTENTIAL_SPEECH", "SPEECH")
                                self.logger.info(f"Speech confirmed by both WebRTC ({webrtc_confidence:.2f}) and Silero ({silero_confidence:.2f})")
                                return True, max(webrtc_confidence, silero_confidence)
                            else:
                                # Silero rejects: stay in potential speech
                                self.stats['silero_rejections'] += 1
                                self.logger.debug(f"WebRTC detected speech ({webrtc_confidence:.2f}) but Silero rejected ({silero_confidence:.2f})")
                                return False, (webrtc_confidence + silero_confidence) / 2
                        else:
                            # No Silero confirmation required
                            self.state = DetectionState.SPEECH
                            self.consecutive_silence_frames = 0
                            self.stats['speech_segments'] += 1
                            self._log_state_transition("POTENTIAL_SPEECH", "SPEECH")
                            return True, webrtc_confidence
                    else:
                        # Not enough evidence yet
                        return False, webrtc_confidence
                else:
                    # Speech not sustained: go back to silence
                    self.state = DetectionState.SILENCE
                    self.consecutive_speech_frames = 0
                    self._log_state_transition("POTENTIAL_SPEECH", "SILENCE")
                    return False, webrtc_confidence
                    
            elif self.state == DetectionState.SPEECH:
                if not webrtc_speech:
                    # Potential end of speech
                    self.state = DetectionState.POTENTIAL_SILENCE
                    self.consecutive_silence_frames = 1
                    self._log_state_transition("SPEECH", "POTENTIAL_SILENCE")
                    return True, webrtc_confidence  # Still report as speech until confirmed silence
                else:
                    # Continued speech
                    self.current_speech_duration = current_time - self.speech_start_time
                    return True, webrtc_confidence
                    
            elif self.state == DetectionState.POTENTIAL_SILENCE:
                if not webrtc_speech:
                    # More evidence of silence
                    self.consecutive_silence_frames += 1
                    
                    # Check if we have enough evidence to confirm silence
                    if self.consecutive_silence_frames >= self.silence_confirmation_frames:
                        # End of speech confirmed
                        self.state = DetectionState.SILENCE
                        self.consecutive_speech_frames = 0
                        
                        # Update statistics
                        speech_duration = current_time - self.speech_start_time
                        self.stats['total_speech_duration'] += speech_duration
                        self.stats['avg_speech_duration'] = (
                            self.stats['total_speech_duration'] / self.stats['speech_segments']
                            if self.stats['speech_segments'] > 0 else 0.0
                        )
                        
                        self._log_state_transition("POTENTIAL_SILENCE", "SILENCE")
                        return False, webrtc_confidence
                    else:
                        # Not enough evidence yet
                        return True, webrtc_confidence  # Still report as speech
                else:
                    # Speech resumed
                    self.state = DetectionState.SPEECH
                    self.consecutive_silence_frames = 0
                    self._log_state_transition("POTENTIAL_SILENCE", "SPEECH")
                    return True, webrtc_confidence
            
            # Default fallback (should not reach here)
            return webrtc_speech, webrtc_confidence
            
        except Exception as e:
            self.logger.error(f"Error in combined speech detection: {e}")
            return False, 0.0
    
    def _log_state_transition(self, from_state: str, to_state: str) -> None:
        """Log state transitions for debugging"""
        self.logger.debug(f"VAD State Transition: {from_state} → {to_state}")
        self.stats['state_transitions'] += 1
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the voice activity detector with the provided parameters.
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            bool: True if configuration was successful, False otherwise
        """
        try:
            # WebRTC-specific configurations
            webrtc_config = {}
            
            if 'webrtc_aggressiveness' in config:
                webrtc_aggressiveness = config['webrtc_aggressiveness']
                webrtc_config['aggressiveness'] = webrtc_aggressiveness
            
            if 'webrtc_threshold' in config:
                webrtc_threshold = config['webrtc_threshold']
                webrtc_config['speech_threshold'] = webrtc_threshold
            
            # Silero-specific configurations
            silero_config = {}
            
            if 'silero_threshold' in config:
                silero_threshold = config['silero_threshold']
                silero_config['threshold'] = silero_threshold
            
            # Common configurations
            if 'sample_rate' in config:
                sample_rate = config['sample_rate']
                self.sample_rate = sample_rate
                webrtc_config['sample_rate'] = sample_rate
                silero_config['sample_rate'] = sample_rate
            
            if 'frame_duration_ms' in config:
                frame_duration_ms = config['frame_duration_ms']
                self.frame_duration_ms = frame_duration_ms
                webrtc_config['frame_duration_ms'] = frame_duration_ms
                
                # Calculate window size ensuring minimum of 512 samples for Silero
                window_size_samples = int(self.sample_rate * frame_duration_ms / 1000)
                if window_size_samples < 512:
                    self.logger.warning(f"Calculated window size {window_size_samples} is too small for Silero VAD (minimum 512), adjusting to 512")
                    window_size_samples = 512
                silero_config['window_size_samples'] = window_size_samples
            
            # State machine configurations
            if 'speech_confirmation_frames' in config:
                speech_confirmation_frames = config['speech_confirmation_frames']
                if speech_confirmation_frames < 1:
                    self.logger.warning(f"Invalid speech_confirmation_frames {speech_confirmation_frames}, must be >= 1")
                else:
                    self.speech_confirmation_frames = speech_confirmation_frames
            
            if 'silence_confirmation_frames' in config:
                silence_confirmation_frames = config['silence_confirmation_frames']
                if silence_confirmation_frames < 1:
                    self.logger.warning(f"Invalid silence_confirmation_frames {silence_confirmation_frames}, must be >= 1")
                else:
                    self.silence_confirmation_frames = silence_confirmation_frames
            
            if 'speech_buffer_size' in config:
                speech_buffer_size = config['speech_buffer_size']
                if speech_buffer_size < 1:
                    self.logger.warning(f"Invalid speech_buffer_size {speech_buffer_size}, must be >= 1")
                else:
                    self.speech_buffer_size = speech_buffer_size
                    webrtc_config['history_size'] = speech_buffer_size
            
            if 'use_silero_confirmation' in config:
                self.use_silero_confirmation = config['use_silero_confirmation']
            
            # Apply configurations to detectors
            if webrtc_config:
                self.webrtc_detector.configure(webrtc_config)
            
            # Always configure Silero detector regardless of use_silero_confirmation setting
            # This ensures that configuration changes are preserved even if Silero is only enabled later
            if silero_config:
                self.logger.debug(f"Configuring Silero VAD with: {silero_config}")
                result = self.silero_detector.configure(silero_config)
                self.logger.debug(f"Silero configuration result: {result}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring Combined VAD: {e}")
            return False
    
    def reset(self) -> None:
        """
        Reset the internal state of the voice activity detector.
        """
        self.webrtc_detector.reset()
        self.silero_detector.reset()
        
        self.state = DetectionState.SILENCE
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.speech_start_time = 0.0
        self.current_speech_duration = 0.0
        
        # Reset statistics
        self.stats = {
            'webrtc_positives': 0,
            'silero_confirmations': 0,
            'silero_rejections': 0,
            'state_transitions': 0,
            'speech_segments': 0,
            'avg_speech_duration': 0.0,
            'total_speech_duration': 0.0
        }
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration of the voice activity detector.
        
        Returns:
            Dict[str, Any]: Dictionary of current configuration parameters
        """
        webrtc_config = self.webrtc_detector.get_configuration()
        silero_config = self.silero_detector.get_configuration()
        
        return {
            'detector_type': self.get_name(),
            'sample_rate': self.sample_rate,
            'frame_duration_ms': self.frame_duration_ms,
            'speech_confirmation_frames': self.speech_confirmation_frames,
            'silence_confirmation_frames': self.silence_confirmation_frames,
            'speech_buffer_size': self.speech_buffer_size,
            'use_silero_confirmation': self.use_silero_confirmation,
            'webrtc_aggressiveness': webrtc_config['aggressiveness'],
            'webrtc_threshold': webrtc_config['speech_threshold'],
            'silero_threshold': silero_config['threshold'],
            'current_state': self.state.name
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current detection statistics.
        
        Returns:
            Dict[str, Any]: Dictionary of detection statistics
        """
        return self.stats
    
    def get_name(self) -> str:
        """
        Get the name of the voice activity detector implementation.
        
        Returns:
            str: Name of the detector
        """
        return "Combined VAD (WebRTC + Silero)"
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the voice activity detector.
        """
        self.webrtc_detector.cleanup()
        self.silero_detector.cleanup()