"""
VoiceActivityHandler for processing voice activity detection commands.

This handler implements the ICommandHandler interface to process VAD-related commands
like detecting speech and configuring VAD parameters.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Type, cast, Deque
from collections import deque

# Core imports
from src.Core.Common.Interfaces.command_handler import ICommandHandler
from src.Core.Commands.command import Command
from src.Core.Events.event_bus import IEventBus
from src.Core.Common.Interfaces.voice_activity_detector import IVoiceActivityDetector

# Cross-feature dependencies
from src.Features.AudioCapture.Events.AudioChunkCapturedEvent import AudioChunkCapturedEvent
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk

# Feature-specific imports
from src.Features.VoiceActivityDetection.Commands.DetectVoiceActivityCommand import DetectVoiceActivityCommand
from src.Features.VoiceActivityDetection.Commands.ConfigureVadCommand import ConfigureVadCommand
from src.Features.VoiceActivityDetection.Events.SpeechDetectedEvent import SpeechDetectedEvent
from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
from src.Features.VoiceActivityDetection.Detectors.WebRtcVadDetector import WebRtcVadDetector
from src.Features.VoiceActivityDetection.Detectors.SileroVadDetector import SileroVadDetector
from src.Features.VoiceActivityDetection.Detectors.CombinedVadDetector import CombinedVadDetector


class VoiceActivityHandler(ICommandHandler[Any]):
    """
    Handler for voice activity detection commands.
    
    This handler processes commands related to voice activity detection and
    manages the VAD detectors. It also subscribes to audio events to perform
    continuous speech detection.
    """
    
    def __init__(self, event_bus: IEventBus):
        """
        Initialize the voice activity handler.
        
        Args:
            event_bus: Event bus for publishing and subscribing to events
        """
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus
        
        # Initialize detector registry
        self.detectors: Dict[str, IVoiceActivityDetector] = {
            'webrtc': WebRtcVadDetector(),
            'silero': SileroVadDetector(),
            'combined': CombinedVadDetector()
        }
        
        # Set up all detectors
        for name, detector in self.detectors.items():
            if not detector.setup():
                self.logger.warning(f"Failed to set up {name} detector")
        
        # Set default active detector
        self.active_detector_name = 'webrtc'  # Use WebRTC by default (lightweight)
        
        # State tracking
        self.in_speech = False
        self.current_speech_id = ""
        self.speech_start_time = 0.0
        self.last_audio_timestamp = 0.0
        
        # Buffer configuration
        self.buffer_limit = 10000  # Maximum number of chunks to buffer (for 5+ minutes of speech)
        self.pre_speech_buffer_size = 64  # ~2 seconds at 32ms/chunk
        
        # Buffers
        # Pre-speech buffer: Continuously maintains the last N chunks of audio, even before speech is detected
        self.pre_speech_buffer: Deque[AudioChunk] = deque(maxlen=self.pre_speech_buffer_size)
        
        # Speech buffer: Contains the full speech segment, including pre-speech + active speech
        self.speech_buffer: Deque[AudioChunk] = deque(maxlen=self.buffer_limit)
        
        # Register for audio events
        self.event_bus.subscribe(AudioChunkCapturedEvent, self._on_audio_chunk_captured)
    
    def handle(self, command: Command) -> Any:
        """
        Handle a voice activity detection command and produce a result.
        
        Args:
            command: The command to handle
            
        Returns:
            The result of the command execution (type depends on command)
            
        Raises:
            TypeError: If the command is not of the expected type
            ValueError: If the command contains invalid data
            Exception: If an error occurs during command execution
        """
        if isinstance(command, DetectVoiceActivityCommand):
            return self._handle_detect_voice_activity(command)
        elif isinstance(command, ConfigureVadCommand):
            return self._handle_configure_vad(command)
        else:
            raise TypeError(f"Unsupported command type: {type(command).__name__}")
    
    def can_handle(self, command: Command) -> bool:
        """
        Check if this handler can handle the given command.
        
        Args:
            command: The command to check
            
        Returns:
            bool: True if this handler can handle the command, False otherwise
        """
        return isinstance(command, (
            DetectVoiceActivityCommand,
            ConfigureVadCommand
        ))
    
    def _handle_detect_voice_activity(self, command: DetectVoiceActivityCommand) -> Union[bool, Dict[str, Any]]:
        """
        Handle a DetectVoiceActivityCommand.
        
        This processes an audio chunk to detect voice activity.
        
        Args:
            command: The DetectVoiceActivityCommand
            
        Returns:
            Union[bool, Dict[str, Any]]: Boolean result or dict with result and confidence
        """
        self.logger.debug("Handling DetectVoiceActivityCommand")
        
        # Choose detector to use
        detector_name = command.detector_type or self.active_detector_name
        detector = self._get_detector(detector_name)
        
        # Adjust sensitivity if provided
        if command.sensitivity is not None:
            detector.configure({'threshold': command.sensitivity})
        
        # Detect speech
        audio_data = command.audio_chunk.raw_data
        sample_rate = command.audio_chunk.sample_rate
        
        if command.return_confidence:
            is_speech, confidence = detector.detect_with_confidence(audio_data, sample_rate)
            return {
                'is_speech': is_speech,
                'confidence': confidence,
                'detector': detector_name
            }
        else:
            is_speech = detector.detect(audio_data, sample_rate)
            return is_speech
    
    def _handle_configure_vad(self, command: ConfigureVadCommand) -> bool:
        """
        Handle a ConfigureVadCommand.
        
        This configures the VAD system parameters.
        
        Args:
            command: The ConfigureVadCommand
            
        Returns:
            bool: True if configuration was successful
        """
        self.logger.info(f"Handling ConfigureVadCommand for detector_type={command.detector_type}")
        
        # Update active detector
        self.active_detector_name = command.detector_type
        
        # Get detector
        detector = self._get_detector(command.detector_type)
        
        # Map command parameters to detector configuration
        config = command.map_to_detector_config()
        
        # Configure detector
        success = detector.configure(config)
        
        # Update buffer limit if specified
        if 'buffer_limit' in command.parameters:
            self.buffer_limit = command.parameters['buffer_limit']
            
        # Update pre-speech buffer size if specified
        if 'pre_speech_buffer_size' in config:
            # Get the new buffer size
            new_size = config['pre_speech_buffer_size']
            
            # Only update if it's different from current size
            if new_size != self.pre_speech_buffer_size:
                self.logger.info(f"Updating pre-speech buffer size from {self.pre_speech_buffer_size} to {new_size}")
                self.pre_speech_buffer_size = new_size
                
                # Create a new buffer with new size, preserving data if possible
                # First convert current buffer to list
                current_data = list(self.pre_speech_buffer)
                
                # Create new buffer with new size
                self.pre_speech_buffer = deque(current_data, maxlen=new_size)
                
                # Log the update
                self.logger.info(f"Pre-speech buffer updated: size={new_size}, "
                               f"containing {len(self.pre_speech_buffer)} chunks")
        
        return success
    
    def _get_detector(self, detector_name: str) -> IVoiceActivityDetector:
        """
        Get the specified detector.
        
        Args:
            detector_name: Name of the detector to get
            
        Returns:
            IVoiceActivityDetector: The requested detector
            
        Raises:
            ValueError: If the detector name is not found
        """
        if detector_name not in self.detectors:
            available_detectors = ", ".join(self.detectors.keys())
            raise ValueError(f"Detector '{detector_name}' not found. Available detectors: {available_detectors}")
        
        return self.detectors[detector_name]
    
    def _on_audio_chunk_captured(self, event: AudioChunkCapturedEvent) -> None:
        """
        Handle an audio chunk captured event.
        
        This method is called when a new audio chunk is available, and it performs
        voice activity detection on the chunk.
        
        Args:
            event: The AudioChunkCapturedEvent
        """
        # Only process if we have an active detector
        if not self.active_detector_name:
            return
        
        audio_chunk = event.audio_chunk
        self.last_audio_timestamp = audio_chunk.timestamp
        
        # Always add new audio chunk to the pre-speech buffer
        # This maintains a sliding window of recent audio regardless of speech detection
        self.pre_speech_buffer.append(audio_chunk)
        
        try:
            # Skip using the command object entirely and call the detector directly
            detector = self._get_detector(self.active_detector_name)
            
            # Detect speech directly
            is_speech, confidence = detector.detect_with_confidence(
                audio_data=audio_chunk.raw_data, 
                sample_rate=audio_chunk.sample_rate
            )
            
            # Handle state transitions
            self._update_speech_state(is_speech, confidence, audio_chunk)
            
        except Exception as e:
            self.logger.error(f"Error processing audio chunk for VAD: {e}")
    
    def _update_speech_state(self, is_speech: bool, confidence: float, audio_chunk: AudioChunk) -> None:
        """
        Update the speech detection state and trigger events.
        
        Args:
            is_speech: Whether speech was detected
            confidence: Confidence of the detection
            audio_chunk: The audio chunk being processed
        """
        current_time = time.time()
        
        # Transition from silence to speech
        if is_speech and not self.in_speech:
            self.in_speech = True
            self.current_speech_id = str(time.time_ns())
            
            # Include pre-speech buffer in the speech buffer to capture audio before detection
            # Create a copy of the pre-speech buffer to avoid reference issues
            pre_speech_chunks = list(self.pre_speech_buffer)
            
            # Calculate the duration of the pre-speech buffer for timing adjustment
            pre_speech_duration = sum(chunk.get_duration() for chunk in pre_speech_chunks)
            
            # Adjust speech start time to account for pre-speech buffer
            self.speech_start_time = current_time - pre_speech_duration
            
            # Initialize speech buffer with pre-speech chunks plus current chunk
            self.speech_buffer = deque(pre_speech_chunks + [audio_chunk], maxlen=self.buffer_limit)
            
            # Log the pre-speech buffer inclusion
            self.logger.info(f"Including {len(pre_speech_chunks)} pre-speech chunks "
                           f"({pre_speech_duration:.2f}s) in speech detection")
            
            # Publish speech detected event
            self.event_bus.publish(SpeechDetectedEvent(
                confidence=confidence,
                audio_timestamp=audio_chunk.timestamp,
                detector_type=self.active_detector_name,
                audio_reference=audio_chunk,  # Keep using current chunk as reference for compatibility
                speech_id=self.current_speech_id
            ))
            self.logger.debug(f"Speech started with confidence {confidence:.2f}")
            
        # Continued speech
        elif is_speech and self.in_speech:
            # Add to speech buffer (deque automatically handles size limiting)
            self.speech_buffer.append(audio_chunk)
                
        # Transition from speech to silence
        elif not is_speech and self.in_speech:
            self.in_speech = False
            speech_duration = current_time - self.speech_start_time
            
            # Convert speech buffer to numpy array for transcription
            # First, extract raw data from each audio chunk in the buffer
            import numpy as np
            
            if len(self.speech_buffer) > 0:
                try:
                    # First check if all arrays have the same shape
                    raw_data_list = []
                    for chunk in self.speech_buffer:
                        # Use the to_float32 method to get normalized numpy array
                        chunk_data = chunk.to_float32()
                        if chunk_data is not None and chunk_data.size > 0:
                            # Ensure it's at least 1D
                            if chunk_data.ndim == 0:
                                chunk_data = np.array([chunk_data.item()], dtype=np.float32)
                            raw_data_list.append(chunk_data)
                    
                    if raw_data_list:
                        # Combine all valid audio chunks into a single numpy array
                        audio_data = np.concatenate(raw_data_list)
                        
                        # Ensure the array is the right type and normalized
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32)
                        
                        # Normalize if not already in [-1, 1] range
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 0 and max_val > 1.0:
                            audio_data = audio_data / max_val
                    else:
                        # No valid chunks found
                        audio_data = np.array([0.0], dtype=np.float32)  # Create a single sample silent array
                except Exception as e:
                    self.logger.error(f"Error processing speech buffer: {e}")
                    # Create a fallback array
                    audio_data = np.array([0.0], dtype=np.float32)  # Create a single sample silent array
            else:
                # Empty buffer case - create an empty array with at least one sample
                audio_data = np.array([0.0], dtype=np.float32)
            
            # Publish silence detected event with the numpy array
            self.event_bus.publish(SilenceDetectedEvent(
                speech_duration=speech_duration,
                audio_timestamp=audio_chunk.timestamp,
                speech_start_time=self.speech_start_time,
                speech_end_time=current_time,
                audio_reference=audio_data,
                speech_id=self.current_speech_id
            ))
            self.logger.debug(f"Speech ended, duration: {speech_duration:.2f}s")
            
            # Clear speech buffer
            self.speech_buffer = deque(maxlen=self.buffer_limit)
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the handler.
        
        This should be called when the handler is no longer needed.
        """
        # Unsubscribe from events
        self.event_bus.unsubscribe(AudioChunkCapturedEvent, self._on_audio_chunk_captured)
        
        # Clean up detectors
        for detector in self.detectors.values():
            detector.cleanup()
        
        # Clear buffers
        self.speech_buffer.clear()
        
    def get_buffer_duration(self) -> float:
        """
        Get the total duration of audio in the speech buffer.
        
        Returns:
            float: Duration in seconds
        """
        return sum(chunk.get_duration() for chunk in self.speech_buffer)
        
    def set_buffer_limit(self, max_chunks: int) -> None:
        """
        Set the maximum number of audio chunks to buffer.
        
        Args:
            max_chunks: Maximum number of chunks
        """
        if max_chunks < 1:
            raise ValueError(f"Buffer limit must be at least 1, got {max_chunks}")
            
        # Create a new deque with the new limit
        new_buffer = deque(self.speech_buffer, maxlen=max_chunks)
        self.speech_buffer = new_buffer
        self.buffer_limit = max_chunks