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

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

# Cross-feature dependencies
from src.Features.AudioCapture.Events.AudioChunkCapturedEvent import AudioChunkCapturedEvent
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk

# Feature-specific imports
from src.Features.VoiceActivityDetection.Commands.DetectVoiceActivityCommand import DetectVoiceActivityCommand
from src.Features.VoiceActivityDetection.Commands.ConfigureVadCommand import ConfigureVadCommand
from src.Features.VoiceActivityDetection.Commands.EnableVadProcessingCommand import EnableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.DisableVadProcessingCommand import DisableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.ClearVadPreSpeechBufferCommand import ClearVadPreSpeechBufferCommand
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
        self.logger = LoggingModule.get_logger(__name__)
        self.event_bus = event_bus
        
        # Initialize empty detector registry - detectors will be created lazily
        self.detectors: Dict[str, IVoiceActivityDetector] = {}
        
        # Store detector configurations for lazy initialization
        self.detector_configs: Dict[str, Dict[str, Any]] = {
            'webrtc': {},
            'silero': {},
            'combined': {}
        }
        
        # Set default active detector
        self.active_detector_name = 'combined'  # Default to combined for best accuracy
        
        # Processing control - start with processing disabled to save resources
        self.processing_enabled = False
        
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
        
        # Optimization: Track pre-speech buffer duration for faster access
        self._pre_speech_buffer_duration = 0.0
        self._pre_speech_durations: Deque[float] = deque(maxlen=self.pre_speech_buffer_size)
        
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
        elif isinstance(command, EnableVadProcessingCommand):
            return self._handle_enable_vad_processing(command)
        elif isinstance(command, DisableVadProcessingCommand):
            return self._handle_disable_vad_processing(command)
        elif isinstance(command, ClearVadPreSpeechBufferCommand):
            return self._handle_clear_vad_pre_speech_buffer(command)
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
            ConfigureVadCommand,
            EnableVadProcessingCommand,
            DisableVadProcessingCommand,
            ClearVadPreSpeechBufferCommand
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
        
        # Check for warning conditions - moved from the command model to the handler
        if command.pre_speech_buffer_size < 16:
            self.logger.warning(
                f"Pre-speech buffer size is very small ({command.pre_speech_buffer_size}). "
                f"This may not capture enough audio before speech detection. "
                f"Recommended minimum size is 16 chunks (~0.5 seconds)."
            )
        
        # Update active detector
        self.active_detector_name = command.detector_type
        
        # Map command parameters to detector configuration
        config = command.map_to_detector_config()
        
        # Store configuration for this detector type
        self.detector_configs[command.detector_type] = config
        
        # If detector already exists, reconfigure it
        if command.detector_type in self.detectors:
            detector = self.detectors[command.detector_type]
            success = detector.configure(config)
        else:
            # Detector will be created with this config when first used
            self.logger.info(f"Stored configuration for {command.detector_type} detector (will be created on first use)")
            success = True
        
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
                
                # Optimization: Preserve the existing data without unnecessary list conversion
                # Create new deque directly with new maxlen
                self.pre_speech_buffer = deque(self.pre_speech_buffer, maxlen=new_size)
                
                # Reset and rebuild duration tracking with the new buffer
                self._pre_speech_durations = deque([chunk.get_duration() for chunk in self.pre_speech_buffer], maxlen=new_size)
                self._pre_speech_buffer_duration = sum(self._pre_speech_durations)
                
                # Log the update
                self.logger.info(f"Pre-speech buffer updated: size={new_size}, "
                               f"containing {len(self.pre_speech_buffer)} chunks, "
                               f"duration={self._pre_speech_buffer_duration:.2f}s")
        
        return success
    
    def _get_detector(self, detector_name: str) -> IVoiceActivityDetector:
        """
        Get the specified detector, creating it lazily if needed.
        
        Args:
            detector_name: Name of the detector to get
            
        Returns:
            IVoiceActivityDetector: The requested detector
            
        Raises:
            ValueError: If the detector name is not found
        """
        # Check if detector is already created
        if detector_name in self.detectors:
            return self.detectors[detector_name]
        
        # Check if this is a valid detector type
        if detector_name not in self.detector_configs:
            available_detectors = ", ".join(self.detector_configs.keys())
            raise ValueError(f"Detector '{detector_name}' not found. Available detectors: {available_detectors}")
        
        # Create the detector with stored configuration
        detector = self._create_detector(detector_name, self.detector_configs[detector_name])
        if detector:
            self.detectors[detector_name] = detector
            return detector
        else:
            raise ValueError(f"Failed to create detector '{detector_name}'")
    
    def _create_detector(self, detector_type: str, config: Dict[str, Any]) -> Optional[IVoiceActivityDetector]:
        """
        Create a detector with the given configuration.
        
        Args:
            detector_type: Type of detector to create
            config: Configuration parameters for the detector
            
        Returns:
            Optional[IVoiceActivityDetector]: The created detector or None if failed
        """
        try:
            if detector_type == 'webrtc':
                detector = WebRtcVadDetector(
                    aggressiveness=config.get('aggressiveness', 3),
                    frame_duration_ms=config.get('frame_duration_ms', 30),
                    sample_rate=config.get('sample_rate', 16000),
                    speech_threshold=config.get('speech_threshold', 0.6)
                )
            elif detector_type == 'silero':
                detector = SileroVadDetector(
                    threshold=config.get('threshold', 0.5),
                    sample_rate=config.get('sample_rate', 16000),
                    min_speech_duration_ms=config.get('min_speech_duration_ms', 250),
                    min_silence_duration_ms=config.get('min_silence_duration_ms', 100)
                )
            elif detector_type == 'combined':
                detector = CombinedVadDetector(
                    webrtc_aggressiveness=config.get('webrtc_aggressiveness', 2),
                    silero_threshold=config.get('silero_threshold', 0.6),
                    sample_rate=config.get('sample_rate', 16000),
                    frame_duration_ms=config.get('frame_duration_ms', 30),
                    speech_confirmation_frames=config.get('speech_confirmation_frames', 2),
                    silence_confirmation_frames=config.get('silence_confirmation_frames', 30),
                    speech_buffer_size=config.get('speech_buffer_size', 100),
                    webrtc_threshold=config.get('webrtc_threshold', 0.6),
                    use_silero_confirmation=config.get('use_silero_confirmation', True)
                )
            else:
                self.logger.error(f"Unknown detector type: {detector_type}")
                return None
            
            # Set up the detector
            if detector.setup():
                self.logger.info(f"Successfully created and set up {detector_type} detector")
                return detector
            else:
                self.logger.error(f"Failed to set up {detector_type} detector")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating {detector_type} detector: {e}")
            return None
    
    def _handle_enable_vad_processing(self, command: EnableVadProcessingCommand) -> bool:
        """
        Handle EnableVadProcessingCommand.
        
        This method enables VAD processing for audio chunks.
        
        Args:
            command: The EnableVadProcessingCommand
            
        Returns:
            bool: True to indicate successful execution
        """
        if not self.processing_enabled:
            self.processing_enabled = True
            self.logger.info("VAD audio processing enabled")
        else:
            self.logger.debug("VAD audio processing already enabled")
            
        return True
        
    def _handle_disable_vad_processing(self, command: DisableVadProcessingCommand) -> bool:
        """
        Handle DisableVadProcessingCommand.
        
        This method disables VAD processing for audio chunks to save resources.
        
        Args:
            command: The DisableVadProcessingCommand
            
        Returns:
            bool: True to indicate successful execution
        """
        if self.processing_enabled:
            self.processing_enabled = False
            self.logger.info("VAD audio processing disabled")
        else:
            self.logger.debug("VAD audio processing already disabled")
            
        return True
        
    def _handle_clear_vad_pre_speech_buffer(self, command: ClearVadPreSpeechBufferCommand) -> None:
        """
        Handle ClearVadPreSpeechBufferCommand by clearing the pre-speech buffer.
        
        This method is typically called after a wake word is detected to ensure
        that the audio transcribed doesn't include the wake word itself or
        audio captured before it.
        
        Args:
            command: The ClearVadPreSpeechBufferCommand
            
        Returns:
            None: No specific return value needed for this command type
        """
        self.logger.info("Clearing VAD pre-speech buffer on command.")
        self.pre_speech_buffer.clear()
        self._pre_speech_durations.clear()
        self._pre_speech_buffer_duration = 0.0
        return None
    
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
        
        # Add new audio chunk to the pre-speech buffer and update duration tracking
        chunk_duration = audio_chunk.get_duration()
        
        # If the buffer is full, we need to subtract the duration of the oldest chunk
        if len(self.pre_speech_buffer) == self.pre_speech_buffer_size:
            # When the buffer is full, appending will drop the oldest item
            # So we need to keep track of the duration we're losing
            if self._pre_speech_durations:
                oldest_duration = self._pre_speech_durations[0]
                self._pre_speech_buffer_duration -= oldest_duration
        
        # Add the new chunk to the buffer
        self.pre_speech_buffer.append(audio_chunk)
        
        # Update duration tracking
        self._pre_speech_durations.append(chunk_duration)
        self._pre_speech_buffer_duration += chunk_duration
        
        # Skip VAD processing if disabled - this is the key optimization
        if not self.processing_enabled:
            return
        
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
            
            # Use the pre-calculated duration for the pre-speech buffer
            # This avoids recalculating durations for each chunk
            pre_speech_duration = self._pre_speech_buffer_duration
            
            # Adjust speech start time to account for pre-speech buffer
            self.speech_start_time = current_time - pre_speech_duration
            
            # Initialize speech buffer with pre-speech chunks plus current chunk
            # Using a direct approach to avoid unnecessary list conversion
            buffer_size = len(self.pre_speech_buffer)
            self.speech_buffer.clear()
            
            # Add pre-speech buffer contents
            for chunk in self.pre_speech_buffer:
                self.speech_buffer.append(chunk)
                
            # Add current chunk
            self.speech_buffer.append(audio_chunk)
            
            # Log the pre-speech buffer inclusion
            self.logger.info(f"Including {buffer_size} pre-speech chunks "
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
                    # Calculate total size in advance to pre-allocate memory
                    total_size = 0
                    for chunk in self.speech_buffer:
                        chunk_data = chunk.to_float32()
                        if chunk_data is not None and chunk_data.size > 0:
                            total_size += chunk_data.size
                    
                    if total_size > 0:
                        # Pre-allocate array of correct size to avoid multiple resizes
                        audio_data = np.zeros(total_size, dtype=np.float32)
                        
                        # Fill the pre-allocated array
                        position = 0
                        for chunk in self.speech_buffer:
                            chunk_data = chunk.to_float32()
                            if chunk_data is not None and chunk_data.size > 0:
                                # Copy data to the right position
                                chunk_size = chunk_data.size
                                audio_data[position:position+chunk_size] = chunk_data
                                position += chunk_size
                        
                        # No need to convert type as we created float32 array directly
                        
                        # Check if normalization is needed
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
        
        # Clear all buffers to free memory
        self.speech_buffer.clear()
        self.pre_speech_buffer.clear()
        self._pre_speech_durations.clear()
        
        # Reset counters and tracking
        self._pre_speech_buffer_duration = 0.0
        
        # Remove detector references
        self.detectors.clear()
        
        # Log cleanup
        self.logger.info("VoiceActivityHandler resources cleaned up")
        
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