"""
WakeWordCommandHandler for processing wake word detection commands.

This handler implements the ICommandHandler interface to process wake word-related commands
like detecting wake words and configuring wake word detection parameters.
"""

import time
from typing import Any, Dict, List, Optional, Union, Type, cast, Deque
from collections import deque
from enum import Enum, auto

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

# Core imports
from src.Core.Common.Interfaces.command_handler import ICommandHandler
from src.Core.Commands.command import Command
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import IEventBus
from src.Core.Common.Interfaces.wake_word_detector import IWakeWordDetector

# Cross-feature dependencies
from src.Features.AudioCapture.Events.AudioChunkCapturedEvent import AudioChunkCapturedEvent
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk
from src.Features.VoiceActivityDetection.Events.SpeechDetectedEvent import SpeechDetectedEvent
from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
from src.Features.VoiceActivityDetection.Commands.ConfigureVadCommand import ConfigureVadCommand
from src.Features.VoiceActivityDetection.Commands.EnableVadProcessingCommand import EnableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.DisableVadProcessingCommand import DisableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.ClearVadPreSpeechBufferCommand import ClearVadPreSpeechBufferCommand

# Feature-specific imports
from src.Features.WakeWordDetection.Commands.ConfigureWakeWordCommand import ConfigureWakeWordCommand
from src.Features.WakeWordDetection.Commands.StartWakeWordDetectionCommand import StartWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.StopWakeWordDetectionCommand import StopWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.DetectWakeWordCommand import DetectWakeWordCommand
from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
from src.Features.WakeWordDetection.Events.WakeWordDetectionStartedEvent import WakeWordDetectionStartedEvent
from src.Features.WakeWordDetection.Events.WakeWordDetectionStoppedEvent import WakeWordDetectionStoppedEvent
from src.Features.WakeWordDetection.Events.WakeWordTimeoutEvent import WakeWordTimeoutEvent
from src.Features.WakeWordDetection.Detectors.PorcupineWakeWordDetector import PorcupineWakeWordDetector
from src.Features.WakeWordDetection.Models.WakeWordConfig import WakeWordConfig


class DetectorState(Enum):
    """State machine states for wake word detection."""
    INACTIVE = auto()
    WAKE_WORD = auto()
    LISTENING = auto()
    RECORDING = auto()
    PROCESSING = auto()


class WakeWordCommandHandler(ICommandHandler[Any]):
    """
    Handler for wake word detection commands.
    
    This handler processes commands related to wake word detection and
    manages the wake word detectors. It also subscribes to audio events 
    to perform continuous wake word detection.
    """
    
    def __init__(self, event_bus: IEventBus, command_dispatcher: CommandDispatcher):
        """
        Initialize the wake word handler.
        
        Args:
            event_bus: Event bus for publishing and subscribing to events
            command_dispatcher: Command dispatcher for sending commands
        """
        self.logger = LoggingModule.get_logger(__name__)
        self.event_bus = event_bus
        self.command_dispatcher = command_dispatcher
        
        # Initialize detector registry
        self.detectors: Dict[str, IWakeWordDetector] = {
            'porcupine': PorcupineWakeWordDetector(),
        }
        
        # State management
        self.state = DetectorState.INACTIVE
        self.is_detecting = False
        self.active_detector_name = 'porcupine'
        self.config = WakeWordConfig()
        
        # Wake word detection state
        self.wake_word_detected = False
        self.wake_word_detected_time = 0
        self.wake_word_name = ""
        self.listening_for_speech = False
        
        # Audio buffering
        self.buffer_size = int(self.config.buffer_duration * 16000 / 512)  # Assuming 16kHz, 512 samples per frame
        self.audio_buffer: Deque[AudioChunk] = deque(maxlen=self.buffer_size)
        self.last_audio_timestamp = 0.0
        
        # VAD subscription state - start disabled
        self.vad_subscribed = False
        
        # Register for audio events - only audio is needed for wake word detection
        self.event_bus.subscribe(AudioChunkCapturedEvent, self._on_audio_chunk_captured)
        
        # Do NOT subscribe to VAD events here - will be subscribed dynamically
    
    def handle(self, command: Command) -> Any:
        """
        Handle a wake word detection command and produce a result.
        
        Args:
            command: The command to handle
            
        Returns:
            The result of the command execution (type depends on command)
            
        Raises:
            TypeError: If the command is not of the expected type
            ValueError: If the command contains invalid data
            Exception: If an error occurs during command execution
        """
        if isinstance(command, ConfigureWakeWordCommand):
            return self._handle_configure_wake_word(command)
        elif isinstance(command, StartWakeWordDetectionCommand):
            return self._handle_start_wake_word_detection(command)
        elif isinstance(command, StopWakeWordDetectionCommand):
            return self._handle_stop_wake_word_detection(command)
        elif isinstance(command, DetectWakeWordCommand):
            return self._handle_detect_wake_word(command)
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
            ConfigureWakeWordCommand,
            StartWakeWordDetectionCommand,
            StopWakeWordDetectionCommand,
            DetectWakeWordCommand
        ))
    
    def _handle_configure_wake_word(self, command: ConfigureWakeWordCommand) -> bool:
        """
        Handle a ConfigureWakeWordCommand.
        
        This configures the wake word detection system parameters.
        
        Args:
            command: The ConfigureWakeWordCommand
            
        Returns:
            bool: True if configuration was successful
        """
        self.logger.info(f"Handling ConfigureWakeWordCommand with detector_type={command.config.detector_type}")
        
        # Store configuration
        self.config = command.config
        
        # Update active detector
        detector_type = command.config.detector_type
        if detector_type not in self.detectors:
            self.logger.error(f"Unknown detector type: {detector_type}")
            return False
        
        self.active_detector_name = detector_type
        
        # Get detector
        detector = self._get_detector(detector_type)
        
        # Configure the detector
        config_dict = {
            'keywords': command.config.wake_words,
            'sensitivities': command.config.sensitivities,
            'access_key': command.config.access_key,
            'keyword_paths': command.config.keyword_paths
        }
        
        # Update buffer size based on configuration
        self.buffer_size = int(command.config.buffer_duration * 16000 / 512)  # Assuming 16kHz, 512 samples per frame
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Configure detector
        success = detector.configure(config_dict)
        if not success:
            self.logger.warning(f"Failed to configure {detector_type} detector")
        
        return success
    
    def _handle_start_wake_word_detection(self, command: StartWakeWordDetectionCommand) -> bool:
        """
        Handle a StartWakeWordDetectionCommand.
        
        This starts the wake word detection process.
        
        Args:
            command: The StartWakeWordDetectionCommand
            
        Returns:
            bool: True if detection was started successfully
        """
        self.logger.info("Handling StartWakeWordDetectionCommand")
        
        # Override detector type if specified
        if command.detector_type:
            if command.detector_type not in self.detectors:
                self.logger.error(f"Unknown detector type: {command.detector_type}")
                return False
            self.active_detector_name = command.detector_type
        
        # Check if detection is already active
        if self.is_detecting:
            self.logger.warning("Wake word detection is already active")
            return True
        
        # Start detection
        self.is_detecting = True
        self.state = DetectorState.WAKE_WORD
        
        # Clear audio buffer
        self.audio_buffer.clear()
        
        # Reset wake word detection state
        self.wake_word_detected = False
        self.wake_word_detected_time = 0
        self.wake_word_name = ""
        self.listening_for_speech = False
        
        # Ensure VAD subscription and processing are disabled at start
        self._disable_vad_processing()
        
        # Disable VAD processing to save resources until wake word is detected
        self.command_dispatcher.dispatch(DisableVadProcessingCommand())
        
        # Get detector
        detector = self._get_detector(self.active_detector_name)
        
        # Get wake words
        wake_words = self.config.wake_words
        
        # Publish event
        self.event_bus.publish(WakeWordDetectionStartedEvent(
            detector_type=self.active_detector_name,
            wake_words=wake_words
        ))
        
        return True
    
    def _handle_stop_wake_word_detection(self, command: StopWakeWordDetectionCommand) -> bool:
        """
        Handle a StopWakeWordDetectionCommand.
        
        This stops the wake word detection process.
        
        Args:
            command: The StopWakeWordDetectionCommand
            
        Returns:
            bool: True if detection was stopped successfully
        """
        self.logger.info("Handling StopWakeWordDetectionCommand")
        
        # Check if detection is active
        if not self.is_detecting:
            self.logger.warning("Wake word detection is not active")
            return True
        
        # Stop detection
        self.is_detecting = False
        self.state = DetectorState.INACTIVE
        
        # Reset state
        self.wake_word_detected = False
        self.wake_word_detected_time = 0
        self.wake_word_name = ""
        self.listening_for_speech = False
        
        # Explicitly disable VAD processing
        self._disable_vad_processing()
        
        # Also disable actual VAD audio processing
        self.command_dispatcher.dispatch(DisableVadProcessingCommand())
        
        # Publish event
        self.event_bus.publish(WakeWordDetectionStoppedEvent(
            reason="user_requested",
            detector_type=self.active_detector_name
        ))
        
        return True
    
    def _handle_detect_wake_word(self, command: DetectWakeWordCommand) -> Dict[str, Any]:
        """
        Handle a DetectWakeWordCommand.
        
        This processes an audio chunk to detect wake words.
        
        Args:
            command: The DetectWakeWordCommand
            
        Returns:
            Dict[str, Any]: Detection result with keys:
                - detected: bool, True if a wake word was detected
                - wake_word: str, the detected wake word (if any)
                - confidence: float, detection confidence (if requested)
        """
        self.logger.debug("Handling DetectWakeWordCommand")
        
        # Choose detector to use
        detector_name = command.detector_type or self.active_detector_name
        detector = self._get_detector(detector_name)
        
        # Process audio with detector
        audio_data = command.audio_chunk.raw_data
        
        if command.return_confidence:
            detected, confidence, wake_word = detector.detect_with_confidence(audio_data)
            return {
                'detected': detected,
                'wake_word': wake_word,
                'confidence': confidence
            }
        else:
            detected, wake_word = detector.detect(audio_data)
            return {
                'detected': detected,
                'wake_word': wake_word
            }
    
    def _get_detector(self, detector_name: str) -> IWakeWordDetector:
        """
        Get the specified detector.
        
        Args:
            detector_name: Name of the detector to get
            
        Returns:
            IWakeWordDetector: The requested detector
            
        Raises:
            ValueError: If the detector name is not found
        """
        if detector_name not in self.detectors:
            available_detectors = ", ".join(self.detectors.keys())
            raise ValueError(f"Detector '{detector_name}' not found. Available detectors: {available_detectors}")
        
        return self.detectors[detector_name]
    
    def _enable_vad_processing(self) -> None:
        """Enable VAD processing by subscribing to VAD events."""
        if not self.vad_subscribed:
            self.event_bus.subscribe(SpeechDetectedEvent, self._on_speech_detected)
            self.event_bus.subscribe(SilenceDetectedEvent, self._on_silence_detected)
            self.vad_subscribed = True
            self.logger.info("VAD event processing enabled")
    
    def _disable_vad_processing(self) -> None:
        """Disable VAD processing by unsubscribing from VAD events."""
        if self.vad_subscribed:
            self.event_bus.unsubscribe(SpeechDetectedEvent, self._on_speech_detected)
            self.event_bus.unsubscribe(SilenceDetectedEvent, self._on_silence_detected)
            self.vad_subscribed = False
            self.logger.info("VAD event processing disabled")
    
    def _on_audio_chunk_captured(self, event: AudioChunkCapturedEvent) -> None:
        """
        Handle an audio chunk captured event.
        
        This method is called when a new audio chunk is available, and it performs
        wake word detection on the chunk if the detector is active.
        
        Args:
            event: The AudioChunkCapturedEvent
        """
        # Only process if we're actively detecting
        if not self.is_detecting:
            return
        
        # Process audio chunk for wake word detection
        audio_chunk = event.audio_chunk
        self.last_audio_timestamp = audio_chunk.timestamp
        
        # Add to buffer
        self.audio_buffer.append(audio_chunk)
        
        # Check for wake word timeout only when waiting for speech to begin
        # but not when actively recording speech
        if self.wake_word_detected and self.listening_for_speech and self.state == DetectorState.LISTENING:
            timeout_duration = self.config.speech_timeout
            current_time = time.time()
            
            if current_time - self.wake_word_detected_time > timeout_duration:
                # Timeout occurred
                self.logger.info(f"Wake word timeout after {timeout_duration}s without speech")
                
                # Publish timeout event
                self.event_bus.publish(WakeWordTimeoutEvent(
                    wake_word=self.wake_word_name,
                    timeout_duration=timeout_duration
                ))
                
                # Reset state
                self.wake_word_detected = False
                self.wake_word_detected_time = 0
                self.wake_word_name = ""
                self.listening_for_speech = False
                
                # Disable VAD processing on timeout
                self._disable_vad_processing()
                self.command_dispatcher.dispatch(DisableVadProcessingCommand())
                
                # Return to wake word detection state
                self.state = DetectorState.WAKE_WORD
        
        # Detect wake words if in wake word state
        if self.state == DetectorState.WAKE_WORD and not self.wake_word_detected:
            try:
                # Use command for wake word detection
                command = DetectWakeWordCommand(
                    audio_chunk=audio_chunk,
                    detector_type=self.active_detector_name,
                    return_confidence=True
                )
                result = self.handle(command)
                
                if result['detected']:
                    # Wake word detected - handle it
                    self._on_wake_word_detected(
                        result['wake_word'],
                        result['confidence']
                    )
            except Exception as e:
                self.logger.error(f"Error processing audio chunk for wake word detection: {e}")
    
    def _on_wake_word_detected(self, wake_word: str, confidence: float) -> None:
        """
        Handle wake word detection.
        
        This method is called when a wake word is detected. It publishes a
        WakeWordDetectedEvent and starts listening for speech.
        
        Args:
            wake_word: The detected wake word
            confidence: The confidence of the detection
        """
        self.logger.info(f"Wake word detected: {wake_word} (confidence: {confidence:.2f})")
        
        # Update state
        self.wake_word_detected = True
        self.wake_word_detected_time = time.time()
        self.wake_word_name = wake_word
        self.listening_for_speech = True
        self.state = DetectorState.LISTENING
        
        # Publish wake word detected event
        self.event_bus.publish(WakeWordDetectedEvent(
            wake_word=wake_word,
            confidence=confidence,
            audio_timestamp=self.last_audio_timestamp,
            detector_type=self.active_detector_name,
            audio_reference=self._get_buffered_audio()
        ))
        
        # If configured, clear VAD's pre-speech buffer to exclude wake word audio
        if self.config.exclude_pre_wake_word_audio:
            self.logger.info("Wake word detected, dispatching command to clear VAD pre-speech buffer.")
            self.command_dispatcher.dispatch(ClearVadPreSpeechBufferCommand())
        else:
            self.logger.info("Wake word detected, VAD pre-speech buffer will be retained as per configuration.")
            
        # Start listening for speech with VAD
        self.command_dispatcher.dispatch(
            ConfigureVadCommand(
                detector_type="combined",
                sensitivity=0.7
            )
        )
        
        # Enable VAD processing only after wake word detection
        self._enable_vad_processing()
        
        # Also enable actual VAD audio processing
        self.command_dispatcher.dispatch(EnableVadProcessingCommand())
    
    def _on_speech_detected(self, event: SpeechDetectedEvent) -> None:
        """
        Handle speech detection event.
        
        This method is called when speech is detected after a wake word.
        
        Args:
            event: The SpeechDetectedEvent
        """
        self.logger.info("Speech detected after wake word")
        
        # Update state
        self.state = DetectorState.RECORDING
    
    def _on_silence_detected(self, event: SilenceDetectedEvent) -> None:
        """
        Handle silence detection event.
        
        This method is called when silence is detected after speech.
        
        Args:
            event: The SilenceDetectedEvent
        """
        self.logger.info(f"Silence detected, speech duration: {event.speech_duration:.2f}s")
        
        # Only handle if we are currently recording
        if self.state == DetectorState.RECORDING:
            # Update state
            self.state = DetectorState.PROCESSING
            
            # Reset wake word detection state
            self.wake_word_detected = False
            self.wake_word_detected_time = 0
            self.wake_word_name = ""
            self.listening_for_speech = False
            
            # Disable VAD processing after speech is processed
            self._disable_vad_processing()
            
            # Also disable actual VAD audio processing to save resources
            self.command_dispatcher.dispatch(DisableVadProcessingCommand())
            
            # Return to wake word detection state after processing
            self.state = DetectorState.WAKE_WORD
    
    def _get_buffered_audio(self) -> Optional[List[AudioChunk]]:
        """
        Get the buffered audio.
        
        Returns:
            Optional[List[AudioChunk]]: List of audio chunks in the buffer
        """
        if not self.audio_buffer:
            return None
        
        return list(self.audio_buffer)
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the handler.
        
        This should be called when the handler is no longer needed.
        """
        # Unsubscribe from events
        self.event_bus.unsubscribe(AudioChunkCapturedEvent, self._on_audio_chunk_captured)
        
        # Explicitly disable VAD processing
        self._disable_vad_processing()
        
        # Also disable actual VAD audio processing
        self.command_dispatcher.dispatch(DisableVadProcessingCommand())
        
        # Stop detection
        if self.is_detecting:
            self.is_detecting = False
            self.state = DetectorState.INACTIVE
        
        # Clean up detectors
        for detector in self.detectors.values():
            detector.cleanup()
        
        # Clear all buffers to free memory
        self.audio_buffer.clear()
        
        # Remove detector references
        self.detectors.clear()
        
        # Log cleanup
        self.logger.info("WakeWordCommandHandler resources cleaned up")