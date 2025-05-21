"""
VadModule for registering and providing access to voice activity detection functionality.

This module serves as the main entry point for the VoiceActivityDetection feature,
handling registration of commands, events, and the VAD handler.
"""

import logging
from typing import Dict, Any, Optional, Union, Callable

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import IEventBus
from src.Features.VoiceActivityDetection.Commands.DetectVoiceActivityCommand import DetectVoiceActivityCommand
from src.Features.VoiceActivityDetection.Commands.ConfigureVadCommand import ConfigureVadCommand
from src.Features.VoiceActivityDetection.Commands.EnableVadProcessingCommand import EnableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.DisableVadProcessingCommand import DisableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.ClearVadPreSpeechBufferCommand import ClearVadPreSpeechBufferCommand
from src.Features.VoiceActivityDetection.Events.SpeechDetectedEvent import SpeechDetectedEvent
from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
from src.Features.VoiceActivityDetection.Handlers.VoiceActivityHandler import VoiceActivityHandler
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk


class VadModule:
    """
    Module for voice activity detection functionality.
    
    This class provides registration and convenience methods for the VoiceActivityDetection feature.
    It serves as a facade for the underlying components (detectors, handler, etc.).
    """
    
    @staticmethod
    def register(command_dispatcher: CommandDispatcher,
                event_bus: IEventBus,
                default_detector: str = "combined",  # Changed from "webrtc" to "combined" to use the two-stage approach
                default_sensitivity: float = 0.7,  # Increased from 0.5 for more conservative detection
                processing_enabled: bool = False) -> VoiceActivityHandler:  # Default to disabled for resource efficiency
        """
        Register the VoiceActivityDetection feature with the system.
        
        This method:
        1. Creates the VAD handler
        2. Registers it with the command dispatcher
        3. Configures the default detector
        
        Args:
            command_dispatcher: The command dispatcher to register handlers with
            event_bus: The event bus for publishing/subscribing to events
            default_detector: The default detector to use ('webrtc', 'silero', or 'combined')
            default_sensitivity: Default sensitivity level (0.0-1.0)
            
        Returns:
            VoiceActivityHandler: The registered command handler
        """
        logger = logging.getLogger(__name__)
        logger.info("Registering VoiceActivityDetection feature")
        
        # Create the handler
        handler = VoiceActivityHandler(event_bus=event_bus)
        
        # Register with command dispatcher
        command_dispatcher.register_handler(DetectVoiceActivityCommand, handler)
        command_dispatcher.register_handler(ConfigureVadCommand, handler)
        command_dispatcher.register_handler(EnableVadProcessingCommand, handler)
        command_dispatcher.register_handler(DisableVadProcessingCommand, handler)
        command_dispatcher.register_handler(ClearVadPreSpeechBufferCommand, handler)
        
        # Configure default detector
        try:
            config_command = ConfigureVadCommand(
                detector_type=default_detector,
                sensitivity=default_sensitivity
            )
            result = command_dispatcher.dispatch(config_command)
            if result:
                logger.info(f"Successfully configured {default_detector} detector with sensitivity {default_sensitivity}")
            else:
                logger.warning(f"Failed to configure {default_detector} detector")
            
            # Set initial processing state
            if processing_enabled:
                command_dispatcher.dispatch(EnableVadProcessingCommand())
                logger.info("VAD processing initially enabled")
            else:
                command_dispatcher.dispatch(DisableVadProcessingCommand())
                logger.info("VAD processing initially disabled")
                
        except Exception as e:
            logger.error(f"Error configuring default detector: {e}")
        
        return handler
    
    @staticmethod
    def detect_voice_activity(command_dispatcher: CommandDispatcher,
                             audio_chunk: AudioChunk,
                             sensitivity: Optional[float] = None,
                             detector_type: Optional[str] = None,
                             return_confidence: bool = False) -> Union[bool, Dict[str, Any]]:
        """
        Detect voice activity in an audio chunk.
        
        Args:
            command_dispatcher: The command dispatcher to use
            audio_chunk: The audio chunk to analyze
            sensitivity: Optional sensitivity adjustment
            detector_type: Optional detector type to use
            return_confidence: Whether to return confidence score
            
        Returns:
            Union[bool, Dict[str, Any]]: Boolean result or dict with result and confidence
        """
        command = DetectVoiceActivityCommand(
            audio_chunk=audio_chunk,
            sensitivity=sensitivity,
            detector_type=detector_type,
            return_confidence=return_confidence
        )
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def configure_vad(command_dispatcher: CommandDispatcher,
                     detector_type: str = "combined",  # Changed from "webrtc" to "combined" for two-stage detection
                     sensitivity: float = 0.7,  # Increased from 0.5 for more conservative detection
                     window_size: int = 5,
                     min_speech_duration: float = 0.25,
                     **kwargs) -> bool:
        """
        Configure the voice activity detection system.
        
        Args:
            command_dispatcher: The command dispatcher to use
            detector_type: Type of VAD detector to use
            sensitivity: Sensitivity level (0.0-1.0)
                For 'combined' detector, this affects the Silero threshold
                while WebRTC threshold is fixed at 0.6 for better sensitivity
            window_size: Number of frames to consider
            min_speech_duration: Minimum speech segment duration in seconds
            **kwargs: Additional detector-specific parameters
                For direct control, use webrtc_threshold and silero_threshold
            
        Returns:
            bool: True if configuration was successful
        """
        command = ConfigureVadCommand(
            detector_type=detector_type,
            sensitivity=sensitivity,
            window_size=window_size,
            min_speech_duration=min_speech_duration,
            parameters=kwargs
        )
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def on_speech_detected(event_bus: IEventBus,
                          handler: Callable[[float, float, str], None]) -> None:
        """
        Subscribe to speech detected events.
        
        Args:
            event_bus: The event bus to subscribe to
            handler: Function to call when speech is detected
                    Function receives (confidence, timestamp, speech_id)
        """
        event_bus.subscribe(SpeechDetectedEvent, 
                           lambda event: handler(
                               event.confidence,
                               event.audio_timestamp,
                               event.speech_id
                           ))
    
    @staticmethod
    def on_silence_detected(event_bus: IEventBus,
                           handler: Callable[[float, float, float, str], None]) -> None:
        """
        Subscribe to silence detected events.
        
        Args:
            event_bus: The event bus to subscribe to
            handler: Function to call when silence is detected after speech
                    Function receives (speech_duration, start_time, end_time, speech_id)
        """
        event_bus.subscribe(SilenceDetectedEvent,
                           lambda event: handler(
                               event.speech_duration,
                               event.speech_start_time,
                               event.speech_end_time,
                               event.speech_id
                           ))
    
    @staticmethod
    def enable_processing(command_dispatcher: CommandDispatcher) -> bool:
        """
        Enable active processing of audio chunks for voice activity detection.
        
        When enabled, the VAD system actively processes each audio chunk through
        detection algorithms, allowing for speech detection but consuming more CPU.
        
        Args:
            command_dispatcher: The command dispatcher to use
            
        Returns:
            bool: True if the command was successfully dispatched
        """
        command = EnableVadProcessingCommand()
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def disable_processing(command_dispatcher: CommandDispatcher) -> bool:
        """
        Disable active processing of audio chunks for voice activity detection.
        
        When disabled, the VAD system will still receive audio chunks but will skip
        applying detection algorithms, significantly reducing CPU usage during idle periods.
        
        Args:
            command_dispatcher: The command dispatcher to use
            
        Returns:
            bool: True if the command was successfully dispatched
        """
        command = DisableVadProcessingCommand()
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def on_speech_audio(event_bus: IEventBus,
                       speech_handler: Callable[[Any, str], None],
                       silence_handler: Callable[[Any, str, float], None]) -> None:
        """
        Subscribe to both speech and silence events to get the complete audio.
        
        Args:
            event_bus: The event bus to subscribe to
            speech_handler: Function to call when speech begins
                          Function receives (audio_ref, speech_id)
            silence_handler: Function to call when speech ends
                           Function receives (audio_ref, speech_id, duration)
        """
        event_bus.subscribe(SpeechDetectedEvent, 
                           lambda event: speech_handler(
                               event.audio_reference,
                               event.speech_id
                           ))
        
        event_bus.subscribe(SilenceDetectedEvent,
                           lambda event: silence_handler(
                               event.audio_reference,
                               event.speech_id,
                               event.speech_duration
                           ))