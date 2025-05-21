"""
WakeWordModule for registering and providing access to wake word detection functionality.

This module serves as the main entry point for the WakeWordDetection feature,
handling registration of commands, events, and the wake word handler.
"""

import os
from typing import Dict, Any, Optional, List, Union, Callable

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import IEventBus
from src.Features.WakeWordDetection.Commands.ConfigureWakeWordCommand import ConfigureWakeWordCommand
from src.Features.WakeWordDetection.Commands.StartWakeWordDetectionCommand import StartWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.StopWakeWordDetectionCommand import StopWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.DetectWakeWordCommand import DetectWakeWordCommand
from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
from src.Features.WakeWordDetection.Events.WakeWordTimeoutEvent import WakeWordTimeoutEvent
from src.Features.WakeWordDetection.Handlers.WakeWordCommandHandler import WakeWordCommandHandler
from src.Features.WakeWordDetection.Models.WakeWordConfig import WakeWordConfig


class WakeWordModule:
    """
    Module for wake word detection functionality.
    
    This class provides registration and convenience methods for the WakeWordDetection feature.
    It serves as a facade for the underlying components (detectors, handler, etc.).
    """
    
    @staticmethod
    def register(command_dispatcher: CommandDispatcher,
                event_bus: IEventBus,
                detector_type: str = "porcupine",
                wake_words: Optional[List[str]] = None,
                sensitivities: Optional[List[float]] = None,
                access_key: Optional[str] = None) -> WakeWordCommandHandler:
        """
        Register the WakeWordDetection feature with the system.
        
        This method:
        1. Creates the wake word handler
        2. Registers it with the command dispatcher
        3. Configures the default detector
        
        Args:
            command_dispatcher: The command dispatcher to register handlers with
            event_bus: The event bus for publishing/subscribing to events
            detector_type: The default detector to use
            wake_words: Default wake words to detect
            sensitivities: Default sensitivities for each wake word
            access_key: Access key for Porcupine (optional, can use env var)
            
        Returns:
            WakeWordCommandHandler: The registered command handler
        """
        logger = LoggingModule.get_logger(__name__)
        logger.info("Registering WakeWordDetection feature")
        
        # Read PORCUPINE_ACCESS_KEY from environment if not provided
        if not access_key:
            access_key = os.environ.get("PORCUPINE_ACCESS_KEY")
            if access_key:
                logger.info("Using PORCUPINE_ACCESS_KEY from environment")
        
        # Create handler
        handler = WakeWordCommandHandler(event_bus=event_bus, command_dispatcher=command_dispatcher)
        
        # Register with command dispatcher
        command_dispatcher.register_handler(ConfigureWakeWordCommand, handler)
        command_dispatcher.register_handler(StartWakeWordDetectionCommand, handler)
        command_dispatcher.register_handler(StopWakeWordDetectionCommand, handler)
        command_dispatcher.register_handler(DetectWakeWordCommand, handler)
        
        # Configure default detector
        try:
            config = WakeWordConfig(
                detector_type=detector_type,
                wake_words=wake_words or ["porcupine"],
                sensitivities=sensitivities,
                access_key=access_key
            )
            
            config_command = ConfigureWakeWordCommand(config=config)
            result = command_dispatcher.dispatch(config_command)
            
            if result:
                logger.info(f"Successfully configured {detector_type} detector with {len(config.wake_words)} wake words")
            else:
                logger.warning(f"Failed to configure {detector_type} detector")
        except Exception as e:
            logger.error(f"Error configuring default detector: {e}")
        
        return handler
    
    @staticmethod
    def configure(command_dispatcher: CommandDispatcher,
                 config: Union[WakeWordConfig, Dict[str, Any]]) -> bool:
        """
        Configure the wake word detection system.
        
        Args:
            command_dispatcher: The command dispatcher to use
            config: Configuration for wake word detection (either a WakeWordConfig object or a dict)
            
        Returns:
            bool: True if configuration was successful
        """
        if isinstance(config, dict):
            config_obj = WakeWordConfig(**config)
        else:
            config_obj = config
        
        command = ConfigureWakeWordCommand(config=config_obj)
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def start_detection(command_dispatcher: CommandDispatcher,
                       detector_type: Optional[str] = None) -> bool:
        """
        Start wake word detection.
        
        Args:
            command_dispatcher: The command dispatcher to use
            detector_type: Optional detector type override
            
        Returns:
            bool: True if detection was started successfully
        """
        command = StartWakeWordDetectionCommand(detector_type=detector_type)
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def stop_detection(command_dispatcher: CommandDispatcher) -> bool:
        """
        Stop wake word detection.
        
        Args:
            command_dispatcher: The command dispatcher to use
            
        Returns:
            bool: True if detection was stopped successfully
        """
        command = StopWakeWordDetectionCommand()
        return command_dispatcher.dispatch(command)
    
    @staticmethod
    def on_wake_word_detected(event_bus: IEventBus,
                             handler: Callable[[str, float, float], None]) -> None:
        """
        Subscribe to wake word detection events.
        
        Args:
            event_bus: The event bus to subscribe to
            handler: Function to call when a wake word is detected
                    Function receives (wake_word, confidence, timestamp)
        """
        event_bus.subscribe(WakeWordDetectedEvent, 
                           lambda event: handler(
                               event.wake_word,
                               event.confidence,
                               event.audio_timestamp
                           ))
    
    @staticmethod
    def on_wake_word_timeout(event_bus: IEventBus,
                            handler: Callable[[str, float], None]) -> None:
        """
        Subscribe to wake word timeout events.
        
        Args:
            event_bus: The event bus to subscribe to
            handler: Function to call when a wake word times out
                    Function receives (wake_word, timeout_duration)
        """
        event_bus.subscribe(WakeWordTimeoutEvent,
                           lambda event: handler(
                               event.wake_word,
                               event.timeout_duration
                           ))