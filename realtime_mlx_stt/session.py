"""
Session-based API that follows the server's pattern.

This provides a more robust API that properly manages state and follows
the same sequencing as the server example.
"""

import os
import sys
import time
import uuid
import threading
from typing import Optional, Dict, Any, Callable
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Infrastructure.Logging.LoggingModule import LoggingModule

# Import all necessary modules
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule
from src.Features.VoiceActivityDetection.VadModule import VadModule
from src.Features.Transcription.TranscriptionModule import TranscriptionModule
from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule

# Import commands
from src.Features.AudioCapture.Commands.StartRecordingCommand import StartRecordingCommand
from src.Features.AudioCapture.Commands.StopRecordingCommand import StopRecordingCommand
from src.Features.VoiceActivityDetection.Commands.ConfigureVadCommand import ConfigureVadCommand
from src.Features.VoiceActivityDetection.Commands.EnableVadProcessingCommand import EnableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.DisableVadProcessingCommand import DisableVadProcessingCommand
from src.Features.Transcription.Commands.ConfigureTranscriptionCommand import ConfigureTranscriptionCommand
from src.Features.Transcription.Commands.StartTranscriptionSessionCommand import StartTranscriptionSessionCommand
from src.Features.Transcription.Commands.StopTranscriptionSessionCommand import StopTranscriptionSessionCommand
from src.Features.WakeWordDetection.Commands.ConfigureWakeWordCommand import ConfigureWakeWordCommand
from src.Features.WakeWordDetection.Commands.StartWakeWordDetectionCommand import StartWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.StopWakeWordDetectionCommand import StopWakeWordDetectionCommand

# Import events
from src.Features.Transcription.Events.TranscriptionUpdatedEvent import TranscriptionUpdatedEvent
from src.Features.VoiceActivityDetection.Events.SpeechDetectedEvent import SpeechDetectedEvent
from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent

# Import internal configs
from src.Features.WakeWordDetection.Models.WakeWordConfig import WakeWordConfig as InternalWakeWordConfig

from .config import ModelConfig, VADConfig, WakeWordConfig
from .types import TranscriptionResult
from .utils import setup_minimal_logging


class SessionState(Enum):
    """Session states."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class TranscriptionSession:
    """
    Session-based transcription that follows the server pattern.
    
    This provides proper state management and sequencing like the server example.
    """
    
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        vad: Optional[VADConfig] = None,
        wake_word: Optional[WakeWordConfig] = None,
        device_id: Optional[int] = None,
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None,
        on_speech_start: Optional[Callable[[float], None]] = None,
        on_speech_end: Optional[Callable[[float], None]] = None,
        on_wake_word: Optional[Callable[[str, float], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        verbose: bool = False
    ):
        """
        Initialize a transcription session.
        
        Args:
            model: Model configuration
            vad: VAD configuration  
            wake_word: Wake word configuration
            device_id: Audio device ID
            on_transcription: Called with final transcriptions
            on_speech_start: Called when speech is detected
            on_speech_end: Called when speech ends
            on_wake_word: Called when wake word is detected
            on_error: Called on errors
            verbose: Enable verbose logging
        """
        # Store configurations
        self.model_config = model or ModelConfig()
        self.vad_config = vad or VADConfig()
        self.wake_word_config = wake_word
        self.device_id = device_id
        
        # Callbacks
        self.on_transcription = on_transcription
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_wake_word = on_wake_word
        self.on_error = on_error
        
        # Setup logging
        if not verbose:
            setup_minimal_logging()
        self.logger = LoggingModule.get_logger(__name__)
        
        # Core components
        self.command_dispatcher = CommandDispatcher()
        self.event_bus = EventBus()
        
        # State management
        self.state = SessionState.IDLE
        self.session_id = None
        self.active_features = []
        
        # Initialize modules
        self._initialize_modules()
        
    def _initialize_modules(self):
        """Initialize all required modules."""
        # Register core modules
        AudioCaptureModule.register(self.command_dispatcher, self.event_bus)
        VadModule.register(self.command_dispatcher, self.event_bus, processing_enabled=False)
        TranscriptionModule.register(self.command_dispatcher, self.event_bus)
        
        # Register VAD integration
        TranscriptionModule.register_vad_integration(
            event_bus=self.event_bus,
            transcription_handler=TranscriptionModule.register(self.command_dispatcher, self.event_bus),
            auto_start_on_speech=True
        )
        
        # Register wake word if configured
        if self.wake_word_config:
            WakeWordModule.register(self.command_dispatcher, self.event_bus)
        
        # Setup event handlers
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Setup event handlers for callbacks."""
        # Transcription events
        if self.on_transcription:
            def handle_transcription(session_id, text, is_final, confidence):
                if is_final and text.strip():
                    result = TranscriptionResult(
                        text=text,
                        confidence=confidence,
                        timestamp=time.time()
                    )
                    try:
                        self.on_transcription(result)
                    except Exception as e:
                        self._handle_error(e)
            
            TranscriptionModule.on_transcription_updated(self.event_bus, handle_transcription)
        
        # Speech detection events
        if self.on_speech_start:
            def handle_speech_start(confidence, timestamp, speech_id):
                try:
                    self.on_speech_start(timestamp)
                except Exception as e:
                    self._handle_error(e)
            
            VadModule.on_speech_detected(self.event_bus, handle_speech_start)
        
        if self.on_speech_end:
            def handle_speech_end(duration, start_time, end_time, speech_id):
                try:
                    self.on_speech_end(end_time)
                except Exception as e:
                    self._handle_error(e)
            
            VadModule.on_silence_detected(self.event_bus, handle_speech_end)
        
        # Wake word events
        if self.wake_word_config and self.on_wake_word:
            def handle_wake_word(event: WakeWordDetectedEvent):
                try:
                    self.on_wake_word(event.wake_word, event.confidence)
                except Exception as e:
                    self._handle_error(e)
            
            self.event_bus.subscribe(WakeWordDetectedEvent, handle_wake_word)
    
    def start(self) -> bool:
        """
        Start the transcription session.
        
        Returns:
            bool: True if started successfully
        """
        if self.state != SessionState.IDLE:
            self.logger.warning(f"Cannot start session in state: {self.state}")
            return False
        
        try:
            self.state = SessionState.STARTING
            self.logger.info("Starting transcription session...")
            
            # 1. Configure transcription
            self._configure_transcription()
            
            # 2. Configure VAD
            self._configure_vad()
            
            # 3. Configure wake word if enabled
            if self.wake_word_config:
                self._configure_wake_word()
            
            # 4. Start audio recording
            self.logger.info("Starting audio recording...")
            self.command_dispatcher.dispatch(StartRecordingCommand(
                device_id=self.device_id,
                sample_rate=16000,
                chunk_size=512
            ))
            
            # 5. Enable VAD processing (unless using wake word)
            if self.vad_config.enabled and not self.wake_word_config:
                self.logger.info("Enabling VAD processing...")
                self.command_dispatcher.dispatch(EnableVadProcessingCommand())
                self.active_features.append("vad")
            
            # 6. Start transcription session
            self.session_id = str(uuid.uuid4())
            self.logger.info(f"Starting transcription session: {self.session_id}")
            self.command_dispatcher.dispatch(StartTranscriptionSessionCommand(
                session_id=self.session_id
            ))
            self.active_features.append("transcription")
            
            # 7. Start wake word detection if enabled
            if self.wake_word_config:
                self.logger.info("Starting wake word detection...")
                self.command_dispatcher.dispatch(StartWakeWordDetectionCommand())
                self.active_features.append("wake_word")
            
            self.state = SessionState.RUNNING
            self.logger.info("Session started successfully")
            return True
            
        except Exception as e:
            self.state = SessionState.ERROR
            self._handle_error(e)
            return False
    
    def stop(self) -> bool:
        """
        Stop the transcription session.
        
        Returns:
            bool: True if stopped successfully
        """
        if self.state != SessionState.RUNNING:
            self.logger.warning(f"Cannot stop session in state: {self.state}")
            return False
        
        try:
            self.state = SessionState.STOPPING
            self.logger.info("Stopping transcription session...")
            
            # Stop in reverse order
            
            # 1. Stop wake word detection
            if "wake_word" in self.active_features:
                self.logger.info("Stopping wake word detection...")
                self.command_dispatcher.dispatch(StopWakeWordDetectionCommand())
            
            # 2. Stop transcription session
            if "transcription" in self.active_features and self.session_id:
                self.logger.info("Stopping transcription session...")
                self.command_dispatcher.dispatch(StopTranscriptionSessionCommand(
                    session_id=self.session_id
                ))
            
            # 3. Disable VAD processing
            if "vad" in self.active_features:
                self.logger.info("Disabling VAD processing...")
                self.command_dispatcher.dispatch(DisableVadProcessingCommand())
            
            # 4. Stop audio recording
            self.logger.info("Stopping audio recording...")
            self.command_dispatcher.dispatch(StopRecordingCommand())
            
            # Clear state
            self.state = SessionState.IDLE
            self.session_id = None
            self.active_features = []
            
            self.logger.info("Session stopped successfully")
            return True
            
        except Exception as e:
            self.state = SessionState.ERROR
            self._handle_error(e)
            return False
    
    def _configure_transcription(self):
        """Configure the transcription engine."""
        self.logger.info("Configuring transcription...")
        engine = self.model_config.engine.value if hasattr(self.model_config.engine, 'value') else self.model_config.engine
        
        self.command_dispatcher.dispatch(ConfigureTranscriptionCommand(
            engine_type=engine,
            model_name=self.model_config.model,
            language=self.model_config.language,
            options={}
        ))
    
    def _configure_vad(self):
        """Configure VAD settings."""
        if not self.vad_config.enabled:
            return
            
        self.logger.info("Configuring VAD...")
        config_dict = self.vad_config.to_dict()
        
        self.command_dispatcher.dispatch(ConfigureVadCommand(
            detector_type=config_dict.get("detector_type", "combined"),
            sensitivity=config_dict.get("sensitivity", 0.6),
            window_size=config_dict.get("window_size", 5),
            min_speech_duration=config_dict.get("min_speech_duration", 0.25),
            parameters=config_dict.get("parameters", {})
        ))
    
    def _configure_wake_word(self):
        """Configure wake word detection."""
        if not self.wake_word_config:
            return
            
        self.logger.info("Configuring wake word detection...")
        
        # Get access key
        access_key = self.wake_word_config.access_key or os.environ.get('PORCUPINE_ACCESS_KEY')
        if not access_key:
            raise ValueError(
                "Porcupine access key required. Set PORCUPINE_ACCESS_KEY or pass access_key."
            )
        
        # Create internal config
        internal_config = InternalWakeWordConfig(
            wake_words=self.wake_word_config.words,
            sensitivities=[self.wake_word_config.sensitivity] * len(self.wake_word_config.words),
            speech_timeout=float(self.wake_word_config.timeout),
            access_key=access_key
        )
        
        self.command_dispatcher.dispatch(ConfigureWakeWordCommand(
            config=internal_config
        ))
    
    def _handle_error(self, error: Exception):
        """Handle errors."""
        self.logger.error(f"Session error: {error}")
        if self.on_error:
            self.on_error(error)
    
    def get_state(self) -> SessionState:
        """Get current session state."""
        return self.state
    
    def is_running(self) -> bool:
        """Check if session is running."""
        return self.state == SessionState.RUNNING
    
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()