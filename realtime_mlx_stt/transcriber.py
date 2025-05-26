"""
High-level transcription API for Realtime MLX STT.

This module provides a simple, user-friendly interface for speech-to-text
transcription without requiring knowledge of the underlying architecture.
"""

import os
import sys
import time
import threading
from typing import Optional, List, Union, Callable
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule
from src.Features.VoiceActivityDetection.VadModule import VadModule
from src.Features.Transcription.TranscriptionModule import TranscriptionModule
from src.Features.AudioCapture.Commands.StartRecordingCommand import StartRecordingCommand
from src.Features.AudioCapture.Commands.StopRecordingCommand import StopRecordingCommand
from src.Features.Transcription.Commands.TranscribeAudioCommand import TranscribeAudioCommand
from src.Features.Transcription.Events.TranscriptionUpdatedEvent import TranscriptionUpdatedEvent
from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
from src.Infrastructure.Logging.LoggingModule import LoggingModule

from .types import (
    AudioSource, TranscriptionEngine, VADMode, 
    TranscriptionResult, AudioDevice,
    TranscriptionCallback, ErrorCallback
)
from .utils import setup_minimal_logging, list_audio_devices
from .config import ModelConfig, VADConfig, WakeWordConfig, TranscriberConfig


class Transcriber:
    """
    Simple, high-level transcription interface.
    
    Examples:
        # Basic usage
        transcriber = Transcriber()
        text = transcriber.transcribe_from_mic(duration=5)
        print(f"You said: {text}")
        
        # Continuous transcription with callback
        def on_transcription(result):
            print(f"Transcribed: {result.text}")
        
        transcriber = Transcriber(on_transcription=on_transcription)
        transcriber.start_continuous()
        time.sleep(60)  # Listen for 60 seconds
        transcriber.stop()
        
        # With VAD (Voice Activity Detection)
        transcriber = Transcriber(vad_mode=VADMode.SILERO)
        transcriber.start_continuous()  # Will only transcribe when speech is detected
    """
    
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        vad: Optional[VADConfig] = None,
        wake_word: Optional[WakeWordConfig] = None,
        device_index: Optional[int] = None,
        on_transcription: Optional[TranscriptionCallback] = None,
        on_error: Optional[ErrorCallback] = None,
        on_wake_word: Optional[Callable[[str, float], None]] = None,
        verbose: bool = False,
        # Legacy parameters for backward compatibility
        engine: Optional[TranscriptionEngine] = None,
        vad_mode: Optional[VADMode] = None,
        language: Optional[str] = None,
    ):
        """
        Initialize the transcriber.
        
        Args:
            model: Model configuration (ModelConfig object)
            vad: VAD configuration (VADConfig object)
            wake_word: Wake word configuration (WakeWordConfig object)
            device_index: Audio device index or None for default
            on_transcription: Callback for transcription results
            on_error: Callback for errors
            on_wake_word: Callback for wake word detection (word, confidence)
            verbose: Enable verbose logging
            
        Legacy args (for backward compatibility):
            engine: Which transcription engine to use
            vad_mode: Voice activity detection mode
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            
        Examples:
            # Using configuration objects (recommended)
            transcriber = Transcriber(
                model=ModelConfig(engine="openai", model="gpt-4o-transcribe"),
                vad=VADConfig(sensitivity=0.8),
                on_transcription=lambda r: print(r.text)
            )
            
            # Legacy style (still supported)
            transcriber = Transcriber(
                engine=TranscriptionEngine.MLX_WHISPER,
                vad_mode=VADMode.COMBINED,
                language="en"
            )
        """
        # Handle legacy parameters
        if model is None and engine is not None:
            model = ModelConfig(
                engine=engine,
                language=language
            )
        elif model is None:
            model = ModelConfig()
        
        if vad is None and vad_mode is not None:
            # Map VADMode enum to VADDetectorType
            detector_map = {
                VADMode.SILERO: "silero",
                VADMode.WEBRTC: "webrtc", 
                VADMode.COMBINED: "combined",
                VADMode.DISABLED: None
            }
            if vad_mode != VADMode.DISABLED:
                vad = VADConfig(
                    enabled=True,
                    detector_type=detector_map.get(vad_mode, "combined")
                )
            else:
                vad = VADConfig(enabled=False)
        elif vad is None:
            vad = VADConfig()
        
        # Store configurations
        self.model_config = model
        self.vad_config = vad
        self.wake_word_config = wake_word
        self.device_index = device_index
        self.on_transcription = on_transcription
        self.on_error = on_error
        self.on_wake_word = on_wake_word
        self.verbose = verbose
        
        # Setup logging
        if not verbose:
            setup_minimal_logging()
        
        self.logger = LoggingModule.get_logger(__name__)
        
        # Initialize components
        self._setup_components()
        self._is_running = False
        self._transcription_thread = None
        
    def _setup_components(self):
        """Initialize the underlying components."""
        # Create core components
        self.command_dispatcher = CommandDispatcher()
        self.event_bus = EventBus()
        
        # Initialize modules
        self.audio_module = AudioCaptureModule()
        self.transcription_module = TranscriptionModule()
        
        # Register modules
        AudioCaptureModule.register(self.command_dispatcher, self.event_bus)
        self.transcription_handler = TranscriptionModule.register(self.command_dispatcher, self.event_bus)
        
        # Initialize VAD if enabled
        if self.vad_config.enabled:
            # Register VAD with processing enabled
            VadModule.register(
                self.command_dispatcher, 
                self.event_bus, 
                processing_enabled=True
            )
            
            # Configure VAD with our settings
            self._configure_vad()
            
            # Set up VAD-triggered transcription
            TranscriptionModule.register_vad_integration(
                event_bus=self.event_bus,
                transcription_handler=self.transcription_handler,
                auto_start_on_speech=True
            )
        
        # Initialize wake word if configured
        if self.wake_word_config:
            self._setup_wake_word()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Configure transcription model
        self._configure_model()
    
    def _setup_event_handlers(self):
        """Setup event handlers for callbacks."""
        if self.on_transcription:
            def on_transcription_updated(session_id, text, is_final, confidence):
                if is_final and text.strip():  # Only process final transcriptions with text
                    result = TranscriptionResult(
                        text=text,
                        confidence=confidence,
                        timestamp=time.time(),
                        duration=None,  # Could be calculated if needed
                        language=None   # Could be extracted if needed
                    )
                    try:
                        self.on_transcription(result)
                    except Exception as e:
                        self.logger.error(f"Error in transcription callback: {e}")
                        if self.on_error:
                            self.on_error(e)
            
            # Use the static method to register the handler
            TranscriptionModule.on_transcription_updated(
                self.event_bus, 
                on_transcription_updated
            )
    
    def _configure_vad(self):
        """Configure VAD with our settings."""
        config_dict = self.vad_config.to_dict()
        # VadModule.configure_vad expects flat parameters
        VadModule.configure_vad(
            command_dispatcher=self.command_dispatcher,
            detector_type=config_dict.get("detector_type", "combined"),
            sensitivity=config_dict.get("sensitivity", 0.6),
            min_speech_duration=config_dict.get("min_speech_duration", 0.25),
            min_silence_duration=config_dict.get("min_silence_duration", 0.1)
        )
    
    def _configure_model(self):
        """Configure the transcription model."""
        TranscriptionModule.configure(
            command_dispatcher=self.command_dispatcher,
            engine_type=self.model_config.engine.value if hasattr(self.model_config.engine, 'value') else self.model_config.engine,
            model_name=self.model_config.model,
            language=self.model_config.language
        )
    
    def _setup_wake_word(self):
        """Setup wake word detection if configured."""
        if not self.wake_word_config:
            return
            
        # Import here to avoid circular imports
        from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule
        from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
        from src.Features.WakeWordDetection.Models.WakeWordConfig import WakeWordConfig as InternalWakeWordConfig
        
        # Register wake word module
        WakeWordModule.register(self.command_dispatcher, self.event_bus)
        
        # Get access key
        access_key = self.wake_word_config.access_key or os.environ.get('PORCUPINE_ACCESS_KEY')
        
        if not access_key:
            raise ValueError(
                "Porcupine access key required for wake word detection. "
                "Set PORCUPINE_ACCESS_KEY environment variable or pass access_key in WakeWordConfig. "
                "Get your free key at: https://picovoice.ai/"
            )
        
        # Create internal config object
        internal_config = InternalWakeWordConfig(
            detector_type=self.wake_word_config.detector.value,
            wake_words=self.wake_word_config.words,
            sensitivities=[self.wake_word_config.sensitivity] * len(self.wake_word_config.words),
            access_key=access_key,
            keyword_paths=[],  # Empty list for default keywords
            speech_timeout=float(self.wake_word_config.timeout)
        )
        
        # Use the static configure method
        WakeWordModule.configure(self.command_dispatcher, internal_config)
        
        # Subscribe to wake word events
        if self.on_wake_word:
            def handle_wake_word(event: WakeWordDetectedEvent):
                try:
                    self.on_wake_word(event.wake_word, event.confidence)
                except Exception as e:
                    self.logger.error(f"Error in wake word callback: {e}")
                    if self.on_error:
                        self.on_error(e)
            
            self.event_bus.subscribe(WakeWordDetectedEvent, handle_wake_word)
    
    def transcribe_from_mic(self, duration: float = 5.0) -> str:
        """
        Record audio from microphone for a specified duration and transcribe it.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Transcribed text
        """
        # Store the result
        result_text = None
        result_event = threading.Event()
        
        def capture_result(result: TranscriptionResult):
            nonlocal result_text
            result_text = result.text
            result_event.set()
        
        # Temporarily set callback
        old_callback = self.on_transcription
        self.on_transcription = capture_result
        
        try:
            # Start recording
            self.command_dispatcher.dispatch(
                StartRecordingCommand(device_id=self.device_index)
            )
            
            # Record for specified duration
            time.sleep(duration)
            
            # Stop recording
            self.command_dispatcher.dispatch(StopRecordingCommand())
            
            # Wait for transcription result
            if result_event.wait(timeout=10):
                return result_text or ""
            else:
                raise TimeoutError("Transcription timed out")
                
        finally:
            # Restore original callback
            self.on_transcription = old_callback
    
    def transcribe_audio(self, audio_data: Union[bytes, List[float]], sample_rate: int = 16000) -> str:
        """
        Transcribe raw audio data.
        
        Args:
            audio_data: Raw audio data as bytes or float array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        # Convert to bytes if needed
        if isinstance(audio_data, list):
            import numpy as np
            audio_array = np.array(audio_data, dtype=np.float32)
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
        else:
            audio_bytes = audio_data
        
        # Store the result
        result_text = None
        result_event = threading.Event()
        
        def capture_result(result: TranscriptionResult):
            nonlocal result_text
            result_text = result.text
            result_event.set()
        
        # Temporarily set callback
        old_callback = self.on_transcription
        self.on_transcription = capture_result
        
        try:
            # Dispatch transcription command
            self.command_dispatcher.dispatch(
                TranscribeAudioCommand(
                    audio_data=audio_bytes,
                    sample_rate=sample_rate
                )
            )
            
            # Wait for result
            if result_event.wait(timeout=10):
                return result_text or ""
            else:
                raise TimeoutError("Transcription timed out")
                
        finally:
            # Restore original callback
            self.on_transcription = old_callback
    
    def transcribe_file(self, file_path: Union[str, Path]) -> str:
        """
        Transcribe audio from a file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # TODO: Implement file transcription
        # This would use FileAudioProvider
        raise NotImplementedError("File transcription not yet implemented in simple API")
    
    def start_continuous(self):
        """
        Start continuous transcription.
        
        When VAD is enabled, transcription happens automatically when speech ends.
        When VAD is disabled, you need to manually call transcribe() or use timed recordings.
        """
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start recording
        self.command_dispatcher.dispatch(
            StartRecordingCommand(device_id=self.device_index)
        )
        
        # Start wake word detection if configured
        if self.wake_word_config:
            from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule
            WakeWordModule.start_detection(self.command_dispatcher)
        
        self.logger.info("Started continuous transcription")
    
    def stop(self):
        """Stop continuous transcription."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop wake word detection if configured
        if self.wake_word_config:
            from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule
            WakeWordModule.stop_detection(self.command_dispatcher)
        
        # Stop recording
        self.command_dispatcher.dispatch(StopRecordingCommand())
        
        self.logger.info("Stopped continuous transcription")
    
    def list_devices(self) -> List[AudioDevice]:
        """
        List available audio input devices.
        
        Returns:
            List of available audio devices
        """
        return list_audio_devices()
    
    def set_device(self, device_index: int):
        """
        Change the audio input device.
        
        Args:
            device_index: Index of the device to use
        """
        self.device_index = device_index
        
        # If running, restart with new device
        if self._is_running:
            self.stop()
            self.start_continuous()
    
    def set_language(self, language: Optional[str]):
        """
        Set the transcription language.
        
        Args:
            language: Language code (e.g., 'en', 'es') or None for auto-detect
        """
        self.language = language
        
        # Update configuration
        config_command = self.transcription_module.create_configure_command({
            'language': language
        })
        self.command_dispatcher.dispatch(config_command)
    
    def __enter__(self):
        """Context manager support."""
        self.start_continuous()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()