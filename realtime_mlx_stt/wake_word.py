"""
Wake word detection wrapper for the high-level API.
"""

import os
import sys
import time
import threading
from typing import Optional, Callable

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule
from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent

from .transcriber import Transcriber
from .types import TranscriptionResult, VADMode, TranscriptionCallback, ErrorCallback
from .utils import setup_minimal_logging


class WakeWordTranscriber(Transcriber):
    """
    Transcriber with wake word detection.
    
    Listens for a wake word and only transcribes speech after the wake word is detected.
    
    Example:
        def on_wake_word(word, confidence):
            print(f"Wake word '{word}' detected!")
        
        def on_transcription(result):
            print(f"You said: {result.text}")
        
        transcriber = WakeWordTranscriber(
            wake_word="jarvis",
            on_wake_word=on_wake_word,
            on_transcription=on_transcription
        )
        
        transcriber.start()  # Will listen for "jarvis" then transcribe
    """
    
    def __init__(
        self,
        wake_word: str = "porcupine",
        sensitivity: float = 0.5,
        timeout_after_wake: float = 30.0,
        on_wake_word: Optional[Callable[[str, float], None]] = None,
        access_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize wake word transcriber.
        
        Args:
            wake_word: The wake word to listen for
            sensitivity: Wake word detection sensitivity (0.0-1.0)
            timeout_after_wake: Seconds to listen after wake word detected
            on_wake_word: Callback when wake word is detected
            access_key: Porcupine access key (or set PORCUPINE_ACCESS_KEY env var)
            **kwargs: Additional arguments passed to Transcriber
        """
        # Initialize parent
        super().__init__(**kwargs)
        
        self.wake_word = wake_word
        self.sensitivity = sensitivity
        self.timeout_after_wake = timeout_after_wake
        self.on_wake_word = on_wake_word
        self.access_key = access_key or os.environ.get('PORCUPINE_ACCESS_KEY')
        
        if not self.access_key:
            raise ValueError(
                "Porcupine access key required. Set PORCUPINE_ACCESS_KEY "
                "environment variable or pass access_key parameter. "
                "Get your free key at: https://picovoice.ai/"
            )
        
        # Wake word state
        self._wake_word_active = False
        self._wake_timeout_timer = None
        
        # Initialize wake word module
        self._setup_wake_word()
    
    def _setup_wake_word(self):
        """Setup wake word detection."""
        self.wake_word_module = WakeWordModule()
        self.wake_word_module.register(self.command_dispatcher, self.event_bus)
        
        # Configure wake word
        config = {
            'keywords': [self.wake_word],
            'sensitivity': self.sensitivity,
            'access_key': self.access_key
        }
        config_command = self.wake_word_module.create_configure_command(config)
        self.command_dispatcher.dispatch(config_command)
        
        # Subscribe to wake word events
        self.event_bus.subscribe(WakeWordDetectedEvent, self._on_wake_word_detected)
    
    def _on_wake_word_detected(self, event: WakeWordDetectedEvent):
        """Handle wake word detection."""
        self._wake_word_active = True
        
        # Call user callback
        if self.on_wake_word:
            try:
                self.on_wake_word(event.keyword, event.confidence)
            except Exception as e:
                self.logger.error(f"Error in wake word callback: {e}")
                if self.on_error:
                    self.on_error(e)
        
        # Start timeout timer
        if self._wake_timeout_timer:
            self._wake_timeout_timer.cancel()
        
        self._wake_timeout_timer = threading.Timer(
            self.timeout_after_wake,
            self._on_wake_timeout
        )
        self._wake_timeout_timer.start()
    
    def _on_wake_timeout(self):
        """Handle wake word timeout."""
        self._wake_word_active = False
        self.logger.info("Wake word timeout - returning to listening mode")
    
    def start(self):
        """Start wake word detection and transcription."""
        # Start wake word detection
        start_command = self.wake_word_module.create_start_command()
        self.command_dispatcher.dispatch(start_command)
        
        # Start continuous transcription
        super().start_continuous()
        
        self.logger.info(f"Listening for wake word '{self.wake_word}'...")
    
    def stop(self):
        """Stop wake word detection and transcription."""
        # Cancel timeout timer
        if self._wake_timeout_timer:
            self._wake_timeout_timer.cancel()
            self._wake_timeout_timer = None
        
        # Stop wake word detection
        stop_command = self.wake_word_module.create_stop_command()
        self.command_dispatcher.dispatch(stop_command)
        
        # Stop transcription
        super().stop()
    
    def _setup_event_handlers(self):
        """Override to add wake word filtering."""
        def on_transcription_event(event):
            # Only process transcription if wake word was detected
            if self._wake_word_active and self.on_transcription:
                result = TranscriptionResult(
                    text=event.text,
                    confidence=event.confidence,
                    timestamp=event.timestamp,
                    duration=getattr(event, 'duration', None),
                    language=getattr(event, 'language', None)
                )
                try:
                    self.on_transcription(result)
                except Exception as e:
                    self.logger.error(f"Error in transcription callback: {e}")
                    if self.on_error:
                        self.on_error(e)
        
        from src.Features.Transcription.Events import TranscriptionUpdatedEvent
        self.event_bus.subscribe(TranscriptionUpdatedEvent, on_transcription_event)