"""
Client-based API for Realtime MLX STT.

This provides a clean, modern API similar to popular libraries like OpenAI's client.
"""

import os
import sys
import time
import threading
from typing import Optional, Dict, Any, Callable, Iterator, List
from dataclasses import dataclass
from contextlib import contextmanager

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .session import TranscriptionSession, SessionState
from .config import ModelConfig, VADConfig, WakeWordConfig
from .types import TranscriptionResult, AudioDevice
from .utils import list_audio_devices, setup_minimal_logging


@dataclass
class STTConfig:
    """Configuration for STT Client."""
    # API Keys
    openai_api_key: Optional[str] = None
    porcupine_api_key: Optional[str] = None
    
    # Default settings
    default_engine: str = "mlx_whisper"
    default_model: str = "whisper-large-v3-turbo"
    default_language: Optional[str] = None
    default_device: Optional[int] = None
    
    # VAD defaults
    vad_sensitivity: float = 0.6
    vad_min_speech_duration: float = 0.25
    
    # Wake word defaults
    wake_word_sensitivity: float = 0.7
    wake_word_timeout: int = 30
    
    # Client settings
    auto_start: bool = True
    verbose: bool = False


class STTClient:
    """
    Modern client interface for Realtime MLX STT.
    
    Examples:
        # Basic usage
        client = STTClient()
        
        # Listen for 10 seconds
        for result in client.transcribe(duration=10):
            print(result.text)
        
        # Continuous transcription
        with client.stream() as stream:
            for result in stream:
                print(result.text)
                if "stop" in result.text.lower():
                    break
        
        # With API keys
        client = STTClient(
            openai_api_key="sk-...",
            porcupine_api_key="..."
        )
        
        # OpenAI transcription
        for result in client.transcribe(engine="openai"):
            print(result.text)
        
        # Wake word mode
        client.start_wake_word("jarvis")
        # ... transcribes only after "jarvis" is spoken
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        porcupine_api_key: Optional[str] = None,
        default_engine: str = "mlx_whisper",
        default_model: Optional[str] = None,
        default_language: Optional[str] = None,
        device_index: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize STT client.
        
        Args:
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            porcupine_api_key: Porcupine API key (or set PORCUPINE_ACCESS_KEY env var)
            default_engine: Default transcription engine ("mlx_whisper" or "openai")
            default_model: Default model name
            default_language: Default language code (None for auto-detect)
            device_index: Audio device index (None for system default)
            verbose: Enable verbose logging
        """
        # Store API keys
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.porcupine_api_key = porcupine_api_key or os.environ.get('PORCUPINE_ACCESS_KEY')
        
        # Set OpenAI key in environment if provided
        if self.openai_api_key:
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
        
        # Configuration
        self.config = STTConfig(
            openai_api_key=self.openai_api_key,
            porcupine_api_key=self.porcupine_api_key,
            default_engine=default_engine,
            default_model=default_model or self._get_default_model(default_engine),
            default_language=default_language,
            default_device=device_index,
            verbose=verbose
        )
        
        # Setup logging
        if not verbose:
            setup_minimal_logging()
        
        # Active session
        self._session: Optional[TranscriptionSession] = None
        self._stream_active = False
    
    def _get_default_model(self, engine: str) -> str:
        """Get default model for engine."""
        if engine == "openai":
            return "gpt-4o-transcribe"
        return "whisper-large-v3-turbo"
    
    def transcribe(
        self,
        duration: Optional[float] = None,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        vad_sensitivity: Optional[float] = None,
        on_partial: Optional[Callable[[str], None]] = None
    ) -> Iterator[TranscriptionResult]:
        """
        Transcribe audio for a specified duration or until stopped.
        
        Args:
            duration: Maximum duration in seconds (None for continuous)
            engine: Override default engine
            model: Override default model
            language: Override default language
            vad_sensitivity: Override VAD sensitivity
            on_partial: Callback for partial results (if supported)
            
        Yields:
            TranscriptionResult objects
            
        Example:
            for result in client.transcribe(duration=30):
                print(f"{result.text} (confidence: {result.confidence})")
        """
        # Use defaults
        engine = engine or self.config.default_engine
        model = model or self.config.default_model
        language = language or self.config.default_language
        vad_sensitivity = vad_sensitivity or self.config.vad_sensitivity
        
        # Check engine requirements
        if engine == "openai" and not self.openai_api_key:
            raise ValueError(
                "OpenAI API key required. Pass openai_api_key to STTClient "
                "or set OPENAI_API_KEY environment variable."
            )
        
        # Results queue
        results = []
        result_lock = threading.Lock()
        
        def on_transcription(result: TranscriptionResult):
            with result_lock:
                results.append(result)
        
        # Create session
        session = TranscriptionSession(
            model=ModelConfig(engine=engine, model=model, language=language),
            vad=VADConfig(sensitivity=vad_sensitivity),
            on_transcription=on_transcription,
            verbose=self.config.verbose
        )
        
        # Start session
        if not session.start():
            raise RuntimeError("Failed to start transcription session")
        
        try:
            start_time = time.time()
            
            while session.is_running():
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Yield results
                with result_lock:
                    while results:
                        yield results.pop(0)
                
                time.sleep(0.05)
            
            # Final results
            time.sleep(0.5)  # Allow final processing
            with result_lock:
                while results:
                    yield results.pop(0)
                    
        finally:
            session.stop()
    
    @contextmanager
    def stream(
        self,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        vad_sensitivity: Optional[float] = None
    ):
        """
        Context manager for streaming transcription.
        
        Example:
            with client.stream() as stream:
                for result in stream:
                    print(result.text)
                    if "goodbye" in result.text.lower():
                        break
        """
        # Use defaults
        engine = engine or self.config.default_engine
        model = model or self.config.default_model
        language = language or self.config.default_language
        vad_sensitivity = vad_sensitivity or self.config.vad_sensitivity
        
        # Check requirements
        if engine == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key required")
        
        # Results queue
        results = []
        result_lock = threading.Lock()
        self._stream_active = True
        
        def on_transcription(result: TranscriptionResult):
            if self._stream_active:
                with result_lock:
                    results.append(result)
        
        # Create session
        session = TranscriptionSession(
            model=ModelConfig(engine=engine, model=model, language=language),
            vad=VADConfig(sensitivity=vad_sensitivity),
            on_transcription=on_transcription,
            verbose=self.config.verbose
        )
        
        # Stream iterator
        def stream_iterator():
            while self._stream_active and session.is_running():
                with result_lock:
                    while results:
                        yield results.pop(0)
                time.sleep(0.05)
            
            # Final results
            time.sleep(0.5)
            with result_lock:
                while results:
                    yield results.pop(0)
        
        # Start session
        if not session.start():
            raise RuntimeError("Failed to start streaming session")
        
        try:
            yield stream_iterator()
        finally:
            self._stream_active = False
            session.stop()
    
    def start_wake_word(
        self,
        wake_word: str = "jarvis",
        sensitivity: float = 0.7,
        on_wake: Optional[Callable[[str, float], None]] = None,
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
    ):
        """
        Start wake word detection mode.
        
        Args:
            wake_word: Wake word to listen for
            sensitivity: Detection sensitivity (0.0-1.0)
            on_wake: Callback when wake word detected
            on_transcription: Callback for transcriptions after wake word
            
        Example:
            def on_wake(word, confidence):
                print(f"Wake word '{word}' detected!")
            
            def on_result(result):
                print(f"Command: {result.text}")
            
            client.start_wake_word(
                wake_word="computer",
                on_wake=on_wake,
                on_transcription=on_result
            )
        """
        if not self.porcupine_api_key:
            raise ValueError(
                "Porcupine API key required for wake word detection. "
                "Pass porcupine_api_key to STTClient or set PORCUPINE_ACCESS_KEY."
            )
        
        # Stop any existing session
        self.stop()
        
        # Create wake word session
        self._session = TranscriptionSession(
            model=ModelConfig(
                engine=self.config.default_engine,
                model=self.config.default_model,
                language=self.config.default_language
            ),
            vad=VADConfig(sensitivity=self.config.vad_sensitivity),
            wake_word=WakeWordConfig(
                words=[wake_word.lower()],
                sensitivity=sensitivity,
                timeout=self.config.wake_word_timeout,
                access_key=self.porcupine_api_key
            ),
            on_wake_word=on_wake,
            on_transcription=on_transcription,
            verbose=self.config.verbose
        )
        
        if not self._session.start():
            raise RuntimeError("Failed to start wake word session")
    
    def stop(self):
        """Stop any active transcription or wake word detection."""
        if self._session and self._session.is_running():
            self._session.stop()
            self._session = None
        self._stream_active = False
    
    def list_devices(self) -> List[AudioDevice]:
        """List available audio input devices."""
        return list_audio_devices()
    
    def set_device(self, device_index: int):
        """Set the default audio device."""
        self.config.default_device = device_index
    
    def set_language(self, language: Optional[str]):
        """Set the default language."""
        self.config.default_language = language
    
    def is_active(self) -> bool:
        """Check if any session is active."""
        return self._session is not None and self._session.is_running()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        self.stop()


# Convenience function
def create_client(
    openai_api_key: Optional[str] = None,
    porcupine_api_key: Optional[str] = None,
    **kwargs
) -> STTClient:
    """
    Create an STT client.
    
    Args:
        openai_api_key: OpenAI API key
        porcupine_api_key: Porcupine API key
        **kwargs: Additional arguments for STTClient
        
    Returns:
        STTClient instance
        
    Example:
        client = create_client(
            openai_api_key="sk-...",
            default_engine="openai"
        )
    """
    return STTClient(
        openai_api_key=openai_api_key,
        porcupine_api_key=porcupine_api_key,
        **kwargs
    )