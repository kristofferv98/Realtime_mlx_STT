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
    vad_sensitivity: float = 0.5
    vad_min_silence_duration: float = 2.0
    vad_min_speech_duration: float = 0.25
    
    # Wake word defaults
    wake_word_sensitivity: float = 0.7
    wake_word_timeout: int = 30
    
    # Auto-stop settings
    auto_stop_after_silence: bool = False
    silence_timeout: float = 2.0  # seconds of silence before stopping
    
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
        # VAD configuration
        vad_sensitivity: float = 0.5,
        vad_min_silence_duration: float = 2.0,
        vad_min_speech_duration: float = 0.25,
        # Auto-stop settings
        auto_stop_after_silence: bool = False,
        silence_timeout: Optional[float] = None,  # Uses vad_min_silence_duration if None
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
            vad_sensitivity: Voice activity detection sensitivity (0.0-1.0, default: 0.5)
            vad_min_silence_duration: Minimum silence duration to end speech (seconds, default: 2.0)
            vad_min_speech_duration: Minimum speech duration to start transcription (seconds, default: 0.25)
            auto_stop_after_silence: Automatically stop after silence timeout (default: False)
            silence_timeout: Override silence timeout (uses vad_min_silence_duration if None)
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
            vad_sensitivity=vad_sensitivity,
            vad_min_silence_duration=vad_min_silence_duration,
            vad_min_speech_duration=vad_min_speech_duration,
            auto_stop_after_silence=auto_stop_after_silence,
            silence_timeout=silence_timeout or vad_min_silence_duration,
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
        auto_stop_after_silence: Optional[bool] = True,  # Default to True for single utterance
        silence_timeout: Optional[float] = None,
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
            auto_stop_after_silence: Override auto-stop behavior
            silence_timeout: Override silence timeout (seconds)
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
        vad_sensitivity = vad_sensitivity if vad_sensitivity is not None else self.config.vad_sensitivity
        auto_stop_after_silence = auto_stop_after_silence if auto_stop_after_silence is not None else self.config.auto_stop_after_silence
        silence_timeout = silence_timeout or self.config.silence_timeout
        
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
            vad=VADConfig(
                sensitivity=vad_sensitivity,
                min_silence_duration=self.config.vad_min_silence_duration,
                min_speech_duration=self.config.vad_min_speech_duration
            ),
            on_transcription=on_transcription,
            verbose=self.config.verbose
        )
        
        # Start session
        if not session.start():
            raise RuntimeError("Failed to start transcription session")
        
        try:
            start_time = time.time()
            last_result_time = start_time
            last_activity_time = start_time
            has_had_speech = False
            
            while session.is_running():
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Yield results and track timing
                with result_lock:
                    if results:
                        has_had_speech = True
                        last_activity_time = time.time()
                    while results:
                        yield results.pop(0)
                        last_result_time = time.time()
                
                # Check auto-stop condition - only after we've had some speech
                if auto_stop_after_silence and has_had_speech:
                    silence_duration = time.time() - last_activity_time
                    if silence_duration >= silence_timeout:
                        break
                
                time.sleep(0.05)
            
            # Allow extra time for final processing after silence is detected
            if auto_stop_after_silence and has_had_speech:
                time.sleep(2.0)  # Wait longer for transcription to complete
            else:
                time.sleep(0.5)  # Standard wait time
                
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
        vad_sensitivity: Optional[float] = None,
        auto_stop_after_silence: Optional[bool] = False,  # Default to False for continuous streaming
        silence_timeout: Optional[float] = None
    ):
        """
        Context manager for streaming transcription.
        
        Args:
            engine: Override default engine
            model: Override default model  
            language: Override default language
            vad_sensitivity: Override VAD sensitivity
            auto_stop_after_silence: Override auto-stop behavior
            silence_timeout: Override silence timeout (seconds)
        
        Example:
            with client.stream() as stream:
                for result in stream:
                    print(result.text)
                    if "goodbye" in result.text.lower():
                        break
                        
            # Auto-stop after silence
            with client.stream(auto_stop_after_silence=True, silence_timeout=3.0) as stream:
                for result in stream:
                    print(result.text)
                # Stream automatically stops after 3 seconds of silence
        """
        # Use defaults
        engine = engine or self.config.default_engine
        model = model or self.config.default_model
        language = language or self.config.default_language
        vad_sensitivity = vad_sensitivity if vad_sensitivity is not None else self.config.vad_sensitivity
        auto_stop_after_silence = auto_stop_after_silence if auto_stop_after_silence is not None else self.config.auto_stop_after_silence
        silence_timeout = silence_timeout or self.config.silence_timeout
        
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
            vad=VADConfig(
                sensitivity=vad_sensitivity,
                min_silence_duration=self.config.vad_min_silence_duration,
                min_speech_duration=self.config.vad_min_speech_duration
            ),
            on_transcription=on_transcription,
            verbose=self.config.verbose
        )
        
        # Stream iterator
        def stream_iterator():
            last_result_time = time.time()
            last_activity_time = time.time()
            has_had_speech = False
            
            while self._stream_active and session.is_running():
                # Check auto-stop condition - only after we've had some speech
                if auto_stop_after_silence and has_had_speech:
                    silence_duration = time.time() - last_activity_time
                    if silence_duration >= silence_timeout:
                        break
                
                with result_lock:
                    if results:
                        has_had_speech = True
                        last_activity_time = time.time()
                    while results:
                        yield results.pop(0)
                        last_result_time = time.time()
                time.sleep(0.05)
            
            # Allow extra time for final processing after silence is detected
            if auto_stop_after_silence and has_had_speech:
                time.sleep(2.0)  # Wait longer for transcription to complete
            else:
                time.sleep(0.5)  # Standard wait time
                
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
    
    def transcribe_until_silence(
        self,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        vad_sensitivity: Optional[float] = None,
        silence_timeout: Optional[float] = None,
        max_duration: Optional[float] = None
    ) -> str:
        """
        Transcribe audio until silence is detected, then return the complete text.
        
        This is a convenience method that provides the common "record until silence" behavior
        similar to voice assistants.
        
        Args:
            engine: Override default engine
            model: Override default model
            language: Override default language
            vad_sensitivity: Override VAD sensitivity
            silence_timeout: Override silence timeout (seconds)
            max_duration: Maximum recording duration (seconds, default: 60)
            
        Returns:
            Complete transcribed text as a single string
            
        Example:
            client = STTClient()
            text = client.transcribe_until_silence(silence_timeout=2.0)
            print(f"You said: {text}")
        """
        # Use defaults
        silence_timeout = silence_timeout or self.config.silence_timeout
        max_duration = max_duration or 60.0
        
        # Collect all results
        results = []
        
        for result in self.transcribe(
            duration=max_duration,
            engine=engine,
            model=model,
            language=language,
            vad_sensitivity=vad_sensitivity,
            auto_stop_after_silence=True,
            silence_timeout=silence_timeout
        ):
            results.append(result.text)
        
        # Join all transcription results
        return " ".join(results).strip()
    
    def transcribe_utterance(
        self,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        vad_sensitivity: Optional[float] = None,
        initial_vad_sensitivity: Optional[float] = None,
        initial_sensitivity_duration: float = 3.0,
        max_duration: Optional[float] = None
    ) -> str:
        """
        Transcribe a single utterance (speech segment).
        
        This is the most common use case: start recording, detect speech,
        stop after silence, and return the complete transcription IMMEDIATELY
        when results are available.
        
        Args:
            engine: Override default engine
            model: Override default model
            language: Override default language
            vad_sensitivity: Override VAD sensitivity (used after initial period)
            initial_vad_sensitivity: VAD sensitivity for initial recording period (prevents false triggers)
            initial_sensitivity_duration: Duration in seconds to use initial sensitivity (default: 3.0)
            max_duration: Maximum recording duration (seconds, default: 30)
            
        Returns:
            Complete transcribed text as a single string
            
        Example:
            client = STTClient()
            text = client.transcribe_utterance()
            print(f"You said: {text}")
            
            # Prevent false triggers with lower initial sensitivity
            text = client.transcribe_utterance(
                vad_sensitivity=0.6,
                initial_vad_sensitivity=0.4,
                initial_sensitivity_duration=3.0
            )
        """
        # Use defaults
        engine = engine or self.config.default_engine
        model = model or self.config.default_model
        language = language or self.config.default_language
        vad_sensitivity = vad_sensitivity if vad_sensitivity is not None else self.config.vad_sensitivity
        max_duration = max_duration or 30.0
        
        # Handle time-based VAD sensitivity adjustment
        use_initial_sensitivity = initial_vad_sensitivity is not None
        current_sensitivity = initial_vad_sensitivity if use_initial_sensitivity else vad_sensitivity
        
        # Check engine requirements
        if engine == "openai" and not self.openai_api_key:
            raise ValueError(
                "OpenAI API key required. Pass openai_api_key to STTClient "
                "or set OPENAI_API_KEY environment variable."
            )
        
        # Results tracking
        results = []
        result_lock = threading.Lock()
        last_result_time = time.time()
        
        def on_transcription(result: TranscriptionResult):
            nonlocal last_result_time
            with result_lock:
                results.append(result.text)
                last_result_time = time.time()
        
        # Create initial session
        session = TranscriptionSession(
            model=ModelConfig(engine=engine, model=model, language=language),
            vad=VADConfig(
                sensitivity=current_sensitivity,
                min_silence_duration=self.config.vad_min_silence_duration,
                min_speech_duration=self.config.vad_min_speech_duration
            ),
            on_transcription=on_transcription,
            verbose=self.config.verbose
        )
        
        # Start session
        if not session.start():
            raise RuntimeError("Failed to start transcription session")
        
        try:
            start_time = time.time()
            has_had_speech = False
            sensitivity_switched = False
            
            while session.is_running():
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Check if we need to switch to higher sensitivity
                if (use_initial_sensitivity and 
                    not sensitivity_switched and 
                    elapsed_time >= initial_sensitivity_duration):
                    
                    if self.config.verbose:
                        print(f"Switching VAD sensitivity from {current_sensitivity} to {vad_sensitivity}")
                    
                    # Stop current session
                    session.stop()
                    
                    # Create new session with normal sensitivity
                    session = TranscriptionSession(
                        model=ModelConfig(engine=engine, model=model, language=language),
                        vad=VADConfig(
                            sensitivity=vad_sensitivity,
                            min_silence_duration=self.config.vad_min_silence_duration,
                            min_speech_duration=self.config.vad_min_speech_duration
                        ),
                        on_transcription=on_transcription,
                        verbose=self.config.verbose
                    )
                    
                    # Start new session
                    if not session.start():
                        raise RuntimeError("Failed to start transcription session with normal sensitivity")
                    
                    sensitivity_switched = True
                    continue
                
                # Check duration
                if elapsed_time >= max_duration:
                    break
                
                # Check for results
                with result_lock:
                    if results:
                        has_had_speech = True
                        # Return immediately after a short delay (0.3s) to ensure no more results
                        time.sleep(0.3)
                        if results:  # Check again after delay
                            break
                
                time.sleep(0.05)
            
            # Get final results
            with result_lock:
                final_text = " ".join(results).strip()
                
        finally:
            session.stop()
        
        return final_text
    
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