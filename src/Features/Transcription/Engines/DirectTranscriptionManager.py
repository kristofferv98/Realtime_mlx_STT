"""
DirectTranscriptionManager for managing transcription without process isolation.

This module provides a simplified manager for transcription engines without using
separate processes, reducing complexity and potential synchronization issues.
"""

from typing import Dict, Any, Optional, Union

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

from src.Features.Transcription.Engines.DirectMlxWhisperEngine import DirectMlxWhisperEngine
from src.Features.Transcription.Engines.OpenAITranscriptionEngine import OpenAITranscriptionEngine


class DirectTranscriptionManager:
    """
    Simplified transcription manager without process isolation.
    
    This class replaces the TranscriptionProcessManager, providing the same interface
    but without the process isolation complexity. It directly manages transcription
    engines in the same process.
    """
    
    def __init__(self):
        """Initialize the transcription manager."""
        self.logger = LoggingModule.get_logger(__name__)
        self.engine = None
        self._engine_type = None
        
    @property
    def engine_type(self) -> Optional[str]:
        """Get the current engine type."""
        return self._engine_type
    
    def start(self, engine_type: str = "mlx_whisper", engine_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start the transcription engine with the specified engine type.
        
        Args:
            engine_type: Type of transcription engine to use ('mlx_whisper' or 'openai')
            engine_config: Configuration for the transcription engine
            
        Returns:
            bool: True if engine started successfully
        """
        self.logger.info(f"Starting transcription with engine_type={engine_type}")
        
        config = engine_config or {}
        
        try:
            # Store engine type
            self._engine_type = engine_type
            
            # Initialize the appropriate engine based on type
            if engine_type == "mlx_whisper":
                self.engine = DirectMlxWhisperEngine(**config)
                success = self.engine.start()
                if not success:
                    self._engine_type = None
                return success
            elif engine_type == "openai":
                # Create OpenAI transcription engine with API key
                self.engine = OpenAITranscriptionEngine(
                    model_name=config.get("model_name", "gpt-4o-transcribe"),
                    language=config.get("language"),
                    api_key=config.get("openai_api_key"),
                    streaming=config.get("streaming", True)
                )
                success = self.engine.start()
                if not success:
                    self._engine_type = None
                return success
            else:
                self.logger.error(f"Unsupported engine type: {engine_type}")
                self._engine_type = None
                return False
        except Exception as e:
            self.logger.error(f"Error starting transcription engine: {e}", exc_info=True)
            return False
    
    def transcribe(self, 
                  audio_data: Any, 
                  is_first_chunk: bool = False, 
                  is_last_chunk: bool = False, 
                  options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send audio data to transcription engine.
        
        Args:
            audio_data: Audio data to transcribe
            is_first_chunk: Whether this is the first chunk of audio
            is_last_chunk: Whether this is the last chunk of audio
            options: Additional options for transcription
            
        Returns:
            Dict[str, Any]: Transcription result or error
        """
        if not self.is_running():
            return {"error": "Transcription engine not running"}
        
        try:
            # Create a copy of options to avoid modifying the original
            engine_options = dict(options or {})
            
            # Special handling for short audio segments from VAD
            import numpy as np
            if isinstance(audio_data, np.ndarray):
                if audio_data.shape[0] < 8000:  # Less than 0.5 sec at 16kHz
                    self.logger.info(f"Short audio detected ({audio_data.shape[0]} samples) - using recurrent mode")
                    engine_options['quick_mode'] = False
            elif isinstance(audio_data, str) and is_first_chunk and is_last_chunk:
                # For complete files, the engine will handle mode selection based on content length
                pass
            
            # Apply options to engine configuration
            if engine_options:
                self.engine.configure(engine_options)
            
            # Process based on chunk type
            if is_first_chunk and is_last_chunk:
                # Complete audio file
                self.engine.transcribe(audio_data)
            else:
                # Streaming mode
                self.engine.add_audio_chunk(audio_data, is_last=is_last_chunk)
            
            # Wait for and return result
            timeout = engine_options.get('timeout', 60.0)
            result = self.engine.get_result(timeout=timeout)
            if result:
                return result
            else:
                return {"error": "No result available within timeout period"}
                
        except Exception as e:
            self.logger.error(f"Error in transcription: {e}", exc_info=True)
            return {"error": str(e)}
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the transcription engine.
        
        Args:
            config: New configuration for the engine
            
        Returns:
            bool: True if configuration was successful
        """
        if not self.is_running():
            self.logger.warning("Cannot configure - transcription engine not running")
            return False
        
        try:
            return self.engine.configure(config)
        except Exception as e:
            self.logger.error(f"Error configuring engine: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the transcription engine.
        
        Returns:
            bool: True if engine was successfully stopped
        """
        if self.is_running():
            self.logger.info(f"Stopping transcription engine (type: {self._engine_type})")
            
            try:
                if hasattr(self.engine, 'cleanup'):
                    self.engine.cleanup()
                
                self.engine = None
                self._engine_type = None
                return True
                
            except Exception as e:
                self.logger.error(f"Error stopping transcription engine: {e}")
                # Still reset engine reference even if cleanup fails
                self.engine = None
                self._engine_type = None
        
        return True  # Return True even if engine wasn't running
    
    def is_running(self) -> bool:
        """
        Check if transcription engine is running.
        
        Returns:
            bool: True if engine is running
        """
        return self.engine is not None and (
            hasattr(self.engine, 'is_running') and 
            self.engine.is_running()
        )