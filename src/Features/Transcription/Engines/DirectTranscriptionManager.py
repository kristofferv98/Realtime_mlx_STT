"""
DirectTranscriptionManager for managing transcription without process isolation.

This module provides a simplified manager for transcription engines without using
separate processes, reducing complexity and potential synchronization issues.
"""

import logging
from typing import Dict, Any, Optional, Union

from src.Features.Transcription.Engines.DirectMlxWhisperEngine import DirectMlxWhisperEngine


class DirectTranscriptionManager:
    """
    Simplified transcription manager without process isolation.
    
    This class replaces the TranscriptionProcessManager, providing the same interface
    but without the process isolation complexity. It directly manages transcription
    engines in the same process.
    """
    
    def __init__(self):
        """Initialize the transcription manager."""
        self.logger = logging.getLogger(__name__)
        self.engine = None
    
    def start(self, engine_type: str = "mlx_whisper", engine_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start the transcription engine with the specified engine type.
        
        Args:
            engine_type: Type of transcription engine to use
            engine_config: Configuration for the transcription engine
            
        Returns:
            bool: True if engine started successfully
        """
        self.logger.info(f"Starting transcription with engine_type={engine_type}")
        
        config = engine_config or {}
        
        try:
            # Initialize the appropriate engine based on type
            if engine_type == "mlx_whisper":
                self.engine = DirectMlxWhisperEngine(**config)
                return self.engine.start()
            else:
                self.logger.error(f"Unsupported engine type: {engine_type}")
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
            # Apply any options to engine configuration
            if options:
                self.engine.configure(options)
            
            # Process based on chunk type
            if is_first_chunk and is_last_chunk:
                # Complete audio file
                self.engine.transcribe(audio_data)
            else:
                # Streaming mode
                self.engine.add_audio_chunk(audio_data, is_last=is_last_chunk)
            
            # Wait for and return result
            result = self.engine.get_result(timeout=options.get('timeout', 60.0) if options else 60.0)
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
            self.logger.info("Stopping transcription engine")
            
            try:
                if hasattr(self.engine, 'cleanup'):
                    self.engine.cleanup()
                
                self.engine = None
                return True
                
            except Exception as e:
                self.logger.error(f"Error stopping transcription engine: {e}")
        
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