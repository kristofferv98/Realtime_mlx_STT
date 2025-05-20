"""
ConfigureTranscriptionCommand for configuring transcription engine settings.

This command updates the configuration of the transcription system,
allowing runtime changes to the engine behavior.
"""

from typing import Optional, Dict, Any, Literal
from datetime import datetime

from src.Core.Commands.command import Command


class ConfigureTranscriptionCommand(Command):
    """
    Command to configure the transcription engine settings.
    
    When handled, this command will update the configuration of the
    transcription system, such as engine type, model, and parameters.
    
    Args:
        engine_type: Type of transcription engine to use ('mlx_whisper' or 'openai')
        model_name: Name of the model to use (e.g., 'whisper-large-v3-turbo' or 'gpt-4o-transcribe')
        language: Optional language code (e.g., 'en', 'fr') or None for auto-detection
        beam_size: Beam search size for inference (1 for greedy)
        compute_type: Compute precision ('default', 'float16', or 'float32')
        streaming: Whether to enable streaming mode
        openai_api_key: OpenAI API key (required for 'openai' engine type)
        options: Additional engine-specific parameters
    """
    
    def __init__(self,
                engine_type: str = "mlx_whisper",
                model_name: str = "whisper-large-v3-turbo",
                language: Optional[str] = None,
                beam_size: int = 1,
                compute_type: str = "float16",
                streaming: bool = True,
                chunk_duration_ms: int = 1000,
                chunk_overlap_ms: int = 200,
                openai_api_key: Optional[str] = None,
                options: Optional[Dict[str, Any]] = None,
                id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                name: Optional[str] = None):
        """
        Initialize the command.
        
        Args:
            engine_type: Engine type
            model_name: Model name
            language: Language code
            beam_size: Beam search size
            compute_type: Computation precision
            streaming: Streaming mode flag
            chunk_duration_ms: Chunk duration in ms
            chunk_overlap_ms: Chunk overlap in ms
            openai_api_key: Optional OpenAI API key (for 'openai' engine)
            options: Additional options
            id: Optional command ID
            timestamp: Optional command timestamp
            name: Optional command name
        """
        # Initialize the base class
        super().__init__(id=id, timestamp=timestamp, name=name)
        
        # Initialize our fields
        self.engine_type = engine_type
        self.model_name = model_name
        self.language = language
        self.beam_size = beam_size
        self.compute_type = compute_type
        self.streaming = streaming
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_overlap_ms = chunk_overlap_ms
        self.openai_api_key = openai_api_key
        self.options = options or {}