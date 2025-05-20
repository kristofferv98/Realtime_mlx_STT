"""
TranscriptionConfig model.

This module defines the TranscriptionConfig class that configures the behavior
of transcription engines and sessions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Literal


@dataclass
class TranscriptionConfig:
    """
    Configuration for transcription engines and sessions.
    
    This class defines parameters that control how transcription is performed,
    including model selection, performance settings, and behavior options.
    """
    # Engine selection and identification
    engine_type: str = "mlx_whisper"  # Options: "mlx_whisper", "openai"
    model_name: str = "whisper-large-v3-turbo"  # For MLX: "whisper-large-v3-turbo", for OpenAI: "gpt-4o-transcribe" or "gpt-4o-mini-transcribe"
    
    # Language settings
    language: Optional[str] = None  # None means auto-detect
    
    # OpenAI API settings
    openai_api_key: Optional[str] = None  # Will check environment variable OPENAI_API_KEY if None
    
    # Performance settings
    compute_type: Literal["default", "float16", "float32"] = "float16"
    beam_size: int = 1
    
    # Streaming options
    streaming: bool = True
    chunk_duration_ms: int = 1000  # Duration of each audio chunk in milliseconds
    chunk_overlap_ms: int = 200    # Overlap between chunks in milliseconds
    
    # Advanced options
    realtime_factor: float = 0.5   # Target processing speed relative to real-time
    max_context_length: int = 128  # Maximum number of tokens to keep in context
    
    # Additional engine-specific options
    options: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def chunk_duration_samples(self) -> int:
        """Convert chunk duration from milliseconds to samples (at 16kHz)."""
        return int((self.chunk_duration_ms / 1000) * 16000)
    
    @property
    def chunk_overlap_samples(self) -> int:
        """Convert chunk overlap from milliseconds to samples (at 16kHz)."""
        return int((self.chunk_overlap_ms / 1000) * 16000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary representation."""
        return {
            "engine_type": self.engine_type,
            "model_name": self.model_name,
            "language": self.language,
            "openai_api_key": self.openai_api_key,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "streaming": self.streaming,
            "chunk_duration_ms": self.chunk_duration_ms,
            "chunk_overlap_ms": self.chunk_overlap_ms,
            "realtime_factor": self.realtime_factor,
            "max_context_length": self.max_context_length,
            "options": self.options
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TranscriptionConfig':
        """Create a configuration from a dictionary representation."""
        # Extract known fields
        known_fields = {
            k: v for k, v in config_dict.items() 
            if k in [
                "engine_type", "model_name", "language", "openai_api_key", "compute_type", 
                "beam_size", "streaming", "chunk_duration_ms", 
                "chunk_overlap_ms", "realtime_factor", "max_context_length"
            ]
        }
        
        # Extract additional options
        options = config_dict.get("options", {})
        
        # Add any unknown fields to options
        for k, v in config_dict.items():
            if k not in known_fields and k != "options":
                options[k] = v
        
        return cls(**known_fields, options=options)