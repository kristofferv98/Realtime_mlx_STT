"""
TranscriptionModels for Realtime_mlx_STT Server

This module defines the data models used for transcription API requests
and responses in the server.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class TranscriptionConfigRequest(BaseModel):
    """Configuration request for the transcription engine."""
    
    engine_type: str = Field(..., description="Type of transcription engine (mlx_whisper, openai)")
    model_name: str = Field(..., description="Name of the model to use")
    language: Optional[str] = Field(None, description="Optional language code")
    beam_size: Optional[int] = Field(None, description="Beam size for beam search decoding")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional engine-specific options")

class TranscriptionSessionRequest(BaseModel):
    """Request to start a transcription session."""
    
    session_id: Optional[str] = Field(None, description="Optional session ID, will be generated if not provided")

class TranscribeAudioRequest(BaseModel):
    """Request to transcribe an audio chunk."""
    
    audio_data: str = Field(..., description="Base64-encoded audio data")
    session_id: Optional[str] = Field(None, description="Session ID for continuous transcription")
    is_final: bool = Field(False, description="Whether this is the final chunk in the session")

class TranscriptionResult(BaseModel):
    """Result of a transcription operation."""
    
    text: str = Field(..., description="Transcribed text")
    is_final: bool = Field(..., description="Whether this is a final result")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    session_id: Optional[str] = Field(None, description="Session ID for continuous transcription")
    segments: Optional[List[Dict[str, Any]]] = Field(None, description="Segmented transcription results")
    
class TranscriptionStatusResponse(BaseModel):
    """Status of the transcription system."""
    
    active: bool = Field(..., description="Whether transcription is active")
    engine: str = Field(..., description="Current transcription engine")
    model: str = Field(..., description="Current model name")
    language: Optional[str] = Field(None, description="Current language setting")
    sessions: Optional[List[str]] = Field(None, description="Active session IDs")