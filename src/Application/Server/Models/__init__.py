"""
API models for the Realtime_mlx_STT Server.

This module provides Pydantic models for request/response handling
in the REST and WebSocket APIs.
"""

from .SystemModels import (
    ServerStatusResponse,
    ProfileRequest,
    ProfileListResponse,
    ProfileData,
    GeneralConfigRequest,
    SystemErrorResponse
)

from .TranscriptionModels import (
    TranscriptionConfigRequest,
    TranscriptionSessionRequest,
    TranscribeAudioRequest,
    TranscriptionResult,
    TranscriptionStatusResponse
)

__all__ = [
    # System models
    'ServerStatusResponse',
    'ProfileRequest',
    'ProfileListResponse',
    'ProfileData',
    'GeneralConfigRequest',
    'SystemErrorResponse',
    
    # Transcription models
    'TranscriptionConfigRequest',
    'TranscriptionSessionRequest',
    'TranscribeAudioRequest',
    'TranscriptionResult',
    'TranscriptionStatusResponse'
]
