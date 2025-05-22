"""
SystemModels for Realtime_mlx_STT Server

This module defines the data models used for system-wide API requests
and responses in the server.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class ServerStatusResponse(BaseModel):
    """Status information about the server."""
    
    status: str = Field(..., description="Server status")
    version: str = Field(..., description="Server version")
    uptime: float = Field(..., description="Server uptime in seconds")
    active_features: List[str] = Field(..., description="Active feature modules")
    active_connections: int = Field(..., description="Number of active WebSocket connections")

class ProfileRequest(BaseModel):
    """Request to start the system with a specific profile."""
    
    profile: str = Field(..., description="Name of the profile to use")
    custom_config: Optional[Dict[str, Any]] = Field(None, description="Custom configuration to merge with profile")

class ProfileListResponse(BaseModel):
    """Response containing available profiles."""
    
    profiles: List[str] = Field(..., description="List of available profile names")
    default: str = Field(..., description="Default profile name")

class ProfileData(BaseModel):
    """Data for a configuration profile."""
    
    name: str = Field(..., description="Profile name")
    config: Dict[str, Any] = Field(..., description="Profile configuration")

class GeneralConfigRequest(BaseModel):
    """General configuration request for the system."""
    
    transcription: Optional[Dict[str, Any]] = Field(None, description="Transcription configuration")
    vad: Optional[Dict[str, Any]] = Field(None, description="Voice activity detection configuration")
    wake_word: Optional[Dict[str, Any]] = Field(None, description="Wake word configuration")
    audio: Optional[Dict[str, Any]] = Field(None, description="Audio configuration")
    
class SystemErrorResponse(BaseModel):
    """Error response from the system."""
    
    status: str = Field("error", description="Error status")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")