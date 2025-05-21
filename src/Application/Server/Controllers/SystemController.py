"""
SystemController for Realtime_mlx_STT Server

This module provides API endpoints for system-wide operations, including
status, configuration profiles, and system start/stop.
"""

import time
import platform
import os
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status, Body

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Infrastructure.Logging.LoggingModule import LoggingModule

from ..Models.SystemModels import (
    ServerStatusResponse,
    ProfileRequest,
    ProfileListResponse,
    ProfileData,
    GeneralConfigRequest
)
from ..Configuration.ProfileManager import ProfileManager
from .BaseController import BaseController

class SystemController(BaseController):
    """
    Controller for system-wide API endpoints.
    
    This controller provides endpoints for system status, configuration profiles,
    and system start/stop operations.
    """
    
    def __init__(self, command_dispatcher: CommandDispatcher, event_bus: EventBus, 
                 profile_manager: ProfileManager):
        """
        Initialize the system controller.
        
        Args:
            command_dispatcher: Command dispatcher to use for sending commands
            event_bus: Event bus for subscribing to events
            profile_manager: Profile manager for handling configuration profiles
        """
        super().__init__(command_dispatcher, event_bus, prefix="/system")
        self.logger = LoggingModule.get_logger(__name__)
        self.profile_manager = profile_manager
        self.start_time = time.time()
        self.active_features = []  # Will be populated as features are activated
        self.version = "0.1.0"  # TODO: Get this from a central version file
    
    def _register_routes(self):
        """Register routes for this controller."""
        
        @self.router.get("/status", response_model=ServerStatusResponse)
        async def get_status():
            """Get the current status of the server."""
            return ServerStatusResponse(
                status="online",
                version=self.version,
                uptime=time.time() - self.start_time,
                active_features=self.active_features,
                active_connections=len(getattr(self.event_bus, 'subscribers', {}))
            )
        
        @self.router.get("/info", response_model=Dict[str, Any])
        async def get_info():
            """Get system information."""
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
                "cpu_count": os.cpu_count(),
                "version": self.version,
                "features": [
                    "transcription",
                    "voice_activity_detection",
                    "wake_word_detection",
                    "audio_capture"
                ]
            }
        
        @self.router.get("/profiles", response_model=ProfileListResponse)
        async def list_profiles():
            """List available configuration profiles."""
            profiles = self.profile_manager.list_profiles()
            return ProfileListResponse(
                profiles=profiles,
                default=self.profile_manager.PREDEFINED_PROFILES.get("default", "continuous-mlx")
            )
        
        @self.router.get("/profiles/{name}", response_model=ProfileData)
        async def get_profile(name: str):
            """Get a specific configuration profile."""
            profile = self.profile_manager.get_profile(name)
            if not profile:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Profile not found: {name}"
                )
            
            return ProfileData(
                name=name,
                config=profile
            )
        
        @self.router.post("/profiles", response_model=Dict[str, Any])
        async def save_profile(profile: ProfileData = Body(...)):
            """Save a configuration profile."""
            success = self.profile_manager.save_profile(profile.name, profile.config)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not save profile: {profile.name}"
                )
            
            return self.create_standard_response(
                data={"saved": True},
                message=f"Profile '{profile.name}' saved successfully"
            )
        
        @self.router.delete("/profiles/{name}", response_model=Dict[str, Any])
        async def delete_profile(name: str):
            """Delete a configuration profile."""
            success = self.profile_manager.delete_profile(name)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not delete profile: {name}"
                )
            
            return self.create_standard_response(
                data={"deleted": True},
                message=f"Profile '{name}' deleted successfully"
            )
        
        @self.router.post("/start", response_model=Dict[str, Any])
        async def start_system(request: ProfileRequest = Body(...)):
            """Start the system with a specific profile."""
            self.logger.info(f"Starting system with profile: {request.profile}")
            
            # Get the profile configuration
            profile_config = self.profile_manager.get_profile(request.profile)
            if not profile_config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Profile not found: {request.profile}"
                )
            
            # Apply the configuration
            try:
                # TODO: Implement actual configuration application
                # This would dispatch commands to configure and start the different components
                
                # For now, just log and return success
                self.logger.info(f"Applied configuration from profile: {request.profile}")
                return self.create_standard_response(
                    data={"started": True, "profile": request.profile},
                    message=f"System started with profile: {request.profile}"
                )
                
            except Exception as e:
                self.logger.error(f"Error starting system: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error starting system: {str(e)}"
                )
        
        @self.router.post("/stop", response_model=Dict[str, Any])
        async def stop_system():
            """Stop the system."""
            self.logger.info("Stopping system")
            
            # TODO: Implement actual system shutdown
            # This would dispatch commands to stop the different components
            
            return self.create_standard_response(
                data={"stopped": True},
                message="System stopped"
            )
        
        @self.router.post("/config", response_model=Dict[str, Any])
        async def update_config(config: GeneralConfigRequest = Body(...)):
            """Update system configuration."""
            self.logger.info("Updating system configuration")
            
            # TODO: Implement actual configuration update
            # This would dispatch commands to configure the different components
            
            return self.create_standard_response(
                data={"updated": True},
                message="System configuration updated"
            )