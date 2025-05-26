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

# Import necessary commands and modules
from src.Features.AudioCapture import AudioCaptureModule
from src.Features.AudioCapture.Commands.StartRecordingCommand import StartRecordingCommand
from src.Features.AudioCapture.Commands.StopRecordingCommand import StopRecordingCommand
from src.Features.VoiceActivityDetection import VadModule
from src.Features.VoiceActivityDetection.Commands.EnableVadProcessingCommand import EnableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.DisableVadProcessingCommand import DisableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.ConfigureVadCommand import ConfigureVadCommand
from src.Features.Transcription.Commands.ConfigureTranscriptionCommand import ConfigureTranscriptionCommand
from src.Features.Transcription.Commands.StartTranscriptionSessionCommand import StartTranscriptionSessionCommand
from src.Features.Transcription.Commands.StopTranscriptionSessionCommand import StopTranscriptionSessionCommand
from src.Features.WakeWordDetection.Commands.StartWakeWordDetectionCommand import StartWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.StopWakeWordDetectionCommand import StopWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.ConfigureWakeWordCommand import ConfigureWakeWordCommand
from src.Features.VoiceActivityDetection.Commands.ConfigureVadCommand import ConfigureVadCommand
from src.Features.VoiceActivityDetection.Commands.EnableVadProcessingCommand import EnableVadProcessingCommand
from src.Features.VoiceActivityDetection.Commands.DisableVadProcessingCommand import DisableVadProcessingCommand
from src.Features.Transcription.Commands.ConfigureTranscriptionCommand import ConfigureTranscriptionCommand
from src.Features.Transcription.Commands.StartTranscriptionSessionCommand import StartTranscriptionSessionCommand
from src.Features.Transcription.Commands.StopTranscriptionSessionCommand import StopTranscriptionSessionCommand
from src.Features.WakeWordDetection.Commands.ConfigureWakeWordCommand import ConfigureWakeWordCommand
from src.Features.WakeWordDetection.Commands.StartWakeWordDetectionCommand import StartWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.StopWakeWordDetectionCommand import StopWakeWordDetectionCommand

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
        self.current_profile = None
        self.system_running = False
        self.active_session_id = None
    
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
                default=self.profile_manager.PREDEFINED_PROFILES.get("default", "vad-triggered")
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
            
            # Merge custom configuration with profile if provided
            if request.custom_config:
                self.logger.info(f"Applying custom configuration: {request.custom_config}")
                profile_config = self._merge_configurations(profile_config, request.custom_config)
            
            # Apply the configuration
            try:
                # Stop any existing system first
                if self.system_running:
                    await self._stop_system_internal()
                
                # Apply profile configuration
                self.logger.info(f"Applying configuration from profile: {request.profile}")
                
                # 1. Configure Transcription
                transcription_config = profile_config.get("transcription", {})
                if transcription_config:
                    self.logger.info("Configuring transcription...")
                    self.send_command(ConfigureTranscriptionCommand(
                        engine_type=transcription_config.get("engine", "mlx_whisper"),
                        model_name=transcription_config.get("model", "whisper-large-v3-turbo"),
                        language=transcription_config.get("language"),
                        options=transcription_config.get("options", {})
                    ))
                
                # 2. Configure VAD
                vad_config = profile_config.get("vad", {})
                if vad_config:
                    self.logger.info("Configuring VAD...")
                    vad_params = vad_config.get("parameters", {})
                    self.send_command(ConfigureVadCommand(
                        detector_type=vad_config.get("detector_type", "combined"),
                        sensitivity=vad_config.get("sensitivity", 0.6),
                        window_size=vad_config.get("window_size", 5),
                        min_speech_duration=vad_config.get("min_speech_duration", 0.25),
                        parameters=vad_params
                    ))
                
                # 3. Configure Wake Word if enabled
                wake_word_config = profile_config.get("wake_word", {})
                if wake_word_config.get("enabled", False):
                    self.logger.info("Configuring wake word detection...")
                    wake_words = wake_word_config.get("words", ["jarvis"])
                    sensitivity = wake_word_config.get("sensitivity", 0.7)
                    # Create sensitivities list with same sensitivity for all wake words
                    sensitivities = [sensitivity] * len(wake_words)
                    
                    # Create WakeWordConfig with proper parameters
                    from src.Features.WakeWordDetection.Models.WakeWordConfig import WakeWordConfig
                    config = WakeWordConfig(
                        wake_words=wake_words,
                        sensitivities=sensitivities,
                        speech_timeout=wake_word_config.get("timeout", 30)
                    )
                    
                    self.send_command(ConfigureWakeWordCommand(
                        config=config
                    ))
                
                # 4. Start audio recording
                self.logger.info("Starting audio recording...")
                self.send_command(StartRecordingCommand(
                    device_id=None,  # Use default device
                    sample_rate=16000,
                    chunk_size=512  # Optimal for Silero VAD
                ))
                
                # 5. Enable/configure VAD processing based on profile
                if vad_config.get("enabled", True):
                    if not wake_word_config.get("enabled", False):
                        # For VAD-triggered transcription, enable VAD immediately
                        self.logger.info("Enabling VAD processing...")
                        self.send_command(EnableVadProcessingCommand())
                    else:
                        # For wake word mode, VAD will be enabled after wake word detection
                        self.logger.info("VAD processing will be enabled after wake word detection")
                
                # 6. Start transcription session if auto_start is enabled
                if transcription_config.get("auto_start", False):
                    import uuid
                    self.active_session_id = str(uuid.uuid4())
                    self.logger.info(f"Starting transcription session: {self.active_session_id}")
                    self.send_command(StartTranscriptionSessionCommand(
                        session_id=self.active_session_id
                    ))
                
                # 7. Start wake word detection if enabled
                if wake_word_config.get("enabled", False):
                    self.logger.info("Starting wake word detection...")
                    self.send_command(StartWakeWordDetectionCommand())
                
                # Update system state
                self.system_running = True
                self.current_profile = request.profile
                self.active_features = self._get_active_features(profile_config)
                
                self.logger.info(f"System started successfully with profile: {request.profile}")
                return self.create_standard_response(
                    data={
                        "started": True, 
                        "profile": request.profile,
                        "active_features": self.active_features
                    },
                    message=f"System started with profile: {request.profile}"
                )
                
            except Exception as e:
                self.logger.error(f"Error starting system: {e}")
                # Try to clean up on error
                try:
                    await self._stop_system_internal()
                except:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error starting system: {str(e)}"
                )
        
        @self.router.post("/stop", response_model=Dict[str, Any])
        async def stop_system():
            """Stop the system."""
            self.logger.info("Stopping system")
            
            try:
                await self._stop_system_internal()
                
                return self.create_standard_response(
                    data={"stopped": True},
                    message="System stopped"
                )
            except Exception as e:
                self.logger.error(f"Error stopping system: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error stopping system: {str(e)}"
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
    
    async def _stop_system_internal(self):
        """Internal method to stop all system components."""
        self.logger.info("Stopping all system components...")
        
        try:
            # 1. Stop wake word detection if active
            if "wake_word_detection" in self.active_features:
                self.logger.info("Stopping wake word detection...")
                self.send_command(StopWakeWordDetectionCommand())
            
            # 2. Stop transcription session if active
            if self.active_session_id:
                self.logger.info(f"Stopping transcription session: {self.active_session_id}")
                self.send_command(StopTranscriptionSessionCommand(
                    session_id=self.active_session_id
                ))
                self.active_session_id = None
            
            # 3. Disable VAD processing
            self.logger.info("Disabling VAD processing...")
            self.send_command(DisableVadProcessingCommand())
            
            # 4. Stop audio recording
            self.logger.info("Stopping audio recording...")
            self.send_command(StopRecordingCommand())
            
            # Update system state
            self.system_running = False
            self.current_profile = None
            self.active_features = []
            
            self.logger.info("All system components stopped")
            
        except Exception as e:
            self.logger.error(f"Error during system shutdown: {e}")
            # Still mark system as stopped even if there were errors
            self.system_running = False
            raise
    
    def _get_active_features(self, profile_config: Dict[str, Any]) -> List[str]:
        """Determine which features are active based on profile configuration."""
        features = []
        
        # Always include audio capture if we're starting the system
        features.append("audio_capture")
        
        # Check VAD
        if profile_config.get("vad", {}).get("enabled", True):
            features.append("voice_activity_detection")
        
        # Check Wake Word
        if profile_config.get("wake_word", {}).get("enabled", False):
            features.append("wake_word_detection")
        
        # Check Transcription
        if profile_config.get("transcription", {}):
            features.append("transcription")
        
        return features
    
    def _merge_configurations(self, base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge custom configuration with base profile configuration.
        
        Custom configuration takes precedence over base configuration.
        """
        import copy
        merged = copy.deepcopy(base_config)
        
        # Handle transcription configuration
        if "transcription" in custom_config:
            if "transcription" not in merged:
                merged["transcription"] = {}
            merged["transcription"].update(custom_config["transcription"])
        
        # Handle VAD configuration
        if "vad" in custom_config:
            if "vad" not in merged:
                merged["vad"] = {}
            merged["vad"].update(custom_config["vad"])
        
        # Handle wake word configuration
        if "wake_word" in custom_config:
            if "wake_word" not in merged:
                merged["wake_word"] = {}
            merged["wake_word"].update(custom_config["wake_word"])
        
        return merged