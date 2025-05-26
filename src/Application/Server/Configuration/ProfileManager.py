"""
ProfileManager for Realtime_mlx_STT Server

This module provides management of configuration profiles,
allowing predefined and custom configurations to be loaded and saved.
"""

import os
import json
from typing import Dict, Any, List, Optional
from src.Infrastructure.Logging.LoggingModule import LoggingModule

class ProfileManager:
    """
    Manages configuration profiles for the server.
    
    This class handles loading, saving, and applying configuration
    profiles, which define settings for different use cases.
    """
    
    # Predefined profiles with their configurations
    # These define operating modes, not specific models
    PREDEFINED_PROFILES = {
        "vad-triggered": {
            "description": "VAD-triggered transcription - only transcribe when speech is detected",
            "transcription": {
                "auto_start": True
            },
            "vad": {
                "detector_type": "combined",
                "sensitivity": 0.6,
                "enabled": True,
                "min_speech_duration": 0.25,
                "parameters": {
                    "frame_duration_ms": 30,
                    "speech_confirmation_frames": 2,
                    "silence_confirmation_frames": 30,
                    "speech_buffer_size": 100
                }
            },
            "wake_word": {
                "enabled": False
            }
        },
        "wake-word": {
            "description": "Wake word activated - say 'jarvis' to start listening",
            "transcription": {
                "auto_start": False
            },
            "vad": {
                "detector_type": "combined",
                "sensitivity": 0.6,
                "enabled": True,
                "min_speech_duration": 0.25,
                "parameters": {
                    "frame_duration_ms": 30,
                    "speech_confirmation_frames": 2,
                    "silence_confirmation_frames": 30,
                    "speech_buffer_size": 100
                }
            },
            "wake_word": {
                "enabled": True,
                "words": ["jarvis"],
                "sensitivity": 0.7,
                "timeout": 30
            }
        }
    }
    
    def __init__(self, profiles_directory: str = "profiles/"):
        """
        Initialize the profile manager.
        
        Args:
            profiles_directory: Directory to store user profiles
        """
        self.logger = LoggingModule.get_logger(__name__)
        self.profiles_directory = profiles_directory
        
        # Ensure the profiles directory exists
        os.makedirs(profiles_directory, exist_ok=True)
    
    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a profile by name.
        
        This method first checks for a predefined profile with the given name,
        then checks for a user profile in the profiles directory.
        
        Args:
            name: Profile name
            
        Returns:
            Configuration dict for the profile, or None if not found
        """
        # Check for predefined profile
        if name in self.PREDEFINED_PROFILES:
            self.logger.debug(f"Loading predefined profile: {name}")
            return self.PREDEFINED_PROFILES[name]
        
        # Check for user profile
        profile_path = os.path.join(self.profiles_directory, f"{name}.json")
        if os.path.exists(profile_path):
            self.logger.debug(f"Loading user profile: {name}")
            try:
                with open(profile_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.error(f"Error parsing profile: {profile_path}")
                return None
        
        self.logger.warning(f"Profile not found: {name}")
        return None
    
    def save_profile(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Save a user profile.
        
        Args:
            name: Profile name
            config: Configuration to save
            
        Returns:
            True if successful, False otherwise
        """
        # Don't overwrite predefined profiles
        if name in self.PREDEFINED_PROFILES:
            self.logger.warning(f"Cannot save over predefined profile: {name}")
            return False
        
        try:
            profile_path = os.path.join(self.profiles_directory, f"{name}.json")
            with open(profile_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.debug(f"Profile saved: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving profile: {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """
        List all available profiles.
        
        Returns:
            List of profile names
        """
        profiles = list(self.PREDEFINED_PROFILES.keys())
        
        # Add user profiles
        try:
            for filename in os.listdir(self.profiles_directory):
                if filename.endswith('.json'):
                    profile_name = filename[:-5]  # Remove .json extension
                    if profile_name not in profiles:
                        profiles.append(profile_name)
        except FileNotFoundError:
            pass  # Profiles directory doesn't exist
        
        return sorted(profiles)
    
    def delete_profile(self, name: str) -> bool:
        """
        Delete a user profile.
        
        Args:
            name: Profile name
            
        Returns:
            True if successful, False otherwise
        """
        # Can't delete predefined profiles
        if name in self.PREDEFINED_PROFILES:
            self.logger.warning(f"Cannot delete predefined profile: {name}")
            return False
        
        profile_path = os.path.join(self.profiles_directory, f"{name}.json")
        if os.path.exists(profile_path):
            try:
                os.remove(profile_path)
                self.logger.debug(f"Profile deleted: {name}")
                return True
            except Exception as e:
                self.logger.error(f"Error deleting profile: {e}")
                return False
        
        self.logger.warning(f"Profile not found: {name}")
        return False