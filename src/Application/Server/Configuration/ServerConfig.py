"""
ServerConfig for Realtime_mlx_STT Server

This module provides configuration management for the server,
allowing settings to be loaded from environment variables or files.
"""

import os
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from src.Infrastructure.Logging.LoggingModule import LoggingModule

@dataclass
class ServerConfig:
    """
    Configuration settings for the server.
    
    This class defines the configuration options for the server,
    with defaults that can be overridden from environment variables or files.
    """
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    auto_start: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Security settings
    auth_enabled: bool = False
    auth_token: Optional[str] = None
    
    # Profile settings
    profiles_directory: str = "profiles/"
    default_profile: str = "default"
    
    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """
        Create configuration from environment variables.
        
        This method creates a ServerConfig instance with values from 
        environment variables, falling back to defaults if not set.
        
        Returns:
            A ServerConfig instance with values from environment
        """
        logger = LoggingModule.get_logger(__name__)
        logger.debug("Loading server configuration from environment")
        
        config = cls()
        
        # Server settings
        config.host = os.environ.get("STT_SERVER_HOST", config.host)
        try:
            config.port = int(os.environ.get("STT_SERVER_PORT", config.port))
        except ValueError:
            logger.warning(f"Invalid STT_SERVER_PORT: {os.environ.get('STT_SERVER_PORT')}. Using default.")
        
        config.debug = os.environ.get("STT_SERVER_DEBUG", "").lower() == "true"
        config.auto_start = os.environ.get("STT_SERVER_AUTO_START", "true").lower() == "true"
        
        # CORS settings
        cors_env = os.environ.get("STT_SERVER_CORS_ORIGINS")
        if cors_env:
            config.cors_origins = cors_env.split(",")
        
        # Security settings
        config.auth_enabled = os.environ.get("STT_SERVER_AUTH_ENABLED", "").lower() == "true"
        config.auth_token = os.environ.get("STT_SERVER_AUTH_TOKEN", config.auth_token)
        
        # Profile settings
        config.profiles_directory = os.environ.get("STT_SERVER_PROFILES_DIR", config.profiles_directory)
        config.default_profile = os.environ.get("STT_SERVER_DEFAULT_PROFILE", config.default_profile)
        
        return config
    
    @classmethod
    def from_file(cls, path: str) -> 'ServerConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            A ServerConfig instance with values from the file
        """
        logger = LoggingModule.get_logger(__name__)
        logger.debug(f"Loading server configuration from file: {path}")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            config = cls()
            
            # Get server section or use empty dict if not found
            server_data = data.get("server", {})
            
            # Update fields from file
            for key, value in server_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            return config
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration file: {e}")
            # Fall back to defaults with environment overrides
            return cls.from_env()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dict representation of the configuration
        """
        return asdict(self)
    
    def save_to_file(self, path: str) -> bool:
        """
        Save configuration to a JSON file.
        
        Args:
            path: Path to save the configuration to
            
        Returns:
            True if successful, False otherwise
        """
        logger = LoggingModule.get_logger(__name__)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Convert to dict and save as JSON
            config_dict = {"server": self.to_dict()}
            
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.debug(f"Configuration saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {e}")
            return False