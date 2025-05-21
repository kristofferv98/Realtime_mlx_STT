"""
LoggingConfig for the logging configuration system.

This module provides a data class to store and manage logging configuration settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .Models import LogLevel, LogHandler, LogFormat


@dataclass
class LoggingConfig:
    """
    Configuration settings for the logging system.
    
    This class holds all configuration parameters for the logging system,
    including log levels, handler settings, and formatting options.
    """
    # Root logger configuration
    root_level: LogLevel = LogLevel.INFO
    
    # Feature-specific log levels
    feature_levels: Dict[str, LogLevel] = field(default_factory=dict)
    
    # Console handler configuration
    console_enabled: bool = True
    console_level: LogLevel = LogLevel.INFO
    console_format: Union[str, LogFormat] = LogFormat.STANDARD
    
    # File handler configuration
    file_enabled: bool = False
    file_level: LogLevel = LogLevel.DEBUG
    file_path: str = "logs/realtimestt.log"
    file_format: Union[str, LogFormat] = LogFormat.STANDARD
    
    # Rotation settings
    rotation_enabled: bool = False
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    
    # Namespace configuration
    root_namespace: str = "realtimestt"
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # We'll keep the enum objects for type safety, conversion happens at formatter creation time
        # No need to convert here since LoggingConfigurer handles this
            
    def get_feature_level(self, feature_name: str) -> LogLevel:
        """
        Get the log level for a specific feature.
        
        Args:
            feature_name: Name of the feature to get level for
            
        Returns:
            LogLevel: The log level for the feature, or the root level if not specified
        """
        return self.feature_levels.get(feature_name, self.root_level)
    
    def set_feature_level(self, feature_name: str, level: LogLevel) -> None:
        """
        Set the log level for a specific feature.
        
        Args:
            feature_name: Name of the feature to set level for
            level: Log level to set
        """
        self.feature_levels[feature_name] = level
        
    def set_level(self, level: LogLevel) -> None:
        """
        Set the root log level.
        
        Args:
            level: Log level to set for the root logger
        """
        self.root_level = level
        
    @classmethod
    def create_default(cls) -> "LoggingConfig":
        """
        Create a default logging configuration.
        
        Returns:
            LoggingConfig: A new instance with default settings
        """
        return cls()
    
    @classmethod
    def create_development(cls) -> "LoggingConfig":
        """
        Create a development-oriented logging configuration.
        
        Returns:
            LoggingConfig: A new instance with development-friendly settings
        """
        return cls(
            root_level=LogLevel.DEBUG,
            console_enabled=True,
            console_level=LogLevel.DEBUG,
            console_format=LogFormat.DEVELOPMENT,
            file_enabled=True,
            file_level=LogLevel.DEBUG,
            file_path="logs/realtimestt_dev.log",
            file_format=LogFormat.DETAILED,
            rotation_enabled=True
        )
    
    @classmethod
    def create_production(cls) -> "LoggingConfig":
        """
        Create a production-oriented logging configuration.
        
        Returns:
            LoggingConfig: A new instance with production-friendly settings
        """
        return cls(
            root_level=LogLevel.INFO,
            console_enabled=True,
            console_level=LogLevel.INFO,
            console_format=LogFormat.STANDARD,
            file_enabled=True,
            file_level=LogLevel.DEBUG,
            file_path="logs/realtimestt.log",
            file_format=LogFormat.STANDARD,
            rotation_enabled=True,
            max_bytes=52428800,  # 50MB
            backup_count=10
        )