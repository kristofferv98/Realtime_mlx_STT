"""
LogFormat enum for the logging configuration system.

This module provides predefined log formats that can be used in the logging configuration.
"""

from enum import Enum


class LogFormat(Enum):
    """
    Enum representing predefined log formats.
    
    Provides common format strings ready to use with logging handlers.
    """
    # Basic format with timestamp, level and message
    BASIC = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Standard format with timestamp, logger name, level and message
    STANDARD = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Detailed format with timestamp, logger name, level, file location and message
    DETAILED = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    
    # Development format with thread info and more details for debugging
    DEVELOPMENT = "%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(filename)s:%(lineno)d - %(message)s"
    
    # Minimal format with just level and message
    MINIMAL = "%(levelname)s: %(message)s"
    
    @classmethod
    def from_string(cls, format_str: str) -> "LogFormat":
        """
        Convert a string format name to a LogFormat enum value.
        
        Args:
            format_str: String representation of format name (case-insensitive)
            
        Returns:
            LogFormat: The corresponding LogFormat enum value
            
        Raises:
            ValueError: If the string doesn't match a valid format name
        """
        format_str = format_str.upper()
        try:
            return cls[format_str]
        except KeyError:
            raise ValueError(f"Unknown log format: {format_str}")
    
    def __str__(self) -> str:
        """Return the format string."""
        return self.value