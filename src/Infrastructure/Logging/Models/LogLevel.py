"""
LogLevel enum for the logging configuration system.

This module provides a strongly-typed enum for log levels and utilities
for converting between string representations and LogLevel values.
"""

import logging
from enum import Enum, auto


class LogLevel(Enum):
    """
    Enum representing log levels for the logging system.
    
    Maps to standard Python logging levels with enum values.
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    def __lt__(self, other):
        """Compare log levels - lower values are more verbose (DEBUG < INFO)."""
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __gt__(self, other):
        """Compare log levels - higher values are less verbose (ERROR > INFO)."""
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __le__(self, other):
        """Less than or equal comparison for log levels."""
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __ge__(self, other):
        """Greater than or equal comparison for log levels."""
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    
    @classmethod
    def from_string(cls, level_str: str) -> "LogLevel":
        """
        Convert a string log level to a LogLevel enum value.
        
        Args:
            level_str: String representation of log level (case-insensitive)
            
        Returns:
            LogLevel: The corresponding LogLevel enum value
            
        Raises:
            ValueError: If the string doesn't match a valid log level
        """
        level_str = level_str.upper()
        try:
            return cls[level_str]
        except KeyError:
            # Try to convert a numeric string
            try:
                level_int = int(level_str)
                # Find the enum value that matches this int value
                for level in cls:
                    if level.value == level_int:
                        return level
                # No match found
                raise ValueError(f"Invalid log level value: {level_str}")
            except ValueError:
                # Not a number or enum name
                raise ValueError(f"Unknown log level: {level_str}")
    
    @classmethod
    def to_logging_level(cls, level: "LogLevel") -> int:
        """
        Convert a LogLevel enum value to a Python logging level integer.
        
        Args:
            level: LogLevel enum value
            
        Returns:
            int: Python logging level constant
        """
        return level.value