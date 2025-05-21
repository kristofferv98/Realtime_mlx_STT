"""
LogHandler enum for the logging configuration system.

This module provides a strongly-typed enum for log handler types and utilities
for creating and configuring different handler types.
"""

from enum import Enum, auto


class LogHandler(Enum):
    """
    Enum representing types of log handlers.
    
    Provides a type-safe way to specify which handler to configure.
    """
    CONSOLE = auto()     # Console (stdout) handler
    FILE = auto()        # Regular file handler
    ROTATING = auto()    # Rotating file handler
    
    @classmethod
    def from_string(cls, handler_str: str) -> "LogHandler":
        """
        Convert a string handler type to a LogHandler enum value.
        
        Args:
            handler_str: String representation of handler type (case-insensitive)
            
        Returns:
            LogHandler: The corresponding LogHandler enum value
            
        Raises:
            ValueError: If the string doesn't match a valid handler type
        """
        handler_str = handler_str.upper()
        try:
            return cls[handler_str]
        except KeyError:
            raise ValueError(f"Unknown log handler type: {handler_str}")
    
    def __str__(self) -> str:
        """Return string representation of the handler type."""
        return self.name