"""
Commands module for the Realtime_mlx_STT project.

This module provides command-related functionality for the command-mediator pattern,
including the base Command class and CommandDispatcher implementation.
"""

# Import directly in Core/__init__.py to avoid circular imports
# Keep this for backward compatibility
from .command import Command
from .command_dispatcher import CommandDispatcher

__all__ = [
    'Command',
    'CommandDispatcher'
]