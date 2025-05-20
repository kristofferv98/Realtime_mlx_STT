"""
Command Handler interface for the command-mediator pattern.

This module defines the ICommandHandler interface that abstracts the command handling
functionality in the system. It enables decoupling between command senders and handlers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar, ForwardRef

# Use forward reference for Command
Command = ForwardRef('Command')

# Generic type variable for the command handler result
T = TypeVar('T')


class ICommandHandler(Generic[T], ABC):
    """
    Interface for command handlers that process commands and produce results.
    
    The command handler implements the mediator pattern, decoupling command
    senders from command processors.
    
    Type parameter T represents the result type of the command handler.
    """
    
    @abstractmethod
    def handle(self, command: Command) -> T:
        """
        Handle a command and produce a result.
        
        Args:
            command: The command to handle
            
        Returns:
            The result of the command execution, type T
            
        Raises:
            TypeError: If the command is not of the expected type
            ValueError: If the command contains invalid data
            Exception: If an error occurs during command execution
        """
        pass
    
    @abstractmethod
    def can_handle(self, command: Command) -> bool:
        """
        Check if this handler can handle the given command.
        
        Args:
            command: The command to check
            
        Returns:
            bool: True if this handler can handle the command, False otherwise
        """
        pass