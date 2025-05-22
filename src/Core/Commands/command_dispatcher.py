"""
Command Dispatcher for the command-mediator pattern.

This module provides the CommandDispatcher class that routes commands to their appropriate handlers.
It serves as the central mediator between command senders and handlers.
"""

import logging
from typing import Any, Dict, Generic, List, Type, TypeVar, ForwardRef

from .command import Command

# Use forward reference
ICommandHandler = ForwardRef('ICommandHandler')

# Configure logging
logger = logging.getLogger(__name__)

# Generic type variable for the handler result
T = TypeVar('T')


class CommandDispatcher:
    """
    Central dispatcher that routes commands to their appropriate handlers.
    
    The CommandDispatcher maintains a registry of command types and their handlers.
    It serves as the mediator between command senders and handlers.
    """
    
    def __init__(self):
        """Initialize the command dispatcher with an empty handler registry."""
        self._handlers: Dict[Type[Command], List['ICommandHandler']] = {}
    
    def register_handler(self, command_type: Type[Command], handler: 'ICommandHandler') -> None:
        """
        Register a handler for a specific command type.
        
        Args:
            command_type: The type of command the handler can process
            handler: The command handler to register
            
        Raises:
            TypeError: If command_type is not a Command subclass or if handler is not an ICommandHandler
        """
        if not issubclass(command_type, Command):
            raise TypeError(f"Expected Command subclass, got {command_type.__name__}")
        
        # Import here to avoid circular import at module level
        from ..Common.Interfaces.command_handler import ICommandHandler
        if not isinstance(handler, ICommandHandler):
            raise TypeError(f"Expected ICommandHandler, got {type(handler).__name__}")
        
        if command_type not in self._handlers:
            self._handlers[command_type] = []
        
        self._handlers[command_type].append(handler)
        logger.debug(f"Registered handler for {command_type.__name__}: {handler.__class__.__name__}")
    
    def dispatch(self, command: Command) -> List[Any]:
        """
        Dispatch a command to all registered handlers for its type.
        
        Args:
            command: The command to dispatch
            
        Returns:
            List of results from all handlers that processed the command
            
        Raises:
            TypeError: If command is not a Command instance
            ValueError: If no handlers are registered for the command type
        """
        if not isinstance(command, Command):
            raise TypeError(f"Expected Command instance, got {type(command).__name__}")
        
        command_type = type(command)
        handlers = self._get_handlers_for_command_type(command_type)
        
        if not handlers:
            raise ValueError(f"No handlers registered for command type: {command_type.__name__}")
        
        results = []
        for handler in handlers:
            if handler.can_handle(command):
                try:
                    result = handler.handle(command)
                    results.append(result)
                    logger.debug(f"Command {command.name} handled by {handler.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Error handling command {command.name} with {handler.__class__.__name__}: {str(e)}", 
                                exc_info=True)
                    # Re-raise the exception to let the caller handle it
                    raise
        
        return results
    
    def dispatch_to_first_handler(self, command: Command) -> Any:
        """
        Dispatch a command to the first matching handler.
        
        Args:
            command: The command to dispatch
            
        Returns:
            Result from the first handler that processed the command
            
        Raises:
            TypeError: If command is not a Command instance
            ValueError: If no handlers are registered for the command type or if no handler can handle the command
        """
        if not isinstance(command, Command):
            raise TypeError(f"Expected Command instance, got {type(command).__name__}")
        
        command_type = type(command)
        handlers = self._get_handlers_for_command_type(command_type)
        
        if not handlers:
            raise ValueError(f"No handlers registered for command type: {command_type.__name__}")
        
        for handler in handlers:
            if handler.can_handle(command):
                try:
                    result = handler.handle(command)
                    logger.debug(f"Command {command.name} handled by {handler.__class__.__name__}")
                    return result
                except Exception as e:
                    logger.error(f"Error handling command {command.name} with {handler.__class__.__name__}: {str(e)}", 
                                exc_info=True)
                    # Re-raise the exception to let the caller handle it
                    raise
        
        raise ValueError(f"No handler can handle command: {command.name}")
    
    def _get_handlers_for_command_type(self, command_type: Type[Command]) -> List[ICommandHandler]:
        """
        Get all handlers for a specific command type.
        
        This includes handlers registered for the command type and any of its parent types.
        
        Args:
            command_type: The command type to get handlers for
            
        Returns:
            List of handler instances for the command type
        """
        # Get all parent command types
        command_types = [cls for cls in command_type.__mro__ 
                        if cls is Command or (cls is not object and issubclass(cls, Command))]
        
        # Collect handlers for the command type and all its parent types
        all_handlers = []
        for cls in command_types:
            if cls in self._handlers:
                all_handlers.extend(self._handlers[cls])
        
        return all_handlers
    
    def unregister_handler(self, command_type: Type[Command], handler: 'ICommandHandler') -> bool:
        """
        Unregister a handler for a specific command type.
        
        Args:
            command_type: The type of command
            handler: The handler to unregister
            
        Returns:
            bool: True if the handler was successfully unregistered, False otherwise
        """
        if command_type in self._handlers and handler in self._handlers[command_type]:
            self._handlers[command_type].remove(handler)
            logger.debug(f"Unregistered handler for {command_type.__name__}: {handler.__class__.__name__}")
            return True
        return False
    
    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        logger.debug("All command handlers cleared")
    
    def clear_handlers_for_type(self, command_type: Type[Command]) -> None:
        """
        Clear all handlers for a specific command type.
        
        Args:
            command_type: The command type to clear handlers for
        """
        if command_type in self._handlers:
            self._handlers[command_type].clear()
            logger.debug(f"Cleared all handlers for {command_type.__name__}")
    
    def get_handler_count(self, command_type: Type[Command] = None) -> int:
        """
        Get the count of registered handlers, optionally for a specific command type.
        
        Args:
            command_type: Optional specific command type to count handlers for
            
        Returns:
            int: Count of handlers
        """
        if command_type is not None:
            return len(self._handlers.get(command_type, []))
        
        return sum(len(handlers) for handlers in self._handlers.values())