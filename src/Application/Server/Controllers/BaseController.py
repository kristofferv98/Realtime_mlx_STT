"""
BaseController for Realtime_mlx_STT Server

This module provides a base controller class that can be extended 
by specific feature controllers to handle API endpoints.
"""

from typing import Dict, Any, Callable, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Infrastructure.Logging.LoggingModule import LoggingModule

class BaseController:
    """
    Base controller class for API endpoints.
    
    This class provides common functionality for controllers and should be
    extended by feature-specific controllers like TranscriptionController,
    AudioController, etc.
    """
    
    def __init__(self, command_dispatcher: CommandDispatcher, event_bus: EventBus, prefix: str = ""):
        """
        Initialize the base controller.
        
        Args:
            command_dispatcher: Command dispatcher to use for sending commands
            event_bus: Event bus for subscribing to events
            prefix: URL prefix for all routes in this controller
        """
        self.logger = LoggingModule.get_logger(__name__)
        self.command_dispatcher = command_dispatcher
        self.event_bus = event_bus
        self.router = APIRouter(prefix=prefix)
        
        # Register routes for this controller
        self._register_routes()
    
    def _register_routes(self):
        """
        Register routes for this controller.
        
        This method should be overridden by subclasses to add specific routes.
        """
        pass
    
    def send_command(self, command) -> Any:
        """
        Send a command through the command dispatcher.
        
        Args:
            command: The command to send
            
        Returns:
            The result of the command execution
            
        Raises:
            HTTPException: If there's an error executing the command
        """
        try:
            result = self.command_dispatcher.dispatch(command)
            return result
        except Exception as e:
            self.logger.error(f"Error dispatching command: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error executing command: {str(e)}"
            )
    
    def create_standard_response(self, status_code: str = "success", data: Any = None, 
                                 message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a standardized response format.
        
        Args:
            status_code: Status code string ("success", "error", etc.)
            data: Response data
            message: Optional message
            
        Returns:
            Standard response dictionary
        """
        response = {
            "status": status_code,
        }
        
        if data is not None:
            response["data"] = data
            
        if message is not None:
            response["message"] = message
            
        return response