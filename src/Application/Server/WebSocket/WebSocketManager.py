"""
WebSocketManager for Realtime_mlx_STT Server

This module provides WebSocket connection management and event broadcasting
for the server, enabling real-time communication with clients.
"""

from typing import Dict, Any, Set
import json
import asyncio
from fastapi import WebSocket
from src.Infrastructure.Logging.LoggingModule import LoggingModule

class WebSocketManager:
    """
    Manages WebSocket connections and event broadcasting.
    
    This class handles client connections and provides methods for
    broadcasting events to all connected clients.
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: Set[WebSocket] = set()
        self.logger = LoggingModule.get_logger(__name__)
    
    def register(self, websocket: WebSocket):
        """
        Register a new WebSocket client.
        
        Args:
            websocket: The WebSocket connection to register
        """
        self.active_connections.add(websocket)
        self.logger.debug(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
    
    def unregister(self, websocket: WebSocket):
        """
        Unregister a WebSocket client.
        
        Args:
            websocket: The WebSocket connection to unregister
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.debug(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """
        Send a message to a specific client.
        
        Args:
            message: The message to send
            websocket: The WebSocket connection to send to
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            self.logger.error(f"Error sending personal message: {e}")
            self.unregister(websocket)
    
    async def _broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all connected clients asynchronously.
        
        Args:
            message: The message to broadcast
        """
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.unregister(connection)
    
    def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcast an event to all connected clients.
        
        This method creates a task to handle the broadcast asynchronously
        since it might be called from synchronous event handlers.
        
        Args:
            event_type: The type of event
            data: The event data
        """
        message = {
            "event": event_type,
            **data
        }
        
        self.logger.debug(f"Broadcasting event: {event_type}")
        
        # Create a task to broadcast the message asynchronously
        # This approach allows broadcasting from synchronous code
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._broadcast(message))
        except RuntimeError:
            # If there's no event loop (e.g., in a thread), log the error
            self.logger.error("Cannot broadcast message: No event loop running")