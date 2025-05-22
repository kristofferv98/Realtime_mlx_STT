"""
WebSocketManager for Realtime_mlx_STT Server

This module provides WebSocket connection management and event broadcasting
for the server, enabling real-time communication with clients.
"""

from typing import Dict, Any, Set, Optional
import json
import asyncio
import threading
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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._broadcast_queue: asyncio.Queue = None
    
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
    
    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """
        Set the event loop for thread-safe broadcasting.
        
        Args:
            loop: The asyncio event loop from the main thread
        """
        self._loop = loop
        self.logger.debug("Event loop set for WebSocket manager")
    
    def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcast an event to all connected clients.
        
        This method handles broadcasting from any thread by using
        asyncio.run_coroutine_threadsafe when called from a different thread.
        
        Args:
            event_type: The type of event
            data: The event data
        """
        message = {
            "event": event_type,
            **data
        }
        
        self.logger.debug(f"Broadcasting event: {event_type}")
        
        # Try to get the current event loop
        try:
            current_loop = asyncio.get_running_loop()
            # We're in an async context, create task directly
            current_loop.create_task(self._broadcast(message))
        except RuntimeError:
            # We're not in an async context (different thread)
            if self._loop and self._loop.is_running():
                # Use thread-safe method to schedule the coroutine
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(message),
                    self._loop
                )
            else:
                self.logger.error("Cannot broadcast message: No event loop available")