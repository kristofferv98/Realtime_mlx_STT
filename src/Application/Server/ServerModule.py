"""
ServerModule for Realtime_mlx_STT

This module provides a FastAPI-based server that exposes the speech-to-text
functionality via HTTP and WebSocket APIs. The server integrates with the
existing command/event architecture without modifying core functionality.
"""

from typing import Optional, Dict, Any, List
import threading
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Features.Transcription.Events.TranscriptionUpdatedEvent import TranscriptionUpdatedEvent
from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
from src.Infrastructure.Logging.LoggingModule import get_logger

from .WebSocket.WebSocketManager import WebSocketManager
from .Configuration.ServerConfig import ServerConfig

class Server:
    """Server implementation that integrates with the existing command/event system."""
    
    def __init__(self, command_dispatcher: CommandDispatcher, event_bus: EventBus, 
                 host: str = "127.0.0.1", port: int = 8080):
        """
        Initialize the server.
        
        Args:
            command_dispatcher: The command dispatcher to use
            event_bus: The event bus to use
            host: The host to bind to
            port: The port to bind to
        """
        self.logger = get_logger(__name__)
        self.app = FastAPI(title="Speech-to-Text API")
        self.command_dispatcher = command_dispatcher
        self.event_bus = event_bus
        self.host = host
        self.port = port
        self.websocket_manager = WebSocketManager()
        self.running = False
        self.server_thread = None
        
        # Set up CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Set up routes
        self._setup_routes()
        
        # Subscribe to events
        self._register_event_handlers()

    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.get("/status")
        async def get_status():
            """Get system status."""
            return {"status": "online"}
        
        @self.app.post("/config")
        async def update_config(config: Dict[str, Any]):
            """Update system configuration."""
            # Will be implemented by specific controllers
            return {"status": "success"}
        
        @self.app.post("/start")
        async def start_processing():
            """Start speech processing."""
            # Will be implemented by specific controllers
            return {"status": "started"}
        
        @self.app.post("/stop")
        async def stop_processing():
            """Stop speech processing."""
            # Will be implemented by specific controllers
            return {"status": "stopped"}
        
        @self.app.websocket("/events")
        async def websocket_endpoint(websocket: WebSocket):
            """Handle WebSocket connections."""
            await websocket.accept()
            self.websocket_manager.register(websocket)
            try:
                while True:
                    data = await websocket.receive_json()
                    # Handle incoming WebSocket commands - will be implemented
                    pass
            except WebSocketDisconnect:
                self.websocket_manager.unregister(websocket)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                self.websocket_manager.unregister(websocket)
    
    def _register_event_handlers(self):
        """Register handlers for system events."""
        # These handlers will broadcast events to WebSocket clients
        self.event_bus.subscribe(TranscriptionUpdatedEvent, self.handle_transcription_update)
        self.event_bus.subscribe(WakeWordDetectedEvent, self.handle_wake_word_detected)
        # More event handlers will be added
    
    def handle_transcription_update(self, event: TranscriptionUpdatedEvent):
        """Handle transcription update events."""
        self.websocket_manager.broadcast_event("transcription", {
            "text": event.text,
            "is_final": event.is_final,
            "session_id": event.session_id
        })
    
    def handle_wake_word_detected(self, event: WakeWordDetectedEvent):
        """Handle wake word detection events."""
        self.websocket_manager.broadcast_event("wake_word", {
            "word": event.word,
            "timestamp": event.timestamp
        })
    
    def start(self):
        """Start the server in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        self.logger.info(f"Server started on http://{self.host}:{self.port}")
    
    def _run_server(self):
        """Run the server."""
        try:
            uvicorn.run(self.app, host=self.host, port=self.port)
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            self.running = False
    
    def stop(self):
        """Stop the server."""
        if not self.running:
            return
        
        self.running = False
        # Proper shutdown would require more complex logic with uvicorn
        self.logger.info("Server stopped")


class ServerModule:
    """
    Server module that registers with the system.
    
    This module follows the same pattern as other features in the system,
    using the register method to integrate with the command/event architecture.
    """
    
    @staticmethod
    def register(command_dispatcher: CommandDispatcher, event_bus: EventBus, 
                 config: Optional[ServerConfig] = None) -> Server:
        """
        Register the server module with the system.
        
        Args:
            command_dispatcher: The command dispatcher to use
            event_bus: The event bus to use
            config: Optional server configuration
            
        Returns:
            The server instance
        """
        logger = get_logger(__name__)
        logger.info("Registering server module")
        
        if config is None:
            config = ServerConfig.from_env()
        
        # Create server instance
        server = Server(
            command_dispatcher=command_dispatcher,
            event_bus=event_bus,
            host=config.host,
            port=config.port
        )
        
        # Start the server if auto_start is enabled
        if config.auto_start:
            server.start()
        
        return server