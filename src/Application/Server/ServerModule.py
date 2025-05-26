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
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Features.Transcription.Events.TranscriptionUpdatedEvent import TranscriptionUpdatedEvent
from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
from src.Infrastructure.Logging.LoggingModule import LoggingModule

from .WebSocket.WebSocketManager import WebSocketManager
from .Configuration.ServerConfig import ServerConfig
from .Configuration.ProfileManager import ProfileManager
from .Controllers.TranscriptionController import TranscriptionController
from .Controllers.SystemController import SystemController

class Server:
    """Server implementation that integrates with the existing command/event system."""
    
    def __init__(self, command_dispatcher: CommandDispatcher, event_bus: EventBus, 
                 host: str = "127.0.0.1", port: int = 8080,
                 cors_origins: List[str] = None):
        """
        Initialize the server.
        
        Args:
            command_dispatcher: The command dispatcher to use
            event_bus: The event bus to use
            host: The host to bind to
            port: The port to bind to
            cors_origins: List of allowed CORS origins
        """
        self.logger = LoggingModule.get_logger(__name__)
        self.app = FastAPI(title="Speech-to-Text API")
        self.command_dispatcher = command_dispatcher
        self.event_bus = event_bus
        self.host = host
        self.port = port
        self.websocket_manager = WebSocketManager()
        self.running = False
        self.server_thread = None
        
        # Setup profile manager
        self.profile_manager = ProfileManager()
        
        # Set up CORS
        if cors_origins is None:
            cors_origins = ["*"]
            
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register controllers
        self._register_controllers()
        
        # Subscribe to events
        self._register_event_handlers()
    
    def _register_controllers(self):
        """Register API controllers."""
        self.logger.info("Registering API controllers")
        
        # Create controllers
        transcription_controller = TranscriptionController(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus
        )
        
        system_controller = SystemController(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus,
            profile_manager=self.profile_manager
        )
        
        # Include routers in the app
        self.app.include_router(transcription_controller.router)
        self.app.include_router(system_controller.router)
        
        # Setup WebSocket endpoint
        @self.app.websocket("/events")
        async def websocket_endpoint(websocket: WebSocket):
            """Handle WebSocket connections."""
            await websocket.accept()
            self.websocket_manager.register(websocket)
            
            # Set the event loop for the WebSocket manager if not already set
            try:
                loop = asyncio.get_running_loop()
                self.websocket_manager.set_event_loop(loop)
            except:
                pass
            
            try:
                while True:
                    data = await websocket.receive_json()
                    # Handle incoming WebSocket commands
                    # This will be expanded as needed
                    self.logger.debug(f"Received WebSocket message: {data}")
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
        # More event handlers will be added as needed
    
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
            "word": event.wake_word,
            "confidence": event.confidence,
            "timestamp": event.audio_timestamp
        })
    
    def start(self):
        """Start the server in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=False
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
        logger = LoggingModule.get_logger(__name__)
        logger.info("Registering server module")
        
        if config is None:
            config = ServerConfig.from_env()
        
        # Create server instance
        server = Server(
            command_dispatcher=command_dispatcher,
            event_bus=event_bus,
            host=config.host,
            port=config.port,
            cors_origins=config.cors_origins
        )
        
        # Start the server if auto_start is enabled
        if config.auto_start:
            server.start()
        
        return server