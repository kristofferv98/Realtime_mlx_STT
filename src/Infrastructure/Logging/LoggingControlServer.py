"""
LoggingControlServer for the logging configuration system.

This module provides a UDP server that listens for logging control commands,
allowing log levels to be changed at runtime without restarting the application.
"""

import json
import logging
import socket
import threading
import time
from typing import Dict, Optional, Union, Any, Callable

from .LoggingModule import LoggingModule
from .Models import LogLevel


class LoggingControlServer:
    """
    UDP server for listening to logging control commands.
    
    This server allows log levels to be changed at runtime by receiving
    commands via UDP. It runs in a background thread and can be started
    when the application initializes.
    """
    
    # Default port to listen on
    DEFAULT_PORT = 50101
    
    # Buffer size for UDP messages
    BUFFER_SIZE = 1024
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'LoggingControlServer':
        """
        Get the singleton instance of the server.
        
        Returns:
            LoggingControlServer: The singleton instance
        """
        if cls._instance is None:
            cls._instance = LoggingControlServer()
        return cls._instance
    
    def __init__(self, port: int = DEFAULT_PORT):
        """
        Initialize the logging control server.
        
        Args:
            port: Port to listen on for UDP commands
        """
        self.port = port
        self.running = False
        self.server_thread = None
        self.socket = None
        self.logger = logging.getLogger("realtimestt.infrastructure.logging.controlserver")
        
        # Command handlers
        self.command_handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {
            'set_level': self._handle_set_level,
            'get_levels': self._handle_get_levels
        }
    
    def start(self) -> bool:
        """
        Start the server in a background thread.
        
        Returns:
            bool: True if server was started, False if already running
        """
        if self.running:
            self.logger.warning("Server already running")
            return False
            
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            
            # Set timeout to allow for clean shutdown
            self.socket.settimeout(1.0)
            
            # Start server thread
            self.running = True
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="LoggingControlServer"
            )
            self.server_thread.start()
            
            self.logger.info(f"Logging control server started on port {self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start logging control server: {e}")
            self.running = False
            return False
    
    def stop(self) -> bool:
        """
        Stop the server and close the socket.
        
        Returns:
            bool: True if server was stopped, False if not running
        """
        if not self.running:
            self.logger.warning("Server not running")
            return False
            
        self.running = False
        
        # Wait for thread to terminate
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(2.0)
            
        # Close socket
        if self.socket:
            self.socket.close()
            self.socket = None
            
        self.logger.info("Logging control server stopped")
        return True
    
    def _run_server(self) -> None:
        """
        Run the server loop in a background thread.
        
        This method listens for incoming UDP messages and processes them.
        """
        self.logger.debug("Server thread started")
        
        while self.running:
            try:
                # Wait for data
                try:
                    data, addr = self.socket.recvfrom(self.BUFFER_SIZE)
                except socket.timeout:
                    # Timeout is expected, just continue the loop
                    continue
                except OSError:
                    # Socket closed or other error
                    if self.running:
                        self.logger.error("Socket error, stopping server")
                        self.running = False
                    break
                
                # Process data
                try:
                    self._process_command(data, addr)
                except Exception as e:
                    self.logger.error(f"Error processing command: {e}")
                    
            except Exception as e:
                self.logger.error(f"Unexpected error in server loop: {e}")
                # Continue running despite error
        
        self.logger.debug("Server thread exiting")
    
    def _process_command(self, data: bytes, addr: tuple) -> None:
        """
        Process a received command.
        
        Args:
            data: Raw command data
            addr: Address the command was received from
        """
        try:
            # Parse JSON command
            command = json.loads(data.decode('utf-8'))
            
            # Log receipt of command
            self.logger.debug(f"Received command from {addr}: {command}")
            
            # Check if command has required fields
            if 'action' not in command:
                self.logger.warning(f"Received command without 'action' field: {command}")
                return
                
            # Get handler for action
            action = command['action']
            if action in self.command_handlers:
                # Handle command
                self.command_handlers[action](command)
            else:
                self.logger.warning(f"Unknown action: {action}")
                
        except json.JSONDecodeError:
            self.logger.warning(f"Received invalid JSON: {data}")
        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
    
    def _handle_set_level(self, command: Dict[str, Any]) -> None:
        """
        Handle a set_level command.
        
        Args:
            command: The parsed command
        """
        if 'target' not in command or 'level' not in command:
            self.logger.warning(f"set_level command missing required fields: {command}")
            return
            
        target = command['target']
        level_str = command['level']
        
        try:
            # Convert level string to LogLevel
            level = LogLevel.from_string(level_str)
            
            # Set the level
            LoggingModule.set_level(target, level)
            
            self.logger.info(f"Set log level for {target} to {level.name}")
            
        except ValueError as e:
            self.logger.warning(f"Invalid log level: {level_str}")
        except Exception as e:
            self.logger.error(f"Error setting log level: {e}")
    
    def _handle_get_levels(self, command: Dict[str, Any]) -> None:
        """
        Handle a get_levels command.
        
        Args:
            command: The parsed command
        """
        # This is a placeholder for future functionality
        # Currently not implemented
        self.logger.warning("get_levels command not implemented yet")
        
    @classmethod
    def start_server(cls, port: int = DEFAULT_PORT) -> 'LoggingControlServer':
        """
        Start the logging control server.
        
        This is a convenience method for starting the server using the singleton pattern.
        
        Args:
            port: Port to listen on for UDP commands
            
        Returns:
            LoggingControlServer: The started server instance
        """
        server = cls.get_instance()
        if server.port != port:
            # If port changed, need to restart
            if server.running:
                server.stop()
            server.port = port
            
        if not server.running:
            server.start()
            
        return server