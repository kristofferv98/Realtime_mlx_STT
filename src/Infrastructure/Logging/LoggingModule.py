"""
LoggingModule for the logging configuration system.

This module provides a public facade for configuring and using the logging system,
following the same pattern as other modules in the project.
"""

import logging
import os
from typing import Dict, Optional, Union, Any

from .LoggingConfig import LoggingConfig
from .LoggingConfigurer import LoggingConfigurer
from .Models import LogLevel, LogFormat

# Import here to avoid circular import
# The import is at the end of the file
# LoggingControlServer = None


class LoggingModule:
    """
    Module for logging configuration functionality.
    
    This class provides a simplified interface for configuring and using
    the logging system, following the same pattern as other modules in the project.
    """
    
    # Control server management
    _control_server_started = False
    _control_server_port = 50101
    
    @classmethod
    def initialize(
        cls,
        console_level: Union[str, LogLevel] = "INFO",
        file_level: Union[str, LogLevel] = "DEBUG",
        file_enabled: bool = True,
        file_path: Optional[str] = None,
        rotation_enabled: bool = True,
        backup_count: int = 5,
        max_bytes: int = 10485760,  # 10MB
        console_format: Union[str, LogFormat] = LogFormat.STANDARD,
        file_format: Union[str, LogFormat] = LogFormat.STANDARD,
        feature_levels: Dict[str, Union[str, LogLevel]] = None,
        start_control_server: bool = False,
        control_server_port: int = 50101
    ) -> LoggingConfig:
        """
        Initialize the logging system with the specified configuration.
        
        Args:
            console_level: Log level for console output
            file_level: Log level for file output
            file_enabled: Whether to enable file logging
            file_path: Path to log file (default: logs/realtimestt.log)
            rotation_enabled: Whether to enable log rotation
            backup_count: Number of backup files to keep
            max_bytes: Maximum file size before rotation
            console_format: Format for console output
            file_format: Format for file output
            feature_levels: Dict of feature names to log levels
            start_control_server: Whether to start the logging control server for runtime configuration
            control_server_port: Port for the control server to listen on
            
        Returns:
            LoggingConfig: The created and applied configuration
        """
        # Create configuration
        config = LoggingConfig()
        
        # Convert string levels to LogLevel
        if isinstance(console_level, str):
            config.console_level = LogLevel.from_string(console_level)
        else:
            config.console_level = console_level
            
        if isinstance(file_level, str):
            config.file_level = LogLevel.from_string(file_level)
        else:
            config.file_level = file_level
            
        # Set the root level to the more verbose of console and file
        # In logging, lower numeric values are more verbose (DEBUG=10, INFO=20, etc.)
        try:
            config.root_level = min(config.console_level, config.file_level)
        except TypeError as e:
            # If we got here, we might be dealing with a comparison issue
            # Let's ensure we're comparing LogLevel enum values
            logger = logging.getLogger("realtimestt.infrastructure.logging")
            logger.warning(f"Error determining minimum log level: {e}. Using console_level as fallback.")
            config.root_level = config.console_level
        
        # Apply other settings
        config.file_enabled = file_enabled
        
        if file_path:
            config.file_path = file_path
            
        config.rotation_enabled = rotation_enabled
        config.backup_count = backup_count
        config.max_bytes = max_bytes
        
        # Set formats
        if isinstance(console_format, str) and hasattr(LogFormat, console_format.upper()):
            config.console_format = getattr(LogFormat, console_format.upper())
        else:
            config.console_format = console_format
            
        if isinstance(file_format, str) and hasattr(LogFormat, file_format.upper()):
            config.file_format = getattr(LogFormat, file_format.upper())
        else:
            config.file_format = file_format
        
        # Set feature levels
        if feature_levels:
            for feature, level in feature_levels.items():
                if isinstance(level, str):
                    config.feature_levels[feature] = LogLevel.from_string(level)
                else:
                    config.feature_levels[feature] = level
        
        # Ensure log directory exists
        if file_enabled and file_path:
            log_dir = os.path.dirname(file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        
        # Apply configuration
        LoggingConfigurer.configure(config)
        
        # Start control server if requested
        if start_control_server:
            cls._control_server_port = control_server_port
            cls.start_control_server(control_server_port)
        
        return config
    
    @classmethod
    def initialize_from_env(cls, start_control_server: bool = False, control_server_port: int = 50101) -> LoggingConfig:
        """
        Initialize the logging system from environment variables.
        
        Args:
            start_control_server: Whether to start the logging control server
            control_server_port: Port for the control server to listen on
            
        Returns:
            LoggingConfig: The created and applied configuration
        """
        config = LoggingConfigurer.configure_from_env()
        
        # Check for control server settings in environment
        env_control_server = os.environ.get('REALTIMESTT_LOG_CONTROL_SERVER')
        if env_control_server:
            start_control_server = env_control_server.lower() in ('true', 'yes', '1')
            
        env_control_port = os.environ.get('REALTIMESTT_LOG_CONTROL_PORT')
        if env_control_port:
            try:
                control_server_port = int(env_control_port)
            except ValueError:
                pass  # Ignore invalid port
                
        # Start control server if requested
        if start_control_server:
            cls._control_server_port = control_server_port
            cls.start_control_server(control_server_port)
            
        return config
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger with a standardized namespace.
        
        Args:
            name: The logger name or module name to standardize
            
        Returns:
            logging.Logger: A properly namespaced logger
        """
        return LoggingConfigurer.get_logger(name)
    
    @staticmethod
    def set_level(feature_or_logger: str, level: Union[str, LogLevel]) -> None:
        """
        Set the log level for a specific feature or logger.
        
        Args:
            feature_or_logger: Name of the feature or logger to set level for
            level: Log level to set
        """
        # Convert string level to LogLevel
        if isinstance(level, str):
            level_enum = LogLevel.from_string(level)
        else:
            level_enum = level
            
        # If the name looks like a feature (no periods, not 'root')
        if '.' not in feature_or_logger and feature_or_logger.lower() != 'root':
            LoggingConfigurer.set_feature_level(feature_or_logger, level_enum)
        elif feature_or_logger.lower() == 'root':
            LoggingConfigurer.set_root_level(level_enum)
        else:
            # Get the logger and set its level directly
            logger = logging.getLogger(feature_or_logger)
            logger.setLevel(level_enum.value)
    
    @classmethod
    def create_development_config(cls, start_control_server: bool = True) -> LoggingConfig:
        """
        Create a development-oriented logging configuration.
        
        Args:
            start_control_server: Whether to start the logging control server
        
        Returns:
            LoggingConfig: Configuration suitable for development
        """
        config = LoggingConfig.create_development()
        LoggingConfigurer.configure(config)
        
        # Start control server if requested
        if start_control_server:
            cls.start_control_server()
            
        return config
    
    @classmethod
    def create_production_config(cls, start_control_server: bool = False) -> LoggingConfig:
        """
        Create a production-oriented logging configuration.
        
        Args:
            start_control_server: Whether to start the logging control server
        
        Returns:
            LoggingConfig: Configuration suitable for production
        """
        config = LoggingConfig.create_production()
        LoggingConfigurer.configure(config)
        
        # Start control server if requested
        if start_control_server:
            cls.start_control_server()
            
        return config
        
    @classmethod
    def start_control_server(cls, port: int = None) -> bool:
        """
        Start the logging control server for runtime configuration.
        
        This server allows changing log levels at runtime via UDP commands.
        It runs in a background thread and is safe to use in production.
        
        Args:
            port: Port to listen on for control commands (default: 50101)
            
        Returns:
            bool: True if server was started, False if already running or failed
        """
        # Import here to avoid circular import 
        from .LoggingControlServer import LoggingControlServer
        
        # Use provided port or class default
        server_port = port if port is not None else cls._control_server_port
        
        # Start the server
        server = LoggingControlServer.start_server(server_port)
        
        # Update class state
        cls._control_server_started = server.running
        if server.running:
            cls._control_server_port = server_port
            
        return server.running
        
    @classmethod
    def stop_control_server(cls) -> bool:
        """
        Stop the logging control server.
        
        Returns:
            bool: True if server was stopped, False if not running
        """
        # Import here to avoid circular import
        from .LoggingControlServer import LoggingControlServer
        
        # Get the server instance
        server = LoggingControlServer.get_instance()
        
        # Stop the server
        result = server.stop()
        
        # Update class state
        cls._control_server_started = server.running
        
        return result
        
    @classmethod
    def is_control_server_running(cls) -> bool:
        """
        Check if the logging control server is running.
        
        Returns:
            bool: True if server is running, False otherwise
        """
        # Import here to avoid circular import
        from .LoggingControlServer import LoggingControlServer
        
        # Get the server instance
        server = LoggingControlServer.get_instance()
        
        # Update class state
        cls._control_server_started = server.running
        
        return server.running