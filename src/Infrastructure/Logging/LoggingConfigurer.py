"""
LoggingConfigurer for the logging configuration system.

This module provides a static class for configuring and managing the logging system.
"""

import os
import logging
import logging.handlers
from typing import Dict, Optional, Union, List

from .LoggingConfig import LoggingConfig
from .Models import LogLevel, LogHandler, LogFormat


class LoggingConfigurer:
    """
    Static class for configuring the logging system.
    
    This class provides methods to configure the Python logging system
    based on the provided LoggingConfig.
    """
    
    # Class variable to track the current configuration
    _current_config: Optional[LoggingConfig] = None
    
    # Class variable to track existing handlers
    _active_handlers: Dict[str, List[logging.Handler]] = {}
    
    @classmethod
    def configure(cls, config: LoggingConfig) -> None:
        """
        Configure logging based on the provided configuration.
        
        This method sets up the root logger and configures handlers
        according to the provided configuration.
        
        Args:
            config: The logging configuration to apply
        """
        # Store the configuration for later reference
        cls._current_config = config
        
        # Get the root logger
        root_logger = logging.getLogger()
        
        # Set the root logger level
        root_logger.setLevel(config.root_level.value)
        
        # Configure console handler if enabled
        if config.console_enabled:
            cls._configure_console_handler(root_logger, config)
        
        # Configure file handler if enabled
        if config.file_enabled:
            # Ensure the log directory exists
            log_dir = os.path.dirname(config.file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            if config.rotation_enabled:
                cls._configure_rotating_file_handler(root_logger, config)
            else:
                cls._configure_file_handler(root_logger, config)
        
        # Configure feature loggers
        for feature_name, level in config.feature_levels.items():
            feature_logger = logging.getLogger(f"{config.root_namespace}.{feature_name}")
            feature_logger.setLevel(level.value)
    
    @classmethod
    def _configure_console_handler(cls, logger: logging.Logger, config: LoggingConfig) -> None:
        """
        Configure a console handler for the logger.
        
        Args:
            logger: Logger to configure handler for
            config: Logging configuration to use
        """
        # Remove any existing console handlers
        cls._remove_handlers(logger, LogHandler.CONSOLE)
        
        # Create a new console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(config.console_level.value)
        
        # Set formatter - ensure we have a string format
        format_string = config.console_format
        if hasattr(format_string, 'value'):  # If it's an enum
            format_string = format_string.value
            
        formatter = logging.Formatter(format_string)
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Track the handler
        cls._track_handler(logger.name, LogHandler.CONSOLE, console_handler)
    
    @classmethod
    def _configure_file_handler(cls, logger: logging.Logger, config: LoggingConfig) -> None:
        """
        Configure a file handler for the logger.
        
        Args:
            logger: Logger to configure handler for
            config: Logging configuration to use
        """
        # Remove any existing file handlers
        cls._remove_handlers(logger, LogHandler.FILE)
        
        # Create a new file handler
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setLevel(config.file_level.value)
        
        # Set formatter - ensure we have a string format
        format_string = config.file_format
        if hasattr(format_string, 'value'):  # If it's an enum
            format_string = format_string.value
            
        formatter = logging.Formatter(format_string)
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Track the handler
        cls._track_handler(logger.name, LogHandler.FILE, file_handler)
    
    @classmethod
    def _configure_rotating_file_handler(cls, logger: logging.Logger, config: LoggingConfig) -> None:
        """
        Configure a rotating file handler for the logger.
        
        Args:
            logger: Logger to configure handler for
            config: Logging configuration to use
        """
        # Remove any existing rotating file handlers
        cls._remove_handlers(logger, LogHandler.ROTATING)
        
        # Create a new rotating file handler
        rotating_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count
        )
        rotating_handler.setLevel(config.file_level.value)
        
        # Set formatter - ensure we have a string format
        format_string = config.file_format
        if hasattr(format_string, 'value'):  # If it's an enum
            format_string = format_string.value
            
        formatter = logging.Formatter(format_string)
        rotating_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(rotating_handler)
        
        # Track the handler
        cls._track_handler(logger.name, LogHandler.ROTATING, rotating_handler)
    
    @classmethod
    def _remove_handlers(cls, logger: logging.Logger, handler_type: LogHandler) -> None:
        """
        Remove handlers of a specific type from a logger.
        
        Args:
            logger: Logger to remove handlers from
            handler_type: Type of handlers to remove
        """
        handlers_to_remove = []
        
        # Get handlers to remove based on type
        if handler_type == LogHandler.CONSOLE:
            handlers_to_remove = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        elif handler_type == LogHandler.FILE:
            handlers_to_remove = [h for h in logger.handlers if isinstance(h, logging.FileHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)]
        elif handler_type == LogHandler.ROTATING:
            handlers_to_remove = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        
        # Remove handlers
        for handler in handlers_to_remove:
            logger.removeHandler(handler)
            
        # Update tracking
        if logger.name in cls._active_handlers:
            cls._active_handlers[logger.name] = [h for h in cls._active_handlers[logger.name] if h not in handlers_to_remove]
    
    @classmethod
    def _track_handler(cls, logger_name: str, handler_type: LogHandler, handler: logging.Handler) -> None:
        """
        Track a handler for later management.
        
        Args:
            logger_name: Name of the logger the handler is attached to
            handler_type: Type of the handler
            handler: The handler instance to track
        """
        if logger_name not in cls._active_handlers:
            cls._active_handlers[logger_name] = []
            
        cls._active_handlers[logger_name].append(handler)
    
    @classmethod
    def configure_from_env(cls) -> LoggingConfig:
        """
        Create and apply configuration from environment variables.
        
        This method reads environment variables to create a logging configuration,
        then applies it using the configure method.
        
        Returns:
            LoggingConfig: The created and applied configuration
        """
        # Start with default configuration
        config = LoggingConfig()
        
        # Get root log level from environment
        level_str = os.environ.get('REALTIMESTT_LOG_LEVEL')
        if level_str:
            try:
                config.root_level = LogLevel.from_string(level_str)
            except ValueError:
                pass  # Ignore invalid level
        
        # Get console settings
        console_enabled = os.environ.get('REALTIMESTT_LOG_CONSOLE')
        if console_enabled:
            config.console_enabled = console_enabled.lower() in ('true', 'yes', '1')
        
        # Get file settings
        file_enabled = os.environ.get('REALTIMESTT_LOG_FILE')
        if file_enabled:
            config.file_enabled = file_enabled.lower() in ('true', 'yes', '1')
        
        # Get file path
        file_path = os.environ.get('REALTIMESTT_LOG_PATH')
        if file_path:
            config.file_path = file_path
        
        # Get rotation settings
        rotation_enabled = os.environ.get('REALTIMESTT_LOG_ROTATION')
        if rotation_enabled:
            config.rotation_enabled = rotation_enabled.lower() in ('true', 'yes', '1')
        
        # Get feature-specific log levels
        for key, value in os.environ.items():
            if key.startswith('REALTIMESTT_LOG_FEATURE_'):
                feature_name = key[len('REALTIMESTT_LOG_FEATURE_'):].lower()
                try:
                    level = LogLevel.from_string(value)
                    config.feature_levels[feature_name] = level
                except ValueError:
                    pass  # Ignore invalid level
        
        # Apply the configuration
        cls.configure(config)
        
        return config
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with a standardized namespace.
        
        This method ensures all loggers follow the standard namespace convention
        defined in the logging configuration.
        
        Args:
            name: The logger name or module name to standardize
            
        Returns:
            logging.Logger: A properly namespaced logger
        """
        # If no configuration exists, use a default one
        if cls._current_config is None:
            cls.configure(LoggingConfig())
            
        # Standardize the namespace
        if name.startswith('src.'):
            # Convert Python module path to logger namespace
            # e.g., 'src.Features.AudioCapture.AudioCaptureModule' -> 'realtimestt.features.audiocapture'
            parts = name.split('.')
            
            if len(parts) >= 3 and parts[0] == 'src':
                if parts[1] == 'Features':
                    # Feature module
                    feature_name = parts[2].lower()
                    namespace = f"{cls._current_config.root_namespace}.features.{feature_name}"
                elif parts[1] == 'Core':
                    # Core module
                    core_module = parts[2].lower()
                    namespace = f"{cls._current_config.root_namespace}.core.{core_module}"
                elif parts[1] == 'Infrastructure':
                    # Infrastructure module
                    infra_module = parts[2].lower()
                    namespace = f"{cls._current_config.root_namespace}.infrastructure.{infra_module}"
                else:
                    # Other module
                    namespace = f"{cls._current_config.root_namespace}.{parts[1].lower()}"
            else:
                # Fallback to root namespace
                namespace = cls._current_config.root_namespace
        elif name.startswith('test'):
            # Test module
            namespace = f"{cls._current_config.root_namespace}.tests"
        elif '.' in name:
            # Already has namespace structure, keep as is
            namespace = name
        else:
            # Single name, add to root namespace
            namespace = f"{cls._current_config.root_namespace}.{name.lower()}"
        
        # Get the logger
        logger = logging.getLogger(namespace)
        
        # Check if feature-specific level is configured
        for feature_key, level in cls._current_config.feature_levels.items():
            if feature_key in namespace:
                logger.setLevel(level.value)
                break
                
        return logger
    
    @classmethod
    def set_feature_level(cls, feature_name: str, level: LogLevel) -> None:
        """
        Set the log level for a specific feature.
        
        Args:
            feature_name: Name of the feature to set level for
            level: Log level to set
        """
        # If no configuration exists, use a default one
        if cls._current_config is None:
            cls.configure(LoggingConfig())
            
        # Update the configuration
        cls._current_config.set_feature_level(feature_name, level)
        
        # Update existing loggers
        feature_namespace = f"{cls._current_config.root_namespace}.features.{feature_name.lower()}"
        feature_logger = logging.getLogger(feature_namespace)
        feature_logger.setLevel(level.value)
    
    @classmethod
    def set_root_level(cls, level: LogLevel) -> None:
        """
        Set the root log level.
        
        Args:
            level: Log level to set for the root logger
        """
        # If no configuration exists, use a default one
        if cls._current_config is None:
            cls.configure(LoggingConfig())
            
        # Update the configuration
        cls._current_config.set_level(level)
        
        # Update root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level.value)
    
    @classmethod
    def get_current_config(cls) -> Optional[LoggingConfig]:
        """
        Get the current logging configuration.
        
        Returns:
            Optional[LoggingConfig]: The current configuration, or None if not configured
        """
        return cls._current_config