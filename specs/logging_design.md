# Logging Configuration System Design

## Overview

This document outlines the design for a centralized logging configuration system for the Realtime_mlx_STT project. The goal is to create a consistent, configurable, and maintainable logging infrastructure that follows the project's vertical slice architecture.

## Current State

The codebase currently uses Python's built-in logging module with inconsistent configuration:

1. **Core modules** use explicit namespaces like "realtimestt.core.commands"
2. **Feature modules** use `__name__` to create namespace hierarchies
3. **Example scripts** configure logging independently
4. **File logging** is only used in wake_word_detection.py
5. No **log rotation** or centralized configuration exists

## Design Goals

1. Create a centralized, configurable logging system
2. Standardize logger namespaces across the codebase
3. Support file and console logging with rotation
4. Allow environment-based and runtime configuration
5. Ensure backward compatibility with existing logging code
6. Follow the project's vertical slice architecture

## Architecture

### Component Structure

The logging system will be implemented as a new module in the Infrastructure layer:

```
src/
  Infrastructure/
    Logging/
      __init__.py
      LoggingConfig.py       # Configuration data model
      LoggingConfigurer.py   # Logging setup and configuration 
      LoggingModule.py       # Public facade/interface
      Models/
        LogLevel.py          # Enum for log levels
        LogHandler.py        # Enum for handler types
        LogFormat.py         # Enum for predefined formats
```

### Key Components

#### LoggingConfig

A data class to store logging configuration settings:

```python
@dataclass
class LoggingConfig:
    """Configuration settings for the logging system."""
    root_level: LogLevel = LogLevel.INFO
    feature_levels: Dict[str, LogLevel] = field(default_factory=dict)
    console_enabled: bool = True
    console_level: LogLevel = LogLevel.INFO
    file_enabled: bool = False
    file_level: LogLevel = LogLevel.DEBUG
    file_path: str = "logs/realtimestt.log"
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    rotation_enabled: bool = False
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
```

#### LoggingConfigurer

A static class with methods to configure the logging system:

```python
class LoggingConfigurer:
    """Static class for configuring the logging system."""
    
    @staticmethod
    def configure(config: LoggingConfig) -> None:
        """Configure logging based on the provided configuration."""
        # Implementation to set up handlers and configure loggers
        
    @staticmethod
    def configure_from_env() -> LoggingConfig:
        """Create and apply configuration from environment variables."""
        # Implementation to read environment variables
        
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger with the standardized namespace."""
        # Implementation to get loggers with consistent naming
        
    @staticmethod
    def set_feature_level(feature_name: str, level: LogLevel) -> None:
        """Set the log level for a specific feature."""
        # Implementation to update feature-specific log levels
```

#### LoggingModule

A facade for the logging system following the project's module pattern:

```python
class LoggingModule:
    """Module for logging configuration functionality."""
    
    @staticmethod
    def initialize(
        console_level: str = "INFO",
        file_level: str = "DEBUG", 
        file_enabled: bool = True,
        file_path: str = None,
        rotation_enabled: bool = True
    ) -> LoggingConfig:
        """Initialize the logging system with the specified configuration."""
        # Implementation
        
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger with the standardized namespace."""
        return LoggingConfigurer.get_logger(name)
        
    @staticmethod
    def set_level(feature_or_logger: str, level: str) -> None:
        """Set the log level for a specific feature or logger."""
        # Implementation
```

### Namespace Standardization

To ensure consistent logger naming, we'll standardize on a clear namespace hierarchy:

1. **Root namespace**: "realtimestt"
2. **Core components**: "realtimestt.core.{component}"
3. **Features**: "realtimestt.features.{feature_name}.{submodule}"
4. **Testing**: "realtimestt.tests.{feature_name}.{test_name}"

The `LoggingConfigurer.get_logger()` method will handle mapping Python module paths to standardized namespaces.

### Environment Variable Configuration

The system will support configuration via environment variables:

- `REALTIMESTT_LOG_LEVEL`: Root log level (INFO, DEBUG, etc.)
- `REALTIMESTT_LOG_CONSOLE`: Enable console logging (true/false)
- `REALTIMESTT_LOG_FILE`: Enable file logging (true/false)
- `REALTIMESTT_LOG_PATH`: Path to log file
- `REALTIMESTT_LOG_ROTATION`: Enable log rotation (true/false)
- `REALTIMESTT_LOG_FEATURE_{FEATURE}`: Set level for specific feature

### Integration with Example Scripts

Example scripts will be updated to use the new logging module:

```python
from src.Infrastructure.Logging.LoggingModule import LoggingModule

# Initialize logging
LoggingModule.initialize(
    console_level="INFO",
    file_enabled=True,
    file_path="logs/example.log"
)

# Get a logger
logger = LoggingModule.get_logger(__name__)
```

### Backward Compatibility

To ensure backward compatibility:

1. The `LoggingConfigurer.get_logger()` method will handle both `__name__` and explicit namespaces
2. Core loggers will retain their current names but be configured through the central system
3. Root logger configuration will not break existing loggers

## Implementation Plan

1. Create the Infrastructure/Logging directory and files
2. Implement the LoggingConfig and model classes
3. Implement the LoggingConfigurer class
4. Implement the LoggingModule facade
5. Update core components to use the new logging system
6. Update example scripts to use LoggingModule
7. Add documentation and tests

## Benefits

1. **Consistency**: Standardized logging across all components
2. **Configurability**: Central configuration with environment variable support
3. **Maintainability**: Clear logging architecture that follows project structure
4. **Performance**: Proper log rotation and level management
5. **Debugging**: Enhanced logging capabilities for easier troubleshooting