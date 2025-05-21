# Logging Configuration System

This module provides a centralized logging configuration system for the Realtime_mlx_STT project, allowing for consistent logging across the codebase with standardized namespaces, configurable outputs, and log rotation.

## Basic Usage

### Simple Initialization

```python
from src.Infrastructure.Logging import LoggingModule

# Initialize with default settings
LoggingModule.initialize()

# Get a logger
logger = LoggingModule.get_logger(__name__)
logger.info("This is an informational message")
logger.debug("This is a debug message")
```

### Custom Configuration

```python
from src.Infrastructure.Logging import LoggingModule, LogLevel

# Initialize with custom settings
LoggingModule.initialize(
    console_level="DEBUG",
    file_enabled=True,
    file_path="logs/my_application.log",
    rotation_enabled=True,
    backup_count=3,
    feature_levels={
        "AudioCapture": LogLevel.DEBUG,
        "Transcription": LogLevel.INFO
    }
)

# Get a logger
logger = LoggingModule.get_logger(__name__)
```

### Predefined Configurations

```python
from src.Infrastructure.Logging import LoggingModule

# Use development configuration (more verbose)
LoggingModule.create_development_config()

# Or use production configuration
LoggingModule.create_production_config()
```

## Environment Variable Configuration

The logging system supports configuration through environment variables, making it easy to change logging behavior without code changes.

### Available Environment Variables

| Environment Variable | Description | Example |
|----------------------|-------------|---------|
| REALTIMESTT_LOG_LEVEL | Root log level | DEBUG, INFO, WARNING |
| REALTIMESTT_LOG_CONSOLE | Enable console logging | true, false |
| REALTIMESTT_LOG_FILE | Enable file logging | true, false |
| REALTIMESTT_LOG_PATH | Path to log file | logs/app.log |
| REALTIMESTT_LOG_ROTATION | Enable log rotation | true, false |
| REALTIMESTT_LOG_FEATURE_{NAME} | Set level for specific feature | DEBUG, INFO |

### Example: Setting Environment Variables

```bash
# Set global log level to DEBUG
export REALTIMESTT_LOG_LEVEL=DEBUG

# Enable file logging
export REALTIMESTT_LOG_FILE=true

# Set custom log path
export REALTIMESTT_LOG_PATH=logs/custom_log.log

# Set feature-specific log levels
export REALTIMESTT_LOG_FEATURE_TRANSCRIPTION=DEBUG
export REALTIMESTT_LOG_FEATURE_AUDIOCAPTURE=INFO
```

### Applying Environment Configuration

```python
from src.Infrastructure.Logging import LoggingModule

# Load configuration from environment variables
LoggingModule.initialize_from_env()

# Get a logger
logger = LoggingModule.get_logger(__name__)
```

## Runtime Configuration

You can change log levels at runtime without restarting the application:

```python
from src.Infrastructure.Logging import LoggingModule, LogLevel

# Set log level for a specific feature
LoggingModule.set_level("AudioCapture", LogLevel.DEBUG)

# Set root log level
LoggingModule.set_level("root", LogLevel.INFO)

# Set log level for a specific logger
LoggingModule.set_level("realtimestt.features.transcription.engines", "DEBUG")
```

## Log Format Options

The logging system provides several predefined formats:

- `BASIC`: Simple format with timestamp, level, and message
- `STANDARD`: Standard format with timestamp, logger name, level, and message
- `DETAILED`: Detailed format with timestamp, logger name, level, file location, and message
- `DEVELOPMENT`: Development format with thread info and details for debugging
- `MINIMAL`: Minimal format with just level and message

```python
from src.Infrastructure.Logging import LoggingModule, LogFormat

# Use a predefined format
LoggingModule.initialize(
    console_format=LogFormat.DETAILED,
    file_format=LogFormat.STANDARD
)

# Or specify a custom format
LoggingModule.initialize(
    console_format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
```

## Logger Naming Convention

The logging system standardizes logger names to create a consistent hierarchy:

- Core components: `realtimestt.core.{component}`
- Features: `realtimestt.features.{feature_name}.{submodule}`
- Testing: `realtimestt.tests.{feature_name}.{test_name}`

When you use `LoggingModule.get_logger(__name__)`, the module path is automatically converted to a standardized namespace.

## Advanced Usage

### Custom LoggingConfig

For more advanced configurations, you can create a LoggingConfig object directly:

```python
from src.Infrastructure.Logging import LoggingConfig, LoggingConfigurer, LogLevel, LogFormat

# Create a custom configuration
config = LoggingConfig(
    root_level=LogLevel.INFO,
    console_enabled=True,
    console_level=LogLevel.INFO,
    file_enabled=True,
    file_level=LogLevel.DEBUG,
    file_path="logs/custom.log",
    rotation_enabled=True,
    max_bytes=5242880,  # 5MB
    backup_count=3,
    feature_levels={
        "audiocapture": LogLevel.DEBUG,
        "transcription": LogLevel.WARNING
    }
)

# Apply the configuration
LoggingConfigurer.configure(config)
```