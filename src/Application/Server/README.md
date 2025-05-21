# Server Module for Realtime_mlx_STT

This module provides a server-based architecture for the Realtime_mlx_STT system, allowing speech-to-text functionality to be accessed through HTTP and WebSocket APIs. The server integrates with the existing command/event architecture without modifying the core functionality.

## Architecture

The server uses FastAPI to provide:
- RESTful API endpoints for configuration and control
- WebSocket connections for real-time events
- Integration with the existing command/event system

### Key Components

- **ServerModule**: Entry point that follows the same pattern as other features in the system
- **Server**: Main server implementation that integrates with FastAPI
- **Controllers**: Handle specific API endpoints for different features
- **WebSocketManager**: Manages WebSocket connections and event broadcasting
- **Configuration**: Handles server configuration and profile management

## API Endpoints

### System Endpoints

- `GET /system/status` - Get system status
- `GET /system/info` - Get system information
- `GET /system/profiles` - List available configuration profiles
- `GET /system/profiles/{name}` - Get a specific configuration profile
- `POST /system/profiles` - Save a configuration profile
- `DELETE /system/profiles/{name}` - Delete a configuration profile
- `POST /system/start` - Start the system with a profile
- `POST /system/stop` - Stop the system
- `POST /system/config` - Update system configuration

### Transcription Endpoints

- `POST /transcription/configure` - Configure the transcription engine
- `POST /transcription/session/start` - Start a transcription session
- `POST /transcription/session/stop` - Stop a transcription session
- `POST /transcription/audio` - Submit audio for transcription
- `GET /transcription/status` - Get transcription status

### WebSocket Events

Connect to `/events` to receive real-time events:

- `transcription` - Transcription updates
- `wake_word` - Wake word detections
- `speech` - Speech detection events
- `silence` - Silence detection events

## Usage

To integrate the server with your application:

```python
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Application.Server import ServerModule

# Create command dispatcher and event bus
command_dispatcher = CommandDispatcher()
event_bus = EventBus()

# Register other modules...

# Register server module
server = ServerModule.register(
    command_dispatcher=command_dispatcher,
    event_bus=event_bus
)

# Server is automatically started if auto_start is enabled (default)
# Otherwise, start it manually:
# server.start()
```

## Configuration

The server can be configured through:

1. Environment variables:
   - `STT_SERVER_HOST` - Host to bind to (default: "127.0.0.1")
   - `STT_SERVER_PORT` - Port to bind to (default: 8080)
   - `STT_SERVER_DEBUG` - Enable debug mode (default: false)
   - `STT_SERVER_AUTO_START` - Auto-start the server (default: true)
   - `STT_SERVER_CORS_ORIGINS` - Comma-separated list of allowed CORS origins

2. Configuration file:
   ```json
   {
     "server": {
       "host": "127.0.0.1",
       "port": 8080,
       "debug": false,
       "auto_start": true,
       "cors_origins": ["*"]
     }
   }
   ```

3. Directly through the `ServerConfig` class:
   ```python
   from src.Application.Server.Configuration.ServerConfig import ServerConfig
   
   config = ServerConfig()
   config.host = "0.0.0.0"  # Allow external connections
   config.port = 8000
   
   server = ServerModule.register(
       command_dispatcher=command_dispatcher,
       event_bus=event_bus,
       config=config
   )
   ```

## Profiles

The server supports configuration profiles for easy setup:

- `continuous-mlx` - Always-on MLX transcription
- `wake-word-mlx` - Wake word with MLX transcription
- `wake-word-openai` - Wake word with OpenAI transcription
- `wake-word-clipboard` - Wake word with clipboard integration

Profiles can be loaded, saved, and managed through the API.

## Integration with Existing Code

The server integrates with the existing codebase through:

1. Using the command/event system to communicate with other modules
2. Subscribing to events from core features
3. Dispatching commands to trigger actions in other modules

All communication happens through the existing architecture, ensuring that the server doesn't modify any core functionality.