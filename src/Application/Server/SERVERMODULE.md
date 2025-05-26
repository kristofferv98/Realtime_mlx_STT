# ServerModule Documentation

## Overview

The ServerModule provides a FastAPI-based REST/WebSocket server that exposes the speech-to-text functionality via HTTP and WebSocket APIs. It follows the same vertical slice architecture pattern as other features in the system, integrating seamlessly with the existing command/event infrastructure without modifying core functionality.

## Architecture

### Module Registration Pattern

The ServerModule follows the standard feature registration pattern:

```python
server = ServerModule.register(
    command_dispatcher=command_dispatcher,
    event_bus=event_bus,
    config=server_config
)
```

This pattern ensures:
- Consistent integration with the command/event system
- No direct dependencies between modules
- Clean separation of concerns
- Easy testing and mocking

### Component Structure

```
src/Application/Server/
├── ServerModule.py          # Main module registration and server lifecycle
├── Configuration/
│   ├── ServerConfig.py      # Server configuration management
│   └── ProfileManager.py    # Predefined and custom profile management
├── Controllers/
│   ├── BaseController.py    # Base controller with common functionality
│   ├── TranscriptionController.py  # Transcription API endpoints
│   └── SystemController.py  # System management endpoints
├── WebSocket/
│   └── WebSocketManager.py  # WebSocket connection and event broadcasting
└── Models/
    ├── TranscriptionModels.py  # Request/response models for transcription
    └── SystemModels.py      # System-related data models
```

## Server Initialization Process

### 1. Configuration Loading

The server supports multiple configuration sources with this precedence:

1. **Explicit configuration object** - Passed directly to `register()`
2. **Environment variables** - Using `ServerConfig.from_env()`
3. **Configuration files** - Using `ServerConfig.from_file(path)`
4. **Default values** - Built-in defaults in `ServerConfig` class

```python
# Configuration precedence example
config = ServerConfig(
    host="127.0.0.1",        # Default
    port=8080,               # Can be overridden by STT_SERVER_PORT env var
    debug=False,             # Can be overridden by STT_SERVER_DEBUG env var
    auto_start=True,         # Server starts automatically after registration
    cors_origins=["*"]       # Can be overridden by STT_SERVER_CORS_ORIGINS
)
```

### 2. FastAPI Application Setup

The Server class initializes the FastAPI application with:

- **CORS middleware** - Configurable origins for cross-origin requests
- **Standard routers** - For organizing endpoints by feature
- **WebSocket endpoint** - For real-time event streaming
- **Event handlers** - Bridge between internal events and WebSocket broadcasts

```python
# FastAPI app initialization
self.app = FastAPI(title="Speech-to-Text API")

# CORS configuration
self.app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. Controller Registration

Controllers are registered using FastAPI's router pattern:

```python
def _register_controllers(self):
    # Create controller instances
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
```

### 4. WebSocket Integration

The WebSocket endpoint at `/events` provides real-time event streaming:

```python
@self.app.websocket("/events")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    self.websocket_manager.register(websocket)
    
    # Set event loop for thread-safe broadcasting
    loop = asyncio.get_running_loop()
    self.websocket_manager.set_event_loop(loop)
    
    # Handle incoming messages and disconnections
```

### 5. Event Handler Registration

The server subscribes to system events and broadcasts them to WebSocket clients:

```python
def _register_event_handlers(self):
    # Subscribe to transcription events
    self.event_bus.subscribe(TranscriptionUpdatedEvent, self.handle_transcription_update)
    
    # Subscribe to wake word events
    self.event_bus.subscribe(WakeWordDetectedEvent, self.handle_wake_word_detected)
```

### 6. Server Startup

The server runs in a separate thread to avoid blocking the main application:

```python
def start(self):
    self.running = True
    self.server_thread = threading.Thread(
        target=self._run_server,
        daemon=False  # Non-daemon for proper cleanup
    )
    self.server_thread.start()
```

## Module Integration Patterns

### Command Dispatcher Integration

Controllers use the command dispatcher to execute system commands:

```python
def send_command(self, command) -> Any:
    try:
        result = self.command_dispatcher.dispatch(command)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing command: {str(e)}"
        )
```

### Event Bus Integration

The server acts as a bridge between internal events and external clients:

```python
def handle_transcription_update(self, event: TranscriptionUpdatedEvent):
    # Convert internal event to WebSocket message
    self.websocket_manager.broadcast_event("transcription", {
        "text": event.text,
        "is_final": event.is_final,
        "session_id": event.session_id
    })
```

### Thread-Safe Broadcasting

The WebSocketManager handles broadcasting from different threads:

```python
def broadcast_event(self, event_type: str, data: Dict[str, Any]):
    try:
        current_loop = asyncio.get_running_loop()
        # In async context - create task directly
        current_loop.create_task(self._broadcast(message))
    except RuntimeError:
        # Different thread - use thread-safe method
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._broadcast(message),
                self._loop
            )
```

## Configuration Management

### Environment Variables

The server supports these environment variables:

- `STT_SERVER_HOST` - Server host (default: "127.0.0.1")
- `STT_SERVER_PORT` - Server port (default: 8080)
- `STT_SERVER_DEBUG` - Enable debug mode (default: false)
- `STT_SERVER_AUTO_START` - Auto-start server on registration (default: true)
- `STT_SERVER_CORS_ORIGINS` - Comma-separated CORS origins (default: "*")
- `STT_SERVER_AUTH_ENABLED` - Enable authentication (default: false)
- `STT_SERVER_AUTH_TOKEN` - Authentication token (optional)
- `STT_SERVER_PROFILES_DIR` - Profile storage directory (default: "profiles/")
- `STT_SERVER_DEFAULT_PROFILE` - Default profile name (default: "default")

### Profile Management

The ProfileManager provides predefined and custom profiles:

```python
# Predefined profiles
PREDEFINED_PROFILES = {
    "vad-triggered": {
        "description": "VAD-triggered transcription",
        "transcription": {"auto_start": True},
        "vad": {"enabled": True, "detector_type": "combined"},
        "wake_word": {"enabled": False}
    },
    "wake-word": {
        "description": "Wake word activated",
        "transcription": {"auto_start": False},
        "vad": {"enabled": True, "detector_type": "combined"},
        "wake_word": {"enabled": True, "words": ["jarvis"]}
    }
}
```

## Extension Points

### Adding New Controllers

1. Create a new controller extending `BaseController`:

```python
class MyController(BaseController):
    def __init__(self, command_dispatcher, event_bus):
        super().__init__(command_dispatcher, event_bus, prefix="/my-feature")
        
    def _register_routes(self):
        @self.router.get("/status")
        async def get_status():
            return self.create_standard_response(data={"status": "ok"})
```

2. Register in `ServerModule._register_controllers()`:

```python
my_controller = MyController(
    command_dispatcher=self.command_dispatcher,
    event_bus=self.event_bus
)
self.app.include_router(my_controller.router)
```

### Adding New Event Handlers

1. Subscribe to new events in `_register_event_handlers()`:

```python
self.event_bus.subscribe(MyCustomEvent, self.handle_my_custom_event)
```

2. Create handler method:

```python
def handle_my_custom_event(self, event: MyCustomEvent):
    self.websocket_manager.broadcast_event("my_event", {
        "data": event.data
    })
```

### Adding Middleware

Add middleware in the Server `__init__` method:

```python
# Example: Add request logging middleware
@self.app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    self.logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")
    return response
```

### Adding Startup/Shutdown Hooks

Use FastAPI lifecycle events:

```python
@self.app.on_event("startup")
async def startup_event():
    self.logger.info("Server startup tasks...")
    # Initialize resources, connections, etc.

@self.app.on_event("shutdown")
async def shutdown_event():
    self.logger.info("Server shutdown tasks...")
    # Cleanup resources, close connections, etc.
```

## Production vs Development Configuration

### Development Mode

```python
server_config = ServerConfig(
    host="127.0.0.1",  # Localhost only
    port=8080,         # Development port
    debug=True,        # Enable debug logging
    cors_origins=["*"] # Allow all origins
)
```

### Production Mode

```python
server_config = ServerConfig(
    host="0.0.0.0",           # Listen on all interfaces
    port=80,                  # Standard HTTP port
    debug=False,              # Disable debug
    cors_origins=[            # Specific allowed origins
        "https://app.example.com",
        "https://www.example.com"
    ],
    auth_enabled=True,        # Enable authentication
    auth_token=os.environ.get("API_TOKEN")  # From environment
)
```

## Usage Example

```python
# Complete server setup with all modules
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Application.Server.ServerModule import ServerModule
from src.Application.Server.Configuration.ServerConfig import ServerConfig

# Initialize core components
command_dispatcher = CommandDispatcher()
event_bus = EventBus()

# Register feature modules (AudioCapture, VAD, Transcription, etc.)
# ... module registration code ...

# Configure and start server
server_config = ServerConfig.from_env()  # Or from_file("config.json")
server = ServerModule.register(
    command_dispatcher=command_dispatcher,
    event_bus=event_bus,
    config=server_config
)

# Server is now running at http://host:port
# Access API docs at http://host:port/docs
```

## Key Design Decisions

1. **Module Pattern Consistency**: Uses the same `register()` pattern as other features
2. **Thread-Safe Event Broadcasting**: Handles events from different threads safely
3. **Configuration Flexibility**: Multiple configuration sources with clear precedence
4. **Controller Abstraction**: BaseController provides common functionality
5. **WebSocket Integration**: Real-time events alongside REST API
6. **Profile Management**: Predefined profiles for common use cases
7. **Non-Blocking Server**: Runs in separate thread to not block main application
8. **CORS Support**: Built-in for web client integration
9. **Extensible Architecture**: Easy to add new endpoints, events, and middleware
10. **Environment-First Configuration**: Production-ready with environment variables

## Security Considerations

- **Authentication**: Optional token-based authentication (expandable)
- **CORS**: Configurable allowed origins
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Standardized error responses
- **Logging**: Comprehensive logging for debugging and auditing

## Performance Considerations

- **Async/Await**: FastAPI's async support for high concurrency
- **WebSocket Efficiency**: Single persistent connection for events
- **Thread Safety**: Proper handling of cross-thread communication
- **Resource Management**: Proper cleanup on shutdown
- **Connection Pooling**: WebSocket connection management

This architecture ensures the server module integrates cleanly with the existing system while providing a robust, scalable API layer for external clients.