# STT Library Integration Specification

## Overview

This specification outlines a plan to transform the current codebase into a modular, importable library with a server-style architecture that allows developers to easily integrate speech-to-text capabilities into their applications. The design supports both direct library integration and a standalone server mode for maximum flexibility.

## Goals

1. **Simplicity**: Make integration easy with minimal code
2. **Flexibility**: Allow choosing specific components to include
3. **Configurability**: Provide sensible defaults but allow customization
4. **Extensibility**: Enable developers to plug in custom handlers for events
5. **Process Isolation**: Provide options to run speech processing in a separate process
6. **Language Agnostic**: Support non-Python clients through a standard API
7. **Resource Management**: Enable easy startup and shutdown of services

## Architecture Overview

The system will support three primary integration methods:

1. **Direct Library Import**: Traditional library usage within Python applications
2. **Server Mode**: Standalone server with REST/WebSocket API
3. **Embedded Server**: In-process server for Python applications that want API-based usage

![Architecture Diagram](https://mermaid.ink/img/pako:eNqNk99O4zAQxl9l5YtS0W1aQgsLiEKraFsJ2JLuJaqqyHE2jVUnNrZTFUV5d8ZJE5qFWy5iz_f9PM545o_b6MQSvPGG-BZqWVgD7fojuOSzjcqCeRSiD5nJRRwXIAVoM9cTVaktnDmrKtWa9p3OokkzKwYfvYW6y_2JZ-VuJf-FWwH7gjrfmdyH-W2M_gJoYM8KHu5r4FveFW6ZmXAYKhtlKpSjLQKbqRtmVm2pUy9lKczpEEY5pJUlNuV5KdAtBw5eZRMuoWp45dBYrwWdkqkhlbO9UhsxgsMBb_PgWq6KZ2ZLOzraqUo1wuwhbzpKVAHBz6NzSPtUHwQrDMjuSddOqP5Wm5ZDhLpD7pzOMVt-Ib1rpawKBpnSpXvFXdgGz9SSvq5O6d7p1C1KvJrIPW0nZwUttqJLkQhbqZ6rmNd6GGKYW3sbVvE8dPz_e55QUuRy2Md1x5-sZ3A5egSMR9uX7d-t4ehq2aX5XRf3h-XwJQ7LQeN2JbUzABDxh-h0m2bNSUhc51DGWHRBxiSwQd4Ru1TT4yomXQ2fYqZXM5JQUOcZSXeOicTrx90mZDaPyZYkNBlHpJuckcfZhIwn5JjQ_aSITwhNz8l0Rnbx4d1BTcKnmX-ufB5OlzOiUzIeMjJm-XjYp3RGZvT6Q0aaT8_I7TmZpGQ5WZG_aLYbVA?type=png)

## Library & Server Structure

```
src/
├── Infrastructure/
│   ├── Server/
│   │   ├── README.md                   # Server documentation
│   │   ├── ServerModule.py             # Main server module
│   │   ├── Controllers/                # API endpoints
│   │   │   ├── AudioController.py      # Audio capture endpoints
│   │   │   ├── SystemController.py     # System-wide operations
│   │   │   ├── TranscriptionController.py # Transcription endpoints
│   │   │   ├── VadController.py        # VAD configuration endpoints
│   │   │   └── WakeWordController.py   # Wake word endpoints
│   │   ├── Services/                   # Service layer
│   │   │   ├── AudioService.py         # Audio capture service
│   │   │   ├── TranscriptionService.py # Transcription service
│   │   │   └── WakeWordService.py      # Wake word service
│   │   ├── WebSocket/                  # WebSocket communication
│   │   │   ├── WebSocketManager.py     # WebSocket connection management
│   │   │   └── WebSocketHandler.py     # WebSocket event handling
│   │   ├── Models/                     # API models
│   │   │   ├── ServerResponse.py       # Standard response format
│   │   │   ├── ConfigurationOptions.py # Configuration options
│   │   │   └── TranscriptionRequest.py # Transcription request models
│   │   └── Configuration/              # Server configuration
│   │       ├── ServerConfig.py         # Server configuration
│   │       └── ProfileManager.py       # Configuration profile management
│   │
│   └── Client/                         # Client library for Python apps
│       ├── SpeechClient.py             # Main client interface
│       ├── Configuration/              # Client configuration
│       │   └── ClientConfig.py         # Client configuration
│       ├── Models/                     # Client models
│       │   └── ClientModels.py         # Request/response models
│       └── Handlers/                   # Event handlers
│           └── EventHandlers.py        # Handler registration
│
├── Application/                        # Library entry points
│   ├── Facade/                         # Simplified library interfaces
│   │   ├── SpeechProcessor.py          # Main facade for library use
│   │   ├── ServerFacade.py             # Facade for server operations
│   │   └── SpeechBuilder.py            # Builder pattern implementation
```

## Server-Client Architecture Details

### Server Components

1. **HTTP REST API**:
   - `GET /status` - Get system status
   - `POST /config` - Update configuration
   - `POST /start` - Start speech processing
   - `POST /stop` - Stop speech processing
   - `GET /devices` - List audio devices
   - `POST /transcribe` - Submit audio for transcription

2. **WebSocket Interface**:
   - Real-time event streaming
   - Client subscribes to events (transcription, wake word, etc.)
   - Server pushes events as they occur
   - Client can send commands through WebSocket

3. **Configuration Profiles**:
   - Pre-defined configurations for common use cases
   - Ability to save custom configurations
   - Default profiles:
     - `continuous-mlx` - Always-on MLX transcription
     - `wake-word-mlx` - Wake word with MLX transcription
     - `wake-word-openai` - Wake word with OpenAI transcription
     - `wake-word-clipboard` - Wake word with clipboard integration

4. **Process Model**:
   - Server runs in its own process
   - Audio capture and processing run in worker threads
   - Optional process isolation for ML components

### Client Library

1. **Python Client**:
   ```python
   from realtime_stt.client import SpeechClient
   
   # Create client
   client = SpeechClient("http://localhost:8080")
   
   # Register handlers
   client.on_transcription(handle_text)
   client.on_wake_word(handle_wake_word)
   
   # Start with a specific configuration profile
   client.start_with_profile("wake-word-mlx")
   
   # Or configure manually
   client.configure(
       wake_word="jarvis", 
       engine="mlx_whisper",
       vad_sensitivity=0.7
   )
   client.start()
   
   # Stop when done
   client.stop()
   ```

2. **REST API Client Example** (Any language):
   ```javascript
   // JavaScript example
   async function startTranscription() {
     // Configure the system
     await fetch('http://localhost:8080/config', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({
         wake_word: "jarvis",
         engine: "mlx_whisper",
         model: "whisper-large-v3-turbo"
       })
     });
     
     // Start the system
     await fetch('http://localhost:8080/start', { method: 'POST' });
     
     // Connect WebSocket for events
     const ws = new WebSocket('ws://localhost:8080/events');
     ws.onmessage = (event) => {
       const data = JSON.parse(event.data);
       if (data.event === 'transcription') {
         handleTranscription(data.text, data.confidence);
       }
     };
   }
   ```

3. **WebSocket Client Example**:
   ```python
   import websocket
   import json
   
   def on_message(ws, message):
       data = json.loads(message)
       if data['event'] == 'transcription':
           print(f"Transcription: {data['text']}")
   
   def start_client():
       ws = websocket.WebSocketApp("ws://localhost:8080/events",
                                   on_message=on_message)
       ws.run_forever()
   ```

## Server Implementation Details

### 1. Server Module Integration

The server will integrate with the existing system through the command/event architecture:

```python
class ServerModule:
    @staticmethod
    def register(command_dispatcher, event_bus, host="127.0.0.1", port=8080):
        """Register the server module with the system."""
        # Create server instance
        server = Server(command_dispatcher, event_bus, host, port)
        
        # Register event handlers
        event_bus.subscribe(TranscriptionUpdatedEvent, server.handle_transcription_update)
        event_bus.subscribe(WakeWordDetectedEvent, server.handle_wake_word_detected)
        # ...other event subscriptions
        
        # Start the server
        server.start()
        
        return server
```

### 2. WebSocket Event Broadcasting

```python
class WebSocketManager:
    def __init__(self):
        self.clients = set()
    
    def register(self, websocket):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
    
    def unregister(self, websocket):
        """Unregister a WebSocket client."""
        self.clients.remove(websocket)
    
    def broadcast_event(self, event_type, data):
        """Broadcast an event to all connected clients."""
        message = {
            "event": event_type,
            **data
        }
        
        for client in self.clients:
            client.send_json(message)
```

### 3. Controller Example

```python
class TranscriptionController:
    def __init__(self, command_dispatcher):
        self.command_dispatcher = command_dispatcher
    
    async def configure(self, request):
        """Configure transcription engine."""
        command = ConfigureTranscriptionCommand(
            engine_type=request.engine,
            model_name=request.model,
            language=request.language,
            beam_size=request.beam_size,
            options=request.options
        )
        
        result = self.command_dispatcher.dispatch(command)
        return {"status": "success", "result": result}
    
    async def start_session(self, request):
        """Start a transcription session."""
        command = StartTranscriptionSessionCommand(
            session_id=request.session_id
        )
        
        result = self.command_dispatcher.dispatch(command)
        return {"status": "success", "session_id": request.session_id}
```

### 4. Server Configuration

```python
class ServerConfig:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 8080
        self.debug = False
        self.cors_origins = ["*"]
        self.profiles_directory = "profiles/"
        self.auth_enabled = False
        self.auth_token = None
    
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables."""
        config = cls()
        config.host = os.environ.get("STT_SERVER_HOST", config.host)
        config.port = int(os.environ.get("STT_SERVER_PORT", config.port))
        config.debug = os.environ.get("STT_SERVER_DEBUG", "").lower() == "true"
        # ... load other config from env
        return config
    
    @classmethod
    def from_file(cls, path):
        """Load configuration from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
```

## Direct Library Usage

For users who don't want server functionality, the library will still be usable directly:

```python
from realtime_stt import SpeechProcessor

# Use the fluent builder API
processor = (SpeechProcessor.builder()
    .with_wake_word(words=["jarvis"])
    .with_transcription(engine="mlx_whisper")
    .build())

# Register handlers
processor.on_transcription_complete(handle_text)

# Start/stop
processor.start()
# ...
processor.stop()
```

## Server Launch Options

The server can be launched in several ways:

1. **Command Line**:
   ```bash
   # Start with default settings
   python -m realtime_stt.server
   
   # Configure with command line arguments
   python -m realtime_stt.server --host 0.0.0.0 --port 8080 --profile wake-word-mlx
   ```

2. **Programmatic Launch**:
   ```python
   from realtime_stt.server import launch_server
   
   # Start with default settings
   server = launch_server()
   
   # Configure programmatically
   server = launch_server(
       host="0.0.0.0",
       port=8080,
       profile="wake-word-mlx"
   )
   
   # Stop when done
   server.stop()
   ```

3. **Configuration File**:
   ```bash
   # Start with configuration file
   python -m realtime_stt.server --config server_config.json
   ```

## Configuration File Format

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "debug": false,
    "cors_origins": ["*"]
  },
  "speech": {
    "wake_word": {
      "enabled": true,
      "words": ["jarvis"],
      "sensitivity": 0.7
    },
    "vad": {
      "detector_type": "combined",
      "sensitivity": 0.6
    },
    "transcription": {
      "engine": "mlx_whisper",
      "model": "whisper-large-v3-turbo",
      "language": null
    }
  }
}
```

## Detailed Component Architecture

### 1. Server Core

The server core will use FastAPI for the HTTP API and WebSockets:

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

class Server:
    def __init__(self, command_dispatcher, event_bus, host, port):
        self.app = FastAPI(title="Speech-to-Text API")
        self.command_dispatcher = command_dispatcher
        self.event_bus = event_bus
        self.host = host
        self.port = port
        self.websocket_manager = WebSocketManager()
        
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
        self.app.get("/status", self.get_status)
        self.app.post("/config", self.update_config)
        self.app.post("/start", self.start_processing)
        self.app.post("/stop", self.stop_processing)
        self.app.websocket("/events", self.websocket_endpoint)
        # ...more routes
    
    async def websocket_endpoint(self, websocket: WebSocket):
        """Handle WebSocket connections."""
        await websocket.accept()
        self.websocket_manager.register(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                # Handle incoming WebSocket commands
                # ...
        except WebSocketDisconnect:
            self.websocket_manager.unregister(websocket)
    
    def _register_event_handlers(self):
        """Register handlers for system events."""
        # These handlers will broadcast events to WebSocket clients
        self.event_bus.subscribe(TranscriptionUpdatedEvent, self.handle_transcription_update)
        # ...more event handlers
    
    def start(self):
        """Start the server."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
```

### 2. Client Library

The client library will abstract away the HTTP and WebSocket details:

```python
import requests
import websocket
import json
import threading

class SpeechClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.websocket_url = f"ws://{server_url.split('//')[1]}/events"
        self.callbacks = {}
        self.ws = None
        self.ws_thread = None
        self.running = False
    
    def on_transcription(self, callback):
        """Register a callback for transcription events."""
        self.callbacks["transcription"] = callback
        return self
    
    def on_wake_word(self, callback):
        """Register a callback for wake word events."""
        self.callbacks["wake_word"] = callback
        return self
    
    def _handle_ws_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        data = json.loads(message)
        event_type = data.get("event")
        
        if event_type in self.callbacks:
            # Call the appropriate callback
            del data["event"]  # Remove event field
            self.callbacks[event_type](**data)
    
    def start_with_profile(self, profile_name):
        """Start the system using a configuration profile."""
        response = requests.post(
            f"{self.server_url}/start",
            json={"profile": profile_name}
        )
        response.raise_for_status()
        
        # Start WebSocket connection
        self._start_websocket()
        
        return response.json()
    
    def configure(self, **kwargs):
        """Configure the system."""
        response = requests.post(
            f"{self.server_url}/config",
            json=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def start(self):
        """Start the system with current configuration."""
        response = requests.post(f"{self.server_url}/start")
        response.raise_for_status()
        
        # Start WebSocket connection
        self._start_websocket()
        
        return response.json()
    
    def stop(self):
        """Stop the system."""
        # Close WebSocket
        if self.ws:
            self.running = False
            self.ws.close()
        
        # Stop server processing
        response = requests.post(f"{self.server_url}/stop")
        response.raise_for_status()
        return response.json()
    
    def _start_websocket(self):
        """Start WebSocket connection in a separate thread."""
        if self.ws_thread and self.ws_thread.is_alive():
            return  # Already running
        
        self.running = True
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.websocket_url,
            on_message=self._handle_ws_message
        )
        
        # Start WebSocket in a separate thread
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            daemon=True
        )
        self.ws_thread.start()
```

## Implementation Plan

### Phase 1: Server Framework (2 weeks)
1. Set up basic server structure with FastAPI
2. Implement HTTP API endpoints
3. Implement WebSocket event broadcasting
4. Create server configuration system
5. Integrate with existing command/event system

### Phase 2: Client Library (1 week)
1. Implement Python client library
2. Create client configuration system
3. Add event callback mechanisms
4. Build WebSocket client functionality

### Phase 3: Feature Integration (2 weeks)
1. Integrate audio capture controls
2. Integrate VAD configuration
3. Integrate wake word detection
4. Integrate transcription engines
5. Add clipboard functionality

### Phase 4: Testing & Documentation (1 week)
1. Write comprehensive tests
2. Create example client applications
3. Document API endpoints
4. Create user guides

## Conclusion

This server-based architecture provides maximum flexibility for integrating speech recognition into applications. By supporting both direct library usage and client-server interactions, developers can choose the most appropriate integration method for their use case. The system remains true to the vertical slice architecture while adding a powerful new way to access its functionality.

The server approach offers several key advantages:
- Process isolation for stability
- Language-agnostic client support
- Distributed application architecture
- Resource management with clear start/stop controls
- Simplified configuration through profiles

This design balances simplicity with power, allowing both straightforward use cases and complex integrations.