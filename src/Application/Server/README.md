# Realtime MLX STT Server

FastAPI-based REST API and WebSocket server for real-time speech-to-text transcription. Provides network access to the complete feature set via HTTP endpoints and WebSocket events.

## Architecture Overview

The server exposes the underlying Features layer through a clean REST API without modifying core functionality. It follows the established vertical slice architecture principles:

```
Server Layer (REST/WebSocket)
    ↓ Commands
Features Layer (Business Logic)
    ↓ Events
Server Layer (WebSocket Broadcasting)
```

## Quick Start

### Basic Server Setup

```bash
# Start with default configuration
cd example_server
python server_example.py

# Server starts at http://localhost:8000
# Web UI available at http://localhost:8000/
# API docs at http://localhost:8000/docs
```

### Environment Configuration

```bash
# Server settings
export STT_SERVER_HOST="0.0.0.0"
export STT_SERVER_PORT="8080"
export STT_SERVER_DEBUG="true"

# Authentication (if enabled)
export STT_SERVER_AUTH_ENABLED="true"
export STT_SERVER_AUTH_TOKEN="your-secret-token"

# API Keys for features
export OPENAI_API_KEY="sk-..."
export PORCUPINE_ACCESS_KEY="..."

python server_example.py
```

## Module Structure

```
src/Application/Server/
├── README.md              # This documentation
├── ServerModule.py         # Main server module and registration
├── Configuration/          # Profile and server configuration
│   ├── ProfileManager.py   # Predefined and custom profiles
│   └── ServerConfig.py     # Server settings with env var support
├── Controllers/            # REST API endpoints
│   ├── BaseController.py   # Common controller functionality
│   ├── SystemController.py # System management endpoints
│   └── TranscriptionController.py # Transcription endpoints
├── Models/                 # Pydantic request/response models
│   ├── SystemModels.py     # System API models
│   └── TranscriptionModels.py # Transcription API models
└── WebSocket/              # Real-time communication
    └── WebSocketManager.py # WebSocket connection management
```

## API Endpoints

### System Management (`/api/v1/system/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Server status and configuration |
| `/start` | POST | Start transcription with profile |
| `/stop` | POST | Stop all transcription |
| `/profiles` | GET | List available profiles |
| `/profiles/{name}` | GET | Get specific profile |
| `/profiles/{name}` | POST | Save custom profile |
| `/profiles/{name}` | DELETE | Delete custom profile |

### Transcription (`/api/v1/transcription/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/configure` | POST | Configure transcription engine |
| `/sessions` | POST | Start new transcription session |
| `/sessions/{id}` | DELETE | Stop transcription session |
| `/transcribe` | POST | Transcribe audio data |
| `/status` | GET | Transcription system status |

### WebSocket Events (`/api/v1/ws`)

| Event Type | Description | Data |
|------------|-------------|------|
| `transcription` | Real-time transcription updates | `{text, is_final, confidence, timestamp}` |
| `wake_word` | Wake word detection | `{wake_word, confidence, timestamp}` |
| `speech_detected` | Voice activity detection | `{confidence, timestamp}` |
| `system_status` | System state changes | `{running, active_features}` |

## Configuration Profiles

### Predefined Profiles

#### `vad-triggered`
- **Use Case**: Standard voice-activity triggered transcription
- **Auto-start**: Enabled
- **VAD**: Combined Silero + WebRTC (sensitivity: 0.6)
- **Wake Word**: Disabled

```json
{
  "description": "VAD-triggered transcription - only transcribe when speech is detected",
  "transcription": {"auto_start": true},
  "vad": {
    "detector_type": "combined",
    "sensitivity": 0.6,
    "enabled": true,
    "min_speech_duration": 0.25
  },
  "wake_word": {"enabled": false}
}
```

#### `wake-word`
- **Use Case**: Wake word activated transcription
- **Auto-start**: Disabled
- **VAD**: Same as vad-triggered
- **Wake Word**: "jarvis" (sensitivity: 0.7, timeout: 30s)

```json
{
  "description": "Wake word activated - say 'jarvis' to start listening",
  "transcription": {"auto_start": false},
  "vad": {
    "detector_type": "combined",
    "sensitivity": 0.6,
    "enabled": true
  },
  "wake_word": {
    "enabled": true,
    "words": ["jarvis"],
    "sensitivity": 0.7,
    "timeout": 30
  }
}
```

### Custom Profile Example

```bash
# Create custom profile for noisy environments
curl -X POST http://localhost:8000/api/v1/system/profiles/noisy-env \
  -H "Content-Type: application/json" \
  -d '{
    "description": "High sensitivity for noisy environments",
    "vad": {
      "sensitivity": 0.8,
      "min_speech_duration": 0.5
    },
    "transcription": {
      "language": "en"
    }
  }'
```

## Usage Examples

### Starting Transcription

```bash
# Start with predefined profile
curl -X POST http://localhost:8000/api/v1/system/start \
  -H "Content-Type: application/json" \
  -d '{"profile": "vad-triggered"}'

# Start with custom configuration
curl -X POST http://localhost:8000/api/v1/system/start \
  -H "Content-Type: application/json" \
  -d '{
    "profile": "vad-triggered",
    "custom_config": {
      "transcription": {"language": "no"},
      "vad": {"sensitivity": 0.7}
    }
  }'
```

### WebSocket Client Example

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.event) {
        case 'transcription':
            if (data.is_final) {
                console.log(`Final: ${data.text}`);
            } else {
                console.log(`Partial: ${data.text}`);
            }
            break;
            
        case 'wake_word':
            console.log(`Wake word "${data.wake_word}" detected (${data.confidence})`);
            break;
    }
};
```

### Python Client Example

```python
import requests
import websocket
import json
import threading

# Start system
response = requests.post("http://localhost:8000/api/v1/system/start", 
    json={"profile": "vad-triggered"})

# WebSocket client
def on_message(ws, message):
    data = json.loads(message)
    print(f"Event: {data['event']}, Data: {data}")

ws = websocket.WebSocketApp("ws://localhost:8000/api/v1/ws",
                          on_message=on_message)
threading.Thread(target=ws.run_forever, daemon=True).start()
```

## Authentication (Optional)

When enabled, all endpoints require authentication:

```bash
# Configure authentication
export STT_SERVER_AUTH_ENABLED="true"
export STT_SERVER_AUTH_TOKEN="your-secret-token"

# Make authenticated requests
curl -X GET http://localhost:8000/api/v1/system/status \
  -H "Authorization: Bearer your-secret-token"
```

## Error Handling

### Standard Error Response Format

```json
{
  "error": "Error message",
  "details": "Detailed error information",
  "status_code": 400,
  "timestamp": "2025-01-27T10:30:00Z"
}
```

### Common HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid request format or parameters |
| 404 | Not Found | Profile or session not found |
| 409 | Conflict | System already running/stopped |
| 500 | Internal Server Error | Command execution failure |

## Extension Guide

### Adding New Controllers

```python
from .BaseController import BaseController
from fastapi import APIRouter

class CustomController(BaseController):
    def __init__(self, command_dispatcher, logger):
        super().__init__(command_dispatcher, logger)
        self.router = APIRouter(prefix="/api/v1/custom")
        self._setup_routes()
    
    def _setup_routes(self):
        @self.router.post("/endpoint")
        async def custom_endpoint(self, request: CustomRequest):
            # Use self.send_command() for command dispatch
            # Use self.create_response() for consistent formatting
            pass
```

### Adding New WebSocket Events

```python
# In WebSocketManager
def register_custom_events(self, event_bus):
    def on_custom_event(event):
        message = {
            "event": "custom_event",
            "data": event.data,
            "timestamp": event.timestamp
        }
        asyncio.run_coroutine_threadsafe(
            self.broadcast_message(message), 
            self.loop
        )
    
    event_bus.subscribe(CustomEvent, on_custom_event)
```

### Adding Middleware

```python
# In ServerModule
from fastapi import Request
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## Development Setup

### Local Development

```bash
# Install dependencies
pip install -e ".[server]"

# Run with auto-reload
python server_example.py --reload

# Access development tools
# - API docs: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - Web UI: http://localhost:8000/
```

### Environment Variables for Development

```bash
export STT_SERVER_DEBUG="true"
export STT_SERVER_HOST="127.0.0.1"
export STT_SERVER_PORT="8000"
export STT_SERVER_CORS_ORIGINS="http://localhost:3000,http://localhost:8080"
```

## Production Deployment

### Configuration

```bash
# Production settings
export STT_SERVER_HOST="0.0.0.0"
export STT_SERVER_PORT="80"
export STT_SERVER_DEBUG="false"
export STT_SERVER_AUTH_ENABLED="true"
export STT_SERVER_AUTH_TOKEN="secure-random-token"
export STT_SERVER_CORS_ORIGINS="https://yourdomain.com"

# Logging
export LOG_LEVEL="INFO"
export LOG_FORMAT="json"
```

### Process Management

```bash
# Using systemd
sudo systemctl start realtime-mlx-stt
sudo systemctl enable realtime-mlx-stt

# Using PM2
pm2 start server_example.py --name "stt-server"
pm2 startup
pm2 save
```

### Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api/v1/ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Performance Considerations

### Scaling Limits

- **WebSocket Connections**: ~1000 concurrent connections per server instance
- **API Throughput**: Limited by MLX transcription speed (~0.5x realtime)
- **Memory Usage**: ~1MB per WebSocket connection, ~500MB for MLX models

### Optimization Tips

1. **Multiple Instances**: Run multiple server instances behind load balancer
2. **Connection Pooling**: Use connection pooling for database operations
3. **Caching**: Cache profile configurations and model loading
4. **Resource Limits**: Set appropriate ulimits for file descriptors

### Monitoring

```python
# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "connections": len(websocket_manager.connections),
        "uptime": time.time() - start_time
    }
```

## Security Considerations

### Authentication
- Token-based authentication available
- No user management - single server token
- HTTPS recommended for production

### CORS
- Configurable origins via environment variable
- Default allows all origins (development only)
- Restrict in production

### Input Validation
- All input validated via Pydantic models
- Base64 audio data size limits
- Request rate limiting recommended

### Dependencies
- Keep FastAPI and dependencies updated
- Monitor security advisories
- Use dependency scanning tools

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check port availability: `lsof -i :8000`
   - Verify Python path and dependencies
   - Check log output for specific errors

2. **WebSocket connections fail**
   - Verify CORS configuration
   - Check firewall settings
   - Test with simple WebSocket client

3. **API returns 500 errors**
   - Enable debug mode for detailed errors
   - Check command dispatcher registration
   - Verify feature module initialization

4. **Audio transcription fails**
   - Verify MLX installation on Apple Silicon
   - Check audio format (16kHz, 16-bit, mono)
   - Validate base64 encoding

### Debug Mode

```bash
export STT_SERVER_DEBUG="true"
python server_example.py

# Check logs
tail -f logs/server.log
```

## Integration with Other Systems

### Message Queues

```python
# Example: Redis pub/sub integration
import redis

redis_client = redis.Redis()

def publish_transcription(text, is_final):
    redis_client.publish('transcription', json.dumps({
        'text': text,
        'is_final': is_final,
        'timestamp': time.time()
    }))
```

### Databases

```python
# Example: PostgreSQL logging
import asyncpg

async def log_transcription(text, confidence, timestamp):
    conn = await asyncpg.connect("postgresql://...")
    await conn.execute("""
        INSERT INTO transcriptions (text, confidence, timestamp) 
        VALUES ($1, $2, $3)
    """, text, confidence, timestamp)
```

### External APIs

```python
# Example: Webhook notifications
import httpx

async def notify_webhook(event_data):
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://your-webhook.com/transcription",
            json=event_data,
            headers={"Authorization": "Bearer webhook-token"}
        )
```

This server provides a production-ready API layer that maintains the vertical slice architecture while offering comprehensive network access to all transcription features.