# Server Examples

This directory contains examples demonstrating how to use the server module of Realtime_mlx_STT.

## Prerequisites

1. **Install server dependencies:**
   ```bash
   # Using uv (recommended)
   uv pip install -e ".[server]"
   
   # Or using pip
   pip install -e ".[server]"
   ```

2. **Set up API keys (if using OpenAI):**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

## Available Examples

### 1. `server_example.py` - Main Server Application

A complete server that exposes REST API and WebSocket endpoints for the STT system.

**Features:**
- REST API for system control and transcription management
- WebSocket for real-time audio streaming and transcription updates
- Automatic API documentation via FastAPI
- Integration with all STT modules (Audio, VAD, Transcription, Wake Word)
- Custom configuration override support

**Usage:**
```bash
python examples/server_example.py
```

**Endpoints:**
- API Documentation: http://localhost:8000/docs
- Health Check: `GET http://localhost:8000/system/status`
- Profiles: `GET http://localhost:8000/system/profiles`
- Start System: `POST http://localhost:8000/system/start`
- Stop System: `POST http://localhost:8000/system/stop`
- WebSocket: `ws://localhost:8000/events`

### 2. `server_client_example.py` - Python Client

Demonstrates how to interact with the server using Python.

**Features:**
- REST API client functions
- WebSocket client for real-time updates
- Example usage patterns

**Usage:**
```bash
# First start the server
python examples/server_example.py

# In another terminal, run the client
python examples/server_client_example.py
```

### 3. `server_web_client.html` - Web Browser Client

A simple HTML/JavaScript client for browser-based interaction.

**Features:**
- Server health monitoring
- Profile selection
- Start/stop transcription
- WebSocket connection management
- Real-time transcription display
- Configuration UI with:
  - Language selection (auto-detect + Norwegian + 12 other languages)
  - Model selection (MLX Whisper + 2 OpenAI models)
  - VAD sensitivity adjustment (0.0 - 1.0)
  - Minimum speech duration control (0.1 - 2.0 seconds)
- Live configuration status display

**Usage:**
1. Start the server: `python examples/server_example.py`
2. Open `server_web_client.html` in a web browser
3. Click "Check Health" to verify server connection
4. Use the interface to control transcription

## API Examples

### Check Server Health
```bash
curl http://localhost:8000/system/status
```

### Get Available Profiles
```bash
curl http://localhost:8000/system/profiles
```

### Start System with Profile
```bash
curl -X POST http://localhost:8000/system/start \
  -H "Content-Type: application/json" \
  -d '{
    "profile": "vad-triggered"
  }'
```

### Start System with Custom Configuration
```bash
curl -X POST http://localhost:8000/system/start \
  -H "Content-Type: application/json" \
  -d '{
    "profile": "vad-triggered",
    "custom_config": {
      "transcription": {
        "engine": "mlx_whisper",
        "model": "whisper-large-v3-turbo",
        "language": "no"
      },
      "vad": {
        "sensitivity": 0.8,
        "min_speech_duration": 0.15
      }
    }
  }'
```

### Stop System
```bash
curl -X POST http://localhost:8000/system/stop
```

## WebSocket Protocol

The WebSocket endpoint (`ws://localhost:8000/events`) supports:

**Client → Server:**
- Binary messages: Raw audio data (16kHz, 16-bit, mono PCM)
- JSON messages: Control commands

**Server → Client:**
- JSON messages with various event types:
  ```json
  {
    "event": "transcription",
    "text": "Hello world",
    "is_final": true,
    "timestamp": 1234567890.123,
    "language": "en"
  }
  ```
  
  ```json
  {
    "event": "speech_detected",
    "timestamp": 1234567890.123
  }
  ```
  
  ```json
  {
    "event": "wake_word",
    "word": "jarvis",
    "timestamp": 1234567890.123
  }
  ```

## Architecture Notes

The server module follows the application's vertical slice architecture:
- Commands are dispatched through the CommandDispatcher
- Events are published via the EventBus
- No direct module dependencies
- Clean separation of concerns

## Troubleshooting

1. **Server won't start:**
   - Ensure server dependencies are installed: `pip install -e ".[server]"`
   - Check if port 8000 is already in use
   - Verify Python version is 3.8+

2. **WebSocket connection fails:**
   - Check browser console for errors
   - Ensure server is running
   - Verify no proxy/firewall blocking WebSocket

3. **Transcription not working:**
   - Check if MLX models are downloaded
   - Verify OpenAI API key if using OpenAI engine
   - Check server logs for errors

## Configuration Options

### Language Support
The system supports automatic language detection or explicit language selection:
- Auto-detect: Leave language empty or null
- Supported languages: `no`, `en`, `es`, `fr`, `de`, `it`, `pt`, `nl`, `pl`, `ru`, `zh`, `ja`, `ko`

### Model Selection
Choose from the available transcription models:
- `whisper-large-v3-turbo` - MLX-optimized Whisper model for Apple Silicon (recommended for local processing)
- `gpt-4o-transcribe` - OpenAI's GPT-4o transcription model (requires API key)
- `gpt-4o-mini-transcribe` - OpenAI's GPT-4o Mini transcription model (faster, requires API key)

### VAD Configuration
Voice Activity Detection can be fine-tuned:
- **Overall Sensitivity** (0.0 - 1.0):
  - 0.3 - Low sensitivity, good for quiet environments
  - 0.6 - Default, balanced for most situations
  - 0.9 - High sensitivity for noisy environments
- **Min Speech Duration** (0.1 - 2.0 seconds):
  - 0.15 - Quick response, may get false positives
  - 0.25 - Default, balanced
  - 0.5+ - Reduces false positives, slower response

#### Advanced VAD Settings (Individual Thresholds)
For fine-grained control, you can adjust each VAD component separately:
- **WebRTC Aggressiveness** (0-3):
  - 0 - Least aggressive (more permissive)
  - 2 - Default, balanced
  - 3 - Most aggressive (more restrictive)
- **Silero Threshold** (0.1-0.9):
  - 0.1 - Very sensitive (more false positives)
  - 0.6 - Default, balanced
  - 0.9 - Very conservative (may miss quiet speech)
- **WebRTC History Threshold** (0.3-0.9):
  - Controls the threshold for WebRTC's history buffer
  - 0.6 - Default, works well for most cases

Example with individual thresholds:
```javascript
customConfig.vad.parameters = {
    webrtc_aggressiveness: 2,
    silero_threshold: 0.7,
    webrtc_threshold: 0.5
}
```

## Production Considerations

For production deployment:
1. Use environment variables for configuration
2. Implement proper authentication/authorization
3. Add rate limiting and request validation
4. Use HTTPS and WSS for secure connections
5. Configure proper CORS headers
6. Implement robust error handling and logging
7. Consider using a production ASGI server like Gunicorn with Uvicorn workers