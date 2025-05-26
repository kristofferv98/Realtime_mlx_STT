# Controllers Module Documentation

## Overview

The Controllers module provides REST API endpoints for the Realtime MLX STT Server. It follows a modular architecture where each controller handles a specific domain of functionality. Controllers integrate with the application's command/event architecture, dispatching commands through the CommandDispatcher and subscribing to events via the EventBus.

## Architecture

### Base Controller Pattern

All controllers inherit from `BaseController`, which provides:
- Standardized response formatting
- Command dispatcher integration
- Event bus access
- FastAPI router setup
- Centralized error handling

### Key Design Principles

1. **Command-Driven**: Controllers translate HTTP requests into commands
2. **Event-Aware**: Controllers can subscribe to and handle domain events
3. **Stateless**: Controllers maintain minimal state, delegating to feature modules
4. **Standardized**: Consistent response formats and error handling

## Controllers

### BaseController

**Purpose**: Abstract base class providing common functionality for all controllers

**Key Features**:
- Router initialization with URL prefix support
- Command dispatching with error handling
- Standardized response formatting
- Logging integration

**Methods**:
- `send_command(command)`: Dispatches commands with automatic error handling
- `create_standard_response(status_code, data, message)`: Creates consistent API responses
- `_register_routes()`: Abstract method for route registration (override in subclasses)

**Error Handling**:
- Catches command execution exceptions
- Returns HTTP 500 with error details
- Logs errors for debugging

### SystemController

**Purpose**: Manages system-wide operations including profiles, startup/shutdown, and status

**Prefix**: `/system`

**Dependencies**:
- CommandDispatcher
- EventBus
- ProfileManager

**State Management**:
- Tracks system running state
- Maintains active features list
- Stores current profile
- Manages active session ID

#### Endpoints

##### GET `/system/status`
**Purpose**: Get current server status
**Response Model**: `ServerStatusResponse`
**Returns**:
- Server status (online/offline)
- Version
- Uptime in seconds
- Active features list
- Active connection count

##### GET `/system/info`
**Purpose**: Get system information
**Returns**:
- Platform details
- Python version
- Hostname
- CPU count
- Available features

##### GET `/system/profiles`
**Purpose**: List available configuration profiles
**Response Model**: `ProfileListResponse`
**Returns**:
- List of profile names
- Default profile name

##### GET `/system/profiles/{name}`
**Purpose**: Get specific profile configuration
**Parameters**:
- `name`: Profile name (path parameter)
**Response Model**: `ProfileData`
**Error**: 404 if profile not found

##### POST `/system/profiles`
**Purpose**: Save a configuration profile
**Request Body**: `ProfileData`
**Returns**: Standard success response
**Error**: 400 if save fails

##### DELETE `/system/profiles/{name}`
**Purpose**: Delete a configuration profile
**Parameters**:
- `name`: Profile name (path parameter)
**Returns**: Standard success response
**Error**: 400 if delete fails

##### POST `/system/start`
**Purpose**: Start the system with a specific profile
**Request Body**: `ProfileRequest`
```json
{
  "profile": "vad-triggered",
  "custom_config": {
    "transcription": {...},
    "vad": {...}
  }
}
```
**Process**:
1. Load profile configuration
2. Merge with custom config if provided
3. Configure transcription engine
4. Configure VAD
5. Configure wake word (if enabled)
6. Start audio recording
7. Enable VAD or wake word detection
8. Start transcription session (if auto_start)

**Commands Dispatched**:
- `ConfigureTranscriptionCommand`
- `ConfigureVadCommand`
- `ConfigureWakeWordCommand`
- `StartRecordingCommand`
- `EnableVadProcessingCommand`
- `StartTranscriptionSessionCommand`
- `StartWakeWordDetectionCommand`

**Error Handling**: Attempts cleanup on failure

##### POST `/system/stop`
**Purpose**: Stop the system
**Process**:
1. Stop wake word detection
2. Stop transcription session
3. Disable VAD processing
4. Stop audio recording
5. Clear system state

**Commands Dispatched**:
- `StopWakeWordDetectionCommand`
- `StopTranscriptionSessionCommand`
- `DisableVadProcessingCommand`
- `StopRecordingCommand`

##### POST `/system/config`
**Purpose**: Update system configuration (placeholder)
**Request Body**: `GeneralConfigRequest`
**Note**: Currently not fully implemented

### TranscriptionController

**Purpose**: Manages transcription operations including configuration, sessions, and audio processing

**Prefix**: `/transcription`

**Dependencies**:
- CommandDispatcher
- EventBus

**State Management**:
- Active sessions set
- Current configuration (engine, model, language, active status)

#### Endpoints

##### POST `/transcription/configure`
**Purpose**: Configure the transcription engine
**Request Body**: `TranscriptionConfigRequest`
```json
{
  "engine_type": "mlx_whisper",
  "model_name": "whisper-large-v3-turbo",
  "language": "en",
  "beam_size": 5,
  "options": {}
}
```
**Commands Dispatched**: `ConfigureTranscriptionCommand`
**Updates**: Current configuration state

##### POST `/transcription/session/start`
**Purpose**: Start a new transcription session
**Request Body**: `TranscriptionSessionRequest`
```json
{
  "session_id": "optional-custom-id"
}
```
**Returns**: Session ID (generated if not provided)
**Commands Dispatched**: `StartTranscriptionSessionCommand`
**Updates**: Active sessions tracking

##### POST `/transcription/session/stop`
**Purpose**: Stop a transcription session
**Request Body**: `TranscriptionSessionRequest`
```json
{
  "session_id": "required-session-id"
}
```
**Commands Dispatched**: `StopTranscriptionSessionCommand`
**Error**: 400 if session_id missing

##### POST `/transcription/audio`
**Purpose**: Transcribe audio data
**Request Body**: `TranscribeAudioRequest`
```json
{
  "audio_data": "base64-encoded-audio",
  "session_id": "optional-session-id",
  "is_final": false
}
```
**Process**:
1. Decode base64 audio data
2. Convert to numpy array
3. Dispatch transcribe command
4. Results delivered via WebSocket events

**Commands Dispatched**: `TranscribeAudioCommand`
**Special Handling**: Test mode support for unit tests
**Error**: 400 for invalid base64, 500 for processing errors

##### GET `/transcription/status`
**Purpose**: Get transcription system status
**Response Model**: `TranscriptionStatusResponse`
**Returns**:
- Active status
- Current engine
- Current model
- Language setting
- Active session IDs

## Request/Response Models

### System Models

- **ServerStatusResponse**: Server status information
- **ProfileRequest**: Profile activation request with optional custom config
- **ProfileListResponse**: Available profiles list
- **ProfileData**: Profile name and configuration
- **GeneralConfigRequest**: System-wide configuration update
- **SystemErrorResponse**: Standardized error response

### Transcription Models

- **TranscriptionConfigRequest**: Engine configuration
- **TranscriptionSessionRequest**: Session management
- **TranscribeAudioRequest**: Audio data for transcription
- **TranscriptionResult**: Transcription output
- **TranscriptionStatusResponse**: Transcription system status

## Integration Points

### Command Dispatcher
- All business logic executed via commands
- Controllers translate HTTP requests to commands
- Automatic error handling and logging

### Event Bus
- Controllers can subscribe to domain events
- WebSocket integration for real-time updates
- Event-driven architecture support

### WebSocket Events
- Transcription updates broadcast to clients
- Wake word detection notifications
- Real-time status updates

## Extension Guidelines

### Adding a New Controller

1. Create controller class inheriting from `BaseController`
2. Define URL prefix in constructor
3. Override `_register_routes()` method
4. Use `send_command()` for business logic
5. Use `create_standard_response()` for consistency
6. Register in `ServerModule._register_controllers()`

### Adding New Endpoints

1. Define request/response models in Models directory
2. Add route decorator in `_register_routes()`
3. Validate input data
4. Dispatch appropriate commands
5. Handle errors consistently
6. Update controller state if needed

### Best Practices

1. **Thin Controllers**: Keep business logic in commands/handlers
2. **Consistent Responses**: Use standard response format
3. **Error Handling**: Let BaseController handle command errors
4. **Logging**: Log important operations and errors
5. **State Management**: Minimize controller state
6. **Testing**: Mock command dispatcher for unit tests

## Authentication & Authorization

Currently, the controllers do not implement authentication or authorization. The system relies on:
- Network-level security (binding to localhost by default)
- CORS configuration for browser security
- No built-in user authentication

For production use, consider adding:
- API key authentication
- JWT token support
- Rate limiting
- Request validation middleware

## Middleware

The server applies the following middleware:
- **CORS**: Configured in ServerModule with customizable origins
- **No custom middleware**: Controllers rely on FastAPI's built-in features

## Error Handling

### Standard Error Response Format
```json
{
  "status": "error",
  "code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {}
}
```

### HTTP Status Codes
- **200**: Success
- **400**: Bad Request (invalid input)
- **404**: Not Found (resource doesn't exist)
- **500**: Internal Server Error (command execution failure)

### Error Propagation
1. Command execution errors caught in `send_command()`
2. Logged with full stack trace
3. Sanitized error message returned to client
4. Original exception details in logs only

## Performance Considerations

1. **Command Execution**: Synchronous by default
2. **WebSocket Updates**: Asynchronous event broadcasting
3. **Audio Processing**: Base64 encoding overhead for REST API
4. **Session Management**: In-memory session tracking
5. **Profile Loading**: File-based with caching potential

## Security Considerations

1. **Input Validation**: Pydantic models enforce type safety
2. **Base64 Audio**: Prevents binary data injection
3. **Command Validation**: Commands validate their own inputs
4. **Error Messages**: Sanitized to prevent information leakage
5. **CORS**: Configurable origin restrictions

## Future Enhancements

1. **Authentication**: Add API key or JWT support
2. **Rate Limiting**: Prevent abuse
3. **Metrics**: Add performance monitoring
4. **Caching**: Cache profile configurations
5. **Batch Operations**: Support multiple audio chunks
6. **Streaming**: Direct audio streaming support
7. **GraphQL**: Alternative API interface
8. **OpenAPI**: Enhanced API documentation