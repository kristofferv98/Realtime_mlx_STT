# Server API Models Documentation

## Overview

The Server API models define the data contracts for the REST/WebSocket API endpoints in the Realtime_mlx_STT server. All models are built using Pydantic v2 for automatic validation, serialization, and OpenAPI schema generation.

## Model Categories

### 1. System Models (`SystemModels.py`)

These models handle system-wide operations including server status, configuration profiles, and general system configuration.

#### ServerStatusResponse
**Purpose**: Provides comprehensive server status information.

**Fields**:
- `status` (str, required): Server operational status
- `version` (str, required): Server version string
- `uptime` (float, required): Server uptime in seconds
- `active_features` (List[str], required): List of currently active feature modules
- `active_connections` (int, required): Number of active WebSocket connections

**Usage**: Response from `/system/status` endpoint

#### ProfileRequest
**Purpose**: Request to start the system with a specific configuration profile.

**Fields**:
- `profile` (str, required): Name of the profile to activate (e.g., "vad-triggered", "wake-word")
- `custom_config` (Dict[str, Any], optional): Custom configuration to merge with the selected profile

**Usage**: Request body for `/system/profile/start` endpoint

#### ProfileListResponse
**Purpose**: Lists all available configuration profiles.

**Fields**:
- `profiles` (List[str], required): List of available profile names
- `default` (str, required): Name of the default profile

**Usage**: Response from `/system/profiles` endpoint

#### ProfileData
**Purpose**: Contains detailed profile configuration data.

**Fields**:
- `name` (str, required): Profile identifier
- `config` (Dict[str, Any], required): Complete profile configuration as nested dictionary

**Usage**: Response from `/system/profile/{profile_name}` endpoint

#### GeneralConfigRequest
**Purpose**: Allows configuration of multiple system components in a single request.

**Fields**:
- `transcription` (Dict[str, Any], optional): Transcription engine configuration
- `vad` (Dict[str, Any], optional): Voice Activity Detection configuration
- `wake_word` (Dict[str, Any], optional): Wake word detection configuration
- `audio` (Dict[str, Any], optional): Audio capture configuration

**Usage**: Request body for `/system/configure` endpoint

#### SystemErrorResponse
**Purpose**: Standardized error response format for all system endpoints.

**Fields**:
- `status` (str, default="error"): Error status indicator
- `code` (str, required): Machine-readable error code
- `message` (str, required): Human-readable error message
- `details` (Dict[str, Any], optional): Additional error context or debugging information

**Usage**: Error response from any endpoint

### 2. Transcription Models (`TranscriptionModels.py`)

These models handle transcription-specific operations including engine configuration, session management, and audio processing.

#### TranscriptionConfigRequest
**Purpose**: Configure the transcription engine with specific parameters.

**Fields**:
- `engine_type` (str, required): Type of transcription engine ("mlx_whisper" or "openai")
- `model_name` (str, required): Model identifier (e.g., "whisper-large-v3-turbo", "whisper-1")
- `language` (str, optional): ISO language code for transcription (e.g., "en", "es")
- `beam_size` (int, optional): Beam size for beam search decoding (affects accuracy/speed)
- `options` (Dict[str, Any], optional): Engine-specific options (e.g., temperature, prompt)

**Usage**: Request body for `/transcription/configure` endpoint

#### TranscriptionSessionRequest
**Purpose**: Manage transcription sessions for continuous audio streams.

**Fields**:
- `session_id` (str, optional): Session identifier (auto-generated if not provided)

**Usage**: Request body for `/transcription/session/start` and `/transcription/session/stop` endpoints

#### TranscribeAudioRequest
**Purpose**: Submit audio data for transcription.

**Fields**:
- `audio_data` (str, required): Base64-encoded audio data (16kHz, 16-bit, mono PCM)
- `session_id` (str, optional): Session ID for continuous transcription context
- `is_final` (bool, default=False): Indicates if this is the final audio chunk in a stream

**Usage**: Request body for `/transcription/audio` endpoint

#### TranscriptionResult
**Purpose**: Contains transcription results with metadata.

**Fields**:
- `text` (str, required): Transcribed text
- `is_final` (bool, required): Whether this is a final or intermediate result
- `confidence` (float, optional): Confidence score between 0-1
- `session_id` (str, optional): Associated session ID for continuous transcription
- `segments` (List[Dict[str, Any]], optional): Detailed segment information with timestamps

**Segment Structure** (when provided):
```python
{
    "start": 0.0,      # Start time in seconds
    "end": 2.5,        # End time in seconds
    "text": "Hello",   # Segment text
    "tokens": [...]    # Token information (engine-specific)
}
```

**Usage**: Response from transcription operations (REST or WebSocket)

#### TranscriptionStatusResponse
**Purpose**: Provides current transcription system status.

**Fields**:
- `active` (bool, required): Whether transcription is currently active
- `engine` (str, required): Current engine type ("mlx_whisper" or "openai")
- `model` (str, required): Current model name
- `language` (str, optional): Current language setting
- `sessions` (List[str], optional): List of active session IDs

**Usage**: Response from `/transcription/status` endpoint

## Model Relationships

### Configuration Flow
1. **ProfileRequest** → Activates a profile → **ProfileData** configuration applied
2. **GeneralConfigRequest** → Updates specific components → Merges with active configuration
3. **TranscriptionConfigRequest** → Configures transcription → Updates **TranscriptionStatusResponse**

### Transcription Flow
1. **TranscriptionSessionRequest** → Creates session → Returns session_id
2. **TranscribeAudioRequest** → Processes audio → Generates **TranscriptionResult**
3. Session tracking maintained in **TranscriptionStatusResponse**

## Integration with Internal Models

### Model Transformation Patterns

1. **Audio Data Transformation**:
   - External: Base64-encoded string in `TranscribeAudioRequest.audio_data`
   - Internal: NumPy array for `TranscribeAudioCommand.audio_chunk`
   - Conversion: `np.frombuffer(base64.b64decode(audio_data), dtype=np.float32)`

2. **Field Mapping**:
   - `TranscribeAudioRequest.is_final` → `TranscribeAudioCommand.is_last_chunk`
   - External models use REST-friendly names, internal commands use domain-specific names

3. **Configuration Mapping**:
   - External configurations (Dict[str, Any]) are validated and mapped to internal configuration objects
   - Profile configurations are merged with custom configurations before applying

## API Versioning Considerations

Currently, the API uses implicit versioning through the model definitions. Future considerations:

1. **Field Evolution**:
   - All fields with defaults can be added without breaking changes
   - Optional fields allow gradual adoption of new features
   - Pydantic's validation ensures backward compatibility

2. **Model Versioning Strategy**:
   - Models could be versioned using namespaces (e.g., `v1.TranscriptionResult`)
   - API endpoints could include version in path (e.g., `/v1/transcription/audio`)

## Validation Rules

### Implicit Validations (via Pydantic)
- Required fields must be present
- Type checking is automatic
- String fields cannot be None if required
- Lists default to empty if not provided

### Business Logic Validations (in Controllers)
- Session IDs must exist for stop operations
- Audio data must be valid base64
- Engine types must be supported ("mlx_whisper" or "openai")
- Profile names must exist in ProfileManager

## Best Practices for Model Usage

1. **Request Models**:
   - Use `Body(...)` in FastAPI endpoints to ensure proper parsing
   - Provide clear field descriptions for OpenAPI documentation
   - Use optional fields for backward compatibility

2. **Response Models**:
   - Always return consistent structure using `create_standard_response()`
   - Include relevant metadata in responses
   - Use appropriate HTTP status codes

3. **Error Handling**:
   - Return `SystemErrorResponse` for all errors
   - Include specific error codes for client handling
   - Provide actionable error messages

## Example Usage

### Configure and Start Transcription
```python
# 1. Configure transcription engine
config_request = TranscriptionConfigRequest(
    engine_type="mlx_whisper",
    model_name="whisper-large-v3-turbo",
    language="en",
    beam_size=5
)

# 2. Start a session
session_request = TranscriptionSessionRequest()
# Returns: {"session_id": "uuid-here"}

# 3. Send audio
audio_request = TranscribeAudioRequest(
    audio_data="base64-encoded-audio",
    session_id="uuid-here",
    is_final=False
)

# 4. Receive results (via WebSocket or polling)
result = TranscriptionResult(
    text="Hello world",
    is_final=True,
    confidence=0.95,
    session_id="uuid-here"
)
```

### Error Response Example
```python
SystemErrorResponse(
    status="error",
    code="INVALID_AUDIO_FORMAT",
    message="Audio data must be 16kHz, 16-bit, mono PCM",
    details={
        "received_format": "8kHz",
        "expected_format": "16kHz"
    }
)
```

## Model Extension Guidelines

When adding new models:

1. **Naming Convention**:
   - Request models: `{Feature}{Action}Request`
   - Response models: `{Feature}{Action}Response` or `{Feature}Result`
   - Status models: `{Feature}StatusResponse`

2. **Field Guidelines**:
   - Use descriptive field names
   - Provide Field descriptions for documentation
   - Make fields optional when possible for flexibility
   - Use appropriate types (avoid Any when possible)

3. **Documentation**:
   - Add docstrings to model classes
   - Document field purposes in Field descriptions
   - Update this documentation with new models

## Notes

- All models use Pydantic v2 features
- Models are designed for REST/WebSocket API compatibility
- Serialization is handled automatically by FastAPI
- Models provide OpenAPI schema generation out of the box