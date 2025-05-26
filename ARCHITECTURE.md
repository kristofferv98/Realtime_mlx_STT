# Realtime MLX STT Architecture

## Overview

The codebase follows a clean layered architecture with proper separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Examples & CLI                        │
│                  (User entry points)                     │
├─────────────────────────────────────────────────────────┤
│                     API Layer                            │
│            (realtime_mlx_stt package)                    │
│     • STTClient - High-level client interface           │
│     • TranscriptionSession - Session management          │  
│     • Transcriber - Simple transcription API            │
│     • User-friendly configs (ModelConfig, etc.)         │
├─────────────────────────────────────────────────────────┤
│                  Application Layer                       │
│              (src/Application/Server)                    │
│     • WebSocket server implementation                   │
│     • REST API controllers                              │
│     • Network session management                         │
├─────────────────────────────────────────────────────────┤
│                   Features Layer                         │
│                  (src/Features/*)                        │
│     • AudioCapture - Audio input handling               │
│     • VoiceActivityDetection - VAD processing           │
│     • Transcription - STT engines (MLX, OpenAI)         │
│     • WakeWordDetection - Wake word detection           │
│     • NoiseSuppression - Audio enhancement              │
├─────────────────────────────────────────────────────────┤
│                     Core Layer                           │
│                    (src/Core/*)                          │
│     • CommandDispatcher - Command routing               │
│     • EventBus - Event pub/sub system                   │
│     • Interfaces - Common contracts                     │
├─────────────────────────────────────────────────────────┤
│                Infrastructure Layer                      │
│              (src/Infrastructure/*)                      │
│     • Logging - Centralized logging                     │
│     • ProgressBar - UI progress indicators              │
└─────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Vertical Slice Architecture
Each feature is self-contained with its own:
- Commands (inputs)
- Events (outputs)
- Handlers (business logic)
- Models (data structures)

### 2. Command/Event Pattern
Features communicate exclusively through:
- **Commands**: Requests to perform actions
- **Events**: Notifications of state changes
- No direct feature-to-feature dependencies

### 3. Proper Layering
- **API Layer**: Thin wrapper sending commands and listening to events
- **Features Layer**: Contains all business logic
- **No Logic Duplication**: API doesn't reimplement features

## How It Works

### API Layer (User-Facing)
```python
# User creates a session
session = TranscriptionSession(
    model=ModelConfig(engine="mlx_whisper"),
    vad=VADConfig(sensitivity=0.8)
)

# API layer translates to commands
ConfigureTranscriptionCommand(engine="mlx_whisper")
ConfigureVadCommand(sensitivity=0.8)
StartRecordingCommand()

# API layer listens for events
TranscriptionUpdatedEvent → callback(result)
```

### Application Layer (Network Server)
```python
# WebSocket receives request
{"action": "start_session", "config": {...}}

# Server uses same commands
ConfigureTranscriptionCommand(...)
StartTranscriptionSessionCommand(...)

# Server forwards events to WebSocket
TranscriptionUpdatedEvent → ws.send(result)
```

## Configuration Philosophy

### User-Facing Configs (API Layer)
Simple, validated, user-friendly:
```python
ModelConfig(
    engine="openai",  # Simple string
    model="whisper-1",
    language="en"
)
```

### Internal Configs (Features Layer)
Detailed, technical, implementation-specific:
```python
TranscriptionConfig(
    engine_type=EngineType.OPENAI,
    model_name="whisper-1",
    language_code="en-US",
    sample_rate=16000,
    # ... many more technical fields
)
```

## Why This Architecture?

1. **Maintainability**: Changes in features don't affect API
2. **Testability**: Each layer can be tested independently  
3. **Flexibility**: Easy to add new features or engines
4. **Consistency**: Same features work via API or server
5. **Separation**: User API vs implementation details

## Common Misconceptions

❌ **"The API duplicates Feature logic"**
- No, it uses commands/events to delegate to Features

❌ **"Two config systems are redundant"**
- No, they serve different purposes (user vs internal)

❌ **"Server should use the API layer"**
- No, both are peers that use Features differently

## Adding New Features

1. Create feature in `src/Features/YourFeature/`
2. Define commands, events, handlers, models
3. Register with CommandDispatcher and EventBus
4. Add user-friendly wrapper in API layer (if needed)
5. Add WebSocket support in Application layer (if needed)