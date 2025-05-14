# Vertical Slice Architecture for RealtimeSTT_mlx

## Overview

This document outlines a proposed refactoring of the RealtimeSTT_mlx project using vertical slice architecture. Rather than organizing code by technical concerns (layers), this approach organizes code by business capabilities (features), where each feature contains all necessary code across all layers.

## Current Structure

Currently, the project has a traditional layered structure with separate files for different technical concerns:

```
RealtimeSTT/
├── __init__.py
├── audio_input.py          # Audio capture and processing
├── audio_recorder.py       # Main recorder implementation 
├── audio_recorder_client.py # Client for remote transcription
├── mlx_transcriber.py      # MLX transcription interface
└── whisper_turbo.py        # Core MLX model implementation
```

## Proposed Vertical Slice Structure

```
src/
├── Core/
│   ├── Common/
│   │   ├── Configuration/
│   │   │   ├── AudioConfig.py
│   │   │   ├── TranscriptionConfig.py
│   │   │   └── VadConfig.py
│   │   ├── Models/
│   │   │   ├── AudioData.py
│   │   │   └── TranscriptionResult.py
│   │   └── Interfaces/
│   │       ├── IAudioProvider.py
│   │       ├── ITranscriptionEngine.py
│   │       └── IVoiceActivityDetector.py
│   └── Events/
│       ├── AudioEvents.py
│       ├── TranscriptionEvents.py
│       └── EventBus.py
├── Features/
│   ├── AudioCapture/
│   │   ├── Commands/
│   │   │   ├── StartRecordingCommand.py
│   │   │   └── StopRecordingCommand.py
│   │   ├── Events/
│   │   │   ├── AudioChunkCapturedEvent.py
│   │   │   └── RecordingStateChangedEvent.py
│   │   ├── Handlers/
│   │   │   ├── RecordingCommandHandler.py
│   │   │   └── AudioChunkHandler.py
│   │   ├── Providers/
│   │   │   ├── PyAudioInputProvider.py
│   │   │   └── FileAudioProvider.py
│   │   └── AudioCaptureModule.py
│   ├── VoiceActivityDetection/
│   │   ├── Commands/
│   │   │   ├── DetectVoiceActivityCommand.py
│   │   │   └── ConfigureVadCommand.py
│   │   ├── Events/
│   │   │   ├── SpeechDetectedEvent.py
│   │   │   └── SilenceDetectedEvent.py
│   │   ├── Detectors/
│   │   │   ├── WebRtcVadDetector.py
│   │   │   ├── SileroVadDetector.py
│   │   │   └── CombinedVadDetector.py
│   │   ├── Handlers/
│   │   │   └── VoiceActivityHandler.py
│   │   └── VadModule.py
│   ├── WakeWordDetection/
│   │   ├── Commands/
│   │   │   └── DetectWakeWordCommand.py
│   │   ├── Events/
│   │   │   └── WakeWordDetectedEvent.py
│   │   ├── Detectors/
│   │   │   ├── PorcupineDetector.py
│   │   │   └── OpenWakeWordDetector.py
│   │   ├── Handlers/
│   │   │   └── WakeWordHandler.py
│   │   └── WakeWordModule.py
│   ├── Transcription/
│   │   ├── Commands/
│   │   │   ├── TranscribeAudioCommand.py
│   │   │   ├── ProcessAudioChunkCommand.py
│   │   │   └── ConfigureTranscriberCommand.py
│   │   ├── Events/
│   │   │   ├── TranscriptionCompletedEvent.py
│   │   │   └── PartialTranscriptionEvent.py
│   │   ├── Models/
│   │   │   ├── MelSpectrogram.py
│   │   │   └── TranscriptionOptions.py
│   │   ├── Engines/
│   │   │   ├── MlxWhisperEngine.py
│   │   │   ├── BatchTranscriber.py
│   │   │   └── StreamingTranscriber.py
│   │   ├── Handlers/
│   │   │   ├── BatchTranscriptionHandler.py
│   │   │   └── StreamingTranscriptionHandler.py
│   │   └── TranscriptionModule.py
│   └── RemoteProcessing/
│       ├── Commands/
│       │   ├── ConnectToServerCommand.py
│       │   └── SendAudioChunkCommand.py
│       ├── Events/
│       │   ├── ServerConnectionStatusEvent.py
│       │   └── RemoteTranscriptionReceivedEvent.py
│       ├── Client/
│       │   └── WebSocketClient.py
│       ├── Server/
│       │   └── WebSocketServer.py
│       ├── Handlers/
│       │   ├── RemoteCommandHandler.py
│       │   └── RemoteEventHandler.py
│       └── RemoteModule.py
├── Infrastructure/
│   ├── Audio/
│   │   ├── AudioProcessing.py
│   │   └── DeviceManagement.py
│   ├── Models/
│   │   ├── ModelLoader.py
│   │   └── ModelRegistry.py
│   └── Threading/
│       ├── WorkerPool.py
│       └── SafePipe.py
└── Application/
    ├── Facade/
    │   ├── RealtimeSTT.py
    │   └── MLXTranscriber.py
    ├── Configuration/
    │   └── ApplicationConfig.py
    └── Extensions/
        └── FacadeExtensions.py
```

## Key Concepts

### 1. Feature Modules

Each feature module is a complete vertical slice that contains all necessary code for that feature, including:

- **Commands**: Request objects that represent operations to be performed
- **Events**: Notifications that something has happened
- **Handlers**: Process commands and produce events
- **Domain Models**: Feature-specific data structures
- **Services/Providers**: Implementations of functionality needed by the feature

### 2. Core Module

Contains only the minimal shared interfaces, common models, and infrastructure needed across features.

### 3. Application Layer

Provides a simplified API for consumers of the library, hiding the complexity of the vertical slices.

## Implementation Strategy

### 1. Command/Event Pattern

Each feature should implement a command/event pattern:

```python
# Example command
class TranscribeAudioCommand:
    def __init__(self, audio_data: np.ndarray, options: TranscriptionOptions = None):
        self.audio_data = audio_data
        self.options = options or TranscriptionOptions()

# Example event
class TranscriptionCompletedEvent:
    def __init__(self, text: str, confidence: float, processing_time: float):
        self.text = text
        self.confidence = confidence
        self.processing_time = processing_time
```

### 2. Feature Registration

Each feature module should have a registration method to wire up its dependencies:

```python
# Example module registration
class TranscriptionModule:
    @staticmethod
    def register(container):
        # Register all handlers and services
        container.register(TranscriptionCommandHandler)
        container.register(MlxWhisperEngine)
        # Register event handlers
        event_bus = container.resolve(EventBus)
        event_bus.subscribe(SpeechDetectedEvent, lambda e: TranscriptionModule._handle_speech_detected(e, container))
```

### 3. Event-Based Communication

Features should communicate primarily through events:

```python
# Publishing an event
self.event_bus.publish(TranscriptionCompletedEvent(text, confidence, processing_time))

# Subscribing to an event
self.event_bus.subscribe(SpeechDetectedEvent, self._handle_speech_detected)
```

## Migration Path

1. **Create Core Interfaces**: Define interfaces for key components
2. **Implement Feature by Feature**: Start with the most isolated features
3. **Update Facade Last**: Keep the public API mostly unchanged during migration
4. **Write Tests**: Ensure each feature works in isolation and together

## Specific Recommendations for Key Files

### MLXTranscriber.py

1. Split into:
   - Core interface (`ITranscriptionEngine`)
   - Base implementation in the Transcription feature
   - Facade that provides the same API

### AudioRecorder.py

1. Split into multiple features:
   - `AudioCapture` feature for microphone handling
   - `VoiceActivityDetection` feature for speech detection
   - `Transcription` feature for processing audio

### whisper_turbo.py

1. Move to `Transcription/Engines` as a backend implementation
2. Use interfaces to decouple it from the rest of the system

## Benefits of This Approach

1. **High Cohesion**: All code for a feature is together
2. **Low Coupling**: Features interact mainly through events
3. **Easier Testing**: Each feature can be tested in isolation
4. **Better Maintainability**: Changes to a feature are localized
5. **Parallel Development**: Teams can work on different features
