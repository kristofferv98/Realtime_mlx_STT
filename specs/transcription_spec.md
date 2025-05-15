# Transcription Feature Specification

## Overview

The Transcription feature implements speech-to-text functionality using MLX-optimized Whisper large-v3-turbo model for Apple Silicon. It integrates with AudioCapture and VoiceActivityDetection features to provide a complete audio processing pipeline. The implementation follows vertical slice architecture principles with process isolation for stability and performance.

## Architecture

### Core Components

1. **ITranscriptionEngine Interface** (already implemented)
   - Located at: `src/Core/Common/Interfaces/transcription_engine.py`
   - Key methods: `start()`, `transcribe()`, `add_audio_chunk()`, `get_result()`, `cleanup()`, `is_running()`
   - Purpose: Abstract interface for any transcription implementation

2. **Directory Structure** (already set up)
   - Commands: `src/Features/Transcription/Commands/`
   - Engines: `src/Features/Transcription/Engines/`
   - Events: `src/Features/Transcription/Events/`
   - Handlers: `src/Features/Transcription/Handlers/`
   - Models: `src/Features/Transcription/Models/`

## Implementation Plan

### 1. Engine Implementation

**MlxWhisperEngine** (`src/Features/Transcription/Engines/MlxWhisperEngine.py`)
- Implements `ITranscriptionEngine`
- Responsibilities:
  - Load and initialize MLX Whisper model
  - Process audio data using mel spectrogram conversion
  - Generate transcriptions using transformer architecture
  - Manage streaming transcription with KV cache
  - Support both batch and streaming modes
- Key optimizations:
  - Memory management for Apple Silicon
  - Caching of model components and filters
  - Process isolation via multiprocessing

**TranscriptionProcessManager** (`src/Features/Transcription/Engines/TranscriptionProcessManager.py`)
- Manages a separate process for transcription
- Uses pipe-based IPC for communication
- Handles process lifecycle (start, stop, monitoring)
- Provides thread-safe access to transcription results

### 2. Models

**TranscriptionResult** (`src/Features/Transcription/Models/TranscriptionResult.py`)
- Properties: `text`, `confidence`, `language`, `is_final`, `segments`, `timestamps`
- Used to represent transcription outputs

**TranscriptionConfig** (`src/Features/Transcription/Models/TranscriptionConfig.py`)
- Properties for engine configuration: `model_name`, `compute_type`, `language`, `beam_size`
- Additional streaming options: `chunk_duration`, `overlap`, `realtime_factor`

**TranscriptionSession** (`src/Features/Transcription/Models/TranscriptionSession.py`)
- Tracks state for ongoing transcription sessions
- Properties: `session_id`, `start_time`, `audio_chunks`, `current_text`, `language`

### 3. Commands

**TranscribeAudioCommand** (`src/Features/Transcription/Commands/TranscribeAudioCommand.py`)
- Parameters: `audio_chunk`, `session_id`, `is_first_chunk`, `is_last_chunk`, `language`
- Requests transcription of a specific audio chunk

**ConfigureTranscriptionCommand** (`src/Features/Transcription/Commands/ConfigureTranscriptionCommand.py`)
- Parameters: `engine_type`, `model_name`, `language`, `beam_size`, `compute_type`, `options`
- Updates transcription engine configuration

**StartTranscriptionSessionCommand** (`src/Features/Transcription/Commands/StartTranscriptionSessionCommand.py`)
- Parameters: `session_id`, `language`, `config`
- Initializes a new transcription session

**StopTranscriptionSessionCommand** (`src/Features/Transcription/Commands/StopTranscriptionSessionCommand.py`)
- Parameters: `session_id`, `flush_remaining_audio`
- Finalizes and ends a transcription session

### 4. Events

**TranscriptionStartedEvent** (`src/Features/Transcription/Events/TranscriptionStartedEvent.py`)
- Properties: `session_id`, `timestamp`, `language`
- Published when transcription begins for a speech segment

**TranscriptionUpdatedEvent** (`src/Features/Transcription/Events/TranscriptionUpdatedEvent.py`)
- Properties: `session_id`, `text`, `is_final`, `confidence`, `timestamp`
- Published for both partial and final transcription results

**TranscriptionErrorEvent** (`src/Features/Transcription/Events/TranscriptionErrorEvent.py`)
- Properties: `session_id`, `error_message`, `error_type`, `timestamp`
- Published when transcription encounters an error

### 5. Handler Implementation

**TranscriptionCommandHandler** (`src/Features/Transcription/Handlers/TranscriptionCommandHandler.py`)
- Implements `ICommandHandler`
- Handles all transcription-related commands
- Manages transcription sessions and engine instances
- Publishes transcription events
- Coordinates with the process manager for isolated processing

### 6. Module Facade

**TranscriptionModule** (`src/Features/Transcription/TranscriptionModule.py`)
- Public static API for the Transcription feature
- Methods:
  - `register()`: Register with command dispatcher and event bus
  - `configure()`: Configure transcription engine
  - `transcribe_audio()`: Transcribe a specific audio chunk
  - `start_session()`: Start a new transcription session
  - `stop_session()`: End a transcription session
  - Event subscription methods for results and errors

## Integration with Other Features

### AudioCapture Integration
- Subscribes to `AudioChunkCapturedEvent`
- Routes audio chunks to active transcription sessions
- Handles proper audio format conversion

### VoiceActivityDetection Integration
- Subscribes to `SpeechDetectedEvent` and `SilenceDetectedEvent`
- Automatically starts transcription sessions when speech begins
- Finalizes transcription when speech ends
- Maintains session IDs consistent with VAD speech segments

## Technical Requirements

### Process Isolation
- Transcription engine runs in separate Python process
- Communication via pipe-based IPC
- Resource management (memory, CPU, Neural Engine)
- Graceful handling of process failures

### Performance Optimization
- Memory management for Apple Silicon
- Efficient audio buffer handling
- Cached preprocessing for mel spectrograms
- KV cache management for streaming transcription
- Adaptive chunk sizing based on latency measurements

### Error Handling
- Graceful recovery from model errors
- Timeout mechanisms for unresponsive transcription
- Proper resource cleanup
- Detailed error reporting

## Implementation Phases

### Phase 1: Basic Infrastructure
- Implement `TranscriptionProcessManager`
- Create basic `MlxWhisperEngine` (non-streaming)
- Define data models and commands
- Implement command handler with process isolation

### Phase 2: Batch Transcription
- Complete implementation of batch transcription
- Add integration with VAD for complete speech segments
- Implement proper error handling and recovery

### Phase 3: Streaming Transcription
- Add streaming capability with KV cache management
- Implement partial results publishing
- Optimize for low latency
- Add chunking and context management

### Phase 4: Optimization and Testing
- Performance benchmarking
- Memory usage optimization
- Latency reduction
- Comprehensive unit and integration tests

## Testing Strategy

### Unit Tests
- Test each component in isolation with mocks
- Verify command and event handling
- Test process isolation utilities

### Integration Tests
- End-to-end tests with audio input to text output
- Test with various audio conditions
- Test integration with other features

## Acceptance Criteria

1. Successfully transcribe speech with >95% accuracy on clean audio
2. Support both batch and streaming transcription modes
3. Process audio in near real-time (RTF <1.0 on M1/M2 chips)
4. Seamlessly integrate with AudioCapture and VoiceActivityDetection
5. Handle errors gracefully with proper reporting
6. Manage resources efficiently without memory leaks
7. Support configuration changes at runtime