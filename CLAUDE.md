# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Realtime_mlx_STT is a real-time speech-to-text transcription library optimized for Apple Silicon. It combines the real-time audio processing capabilities of RealtimeSTT with the high-performance Whisper large-v3-turbo model optimized through MLX. The project aims to provide low-latency, high-accuracy transcription for macOS applications.

## Current Repository Structure

The project currently follows a modular architecture with clear separation between audio processing, voice activity detection, and transcription components:

```
Realtime_mlx_STT/
├── LICENSE
├── README.md
├── setup.py
├── requirements.txt
├── RealtimeSTT/                # Core library code
│   ├── __init__.py             # Package initialization
│   ├── audio_input.py          # Audio capture and processing
│   ├── audio_recorder.py       # Main recorder implementation
│   ├── audio_recorder_client.py # Client for remote transcription
│   ├── safepipe.py             # Thread-safe data transfer utilities
│   ├── whisper_turbo.py        # MLX-optimized whisper-large-v3-turbo
│   └── warmup_audio.wav        # Sample audio for model initialization
└── tests/                      # Test scripts
    ├── simple_test.py          # Basic functionality test
    ├── realtimestt_test.py     # Real-time transcription test
    └── vad_test.py             # Voice activity detection test
```

## Vertical Slice Architecture Refactoring

We are currently refactoring the project to use a vertical slice architecture approach. This architecture organizes code by business capabilities (features) rather than technical concerns (layers). See `vertical-slice-overview.md` for a detailed overview of the planned structure and migration path.

### Key Benefits of Vertical Slice Architecture

- **High Cohesion**: All code related to a specific feature is located together
- **Low Coupling**: Features interact primarily through events
- **Easier Testing**: Each feature can be tested in isolation
- **Better Maintainability**: Changes to a feature are localized
- **Parallel Development**: Teams can work on different features independently

### Planned New Structure

The new architecture will organize code into feature modules:

```
src/
├── Core/                       # Core interfaces and models
├── Features/                   # Feature-based vertical slices
│   ├── AudioCapture/           # Audio input handling
│   ├── VoiceActivityDetection/ # Speech detection
│   ├── Transcription/          # Audio-to-text processing
│   ├── WakeWordDetection/      # Wake word detection
│   └── RemoteProcessing/       # Remote transcription
├── Infrastructure/             # Cross-cutting concerns
└── Application/                # Public API facades
```

Each feature module will contain all necessary components:
- Commands and events
- Handlers and services
- Feature-specific models and interfaces
- Implementation details

## Current Key Components & Their Roles

- **audio_recorder.py**: Main class that orchestrates audio capture, voice activity detection, and transcription. It handles buffer management, timings, and callbacks.

- **audio_input.py**: Manages microphone input using PyAudio and provides audio data to the recorder.

- **whisper_turbo.py**: MLX-optimized implementation of Whisper large-v3-turbo, including:
  - Audio preprocessing (mel spectrogram generation)
  - Model architecture (encoder-decoder Transformer)
  - Tokenization and text generation
  - Both batch and streaming transcription modes

- **safepipe.py**: Provides thread-safe data transfer between audio capture and transcription processes.

- **audio_recorder_client.py**: Client implementation for potentially remote transcription services.

## Migration Path to Vertical Slice Architecture

In the vertical slice architecture:

1. **audio_recorder.py** will be split into multiple feature modules:
   - `Features/AudioCapture` for microphone handling
   - `Features/VoiceActivityDetection` for speech detection
   - `Features/Transcription` for processing audio

2. **whisper_turbo.py** will move to:
   - `Features/Transcription/Engines/MlxWhisperEngine.py`
   - Will implement the core interface `ITranscriptionEngine` which is already defined
   - Will use process isolation to improve stability and resource management

3. **audio_input.py** will be refactored to:
   - `Features/AudioCapture/Providers/PyAudioInputProvider.py`
   - Will implement a core interface `IAudioProvider`

4. **safepipe.py** will move to:
   - `Infrastructure/Threading/SafePipe.py`

Refer to `vertical-slice-overview.md` for a detailed implementation plan.

## Primary Data Flows

1. **Audio Capture → Transcription**:
   - `audio_input.py` captures audio from microphone
   - Voice Activity Detection (VAD) determines if speech is present
   - Audio chunks are buffered and processed
   - `whisper_turbo.py` converts audio to text using MLX acceleration
   - Results are returned via callbacks

2. **Real-time Mode**:
   - Audio is processed in small chunks as it arrives
   - Partial transcriptions are generated during speech
   - Results are streamed back to the application
   - Final result is generated when silence is detected

3. **Batch Mode**:
   - Larger chunks of audio are processed at once
   - More accurate but higher latency
   - Used when accuracy is prioritized over real-time response

## Technical Implementation Details

### MLX Integration

The project leverages Apple's MLX framework for optimized inference on Apple Silicon. Key integration points:

- **Model Loading**: Uses MLX to load and run the whisper-large-v3-turbo model
- **Audio Processing**: Accelerated mel spectrogram generation
- **Tensor Operations**: Utilizes MLX's optimized matrix operations
- **Memory Efficiency**: Takes advantage of MLX's unified memory model

### Voice Activity Detection

The system uses a two-stage voice activity detection:

1. **WebRTC VAD**: Fast initial detection of speech activity
2. **Silero VAD**: More accurate verification of speech segments
3. **Silence Detection**: Identifies end of speech for timely processing

### Streaming Architecture

- Uses a producer-consumer pattern for audio processing
- Thread-safe queues manage data flow between components
- Reference counting ensures proper resource management
- Avoids blocking the main thread during processing

## Common Development Tasks

### Building and Running

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run basic test
python tests/simple_test.py

# Run real-time test
python tests/realtimestt_test.py
```

### Key Files to Modify for Common Tasks (Current Structure)

- **To modify audio processing**: Edit `RealtimeSTT/audio_input.py`
- **To adjust VAD sensitivity**: Update parameters in `RealtimeSTT/audio_recorder.py`
- **To optimize the transcription model**: Edit `RealtimeSTT/whisper_turbo.py`
- **To change callback behavior**: Modify event handlers in `RealtimeSTT/audio_recorder.py`

### Key Files to Modify (After Vertical Slice Architecture Refactoring)

- **To modify audio processing**: Edit modules in `src/Features/AudioCapture`
- **To adjust VAD sensitivity**: Update configurations in `src/Features/VoiceActivityDetection`
- **To optimize the transcription model**: Edit `src/Features/Transcription/Engines/MlxWhisperEngine.py`
- **To change callback behavior**: Modify event handlers in respective feature modules

### Adding New Features (Current Structure)

1. For new audio processing capabilities:
   - Add methods to `audio_input.py` or extend with a new module
   - Update `audio_recorder.py` to use new capabilities

2. For improved model capabilities:
   - Modify `whisper_turbo.py` or add alternative model implementations
   - Update initialization in `__init__.py` to support model selection

### Adding New Features (After Vertical Slice Architecture Refactoring)

1. Create a new feature module in `src/Features/` with:
   - Commands for operations to perform
   - Events for notifications
   - Handlers to process commands
   - Feature-specific models and services

2. Register the new feature module with the application:
   - Add interfaces to `src/Core` if needed
   - Update the application facade to expose new functionality

## Key Dependencies

- **mlx**: Apple's ML framework optimized for Apple Silicon
- **pyaudio**: Cross-platform audio I/O library
- **numpy**: Numerical computing library for audio processing
- **huggingface_hub**: For model downloading and management
- **webrtcvad**: Voice activity detection library
- **tiktoken**: Tokenizer for the Whisper model

## Special Considerations

1. **Apple Silicon Optimization**:
   - The MLX implementation is specifically optimized for Apple Silicon
   - Performance on Intel Macs will be significantly lower
   - The library won't run on non-macOS platforms without modification

2. **Audio Hardware Dependencies**:
   - Requires proper audio input device configuration
   - May need adjustments for different microphone types
   - Sample rate and audio format requirements are strict

3. **Memory Management**:
   - ML models consume significant memory
   - Long recordings may require memory optimization
   - Consider batch size and model precision for memory constraints

4. **Real-time Performance**:
   - Balances accuracy vs. latency
   - VAD sensitivity affects user experience
   - Buffer sizes impact both latency and accuracy

## Audio Processing Guidelines

- Audio should always be processed at 16kHz, 16-bit, mono format for consistency with Whisper models

- Use a two-stage VAD approach: fast initial detection with WebRTC followed by more accurate Silero verification

- Buffer size for audio processing should balance latency (smaller) and accuracy (larger)

- Always include pre-buffering to capture the start of speech that might occur before VAD detection

## Current Development Status

We are currently working on the AudioCapture and VoiceActivityDetection features in the vertical slice architecture. Specifically:

### AudioCapture Feature (COMPLETED)
- Implementation is complete with directory structure, commands, events, and providers
- PyAudioInputProvider and FileAudioProvider are fully implemented and tested
- Commands for listing devices, selecting devices, and recording control are in place
- Fixed circular import issues in Core modules that affected AudioCapture features
- All tests are passing and properly documented in `tests/Features/AudioCapture/README.md`
- Feature is independent of hardware and can be tested in any environment
- Module is ready for integration with other features

### VoiceActivityDetection Feature (COMPLETED)
- Three voice activity detection implementations:
  - WebRtcVadDetector: Fast, lightweight detector for quick speech detection
  - SileroVadDetector: Higher accuracy ML-based detector that requires torchaudio
  - CombinedVadDetector: Two-stage detector that uses both WebRTC and Silero for best results
- Fixed test suite with comprehensive tests for each detector type
- Updated Silero VAD to handle model loading failures gracefully and support chunked audio processing
- Added support for processing audio in appropriate chunk sizes (512 samples for 16kHz)
- All tests are passing and fully automated

### Passing Tests
- WebRTC VAD Tests: Tests WebRtcVadDetector functionality (PASSING)
- Silero VAD Tests: Tests SileroVadDetector functionality (PASSING with torchaudio installed)
- Combined VAD Tests: Tests CombinedVadDetector functionality (PASSING)
- AudioCapture Tests: 26 tests across all AudioCapture components (PASSING, 1 skipped)
  - PyAudioInputProvider tests with hardware independence
  - FileAudioProvider tests with file system independence
  - AudioCommandHandler tests for command processing
  - AudioCaptureModule facade tests
  - Event publishing and handling tests

### Next Steps
- Ready to begin implementing the Transcription feature in the vertical slice architecture
- Core and supporting features (AudioCapture, VoiceActivityDetection) are fully implemented and tested
- Created a comprehensive specification document at `specs/transcription_spec.md`
- Implementation plan includes:
  1. Implement the MlxWhisperEngine with process isolation
  2. Create command/event structure for transcription operations
  3. Build TranscriptionCommandHandler to manage engine instances
  4. Develop TranscriptionModule facade for public API access
  5. Integrate with AudioCapture and VoiceActivityDetection features
  6. Implement both batch and streaming transcription modes
  7. Optimize for Apple Silicon performance
  8. Develop comprehensive test suite

## Operational Notes
- Don't run infinite loop python scripts rather copy the command to run it with mcp clipboard, and Kristoffer will run it and provide the output.
- The Silero VAD component requires torchaudio: `uv pip install torchaudio`

## Test Files and Media
- We can test file transcription from file with /Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/bok_konge01.mp3 if needed