# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Realtime_mlx_STT is a real-time speech-to-text transcription library optimized for Apple Silicon. It combines the real-time audio processing capabilities of RealtimeSTT with the high-performance Whisper large-v3-turbo model optimized through MLX. The project aims to provide low-latency, high-accuracy transcription for macOS applications.

## Repository Structure

The project follows a modular architecture with clear separation between audio processing, voice activity detection, and transcription components:

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

## Key Components & Their Roles

- **audio_recorder.py**: Main class that orchestrates audio capture, voice activity detection, and transcription. It handles buffer management, timings, and callbacks.

- **audio_input.py**: Manages microphone input using PyAudio and provides audio data to the recorder.

- **whisper_turbo.py**: MLX-optimized implementation of Whisper large-v3-turbo, including:
  - Audio preprocessing (mel spectrogram generation)
  - Model architecture (encoder-decoder Transformer)
  - Tokenization and text generation
  - Both batch and streaming transcription modes

- **safepipe.py**: Provides thread-safe data transfer between audio capture and transcription processes.

- **audio_recorder_client.py**: Client implementation for potentially remote transcription services.

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

### Key Files to Modify for Common Tasks

- **To modify audio processing**: Edit `RealtimeSTT/audio_input.py`
- **To adjust VAD sensitivity**: Update parameters in `RealtimeSTT/audio_recorder.py`
- **To optimize the transcription model**: Edit `RealtimeSTT/whisper_turbo.py`
- **To change callback behavior**: Modify event handlers in `RealtimeSTT/audio_recorder.py`

### Adding New Features

1. For new audio processing capabilities:
   - Add methods to `audio_input.py` or extend with a new module
   - Update `audio_recorder.py` to use new capabilities

2. For improved model capabilities:
   - Modify `whisper_turbo.py` or add alternative model implementations
   - Update initialization in `__init__.py` to support model selection

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

## Operational Notes
- Dont run infinite loop python scripts rather copy the command to run it with mcp clipboard, and Kristoffer will run it and provide the output.

## Test Files and Media
- we can test file transcription from file with /Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/bok_konge01.mp3 if needed