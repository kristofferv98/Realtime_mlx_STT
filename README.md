# Realtime_mlx_STT

Realtime_mlx_STT is a high-performance speech-to-text transcription library optimized exclusively for Apple Silicon. It leverages Apple's MLX framework to run the Whisper large-v3-turbo model with maximum performance on macOS devices with Apple Silicon chips.

## Features

- **Real-time transcription** with low latency for macOS applications
- **Apple Silicon optimization** using MLX with Neural Engine acceleration
- **Voice activity detection** using both WebRTC and Silero models
- **Vertical slice architecture** for modular and maintainable code
- **Event-driven design** for flexible integration
- **Thread-safe operations** for responsive applications

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3) - Required, not optional
- **Python 3.11+**
- **MLX** for Apple Silicon optimization
- **PyAudio** for audio capture
- **WebRTC VAD** and **Silero VAD** for voice activity detection
- **Torch** and **NumPy** for audio processing

> **Important Note**: This library is specifically optimized for Apple Silicon and will not work on Intel-based Macs or other platforms. It requires the Neural Engine found in Apple Silicon chips to achieve optimal performance.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Realtime_mlx_STT.git
cd Realtime_mlx_STT

# Set up Python environment (recommended to use version 3.11)
python -m venv env
source env/bin/activate

# Install with uv using pyproject.toml
uv pip install -e .

# For development, include dev dependencies
uv pip install -e ".[dev]"
```

## Architecture

Realtime_mlx_STT uses a vertical slice architecture, organizing code by features rather than technical layers. Each feature contains all necessary components:

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

### Key Components

- **Audio Capture**: Handles microphone input and audio file processing
- **Voice Activity Detection**: Detects speech using WebRTC, Silero, or combined approaches
- **Transcription**: Processes audio using MLX-optimized Whisper models
- **Event System**: Enables loose coupling between components

## Usage Example

```python
from src.Application.Facade import RealtimeSTT

# Initialize the transcription system
transcriber = RealtimeSTT.create_transcriber(
    vad_type="combined",        # Use combined VAD (WebRTC + Silero)
    vad_sensitivity=0.8,        # Higher sensitivity for voice detection
    model="whisper-large-v3",   # Specify the model to use
    language="en",              # Specify language (or auto-detect)
    streaming=True              # Enable real-time streaming
)

# Register callback for when speech is detected
transcriber.on_speech_detected = lambda confidence: print(f"Speech detected: {confidence:.2f}")

# Register callback for transcription results
transcriber.on_transcription_update = lambda text: print(f"Transcribing: {text}")
transcriber.on_transcription_complete = lambda text: print(f"Final: {text}")

# Start listening
transcriber.start_listening()

# Wait for user input to stop
input("Press Enter to stop listening...")

# Stop and clean up
transcriber.stop_listening()
transcriber.cleanup()
```

## Testing

The project includes comprehensive tests for each feature:

```bash
# Run all tests
python tests/run_tests.py

# Run tests for a specific feature
python tests/run_tests.py -f VoiceActivityDetection

# Run a specific test with verbose output
python tests/run_tests.py -t webrtc_vad_test -v
```

## Performance

On Apple Silicon (M1/M2/M3), the MLX-optimized Whisper-large-v3-turbo model typically achieves:

- **Batch mode**: ~0.3-0.5x realtime (processes 60 seconds of audio in 20-30 seconds)
- **Streaming mode**: ~0.5-0.7x realtime (processes audio with ~2-3 second latency)

The MLX implementation takes full advantage of the Neural Engine in Apple Silicon chips, providing significantly better performance than CPU-based implementations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the base Whisper large-v3-turbo model
- [MLX](https://github.com/ml-explore/mlx) for Apple Silicon optimization
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) for the original audio processing concepts
- [Hugging Face](https://huggingface.co) for model distribution infrastructure