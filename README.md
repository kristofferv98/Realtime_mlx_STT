# Realtime_mlx_STT

Realtime_mlx_STT is a high-performance speech-to-text transcription library optimized exclusively for Apple Silicon. It leverages Apple's MLX framework to run the Whisper large-v3-turbo model with maximum performance on macOS devices with Apple Silicon chips.

## Features

- **Real-time transcription** with low latency for macOS applications
- **Apple Silicon optimization** using MLX with Neural Engine acceleration
- **Voice activity detection** using both WebRTC and Silero models
- **Wake word detection** for hands-free activation using Porcupine
- **Centralized logging system** with runtime configuration and log rotation
- **Vertical slice architecture** for modular and maintainable code
- **Event-driven design** for flexible integration
- **Thread-safe operations** for responsive applications

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3) - Required, not optional
- **Python 3.11+**
- **MLX** for Apple Silicon optimization
- **PyAudio** for audio capture
- **WebRTC VAD** and **Silero VAD** for voice activity detection
- **Porcupine** for wake word detection (optional)
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

# For wake word detection
uv pip install -e ".[wakeword]"
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
│   ├── Logging/                # Centralized logging system
│   └── ProgressBar/            # Progress bar management
└── Application/                # Public API facades
```

### Key Components

- **Audio Capture**: Handles microphone input and audio file processing
- **Voice Activity Detection**: Detects speech using WebRTC, Silero, or combined approaches
- **Wake Word Detection**: Recognizes specific trigger phrases to activate the system
- **Transcription**: Processes audio using MLX-optimized Whisper models
- **Event System**: Enables loose coupling between components
- **Logging System**: Provides centralized logging with runtime configuration and log rotation
- **Progress Bar Control**: Centralized management of progress bars for cleaner user output

## Usage Examples

### Basic API Usage

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

### Included Example Scripts

The repository includes several ready-to-use example scripts:

1. **Continuous Transcription** - Captures and transcribes speech from your microphone using MLX-optimized Whisper:
   ```bash
   python examples/continuous_transcription.py
   ```

2. **Auto-Typing Transcription** - Transcribes speech and automatically types it into any application:
   ```bash
   # Install required dependency
   uv pip install pyautogui
   
   # Run with default settings (typing latest text)
   python examples/continuous_transcription_pasting.py
   
   # Run with full history mode (types all accumulated transcriptions)
   python examples/continuous_transcription_pasting.py --paste-mode full
   ```

3. **Wake Word Activation** - Transcribes speech only after detecting a wake word:
   ```bash
   # Install required dependencies
   uv pip install -e ".[wakeword]"
   
   # Set your Porcupine access key (get from https://console.picovoice.ai/)
   export PORCUPINE_ACCESS_KEY=your_key_here
   
   # Run with default "porcupine" wake word
   python examples/wake_word_detection.py
   
   # Run with custom wake words
   python examples/wake_word_detection.py --wake-words "jarvis,computer" --sensitivity 0.7
   ```

4. **OpenAI Transcription** - Transcribes speech using OpenAI's cloud-based GPT-4o-transcribe model:
   ```bash
   # Install required dependencies
   uv pip install -e ".[openai]"
   
   # Set your OpenAI API key
   export OPENAI_API_KEY=your_api_key_here
   
   # Run with GPT-4o-transcribe (highest quality)
   python examples/openai_transcription.py
   
   # Run with GPT-4o-mini-transcribe (faster, lower cost)
   python examples/openai_transcription.py --model gpt-4o-mini-transcribe
   ```

5. **File Transcription** - Transcribes audio from a file:
   ```bash
   python examples/transcribe_file.py path/to/audio/file.mp3
   ```

6. **Check Audio Devices** - Lists all available audio input devices:
   ```bash
   python examples/check_audio_devices.py
   ```

7. **Configuring Logging and Progress Bars** - The centralized systems can be used in your applications:
   ```python
   # Import modules
   from src.Infrastructure.Logging import LoggingModule, LogLevel
   from src.Infrastructure.ProgressBar.ProgressBarManager import ProgressBarManager

   # Initialize progress bar control first (disable tqdm progress bars globally)
   ProgressBarManager.initialize(disabled=True)
   
   # Initialize logging with desired configuration
   LoggingModule.initialize(
       console_level="INFO",
       file_enabled=True,
       file_path="logs/application.log",
       rotation_enabled=True,
       feature_levels={
           "AudioCapture": LogLevel.DEBUG,
           "Transcription": LogLevel.INFO
       }
   )

   # Get a logger for your module
   logger = LoggingModule.get_logger(__name__)
   logger.info("Application started")
   
   # Enable runtime log level adjustment
   LoggingModule.start_control_server()
   
   # Later, change log levels using the provided utility
   # python scripts/change_log_level.py AudioCapture DEBUG
   ```

   You can control both logging and progress bars with environment variables:
   ```bash
   # Set up environment with progress bars hidden (cleaner output)
   source scripts/set_logging_env.sh prod
   
   # Set up environment with progress bars visible (helpful for debugging)
   source scripts/set_logging_env.sh dev
   
   # Manually control progress bars
   export DISABLE_PROGRESS_BARS=true  # Hide tqdm progress bars
   ```
   
   Command-line arguments for controlling progress bars in example scripts:
   ```bash
   # Run with progress bars hidden (though progress bars are now disabled by default)
   python examples/wake_word_detection.py --no-progress-bars
   python examples/continuous_transcription.py --no-progress-bars
   python examples/openai_transcription.py --no-progress-bars
   ```

## Testing

The project includes comprehensive tests for each feature:

```bash
# Run all tests
python tests/run_tests.py

# Run tests for a specific feature
python tests/run_tests.py -f VoiceActivityDetection

# Run tests for Infrastructure components
python tests/run_tests.py -f Infrastructure

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
- [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) for wake word detection
- [Hugging Face](https://huggingface.co) for model distribution infrastructure