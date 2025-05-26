# Realtime_mlx_STT

Realtime_mlx_STT is a high-performance speech-to-text transcription library optimized exclusively for Apple Silicon. It leverages Apple's MLX framework to run the Whisper large-v3-turbo model with maximum performance on macOS devices with Apple Silicon chips.

## Features

- **Real-time transcription** with low latency for macOS applications
- **Apple Silicon optimization** using MLX with Neural Engine acceleration
- **Voice activity detection** using both WebRTC and Silero models with full configurability
- **Wake word detection** for hands-free activation using Porcupine
- **Server/Client architecture** with HTTP API and WebSocket support
- **Web UI** with comprehensive configuration controls for all parameters
- **Lazy initialization** for optimal resource usage
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
git clone https://github.com/kristofferv98/Realtime_mlx_STT.git
cd Realtime_mlx_STT

# Set up Python environment (requires Python 3.8+ but 3.11+ recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with pip using pyproject.toml (setup.py has been removed)
pip install -e .

# For development, include dev dependencies
pip install -e ".[dev]"

# For OpenAI transcription support
pip install -e ".[openai]"

# For clipboard/auto-typing features
pip install -e ".[clipboard]"
```

> **Note**: The project uses `pyproject.toml` as the single source of truth for dependencies. The old `setup.py` has been removed to avoid configuration conflicts.

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
└── Application/                # Public API and server components
    ├── Facade/                 # Direct library usage API
    ├── Server/                 # HTTP/WebSocket server implementation
    │   ├── Controllers/        # API endpoints for different feature areas
    │   ├── WebSocket/          # Real-time event broadcasting
    │   └── Configuration/      # Server and profile configuration
    └── Client/                 # Python client library for server interaction
```

### Key Components

- **Audio Capture**: Handles microphone input and audio file processing
- **Voice Activity Detection**: Detects speech using WebRTC, Silero, or combined approaches
  - Lazy initialization for optimal resource usage
  - Full configurability of all parameters (thresholds, frame settings, buffers)
  - Individual control over each VAD component
- **Wake Word Detection**: Recognizes specific trigger phrases to activate the system
- **Transcription**: Processes audio using MLX-optimized Whisper models or OpenAI API
- **Server**: FastAPI-based HTTP and WebSocket server for remote access
  - Web UI with comprehensive configuration controls
  - Real-time transcription display
  - Support for custom configuration overrides
- **Client Library**: Python client for interacting with the server
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

### Server/Client Usage

```python
# Server-side (run in a separate process)
from src.Application import ServerModule
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Features import AudioCaptureModule, TranscriptionModule, VadModule, WakeWordModule

# Create command dispatcher and event bus
command_dispatcher = CommandDispatcher()
event_bus = EventBus()

# Register all feature modules
AudioCaptureModule.register(command_dispatcher, event_bus)
VadModule.register(command_dispatcher, event_bus)
TranscriptionModule.register(command_dispatcher, event_bus)
WakeWordModule.register(command_dispatcher, event_bus)

# Start the server (default: http://127.0.0.1:8000)
server = ServerModule.register(command_dispatcher, event_bus)

# Client-side (in another process or machine)
import requests
import websocket
import json
import threading

# Start system with profile and custom configuration
response = requests.post("http://localhost:8000/system/start", 
    json={
        "profile": "vad-triggered",
        "custom_config": {
            "transcription": {
                "engine": "mlx_whisper",
                "model": "whisper-large-v3-turbo",
                "language": "no"  # Norwegian
            },
            "vad": {
                "sensitivity": 0.7,
                "parameters": {
                    # Individual VAD thresholds
                    "webrtc_aggressiveness": 2,
                    "silero_threshold": 0.6,
                    # Frame processing settings
                    "speech_confirmation_frames": 2,
                    "silence_confirmation_frames": 30
                }
            }
        }
    })

# Connect to WebSocket for real-time events
def on_message(ws, message):
    data = json.loads(message)
    if data["event"] == "transcription":
        print(f"Transcription: {data['text']}")

ws = websocket.WebSocketApp("ws://localhost:8000/events",
                          on_message=on_message)
threading.Thread(target=ws.run_forever, daemon=True).start()

# Later, stop the system
requests.post("http://localhost:8000/system/stop")
```

### Included Example Scripts

The repository includes several ready-to-use example scripts:

1. **VAD-Triggered Transcription** - Captures and transcribes speech from your microphone using MLX-optimized Whisper with Voice Activity Detection:
   ```bash
   python examples/vad_transcription.py
   ```

2. **Auto-Typing Transcription** - Transcribes speech and automatically types it into any application:
   ```bash
   # Install required dependency
   uv pip install pyautogui
   
   # Run with default settings (typing latest text)
   python examples/vad_transcription_with_pasting.py
   
   # Run with full history mode (types all accumulated transcriptions)
   python examples/vad_transcription_with_pasting.py --paste-mode full
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

### Server Examples

For server-based usage with web interface, check the `example_server/` directory:

```bash
# Start the server with automatic browser opening
python example_server/server_example.py
```

This provides:
- REST API endpoints for transcription control
- WebSocket for real-time streaming
- Web interface for easy configuration and monitoring
- Python client examples

See [example_server/README.md](example_server/README.md) for detailed documentation.

### Additional Configuration

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
   python examples/vad_transcription.py --no-progress-bars
   python examples/openai_transcription.py --no-progress-bars
   ```

## Testing

The project includes comprehensive tests for each feature and component:

```bash
# Run all tests
python tests/run_tests.py

# Run tests for a specific feature or component
python tests/run_tests.py -f VoiceActivityDetection
python tests/run_tests.py -f Infrastructure
python tests/run_tests.py -f Application  # Server/Client tests

# Run a specific test with verbose output
python tests/run_tests.py -t webrtc_vad_test -v
python tests/run_tests.py -t test_server_module -v

# Test with PYTHONPATH (if imports fail)
PYTHONPATH=/path/to/Realtime_mlx_STT python tests/run_tests.py
```

The Server implementation includes tests for:
- API Controllers (Transcription and System)
- WebSocket connections and event broadcasting
- Configuration and profile management
- Command/Event integration

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

## Recent Updates (January 2025)

- **Configuration Cleanup**: Migrated to `pyproject.toml` as the single source of truth for dependencies
- **Architecture Improvements**: Added proper exports to `__init__.py` files for better import ergonomics
- **Code Quality**: Fixed thread safety issues and improved error handling
- **Consistency**: All features now follow the same structure (Commands, Events, Handlers, Models)
- **Removed Stubs**: Cleaned up empty directories and placeholder features

For detailed changes, see `specs/cleanup_implementation_summary.md`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the base Whisper large-v3-turbo model
- [MLX](https://github.com/ml-explore/mlx) for Apple Silicon optimization
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) for the original audio processing concepts
- [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) for wake word detection
- [Hugging Face](https://huggingface.co) for model distribution infrastructure