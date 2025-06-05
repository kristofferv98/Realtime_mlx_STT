# Realtime_mlx_STT

[![PyPI version](https://badge.fury.io/py/realtime-mlx-stt.svg)](https://badge.fury.io/py/realtime-mlx-stt)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey.svg)](https://support.apple.com/en-us/HT211814)

High-performance speech-to-text transcription library optimized exclusively for Apple Silicon. Leverages MLX framework for real-time on-device transcription with low latency.

> ⚠️ **IMPORTANT: This library is designed for LOCAL USE ONLY on macOS with Apple Silicon.** The included server is a development tool and should NOT be exposed to the internet or used in production environments without implementing proper security measures.

## Features

- **Real-time transcription** with low latency using MLX Whisper
- **Multiple APIs** - Python API, REST API, and WebSocket for different use cases  
- **Apple Silicon optimization** using MLX with Neural Engine acceleration
- **Voice activity detection** with WebRTC and Silero (configurable thresholds)
- **Wake word detection** using Porcupine ("Jarvis", "Alexa", etc.)
- **OpenAI integration** for cloud-based transcription alternative
- **Interactive CLI** for easy exploration of features
- **Web UI** with modern interface and real-time updates
- **Profile system** for quick configuration switching
- **Event-driven architecture** with command pattern
- **Thread-safe** and production-ready

## Language Selection

The Whisper large-v3-turbo model supports 99 languages with intelligent language detection:

- **Language-specific mode**: When you select a specific language (e.g., Norwegian, French, Spanish), the model uses language-specific tokens that significantly improve transcription accuracy for that language
- **Multi-language capability**: Even with a language selected, Whisper can still transcribe other languages if spoken - it's not restricted to only the selected language
- **Accuracy benefit**: Selecting the primary language you'll be speaking provides much more accurate transcription compared to auto-detect mode
- **Auto-detect mode**: When no language is specified, the model attempts to detect the language automatically, though with potentially lower accuracy

For example, if you select Norwegian (`no`) as your language:
- Norwegian speech will be transcribed with high accuracy
- English speech will still be transcribed correctly if spoken
- The model uses the Norwegian language token (50288) to optimize for Norwegian

This behavior matches OpenAI's Whisper API - the language parameter guides but doesn't restrict the model.

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3) - Required, not optional
- **Python 3.9+** (3.11+ recommended for best performance)
- **MLX** for Apple Silicon optimization
- **PyAudio** for audio capture
- **WebRTC VAD** and **Silero VAD** for voice activity detection
- **Porcupine** for wake word detection (optional)
- **Torch** and **NumPy** for audio processing

> **Important Note**: This library is specifically optimized for Apple Silicon and will not work on Intel-based Macs or other platforms. It requires the Neural Engine found in Apple Silicon chips to achieve optimal performance.

## Installation

### Install from PyPI (Recommended)

```bash
# Basic installation
pip install realtime-mlx-stt

# With OpenAI support for cloud transcription
pip install "realtime-mlx-stt[openai]"

# With development tools
pip install "realtime-mlx-stt[dev]"

# With server support for REST/WebSocket APIs
pip install "realtime-mlx-stt[server]"

# Install everything
pip install "realtime-mlx-stt[openai,server,dev]"
```

## 📚 Documentation

- **[Usage Guide](USAGE_GUIDE.md)** - Common patterns and troubleshooting
- **[API Reference](realtime_mlx_stt/README.md)** - Detailed API documentation
- **[Examples](examples/)** - Working code examples

### Install from Source

```bash
# Clone the repository
git clone https://github.com/kristofferv98/Realtime_mlx_STT.git
cd Realtime_mlx_STT

# Set up Python environment (requires Python 3.9+ but 3.11+ recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Quick Start

### Interactive CLI (Recommended)

The easiest way to explore all features:

```bash
python examples/cli.py
```

This provides a menu-driven interface for:
- Quick 10-second transcription
- Continuous streaming mode
- OpenAI cloud transcription
- Wake word detection
- Audio device selection
- Language configuration

### Python API

```python
from realtime_mlx_stt import STTClient

# Simple transcription
client = STTClient()
for result in client.transcribe(duration=10):
    print(result.text)

# With OpenAI
client = STTClient(openai_api_key="sk-...")
for result in client.transcribe(engine="openai"):
    print(result.text)

# Wake word mode
client.start_wake_word("jarvis")
```

### Server Mode

> **Security Note**: The server is for local development only and binds to localhost by default. Do NOT expose it to the internet without proper authentication and security measures.

```bash
# Start server (localhost only - safe)
cd example_server
python server_example.py

# Opens web UI at http://localhost:8000
```

## Architecture

The library provides two specialized interfaces built on a common Features layer:

```
┌─────────────────────────────────────────────────┐
│          User Interfaces                         │
│  • CLI (examples/cli.py)                        │
│  • Web UI (example_server/)                     │
├─────────────────────────────────────────────────┤
│          API Layers                             │
│  • Python API (realtime_mlx_stt/)              │
│  • REST/WebSocket (src/Application/Server/)    │
├─────────────────────────────────────────────────┤
│          Features Layer                         │
│  • AudioCapture                                │
│  • VoiceActivityDetection                      │
│  • Transcription (MLX/OpenAI)                  │
│  • WakeWordDetection                           │
├─────────────────────────────────────────────────┤
│          Core & Infrastructure                  │
│  • Command/Event System                         │
│  • Logging & Configuration                      │
└─────────────────────────────────────────────────┘
```

### Key Design Principles

- **Vertical Slice Architecture**: Each feature is self-contained with Commands, Events, Handlers, and Models
- **Dual API Design**: Python API optimized for direct use, Server API optimized for multi-client scenarios
- **Event-Driven**: Features communicate via commands and events, not direct dependencies
- **Production Ready**: Thread-safe, lazy initialization, comprehensive error handling

## API Documentation

### Python API (realtime_mlx_stt)

```python
from realtime_mlx_stt import STTClient, TranscriptionSession, create_transcriber

# Method 1: Modern Client API
client = STTClient(
    openai_api_key="sk-...",     # Optional
    default_engine="mlx_whisper", # or "openai"
    default_language="en"         # or None for auto-detect
)

# Transcribe for fixed duration
for result in client.transcribe(duration=10):
    print(f"{result.text} (confidence: {result.confidence})")

# Streaming with stop word
with client.stream() as stream:
    for result in stream:
        print(result.text)
        if "stop" in result.text.lower():
            break

# Method 2: Session-based API
from realtime_mlx_stt import TranscriptionSession, ModelConfig, VADConfig

session = TranscriptionSession(
    model=ModelConfig(engine="mlx_whisper", language="no"),
    vad=VADConfig(sensitivity=0.8),
    on_transcription=lambda r: print(r.text)
)

with session:
    time.sleep(30)  # Listen for 30 seconds

# Method 3: Simple Transcriber
from realtime_mlx_stt import Transcriber
transcriber = Transcriber(language="es")
text = transcriber.transcribe_from_mic(duration=5)
print(f"You said: {text}")
```

### REST API

```bash
# Start system with profile
curl -X POST http://localhost:8000/api/v1/system/start \
  -H "Content-Type: application/json" \
  -d '{
    "profile": "vad-triggered",
    "custom_config": {
      "transcription": {"language": "fr"},
      "vad": {"sensitivity": 0.7}
    }
  }'

# Get system status
curl http://localhost:8000/api/v1/system/status

# Transcribe audio file
curl -X POST http://localhost:8000/api/v1/transcription/audio \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "base64_encoded_audio_data"}'
```

### WebSocket Events

```javascript
const ws = new WebSocket('ws://localhost:8000/events');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'transcription':
            if (data.is_final) {
                console.log(`Final: ${data.text}`);
            } else {
                console.log(`Transcribing: ${data.text}`);
            }
            break;
        case 'wake_word':
            console.log(`Wake word: ${data.wake_word}`);
            break;
    }
```

## Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-..."        # For OpenAI transcription
export PORCUPINE_ACCESS_KEY="..."     # For wake word detection
# Alternative names for Picovoice universal key (same as PORCUPINE_ACCESS_KEY):
# export PICOVOICE_ACCESS_KEY="..."
# export PICOVOICE_API_KEY="..."

# Logging
export LOG_LEVEL="INFO"               # DEBUG, INFO, WARNING, ERROR
export LOG_FORMAT="human"             # human, json, detailed
```

### Python Configuration

```python
from realtime_mlx_stt import ModelConfig, VADConfig, WakeWordConfig

# Model configuration
model = ModelConfig(
    engine="mlx_whisper",        # or "openai"
    model="whisper-large-v3-turbo",
    language="en"                # or None for auto-detect
)

# VAD configuration
vad = VADConfig(
    enabled=True,
    sensitivity=0.6,             # 0.0-1.0
    min_speech_duration=0.25,    # seconds
    min_silence_duration=0.1     # seconds
)

# Wake word configuration
# Note: Requires PORCUPINE_ACCESS_KEY environment variable
wake_word = WakeWordConfig(
    words=["jarvis", "computer"],
    sensitivity=0.7,
    timeout=30                   # seconds
)

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

## Recent Updates

- **New Python API**: Added high-level `realtime_mlx_stt` package with STTClient, TranscriptionSession, and Transcriber
- **Interactive CLI**: New user-friendly CLI at `examples/cli.py` for exploring all features
- **Dual API Architecture**: Python API optimized for direct use, Server API for multi-client scenarios
- **Improved Examples**: Consolidated examples with clear documentation
- **Architecture Documentation**: Added comprehensive architecture documentation
- **OpenAI Integration**: Support for OpenAI's transcription API as alternative to local MLX

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the base Whisper large-v3-turbo model
- [MLX](https://github.com/ml-explore/mlx) for Apple Silicon optimization
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) for the original audio processing concepts
- [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) for wake word detection
- [Hugging Face](https://huggingface.co) for model distribution infrastructure