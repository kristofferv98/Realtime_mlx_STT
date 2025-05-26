# Realtime MLX STT - Python API

High-level Python API for speech-to-text transcription, optimized for developer experience.

## Installation

```bash
pip install -e .

# For OpenAI support
pip install -e ".[openai]"
```

## Quick Start

```python
from realtime_mlx_stt import STTClient

# Simple usage
client = STTClient()
for result in client.transcribe(duration=10):
    print(result.text)
```

## API Levels

### 1. STTClient (`client.py`) - Modern API
```python
client = STTClient(openai_api_key="sk-...")

# Fixed duration
for result in client.transcribe(duration=10):
    print(result.text)

# Streaming
with client.stream() as stream:
    for result in stream:
        if "stop" in result.text.lower():
            break
```

### 2. TranscriptionSession (`session.py`) - Session-based
```python
from realtime_mlx_stt import TranscriptionSession, ModelConfig

session = TranscriptionSession(
    model=ModelConfig(engine="mlx_whisper"),
    on_transcription=lambda r: print(r.text)
)

with session:
    time.sleep(30)
```

### 3. Transcriber (`transcriber.py`) - Simple API
```python
from realtime_mlx_stt import create_transcriber

transcriber = create_transcriber()
text = transcriber.transcribe_from_mic(duration=5)
```

## Module Structure

```
realtime_mlx_stt/
├── __init__.py         # Package exports
├── client.py           # Modern client API
├── config.py           # Configuration classes
├── session.py          # Session-based API
├── transcriber.py      # Simple API
├── types.py            # Type definitions
├── utils.py            # Helper functions
└── wake_word.py        # Wake word utilities
```

## Configuration

```python
from realtime_mlx_stt import ModelConfig, VADConfig, WakeWordConfig

# Type-safe configuration
model = ModelConfig(
    engine="mlx_whisper",    # or "openai"
    model="whisper-large-v3-turbo",
    language="en"            # or None for auto
)

vad = VADConfig(
    sensitivity=0.6,         # 0.0-1.0
    min_speech_duration=0.25
)

wake_word = WakeWordConfig(
    words=["jarvis"],
    sensitivity=0.7
)
```

## Architecture

This API layer is a thin wrapper around the Features layer:
- Uses CommandDispatcher to send commands
- Subscribes to EventBus for results
- Provides Pythonic interfaces (callbacks, context managers)
- Handles threading and state management

## Design Principles

1. **Developer Experience** - Simple things simple, complex things possible
2. **Type Safety** - Full type hints with validation
3. **Multiple Styles** - Client, session, or simple API
4. **Zero Configuration** - Works with defaults
5. **Progressive Disclosure** - Advanced features available when needed