# Realtime MLX STT - Usage Guide

This guide addresses common usage patterns and potential issues when using the realtime-mlx-stt package.

## Installation

```bash
pip install realtime-mlx-stt
```

## Common Issues and Solutions

### 1. Import Errors

If you encounter `ModuleNotFoundError: No module named 'src'`, please update to version 0.1.1 or later:
```bash
pip install realtime-mlx-stt>=0.1.1
```

### 2. Model Download Issues

The MLX Whisper models are automatically downloaded on first use. If you encounter download issues:

- **First run may take time**: The whisper-large-v3-turbo model is ~3.8GB
- **Network issues**: Ensure you have a stable internet connection
- **Disk space**: Ensure you have sufficient disk space in `~/.cache/huggingface/`

### 3. STTClient Initialization

**Correct usage:**
```python
from realtime_mlx_stt import STTClient

# All valid parameters:
client = STTClient(
    openai_api_key="sk-...",      # Optional: for OpenAI engine
    porcupine_api_key="...",      # Optional: for wake word detection
    default_engine="mlx_whisper", # "mlx_whisper" or "openai"
    default_model=None,           # Model name or None for default
    default_language="en",        # Language code or None for auto
    device_index=None,            # Audio device index or None for default
    verbose=False                 # Enable debug logging
)
```

**Common mistakes:**
```python
# ❌ WRONG - These parameters don't exist:
client = STTClient(
    wake_word_sensitivity=0.7,    # Not a parameter
    vad_sensitivity=0.6,          # Not a parameter
    vad_min_speech_duration=0.25  # Not a parameter
)
```

### 4. Wake Word Detection

Wake word detection requires the `PORCUPINE_ACCESS_KEY` environment variable:

```bash
# Get your free key at: https://picovoice.ai/
export PORCUPINE_ACCESS_KEY='your-key-here'
```

**Correct wake word usage (callback-based):**
```python
def on_wake_word(word, confidence):
    print(f"Wake word '{word}' detected!")

def on_transcription(result):
    print(f"You said: {result.text}")

# Start wake word detection (non-blocking)
client.start_wake_word(
    wake_word="jarvis",           # Or ["jarvis", "computer"] for multiple
    on_wake=on_wake_word,         # Required: wake word callback
    on_transcription=on_transcription  # Required: transcription callback
)

# Keep the program running
import time
time.sleep(60)  # Listen for 60 seconds

# Stop when done
client.stop()
```

**Common mistakes:**
```python
# ❌ WRONG - start_wake_word is not blocking and doesn't return boolean
if client.start_wake_word("jarvis"):
    # This won't work
    pass

# ❌ WRONG - No wait_for_wake_word method exists
client.wait_for_wake_word("jarvis")

# ❌ WRONG - No timeout parameter
client.start_wake_word("jarvis", timeout=30)
```

### 5. Simple Transcription Patterns

#### Fixed Duration Transcription
```python
# Transcribe for 10 seconds
for result in client.transcribe(duration=10):
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence}")
```

#### Continuous Streaming
```python
# Stream until you say "stop"
with client.stream() as stream:
    for result in stream:
        print(result.text)
        if "stop" in result.text.lower():
            break
```

#### Wake Word Triggered Transcription
```python
# Complete example with wake word
import time
from realtime_mlx_stt import STTClient

client = STTClient()

def handle_wake_word(word, confidence):
    print(f"\n✨ '{word}' detected! Listening...")

def handle_command(result):
    print(f"📝 Command: {result.text}\n")
    print("Say 'jarvis' again...")

# Start listening
client.start_wake_word(
    wake_word="jarvis",
    on_wake=handle_wake_word,
    on_transcription=handle_command
)

try:
    # Run for 5 minutes
    time.sleep(300)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    client.stop()
```

## Engine Selection

### MLX Whisper (Default)
- Optimized for Apple Silicon
- Runs locally, no API key needed
- Models downloaded automatically on first use

### OpenAI Whisper API
```python
client = STTClient(
    openai_api_key="sk-...",
    default_engine="openai"
)
```

## Language Settings

```python
# Specify language
client = STTClient(default_language="es")  # Spanish

# Or change dynamically
client.set_language("fr")  # French

# Auto-detect language
client.set_language(None)
```

## Audio Device Selection

```python
# List available devices
devices = client.list_devices()
for device in devices:
    print(f"[{device.index}] {device.name}")

# Select specific device
client.set_device(device_index=2)
```

## Error Handling

```python
try:
    client = STTClient()
    for result in client.transcribe(duration=5):
        print(result.text)
except Exception as e:
    print(f"Error: {e}")
    # Common errors:
    # - No microphone access
    # - Missing API keys
    # - Network issues (for OpenAI)
    # - Model download failures
```

## Best Practices

1. **Always use callbacks for wake word detection** - It's event-driven, not polling
2. **Set environment variables before importing** - Especially for API keys
3. **Handle exceptions** - Microphone access and network issues are common
4. **Use context managers for streaming** - Ensures proper cleanup
5. **Check device permissions** - macOS requires microphone access approval

## Need Help?

- Check the [examples folder](examples/example_scripts/) for complete working examples
- Review the [API documentation](realtime_mlx_stt/README.md)
- Report issues at: https://github.com/kristofferv98/Realtime_mlx_STT/issues