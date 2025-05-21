# STT Library Integration Specification

## Overview

This specification outlines a plan to transform the current codebase into a modular, importable library that allows developers to easily integrate speech-to-text capabilities into their applications. The library will follow a builder pattern approach with a simple, fluent API for configuration.

## Goals

1. **Simplicity**: Make integration easy with minimal code
2. **Flexibility**: Allow choosing specific components to include
3. **Configurability**: Provide sensible defaults but allow customization
4. **Extensibility**: Enable developers to plug in custom handlers for events
5. **Minimal Dependencies**: Allow importing only what's needed

## Library Structure

```
realtime_stt/
├── __init__.py
├── core/
│   ├── event_bus.py
│   └── command_dispatcher.py
├── features/
│   ├── audio_capture/
│   ├── vad/
│   ├── wake_word/
│   └── transcription/
│       ├── engines/
│       │   ├── mlx_whisper.py
│       │   └── openai.py
│       └── ...
└── utils/
    ├── clipboard.py
    └── ...
```

## API Design

The library will use a builder pattern for configuration, making it intuitive and chainable:

```python
from realtime_stt import SpeechProcessor

# Basic setup with all defaults
processor = SpeechProcessor.builder().build()

# Configured setup
processor = (SpeechProcessor.builder()
    .with_wake_word(words=["jarvis"], sensitivity=0.5)
    .with_vad(detector_type="combined", sensitivity=0.7)
    .with_transcription(
        engine="mlx_whisper",  # or "openai"
        model="whisper-large-v3-turbo",  # or "gpt-4o-transcribe" for OpenAI
        language="en"
    )
    .with_clipboard_copy(auto_paste=True)
    .build())

# Start processing
processor.start()

# Register event handlers
processor.on_transcription_complete(your_handler_function)
processor.on_wake_word_detected(your_wake_word_handler)
```

## Component Configuration Options

### 1. Audio Capture

```python
.with_audio_capture(
    device_id=None,  # None = default device
    sample_rate=16000,
    chunk_size=512,
    channels=1
)
```

### 2. Voice Activity Detection

```python
.with_vad(
    detector_type="combined",  # "webrtc", "silero", or "combined"
    sensitivity=0.6,  # 0.0 to 1.0
    window_size=5,
    min_speech_duration=0.25,
    speech_buffer_size=30,  # seconds of speech to buffer
    pre_speech_buffer_duration=1.0  # seconds of audio to keep before detected speech
)
```

### 3. Wake Word Detection

```python
.with_wake_word(
    enabled=True,  # Set to False to disable wake word (always-on mode)
    words=["jarvis"],  # List of wake words to detect
    sensitivity=0.5,  # 0.0 to 1.0
    access_key="YOUR_PORCUPINE_KEY",  # or use environment variable
    timeout=5.0  # seconds to wait for speech after wake word
)
```

### 4. Transcription

```python
.with_transcription(
    engine="mlx_whisper",  # "mlx_whisper" or "openai"
    model="whisper-large-v3-turbo",  # engine-specific model name
    language=None,  # None for auto-detection
    beam_size=1,
    streaming=True,
    
    # MLX-specific options
    quick_mode=True,  # faster but slightly less accurate
    
    # OpenAI-specific options
    api_key="YOUR_OPENAI_KEY"  # or use environment variable
)
```

### 5. Post-Processing

```python
.with_clipboard_copy(
    enabled=True,
    auto_paste=False,
    copy_delay=0.1,
    paste_delay=0.2
)
```

## Event Handlers

The library will expose the following events:

```python
# Recording events
processor.on_recording_started(callback)
processor.on_recording_stopped(callback)

# VAD events
processor.on_speech_detected(callback)
processor.on_silence_detected(callback)

# Wake word events
processor.on_wake_word_detected(callback)
processor.on_wake_word_timeout(callback)

# Transcription events
processor.on_transcription_started(callback)
processor.on_transcription_updated(callback)  # Intermediate results
processor.on_transcription_complete(callback)  # Final results
processor.on_transcription_error(callback)
```

## Configuration Presets

The library will include common presets for quick setup:

```python
# Always-on transcription (no wake word)
processor = SpeechProcessor.create_continuous()

# Wake word with MLX transcription
processor = SpeechProcessor.create_wake_word_mlx()

# Wake word with OpenAI transcription
processor = SpeechProcessor.create_wake_word_openai(api_key="YOUR_KEY")

# Wake word with clipboard integration
processor = SpeechProcessor.create_wake_word_clipboard(engine="mlx_whisper")
```

## Command Execution Framework

We'll include a simple framework for executing commands based on transcribed text:

```python
from realtime_stt import SpeechProcessor, CommandRegistry

# Register commands
commands = CommandRegistry()
commands.add("open {app}", open_application)
commands.add("search for {query}", web_search)
commands.add("what time is it", tell_time)

# Create processor with command execution
processor = (SpeechProcessor.builder()
    .with_wake_word(words=["computer"])
    .with_transcription(engine="mlx_whisper")
    .with_command_execution(commands)
    .build())

processor.start()
```

## Usage Examples

### Basic Wake Word + Transcription

```python
from realtime_stt import SpeechProcessor

def handle_transcription(text, confidence):
    print(f"Transcribed: {text} (confidence: {confidence:.2f})")

# Create a speech processor with wake word and MLX transcription
processor = (SpeechProcessor.builder()
    .with_wake_word(words=["computer"])
    .with_transcription(engine="mlx_whisper")
    .build())

# Register handler for transcription results
processor.on_transcription_complete(handle_transcription)

# Start processing
processor.start()

# Keep the program running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    processor.stop()
```

### Integration with Another Application

```python
from realtime_stt import SpeechProcessor
from your_app import SmartHomeController

controller = SmartHomeController()

def execute_command(text, confidence):
    # Forward transcribed commands to your application
    controller.process_voice_command(text)

# Create processor with OpenAI for high accuracy
processor = (SpeechProcessor.builder()
    .with_wake_word(words=["home"])
    .with_transcription(
        engine="openai", 
        model="gpt-4o-transcribe"
    )
    .build())

processor.on_transcription_complete(execute_command)
processor.start()

# Run your application's main loop
controller.run()
```

## Implementation Roadmap

1. **Phase 1: Core Architecture**
   - Create builder pattern implementation
   - Abstract existing modules into library structure
   - Implement basic event handling system

2. **Phase 2: Feature Modules**
   - Refactor audio capture into configurable module
   - Refactor VAD into configurable module
   - Refactor wake word detection into configurable module
   - Refactor transcription engines into configurable modules

3. **Phase 3: Integration & Utilities**
   - Implement clipboard and post-processing utilities
   - Create preset configurations
   - Build command execution framework

4. **Phase 4: Documentation & Examples**
   - Write comprehensive API documentation
   - Create example applications
   - Build tutorial guides

## Conclusion

This library design provides a flexible, easy-to-use interface for integrating speech recognition capabilities into Python applications. By following a builder pattern with sensible defaults, developers can quickly add powerful speech-to-text features with minimal configuration, while still having full access to customize behavior when needed.

The event-based architecture ensures that applications can easily respond to various speech processing events, making it ideal for building voice-controlled interfaces and assistants.