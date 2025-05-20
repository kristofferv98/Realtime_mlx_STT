# Wake Word Detection Feature

This module provides wake word detection capabilities for the Realtime_mlx_STT project.

## Overview

The Wake Word Detection feature allows the system to remain in a low-resource "listening" mode until a specific keyword or phrase is detected, at which point it will activate full speech transcription. This is crucial for hands-free operation and always-on voice assistant applications.

## Components

- **Detectors**: Implementations of wake word detection algorithms
  - `PorcupineWakeWordDetector`: Integrates with Picovoice's Porcupine engine
  
- **Commands**: Actions that can be performed on the wake word detection system
  - `ConfigureWakeWordCommand`: Sets up wake word detection parameters
  - `StartWakeWordDetectionCommand`: Begins listening for wake words
  - `StopWakeWordDetectionCommand`: Stops listening for wake words
  - `DetectWakeWordCommand`: Processes an audio chunk for wake word detection
  
- **Events**: Notifications about wake word detection state
  - `WakeWordDetectedEvent`: Published when a wake word is detected
  - `WakeWordDetectionStartedEvent`: Published when detection begins
  - `WakeWordDetectionStoppedEvent`: Published when detection ends
  - `WakeWordTimeoutEvent`: Published when no speech follows a wake word
  
- **Handler**: Processes commands and manages the wake word detection state
  - `WakeWordCommandHandler`: Coordinates wake word detection and integration with other features
  
- **Models**: Data structures for wake word detection
  - `WakeWordConfig`: Configuration parameters for wake word detection

## Usage

### Basic Usage

```python
from src.Core.Events.event_bus import EventBus
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule

# Create event bus and command dispatcher
event_bus = EventBus()
command_dispatcher = CommandDispatcher()

# Register the wake word module
WakeWordModule.register(
    command_dispatcher=command_dispatcher,
    event_bus=event_bus,
    wake_words=["computer", "jarvis"]
)

# Start wake word detection
WakeWordModule.start_detection(command_dispatcher)

# Subscribe to wake word events
def on_wake_word(wake_word, confidence, timestamp):
    print(f"Wake word detected: {wake_word} (confidence: {confidence:.2f})")
    
WakeWordModule.on_wake_word_detected(event_bus, on_wake_word)
```

### Advanced Configuration

```python
from src.Features.WakeWordDetection.Models.WakeWordConfig import WakeWordConfig

# Configure with built-in wake words
config = WakeWordConfig(
    detector_type="porcupine",
    wake_words=["alexa", "porcupine"],
    sensitivities=[0.7, 0.6],
    access_key="YOUR_PORCUPINE_ACCESS_KEY",
    speech_timeout=3.0,
    buffer_duration=0.2
)

WakeWordModule.configure(command_dispatcher, config)
```

## Integration with Other Features

The Wake Word Detection feature integrates with:

1. **AudioCapture**: Receives audio data for wake word detection
2. **VoiceActivityDetection**: Used to detect speech after a wake word is detected
3. **Transcription**: Triggered to start transcribing speech after a wake word and subsequent speech is detected

## State Machine

The wake word detection process follows a state machine:

1. **Inactive**: Not listening for anything
2. **WakeWord**: Actively listening for wake words
3. **Listening**: Wake word detected, now listening for speech using VAD
4. **Recording**: Speech detected after wake word, now recording for transcription
5. **Processing**: Audio capture complete, now processing transcription

## Dependencies

- **Porcupine**: `pip install pvporcupine`
- Requires a Picovoice access key (available from [Picovoice Console](https://console.picovoice.ai/))

## Environment Variables

- `PORCUPINE_ACCESS_KEY`: API key for Porcupine (optional, can also be provided in config)