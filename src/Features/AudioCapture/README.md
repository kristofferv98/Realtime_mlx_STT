# Audio Capture Feature Slice

## Overview
Audio Capture provides a modular, event-driven system for capturing audio from various sources and making it available to other components through a standardized interface.

## Key Components

### Commands
- `StartRecordingCommand`: Initiates audio capture with configurable parameters
- `StopRecordingCommand`: Stops active audio capture
- `SelectDeviceCommand`: Changes the active audio input device
- `ListDevicesCommand`: Queries available audio input devices

### Events
- `AudioChunkCapturedEvent`: Published when new audio data is available
- `RecordingStateChangedEvent`: Published on recording state changes (started, stopped, error)

### Models
- `AudioChunk`: Encapsulates audio data with metadata (format, sample rate, etc.)
- `DeviceInfo`: Represents audio device capabilities and properties

### Providers
- `PyAudioInputProvider`: Captures audio from microphone using PyAudio
- `FileAudioProvider`: Streams audio from files for testing/playback

### Handler
- `AudioCommandHandler`: Processes audio-related commands and publishes events

## Usage Example

```python
# Register module
AudioCaptureModule.register(command_dispatcher, event_bus)

# List audio devices
devices = AudioCaptureModule.list_devices(command_dispatcher)

# Start recording
command_dispatcher.dispatch(StartRecordingCommand(
    device_id=0,  # Use default device
    sample_rate=16000,
    chunk_size=512
))

# Subscribe to audio events
event_bus.subscribe(AudioChunkCapturedEvent, process_audio_data)
event_bus.subscribe(RecordingStateChangedEvent, handle_state_change)

# Stop recording
command_dispatcher.dispatch(StopRecordingCommand())
```

## Features
- Thread-safe resource management
- Automatic sample rate conversion
- Multiple audio source support
- Event-based data flow
- Configurable parameters
- Device enumeration

## Dependencies
- PyAudio
- NumPy
- SciPy (for resampling)
- Core command/event infrastructure