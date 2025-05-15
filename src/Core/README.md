# Core Infrastructure

## Overview

The Core infrastructure provides the foundational components necessary for implementing a vertical slice architecture in the Realtime_mlx_STT system. It enables loosely coupled, event-driven communication between feature modules and defines key interfaces that standardize feature implementations.

## Key Components

### Event System

The event system enables asynchronous, pub-sub style communication between features:

- **Event Base Class**: Foundation for all events with built-in ID, timestamp, and naming
- **IEventBus Interface**: Contract for publishing and subscribing to events
- **EventBus Implementation**: Thread-safe event dispatcher with inheritance support

```python
# Publishing an event
event_bus.publish(AudioChunkCapturedEvent(audio_data, sample_rate))

# Subscribing to events
event_bus.subscribe(SpeechDetectedEvent, handle_speech_detected)
```

### Command System

The command system implements the mediator pattern for request-response operations:

- **Command Base Class**: Foundation for all command objects
- **ICommandHandler Interface**: Contract for components that process commands
- **CommandDispatcher**: Central mediator that routes commands to handlers

```python
# Dispatching a command
result = command_dispatcher.dispatch(StartRecordingCommand(device_id=1))

# Implementing a handler
class AudioCommandHandler(ICommandHandler):
    def handle(self, command: Command) -> Any:
        if isinstance(command, StartRecordingCommand):
            # Start recording logic
            return True
```

### Core Interfaces

Four primary feature interfaces define the key capabilities of the system:

- **IAudioProvider**: Defines audio capture functionality
- **IVoiceActivityDetector**: Defines speech detection capabilities
- **IWakeWordDetector**: Defines wake word/hotword detection
- **ITranscriptionEngine**: Defines speech-to-text conversion

## Architecture Benefits

This infrastructure enables several key benefits:

1. **Low Coupling**: Features interact through events and commands without direct dependencies
2. **High Cohesion**: Related functionality stays together in feature modules
3. **Testability**: Interfaces facilitate unit testing with mock implementations
4. **Extensibility**: New features can be added without modifying existing code
5. **Thread Safety**: Concurrent operations are handled safely through proper locking

## Usage Example

```python
# Setting up the core infrastructure
event_bus = EventBus()
command_dispatcher = CommandDispatcher()

# Registering command handlers
command_dispatcher.register_handler(StartRecordingCommand, audio_command_handler)

# Subscribing to events
event_bus.subscribe(AudioChunkCapturedEvent, process_audio_data)

# Dispatching a command
command_dispatcher.dispatch(StartRecordingCommand(device_id=0))

# Events will be published by handlers and processed by subscribers
```

## Integration with Features

Feature modules build on this core infrastructure by:

1. Defining feature-specific commands and events
2. Implementing the core interfaces
3. Creating command handlers that process requests
4. Publishing events to notify other features of changes

This approach creates a modular system where features can evolve independently while maintaining a coherent overall architecture.