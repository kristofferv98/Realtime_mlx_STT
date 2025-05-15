# Voice Activity Detection Feature Slice

## Overview
Voice Activity Detection provides a modular, event-driven system for detecting speech segments in audio streams. It enables applications to distinguish between speech and silence, improving efficiency by processing only relevant audio containing speech.

## Key Components

### Detectors
- `WebRtcVadDetector`: Fast, lightweight VAD using WebRTC algorithm
- `SileroVadDetector`: High-accuracy ML-based detection using Silero model
- `CombinedVadDetector`: Two-stage approach balancing speed and accuracy

### Commands
- `DetectVoiceActivityCommand`: Process an audio chunk to detect speech
- `ConfigureVadCommand`: Set parameters for VAD operation

### Events
- `SpeechDetectedEvent`: Published when speech begins in the audio
- `SilenceDetectedEvent`: Published when speech ends and silence begins

### Handler
- `VoiceActivityHandler`: Processes commands and manages detection state

## Usage Example

```python
# Register module
VadModule.register(command_dispatcher, event_bus)

# Configure VAD detection
VadModule.configure_vad(
    command_dispatcher,
    detector_type="combined",
    sensitivity=0.8,
    window_size=4,
    min_speech_duration=0.5
)

# Subscribe to speech events
VadModule.on_speech_detected(
    event_bus,
    lambda confidence, timestamp, speech_id: print(f"Speech detected: {confidence:.2f}")
)

VadModule.on_silence_detected(
    event_bus,
    lambda duration, start, end, speech_id: print(f"Speech ended: {duration:.2f}s")
)

# Process a specific audio chunk manually
is_speech = VadModule.detect_voice_activity(command_dispatcher, audio_chunk)
```

## Features
- Multiple detector implementations for different needs
- State machine for smooth detection transitions
- Configurable sensitivity and parameters
- Speech segment tracking with unique IDs
- Confidence scores for detections
- Buffer management for complete speech segments
- Event-based architecture for loose coupling
- Automatic audio processing via event subscription

## Dependencies
- WebRTC VAD (webrtcvad)
- Silero VAD model (torch)
- NumPy
- Core command/event infrastructure
- AudioCapture feature (for audio input)