# Voice Activity Detection Feature Slice

## Overview
Voice Activity Detection provides a modular, event-driven system for detecting speech segments in audio streams. It enables applications to distinguish between speech and silence, improving efficiency by processing only relevant audio containing speech.

The module now features **lazy initialization** for optimal resource usage - detectors are only created when actually needed, not when the module is registered. All VAD parameters are **fully configurable** at runtime, including individual thresholds for each detector type.

## Key Components

### Detectors
- `WebRtcVadDetector`: Fast, lightweight VAD using WebRTC algorithm
  - Configurable: aggressiveness (0-3), frame duration (10/20/30ms)
- `SileroVadDetector`: High-accuracy ML-based detection using Silero model  
  - Configurable: threshold (0.0-1.0), min speech duration, min silence duration
- `CombinedVadDetector`: Two-stage approach balancing speed and accuracy
  - Uses WebRTC for initial detection, Silero for confirmation
  - Inherits configuration from both detectors

### Commands
- `DetectVoiceActivityCommand`: Process an audio chunk to detect speech
- `ConfigureVadCommand`: Set parameters for VAD operation
- `EnableVadProcessingCommand`: Enable automatic VAD processing
- `DisableVadProcessingCommand`: Disable automatic VAD processing
- `ClearVadPreSpeechBufferCommand`: Clear the pre-speech audio buffer

### Events
- `SpeechDetectedEvent`: Published when speech begins in the audio
- `SilenceDetectedEvent`: Published when speech ends and silence begins

### Handler
- `VoiceActivityHandler`: Processes commands and manages detection state
  - Features lazy initialization - detectors created on first use
  - Maintains detector registry with independent configurations
  - Handles state transitions and buffer management

## Usage Example

```python
# Register module (detectors not created yet - lazy initialization)
VadModule.register(command_dispatcher, event_bus)

# Configure VAD detection with full parameter control
VadModule.configure_vad(
    command_dispatcher,
    detector_type="combined",
    sensitivity=0.8,
    window_size=4,
    min_speech_duration=0.5,
    # Individual detector thresholds
    webrtc_aggressiveness=2,
    silero_threshold=0.5,
    # Frame processing settings
    frame_duration_ms=30,
    speech_confirmation_frames=3,
    silence_confirmation_frames=10
)

# Configure individual detectors separately if needed
VadModule.configure_vad(
    command_dispatcher,
    detector_type="webrtc",
    webrtc_aggressiveness=3,  # Most aggressive
    frame_duration_ms=10      # Shortest frame for low latency
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

# Enable automatic VAD processing
VadModule.enable_vad_processing(command_dispatcher)

# Process a specific audio chunk manually
is_speech = VadModule.detect_voice_activity(command_dispatcher, audio_chunk)
```

## Features
- **Lazy Initialization**: Detectors created only when needed, not at registration
- **Full Configurability**: All parameters exposed and configurable at runtime
- Multiple detector implementations for different needs
- Individual threshold control for each detector type
- State machine for smooth detection transitions
- Configurable frame duration and confirmation windows
- Speech segment tracking with unique IDs
- Confidence scores for detections
- Buffer management for complete speech segments
- Event-based architecture for loose coupling
- Automatic audio processing via event subscription
- Resource-efficient design with on-demand loading

## Configuration Parameters

### Common Parameters
- `detector_type`: "webrtc", "silero", or "combined"
- `sensitivity`: Overall sensitivity (0.0-1.0)
- `window_size`: Number of frames for decision smoothing
- `min_speech_duration`: Minimum speech length to trigger detection
- `frame_duration_ms`: Audio frame size (10, 20, or 30ms)
- `speech_confirmation_frames`: Frames needed to confirm speech start
- `silence_confirmation_frames`: Frames needed to confirm speech end

### WebRTC-specific
- `webrtc_aggressiveness`: 0-3 (higher = more aggressive filtering)

### Silero-specific
- `silero_threshold`: 0.0-1.0 (detection threshold)
- `min_silence_duration_ms`: Minimum silence to end speech segment

## Dependencies
- WebRTC VAD (webrtcvad)
- Silero VAD model (torch) - loaded only when Silero detector is used
- NumPy
- Core command/event infrastructure
- AudioCapture feature (for audio input)