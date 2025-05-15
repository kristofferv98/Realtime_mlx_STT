# Voice Activity Detection Feature Specification

## Overview

The Voice Activity Detection (VAD) feature is responsible for detecting speech segments in audio streams. It provides a modular, event-driven system to analyze audio data and signal when speech starts and ends. This enables the application to process only relevant audio containing speech, improving efficiency and user experience.

## Key Components

### Commands

- **DetectVoiceActivityCommand**: Process an audio chunk to detect speech
  - Parameters: audio_chunk (AudioChunk), sensitivity (float, optional)
  - Returns: Boolean indicating speech presence

- **ConfigureVadCommand**: Set parameters for VAD operation
  - Parameters: detector_type (string), sensitivity (float), window_size (int), min_speech_duration (float)
  - Returns: Success status

### Events

- **SpeechDetectedEvent**: Published when speech is detected in the audio
  - Properties: timestamp, confidence, audio_reference

- **SilenceDetectedEvent**: Published when speech ends and silence begins
  - Properties: timestamp, speech_duration, audio_reference

### Detectors

- **WebRtcVadDetector**: Fast, lightweight VAD implementation using WebRTC VAD
  - Good for initial, low-latency detection
  - Configurable aggressiveness levels (0-3)

- **SileroVadDetector**: Higher accuracy VAD using the Silero model
  - Better accuracy but higher computational cost
  - ML-based approach for more nuanced speech detection

- **CombinedVadDetector**: Two-stage approach combining both detectors
  - Uses WebRTC for fast initial detection
  - Confirms with Silero for higher accuracy

### Handler

- **VoiceActivityHandler**: Processes VAD commands and publishes events
  - Manages detector instances
  - Processes audio chunks
  - Maintains speech detection state (speech vs. silence)

### Module

- **VadModule**: Registration point for VAD feature
  - Wires up command handlers
  - Subscribes to needed events (AudioChunkCapturedEvent)
  - Provides factory methods for commonly used operations

## Workflow

1. System captures audio via AudioCapture feature
2. AudioChunkCapturedEvent triggers VAD processing
3. VoiceActivityHandler processes audio through configured detectors
4. When speech is detected, SpeechDetectedEvent is published
5. Other components (like Transcription) respond to speech events
6. When speech ends, SilenceDetectedEvent is published
7. Components can reconfigure VAD sensitivity as needed

## Integration Points

- **Input**: Consumes AudioChunkCapturedEvent from AudioCapture feature
- **Output**: Produces SpeechDetectedEvent and SilenceDetectedEvent
- **Configuration**: Accepts settings via ConfigureVadCommand

## Implementation Details

### Detector Implementations

- WebRTCVadDetector:
  - Wraps the webrtcvad Python library
  - Maintains internal frame buffer for context
  - Configurable sensitivity/aggressiveness

- SileroVadDetector:
  - Uses pretrained Silero VAD model from torch hub
  - Handles batched processing for efficiency
  - Provides confidence scores for detections

- CombinedVadDetector:
  - Uses WebRTC for initial fast detection
  - Confirms positive detections with Silero
  - Maintains state machine for detection stability

### State Management

1. **Silent State**: No speech detected, monitoring for speech onset
2. **PotentialSpeech State**: Initial speech detected, gathering confirmation
3. **Speech State**: Confirmed speech, monitoring for silence
4. **PotentialSilence State**: Initial silence detected, confirming end of speech

### Performance Considerations

- WebRTC VAD is lightweight and suitable for continuous processing
- Silero VAD is more resource-intensive but more accurate
- Combined approach balances performance and accuracy
- MLX optimization may be applied to Silero model for Apple Silicon

## Usage Example

```python
# Register VAD module
VadModule.register(command_dispatcher, event_bus)

# Configure VAD detection
command_dispatcher.dispatch(ConfigureVadCommand(
    detector_type="combined",
    sensitivity=0.8,
    window_size=4,
    min_speech_duration=0.5
))

# Subscribe to speech events
event_bus.subscribe(SpeechDetectedEvent, start_transcription)
event_bus.subscribe(SilenceDetectedEvent, finalize_transcription)

# AudioChunkCapturedEvent from AudioCapture feature will automatically
# trigger voice activity detection via event subscription
```

## Dependencies

- webrtcvad
- torch (for Silero VAD model)
- numpy
- Core command/event infrastructure
- AudioCapture feature (for audio input)