# VAD Hardcoded Values Analysis

## Summary of Findings

After searching the codebase for hardcoded VAD parameters, I found several areas where default values are set. While these can be overridden through configuration, there are some concerns about initialization.

## Hardcoded Default Values

### 1. ConfigureVadCommand defaults (src/Features/VoiceActivityDetection/Commands/ConfigureVadCommand.py)
```python
sensitivity: float = 0.5
window_size: int = 5
min_speech_duration: float = 0.25
speech_pad_ms: int = 100
pre_speech_buffer_size: int = 64  # ~2 seconds at 32ms/chunk
```

### 2. CombinedVadDetector constructor defaults (src/Features/VoiceActivityDetection/Detectors/CombinedVadDetector.py)
```python
webrtc_aggressiveness: int = 2
silero_threshold: float = 0.6
frame_duration_ms: int = 30
speech_confirmation_frames: int = 2
silence_confirmation_frames: int = 30
speech_buffer_size: int = 100
webrtc_threshold: float = 0.6
```

### 3. SileroVadDetector defaults (src/Features/VoiceActivityDetection/Detectors/SileroVadDetector.py)
```python
threshold: float = 0.5
```

### 4. WebRtcVadDetector defaults (src/Features/VoiceActivityDetection/Detectors/WebRtcVadDetector.py)
```python
aggressiveness: int = 3
speech_threshold: float = 0.6
```

### 5. VadModule registration defaults (src/Features/VoiceActivityDetection/VadModule.py)
```python
default_detector: str = "combined"
default_sensitivity: float = 0.7
```

## Critical Issue Found

**VoiceActivityHandler initialization** (src/Features/VoiceActivityDetection/Handlers/VoiceActivityHandler.py):
```python
# Initialize detector registry
self.detectors: Dict[str, IVoiceActivityDetector] = {
    'webrtc': WebRtcVadDetector(),
    'silero': SileroVadDetector(),
    'combined': CombinedVadDetector()
}
```

The detectors are initialized with their default constructor values and cannot be changed until a ConfigureVadCommand is sent. This means:
1. The initial detector instances use hardcoded defaults
2. Configuration from the Application layer happens after initialization
3. Some parameters might not be configurable after initialization

## Recommendations

1. **Lazy Initialization**: Initialize detectors only when first used, not in the constructor
2. **Factory Pattern**: Create detectors with configuration parameters
3. **Ensure All Parameters Are Configurable**: Verify that all constructor parameters can be updated via configure() method

## Current Configuration Flow

1. ProfileManager defines profiles with VAD settings
2. SystemController sends ConfigureVadCommand with settings
3. VoiceActivityHandler receives command and calls detector.configure()
4. Detector updates its internal parameters

## Configurable Parameters from Application Layer

The following can be set through custom_config in the Application layer:
- `sensitivity` (maps to different thresholds for each detector)
- `min_speech_duration`
- `window_size`
- `speech_pad_ms`
- `pre_speech_buffer_size`
- Individual thresholds via `parameters` dict:
  - `webrtc_aggressiveness`
  - `silero_threshold`
  - `webrtc_threshold`

## Non-configurable Parameters

These are currently hardcoded and cannot be changed from the Application layer:
- `frame_duration_ms` (30ms)
- `speech_confirmation_frames` (2)
- `silence_confirmation_frames` (30)
- `speech_buffer_size` (100)

These should be added to the ConfigureVadCommand parameters dict if needed.