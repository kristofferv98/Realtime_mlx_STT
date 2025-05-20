# Realtime_mlx_STT Tests

This directory contains tests for the Realtime_mlx_STT project, organized according to the vertical slice architecture.

## Completed Features

The following features have been fully implemented and tested:

### 1. AudioCapture
- **Status**: ✅ COMPLETED
- **Test Directory**: `Features/AudioCapture`
- **Tests**: 26 passing tests (1 skipped)
- **Documentation**: 
  - `tests/Features/AudioCapture/README.md`
  - `src/Features/AudioCapture/README.md`
  - `src/Features/AudioCapture/COMPLETION.md`

The AudioCapture feature provides audio input functionality through microphone and file-based providers. All tests are designed to be hardware-independent through comprehensive mocking.

### 2. VoiceActivityDetection
- **Status**: ✅ COMPLETED
- **Test Directory**: `Features/VoiceActivityDetection`
- **Tests**: 3 passing tests
- **Documentation**:
  - `tests/Features/VoiceActivityDetection/README.md`
  - `src/Features/VoiceActivityDetection/README.md`
  - `src/Features/VoiceActivityDetection/COMPLETION.md`

The VoiceActivityDetection feature provides speech detection capabilities through multiple detector implementations (WebRTC, Silero, Combined).

## In-Progress Features

### 3. Transcription
- **Status**: 🔄 PLANNED
- **Description**: Will implement the speech-to-text conversion using MLX-optimized Whisper model

## Running Tests

### Feature-Specific Tests

Each feature has its own test runner script that can be used to run all tests for that feature:

```bash
# Run AudioCapture tests
python tests/Features/AudioCapture/run_all_tests.py

# Run VoiceActivityDetection tests
python tests/Features/VoiceActivityDetection/run_all_tests.py
```

### Individual Tests

You can also run individual test files:

```bash
# Example: Run a specific test file
python -m unittest tests/Features/AudioCapture/pyaudio_provider_test.py
```

## Test Design Principles

All tests follow these principles:

1. **Hardware Independence**: Tests use mocking to avoid hardware dependencies
2. **Thread Safety**: Tests properly manage threading to avoid race conditions
3. **Deterministic Results**: Tests produce consistent results across runs
4. **Comprehensive Coverage**: Tests cover all public methods and edge cases
5. **Fast Execution**: Tests are optimized for speed to enable rapid development

## Note on Dependencies

Some tests may require additional dependencies:

- VoiceActivityDetection tests require `webrtcvad` and optionally `torch` and `torchaudio`
- Tests will skip automatically with clear messages if dependencies are missing