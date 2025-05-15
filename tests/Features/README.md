# Feature Tests

This directory contains test cases for the vertical slice feature modules.

## Structure

- `AudioCapture/`: Tests for the AudioCapture feature
- `VoiceActivityDetection/`: Tests for the VoiceActivityDetection feature

## Test Philosophy

Tests in this directory follow these principles:

1. **Feature Independence**: Each feature's tests are contained within their own directory
2. **Comprehensive Coverage**: Tests cover all aspects of the feature (commands, events, handlers, etc.)
3. **Isolation**: Tests should focus on the feature's behavior, mocking dependencies where appropriate
4. **Real-world Usage**: Tests should represent actual usage patterns

## Running Tests

Tests can be run using pytest:

```bash
# Run all feature tests
pytest tests/Features

# Run tests for a specific feature
pytest tests/Features/AudioCapture
pytest tests/Features/VoiceActivityDetection
```