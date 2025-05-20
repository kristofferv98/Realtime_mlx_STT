# Refactored Main Branch Specification

## Overview

This document outlines the file structure and components to be included in the refactored main branch of the Realtime_mlx_STT project. The goal is to create a clean, well-organized codebase that follows the vertical slice architecture while preserving all implemented features.

## Project Architecture

The project follows a vertical slice architecture organized by features. Each feature contains its own commands, events, handlers, and models.

### Core Directory Structure

```
src/
├── Core/                       # Core interfaces and base classes
├── Features/                   # Feature-based vertical slices
│   ├── AudioCapture/           # Audio input handling (COMPLETED)
│   ├── VoiceActivityDetection/ # Speech detection (COMPLETED)
│   ├── Transcription/          # Audio-to-text processing (COMPLETED)
│   ├── WakeWordDetection/      # Future feature
│   └── RemoteProcessing/       # Future feature
├── Infrastructure/             # Cross-cutting concerns
└── Application/                # Public API facades
```

## Files to Include

### Core Components

All core components will be included as they provide the foundation for the entire system:

- `src/Core/Commands/command.py`
- `src/Core/Commands/command_dispatcher.py`
- `src/Core/Commands/__init__.py`
- `src/Core/Common/Interfaces/*.py`
- `src/Core/Events/event.py`
- `src/Core/Events/event_bus.py`
- `src/Core/Events/__init__.py`
- `src/Core/__init__.py`
- `src/Core/README.md`

### Completed Features

Based on the COMPLETION.md files, the following features are marked as complete and will be included:

1. **AudioCapture Feature**:
   - All files in `src/Features/AudioCapture/`
   - Associated tests in `tests/Features/AudioCapture/`

2. **VoiceActivityDetection Feature**:
   - All files in `src/Features/VoiceActivityDetection/`
   - Associated tests in `tests/Features/VoiceActivityDetection/`

3. **Transcription Feature**:
   - All files in `src/Features/Transcription/`
   - Associated tests in `tests/Features/Transcription/`
   - Note: We'll exclude the `OLD_BACKUP` directories as they contain superseded implementations

### Example Code

We'll include all examples to demonstrate usage patterns:

- `examples/check_audio_devices.py`
- `examples/continuous_transcription.py`
- `examples/transcribe_file.py`
- `examples/transcribe_with_vad.py`

### Tests

Include all test files except those in backup directories:

- `tests/run_tests.py`
- `tests/README.md`
- All files in `tests/Features/` except those in `OLD_BACKUP` directories

### Project Configuration

Include all project configuration files:

- `pyproject.toml`
- `setup.py`
- `CLAUDE.md`
- `README.md`
- `LICENSE`
- `.python-version` 

### Sample/Test Data

Include minimal test data necessary for tests:

- `mel_filters.npz` (required for MLX Whisper models)

## Files to Exclude

- `RealtimeSTT/PRE_REFACTOR/` (deprecated code)
- Any `OLD_BACKUP` directories
- Temporary audio files (unless used in tests)
- `venv/` and `env_mlx_test/` directories
- `__pycache__` directories and `.pyc` files

## Integration Testing

The integration between the three completed features (AudioCapture, VoiceActivityDetection, and Transcription) should be thoroughly tested using:

- `tests/Features/*/run_all_tests.py` for individual feature tests
- End-to-end tests that verify the complete pipeline

## Application Facade

The `src/Application/Facade` directory currently has minimal implementation. As part of the refactoring, a proper facade should be developed to provide a clean public API for the library, similar to the example usage shown in the README.

## Future Considerations

The refactored main branch should provide hooks and extension points for future features:

1. **WakeWordDetection**: Interfaces are defined but implementation is pending
2. **RemoteProcessing**: Skeleton directory exists for future cloud-based processing

## Backward Compatibility

The refactored code should maintain backward compatibility with existing APIs where possible, especially for features that have been integrated by users.

## Python and Dependency Management

The project uses:
- Python 3.11+ (specified in `.python-version`)
- `uv sync pyproject.toml` for dependency management (NOT pip)
- MLX-specific optimizations for Apple Silicon