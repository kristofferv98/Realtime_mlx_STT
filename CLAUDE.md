# Realtime_mlx_STT

## Overview
High-performance speech-to-text transcription library optimized exclusively for Apple Silicon, leveraging MLX framework for real-time on-device transcription with low latency.

## Architecture
- **Design Pattern**: Vertical slice architecture with command-driven communication
- **Layer Structure**: Features, Core, Infrastructure, and Application layers
- **Feature Organization**: Each feature contains `Commands/`, `Events/`, `Handlers/`, and `Models/` subdirectories
- **Communication**: Features communicate via `CommandDispatcher` and `EventBus`, never directly
- **Key Decisions**: Direct MLX integration preferred over process isolation for performance

## Tech Stack
- **Runtime**: Python 3.8+ (3.11+ recommended for Apple Silicon)
- **ML Framework**: MLX with whisper-large-v3-turbo model
- **Audio**: PyAudio at 16kHz, 16-bit, mono format
- **VAD**: Combined Silero + WebRTC with lazy initialization
- **Wake Word**: Porcupine (requires PORCUPINE_ACCESS_KEY)
- **Server**: FastAPI + WebSocket for real-time communication
- **Testing**: pytest with mock-based testing

## Coding Standards
- **Architecture**: ALWAYS follow vertical slice architecture
- **Type Hints**: Required for all new code
- **Imports**: Use populated `__init__.py` files (e.g., `from src.Features import TranscriptionModule`)
- **Logging**: Use `LoggingModule.get_logger(__name__)` exclusively
- **Threading**: Non-daemon threads with proper cleanup
- **Audio Format**: Standardized at 16kHz, 16-bit, mono

## Development Workflow
- **Installation**: `pip install -e .` or `uv pip install -e .` (pyproject.toml is single source of truth)
- **Testing**: Run `python tests/run_tests.py` for all tests
- **Commit Format**: Use conventional commits with descriptive messages
- **Branch Naming**: Current branch is `application_client`

## Important Notes
- **Platform Requirement**: Apple Silicon (M1/M2/M3) - Intel Macs not supported
- **Completed Features**: AudioCapture, VoiceActivityDetection, WakeWordDetection, Transcription
- **Pending Work**: WakeWordDetection tests need implementation
- **Environment Limitation**: Claude cannot execute example scripts directly - user must run them
- **Performance**: MLX optimizations required for all transcription code

## Project Structure
```
src/
├── Features/           # Feature modules with vertical slice architecture
├── Core/              # Interfaces and communication infrastructure
├── Infrastructure/    # Cross-cutting concerns (Logging, ProgressBar)
└── Application/       # Server implementation
```

## Configuration
- **Profiles**: continuous, vad-triggered, wake-word
- **Environment Variables**: 
  - `OPENAI_API_KEY` (for cloud transcription)
  - `PORCUPINE_ACCESS_KEY` (for wake word)
  - `LOG_FORMAT`, `LOG_LEVEL` (logging control)

## Author
- Kristoffer Vatnehol (kristoffer.vatnehol@gmail.com)
- GitHub: github.com/kristofferv98/