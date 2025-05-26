# Realtime_mlx_STT

## Overview
High-performance speech-to-text transcription library optimized exclusively for Apple Silicon, leveraging MLX framework for real-time on-device transcription with low latency.

## Architecture
- **Design Pattern**: Vertical slice architecture with command-driven communication
- **Dual API Design**: Python API (`realtime_mlx_stt/`) and Server API (`src/Application/Server/`)
- **Layer Structure**: Features, Core, Infrastructure, and Application layers
- **Feature Organization**: Each feature contains `Commands/`, `Events/`, `Handlers/`, and `Models/` subdirectories
- **Communication**: Features communicate via `CommandDispatcher` and `EventBus`, never directly
- **Key Decision**: Separate orchestration for Python and Server APIs - each optimized for its use case

## Tech Stack
- **Runtime**: Python 3.9+ (3.11+ recommended for Apple Silicon)
- **ML Framework**: MLX with whisper-large-v3-turbo model
- **Audio**: PyAudio at 16kHz, 16-bit, mono format
- **VAD**: Combined Silero + WebRTC with lazy initialization
- **Wake Word**: Porcupine (requires PORCUPINE_ACCESS_KEY)
- **Server**: FastAPI + WebSocket for real-time communication
- **Cloud Alternative**: OpenAI API integration (optional)
- **Testing**: pytest with mock-based testing

## Coding Standards
- **Architecture**: ALWAYS follow vertical slice architecture
- **Type Hints**: Required for all new code
- **Imports**: Use populated `__init__.py` files (e.g., `from src.Features import TranscriptionModule`)
- **Logging**: Use `LoggingModule.get_logger(__name__)` exclusively
- **Threading**: Non-daemon threads with proper cleanup
- **Audio Format**: Standardized at 16kHz, 16-bit, mono
- **API Design**: Maintain backward compatibility in public APIs

## Development Workflow
- **Installation**: `pip install -e .` or `uv pip install -e .` (pyproject.toml is single source of truth)
- **Testing**: Run `python tests/run_tests.py` for all tests
- **Examples**: Start with `python examples/cli.py` for interactive exploration
- **Commit Format**: Use conventional commits with descriptive messages
- **Branch Naming**: Current branch is `application_client`

## Important Notes
- **Platform Requirement**: Apple Silicon (M1/M2/M3) - Intel Macs not supported
- **API Layers**: Two separate APIs serve different needs - this is intentional, not duplication
- **Quick Start**: Use the interactive CLI at `examples/cli.py`
- **Environment Limitation**: Claude cannot execute example scripts directly - user must run them
- **Performance**: MLX optimizations required for all transcription code

## Project Structure
```
realtime_mlx_stt/      # Python API (direct use)
├── client.py          # Modern client interface
├── session.py         # Session-based API
├── transcriber.py     # Simple transcription API
└── config.py          # User-facing configurations

src/                   # Core implementation
├── Features/          # Feature modules with vertical slice architecture
├── Core/              # Interfaces and communication infrastructure
├── Infrastructure/    # Cross-cutting concerns (Logging, ProgressBar)
└── Application/       # Server implementation
    └── Server/        # REST/WebSocket API

examples/              # User examples
├── cli.py            # Interactive CLI (main entry point)
└── example_scripts/  # Additional example implementations
```

## Configuration
- **Profiles**: vad-triggered, wake-word
- **Environment Variables**: 
  - `OPENAI_API_KEY` (for cloud transcription)
  - `PORCUPINE_ACCESS_KEY` (for wake word)
  - `LOG_FORMAT`, `LOG_LEVEL` (logging control)

## Author
- Kristoffer Vatnehol (kristoffer.vatnehol@gmail.com)
- GitHub: github.com/kristofferv98/