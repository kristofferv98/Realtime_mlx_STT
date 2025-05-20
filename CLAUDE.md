# CLAUDE.md for Realtime_mlx_STT

## CRITICAL DIRECTIVES
- **ALWAYS** follow vertical slice architecture for all code changes
- **ALWAYS** implement features within src/Features/ with Commands/, Events/, Handlers/, and Models/ subdirectories
- **NEVER** modify code in RealtimeSTT/ directory except for bug fixes
- **ALWAYS** use `uv sync pyproject.toml` for dependency management
- **ALWAYS** implement interfaces defined in Core/ when creating feature components

## CODE PATTERNS
- Command objects define actions to be performed
- Event objects notify about state changes
- CommandHandler implementations process specific commands
- Features communicate via CommandDispatcher and EventBus, never directly
- Use process isolation for ML components to improve stability
- All audio processing at 16kHz, 16-bit, mono format

## IMPLEMENTATION GUIDELINES
- AudioCapture and VoiceActivityDetection and Transcription features are COMPLETED
- MLX optimizations are REQUIRED for all transcription code
- Type hints MUST be used throughout all new code
- Tests MUST be written for all new functionality

## TECHNICAL CONSTRAINTS
- Target platform is Apple Silicon (macOS)
- Balance latency vs. accuracy in real-time processing
- Memory efficiency is critical for ML model loading

## USER INFO
- Kristoffer Vatnehol (kristoffer.vatnehol@gmail.com)
- GitHub: github.com/kristofferv98/