# CLAUDE.md for Realtime_mlx_STT

## CRITICAL DIRECTIVES
- **ALWAYS** follow vertical slice architecture for all code changes.
- **ALWAYS** implement features within `src/Features/` with `Commands/`, `Events/`, `Handlers/`, and `Models/` subdirectories. Infrastructure components (like Logging) reside in `src/Infrastructure/`.
- **ALWAYS** use `uv sync pyproject.toml` for dependency management.
- **ALWAYS** implement interfaces defined in `Core/` when creating feature components.

## CODE PATTERNS
- Command objects define actions to be performed.
- Event objects notify about state changes.
- CommandHandler implementations process specific commands.
- Features communicate via `CommandDispatcher` and `EventBus`, never directly.
- **Consider** process isolation for ML components for stability, but direct integration (e.g., for MLX Whisper) is **preferred** for performance when feasible.
- All audio processing at 16kHz, 16-bit, mono format.

## IMPLEMENTATION GUIDELINES
- AudioCapture, VoiceActivityDetection, WakeWordDetection Transcription features are COMPLETED.
- MLX optimizations are REQUIRED for all transcription code running on Apple Silicon.
- Type hints MUST be used throughout all new code.
- Tests MUST be written for all new functionality.
- **ALWAYS** use `src.Infrastructure.Logging.LoggingModule.get_logger(__name__)` for acquiring loggers to ensure standardized namespaces and centralized control as defined in `specs/logging_design.md`.

## TECHNICAL CONSTRAINTS
- Target platform is Apple Silicon (macOS).
- Balance latency vs. accuracy in real-time processing.
- Memory efficiency is critical for ML model loading.
- Claude (you) can not run scripts in the example script, since in the enviorment it breaks. The user will do so.

## USER INFO
- Kristoffer Vatnehol (kristoffer.vatnehol@gmail.com)
- GitHub: github.com/kristofferv98/