# Realtime MLX STT API

This directory contains the high-level Python API for Realtime MLX STT.

## API Levels

### 1. Simple API (`transcriber.py`)
- Easy to use for beginners
- Backward compatible with early versions
- Good for quick prototypes

### 2. Configuration API (`config.py`)
- Type-safe configuration classes
- Validation at configuration time
- Full control over all parameters

### 3. Session API (`session.py`)
- Follows the server's proven pattern
- Proper state management
- Recommended for production use

## Module Structure

```
realtime_mlx_stt/
├── __init__.py      # Package exports
├── config.py        # Configuration classes
├── session.py       # Session-based API
├── transcriber.py   # Simple high-level API
├── types.py         # Type definitions
├── utils.py         # Helper functions
└── wake_word.py     # Wake word wrapper
```

## Usage Examples

See the `examples/` directory for complete examples:
- `simple_api_demo.py` - Basic usage
- `config_api_demo.py` - Configuration-based usage
- `session_api_demo.py` - Session-based usage (recommended)

## Design Principles

1. **Multiple API Levels** - From simple to advanced
2. **Type Safety** - Full type hints and validation
3. **Event-Driven** - Callbacks for real-time updates
4. **State Management** - Proper lifecycle handling
5. **Backward Compatible** - Old code continues to work