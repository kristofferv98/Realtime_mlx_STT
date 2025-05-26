# Realtime MLX STT Examples

## 🚀 Interactive CLI

Explore all features through our user-friendly interactive CLI:

```bash
python examples/cli.py
```

The CLI provides a clean menu interface for:
- **Quick Transcription** - 10-second local MLX transcription
- **Streaming Mode** - Continuous transcription with stop command
- **OpenAI Cloud** - Compare with cloud-based transcription
- **Wake Word Detection** - "Jarvis" activation demo
- **Device Selection** - Choose your microphone
- **Language Settings** - 16 languages + auto-detect
- **Help & Documentation** - Built-in guide

## 📁 Additional Example Scripts

For developers who want to see implementation details, additional example scripts are available in the `example_scripts/` directory:

- `transcribe.py` - Full-featured command-line interface
- `client_example.py` - High-level API usage patterns
- `openai_only.py` - Dedicated OpenAI integration
- `wake_word_example.py` - Wake word detection focus
- `check_audio_devices.py` - Audio device utility

## 🔑 API Keys

Some features require API keys:

- **OpenAI Transcription**: Set `OPENAI_API_KEY` environment variable
- **Wake Word Detection**: Set `PORCUPINE_ACCESS_KEY` environment variable

Get your keys:
- OpenAI: https://platform.openai.com/
- Porcupine: https://picovoice.ai/ (free tier available)

## 📋 Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python 3.9+
- Working microphone
- Optional: API keys for cloud features