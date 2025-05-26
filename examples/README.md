# Realtime_mlx_STT Examples

This folder contains simple, clear examples demonstrating the main features of Realtime_mlx_STT.

## Examples

### 1. check_audio_devices.py
Lists all available audio input devices on your system.

```bash
python check_audio_devices.py
```

Use this to find the device ID for your microphone.

### 2. simple_vad_example.py
The simplest VAD transcription example - minimal code, maximum clarity.
- Uses Voice Activity Detection (VAD) 
- Transcribes automatically when you stop speaking
- Good starting point for understanding the library

```bash
python simple_vad_example.py
```

### 3. vad_transcription.py
Full-featured VAD transcription with more options.
- All features from simple_vad_example.py
- Command-line arguments for device and language selection
- More detailed logging and statistics
- Class-based structure for integration into larger projects

```bash
# Use default microphone and auto-detect language
python vad_transcription.py

# Specify device and language
python vad_transcription.py --device 1 --language en
```

### 4. wake_word_example.py
Wake word triggered transcription (like "Hey Siri" or "OK Google").
- Listens for a wake word (default: "porcupine")
- Only transcribes speech after the wake word
- Returns to listening mode after timeout

```bash
# Requires PORCUPINE_ACCESS_KEY environment variable
export PORCUPINE_ACCESS_KEY='your-key-here'

# Use default wake word "porcupine"
python wake_word_example.py

# Use custom wake word and timeout
python wake_word_example.py --wake-word jarvis --timeout 60
```

Get your free Porcupine access key at: https://picovoice.ai/

## Requirements

All examples require:
- Apple Silicon Mac (M1/M2/M3)
- Python 3.9+
- Microphone access
- Dependencies installed: `pip install -e .`

## Key Differences

- **simple_vad_example.py**: Minimal code, easiest to understand, uses VAD for automatic transcription
- **vad_transcription.py**: Same as simple but with more features, options, and better structure
- **wake_word_example.py**: Requires saying a wake word before transcription begins

## Tips

- Run `check_audio_devices.py` first to find your microphone's device ID
- Start with `simple_vad_example.py` to understand the basics
- Use `vad_transcription.py` for production use cases
- Wake word detection requires a Picovoice access key (free tier available)