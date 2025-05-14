# MLX Backend Installation Guide

This guide explains how to set up and use the MLX backend for Realtime_mlx_STT. The MLX backend provides optimized speech-to-text transcription on Apple Silicon hardware, enabling high-performance, low-latency transcription for macOS applications.

## Requirements

- **macOS** running on Apple Silicon (M1/M2/M3 chip)
- **Python 3.8+**
- **mlx** package and its dependencies

## Installation Steps

1. **Set up a Python environment**:
   ```bash
   # Create a virtual environment
   python -m venv env
   
   # Activate the environment
   source env/bin/activate  # On macOS/Linux
   ```

2. **Install MLX**:
   ```bash
   # Install MLX using uv for better performance
   uv pip install mlx
   ```

3. **Install Realtime_mlx_STT**:
   ```bash
   # Clone the repository (if you haven't already)
   git clone https://github.com/yourusername/Realtime_mlx_STT.git
   cd Realtime_mlx_STT
   
   # Install the package in development mode
   uv pip install -e ".[dev]"
   ```

## Using the MLX Backend

### Basic Usage

```python
from RealtimeSTT import AudioToTextRecorder

# Initialize recorder
recorder = AudioToTextRecorder(
    mlx_model_path="openai/whisper-large-v3-turbo",  # Model to use
    mlx_quick_mode=True,  # Enable parallel processing for faster transcription
    enable_realtime_transcription=True,  # Enable streaming transcription
    language="",  # Empty string for auto-detection, or use language code (e.g., "en")
)

# Start recording and transcription
recorder.start()

# Get text (blocks until recording/transcription completes)
text = recorder.text()
print(f"Transcription: {text}")

# Clean up
recorder.shutdown()
```

### Real-time Callbacks

```python
from RealtimeSTT import AudioToTextRecorder

# Define callback functions
def on_update(text):
    print(f"Real-time update: {text}")

def on_stabilized(text):
    print(f"Stabilized text: {text}")

# Initialize recorder with callbacks
recorder = AudioToTextRecorder(
    mlx_model_path="openai/whisper-large-v3-turbo",
    mlx_quick_mode=True,
    enable_realtime_transcription=True,
    on_realtime_transcription_update=on_update,
    on_realtime_transcription_stabilized=on_stabilized,
)

# Start recording
recorder.start()

# Wait for user to press Enter
input("Press Enter to stop recording...")

# Clean up
recorder.shutdown()
```

## Testing

To test the MLX backend:

```bash
# Run the basic MLX transcriber test
python tests/mlx_transcriber_test.py

# Test MLX streaming transcription
python tests/streaming_transcriber_test.py --audio /path/to/audio.wav --chunk-size 4000 --buffer-size 16000

# Test MLX backend integration with AudioToTextRecorder
python tests/mlx_audio_recorder_test.py

# Test with streaming mode
python tests/mlx_audio_recorder_test.py --streaming

# Test with a specific audio file
python tests/mlx_audio_recorder_test.py --audio /path/to/audio.wav

# Benchmark MLX performance
python tests/benchmark_mlx.py
```

## Performance Tuning

The MLX backend can be fine-tuned for performance vs. accuracy:

- **Batch Mode**: Better accuracy but higher latency
  - Set `enable_realtime_transcription=False`
  
- **Streaming Mode**: Lower latency but may have minor accuracy tradeoffs
  - Set `enable_realtime_transcription=True`
  
- **Quick Mode**: Enables parallel processing for faster results
  - Set `mlx_quick_mode=True` (recommended for most use cases)
  
- **Language Settings**: Impacts detection speed and accuracy
  - Set `language=""` for auto-detection (slower initial processing)
  - Set `language="en"` (or other language code) for faster processing when language is known

- **Buffer Size**: Affects transcription quality vs. latency
  - Larger buffer size (via MLXTranscriber) provides better accuracy but higher latency
  - Smaller buffer size provides lower latency but may reduce accuracy

- **Memory Management**: 
  - Always call `cleanup()` or `shutdown()` when done to release resources
  - For long-running applications, periodically restart transcribers to free memory

## Troubleshooting

- **Error: "MLX backend requires Apple Silicon"**
  - The MLX backend only works on Macs with M1/M2/M3 chips
  - This library is specifically optimized for Apple Silicon and is not compatible with Intel Macs
  - Cannot be used on Windows, Linux, or Intel-based Macs

- **High Memory Usage**
  - The Whisper large-v3-turbo model requires significant memory (1-2GB)
  - Close other memory-intensive applications
  - Call `cleanup()` or `shutdown()` when done to release resources
  - For long-running applications, periodically restart the transcriber
  - Check for memory leaks with Activity Monitor

- **Slow Transcription**
  - Enable `mlx_quick_mode=True` for parallel processing
  - Check if other applications are using the GPU/Neural Engine
  - Ensure your Mac is not thermal throttling (check with Activity Monitor)
  - Specify a language instead of using auto-detection
  - For repeated transcriptions, keep the transcriber running instead of restarting

- **Model Download Issues**
  - If you encounter SSL or network issues when downloading the model:
    - Use a VPN if Hugging Face is blocked in your region
    - Check your firewall settings
    - Try downloading the model manually and place it in ~/.cache/huggingface/
    - Ensure you have at least 4GB of free disk space

- **Low Transcription Accuracy**
  - Increase the audio quality (reduce background noise)
  - Use batch mode instead of streaming for final transcriptions
  - If using streaming mode, increase buffer size and overlap parameters
  - For non-English languages, explicitly specify the language code