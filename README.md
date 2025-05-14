# Realtime_mlx_STT

Realtime_mlx_STT is a high-performance speech-to-text transcription library optimized exclusively for Apple Silicon. It leverages Apple's MLX framework to run the Whisper large-v3-turbo model with maximum performance on macOS devices with Apple Silicon chips.

## Features

- **Real-time transcription** with low latency for macOS applications
- **Apple Silicon optimization** using MLX with Neural Engine acceleration
- **Voice activity detection** to automatically process speech segments
- **Streaming and batch modes** for flexible transcription options
- **Thread-safe design** for responsive applications
- **Simple API** with both high-level recorder and direct transcriber access

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3) - Required, not optional
- **Python 3.8+**
- **MLX** for Apple Silicon optimization
- **PyAudio** for audio capture
- **NumPy** for audio processing
- **Hugging Face Hub** for model downloading

> **Important Note**: This library is specifically optimized for Apple Silicon and will not work on Intel-based Macs or other platforms. It requires the Neural Engine found in Apple Silicon chips to achieve optimal performance.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Realtime_mlx_STT.git
cd Realtime_mlx_STT

# Create a virtual environment (recommended)
python -m venv env
source env/bin/activate

# Install MLX
uv pip install mlx

# Install the package in development mode
uv pip install -e ".[dev]"
```

For more detailed installation instructions including troubleshooting, see [MLX_INSTALLATION_GUIDE.md](MLX_INSTALLATION_GUIDE.md).

## Quick Start

### Basic Transcription

```python
from RealtimeSTT import MLXTranscriber
import numpy as np
import soundfile as sf

# Load audio file
audio, sample_rate = sf.read("audio.wav")

# Initialize transcriber
transcriber = MLXTranscriber(
    model_path="openai/whisper-large-v3-turbo",
    any_lang=False,  # Set to True for language auto-detection
    language="en"    # Specify language if any_lang is False
)
transcriber.start()

# Transcribe audio
transcriber.transcribe(audio)

# Get the result
result = transcriber.get_result(timeout=10.0)
if result:
    print(f"Transcription: {result['text']}")
    print(f"Processing time: {result['processing_time']:.2f}s")

# Clean up
transcriber.stop()
transcriber.cleanup()
```

### Using AudioToTextRecorder

```python
from RealtimeSTT import AudioToTextRecorder

# Initialize recorder
recorder = AudioToTextRecorder(
    mlx_model_path="openai/whisper-large-v3-turbo",  # MLX-specific parameter
    mlx_quick_mode=True,                            # Enables parallel processing
    language="en",                                  # Leave empty for auto-detection
    enable_realtime_transcription=True,            # Enable streaming mode
)

# Define callback for real-time updates
def on_update(text):
    print(f"Partial transcription: {text}")
recorder.on_realtime_transcription_update = on_update

# Start recording
recorder.start()

# Get transcription (blocks until speech is detected and transcribed)
text = recorder.text()
print(f"Final transcription: {text}")

# Clean up
recorder.shutdown()
```

### Real-time Microphone Transcription

```python
from RealtimeSTT import AudioInput, MLXTranscriber
import numpy as np
import struct
import time
import threading

# Initialize components
audio_input = AudioInput()
transcriber = MLXTranscriber(
    model_path="openai/whisper-large-v3-turbo",
    realtime_mode=True,    # Enable streaming mode
    any_lang=False,        # Set to True for language auto-detection
    language="en",         # Specify language code
    quick=True             # Enable parallel processing
)

# Set up audio
if not audio_input.setup():
    print("Failed to set up audio input")
    exit(1)

# Start transcriber
transcriber.start()

# Flag to control recording
running = True

def process_results():
    while running:
        result = transcriber.get_result(timeout=0.5)
        if result:
            # For streaming results, you'll get both full text and new segments
            if 'new_text' in result:
                print(f"New segment: {result['new_text']}")
            print(f"Full transcription: {result['text']}")

# Start result processing thread
result_thread = threading.Thread(target=process_results)
result_thread.daemon = True
result_thread.start()

try:
    print("Recording... Speak now! (Press Ctrl+C to stop)")
    while running:
        # Read audio chunk
        raw_data = audio_input.read_chunk()
        
        # Convert to numpy array
        fmt = f"{audio_input.chunk_size}h"
        pcm_data = np.array(struct.unpack(fmt, raw_data)) / 32768.0
        
        # Resample if needed
        if (audio_input.device_sample_rate != audio_input.target_samplerate and
            audio_input.resample_to_target):
            pcm_data = audio_input.resample_audio(
                pcm_data,
                audio_input.target_samplerate,
                audio_input.device_sample_rate
            )
        
        # Process audio chunk - specify if this is the last chunk
        # For continuous processing, is_last=False
        transcriber.add_audio_chunk(pcm_data, is_last=False)
        
except KeyboardInterrupt:
    print("Stopping...")
    running = False
    # Send a final chunk with is_last=True to get final result
    transcriber.add_audio_chunk(np.zeros(1), is_last=True)
    time.sleep(0.5)  # Allow final processing to complete

# Clean up
transcriber.stop()
transcriber.cleanup()
audio_input.cleanup()
```

## Test Scripts

The repository includes several test scripts to verify functionality:

### MLX Transcriber Test

Tests the MLX transcriber with a sample audio file:

```bash
python tests/mlx_transcriber_test.py --audio path/to/audio.wav
```

### Streaming Transcriber Test

Tests the MLX streaming transcriber with a sample audio file:

```bash
python tests/streaming_transcriber_test.py --audio path/to/audio.wav --chunk-size 4000 --buffer-size 16000
```

### Audio Recorder Test

Tests the AudioToTextRecorder with MLX backend:

```bash
python tests/mlx_audio_recorder_test.py --streaming  # Test with streaming mode
```

### Integration Test

Tests microphone integration with the transcriber:

```bash
# List available audio devices
python tests/integration_test.py --list-devices

# Test with default microphone for 10 seconds
python tests/integration_test.py --duration 10

# Test with specific device
python tests/integration_test.py --device 1 --duration 15
```

### Benchmark Test

Tests the performance of the MLX transcriber:

```bash
python tests/benchmark_mlx.py --audio path/to/audio.wav
```

## Components

### Audio Input

The `AudioInput` class handles microphone input using PyAudio, providing audio data to the recorder. It includes:

- Sample rate detection and conversion
- Device selection
- Audio resampling with anti-aliasing filters

### MLX Transcriber

The `MLXTranscriber` class provides the core transcription functionality:

- MLX-optimized Whisper large-v3-turbo model
- Streaming and batch processing modes
- Thread-safe operation
- Optimized for Apple Silicon with Neural Engine acceleration

### AudioToTextRecorder

The `AudioToTextRecorder` class orchestrates the transcription process:

- Voice activity detection (VAD) for automatic speech detection
- Real-time transcription capabilities with incremental updates
- Callback system for integrating with applications
- Thread-safe audio buffering and processing

## Performance

On Apple Silicon (M1/M2/M3), the MLX-optimized Whisper-large-v3-turbo model typically achieves:

- **Batch mode**: ~0.3-0.5x realtime (processes 60 seconds of audio in 20-30 seconds)
- **Streaming mode**: ~0.5-0.7x realtime (processes audio with ~2-3 second latency)

Performance varies by Apple Silicon generation:

| Device | Batch Processing | Streaming Latency | Memory Usage |
|--------|------------------|-------------------|--------------|
| M1     | ~0.5x realtime   | ~3 second         | ~1-2 GB      |
| M2     | ~0.4x realtime   | ~2.5 second       | ~1-2 GB      |
| M3     | ~0.3x realtime   | ~2 second         | ~1-2 GB      |

The MLX implementation takes full advantage of the Neural Engine in Apple Silicon chips, providing significantly better performance than CPU-based implementations. Key optimizations include:

- Mixed-precision computation optimized for Apple Silicon
- Parallel processing with `quick_mode=True`
- Efficient key-value caching for streaming transcription
- Direct NumPy array integration with zero disk I/O

## Documentation

- [Comprehensive Documentation](COMPREHENSIVE_DOCUMENTATION.md) - Full library documentation
- [MLX Installation Guide](MLX_INSTALLATION_GUIDE.md) - Detailed installation instructions
- [API Reference](COMPREHENSIVE_DOCUMENTATION.md#api-reference) - Complete API reference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the base Whisper large-v3-turbo model
- [MLX](https://github.com/ml-explore/mlx) for Apple Silicon optimization
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) for the original audio processing framework
- [Hugging Face](https://huggingface.co) for model distribution infrastructure