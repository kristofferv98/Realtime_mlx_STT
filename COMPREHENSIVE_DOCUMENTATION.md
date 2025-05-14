# Realtime_mlx_STT: Comprehensive Documentation

This document provides a complete reference for the Realtime_mlx_STT library, which implements high-performance, real-time speech-to-text transcription optimized for Apple Silicon using the MLX framework.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements & Installation](#requirements--installation)
3. [Core Architecture & Components](#core-architecture--components)
4. [Key Functionalities](#key-functionalities)
5. [Usage](#usage)
   - [Basic Transcription](#basic-transcription)
   - [Real-time Transcription](#real-time-transcription)
   - [Advanced: Direct StreamingTranscriber Usage](#advanced-direct-streamingtranscriber-usage)
6. [API Reference](#api-reference)
7. [Performance](#performance)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Limitations & Future Work](#limitations--future-work)

## Project Overview

Realtime_mlx_STT is a high-performance speech-to-text library optimized exclusively for Apple Silicon. It combines real-time audio processing with the Whisper large-v3-turbo model optimized through Apple's MLX framework, providing low-latency, high-accuracy transcription for macOS applications.

The library offers:
- Real-time speech recognition with voice activity detection (VAD)
- Both batch and streaming transcription modes
- Direct access to raw transcription results or simple callback-driven operation
- Fully optimized performance on Apple Silicon (M1/M2/M3) using the Neural Engine and GPU
- Wake word detection capabilities

## Requirements & Installation

### System Requirements

- **Apple Silicon Mac** (M1, M2, or M3 series) - This is mandatory as the library uses MLX
- **macOS** (12+)
- **Python** 3.8+

### Dependencies

- **Core libraries**: mlx, numpy, pyaudio, tiktoken, huggingface_hub
- **Audio processing**: scipy, soundfile
- **VAD components**: webrtcvad, torch (for Silero VAD)
- **Optional**: openwakeword, pvporcupine (for wake word detection)

### Installation

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate

# Install MLX
pip install mlx

# Install the package
pip install -e ".[dev]"
```

## Core Architecture & Components

Realtime_mlx_STT is built around a modular architecture that enables efficient capture, processing, and transcription of audio:

### Key Components

1. **`AudioInput`** (audio_input.py)
   - Manages microphone input via PyAudio
   - Handles device selection and sample rate negotiation
   - Provides audio resampling capabilities

2. **`AudioToTextRecorder`** (audio_recorder.py)
   - Central orchestrator for the transcription process
   - Manages voice activity detection (VAD)
   - Handles audio buffering and preprocessing
   - Controls recording state and callbacks
   - Integrates with the MLXTranscriber for transcription

3. **`MLXTranscriber`** (mlx_transcriber.py)
   - Bridge between audio processing and the MLX-optimized Whisper model
   - Manages worker thread for non-blocking transcription
   - Supports both batch and streaming transcription modes
   - Handles model loading and resource management

4. **`whisper_turbo.py`**
   - Implements the core Whisper large-v3-turbo model using MLX
   - Provides audio preprocessing (mel spectrogram generation)
   - Includes both batch (`Transcriber`) and streaming (`StreamingTranscriber`) implementations
   - Optimized for Apple Silicon performance

### Batch Mode Data Flow

1. **Audio Capture**
   - Audio is captured in chunks from the microphone via PyAudio
   - Raw audio chunks are placed in a queue for processing

2. **Voice Activity Detection**
   - Combines WebRTC VAD (fast initial detection) and Silero VAD (accurate verification)
   - Controls when recording starts and stops based on detected speech

3. **Audio Buffering**
   - Pre-recording buffer captures audio just before speech is detected
   - Recording buffer collects audio during active speech
   - Audio is converted to normalized NumPy arrays for transcription

4. **Transcription**
   - Complete audio segment is processed at once via `MLXTranscriber.transcribe()`
   - MLXTranscriber passes audio to `whisper_turbo.Transcriber`
   - Results are returned via callbacks or direct method returns

### Streaming Mode Data Flow

1. **Audio Capture & Buffering**
   - Same as batch mode, but audio chunks are sent continuously

2. **Streaming Transcription**
   - Chunks are submitted via `MLXTranscriber.add_audio_chunk()`
   - `StreamingTranscriber` processes chunks incrementally
   - Maintains KV cache state between processing steps for efficiency
   - Returns partial transcripts that build up over time

3. **Result Handling**
   - Real-time updates are provided during transcription
   - Final transcript is returned when speech ends

## Key Functionalities

### Audio Processing

- 16kHz, 16-bit, mono audio format throughout the pipeline
- Automatic sample rate conversion when needed
- Thread-safe buffering between audio capture and transcription

### Voice Activity Detection

- Two-stage VAD combining WebRTC (fast, low resource) and Silero (accurate, ML-based)
- Configurable sensitivity for different environments
- Silence detection for determining end of speech segments
- Buffering system captures pre-speech audio to avoid missing initial words

### MLX-Optimized Transcription

- Uses Whisper large-v3-turbo model optimized with Apple's MLX framework
- Supports both batch transcription and streaming real-time transcription
- Maintains key-value cache across audio chunks for context preservation in streaming
- Direct NumPy array support eliminates disk I/O for temporary files

### Wake Word Detection (Optional)

- Support for Porcupine or OpenWakeWord libraries
- Configurable sensitivity and wake word phrases
- Automatic activation of transcription on wake word detection

## Usage

### Basic Transcription

For simple batch transcription of audio:

```python
from RealtimeSTT import MLXTranscriber
import numpy as np
import soundfile as sf

# Load audio file
audio, sample_rate = sf.read("audio.wav")
if sample_rate != 16000:
    # Resample to 16kHz if needed
    from scipy import signal
    audio = signal.resample(audio, int(len(audio) * 16000 / sample_rate))

# Initialize transcriber with MLX-specific parameters
transcriber = MLXTranscriber(
    model_path="openai/whisper-large-v3-turbo",  # Model to use
    realtime_mode=False,                         # Batch mode for this example
    any_lang=False,                              # False if language is known
    quick=True,                                  # Enable parallel processing
    language="en"                                # Specific language code, or None for auto-detection
)
transcriber.start()

try:
    # Transcribe audio
    transcriber.transcribe(audio)
    
    # Get the result
    result = transcriber.get_result(timeout=10.0)
    if result:
        print(f"Transcription: {result['text']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
finally:
    # Clean up
    transcriber.stop()
    transcriber.cleanup()
```

### Real-time Transcription

For real-time transcription with automatic speech detection:

```python
from RealtimeSTT import AudioToTextRecorder

# Define callbacks for real-time updates
def on_update(text):
    print(f"Partial: {text}")

def on_stabilized(text):
    print(f"Stabilized: {text}")

# Define additional callbacks (optional)
def on_recording_start():
    print("Recording started - Voice activity detected")
    
def on_recording_stop():
    print("Recording stopped - Voice activity ended")
    
def on_transcription_start(audio_data):
    print(f"Starting transcription of {len(audio_data)} samples")
    return True  # Return False to abort transcription

# Initialize recorder with MLX-specific parameters
recorder = AudioToTextRecorder(
    # MLX-specific parameters
    mlx_model_path="openai/whisper-large-v3-turbo",  # Model to use
    mlx_quick_mode=True,                            # Enable parallel processing
    
    # General parameters
    language="en",                                  # Specific language or "" for auto
    input_device_index=None,                        # Audio device to use (None for default)
    
    # Event callbacks
    on_recording_start=on_recording_start,
    on_recording_stop=on_recording_stop,
    on_transcription_start=on_transcription_start,
    
    # Real-time parameters
    enable_realtime_transcription=True,             # Enable streaming mode
    on_realtime_transcription_update=on_update,
    on_realtime_transcription_stabilized=on_stabilized,
    
    # VAD parameters
    silero_sensitivity=0.4,                         # VAD sensitivity (0.0-1.0)
    webrtc_sensitivity=3,                           # WebRTC VAD mode (0-3)
    post_speech_silence_duration=0.6,               # Seconds of silence to end recording
)

try:
    # Start recording and listening for speech
    recorder.start()
    print("Listening... (speak now)")
    
    # Get final transcription (blocks until speech is detected and transcribed)
    text = recorder.text()
    print(f"Final: {text}")
finally:
    # Clean up
    recorder.shutdown()
```

### Advanced: Direct StreamingTranscriber Usage

For lower-level control over streaming transcription:

```python
from RealtimeSTT.whisper_turbo import create_streaming_transcriber
import numpy as np
import soundfile as sf
import time

# Create a streaming transcriber with specific parameters
transcriber = create_streaming_transcriber(
    buffer_size=16000,  # Buffer size in samples (1 second at 16kHz)
    overlap=2000,       # Overlap between segments in samples
    config_path=None,   # Optional path to custom config (default: use built-in config)
    weights_path=None   # Optional path to custom weights (default: use built-in weights)
)

# Load audio file and split into chunks
audio, sr = sf.read("audio.wav")
if sr != 16000:
    # Resample to 16kHz if needed
    from scipy import signal
    audio = signal.resample(audio, int(len(audio) * 16000 / sr))

# Create equal-sized chunks
chunk_size = 4000  # 250ms chunks at 16kHz
chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

# Process chunks, simulating real-time input
start_time = time.time()
for i, chunk in enumerate(chunks):
    # Check if this is the last chunk
    is_last = (i == len(chunks) - 1)
    
    # Process this chunk
    result = transcriber.process_chunk(
        chunk,
        is_last=is_last,
        language="en"  # Specify language (None for auto-detection)
    )
    
    # Use the result
    print(f"Chunk {i+1}/{len(chunks)}")
    
    # New text is the incremental addition from this chunk
    if result['new_text']:
        print(f"New text: {result['new_text']}")
    
    # Full text is the complete transcription so far
    if i % 5 == 0 or is_last:  # Print full text every 5 chunks or at the end
        print(f"Full text so far: {result['text']}")
    
    # Check if transcription is marked as final (on last chunk)
    if result['is_final']:
        elapsed = time.time() - start_time
        print(f"Final result: {result['text']}")
        print(f"Total processing time: {elapsed:.2f}s for {len(audio)/16000:.2f}s audio")
        print(f"Realtime factor: {elapsed/(len(audio)/16000):.2f}x")
        break
        
    # Optional: Simulate real-time by adding a small delay
    # time.sleep(0.05)  # Uncomment to add deliberate latency

# Reset the transcriber state for processing a new stream
transcriber.reset()
```

## API Reference

### AudioToTextRecorder

The main class for capturing and transcribing audio.

#### Constructor Parameters

```python
AudioToTextRecorder(
    # MLX-specific parameters (required for this implementation)
    mlx_model_path: str = "openai/whisper-large-v3-turbo",  # Hugging Face model path
    mlx_quick_mode: bool = True,                           # Enable parallel processing
    
    # General parameters
    language: str = "",  # Empty for auto-detection, or language code (e.g., "en", "fr")
    input_device_index: int = None,  # Audio device index, None for default
    
    # Callback parameters
    on_recording_start=None,         # Called when recording starts
    on_recording_stop=None,          # Called when recording stops
    on_transcription_start=None,     # Called before transcription starts, can abort if returns False
    
    # Formatting options
    ensure_sentence_starting_uppercase=True,  # Capitalize first letter
    ensure_sentence_ends_with_period=True,    # Add period at end if missing
    
    # Behavior flags
    use_microphone=True,  # Use microphone input (False for programmatic audio feeding)
    spinner=True,         # Show progress spinner in terminal
    debug_mode=False,     # Enable extra logging
    
    # Real-time parameters
    enable_realtime_transcription=False,               # Enable streaming mode
    realtime_processing_pause=0.2,                     # Seconds between realtime updates
    init_realtime_after_seconds=0.2,                   # Seconds to wait before first realtime transcription
    on_realtime_transcription_update=None,             # Callback for each update
    on_realtime_transcription_stabilized=None,         # Callback for stabilized text
    
    # VAD parameters
    silero_sensitivity: float = 0.4,                   # Silero VAD sensitivity (0.0-1.0)
    webrtc_sensitivity: int = 3,                       # WebRTC VAD mode (0-3)
    post_speech_silence_duration: float = 0.6,         # Seconds of silence to end recording
    min_length_of_recording: float = 0.5,              # Minimum recording length in seconds
    min_gap_between_recordings: float = 0,             # Minimum time between recordings
    pre_recording_buffer_duration: float = 1.0,        # Seconds of audio to keep before speech
    silero_use_onnx: bool = False,                     # Use ONNX runtime for Silero
    silero_deactivity_detection: bool = False,         # Use Silero for deactivation detection
    
    # Wake word parameters (optional)
    wakeword_backend: str = "",                        # "pvporcupine" or "oww"
    wake_words: str = "",                              # Comma-separated wake word phrase(s)
    wake_words_sensitivity: float = 0.6,               # Wake word detection sensitivity
    wake_word_activation_delay: float = 0.0,           # Delay after wake word detection
    wake_word_timeout: float = 5.0,                    # Timeout for wake word detection
    wake_word_buffer_duration: float = 0.1,            # Audio buffer for wake word
    openwakeword_model_paths: str = None,              # Custom OpenWakeWord model paths
    openwakeword_inference_framework: str = "onnx",    # Framework for OpenWakeWord
    
    # Advanced parameters
    on_vad_start=None,                                 # Called when VAD detects speech
    on_vad_stop=None,                                  # Called when VAD detects silence
    on_vad_detect_start=None,                          # Called when VAD detection starts
    on_vad_detect_stop=None,                           # Called when VAD detection stops
    on_wakeword_detected=None,                         # Called when wake word detected
    on_wakeword_timeout=None,                          # Called when wake word times out
    on_wakeword_detection_start=None,                  # Called when wake word detection starts
    on_wakeword_detection_end=None,                    # Called when wake word detection ends
    on_turn_detection_start=None,                      # Called when turn detection starts
    on_turn_detection_stop=None,                       # Called when turn detection stops
    on_recorded_chunk=None,                            # Called with each recorded chunk
    handle_buffer_overflow: bool = True,               # Handle audio buffer overflow
    buffer_size: int = 512,                            # PyAudio buffer size
    sample_rate: int = 16000,                          # Sample rate (must be 16kHz)
    print_transcription_time: bool = False,            # Print processing time
    early_transcription_on_silence: int = 0,           # Early transcribe after ms silence
    allowed_latency_limit: int = 100,                  # Max queued audio chunks
    normalize_audio: bool = False,                     # Normalize audio levels
    start_callback_in_new_thread: bool = False,        # Run callbacks in new thread
)
```

#### Key Methods

- **`text()`**: Blocks until speech is detected and transcribed, returns the transcription
- **`start()`**: Starts the recorder (listening for speech)
- **`stop()`**: Stops the recorder
- **`feed_audio(audio_data)`**: Passes audio data directly to the recorder
- **`shutdown()`**: Cleans up all resources

### MLXTranscriber

Bridge between audio processing and MLX transcription.

#### Constructor Parameters

```python
MLXTranscriber(
    model_path: str = "openai/whisper-large-v3-turbo",  # Hugging Face model path
    realtime_mode: bool = True,     # True for streaming mode, False for batch mode
    device: str = "auto",           # "auto", "cpu", or "gpu" (auto selects best available)
    any_lang: bool = False,         # True for language auto-detection, False to use specific language
    quick: bool = True,             # True enables parallel processing for faster results
    language: str = None            # Language code (e.g., "en", "fr") or None for auto-detection
)
```

#### Key Methods

- **`start()`**: Starts the transcription worker thread
- **`transcribe(audio_data)`**: Submits audio for batch transcription
- **`add_audio_chunk(audio_chunk, is_last=False)`**: Adds an audio chunk for streaming transcription
- **`get_result(timeout=0.1)`**: Retrieves a transcription result if available
- **`stop()`**: Stops the worker thread
- **`cleanup()`**: Releases resources

## Performance

On Apple Silicon, the MLX-optimized Whisper-large-v3-turbo model achieves excellent performance:

### Batch Mode
- **Speed**: ~0.3-0.5x realtime (processes 60 seconds of audio in 20-30 seconds)
- **Memory**: ~1-2GB during transcription
- **Accuracy**: Comparable to OpenAI Whisper (WER typically <5% on clear speech)

### Streaming Mode
- **Speed**: ~0.5-0.7x realtime (processes audio with ~2-3 second latency)
- **Memory**: ~1.5-2.5GB during transcription (higher due to KV cache)
- **Accuracy**: Slightly lower than batch mode but still very good

### Performance Comparison

| Feature | MLX Streaming | MLX Batch |
|---------|---------------|-----------|
| Latency | Low | High |
| Accuracy | Good | Best |
| Memory Usage | Higher | Lower |
| CPU Usage | High | Medium |
| Real-time Factor | 0.5-0.7x | 0.3-0.5x |
| Timestamp Support | No | Yes |
| Platform Compatibility | Apple Silicon only | Apple Silicon only |
| KV Cache State | Maintained between chunks | N/A |
| Resource Efficiency | Excellent | Very Good |

### Optimizing Performance

1. **Buffer Size**: Adjust based on latency vs. accuracy tradeoff
   - Smaller buffer (e.g., 8000 samples) = Lower latency, potentially lower accuracy
   - Larger buffer (e.g., 24000 samples) = Higher accuracy, higher latency

2. **Quick Mode**: Set `mlx_quick_mode=True` for faster processing with parallel execution

3. **Memory Management**: Call `cleanup()` when done to release resources

4. **Audio Quality**: Better quality audio input yields better transcription results

## Testing

The repository includes comprehensive test scripts:

### Basic Tests

```bash
# Basic functionality test
python tests/mlx_audio_recorder_test.py

# Streaming test
python tests/mlx_audio_recorder_test.py --streaming

# Integration test with microphone
python tests/integration_test.py
```

### Performance Testing

```bash
# Benchmark MLX performance
python tests/benchmark_mlx.py

# Test streaming transcription with various parameters
python tests/streaming_transcriber_test.py --audio /path/to/audio.wav --chunk-size 4000 --buffer-size 16000 --overlap 2000

# Generate performance graphs
python tests/streaming_transcriber_test.py --audio /path/to/audio.wav --output results.json --plot results.png
```

## Troubleshooting

### Common Issues

1. **"RuntimeError: MLX backend requires Apple Silicon"**:
   - This library only works on Apple Silicon (M1/M2/M3) Macs.
   - Verify you're using a supported device.

2. **High memory usage**:
   - Reduce buffer size
   - Ensure `cleanup()` is called when done
   - Reset streaming state between sessions

3. **Poor transcription quality**:
   - Increase buffer size
   - Increase overlap between chunks in streaming mode
   - Ensure audio is properly normalized
   - Check microphone quality and environment noise

4. **High latency**:
   - Decrease buffer size
   - Use `quick=True` mode
   - Process smaller chunks

5. **Crashes or errors**:
   - Check for minimum audio length (at least 512 samples)
   - Ensure audio format is correct (16kHz, float32, mono)
   - Verify Apple Silicon hardware is used

### Recommended Settings for Different Use Cases

- **Accuracy-focused**: Use batch mode with `enable_realtime_transcription=False`
- **Low-latency**: Use streaming mode with smaller buffer size
- **Balanced**: Use streaming mode with default buffer size

## Limitations & Future Work

### Current Limitations

- No timestamp information in streaming mode
- No interim confidence scores
- No word-level alignment
- Language detection only happens at the start of streaming
- The implementation is optimized for Apple Silicon and requires MLX
- No cross-platform support (macOS with Apple Silicon only)

### Planned Enhancements

- Add timestamp support for streaming mode
- Implement word-level confidence scores
- Support for dynamic language switching
- Optimize KV cache management for very long streams
- Add adaptive buffer sizing based on speech patterns
- Implement more advanced memory management for extended operation
- Improve handling of diverse speech patterns and accents

### Memory Optimization Opportunities

- Analyze and reduce memory usage for extended transcription
- Implement more aggressive cleanup for unused resources
- Add support for smaller models for memory-constrained environments

## Conclusion

Realtime_mlx_STT provides a high-performance, easy-to-use solution for speech-to-text transcription optimized for Apple Silicon. Its modular architecture, comprehensive API, and extensive configuration options make it suitable for a wide range of applications from simple command-line tools to complex interactive applications requiring real-time transcription.