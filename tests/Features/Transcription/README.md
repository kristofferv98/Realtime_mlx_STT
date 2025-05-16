# Transcription Feature Tests - COMPLETED

This directory contains tests for the Transcription feature, including a real-world test that uses the actual MLX-optimized Whisper model. All tests have been implemented and are passing.

## Test Status

- ✅ `simple_test.py`: Tests basic imports and initialization of the Transcription feature
- ✅ `transcribe_real_file_test.py`: Tests end-to-end functionality with a real audio file (using a mock engine)
- ✅ `real_world_transcription_test.py`: Tests with the actual MLX Whisper model (resource intensive)
- ✅ All commands properly tested
- ✅ Test runs to completion without errors

## Hardware Requirements

The real-world test `real_world_transcription_test.py` runs the actual MLX-optimized Whisper model and requires:

- Apple Silicon hardware (M1, M2 or M3 chip)
- macOS environment
- At least 8GB RAM (16GB recommended)
- MLX framework properly installed
- Internet connection for model downloading (first run only)

## Test Files

### simple_test.py

A basic test that verifies all imports and initialization of the Transcription feature.

**Usage:**
```bash
python -m tests.Features.Transcription.simple_test
```

### transcribe_real_file_test.py

A standalone test that transcribes a real audio file to verify end-to-end functionality of the Transcription feature.
This test runs with mocked engine components and is designed to test the integration and flow, not the actual transcription.

**Usage:**
```bash
# Run with default settings
python -m tests.Features.Transcription.transcribe_real_file_test

# Run with custom settings
python -m tests.Features.Transcription.transcribe_real_file_test \
  --audio-file /path/to/audio.mp3 \
  --output-path /path/to/output.txt \
  --language en \
  --quiet
```

**Arguments:**
- `--audio-file` - Path to the audio file to transcribe (default: bok_konge01.mp3 in project root)
- `--output-path` - Path to save the transcription result (default: next to the audio file)
- `--language` - Language code for transcription (default: 'no' for Norwegian)
- `--quiet` - Run without console output

**Outputs:**
- Plain text transcription file
- JSON file with detailed results and performance metrics
- Log file with execution details

### real_world_transcription_test.py

A comprehensive test that uses the actual MLX-optimized Whisper model to transcribe audio. This test is designed for systems with Apple Silicon hardware and sufficient resources.

**Usage:**
```bash
# Run with default settings
python -m tests.Features.Transcription.real_world_transcription_test

# Run with custom settings
python -m tests.Features.Transcription.real_world_transcription_test \
  --audio-file /path/to/audio.mp3 \
  --output-path /path/to/output.txt \
  --language en \
  --model whisper-large-v3-turbo \
  --compute-type float16 \
  --beam-size 1 \
  --quiet
```

**Arguments:**
- `--audio-file` - Path to the audio file to transcribe (default: bok_konge01.mp3)
- `--output-path` - Path to save the transcription result (default: next to the audio file)
- `--language` - Language code or None for auto-detection (default: None)
- `--model` - Whisper model to use (default: whisper-large-v3-turbo)
- `--compute-type` - Computation precision: float16 or float32 (default: float16)
- `--beam-size` - Beam search size for inference (default: 1)
- `--quiet` - Run without console output

**Outputs:**
- Plain text transcription file with "_real" suffix
- JSON file with detailed results and performance metrics
- Log file with execution details

## Running the Tests

To run all Transcription tests, use the combined test runner:

```bash
# Navigate to the project root
cd /path/to/Realtime_mlx_STT

# Run all tests
python -m tests.Features.Transcription.run_all_tests

# Skip the resource-intensive real-world test
python -m tests.Features.Transcription.run_all_tests --skip-real-world

# Run with specific options
python -m tests.Features.Transcription.run_all_tests \
  --audio-file /path/to/audio.mp3 \
  --language en \
  --model whisper-large-v3-turbo \
  --compute-type float16 \
  --beam-size 2 \
  --quiet
```

**Test Runner Arguments:**
- All arguments from the individual tests
- `--skip-real-file` - Skip the real file test
- `--skip-real-world` - Skip the resource-intensive real-world test with actual MLX model

## Test Requirements

These tests require:
- MLX framework installed
- LibROSA for audio file handling
- NumPy for audio data manipulation
- An actual audio file to test with (default: bok_konge01.mp3)

## Expected Results

The test should successfully:
1. Load the audio file
2. Process it through the Transcription feature
3. Generate a text transcription
4. Save the results to files
5. Report performance metrics

## Troubleshooting

If tests fail with:
- **Audio loading errors**: Ensure LibROSA is installed and the audio file is valid
- **Transcription errors**: Check that MLX is properly installed and configured
- **Memory errors**: MLX requires significant memory, ensure your system has enough available