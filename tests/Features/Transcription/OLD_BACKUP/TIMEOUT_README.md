# Transcription Test Timeout Fix

This document explains the changes made to prevent the transcription tests from running indefinitely.

## Problem

The `real_world_transcription_test.py` script could potentially run indefinitely due to:

1. Network issues during model download from HuggingFace
2. Long-running transcription processes without proper timeouts
3. Event-based completion mechanism that might never trigger if events aren't properly handled

## Changes Made

1. Added a global timeout using the SIGALRM signal handler:
   - The test now accepts a `--timeout` parameter (default: 300 seconds/5 minutes)
   - If the test takes longer than this timeout, it will automatically terminate

2. Added dynamic transcription timeout:
   - The transcription timeout is now calculated based on audio duration (10x audio length or at least 3 minutes)
   - This accounts for longer audio files needing more processing time

3. Improved the test runner:
   - `run_all_tests.py` now has a `--test-timeout` parameter 
   - Uses `subprocess.Popen` with a timeout instead of `subprocess.run`
   - Will forcibly terminate tests that run too long

## Usage

### Running Individual Tests

```bash
# Run with default 5-minute timeout
python -m tests.Features.Transcription.real_world_transcription_test

# Run with custom 10-minute timeout
python -m tests.Features.Transcription.real_world_transcription_test --timeout 600
```

### Running All Tests

```bash
# Run all tests with default timeouts
python -m tests.Features.Transcription.run_all_tests

# Run with custom 10-minute timeout for real-world test
python -m tests.Features.Transcription.run_all_tests --test-timeout 600

# Skip the resource-intensive real-world test
python -m tests.Features.Transcription.run_all_tests --skip-real-world
```

## Troubleshooting

If the tests still timeout:

1. **Model Download Issues**: Ensure you have a good internet connection. The model is quite large (~2GB).

2. **Resource Constraints**: The MLX Whisper model requires a significant amount of memory and CPU resources. Consider:
   - Closing other applications to free up memory
   - Using a smaller model variant (e.g., `--model whisper-tiny` for testing)
   - Using a shorter audio file for testing

3. **MLX Installation**: Ensure MLX is properly installed and functioning on your system.

4. **Check Logs**: If a test times out, check the generated log files for more detailed error information.