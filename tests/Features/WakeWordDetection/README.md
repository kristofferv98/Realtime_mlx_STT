# Wake Word Detection Tests

This directory contains tests for the Wake Word Detection feature.

## Test Files

- `porcupine_detector_test.py` - Tests for the Porcupine wake word detector
- `wake_word_handler_test.py` - Tests for the wake word command handler
- `run_all_tests.py` - Script to run all wake word detection tests

## Running Tests

To run all wake word detection tests:

```bash
python tests/Features/WakeWordDetection/run_all_tests.py
```

To run a specific test:

```bash
python tests/Features/WakeWordDetection/porcupine_detector_test.py
```

## Notes

- Some tests require the Porcupine access key set as an environment variable `PORCUPINE_ACCESS_KEY`
- If you don't have an access key, some tests will be skipped
- Get a free access key from [Picovoice Console](https://console.picovoice.ai/)