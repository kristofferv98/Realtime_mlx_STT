#!/usr/bin/env python3
"""
MLX Array Direct Test

This test verifies that MLX arrays can be used directly with the transcription
functions, ensuring proper handling of different array types.
"""

import os
import sys
import logging
import numpy as np
import mlx.core as mx
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('mlx_array_test')

# Add parent directory to path to import from RealtimeSTT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RealtimeSTT.whisper_turbo import transcribe
from RealtimeSTT.mlx_transcriber import MLXTranscriber

def test_mlx_array_transcribe():
    """Test direct MLX array transcription with whisper_turbo.transcribe"""
    logger.info("Testing direct MLX array transcription with whisper_turbo.transcribe")
    
    # Create a simple test audio (1 second of silence)
    # We use zeros which should typically result in minimal or empty transcription
    sample_rate = 16000
    audio_np = np.zeros(sample_rate, dtype=np.float32)
    audio_mx = mx.array(audio_np)
    
    start_time = time.time()
    
    # Try direct transcription with MLX array
    logger.info('Attempting direct transcription with MLX array...')
    result = transcribe(audio_input=audio_mx, any_lang=False, quick=True, language='en')
    
    elapsed = time.time() - start_time
    logger.info(f'Transcription complete in {elapsed:.2f}s')
    logger.info(f'Transcription result: "{result}"')
    
    return result is not None

def test_mlx_array_with_mlxtranscriber():
    """Test MLX array transcription with MLXTranscriber"""
    logger.info("Testing MLX array transcription with MLXTranscriber")
    
    # Create a simple test audio (1 second of silence)
    sample_rate = 16000
    audio_np = np.zeros(sample_rate, dtype=np.float32)
    audio_mx = mx.array(audio_np)
    
    # Initialize the transcriber
    transcriber = MLXTranscriber(
        model_path='openai/whisper-large-v3-turbo',
        realtime_mode=False,
        any_lang=False,
        quick=True,
        language='en'
    )
    
    # Start the transcription worker
    transcriber.start()
    
    # Force model loading
    while transcriber.transcriber is None:
        time.sleep(0.1)
        logger.info('Waiting for model initialization...')
    
    start_time = time.time()
    
    # Test with MLX array
    logger.info('Processing with MLX array...')
    result = transcriber._process_audio(audio_mx)
    
    elapsed = time.time() - start_time
    logger.info(f'MLX array processing complete in {elapsed:.2f}s')
    logger.info(f'MLX array result: "{result}"')
    
    # Stop the worker
    transcriber.stop()
    
    return result is not None

def main():
    """Run all tests"""
    try:
        # Test with direct transcribe function
        success1 = test_mlx_array_transcribe()
        
        # Test with MLXTranscriber
        success2 = test_mlx_array_with_mlxtranscriber()
        
        # Overall test result
        if success1 and success2:
            logger.info("All tests completed successfully!")
            return 0
        else:
            logger.error("Some tests failed.")
            return 1
            
    except Exception as e:
        logger.exception(f"Error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())