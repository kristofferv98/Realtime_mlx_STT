#!/usr/bin/env python3
"""
MLX Transcriber Test

This script tests the MLX-optimized transcriber for the Realtime_mlx_STT library.
It tests basic functionality of the MLXTranscriber class with a sample audio file.
"""

import os
import sys
import time
import numpy as np
import argparse
import logging

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RealtimeSTT.mlx_transcriber import MLXTranscriber

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlx_transcriber_test")

def test_basic_transcription(audio_file, any_lang=False, quick=True):
    """
    Test basic functionality of the MLX transcriber.
    
    Args:
        audio_file (str): Path to an audio file for testing
        any_lang (bool): Whether to use language detection
        quick (bool): Whether to use quick (parallel) processing
    """
    logger.info(f"Testing MLX transcriber with file: {audio_file}")
    
    # Create the transcriber
    transcriber = MLXTranscriber(
        realtime_mode=False,
        any_lang=any_lang,
        quick=quick
    )
    
    # Start the transcriber
    transcriber.start()
    
    try:
        # Load the audio file
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Submit for transcription
        start_time = time.time()
        transcriber.transcribe(audio_data)
        
        # Wait for and retrieve the result
        result = None
        max_wait = 30  # Maximum wait time in seconds
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait:
            result = transcriber.get_result(timeout=0.5)
            if result:
                break
            logger.info("Waiting for transcription result...")
        
        # Calculate and display metrics
        total_time = time.time() - start_time
        audio_duration = len(audio_data) / sample_rate
        
        if result:
            logger.info(f"Transcription completed in {total_time:.2f} seconds")
            logger.info(f"Audio duration: {audio_duration:.2f} seconds")
            logger.info(f"Realtime factor: {total_time / audio_duration:.2f}x")
            logger.info(f"Result: {result['text']}")
            return result['text']
        else:
            logger.error("No transcription result received within timeout")
            return None
            
    finally:
        # Clean up
        transcriber.stop()
        transcriber.cleanup()

def test_streaming_transcription(audio_file, chunk_size=4000, any_lang=False, quick=True):
    """
    Test streaming transcription with the MLX transcriber.
    
    Args:
        audio_file (str): Path to an audio file for testing
        chunk_size (int): Size of audio chunks in samples
        any_lang (bool): Whether to use language detection
        quick (bool): Whether to use quick (parallel) processing
    """
    logger.info(f"Testing MLX streaming transcription with file: {audio_file}")
    
    # Create the transcriber
    transcriber = MLXTranscriber(
        realtime_mode=True,
        any_lang=any_lang,
        quick=quick
    )
    
    # Start the transcriber
    transcriber.start()
    
    try:
        # Load the audio file
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Process audio in chunks to simulate streaming
        start_time = time.time()
        total_chunks = len(audio_data) // chunk_size + 1
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            is_last = (i + chunk_size >= len(audio_data))
            
            logger.info(f"Processing chunk {i // chunk_size + 1}/{total_chunks}" +
                       f" ({len(chunk)} samples, {'last' if is_last else 'not last'})")
            
            # Add the chunk to the transcriber
            transcriber.add_audio_chunk(chunk, is_last=is_last)
            
            # Check for intermediate results
            result = transcriber.get_result(timeout=0.1)
            if result:
                logger.info(f"Intermediate result: {result['text'][:50]}...")
            
            # Small delay to simulate real-time processing
            time.sleep(0.1)
        
        # Wait for and retrieve the final result
        result = None
        max_wait = 30  # Maximum wait time in seconds
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait:
            result = transcriber.get_result(timeout=0.5)
            if result:
                break
            logger.info("Waiting for final transcription result...")
        
        # Calculate and display metrics
        total_time = time.time() - start_time
        audio_duration = len(audio_data) / sample_rate
        
        if result:
            logger.info(f"Streaming transcription completed in {total_time:.2f} seconds")
            logger.info(f"Audio duration: {audio_duration:.2f} seconds")
            logger.info(f"Realtime factor: {total_time / audio_duration:.2f}x")
            logger.info(f"Final result: {result['text']}")
            return result['text']
        else:
            logger.error("No final transcription result received within timeout")
            return None
            
    finally:
        # Clean up
        transcriber.stop()
        transcriber.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Test the MLX Transcriber")
    parser.add_argument("--audio", type=str, default="../RealtimeSTT/warmup_audio.wav",
                        help="Path to an audio file for testing")
    parser.add_argument("--streaming", action="store_true",
                        help="Test streaming transcription")
    parser.add_argument("--any-lang", action="store_true",
                        help="Use language detection")
    parser.add_argument("--no-quick", action="store_true",
                        help="Disable quick (parallel) processing")
    
    args = parser.parse_args()
    
    audio_file = args.audio
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    logger.info("Starting MLX transcriber test")
    
    if args.streaming:
        test_streaming_transcription(
            audio_file,
            any_lang=args.any_lang,
            quick=not args.no_quick
        )
    else:
        test_basic_transcription(
            audio_file,
            any_lang=args.any_lang,
            quick=not args.no_quick
        )

if __name__ == "__main__":
    main()