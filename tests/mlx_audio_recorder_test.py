#!/usr/bin/env python3
"""
MLX AudioToTextRecorder Integration Test

This script tests the integration of MLX backend with AudioToTextRecorder.
It tests both batch and streaming transcription modes.
"""

import os
import sys
import time
import argparse
import logging
import soundfile as sf
import numpy as np

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RealtimeSTT import AudioToTextRecorder

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mlx_audio_recorder_test")

def test_batch_transcription(audio_file):
    """
    Test batch transcription using AudioToTextRecorder.
    
    Args:
        audio_file (str): Path to an audio file for testing
    """
    logger.info(f"Testing batch transcription using file: {audio_file}")
    
    # Load audio file
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
    
    # Create recorder
    recorder = AudioToTextRecorder(
        model_path="openai/whisper-large-v3-turbo",
        quick_mode=True,
        enable_realtime_transcription=False,
        use_vad=False,  # Disable VAD for direct testing
        spinner=False,
        debug_mode=True
    )
    
    try:
        # Manually set the audio buffer
        recorder.audio = audio_data
        
        # Perform transcription
        start_time = time.time()
        result = recorder.text()
        total_time = time.time() - start_time
        
        # Display results
        logger.info(f"Transcription completed in {total_time:.2f} seconds")
        logger.info(f"Audio duration: {len(audio_data) / sample_rate:.2f} seconds")
        logger.info(f"Realtime factor: {total_time / (len(audio_data) / sample_rate):.2f}x")
        logger.info(f"Result: {result}")
        
        return result
    finally:
        # Clean up
        recorder.shutdown()

def test_streaming_transcription(audio_file, chunk_size=4000):
    """
    Test streaming transcription using AudioToTextRecorder.
    
    Args:
        audio_file (str): Path to an audio file for testing
        chunk_size (int): Size of audio chunks in samples
    """
    logger.info(f"Testing streaming transcription using file: {audio_file}")
    
    # Track streaming updates
    streaming_results = []
    
    def on_realtime_update(text):
        logger.info(f"Realtime update: {text[:50]}..." if len(text) > 50 else text)
        streaming_results.append(text)
    
    # Create recorder
    recorder = AudioToTextRecorder(
        model_path="openai/whisper-large-v3-turbo",
        quick_mode=True,
        enable_realtime_transcription=True,
        use_vad=False,  # Disable VAD for direct testing
        spinner=False,
        on_realtime_transcription_update=on_realtime_update,
        debug_mode=True
    )
    
    try:
        # Load audio file
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
        
        # Start recorder
        recorder.start()
        
        # Manually feed audio chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            is_last = (i + chunk_size >= len(audio_data))
            
            # Set frames directly (bypassing the normal audio capture flow)
            with recorder.frame_lock:
                recorder.frames = chunk
                recorder.recording = True
            
            # Allow time for processing
            time.sleep(0.5)
            
            if is_last:
                # For the last chunk, wait a bit longer
                time.sleep(1.0)
        
        # Get final result
        final_result = recorder.text()
        
        # Display results
        logger.info(f"Final result: {final_result}")
        logger.info(f"Received {len(streaming_results)} streaming updates")
        
        return final_result, streaming_results
    finally:
        # Clean up
        recorder.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Test AudioToTextRecorder")
    parser.add_argument("--audio", type=str, default="../RealtimeSTT/warmup_audio.wav",
                        help="Path to an audio file for testing")
    parser.add_argument("--streaming", action="store_true",
                        help="Test streaming transcription")
    parser.add_argument("--chunk-size", type=int, default=4000,
                        help="Size of audio chunks for streaming test")
    
    args = parser.parse_args()
    
    audio_file = args.audio
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    logger.info("Starting AudioToTextRecorder integration test")
    
    if args.streaming:
        test_streaming_transcription(
            audio_file,
            chunk_size=args.chunk_size
        )
    else:
        test_batch_transcription(audio_file)

if __name__ == "__main__":
    main()