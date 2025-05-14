#!/usr/bin/env python3
"""
Test the direct NumPy array support in whisper_turbo.py.

This script tests the modified log_mel_spectrogram function and MLXTranscriber
to ensure they correctly handle NumPy arrays without creating temporary files.
"""

import os
import sys
import time
import numpy as np
import logging
import argparse
from pprint import pprint

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mlx_direct_numpy_test')

def test_whisper_turbo_direct():
    """Test the whisper_turbo.py direct NumPy support."""
    from RealtimeSTT.whisper_turbo import log_mel_spectrogram, transcribe
    
    # Create a simple sine wave as test audio (16kHz, 1 second)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    logger.info("Testing whisper_turbo log_mel_spectrogram with NumPy array...")
    
    # Time the conversion
    start_time = time.time()
    mel = log_mel_spectrogram(audio, padding=0)
    conversion_time = time.time() - start_time
    
    logger.info(f"log_mel_spectrogram conversion time: {conversion_time:.4f}s")
    logger.info(f"Mel spectrogram shape: {mel.shape}")
    
    # Test full transcription pipeline with NumPy array
    logger.info("Testing whisper_turbo transcribe with NumPy array...")
    
    # Create a larger audio sample (2 seconds) with the word "hello"
    # This is just a test signal - in a real test, you would load actual speech
    audio_longer = np.concatenate([audio, 0.2 * np.sin(2 * np.pi * 330 * t)])
    
    start_time = time.time()
    text = transcribe(audio_longer, quick=True)
    transcription_time = time.time() - start_time
    
    logger.info(f"Transcription time: {transcription_time:.4f}s")
    logger.info(f"Transcription result: '{text}'")
    
    return {
        "conversion_time": conversion_time,
        "mel_shape": mel.shape,
        "transcription_time": transcription_time,
        "transcription_result": text
    }

def test_mlx_transcriber():
    """Test the MLXTranscriber with direct NumPy array support."""
    from RealtimeSTT.mlx_transcriber import MLXTranscriber
    
    # Create a simple sine wave as test audio (16kHz, 2 seconds)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Create a test signal - in a real test, you would load actual speech audio
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    logger.info("Creating MLXTranscriber...")
    transcriber = MLXTranscriber(realtime_mode=False, quick=True)
    
    logger.info("Starting MLXTranscriber...")
    transcriber.start()
    
    try:
        logger.info("Submitting audio for transcription...")
        start_time = time.time()
        transcriber.transcribe(audio)
        
        logger.info("Waiting for result...")
        result = None
        max_wait = 30  # seconds
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait:
            result = transcriber.get_result(timeout=0.5)
            if result:
                break
            logger.info("Waiting for transcription result...")
        
        processing_time = time.time() - start_time
        
        if result:
            logger.info(f"Transcription time: {processing_time:.4f}s")
            logger.info(f"Model processing time: {result['processing_time']:.4f}s")
            logger.info(f"Transcription result: '{result['text']}'")
            return {
                "processing_time": processing_time,
                "model_time": result['processing_time'],
                "transcription_result": result['text']
            }
        else:
            logger.error("No transcription result received within timeout")
            return {
                "error": "Timeout waiting for result"
            }
    
    finally:
        logger.info("Cleaning up MLXTranscriber...")
        transcriber.cleanup()

def test_with_real_audio(file_path):
    """Test the MLXTranscriber with a real audio file."""
    from RealtimeSTT.mlx_transcriber import MLXTranscriber
    import soundfile as sf
    
    logger.info(f"Loading audio file: {file_path}")
    try:
        audio, sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy import signal
            audio = signal.resample_poly(audio, 16000, sample_rate)
            logger.info(f"Resampled audio from {sample_rate}Hz to 16000Hz")
        
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        logger.info(f"Audio duration: {len(audio)/16000:.2f}s, shape: {audio.shape}")
        
        # Test MLXTranscriber
        logger.info("Creating MLXTranscriber...")
        transcriber = MLXTranscriber(realtime_mode=False, quick=True)
        
        logger.info("Starting MLXTranscriber...")
        transcriber.start()
        
        try:
            logger.info("Submitting audio for transcription...")
            start_time = time.time()
            transcriber.transcribe(audio)
            
            logger.info("Waiting for result...")
            result = None
            max_wait = 60  # seconds
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait:
                result = transcriber.get_result(timeout=0.5)
                if result:
                    break
                logger.info("Waiting for transcription result...")
            
            processing_time = time.time() - start_time
            
            if result:
                logger.info(f"Transcription time: {processing_time:.4f}s")
                logger.info(f"Model processing time: {result['processing_time']:.4f}s")
                logger.info(f"Transcription result: '{result['text']}'")
                return {
                    "processing_time": processing_time,
                    "model_time": result['processing_time'],
                    "transcription_result": result['text']
                }
            else:
                logger.error("No transcription result received within timeout")
                return {
                    "error": "Timeout waiting for result"
                }
        
        finally:
            logger.info("Cleaning up MLXTranscriber...")
            transcriber.cleanup()
    
    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        return {
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Test MLX-optimized Whisper with direct NumPy arrays")
    parser.add_argument("--audio", "-a", type=str, help="Path to an audio file for testing")
    parser.add_argument("--test-type", "-t", type=str, choices=["whisper", "transcriber", "both"], default="both",
                      help="Type of test to run (default: both)")
    
    args = parser.parse_args()
    
    results = {}
    
    # Run tests based on arguments
    if args.audio:
        logger.info("Running test with real audio file")
        results["real_audio"] = test_with_real_audio(args.audio)
    else:
        if args.test_type in ["whisper", "both"]:
            logger.info("Running whisper_turbo direct test")
            results["whisper_turbo"] = test_whisper_turbo_direct()
            
        if args.test_type in ["transcriber", "both"]:
            logger.info("Running MLXTranscriber test")
            results["mlx_transcriber"] = test_mlx_transcriber()
    
    # Print summary of results
    logger.info("Test Results:")
    pprint(results)

if __name__ == "__main__":
    main()