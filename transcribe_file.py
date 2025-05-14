#!/usr/bin/env python3
"""
File Transcription Test

This script transcribes an audio file using the MLX-optimized whisper-large-v3-turbo model.
"""

import os
import sys
import time
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('transcribe_file')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Parse arguments
parser = argparse.ArgumentParser(description="Transcribe an audio file using MLX-optimized Whisper")
parser.add_argument('--file', type=str, default="/Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/bok_konge01.mp3",
                    help="Path to audio file")
parser.add_argument('--language', type=str, default="",
                    help="Language code (empty for auto-detection)")
parser.add_argument('--quick', action='store_true', default=True,
                    help="Use quick mode for faster transcription")
args = parser.parse_args()

def transcribe_with_direct_api(file_path, language="", quick=True):
    """Transcribe using the direct transcribe function from whisper_turbo"""
    from RealtimeSTT.whisper_turbo import transcribe
    
    logger.info(f"Transcribing with direct API: {file_path}")
    logger.info(f"Parameters: language='{language}', quick={quick}")
    
    start_time = time.time()
    result = transcribe(
        audio_input=file_path,
        any_lang=not language,
        quick=quick,
        language=language if language else None
    )
    elapsed = time.time() - start_time
    
    logger.info(f"Transcription completed in {elapsed:.2f} seconds")
    return result, elapsed

def transcribe_with_mlxtranscriber(file_path, language="", quick=True):
    """Transcribe using the MLXTranscriber class"""
    import soundfile as sf
    import numpy as np
    from RealtimeSTT.mlx_transcriber import MLXTranscriber
    
    logger.info(f"Transcribing with MLXTranscriber: {file_path}")
    logger.info(f"Parameters: language='{language}', quick={quick}")
    
    # Load audio file
    logger.info("Loading audio file...")
    try:
        audio, sample_rate = sf.read(file_path)
    except Exception as e:
        # If soundfile fails, try using librosa
        import librosa
        logger.info("Using librosa for audio loading...")
        audio, sample_rate = librosa.load(file_path, sr=16000)
    
    # Convert to mono if needed
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    
    # Ensure correct format
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    max_val = np.max(np.abs(audio))
    if max_val > 0 and max_val > 1.0:
        audio = audio / max_val
    
    # Create transcriber
    transcriber = MLXTranscriber(
        model_path="openai/whisper-large-v3-turbo",
        realtime_mode=False,
        any_lang=not language,
        quick=quick,
        language=language if language else None
    )
    
    # Start the transcriber
    transcriber.start()
    
    # Submit audio for transcription
    start_time = time.time()
    transcriber.transcribe(audio)
    
    # Wait for result
    result = None
    max_wait = 60  # seconds
    wait_start = time.time()
    
    while time.time() - wait_start < max_wait:
        result = transcriber.get_result(timeout=0.5)
        if result:
            break
        logger.info("Waiting for transcription result...")
    
    elapsed = time.time() - start_time
    
    # Stop the transcriber
    transcriber.stop()
    
    if result:
        logger.info(f"Transcription completed in {elapsed:.2f} seconds")
        return result['text'], elapsed
    else:
        logger.error("Transcription failed: timeout")
        return None, elapsed

def main():
    """Run transcription using both methods"""
    file_path = args.file
    language = args.language
    quick = args.quick
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 1
    
    logger.info(f"Processing file: {file_path}")
    logger.info(f"Language: {language or 'auto-detect'}")
    logger.info(f"Quick mode: {quick}")
    
    try:
        # Method 1: Direct API
        logger.info("\n=== Method 1: Direct whisper_turbo.transcribe API ===")
        direct_result, direct_time = transcribe_with_direct_api(file_path, language, quick)
        
        # Method 2: MLXTranscriber
        logger.info("\n=== Method 2: MLXTranscriber ===")
        mlxt_result, mlxt_time = transcribe_with_mlxtranscriber(file_path, language, quick)
        
        # Print results
        logger.info("\n=== Results ===")
        logger.info(f"Direct API time: {direct_time:.2f} seconds")
        logger.info(f"MLXTranscriber time: {mlxt_time:.2f} seconds")
        
        logger.info("\n=== Transcription Direct API ===")
        logger.info(direct_result)
        
        logger.info("\n=== Transcription MLXTranscriber ===")
        logger.info(mlxt_result)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error during transcription: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())