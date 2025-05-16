#!/usr/bin/env python3
"""
Example of using the refactored Transcription feature to transcribe an audio file.

This script demonstrates how to use the DirectMlxWhisperEngine to transcribe 
an audio file without process isolation.
"""

import os
import sys
import time
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus

# Feature imports
from src.Features.Transcription.TranscriptionModule import TranscriptionModule


def transcribe_file(file_path, language=None, quick_mode=True, beam_size=1, output_file=None):
    """
    Transcribe an audio file using the DirectMlxWhisperEngine.
    
    Args:
        file_path: Path to the audio file to transcribe
        language: Language code (e.g., 'en', 'no') or None for auto-detection
        quick_mode: Whether to use quick/parallel mode for faster processing
        beam_size: Beam search size (larger = more accurate but slower)
        output_file: Path to save the transcription to (if None, uses input filename + _transcription.txt)
        
    Returns:
        dict: Transcription result
    """
    # Initialize components
    command_dispatcher = CommandDispatcher()
    event_bus = EventBus()
    
    # Track transcription progress
    progress = {
        "updates": 0,
        "latest_text": "",
        "latest_is_final": False
    }
    
    # Register the Transcription feature
    transcription_handler = TranscriptionModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        default_engine="mlx_whisper",
        default_model="whisper-large-v3-turbo"
    )
    
    # Subscribe to transcription events
    def on_transcription_updated(session_id, text, is_final, confidence):
        progress["updates"] += 1
        progress["latest_text"] = text
        progress["latest_is_final"] = is_final
        
        status = "FINAL" if is_final else "partial"
        logger.info(f"Transcription update ({status}): {text[:50]}..." if len(text) > 50 else text)
        logger.info(f"  Confidence: {confidence:.2f}")
    
    TranscriptionModule.on_transcription_updated(event_bus, on_transcription_updated)
    
    # Configure transcription
    TranscriptionModule.configure(
        command_dispatcher=command_dispatcher,
        engine_type="mlx_whisper",
        model_name="whisper-large-v3-turbo",
        language=language,
        streaming=not quick_mode,  # Quick mode = not streaming
        beam_size=beam_size,
        options={
            "quick_mode": quick_mode
        }
    )
    
    logger.info(f"Transcribing file: {file_path}")
    logger.info(f"  Language: {language or 'auto-detect'}")
    logger.info(f"  Mode: {'quick/parallel' if quick_mode else 'recurrent/sequential'}")
    logger.info(f"  Beam size: {beam_size}")
    
    start_time = time.time()
    
    # Transcribe the file
    result = TranscriptionModule.transcribe_file(
        command_dispatcher=command_dispatcher,
        file_path=file_path,
        language=language,
        options={"quick_mode": quick_mode},
        beam_size=beam_size
    )
    
    processing_time = time.time() - start_time
    audio_duration = progress.get("audio_duration", 0)
    
    # Check if result is a list (which happens sometimes) and get the first item
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    
    # Log results
    if isinstance(result, dict) and 'error' in result:
        logger.error(f"Transcription error: {result['error']}")
        return result
    
    logger.info(f"Transcription complete in {processing_time:.2f} seconds")
    logger.info(f"  Updates received: {progress['updates']}")
    logger.info(f"  Final text length: {len(result['text'])} characters")
    
    if audio_duration > 0:
        realtime_factor = processing_time / audio_duration
        logger.info(f"  Processing speed: {realtime_factor:.2f}x real-time")
    
    # Save output if requested
    if output_file is None:
        output_file = os.path.splitext(file_path)[0] + "_transcription.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result['text'])
    
    logger.info(f"Transcription saved to: {output_file}")
    
    return result


def main():
    """Parse arguments and run transcription."""
    parser = argparse.ArgumentParser(description="Transcribe an audio file using MLX Whisper")
    parser.add_argument("file_path", help="Path to the audio file to transcribe")
    parser.add_argument("--language", "-l", default=None, 
                        help="Language code (e.g., 'en', 'no'). Default: auto-detect")
    parser.add_argument("--output", "-o", help="Path to save transcription to")
    parser.add_argument("--quick", "-q", action="store_true", default=True, 
                        help="Use quick/parallel mode (faster). Default: enabled")
    parser.add_argument("--beam-size", "-b", type=int, default=1, 
                        help="Beam search size (default: 1)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return 1
    
    try:
        result = transcribe_file(
            file_path=args.file_path,
            language=args.language,
            quick_mode=args.quick,
            beam_size=args.beam_size,
            output_file=args.output
        )
        
        # Output a short summary
        print("\nTranscription summary:")
        print("=====================")
        if isinstance(result, dict) and 'text' in result:
            print(f"First 300 characters:")
            print(result['text'][:300] + ('...' if len(result['text']) > 300 else ''))
        else:
            print("Error: Failed to get transcription result")
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())