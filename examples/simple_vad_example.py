#!/usr/bin/env python3
"""
Basic Transcription Example

This is the simplest working example of using Realtime_mlx_STT.
It uses VAD to detect speech and automatically transcribes when you stop speaking.

Usage:
    python basic_transcription.py
    
Note: This example uses the built-in VAD integration for simplicity.
      Speech is automatically detected and transcribed when you pause.
"""

import os
import sys
import time
import signal
import threading
import argparse
import logging

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Suppress progress bars for cleaner output
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus

# Feature imports
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule
from src.Features.VoiceActivityDetection.VadModule import VadModule
from src.Features.Transcription.TranscriptionModule import TranscriptionModule


def main():
    """Basic transcription example using VAD."""
    parser = argparse.ArgumentParser(description="Simple VAD transcription example")
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed logs')
    args = parser.parse_args()
    
    # Configure logging based on verbose flag
    if not args.verbose:
        logging.getLogger().setLevel(logging.ERROR)
        # Also set specific loggers to ERROR
        for logger_name in ['realtimestt', 'src']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        # Suppress warnings
        import warnings
        warnings.filterwarnings('ignore')
    
    print("Basic Transcription Example")
    print("-" * 50)
    print("This example uses VAD to automatically detect and transcribe speech.")
    print("Speak clearly and pause when done. Press Ctrl+C to exit.\n")
    
    # Initialize components
    command_dispatcher = CommandDispatcher()
    event_bus = EventBus()
    
    # Register modules (register returns the handler for transcription)
    AudioCaptureModule.register(command_dispatcher, event_bus)
    VadModule.register(command_dispatcher, event_bus, processing_enabled=True)
    transcription_handler = TranscriptionModule.register(command_dispatcher, event_bus)
    
    # Track transcriptions
    transcription_count = 0
    
    # Set up event handlers
    def on_transcription_updated(session_id, text, is_final, confidence):
        nonlocal transcription_count
        if is_final and text.strip():
            transcription_count += 1
            print(f"\n[{transcription_count}] Transcription: {text}")
            print(f"    Confidence: {confidence:.2f}")
    
    def on_speech_detected(confidence, timestamp, speech_id):
        print("\n🎤 Listening...", end='', flush=True)
    
    def on_silence_detected(duration, start_time, end_time, speech_id):
        print(f" (duration: {duration:.1f}s)")
    
    # Register event handlers
    TranscriptionModule.on_transcription_updated(event_bus, on_transcription_updated)
    VadModule.on_speech_detected(event_bus, on_speech_detected)
    VadModule.on_silence_detected(event_bus, on_silence_detected)
    
    # Set up VAD-triggered transcription
    # This is the key - it automatically connects VAD to transcription
    TranscriptionModule.register_vad_integration(
        event_bus=event_bus,
        transcription_handler=transcription_handler,  # Use the handler, not dispatcher
        auto_start_on_speech=True
    )
    
    # Configure VAD with reasonable defaults
    VadModule.configure_vad(
        command_dispatcher=command_dispatcher,
        detector_type="combined",
        sensitivity=0.6,
        min_speech_duration=0.25
    )
    
    # Start recording
    print("Starting audio capture...")
    AudioCaptureModule.start_recording(
        command_dispatcher=command_dispatcher,
        sample_rate=16000,
        chunk_size=512
    )
    
    print("Ready! Start speaking...\n")
    
    # Keep running until Ctrl+C
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n\nStopping...")
        
        # Stop recording
        try:
            AudioCaptureModule.stop_recording(command_dispatcher)
        except:
            pass
            
        print(f"Total transcriptions: {transcription_count}")
        print("Done!")
        
        # Force exit after a short delay
        def force_exit():
            time.sleep(1)
            os._exit(0)
        
        threading.Thread(target=force_exit, daemon=True).start()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()