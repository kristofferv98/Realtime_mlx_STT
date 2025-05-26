#!/usr/bin/env python3
"""
Wake Word Detection Example

This example demonstrates wake word detection to trigger transcription.
The system listens for a wake word (default: "porcupine") and only
transcribes speech that follows the wake word.

Features:
- Continuous wake word detection
- Automatic timeout after wake word (30 seconds default)
- Transcribes only after wake word is spoken
- Returns to listening mode after timeout or silence

Usage:
    python wake_word_example.py [--wake-word WORD] [--timeout SECONDS]
    
    Arguments:
        --wake-word, -w  Wake word to listen for (default: porcupine)
        --timeout, -t    Seconds to listen after wake word (default: 30)
    
    Examples:
        python wake_word_example.py
        python wake_word_example.py --wake-word jarvis --timeout 60

Note: Requires PORCUPINE_ACCESS_KEY environment variable to be set.
Get your free key at: https://picovoice.ai/
"""

import os
import sys
import time
import signal
import argparse
from typing import Optional

# Add project root to path first

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Suppress progress bars for cleaner output
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import logging

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus

# Feature imports
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule
from src.Features.VoiceActivityDetection.VadModule import VadModule
from src.Features.Transcription.TranscriptionModule import TranscriptionModule
from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule


class WakeWordApp:
    """Wake word triggered transcription application."""
    
    def __init__(self, wake_word: str = "porcupine", timeout: int = 30):
        self.wake_word = wake_word
        self.timeout = timeout
        self.is_running = False
        
        # Initialize components
        self.command_dispatcher = CommandDispatcher()
        self.event_bus = EventBus()
        
    def setup(self):
        """Set up all modules and event handlers."""
        # Check for access key
        if not os.environ.get('PORCUPINE_ACCESS_KEY'):
            print("\n❌ ERROR: PORCUPINE_ACCESS_KEY environment variable not set!")
            print("Get your free key at: https://picovoice.ai/")
            print("Then run: export PORCUPINE_ACCESS_KEY='your-key-here'\n")
            sys.exit(1)
        
        # Register modules
        AudioCaptureModule.register(self.command_dispatcher, self.event_bus)
        VadModule.register(self.command_dispatcher, self.event_bus, processing_enabled=False)
        self.transcription_handler = TranscriptionModule.register(self.command_dispatcher, self.event_bus)
        WakeWordModule.register(self.command_dispatcher, self.event_bus)
        
        # Configure wake word detection
        WakeWordModule.configure(
            self.command_dispatcher,
            config={
                'wake_words': [self.wake_word],
                'speech_timeout': self.timeout,
                'sensitivities': [0.5]  # Default sensitivity
            }
        )
        
        # Set up event handlers
        def on_wake_word_detected(wake_word, confidence, timestamp):
            print(f"\n✨ Wake word '{wake_word}' detected! (confidence: {confidence:.2f}) Listening for {self.timeout}s...")
            # Enable VAD after wake word
            VadModule.enable_processing(self.command_dispatcher)
        
        def on_wake_word_timeout(wake_word, timeout_duration):
            print(f"💤 Timeout after {timeout_duration}s - returning to wake word detection mode")
            # Disable VAD and return to wake word detection
            VadModule.disable_processing(self.command_dispatcher)
        
        def on_transcription_updated(session_id, text, is_final, confidence):
            if is_final and text.strip():
                print(f"\n📝 Transcription: {text}")
                print(f"   Confidence: {confidence:.2f}")
        
        def on_speech_detected(confidence, timestamp, speech_id):
            print("🎤 Speaking...", end='', flush=True)
        
        def on_silence_detected(duration, start_time, end_time, speech_id):
            print(f" (duration: {duration:.1f}s)")
        
        # Register event handlers
        WakeWordModule.on_wake_word_detected(self.event_bus, on_wake_word_detected)
        WakeWordModule.on_wake_word_timeout(self.event_bus, on_wake_word_timeout)
        TranscriptionModule.on_transcription_updated(self.event_bus, on_transcription_updated)
        VadModule.on_speech_detected(self.event_bus, on_speech_detected)
        VadModule.on_silence_detected(self.event_bus, on_silence_detected)
        
        # Set up VAD-triggered transcription (but VAD starts disabled)
        TranscriptionModule.register_vad_integration(
            self.event_bus,
            self.transcription_handler,
            auto_start_on_speech=True
        )
        
    def start(self):
        """Start the wake word detection system."""
        print("\nWake Word Detection Example")
        print("=" * 50)
        print(f"Wake word: '{self.wake_word}'")
        print(f"Timeout: {self.timeout} seconds")
        print(f"\nSay '{self.wake_word}' to start listening...")
        print("Press Ctrl+C to stop.\n")
        
        # Start audio recording
        self.is_running = True
        AudioCaptureModule.start_recording(
            self.command_dispatcher,
            sample_rate=16000,
            chunk_size=512
        )
        
        # Start wake word detection
        WakeWordModule.start_detection(self.command_dispatcher)
        
    def stop(self):
        """Stop all processing."""
        print("\n\nStopping...")
        WakeWordModule.stop_detection(self.command_dispatcher)
        AudioCaptureModule.stop_recording(self.command_dispatcher)
        self.is_running = False
        

def main():
    """Run the wake word example."""
    parser = argparse.ArgumentParser(
        description="Wake word triggered transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--wake-word', '-w', default='porcupine',
                        help='Wake word to detect (default: porcupine)')
    parser.add_argument('--timeout', '-t', type=int, default=30,
                        help='Seconds to listen after wake word (default: 30)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed logs')
    
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
    
    # Create and run the app
    app = WakeWordApp(wake_word=args.wake_word, timeout=args.timeout)
    app.setup()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start and run until interrupted
    app.start()
    while app.is_running:
        time.sleep(0.1)


if __name__ == "__main__":
    main()