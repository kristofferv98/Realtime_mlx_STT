#!/usr/bin/env python3
"""
VAD-Triggered Transcription Example

This example demonstrates real-time speech transcription using Voice Activity Detection (VAD).
It continuously listens for speech, detects when you start and stop speaking,
and transcribes each speech segment automatically.

Features:
- Automatic speech detection using VAD
- Real-time transcription when speech ends
- Continuous listening until you press Ctrl+C

Usage:
    python vad_transcription.py [--device DEVICE_ID] [--language LANGUAGE_CODE]
    
    Arguments:
        --device, -d    Audio device index (default: system default)
        --language, -l  Language code like 'en', 'es', 'fr' (default: auto-detect)
    
    Examples:
        python vad_transcription.py
        python vad_transcription.py --device 1 --language en
"""

import os
import sys
import time
import signal
import argparse
from typing import Optional

# Add project root to path
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


class VadTranscriptionApp:
    """Simple VAD-triggered transcription application."""
    
    def __init__(self, device_index: Optional[int] = None, language: Optional[str] = None):
        self.device_index = device_index
        self.language = language
        self.is_running = False
        self.transcription_count = 0
        
        # Initialize components
        self.command_dispatcher = CommandDispatcher()
        self.event_bus = EventBus()
        
    def setup(self):
        """Set up all modules and event handlers."""
        # Register modules
        AudioCaptureModule.register(self.command_dispatcher, self.event_bus)
        VadModule.register(self.command_dispatcher, self.event_bus, processing_enabled=True)
        self.transcription_handler = TranscriptionModule.register(
            self.command_dispatcher, 
            self.event_bus,
            default_language=self.language
        )
        
        # Configure VAD for good speech detection
        VadModule.configure_vad(
            self.command_dispatcher,
            detector_type="combined",
            sensitivity=0.6,
            min_speech_duration=0.25
        )
        
        # Set up event handlers
        def on_transcription_updated(session_id, text, is_final, confidence):
            if is_final and text.strip():
                self.transcription_count += 1
                print(f"\n[{self.transcription_count}] Transcription: {text}")
                print(f"    Confidence: {confidence:.2f}\n")
        
        def on_speech_detected(confidence, timestamp, speech_id):
            print("🎤 Listening...", end='', flush=True)
        
        def on_silence_detected(duration, start_time, end_time, speech_id):
            print(f" (spoke for {duration:.1f}s)")
        
        # Register event handlers
        TranscriptionModule.on_transcription_updated(self.event_bus, on_transcription_updated)
        VadModule.on_speech_detected(self.event_bus, on_speech_detected)
        VadModule.on_silence_detected(self.event_bus, on_silence_detected)
        
        # Set up VAD-triggered transcription
        TranscriptionModule.register_vad_integration(
            self.event_bus,
            self.transcription_handler,
            auto_start_on_speech=True
        )
        
    def start(self):
        """Start recording and processing."""
        print("\nVAD-Triggered Transcription")
        print("=" * 50)
        
        # List available devices if none specified
        if self.device_index is None:
            devices = AudioCaptureModule.list_devices(self.command_dispatcher)
            if devices and isinstance(devices[0], list):
                devices = devices[0]
            
            print("\nAvailable audio devices:")
            for device in devices:
                device_id = device.get('device_id', device.get('index', 0))
                device_name = device.get('name', 'Unknown')
                if device.get('is_default'):
                    print(f"  [{device_id}] {device_name} (default)")
                    self.device_index = device_id
                else:
                    print(f"  [{device_id}] {device_name}")
        
        print(f"\nUsing device: {self.device_index}")
        print(f"Language: {self.language or 'auto-detect'}")
        print("\nStarting... Speak and I'll transcribe when you pause.")
        print("Press Ctrl+C to stop.\n")
        
        # Start recording
        self.is_running = True
        AudioCaptureModule.start_recording(
            self.command_dispatcher,
            device_id=self.device_index,
            sample_rate=16000,
            chunk_size=512
        )
        
    def stop(self):
        """Stop recording and clean up."""
        print("\n\nStopping...")
        AudioCaptureModule.stop_recording(self.command_dispatcher)
        self.is_running = False
        print(f"Total transcriptions: {self.transcription_count}")
        

def main():
    """Run the VAD transcription example."""
    parser = argparse.ArgumentParser(
        description="Real-time speech transcription with VAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--device', '-d', type=int, help='Audio device index')
    parser.add_argument('--language', '-l', type=str, help='Language code (e.g., en, es, fr)')
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
    
    # Create and run the app
    app = VadTranscriptionApp(device_index=args.device, language=args.language)
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