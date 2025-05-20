#!/usr/bin/env python3
"""
Wake word detection example.

This script demonstrates how to use the wake word detection functionality
to trigger transcription only after a wake word is detected.
"""

import os
import sys
import signal
import argparse
import time
from typing import Optional, List

from src.Core.Events.event_bus import EventBus
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule
from src.Features.VoiceActivityDetection.VadModule import VadModule
from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule
from src.Features.Transcription.TranscriptionModule import TranscriptionModule


def main():
    """Run the wake word detection example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Wake word detection example")
    parser.add_argument("--wake-words", type=str, default="porcupine",
                        help="Comma-separated list of wake words to detect")
    parser.add_argument("--sensitivity", type=float, default=0.5,
                        help="Sensitivity for wake word detection (0.0-1.0)")
    parser.add_argument("--access-key", type=str, default=None,
                        help="Porcupine access key (will check PORCUPINE_ACCESS_KEY env var if not provided)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Timeout in seconds to wait for speech after wake word detection")
    
    args = parser.parse_args()
    
    # Get wake words from command line (comma-separated)
    wake_words = [word.strip() for word in args.wake_words.split(",")]
    
    # Get access key from command line or environment
    access_key = args.access_key or os.environ.get("PORCUPINE_ACCESS_KEY")
    if not access_key:
        print("Error: Porcupine access key not provided and PORCUPINE_ACCESS_KEY environment variable not set")
        print("Get an access key from https://console.picovoice.ai/")
        return 1
    
    # Create event bus and command dispatcher
    event_bus = EventBus()
    command_dispatcher = CommandDispatcher()
    
    # Create log directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/wake_word_example.log"),
            logging.StreamHandler()
        ]
    )
    
    # Register modules
    print(f"Initializing modules...")
    audio_module = AudioCaptureModule.register(command_dispatcher, event_bus)
    vad_module = VadModule.register(command_dispatcher, event_bus)
    transcription_module = TranscriptionModule.register(command_dispatcher, event_bus)
    
    # Register wake word module with specified wake words and sensitivity
    wake_word_module = WakeWordModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        wake_words=wake_words,
        sensitivities=[args.sensitivity] * len(wake_words),
        access_key=access_key
    )
    
    # Connect the VAD system to the transcription system
    TranscriptionModule.register_vad_integration(
        event_bus=event_bus,
        transcription_handler=transcription_module,
        session_id=None,  # Generate unique session for each speech segment
        auto_start_on_speech=True
    )
    
    # Set up event handlers
    def on_wake_word_detected(wake_word, confidence, timestamp):
        """Handle wake word detection events."""
        print(f"\n🔊 Wake word detected: '{wake_word}' (confidence: {confidence:.2f})")
        print("Listening for speech... (speak now)")
    
    def on_wake_word_timeout(wake_word, timeout_duration):
        """Handle wake word timeout events."""
        print(f"\n⏱️ Timed out after {timeout_duration:.1f}s without speech")
        print(f"Listening for wake word '{args.wake_words}'...")
    
    def on_transcription_update(session_id, text, is_final, confidence):
        """Handle transcription update events."""
        if is_final:
            print(f"\n🎤 Final transcription: {text} (confidence: {confidence:.2f})")
            print(f"\nListening for wake word '{args.wake_words}'...")
        else:
            # Print intermediate results on same line
            print(f"\r🎤 {text}", end="", flush=True)
    
    # Subscribe to events
    WakeWordModule.on_wake_word_detected(event_bus, on_wake_word_detected)
    WakeWordModule.on_wake_word_timeout(event_bus, on_wake_word_timeout)
    TranscriptionModule.on_transcription_updated(event_bus, on_transcription_update)
    
    # Start audio capture
    print("Starting audio capture...")
    AudioCaptureModule.start_recording(command_dispatcher)
    
    # Start wake word detection
    print(f"Listening for wake word '{args.wake_words}'...")
    WakeWordModule.start_detection(command_dispatcher)
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        WakeWordModule.stop_detection(command_dispatcher)
        AudioCaptureModule.stop_recording(command_dispatcher)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        WakeWordModule.stop_detection(command_dispatcher)
        AudioCaptureModule.stop_recording(command_dispatcher)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())