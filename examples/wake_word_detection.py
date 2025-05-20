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
    
    # Import needed event types
    from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
    from src.Features.VoiceActivityDetection.Events.SpeechDetectedEvent import SpeechDetectedEvent
    from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
    
    # Create a special SilenceDetectedEvent handler that only processes events when wake word is active
    def conditional_silence_handler(event):
        if is_wake_word_active and hasattr(event, 'audio_reference') and event.audio_reference is not None:
            # Process the complete speech segment with transcription
            try:
                # We have the complete audio segment, now transcribe it
                transcription_module.on_silence_detected(
                    session_id=f"wake-{event.speech_id}",
                    audio_reference=event.audio_reference,
                    duration=event.speech_duration
                )
            except Exception as e:
                print(f"Error transcribing speech: {e}")
    
    # Register our special handler with the event bus
    event_bus.subscribe(SilenceDetectedEvent, conditional_silence_handler)
    
    # Set up event handlers
    def on_wake_word_detected(wake_word, confidence, timestamp):
        """Handle wake word detection events."""
        print(f"\n🔊 Wake word detected: '{wake_word}' (confidence: {confidence:.2f})")
        print("Listening for speech... (speak now)")
    
    def on_wake_word_timeout(wake_word, timeout_duration):
        """Handle wake word timeout events."""
        nonlocal is_wake_word_active
        
        print(f"\n⏱️ Timed out after {timeout_duration:.1f}s without speech")
        print(f"Listening for wake word '{args.wake_words}'...")
        
        # Reset wake word active flag
        is_wake_word_active = False
    
    def on_transcription_update(session_id, text, is_final, confidence):
        """Handle transcription update events."""
        if is_final:
            print(f"\n🎤 Final transcription: {text} (confidence: {confidence:.2f})")
            print(f"\nListening for wake word '{args.wake_words}'...")
        else:
            # Print intermediate results on same line
            print(f"\r🎤 {text}", end="", flush=True)
    
    # Variables to track state
    is_wake_word_active = False
    
    # Subscribe to VAD events directly to handle proper state management
    def on_speech_detected(confidence, timestamp, speech_id):
        """Handle speech detected events, but only log when after wake word."""
        if is_wake_word_active:
            print(f"Speech detected after wake word!")
    
    def on_silence_detected(speech_duration, start_time, end_time, speech_id):
        """Handle silence detected events, mainly for state management."""
        nonlocal is_wake_word_active
        
        if is_wake_word_active:
            print(f"Processing speech after wake word (duration: {speech_duration:.2f}s)")
            
            # Reset wake word active flag after processing
            is_wake_word_active = False
            
            # The transcription part is handled by our conditional_silence_handler
            # which has direct access to the audio data
    
    # Enhanced wake word detection handler
    def on_wake_word_detected(wake_word, confidence, timestamp):
        """Handle wake word detection events."""
        nonlocal is_wake_word_active
        
        print(f"\n🔊 Wake word detected: '{wake_word}' (confidence: {confidence:.2f})")
        print("Listening for speech... (speak now)")
        
        # Set wake word active flag
        is_wake_word_active = True
    
    # Subscribe to events
    WakeWordModule.on_wake_word_detected(event_bus, on_wake_word_detected)
    WakeWordModule.on_wake_word_timeout(event_bus, on_wake_word_timeout)
    TranscriptionModule.on_transcription_updated(event_bus, on_transcription_update)
    VadModule.on_speech_detected(event_bus, on_speech_detected)
    VadModule.on_silence_detected(event_bus, on_silence_detected)
    
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