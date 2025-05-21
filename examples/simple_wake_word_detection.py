#!/usr/bin/env python3
"""
Simple wake word detection example.

This is a simplified version of the wake_word_detection.py example that avoids 
importing the huggingface_hub dependency, which can cause compatibility issues
with certain Python versions.
"""

import os
import sys
import signal
import argparse
import time
from typing import Optional, List

# Set up logging first
import logging
from src.Infrastructure.Logging import LoggingModule, LogLevel

# Initialize event system
from src.Core.Events.event_bus import EventBus
from src.Core.Commands.command_dispatcher import CommandDispatcher

# Import required modules
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule
from src.Features.VoiceActivityDetection.VadModule import VadModule
from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule

# Import event types
from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
from src.Features.VoiceActivityDetection.Events.SpeechDetectedEvent import SpeechDetectedEvent
from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
from src.Features.WakeWordDetection.Events.WakeWordTimeoutEvent import WakeWordTimeoutEvent

def main():
    """Run the simplified wake word detection example."""
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
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output for VAD and wake word processing")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed log messages (quiet mode is default)")
    
    args = parser.parse_args()
    
    # Get wake words from command line (comma-separated)
    wake_words = [word.strip() for word in args.wake_words.split(",")]
    
    # Get access key from command line or environment
    access_key = args.access_key or os.environ.get("PORCUPINE_ACCESS_KEY")
    if not access_key:
        print("Error: Porcupine access key not provided and PORCUPINE_ACCESS_KEY environment variable not set")
        print("Get an access key from https://console.picovoice.ai/")
        return 1
    
    # Initialize logging with the appropriate level
    LoggingModule.initialize(
        # Set console level based on command line arguments
        console_level=LogLevel.DEBUG if args.debug else (LogLevel.INFO if args.verbose else LogLevel.ERROR),
        file_enabled=True,  # Always log to file for debugging
        file_path="logs/wake_word_example.log",
        rotation_enabled=True,
        feature_levels={
            "WakeWordDetection": LogLevel.DEBUG if args.debug else LogLevel.INFO,
            "VoiceActivityDetection": LogLevel.DEBUG if args.debug else LogLevel.INFO,
            "AudioCapture": LogLevel.DEBUG if args.debug else LogLevel.INFO
        }
    )
    
    # Get a logger for this module
    logger = LoggingModule.get_logger(__name__)
    
    # Create event bus and command dispatcher
    event_bus = EventBus()
    command_dispatcher = CommandDispatcher()
    
    # Variables to track state
    is_wake_word_active = False
    vad_processing_enabled = False
    
    # Register modules
    logger.info("Initializing modules...")
    audio_module = AudioCaptureModule.register(command_dispatcher, event_bus)
    vad_module = VadModule.register(command_dispatcher, event_bus, processing_enabled=False)  # Start with VAD disabled
    
    # Register wake word module with specified wake words and sensitivity
    wake_word_module = WakeWordModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        wake_words=wake_words,
        sensitivities=[args.sensitivity] * len(wake_words),
        access_key=access_key
    )
    
    # Define the silence handler
    def on_silence_detected(speech_duration, start_time, end_time, speech_id):
        """Handle silence detected events, mainly for state management."""
        nonlocal is_wake_word_active, vad_processing_enabled
        
        if is_wake_word_active:
            # Log technical details
            logger.info(f"Speech detected after wake word (duration: {speech_duration:.2f}s)")
            print(f"Speech detected with duration: {speech_duration:.2f}s")
            
            # Reset wake word active flag
            is_wake_word_active = False
            
            # Disable VAD processing after speech is processed
            VadModule.disable_processing(command_dispatcher)
            vad_processing_enabled = False
            logger.debug("Disabled VAD processing after speech")
            
            # Print ready status
            print(f"\nListening for wake word '{args.wake_words}'...")
    
    # Set up event handlers
    def on_wake_word_detected(wake_word, confidence, timestamp):
        """Handle wake word detection events."""
        nonlocal is_wake_word_active, vad_processing_enabled
        
        # Log wake word detection
        logger.info(f"Wake word detected: '{wake_word}' (confidence: {confidence:.2f})")
        print(f"\n🔊 Wake word detected: '{wake_word}' (confidence: {confidence:.2f})")
        print("Listening for speech... (speak now)")
        
        # Set wake word active flag
        is_wake_word_active = True
        
        # Enable VAD processing when wake word is detected
        VadModule.enable_processing(command_dispatcher)
        vad_processing_enabled = True
        logger.debug("Enabled VAD processing after wake word")
    
    def on_wake_word_timeout(wake_word, timeout_duration):
        """Handle wake word timeout events."""
        nonlocal is_wake_word_active, vad_processing_enabled
        
        logger.info(f"Timed out after {timeout_duration:.1f}s without speech")
        print(f"\n⏱️ Timed out after {timeout_duration:.1f}s without speech")
        print(f"Listening for wake word '{args.wake_words}'...")
        
        # Reset wake word active flag
        is_wake_word_active = False
        
        # Disable VAD processing when timeout occurs
        VadModule.disable_processing(command_dispatcher)
        vad_processing_enabled = False
        logger.debug("Disabled VAD processing after timeout")
    
    # Subscribe to events
    WakeWordModule.on_wake_word_detected(event_bus, on_wake_word_detected)
    WakeWordModule.on_wake_word_timeout(event_bus, on_wake_word_timeout)
    VadModule.on_silence_detected(event_bus, on_silence_detected)
    
    # Start audio capture
    logger.info("Starting audio capture...")
    print("Starting audio capture...")
    AudioCaptureModule.start_recording(command_dispatcher)
    
    # Start wake word detection
    logger.info(f"Listening for wake word '{args.wake_words}'...")
    print(f"Listening for wake word '{args.wake_words}'...")
    WakeWordModule.start_detection(command_dispatcher)
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        print("\nShutting down...")
        WakeWordModule.stop_detection(command_dispatcher)
        AudioCaptureModule.stop_recording(command_dispatcher)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Keep running until interrupted
    try:
        print("Press Ctrl+C to exit")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()