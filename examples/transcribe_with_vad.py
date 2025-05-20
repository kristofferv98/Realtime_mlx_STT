#!/usr/bin/env python3
"""
Example of using the refactored Transcription feature with VAD integration.

This script demonstrates how to use the DirectMlxWhisperEngine with
Voice Activity Detection for efficient, complete-sentence transcription.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# System imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus

# Feature imports
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule
from src.Features.VoiceActivityDetection.VadModule import VadModule
from src.Features.Transcription.TranscriptionModule import TranscriptionModule


def main():
    """Run the example application."""
    parser = argparse.ArgumentParser(description="Transcribe audio with VAD integration")
    
    parser.add_argument("--input-file", type=str, default=None,
                        help="Input audio file to transcribe (default: use microphone)")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio device ID to use (default: system default)")
    parser.add_argument("--language", type=str, default=None,
                        help="Language code (default: auto-detect)")
    parser.add_argument("--model", type=str, default="whisper-large-v3-turbo",
                        help="Model name (default: whisper-large-v3-turbo)")
    parser.add_argument("--vad-type", type=str, default="combined",
                        choices=["webrtc", "silero", "combined"],
                        help="VAD detector type (default: combined)")
    parser.add_argument("--vad-threshold", type=float, default=0.8,
                        help="VAD sensitivity threshold (0.0-1.0)")
    parser.add_argument("--quick-mode", action="store_true",
                        help="Use quick/parallel mode (faster but slightly less accurate)")
    parser.add_argument("--record-seconds", type=int, default=0,
                        help="Record for specified number of seconds (default: 0 = until Ctrl+C)")
    
    args = parser.parse_args()
    
    # Initialize command dispatcher and event bus
    command_dispatcher = CommandDispatcher()
    event_bus = EventBus()
    
    # Track transcriptions for output
    current_transcription = ""
    
    # Callback for transcription updates
    def on_transcription_updated(session_id, text, is_final, confidence):
        nonlocal current_transcription
        
        if is_final:
            current_transcription = text
            print(f"\n[TRANSCRIPTION] {text}")
            print("-" * 80)
        
    # Callback for speech events
    def on_speech_detected(speech_id, confidence, timestamp):
        print(f"Speech detected (confidence: {confidence:.2f})")
        
    def on_silence_detected(speech_id, duration, audio_ref, start_time, end_time):
        print(f"Speech ended (duration: {duration:.2f}s)")
        
    # Register and connect features
    print("Initializing components...")
    
    # Register the Transcription module
    transcription_handler = TranscriptionModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        default_engine="mlx_whisper",
        default_model=args.model,
        default_language=args.language
    )
    
    # Configure transcription
    TranscriptionModule.configure(
        command_dispatcher=command_dispatcher,
        engine_type="mlx_whisper",
        model_name=args.model,
        language=args.language,
        streaming=not args.quick_mode,  # Streaming mode is more accurate but slower
        quick_mode=args.quick_mode
    )
    
    # Register the VAD module
    vad_handler = VadModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        detector_type=args.vad_type,
        threshold=args.vad_threshold
    )
    
    # Register AudioCapture last, as it will start capturing audio
    if args.input_file:
        # Use file-based audio input
        print(f"Using audio file: {args.input_file}")
        audio_handler = AudioCaptureModule.register(
            command_dispatcher=command_dispatcher,
            event_bus=event_bus,
            provider_type="file",
            file_path=args.input_file
        )
    else:
        # Use microphone input
        print("Using microphone input")
        audio_handler = AudioCaptureModule.register(
            command_dispatcher=command_dispatcher,
            event_bus=event_bus,
            provider_type="pyaudio",
            device_id=args.device
        )
    
    # Connect VAD with Transcription
    TranscriptionModule.register_vad_integration(
        event_bus=event_bus,
        transcription_handler=transcription_handler,
        session_id=None,  # Create new session for each speech segment
        auto_start_on_speech=True  # Start transcription when speech is detected
    )
    
    # Subscribe to events
    TranscriptionModule.on_transcription_updated(
        event_bus=event_bus,
        handler=on_transcription_updated
    )
    
    VadModule.on_speech_detected(
        event_bus=event_bus,
        handler=on_speech_detected
    )
    
    VadModule.on_silence_detected(
        event_bus=event_bus,
        handler=on_silence_detected
    )
    
    # Start capturing audio
    print("\nStarting audio capture...")
    print("Speak clearly and naturally, preferably with pauses between sentences.")
    print("Press Ctrl+C to stop.")
    print("-" * 80)
    
    # List available audio devices if using microphone
    if not args.input_file and args.device is None:
        devices = AudioCaptureModule.list_devices(command_dispatcher)
        print("\nAvailable audio devices:")
        for device in devices:
            print(f"  {device['id']}: {device['name']}")
        print()
    
    # Start recording
    AudioCaptureModule.start_recording(
        command_dispatcher=command_dispatcher, 
        device_id=args.device
    )
    
    try:
        # Record for specified time or until interrupted
        if args.record_seconds > 0:
            print(f"Recording for {args.record_seconds} seconds...")
            time.sleep(args.record_seconds)
        else:
            print("Recording until Ctrl+C is pressed...")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        # Stop recording
        AudioCaptureModule.stop_recording(command_dispatcher)
        
        # Final summary
        print("\nRecording stopped.")
        if current_transcription:
            print("\nFinal transcription:")
            print("-" * 80)
            print(current_transcription)
            print("-" * 80)
        
        print("Cleaning up...")
        
        # Clean up resources
        transcription_handler.cleanup()


if __name__ == "__main__":
    main()