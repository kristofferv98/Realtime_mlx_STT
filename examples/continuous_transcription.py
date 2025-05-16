#!/usr/bin/env python3
"""
Continuous Transcription Example

This example demonstrates continuous audio capture with Voice Activity Detection (VAD)
and real-time transcription. It:

1. Captures audio from the default microphone
2. Performs VAD to detect speech segments
3. Transcribes complete speech segments when silence is detected
4. Prints the transcribed text to the terminal

Press Ctrl+C to stop recording and exit.
"""

import os
import sys
import time
import logging
import argparse
import threading
import signal
from typing import Dict, Any, List, Optional

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
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule
from src.Features.VoiceActivityDetection.VadModule import VadModule
from src.Features.Transcription.TranscriptionModule import TranscriptionModule


class ContinuousTranscriptionApp:
    """Main application for continuous transcription with VAD."""
    
    def __init__(self, 
                device_index: Optional[int] = None,
                vad_aggressiveness: int = 2,
                language: Optional[str] = None,
                beam_size: int = 1,
                quick_mode: bool = True):
        """
        Initialize the application.
        
        Args:
            device_index: Index of audio device to use (None=default)
            vad_aggressiveness: VAD aggressiveness (0-3, higher=more aggressive)
            language: Language code or None for auto-detection
            beam_size: Beam search size for transcription
            quick_mode: Whether to use quick/parallel mode for faster transcription
        """
        self.device_index = device_index
        self.vad_aggressiveness = vad_aggressiveness
        self.language = language
        self.beam_size = beam_size
        self.quick_mode = quick_mode
        
        # Stats
        self.speech_count = 0
        self.total_duration = 0.0
        self.transcriptions: List[Dict[str, Any]] = []
        
        # Status flags
        self.is_running = False
        self.is_recording = False
        self.is_processing = False
        
        # Initialize components
        self.command_dispatcher = CommandDispatcher()
        self.event_bus = EventBus()
        
        # Set up the event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event handlers for the application."""
        # Register event handlers for transcription results
        def on_transcription_updated(session_id, text, is_final, confidence):
            """Handle transcription update events."""
            # Only print final transcriptions
            if is_final:
                # Format the output nicely
                logger.info(f"Transcription complete (confidence: {confidence:.2f})")
                print("\n" + "-" * 80)
                print(f"TRANSCRIPTION [{session_id[:8]}]:")
                print(f"{text}")
                print("-" * 80)
                
                # Store the result
                self.transcriptions.append({
                    'session_id': session_id,
                    'text': text,
                    'is_final': is_final,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
        
        # Track speech stats
        def on_speech_detected(confidence, timestamp, speech_id):
            """Handle speech detected events."""
            self.speech_count += 1
            logger.info(f"Speech detected [{speech_id[:8]}] (confidence: {confidence:.2f})")
        
        def on_silence_detected(speech_duration, start_time, end_time, speech_id):
            """Handle silence detected events."""
            self.total_duration += speech_duration
            avg_duration = self.total_duration / max(1, self.speech_count)
            logger.info(f"Speech ended [{speech_id[:8]}] (duration: {speech_duration:.2f}s, " 
                       f"avg: {avg_duration:.2f}s)")
        
        # Set up the recording state handler
        def on_recording_state_changed(is_recording, device_info, source_id):
            """Handle recording state changes."""
            self.is_recording = is_recording
            if is_recording:
                logger.info(f"Recording started on device: {device_info['name']}")
            else:
                logger.info("Recording stopped")
        
        # Register the handlers with the event bus
        TranscriptionModule.on_transcription_updated(self.event_bus, on_transcription_updated)
        VadModule.on_speech_detected(self.event_bus, on_speech_detected)
        VadModule.on_silence_detected(self.event_bus, on_silence_detected)
        AudioCaptureModule.on_recording_state_changed(self.event_bus, on_recording_state_changed)
    
    def initialize(self):
        """Initialize all modules and configure them."""
        logger.info("Initializing application components...")
        
        # Register features
        self.audio_handler = AudioCaptureModule.register(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus
        )
        
        self.vad_handler = VadModule.register(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus,
            default_detector="combined",
            default_aggressiveness=self.vad_aggressiveness
        )
        
        self.transcription_handler = TranscriptionModule.register(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus,
            default_engine="mlx_whisper",
            default_model="whisper-large-v3-turbo",
            default_language=self.language
        )
        
        # Configure transcription
        TranscriptionModule.configure(
            command_dispatcher=self.command_dispatcher,
            engine_type="mlx_whisper",
            model_name="whisper-large-v3-turbo",
            language=self.language,
            streaming=not self.quick_mode,  # quick_mode is opposite of streaming
            beam_size=self.beam_size,
            options={
                "quick_mode": self.quick_mode
            }
        )
        
        # Set up VAD integration with transcription
        # This automatically handles silence detection and transcription
        TranscriptionModule.register_vad_integration(
            event_bus=self.event_bus,
            transcription_handler=self.transcription_handler,
            session_id=None,  # Generate unique session for each speech segment
            auto_start_on_speech=True
        )
        
        logger.info("Application components initialized successfully")
    
    def start(self):
        """Start the application and begin processing audio."""
        if self.is_running:
            logger.warning("Application is already running")
            return
        
        # Start running
        self.is_running = True
        logger.info("Starting continuous transcription...")
        
        # If no device specified, list available devices
        available_devices = AudioCaptureModule.list_devices(self.command_dispatcher)
        if self.device_index is None:
            # Print available devices for user information
            logger.info("Available audio devices:")
            for idx, device in enumerate(available_devices):
                logger.info(f"  [{idx}]: {device['name']}")
            
            # Use default device (usually index 0)
            self.device_index = 0
            logger.info(f"Using default device: [{self.device_index}]: "
                       f"{available_devices[self.device_index]['name']}")
        
        # Start recording with the selected device
        audio_result = AudioCaptureModule.start_recording(
            command_dispatcher=self.command_dispatcher,
            device_index=self.device_index,
            sample_rate=16000,  # 16kHz required for VAD and transcription
            channels=1,  # Mono audio
            chunk_duration_ms=30  # 30ms chunks recommended for VAD
        )
        
        if not audio_result.get('success', False):
            logger.error(f"Failed to start recording: {audio_result.get('error', 'Unknown error')}")
            self.is_running = False
            return False
        
        # Start VAD processing on the audio stream
        vad_result = VadModule.start_detection(
            command_dispatcher=self.command_dispatcher,
            aggressiveness=self.vad_aggressiveness
        )
        
        if not vad_result.get('success', False):
            logger.error(f"Failed to start VAD: {vad_result.get('error', 'Unknown error')}")
            # Stop recording since we couldn't start VAD
            AudioCaptureModule.stop_recording(self.command_dispatcher)
            self.is_running = False
            return False
        
        logger.info(f"Continuous transcription started. Press Ctrl+C to stop.")
        return True
    
    def stop(self):
        """Stop all processing and clean up resources."""
        if not self.is_running:
            return
        
        logger.info("Stopping continuous transcription...")
        
        # Stop VAD processing
        try:
            VadModule.stop_detection(self.command_dispatcher)
        except Exception as e:
            logger.error(f"Error stopping VAD: {e}")
        
        # Stop audio recording
        try:
            AudioCaptureModule.stop_recording(self.command_dispatcher)
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
        
        # Print summary
        session_time = time.time()
        logger.info(f"Session summary:")
        logger.info(f"  Total speeches detected: {self.speech_count}")
        logger.info(f"  Total speech duration: {self.total_duration:.2f} seconds")
        if self.speech_count > 0:
            logger.info(f"  Average speech duration: {self.total_duration / self.speech_count:.2f} seconds")
        logger.info(f"  Total transcriptions: {len(self.transcriptions)}")
        
        self.is_running = False


def main():
    """Parse arguments and run the continuous transcription example."""
    parser = argparse.ArgumentParser(
        description="Continuous transcription with VAD example"
    )
    parser.add_argument("--device", "-d", type=int, default=None,
                      help="Audio device index (default: system default)")
    parser.add_argument("--vad-aggressiveness", "-v", type=int, default=2, choices=[0, 1, 2, 3],
                      help="VAD aggressiveness (0-3, higher=more aggressive, default: 2)")
    parser.add_argument("--language", "-l", type=str, default=None,
                      help="Language code (default: auto-detect)")
    parser.add_argument("--beam-size", "-b", type=int, default=1,
                      help="Beam search size for transcription (default: 1)")
    parser.add_argument("--no-quick-mode", action="store_true", 
                      help="Disable quick mode (more accurate but slower)")
    
    args = parser.parse_args()
    
    # Create and initialize the application
    app = ContinuousTranscriptionApp(
        device_index=args.device,
        vad_aggressiveness=args.vad_aggressiveness,
        language=args.language,
        beam_size=args.beam_size,
        quick_mode=not args.no_quick_mode
    )
    
    app.initialize()
    
    # Handle graceful shutdown with Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping transcription...")
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the application
    if app.start():
        # Keep running until user interrupts
        try:
            while app.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            app.stop()
            print("Application stopped by user")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())