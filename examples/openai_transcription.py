#!/usr/bin/env python3
# Set environment variables to disable progress bars before ANY other imports
import os
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

"""
OpenAI Transcription Example

This example demonstrates continuous audio capture with Voice Activity Detection (VAD)
and real-time transcription using the OpenAI GPT-4o-transcribe model. It:

1. Captures audio from the default microphone
2. Performs VAD to detect speech segments
3. Transcribes complete speech segments using OpenAI's cloud service
4. Prints the transcribed text to the terminal

Press Ctrl+C to stop recording and exit.

Note: This requires an OpenAI API key set in the OPENAI_API_KEY environment variable or
passed via the --api-key parameter.
"""

import os
import sys
import time
import logging
import argparse
import threading
import signal
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import ProgressBarManager for controlling tqdm progress bars
from src.Infrastructure.ProgressBar.ProgressBarManager import ProgressBarManager
from src.Infrastructure.Logging import LoggingModule

# Initialize the ProgressBarManager first - will be set based on arguments later
ProgressBarManager.initialize(disabled=False)  # Default to showing progress bars

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = LoggingModule.get_logger(__name__)

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus

# Feature imports
from src.Features.AudioCapture.AudioCaptureModule import AudioCaptureModule
from src.Features.VoiceActivityDetection.VadModule import VadModule
from src.Features.Transcription.TranscriptionModule import TranscriptionModule


class OpenAITranscriptionApp:
    """
    Application for OpenAI-based transcription with VAD.
    
    This application uses OpenAI's GPT-4o-transcribe model for high-quality
    cloud-based transcription.
    """
    
    def __init__(self, 
                device_index: Optional[int] = None,
                vad_aggressiveness: int = 2,
                language: Optional[str] = None,
                api_key: Optional[str] = None,
                model_name: str = "gpt-4o-transcribe",
                keep_history: bool = True,
                history_length: int = 10):
        """
        Initialize the application.
        
        Args:
            device_index: Index of audio device to use (None=default)
            vad_aggressiveness: VAD aggressiveness (0-3, higher=more aggressive)
            language: Language code or None for auto-detection
            api_key: OpenAI API key (will use OPENAI_API_KEY environment variable if None)
            model_name: OpenAI model name ("gpt-4o-transcribe" or "gpt-4o-mini-transcribe")
            keep_history: Whether to maintain transcription history
            history_length: Number of recent transcriptions to keep in history
        """
        self.device_index = device_index
        self.vad_aggressiveness = vad_aggressiveness
        self.language = language
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        
        # Verify API key is available
        if not self.api_key:
            logger.error("OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key.")
            sys.exit(1)
        
        # Transcription history settings
        self.keep_history = keep_history
        self.history_length = history_length
        self.transcription_history = []  # List of past transcribed segments
        
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
                
                # Store the result in transcription list
                result = {
                    'session_id': session_id,
                    'text': text,
                    'is_final': is_final,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                self.transcriptions.append(result)
                
                # Add to history if enabled
                if self.keep_history and text.strip():  # Only add non-empty transcriptions
                    self.transcription_history.append(text.strip())
                    # Limit the history length
                    if len(self.transcription_history) > self.history_length:
                        self.transcription_history = self.transcription_history[-self.history_length:]
                
                # Get combined history text
                history_text = ""
                if self.keep_history and self.transcription_history:
                    history_text = " ".join(self.transcription_history)
                
                # Print the transcription
                print("\n" + "-" * 80)
                print(f"OPENAI TRANSCRIPTION [{session_id[:8]}]:")
                print(f"{text}")
                
                # If history is maintained, show full history too
                if self.keep_history and len(self.transcription_history) > 1:
                    print("\nFULL HISTORY:")
                    print(f"{history_text}")
                    
                print("-" * 80)
        
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
        def on_recording_state_changed(previous_state, current_state):
            """Handle recording state changes."""
            from src.Features.AudioCapture.Events.RecordingStateChangedEvent import RecordingState
            
            # Update recording state
            self.is_recording = (current_state == RecordingState.RECORDING)
            
            # Log state change
            logger.info(f"Recording state changed: {previous_state.name} -> {current_state.name}")
            
            if current_state == RecordingState.RECORDING:
                logger.info(f"Recording started on device ID: {self.device_index}")
            elif current_state == RecordingState.STOPPED:
                logger.info("Recording stopped")
            elif current_state == RecordingState.ERROR:
                logger.error("Recording error occurred")
        
        # Handle transcription errors
        def on_transcription_error(session_id, error_message, error_type):
            """Handle transcription error events."""
            logger.error(f"Transcription error [{session_id[:8]}]: {error_message} (Type: {error_type})")
        
        # Register the handlers with the event bus
        TranscriptionModule.on_transcription_updated(self.event_bus, on_transcription_updated)
        TranscriptionModule.on_transcription_error(self.event_bus, on_transcription_error)
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
        
        # Register VAD module
        self.vad_handler = VadModule.register(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus,
            default_detector="combined",
            processing_enabled=True  # Enable processing immediately since we don't use wake word
        )
        
        # Configure VAD with the desired aggressiveness
        VadModule.configure_vad(
            command_dispatcher=self.command_dispatcher,
            detector_type="combined",
            sensitivity=self.vad_aggressiveness / 3.0,  # Convert 0-3 scale to 0-1 scale
            window_size=5
        )
        
        # Register transcription module with OpenAI engine
        self.transcription_handler = TranscriptionModule.register(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus,
            default_engine="openai",
            default_model=self.model_name,
            default_language=self.language,
            openai_api_key=self.api_key
        )
        
        # Verify OpenAI API key is valid by configuring the engine explicitly
        transcription_config_result = TranscriptionModule.configure(
            command_dispatcher=self.command_dispatcher,
            engine_type="openai",
            model_name=self.model_name,
            language=self.language,
            openai_api_key=self.api_key,
            streaming=True  # OpenAI works best with streaming mode
        )
        
        if not transcription_config_result:
            logger.error("Failed to configure OpenAI transcription engine. Check your API key.")
            return False
        
        # Set up VAD integration with transcription
        TranscriptionModule.register_vad_integration(
            event_bus=self.event_bus,
            transcription_handler=self.transcription_handler,
            session_id=None,  # Generate unique session for each speech segment
            auto_start_on_speech=True
        )
        
        logger.info("Application components initialized successfully")
        return True
    
    def start(self):
        """Start the application and begin processing audio."""
        if self.is_running:
            logger.warning("Application is already running")
            return
        
        # Start running
        self.is_running = True
        logger.info("Starting OpenAI-based transcription...")
        
        # List available devices and select one
        devices_list = AudioCaptureModule.list_devices(self.command_dispatcher)
        
        # The list_devices method returns a list that contains a list of device dicts
        # Need to flatten the structure to get the actual device list
        available_devices = []
        if isinstance(devices_list, list):
            # Handle nested list structure
            if len(devices_list) > 0 and isinstance(devices_list[0], list):
                available_devices = devices_list[0]
            else:
                # Try to use the outer list directly
                available_devices = devices_list
        
        logger.debug(f"Found {len(available_devices)} audio devices")
        
        if self.device_index is None:
            # Print available devices for user information
            logger.info("Available audio devices:")
            for device in available_devices:
                device_id = device.get('device_id', device.get('index', 0))
                device_name = device.get('name', 'Unknown Device')
                logger.info(f"  [{device_id}]: {device_name}")
            
            # Use default device (usually device_id 0 or the first one in the list)
            default_device = next((d for d in available_devices if d.get('is_default', False)), 
                                 available_devices[0] if available_devices else None)
            
            if default_device:
                self.device_index = default_device.get('device_id', default_device.get('index', 0))
                device_name = default_device.get('name', 'Unknown Device')
                logger.info(f"Using default device: [{self.device_index}]: {device_name}")
            else:
                # Fall back to device 0 if no devices found
                self.device_index = 0
                logger.warning(f"No devices found. Trying to use device index {self.device_index}")
        
        # Start recording with the selected device
        audio_result = AudioCaptureModule.start_recording(
            command_dispatcher=self.command_dispatcher,
            device_id=self.device_index,
            sample_rate=16000,  # 16kHz required for VAD and transcription
            chunk_size=512  # Use 512 samples (32ms) which is recommended for Silero VAD
        )
        
        # Handle the return value properly based on its type
        success = False
        if isinstance(audio_result, bool):
            success = audio_result
        elif isinstance(audio_result, list) and audio_result:
            # Non-empty list usually indicates success
            success = True
        
        if not success:
            logger.error(f"Failed to start recording")
            self.is_running = False
            return False
        
        # Start VAD processing by ensuring it's properly configured
        # Convert aggressiveness (0-3) to sensitivity (0-1)
        sensitivity = self.vad_aggressiveness / 3.0
        
        vad_result = VadModule.configure_vad(
            command_dispatcher=self.command_dispatcher,
            detector_type="combined",
            sensitivity=sensitivity,
            window_size=5,
            min_speech_duration=0.25
        )
        
        if not vad_result:
            logger.error(f"Failed to start VAD")
            # Stop recording since we couldn't start VAD
            AudioCaptureModule.stop_recording(self.command_dispatcher)
            self.is_running = False
            return False
        
        logger.info(f"OpenAI transcription started with model: {self.model_name}")
        logger.info(f"Press Ctrl+C to stop.")
        return True
    
    def stop(self):
        """Stop all processing and clean up resources."""
        if not self.is_running:
            return
        
        logger.info("Stopping OpenAI transcription...")
        
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
        
        # Print full history if enabled
        if self.keep_history and self.transcription_history:
            full_history = " ".join(self.transcription_history)
            print("\n" + "=" * 80)
            print("FULL TRANSCRIPTION HISTORY:")
            print(full_history)
            print("=" * 80)
        
        self.is_running = False


def main():
    """Parse arguments and run the OpenAI transcription example."""
    parser = argparse.ArgumentParser(
        description="OpenAI-based transcription with VAD example"
    )
    parser.add_argument("--device", "-d", type=int, default=None,
                      help="Audio device index (default: system default)")
    parser.add_argument("--vad-aggressiveness", "-v", type=int, default=2, choices=[0, 1, 2, 3],
                      help="VAD aggressiveness (0-3, higher=more aggressive, default: 2)")
    parser.add_argument("--language", "-l", type=str, default=None,
                      help="Language code (default: auto-detect)")
    parser.add_argument("--api-key", type=str, default=None,
                      help="OpenAI API key (will use OPENAI_API_KEY environment variable if not provided)")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-transcribe",
                      choices=["gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
                      help="OpenAI model to use (default: gpt-4o-transcribe)")
    parser.add_argument("--no-history", action="store_true",
                      help="Disable transcription history accumulation")
    parser.add_argument("--history-length", type=int, default=10,
                      help="Number of recent transcriptions to maintain in history (default: 10)")
    parser.add_argument("--no-progress-bars", action="store_true",
                      help="Hide progress bars from tqdm and huggingface-hub")
    
    args = parser.parse_args()
    
    # Update ProgressBarManager based on arguments
    ProgressBarManager.initialize(disabled=args.no_progress_bars)
    
    # Create and initialize the application
    app = OpenAITranscriptionApp(
        device_index=args.device,
        vad_aggressiveness=args.vad_aggressiveness,
        language=args.language,
        api_key=args.api_key,
        model_name=args.model,
        keep_history=not args.no_history,
        history_length=args.history_length
    )
    
    if not app.initialize():
        logger.error("Failed to initialize the application. Exiting.")
        return 1
    
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