#!/usr/bin/env python3
"""
Continuous Transcription Example with Auto-Typing

This example demonstrates continuous audio capture with Voice Activity Detection (VAD),
real-time transcription, and automatic text insertion (as if typing). It:

1. Captures audio from the default microphone
2. Performs VAD to detect speech segments
3. Transcribes complete speech segments when silence is detected
4. Prints the transcribed text to the terminal
5. Automatically pastes the transcribed text into the active application

Features:
- Uses the clipboard and Command+V shortcut for pasting, ensuring proper handling of
  international character sets and keyboard layouts (Norwegian, etc.)
- Falls back to direct typing only if clipboard method fails
- Supports both latest-only and full-history modes

Press Ctrl+C to stop recording and exit.
"""

# Clear Python cache to ensure we use the latest code
import sys
import importlib
importlib.invalidate_caches()

# Clear any stale pyc files
if __name__ == "__main__":
    import os
    import re
    
    # Get the project root directory
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Function to clean up pyc files
    def clean_pyc_files(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".pyc"):
                    os.remove(os.path.join(root, file))
            
            # Clean up __pycache__ directories
            if "__pycache__" in dirs:
                pycache_dir = os.path.join(root, "__pycache__")
                for file in os.listdir(pycache_dir):
                    os.remove(os.path.join(pycache_dir, file))
    
    # Clean up the necessary directories
    clean_pyc_files(os.path.join(project_root, "src"))
    # Extra cleaning of critical modules
    clean_pyc_files(os.path.join(project_root, "src/Features/VoiceActivityDetection"))
    clean_pyc_files(os.path.join(project_root, "examples"))

import os
import sys
import time
import logging
import argparse
import threading
import signal
import subprocess
from typing import Dict, Any, List, Optional

# Try to import pyautogui - we'll need this for typing
try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable fail-safe feature
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("Warning: pyautogui is not installed. Auto-typing will not work.")
    print("To enable auto-typing, install pyautogui using UV:")
    print("  uv pip install pyautogui")
    print("Or with pip:")
    print("  pip install pyautogui")

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


def auto_type_text(text):
    """
    Automatically type text into the active window (simulating keyboard input)
    
    Args:
        text: The text to type
    """
    if not PYAUTOGUI_AVAILABLE:
        logger.error("pyautogui is not available. Cannot auto-type text.")
        return False
    
    try:
        # Give the user a moment to switch to their target application
        time.sleep(0.5)
        
        # Method 1: Use the clipboard for accurate character support
        # This avoids keyboard layout issues by using the system clipboard
        # which correctly handles all Unicode characters and special symbols
        import subprocess
        
        # Copy the text to clipboard
        process = subprocess.Popen('pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
        process.communicate(text.encode('utf-8'))
        
        # Paste the text using keyboard shortcut (Command+V)
        pyautogui.hotkey('command', 'v')
        
        logger.info("Text has been pasted via clipboard for accurate character support.")
        return True
    except Exception as e:
        logger.error(f"Failed to auto-type text: {e}")
        
        # Fallback: try direct typing if clipboard method fails
        try:
            logger.info("Attempting direct typing as fallback...")
            pyautogui.write(text)
            return True
        except:
            return False


class ContinuousTranscriptionApp:
    """
    Main application for continuous transcription with VAD and auto-typing.
    
    Important note about chunk sizes:
    Silero VAD models were trained using specific chunk sizes:
    - For 16kHz sample rate: 512, 1024, or 1536 samples
    - For 8kHz sample rate: 256, 512, or 768 samples
    Using other values may reduce the model's accuracy.
    """
    
    def __init__(self, 
                device_index: Optional[int] = None,
                vad_aggressiveness: int = 2,
                language: Optional[str] = None,
                beam_size: int = 1,
                quick_mode: bool = True,
                keep_history: bool = True,
                history_length: int = 10,
                paste_mode: str = "latest"):
        """
        Initialize the application.
        
        Args:
            device_index: Index of audio device to use (None=default)
            vad_aggressiveness: VAD aggressiveness (0-3, higher=more aggressive)
            language: Language code or None for auto-detection
            beam_size: Beam search size for transcription
            quick_mode: Whether to use quick/parallel mode for faster transcription
            keep_history: Whether to maintain transcription history
            history_length: Number of recent transcriptions to keep in history
            paste_mode: How to auto-type text ('latest' or 'full')
        """
        self.device_index = device_index
        self.vad_aggressiveness = vad_aggressiveness
        self.language = language
        self.beam_size = beam_size
        self.quick_mode = quick_mode
        
        # Transcription history settings
        self.keep_history = keep_history
        self.history_length = history_length
        self.transcription_history = []  # List of past transcribed segments
        
        # Auto-typing settings
        self.paste_mode = paste_mode
        
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
                
                # Get text to auto-type
                paste_text = ""
                if self.paste_mode == "full" and self.keep_history and self.transcription_history:
                    paste_text = " ".join(self.transcription_history)
                else:  # Default to latest mode
                    paste_text = text.strip()
                
                # Auto-type the text
                success = False
                if paste_text:
                    success = auto_type_text(paste_text)
                
                # Get combined history text for display
                history_text = ""
                if self.keep_history and self.transcription_history:
                    history_text = " ".join(self.transcription_history)
                
                # Print the transcription
                print("\n" + "-" * 80)
                print(f"TRANSCRIPTION [{session_id[:8]}]:")
                print(f"{text}")
                
                # If history is maintained, show full history too
                if self.keep_history and len(self.transcription_history) > 1:
                    print("\nFULL HISTORY:")
                    print(f"{history_text}")
                    
                print("-" * 80)
                if success:
                    print(f"✓ Pasted: {'Full history' if self.paste_mode == 'full' else 'Latest text'}")
                else:
                    print("⚠ Failed to paste text")
        
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
        logger.info("Starting continuous transcription with auto-typing...")
        logger.info(f"Auto-typing mode: {self.paste_mode.upper()} - will type {'full history' if self.paste_mode == 'full' else 'latest text'}")
        
        if not PYAUTOGUI_AVAILABLE:
            logger.warning("pyautogui is not installed! Auto-typing feature will not work.")
            logger.warning("Install with: pip install pyautogui")
        
        # If no device specified, list available devices
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
        
        print("\n" + "=" * 80)
        print("CONTINUOUS TRANSCRIPTION WITH AUTO-PASTING")
        print("=" * 80)
        print(f"✓ Recording started on device [{self.device_index}]")
        print(f"✓ Paste mode: {self.paste_mode.upper()}")
        print(f"✓ After each transcription, text will be pasted using Command+V")
        print(f"✓ Focus your cursor where you want text to appear")
        print(f"✓ Using clipboard method for international keyboard support (Norwegian etc.)")
        print("=" * 80)
        print("Press Ctrl+C to stop.")
        
        return True
    
    def stop(self):
        """Stop all processing and clean up resources."""
        if not self.is_running:
            return
        
        logger.info("Stopping continuous transcription...")
        
        # No need to stop VAD processing - it's just listening to events
        # The VAD will stop when audio stops sending events
        
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
            
            # Paste the final full history when stopping, if in full mode
            if self.paste_mode == "full" and PYAUTOGUI_AVAILABLE:
                if auto_type_text(full_history):
                    print("✓ Final full history has been pasted")
        
        self.is_running = False


def main():
    """Parse arguments and run the continuous transcription example with auto-typing."""
    parser = argparse.ArgumentParser(
        description="Continuous transcription with VAD and auto-typing example"
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
    parser.add_argument("--no-history", action="store_true",
                      help="Disable transcription history accumulation")
    parser.add_argument("--history-length", type=int, default=10,
                      help="Number of recent transcriptions to maintain in history (default: 10)")
    parser.add_argument("--paste-mode", type=str, default="latest", choices=["latest", "full"],
                      help="What to paste: 'latest' (only the most recent transcription) or 'full' (entire history)")
    
    args = parser.parse_args()
    
    # Create and initialize the application
    app = ContinuousTranscriptionApp(
        device_index=args.device,
        vad_aggressiveness=args.vad_aggressiveness,
        language=args.language,
        beam_size=args.beam_size,
        quick_mode=not args.no_quick_mode,
        keep_history=not args.no_history,
        history_length=args.history_length,
        paste_mode=args.paste_mode
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