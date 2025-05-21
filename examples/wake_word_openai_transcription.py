#!/usr/bin/env python3
# Set environment variables to disable progress bars before ANY other imports
import os
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Only show wanted outputs in quiet mode (default)
import sys
import warnings
import builtins
import tempfile
import importlib

# Store original stdout and print
original_stdout = sys.stdout
original_print = builtins.print

# Special print function that only prints allowed messages
def selective_print(*args, **kwargs):
    # Convert all args to strings and join
    text = " ".join(str(arg) for arg in args if arg is not None)
    # Allowed prefixes for user-friendly output
    allowed_prefixes = [
        "Starting audio", "Listening for", "👂", "🔊", "💤", "Say '",
        "Speech detected", "🎤 Final", "⏱️"
    ]
    # Only print if message starts with one of the allowed prefixes
    if any(text.strip().startswith(prefix) for prefix in allowed_prefixes):
        # Use original stdout and print
        kwargs.pop('file', None)  # Remove file if present
        original_print(*args, file=original_stdout, **kwargs)

# Define quiet mode condition
is_quiet_mode = "--debug" not in sys.argv and "--verbose" not in sys.argv
seen_transcriptions = set()  # Track seen transcriptions to avoid duplicates
null_file = open(os.devnull, 'w')

# If in quiet mode, suppress all output except selected prints
if is_quiet_mode:
    # Override print to our selective print
    builtins.print = selective_print
    # Redirect stdout to null
    sys.stdout = null_file
    # Suppress all Python warnings
    warnings.filterwarnings('ignore')

# Environment variables for quieter operation
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
os.environ['PYTHONIOENCODING'] = 'UTF-8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# We'll use ProgressBarManager for tqdm progress bars
# But import it later after command-line arguments are parsed

"""
Wake word detection with OpenAI transcription example.

This script demonstrates how to use the wake word detection functionality
to trigger OpenAI-based transcription only after a wake word is detected. 
It implements a two-stage activation approach for improved efficiency:

1. First stage: Only wake word detection runs continuously
2. Second stage: VAD and OpenAI transcription activated only after wake word detection

Running modes:
* Default (quiet): Shows only user output with emojis, hides log messages
  $ python -m examples.wake_word_openai_transcription
  
* Verbose mode: Shows detailed INFO-level logs and user output
  $ python -m examples.wake_word_openai_transcription --verbose
  
* Debug mode: Shows all DEBUG-level logs for detailed troubleshooting
  $ python -m examples.wake_word_openai_transcription --debug
  
Note: This requires an OpenAI API key set in the OPENAI_API_KEY environment variable or
passed via the --api-key parameter.
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
    """Run the wake word detection with OpenAI transcription example."""
    # We already set up quiet mode and print redirection at the top of the file
    global is_quiet_mode, seen_transcriptions
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Wake word detection with OpenAI transcription example")
    parser.add_argument("--wake-words", type=str, default="jarvis",
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
    parser.add_argument("--no-progress-bars", action="store_true",
                        help="Hide progress bars from tqdm and huggingface-hub")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (will check OPENAI_API_KEY env var if not provided)")
    parser.add_argument("--model", type=str, default="gpt-4o-transcribe",
                        help="OpenAI model to use for transcription (default: gpt-4o-transcribe)")
    
    args = parser.parse_args()
    
    # Get wake words from command line (comma-separated)
    wake_words = [word.strip() for word in args.wake_words.split(",")]
    
    # Get access key from command line or environment
    access_key = args.access_key or os.environ.get("PORCUPINE_ACCESS_KEY")
    if not access_key:
        print("Error: Porcupine access key not provided and PORCUPINE_ACCESS_KEY environment variable not set")
        print("Get an access key from https://console.picovoice.ai/")
        return 1
    
    # Get OpenAI API key from command line or environment
    openai_api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        print("Get an API key from https://platform.openai.com/")
        return 1
    
    # Create event bus and command dispatcher
    event_bus = EventBus()
    command_dispatcher = CommandDispatcher()
    
    # Setup logging using the centralized logging system
    import logging
    from src.Infrastructure.Logging import LoggingModule, LogLevel
    from src.Infrastructure.ProgressBar.ProgressBarManager import ProgressBarManager
    
    # Initialize the ProgressBarManager to control tqdm progress bars
    # Disable progress bars if explicitly requested with --no-progress-bars or in quiet mode
    ProgressBarManager.initialize(disabled=args.no_progress_bars or is_quiet_mode)
    
    # Initialize logging with the appropriate level
    LoggingModule.initialize(
        # Set console level based on command line arguments - default to quiet (ERROR level)
        console_level=LogLevel.DEBUG if args.debug else (LogLevel.INFO if args.verbose else LogLevel.ERROR),
        file_enabled=True,  # Always log to file for debugging regardless of console settings
        file_path="logs/wake_word_openai_example.log",
        rotation_enabled=True,
        start_control_server=True,  # Enable runtime log level adjustment
        # Specify format as strings to avoid any enum conversion issues
        console_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        feature_levels={
            "WakeWordDetection": LogLevel.DEBUG if args.debug else LogLevel.INFO,
            "VoiceActivityDetection": LogLevel.DEBUG if args.debug else LogLevel.INFO,
            "Transcription": LogLevel.DEBUG if args.debug else LogLevel.INFO,
            "AudioCapture": LogLevel.DEBUG if args.debug else LogLevel.INFO
        }
    )
    
    # Get a logger for this module
    logger = LoggingModule.get_logger(__name__)
    
    # Import needed event types
    from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
    from src.Features.VoiceActivityDetection.Events.SpeechDetectedEvent import SpeechDetectedEvent
    from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
    from src.Features.WakeWordDetection.Events.WakeWordTimeoutEvent import WakeWordTimeoutEvent
    
    # Variables to track state
    is_wake_word_active = False
    vad_processing_enabled = False
    # Make is_quiet_mode available to inner functions (already defined at the top of main())
    
    # Debug print function for monitoring state changes using logger
    def debug_print(message):
        # Use logger.debug instead of printing directly
        logger.debug(message)
    
    # Define the transcription handler
    def conditional_silence_handler(event):
        # Log detailed technical information for debugging purposes
        # These logs are primarily for developers and debugging, not for end users
        logger.debug(f"Transcription handler called")
        logger.debug(f"is_wake_word_active: {is_wake_word_active}")
        logger.debug(f"event has audio_reference: {hasattr(event, 'audio_reference')}")
        logger.debug(f"audio_reference is not None: {hasattr(event, 'audio_reference') and event.audio_reference is not None}")
        
        # Process regardless of wake_word_active state for debugging
        if hasattr(event, 'audio_reference') and event.audio_reference is not None:
            logger.debug("Audio reference found, attempting transcription")
            # Process the complete speech segment with transcription
            try:
                # We have the complete audio segment, now transcribe it using TranscriptionModule's static methods
                # Create a unique session ID for this transcription
                session_id = f"wake-{event.speech_id}"
                
                # Use transcribe_audio with the audio reference
                audio_data = event.audio_reference
                # Debug-level logging for technical details about the audio data
                logger.debug(f"Audio data ready for transcription - type: {type(audio_data)}")
                if hasattr(audio_data, 'shape'):
                    logger.debug(f"Audio data shape: {audio_data.shape}")
                elif hasattr(audio_data, '__len__'):
                    logger.debug(f"Audio data length: {len(audio_data)}")
                else:
                    logger.debug(f"Audio data has no shape or length attributes")
                
                try:
                    # Use debug level for technical execution steps
                    logger.debug("Calling TranscriptionModule.transcribe_audio...")
                    result = TranscriptionModule.transcribe_audio(
                        command_dispatcher,
                        audio_data=audio_data,
                        session_id=session_id,
                        is_first_chunk=True,
                        is_last_chunk=True
                    )
                    logger.debug("TranscriptionModule.transcribe_audio completed")
                except Exception as e:
                    logger.error(f"Error in TranscriptionModule.transcribe_audio: {e}", exc_info=True)
                    raise  # Re-raise to be caught by outer exception handler
                
                logger.debug(f"transcription result: {result}")
                
                # Print the result directly
                if result:
                    # TranscriptionModule.transcribe_audio returns a dictionary with text and other fields,
                    # but the structure might be nested in a list of results
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('text', '')
                        confidence = result[0].get('confidence', 0.0)
                        # Log technical details at debug level, user-friendly at info level
                        logger.debug(f"Final transcription text: {text}")
                        logger.info(f"Transcription complete with confidence: {confidence:.2f}")
                        # Check if we've already seen this transcription to avoid duplicates
                        transcription_key = f"{text}_{confidence:.2f}"
                        if transcription_key not in seen_transcriptions:
                            seen_transcriptions.add(transcription_key)
                            print(f"\n🎤 Final transcription: {text} (confidence: {confidence:.2f})")
                            print(f"\nListening for wake word '{args.wake_words}'...")
                    elif isinstance(result, dict):
                        if "text" in result:
                            # Direct dictionary with text field
                            # Log technical details at debug level, user-friendly at info level
                            logger.debug(f"Final transcription text: {result['text']}")
                            logger.info(f"Transcription complete with confidence: {result.get('confidence', 0.0):.2f}")
                            # Check if we've already seen this transcription to avoid duplicates
                            transcription_key = f"{result['text']}_{result.get('confidence', 0.0):.2f}"
                            if transcription_key not in seen_transcriptions:
                                seen_transcriptions.add(transcription_key)
                                print(f"\n🎤 Final transcription: {result['text']} (confidence: {result.get('confidence', 0.0):.2f})")
                                print(f"\nListening for wake word '{args.wake_words}'...")
                        elif "error" in result:
                            # Error occurred
                            logger.error(f"Transcription error: {result['error']}")
                            print(f"\n❌ Transcription error: {result['error']}")
                            print(f"\nListening for wake word '{args.wake_words}'...")
                        else:
                            # Dictionary structure without text field
                            logger.info(f"Final transcription: {result}")
                            print(f"\n🎤 Final transcription: {result}")
                            print(f"\nListening for wake word '{args.wake_words}'...")
                    else:
                        # Unknown result structure
                        logger.info(f"Transcription complete: {result}")
                        print(f"\n🎤 Transcription complete: {result}")
                        print(f"\nListening for wake word '{args.wake_words}'...")
                else:
                    logger.warning("No transcription result returned")
                    print("\nNo transcription result returned")
                    print(f"\nListening for wake word '{args.wake_words}'...")
            except Exception as e:
                logger.error(f"Error transcribing speech: {e}", exc_info=True)
                print(f"Error transcribing speech: {e}")
                print(f"\nListening for wake word '{args.wake_words}'...")
                # traceback will be included in the log due to exc_info=True
    
    # Register the transcription handler BEFORE registering modules
    # to ensure it has priority over other handlers
    event_bus.subscribe(SilenceDetectedEvent, conditional_silence_handler)
    logger.info("Registered conditional silence handler for transcription")
    
    # Register modules
    logger.info("Initializing modules...")
    audio_module = AudioCaptureModule.register(command_dispatcher, event_bus)
    vad_module = VadModule.register(command_dispatcher, event_bus, processing_enabled=False)  # Start with VAD disabled
    
    # Register transcription module with OpenAI engine
    transcription_module = TranscriptionModule.register(
        command_dispatcher, 
        event_bus,
        default_engine="openai",
        default_model=args.model,
        openai_api_key=openai_api_key
    )
    
    # Register wake word module with specified wake words and sensitivity
    wake_word_module = WakeWordModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        wake_words=wake_words,
        sensitivities=[args.sensitivity] * len(wake_words),
        access_key=access_key
    )
    
    # Set up event handlers
    def on_vad_status_change(enabled):
        """Monitor VAD processing status changes."""
        nonlocal vad_processing_enabled
        vad_processing_enabled = enabled
        debug_print(f"VAD processing {'enabled' if enabled else 'disabled'}")
    
    def on_wake_word_detected(wake_word, confidence, timestamp):
        """Handle wake word detection events."""
        nonlocal is_wake_word_active, vad_processing_enabled
        
        # Log wake word detection with technical details
        logger.info(f"Wake word detected: '{wake_word}' (confidence: {confidence:.2f})")
        # Print user-friendly output with emoji
        print(f"\n🔊 Wake word detected: '{wake_word}' (confidence: {confidence:.2f})")
        print("Listening for speech... (speak now)")
        
        # Set wake word active flag
        is_wake_word_active = True
        
        # Enable VAD processing when wake word is detected
        VadModule.enable_processing(command_dispatcher)
        vad_processing_enabled = True
        logger.debug("Explicitly enabled VAD processing after wake word")
    
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
        logger.debug("Explicitly disabled VAD processing after timeout")
    
    def on_transcription_update(session_id, text, is_final, confidence):
        """Handle transcription update events."""
        # We're handling final transcriptions in conditional_silence_handler
        # Only handle intermediate results here to avoid duplicates
        if not is_final:
            # Print intermediate results on same line
            print(f"\r🎤 {text}", end="", flush=True)
    
    # Only subscribe to speech detected events for logging purposes
    def on_speech_detected(confidence, timestamp, speech_id):
        """Handle speech detected events, but only log when after wake word."""
        if is_wake_word_active:
            logger.debug(f"Speech detected after wake word with confidence {confidence:.2f}")
            # Keep this console output since it's user-facing
            print(f"Speech detected after wake word!")
    
    def on_silence_detected(speech_duration, start_time, end_time, speech_id):
        """Handle silence detected events, mainly for state management."""
        nonlocal is_wake_word_active, vad_processing_enabled
        
        if is_wake_word_active:
            # Log technical details but only print in verbose/debug mode
            logger.info(f"Processing speech after wake word (duration: {speech_duration:.2f}s)")
            if not is_quiet_mode:
                print(f"Processing speech after wake word (duration: {speech_duration:.2f}s)")
            logger.debug(f"on_silence_detected called, is_wake_word_active: {is_wake_word_active}, speech_id: {speech_id}")
            
            # Reset wake word active flag - the handler will take care of processing
            is_wake_word_active = False
            
            # Explicitly disable VAD processing after speech is processed
            VadModule.disable_processing(command_dispatcher)
            vad_processing_enabled = False
            logger.debug("Explicitly disabled VAD processing after speech processing")
    
    # Subscribe to events
    WakeWordModule.on_wake_word_detected(event_bus, on_wake_word_detected)
    WakeWordModule.on_wake_word_timeout(event_bus, on_wake_word_timeout)
    TranscriptionModule.on_transcription_updated(event_bus, on_transcription_update)
    VadModule.on_speech_detected(event_bus, on_speech_detected)
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
        # Stop the control server if it's running
        LoggingModule.stop_control_server()
        WakeWordModule.stop_detection(command_dispatcher)
        AudioCaptureModule.stop_recording(command_dispatcher)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Verify VAD processing is actually disabled at startup
    disabled_result = VadModule.disable_processing(command_dispatcher)
    debug_print(f"Initial VAD disable result: {disabled_result}")
    
    # Log system state information
    logger.info("Wake word detection active (only wake word processing)")
    logger.info("VAD processing disabled (will be activated after wake word)")
    logger.info("System is in low-power mode until wake word is detected")
    logger.info(f"Using OpenAI '{args.model}' for transcription")
    
    # Print user-friendly messages with emojis
    print(f"👂 Wake word detection active (only wake word processing)")
    print(f"🔊 VAD processing disabled (will be activated after wake word)")
    print(f"💤 System is in low-power mode until wake word is detected")
    print(f"🤖 Using OpenAI '{args.model}' for transcription")
    print(f"Say '{args.wake_words}' to activate")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        print("\nShutting down...")
        # Stop the control server if it's running
        LoggingModule.stop_control_server()
        WakeWordModule.stop_detection(command_dispatcher)
        AudioCaptureModule.stop_recording(command_dispatcher)
        # Log a final message
        logger.info("Application terminated")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())