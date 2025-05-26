#!/usr/bin/env python3
"""
Server Example for Realtime_mlx_STT

This example demonstrates how to start the server with all modules properly registered.
It shows the correct sequence of module registration to enable full audio pipeline functionality.
"""

import os
import sys
import logging
import signal
import webbrowser
import threading
import time

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Configure logging
from src.Infrastructure.Logging import LoggingModule
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
from src.Features.WakeWordDetection.WakeWordModule import WakeWordModule

# Server imports
from src.Application.Server.ServerModule import ServerModule
from src.Application.Server.Configuration.ServerConfig import ServerConfig


def main():
    """Main function to start the server with all modules."""
    logger.info("Initializing Realtime_mlx_STT Server...")
    
    # Create core components
    command_dispatcher = CommandDispatcher()
    event_bus = EventBus()
    
    # Register all feature modules in the correct order
    # 1. Audio Capture - Provides audio input
    audio_handler = AudioCaptureModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus
    )
    logger.info("AudioCapture module registered")
    
    # 2. Voice Activity Detection - Processes audio chunks
    vad_handler = VadModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        default_detector="combined",
        processing_enabled=False  # Will be enabled by profiles
    )
    logger.info("VAD module registered")
    
    # 3. Wake Word Detection - Optional, for wake-word profiles
    wake_word_handler = WakeWordModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus
    )
    logger.info("WakeWord module registered")
    
    # 4. Transcription - Processes speech segments
    transcription_handler = TranscriptionModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        default_engine="mlx_whisper",
        default_model="whisper-large-v3-turbo",
        default_language=None
    )
    logger.info("Transcription module registered")
    
    # Set up VAD integration with transcription
    # This is critical for automatic transcription when silence is detected
    TranscriptionModule.register_vad_integration(
        event_bus=event_bus,
        transcription_handler=transcription_handler,
        session_id=None,  # Generate unique session for each speech segment
        auto_start_on_speech=True
    )
    logger.info("VAD-Transcription integration configured")
    
    # Configure server
    server_config = ServerConfig(
        host="127.0.0.1",
        port=8000,
        debug=False,
        auto_start=True,
        cors_origins=["*"]
    )
    
    # Register server module
    server = ServerModule.register(
        command_dispatcher=command_dispatcher,
        event_bus=event_bus,
        config=server_config
    )
    logger.info(f"Server started on http://{server_config.host}:{server_config.port}")
    logger.info("Use the API endpoints to start transcription with a profile")
    logger.info("Example: POST /system/start with {'profile': 'vad-triggered'}")
    
    # Open the web client in browser after a short delay
    def open_browser():
        import time  # Import here to avoid scope issues
        time.sleep(1.5)  # Wait for server to fully start
        web_client_path = os.path.join(os.path.dirname(__file__), 'server_web_client.html')
        if os.path.exists(web_client_path):
            logger.info(f"Opening web client in browser...")
            webbrowser.open(f'file://{os.path.abspath(web_client_path)}')
        else:
            logger.warning(f"Web client not found at {web_client_path}")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nShutting down server...")
        server.stop()
        # Stop all modules
        try:
            AudioCaptureModule.stop_recording(command_dispatcher)
        except:
            pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Keep the main thread alive
    try:
        signal.pause()  # Wait for signal
    except AttributeError:
        # Windows doesn't have signal.pause
        import time
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()