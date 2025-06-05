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
import argparse

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Realtime_mlx_STT Server")
    parser.add_argument('--no-browser', action='store_true', 
                       help='Do not automatically open the web client in browser')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Host to bind server to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind server to (default: 8000)')
    args = parser.parse_args()
    
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
    
    # Security warning for non-localhost binding
    if args.host not in ['127.0.0.1', 'localhost', '::1']:
        logger.warning("="*60)
        logger.warning("⚠️  SECURITY WARNING: Server binding to %s", args.host)
        logger.warning("This server is designed for LOCAL DEVELOPMENT ONLY!")
        logger.warning("It lacks authentication and security features.")
        logger.warning("DO NOT expose this server to the internet!")
        logger.warning("="*60)
        
        # Give user time to read warning
        time.sleep(3)
    
    # Configure server
    server_config = ServerConfig(
        host=args.host,
        port=args.port,
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
    logger.info(f"Server starting on http://{server_config.host}:{server_config.port}")
    
    if not args.no_browser:
        logger.info("Web interface will open automatically in your browser...")
        logger.info("If it doesn't open, manually navigate to the URL above")
    else:
        logger.info("Navigate to the URL above to access the web interface")
    
    logger.info("Press Ctrl+C to stop the server")
    
    # Open the web client in browser after a short delay (unless disabled)
    if not args.no_browser:
        def open_browser():
            import time  # Import here to avoid scope issues
            time.sleep(2.0)  # Wait for server to fully start
            web_client_path = os.path.join(os.path.dirname(__file__), 'server_web_client.html')
            if os.path.exists(web_client_path):
                try:
                    webbrowser.open(f'file://{os.path.abspath(web_client_path)}')
                    logger.info("Web client opened in browser")
                except Exception as e:
                    logger.warning(f"Could not open browser automatically: {e}")
                    logger.info(f"Please open {web_client_path} manually")
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