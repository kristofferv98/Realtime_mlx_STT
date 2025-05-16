#!/usr/bin/env python3
"""
Simple test script to verify basic Transcription feature functionality.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleTest")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Run a basic import test
def test_imports():
    try:
        from src.Core.Commands.command_dispatcher import CommandDispatcher
        from src.Core.Events.event_bus import EventBus
        
        # Import each module separately to isolate issues
        logger.info("Testing Core imports... OK")
        
        # Test TranscriptionModule import
        try:
            from src.Features.Transcription.TranscriptionModule import TranscriptionModule
            logger.info("TranscriptionModule import... OK")
        except Exception as e:
            logger.error(f"TranscriptionModule import failed: {str(e)}")
            
        # Test Models imports
        try:
            from src.Features.Transcription.Models import TranscriptionConfig, TranscriptionResult, TranscriptionSession
            logger.info("Models imports... OK")
        except Exception as e:
            logger.error(f"Models imports failed: {str(e)}")
            
        # Test Commands imports
        try:
            from src.Features.Transcription.Commands import (
                TranscribeAudioCommand, ConfigureTranscriptionCommand,
                StartTranscriptionSessionCommand, StopTranscriptionSessionCommand
            )
            logger.info("Commands imports... OK")
        except Exception as e:
            logger.error(f"Commands imports failed: {str(e)}")
            
        # Test Events imports
        try:
            from src.Features.Transcription.Events import (
                TranscriptionStartedEvent, TranscriptionUpdatedEvent, TranscriptionErrorEvent
            )
            logger.info("Events imports... OK")
        except Exception as e:
            logger.error(f"Events imports failed: {str(e)}")
            
        # Test Engines imports
        try:
            from src.Features.Transcription.Engines import DirectMlxWhisperEngine, DirectTranscriptionManager
            # Also test legacy names for backward compatibility
            from src.Features.Transcription.Engines import MlxWhisperEngine, TranscriptionProcessManager
            logger.info("Engines imports... OK")
        except Exception as e:
            logger.error(f"Engines imports failed: {str(e)}")
            
        return True
    except Exception as e:
        logger.error(f"Import test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running simple Transcription test...")
    result = test_imports()
    print(f"Test completed: {'SUCCESS' if result else 'FAILURE'}")
    sys.exit(0 if result else 1)