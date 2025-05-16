#!/usr/bin/env python3
"""
Real-world Transcription Test.

This test performs actual transcription using the DirectMlxWhisperEngine on a real audio file.
No mocking is used, allowing us to verify the end-to-end functionality with real audio input.
"""

import os
import sys
import unittest
import time
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus

# Feature imports
from src.Features.Transcription.TranscriptionModule import TranscriptionModule


class RealTranscriptionTest(unittest.TestCase):
    """Test transcription with real audio without mocking."""
    
    def setUp(self):
        """Set up test environment with real components."""
        self.command_dispatcher = CommandDispatcher()
        self.event_bus = EventBus()
        
        # Store transcription results
        self.results = []
        self.is_final_received = False
        
        # Register the Transcription Module
        self.transcription_handler = TranscriptionModule.register(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus,
            default_engine="mlx_whisper",
            default_model="whisper-large-v3-turbo"
        )
        
        # Subscribe to transcription events
        def on_transcription_updated(session_id, text, is_final, confidence):
            logger.info(f"Transcription{'(FINAL)' if is_final else ''}: {text[:100]}{'...' if len(text) > 100 else ''}")
            logger.info(f"  Confidence: {confidence:.2f}")
            
            self.results.append({
                'session_id': session_id,
                'text': text,
                'is_final': is_final,
                'confidence': confidence
            })
            
            if is_final:
                self.is_final_received = True
        
        TranscriptionModule.on_transcription_updated(self.event_bus, on_transcription_updated)
        
        # Make sure the test audio file exists
        self.audio_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../../../bok_konge01.mp3'
        ))
        
        self.assertTrue(os.path.exists(self.audio_file), 
                       f"Test audio file not found: {self.audio_file}")
        
        logger.info(f"Using test audio file: {self.audio_file}")
    
    def test_transcribe_file(self):
        """Test transcription of a real audio file without mocking."""
        logger.info("Starting real transcription test")
        
        # Configure the transcription engine
        TranscriptionModule.configure(
            command_dispatcher=self.command_dispatcher,
            engine_type="mlx_whisper",
            model_name="whisper-large-v3-turbo",
            language=None,  # Auto-detect language
            streaming=False,  # Use parallel mode for faster results
            beam_size=1,  # Use smaller beam size for faster processing
            options={
                "quick_mode": True  # Use quick mode for faster processing (via options dict)
            }
        )
        
        start_time = time.time()
        
        # Transcribe the file
        result = TranscriptionModule.transcribe_file(
            command_dispatcher=self.command_dispatcher,
            file_path=self.audio_file,
            language=None,  # Auto-detect language
            options={"quick_mode": True},
            beam_size=1
        )
        
        processing_time = time.time() - start_time
        
        # Log the result and processing time
        logger.info(f"Transcription complete in {processing_time:.2f} seconds")
        
        # Check if result is a list (which happens sometimes) and get the first item
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        if isinstance(result, dict) and 'error' in result:
            logger.error(f"Transcription error: {result['error']}")
        
        # Verify the result
        self.assertIsInstance(result, dict, f"Expected dict result, got {type(result)}")
        self.assertNotIn('error', result, f"Transcription failed with error: {result.get('error', 'Unknown error')}")
        self.assertIn('text', result, "Transcription result doesn't contain text")
        self.assertTrue(len(result['text']) > 0, "Transcription text is empty")
        
        # Save the transcription result to a file for review
        output_file = os.path.join(
            os.path.dirname(self.audio_file),
            os.path.basename(self.audio_file).replace('.mp3', '_transcription.txt')
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        logger.info(f"Saved transcription to: {output_file}")
        
        # Wait for the final result if we haven't received it yet
        timeout = 30  # 30 seconds max wait time
        wait_start = time.time()
        while not self.is_final_received and time.time() - wait_start < timeout:
            time.sleep(0.5)
        
        # Verify we got final results
        self.assertTrue(self.is_final_received, "No final transcription result received")
        self.assertTrue(any(r['is_final'] for r in self.results), "No final result in event updates")
        
        # Log a summary of all results received
        logger.info(f"Received {len(self.results)} transcription updates")
        logger.info(f"Final transcription: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
        
        # Check for Norwegian content in the transcription (instead of relying on language tag)
        norwegian_words = ["Norge", "norske", "kongedømme", "monarkiet"]
        text_contains_norwegian = any(word in result['text'] for word in norwegian_words)
        self.assertTrue(text_contains_norwegian, "Transcription doesn't appear to contain Norwegian text")
        
        return result


if __name__ == "__main__":
    # When run directly, use a custom test runner
    print("Running real-world transcription test...")
    start_time = time.time()
    
    test = RealTranscriptionTest()
    test.setUp()
    result = test.test_transcribe_file()
    
    total_time = time.time() - start_time
    print(f"Test complete in {total_time:.2f} seconds")
    print(f"Transcription result (first 300 chars):")
    print("=========================================")
    print(result['text'][:300] + ('...' if len(result['text']) > 300 else ''))
    print("=========================================")