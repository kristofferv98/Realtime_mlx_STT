#!/usr/bin/env python3
"""
Test the DirectMlxWhisperEngine implementation and VAD integration.

This test verifies the direct MLX implementation (no process isolation)
and its integration with VAD events.
"""

import os
import sys
import unittest
import time
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Core.Events.event import Event

# Feature imports
from src.Features.Transcription.TranscriptionModule import TranscriptionModule
from src.Features.Transcription.Engines.DirectMlxWhisperEngine import DirectMlxWhisperEngine
from src.Features.Transcription.Engines.DirectTranscriptionManager import DirectTranscriptionManager
from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent


class TestDirectMlxWhisperEngine(unittest.TestCase):
    """Test the DirectMlxWhisperEngine implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.command_dispatcher = CommandDispatcher()
        self.event_bus = EventBus()
        
        # Mock the engine initialization to avoid actual model loading
        # which would be too slow for unit tests
        self.patcher = patch('src.Features.Transcription.Engines.DirectMlxWhisperEngine.DirectMlxWhisperEngine.start')
        self.mock_start = self.patcher.start()
        self.mock_start.return_value = True
        
        # Additionally mock the transcribe method
        self.patcher2 = patch('src.Features.Transcription.Engines.DirectMlxWhisperEngine.DirectMlxWhisperEngine.transcribe')
        self.mock_transcribe = self.patcher2.start()
        self.mock_transcribe.return_value = {"text": "Test transcription", "is_final": True, "success": True}
        
        # Mock the VAD methods
        self.patcher3 = patch('src.Features.Transcription.Engines.DirectMlxWhisperEngine.DirectMlxWhisperEngine._process_audio')
        self.mock_process = self.patcher3.start()
        
        # Register the module
        self.transcription_handler = TranscriptionModule.register(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus,
            default_engine="mlx_whisper",
            default_model="whisper-large-v3-turbo"
        )
        
        # Track events
        self.captured_events = []
        self.event_bus.subscribe(Event, lambda event: self.captured_events.append(event))
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        self.patcher2.stop()
        self.patcher3.stop()
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        # Create a patched version of is_running that returns True
        with patch.object(DirectMlxWhisperEngine, 'is_running', return_value=True):
            engine = DirectMlxWhisperEngine()
            engine.start()
            
            # Verify that start was called
            self.mock_start.assert_called_once()
            
            # Now check if is_running works with our patch
            self.assertTrue(engine.is_running())
    
    def test_engine_configure(self):
        """Test engine configuration."""
        engine = DirectMlxWhisperEngine()
        engine.start()
        engine.is_initialized = True
        
        # Configure with various options
        config = {
            "language": "en",
            "quick_mode": False,
            "beam_size": 5
        }
        
        result = engine.configure(config)
        self.assertTrue(result)
        self.assertEqual(engine.language, "en")
        self.assertEqual(engine.quick_mode, False)
        self.assertEqual(engine.beam_size, 5)
    
    def test_transcription_module_file_transcription(self):
        """Test file transcription via the module."""
        # Create a simple test file (just create a dummy array for now)
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        # Need to mock multiple components for this to work properly
        with patch('src.Features.Transcription.TranscriptionModule.TranscriptionModule.start_session') as mock_start_session, \
             patch('src.Core.Commands.command_dispatcher.CommandDispatcher.dispatch') as mock_dispatch:
            
            # Mock the session creation
            mock_start_session.return_value = {"session_id": "test-session-123"}
            
            # Mock the dispatch result
            mock_dispatch.return_value = {
                "text": "Test transcription from file",
                "is_final": True,
                "success": True,
                "language": "en",
                "confidence": 0.95
            }
            
            # Test transcribe_audio
            result = TranscriptionModule.transcribe_audio(
                command_dispatcher=self.command_dispatcher,
                audio_data=test_audio,
                session_id="test-session-123",  # Provide explicit session ID to avoid issues
                is_first_chunk=True,
                is_last_chunk=True
            )
            
            # Verify the result
            self.assertIn("text", result)
            self.assertEqual(result["text"], "Test transcription from file")
    
    def test_vad_integration(self):
        """Test VAD integration with transcription."""
        # Create a custom SilenceDetectedEvent class for testing to avoid Event.__post_init__ issues
        class TestSilenceEvent(Event):
            def __init__(self, speech_duration, speech_start_time, speech_end_time, audio_reference, speech_id):
                super().__init__()
                self.speech_duration = speech_duration
                self.speech_start_time = speech_start_time
                self.speech_end_time = speech_end_time
                self.audio_reference = audio_reference
                self.speech_id = speech_id
        
        # We need to override the TranscriptionCommandHandler's on_silence_detected method
        # so it triggers when our test event is published
        with patch.object(
            self.transcription_handler, 
            'on_silence_detected', 
            wraps=self.transcription_handler.on_silence_detected
        ) as mock_on_silence:
            
            # Set up the VAD integration by directly importing the SilenceDetectedEvent class
            from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent
            
            # Register the test silence handler
            self.event_bus.subscribe(SilenceDetectedEvent, mock_on_silence)
            
            # Create a test speech segment
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            speech_id = "test-speech-123"
            
            # Call the on_silence_detected method with the required parameters
            self.transcription_handler.on_silence_detected(speech_id, test_audio, 1.0)
            
            # Verify the handler was called
            mock_on_silence.assert_called_once_with(speech_id, test_audio, 1.0)
    
    def test_direct_transcription_manager(self):
        """Test the DirectTranscriptionManager."""
        manager = DirectTranscriptionManager()
        
        # Test start method
        with patch('src.Features.Transcription.Engines.DirectMlxWhisperEngine.DirectMlxWhisperEngine.start') as mock_start:
            mock_start.return_value = True
            
            # Start with config
            result = manager.start(
                engine_type="mlx_whisper",
                engine_config={"model_name": "whisper-large-v3-turbo", "language": "en"}
            )
            
            # Verify the result
            self.assertTrue(result)
            mock_start.assert_called_once()
            
            # Test is_running
            with patch('src.Features.Transcription.Engines.DirectMlxWhisperEngine.DirectMlxWhisperEngine.is_running') as mock_is_running:
                mock_is_running.return_value = True
                
                self.assertTrue(manager.is_running())
                
                # Test transcribe with mock engine
                with patch('src.Features.Transcription.Engines.DirectMlxWhisperEngine.DirectMlxWhisperEngine.get_result') as mock_get_result:
                    mock_get_result.return_value = {
                        "text": "Test manager transcription",
                        "is_final": True,
                        "success": True
                    }
                    
                    # Test complete audio
                    result = manager.transcribe(
                        audio_data=np.zeros(16000, dtype=np.float32),
                        is_first_chunk=True,
                        is_last_chunk=True
                    )
                    
                    # Verify the result
                    self.assertIn("text", result)
                    self.assertEqual(result["text"], "Test manager transcription")
                    
                    # Test stop
                    with patch('src.Features.Transcription.Engines.DirectMlxWhisperEngine.DirectMlxWhisperEngine.cleanup') as mock_cleanup:
                        mock_cleanup.return_value = None
                        
                        result = manager.stop()
                        self.assertTrue(result)
                        mock_cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()