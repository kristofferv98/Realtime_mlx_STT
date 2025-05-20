#!/usr/bin/env python3
"""
Test the OpenAITranscriptionEngine implementation.

This test verifies the OpenAI API-based transcription engine functionality
using mocked API responses to avoid actual API calls.
"""

import os
import sys
import unittest
import time
import numpy as np
from unittest.mock import MagicMock, patch, Mock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the component to test
from src.Features.Transcription.Engines.OpenAITranscriptionEngine import OpenAITranscriptionEngine


class TestOpenAITranscriptionEngine(unittest.TestCase):
    """Test the OpenAITranscriptionEngine implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock the requests module to avoid actual API calls
        self.patcher_requests = patch('src.Features.Transcription.Engines.OpenAITranscriptionEngine.requests')
        self.mock_requests = self.patcher_requests.start()
        
        # Set up a mock response for API connectivity test
        mock_response = Mock()
        mock_response.status_code = 200
        self.mock_requests.get.return_value = mock_response
        
        # Mock the OpenAI client and its methods
        self.patcher_openai = patch('src.Features.Transcription.Engines.OpenAITranscriptionEngine.openai', create=True)
        self.mock_openai = self.patcher_openai.start()
        
        # Create a mock OpenAI client with a mock transcriptions.create method
        self.mock_client = Mock()
        self.mock_openai.OpenAI.return_value = self.mock_client
        
        # Mock the audio module and transcriptions chain
        self.mock_client.audio = Mock()
        self.mock_client.audio.transcriptions = Mock()
        self.mock_client.audio.transcriptions.create = Mock()
        
        # Mock soundfile to avoid file operations
        self.patcher_sf = patch('src.Features.Transcription.Engines.OpenAITranscriptionEngine.sf')
        self.mock_sf = self.patcher_sf.start()
        
        # Mock tempfile to avoid file operations
        self.patcher_tempfile = patch('src.Features.Transcription.Engines.OpenAITranscriptionEngine.tempfile')
        self.mock_tempfile = self.patcher_tempfile.start()
        
        # Mock threading to avoid actual thread creation
        self.patcher_threading = patch('src.Features.Transcription.Engines.OpenAITranscriptionEngine.threading')
        self.mock_threading = self.patcher_threading.start()
        
        # Setup mock thread that executes target function directly
        self.mock_thread = Mock()
        def thread_side_effect(target=None, args=(), daemon=True):
            if target and args:
                target(*args)
            return self.mock_thread
        self.mock_threading.Thread.side_effect = thread_side_effect
        
        # Set up mock tempfile
        mock_named_tempfile = Mock()
        self.mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_named_tempfile
        mock_named_tempfile.name = "/tmp/test.wav"
        
        # Initialize with a test API key
        self.test_api_key = "test_api_key_12345"
        self.engine = OpenAITranscriptionEngine(api_key=self.test_api_key)
        
    def tearDown(self):
        """Clean up after tests."""
        self.patcher_requests.stop()
        self.patcher_openai.stop()
        self.patcher_sf.stop()
        self.patcher_tempfile.stop()
        self.patcher_threading.stop()
    
    def test_initialization(self):
        """Test basic initialization of the engine."""
        # Test default parameters
        engine = OpenAITranscriptionEngine(api_key=self.test_api_key)
        self.assertEqual(engine.model_name, "gpt-4o-transcribe")
        self.assertEqual(engine.api_key, self.test_api_key)
        self.assertEqual(engine.sample_rate, 16000)
        self.assertIsNone(engine.language)
        self.assertTrue(engine.streaming)
        
        # Test custom parameters
        custom_engine = OpenAITranscriptionEngine(
            model_name="gpt-4o-mini-transcribe",
            language="en",
            api_key="custom_key",
            streaming=False,
            sample_rate=44100
        )
        self.assertEqual(custom_engine.model_name, "gpt-4o-mini-transcribe")
        self.assertEqual(custom_engine.language, "en")
        self.assertEqual(custom_engine.api_key, "custom_key")
        self.assertFalse(custom_engine.streaming)
        self.assertEqual(custom_engine.sample_rate, 44100)
    
    def test_start_success(self):
        """Test successful engine startup with API connectivity."""
        # Set up successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        self.mock_requests.get.return_value = mock_response
        
        # Start the engine
        result = self.engine.start()
        
        # Verify results
        self.assertTrue(result)
        self.assertTrue(self.engine._running)
        self.mock_requests.get.assert_called_once_with(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {self.test_api_key}"},
            timeout=5
        )
    
    def test_start_failure(self):
        """Test engine startup failure due to API issues."""
        # Set up failed API response
        mock_response = Mock()
        mock_response.status_code = 401  # Unauthorized
        self.mock_requests.get.return_value = mock_response
        
        # Start the engine
        result = self.engine.start()
        
        # Verify results
        self.assertFalse(result)
        self.assertFalse(self.engine._running)
    
    def test_is_running(self):
        """Test is_running method."""
        # Initially not running
        self.assertFalse(self.engine.is_running())
        
        # After successful start
        self.engine._running = True
        self.engine._client = Mock()
        self.assertTrue(self.engine.is_running())
        
        # With running flag but no client
        self.engine._running = True
        self.engine._client = None
        self.assertFalse(self.engine.is_running())
    
    def test_configure(self):
        """Test engine configuration updates."""
        # Set up initial state
        self.engine._running = True
        self.engine._client = Mock()
        self.mock_openai.reset_mock()
        
        # Update configuration
        config = {
            "language": "fr",
            "model_name": "gpt-4o-mini-transcribe",
            "api_key": "new_api_key",
            "streaming": False
        }
        
        result = self.engine.configure(config)
        
        # Verify results
        self.assertTrue(result)
        self.assertEqual(self.engine.language, "fr")
        self.assertEqual(self.engine.model_name, "gpt-4o-mini-transcribe")
        self.assertEqual(self.engine.api_key, "new_api_key")
        self.assertFalse(self.engine.streaming)
        
        # Verify client was reinitialized with new API key
        self.mock_openai.OpenAI.assert_called_once_with(api_key="new_api_key")
    
    def test_transcribe(self):
        """Test transcription with mocked API response."""
        # Set up engine
        self.engine.start()  # Ensure engine is properly set up
        
        # Set up mock API response
        mock_response = Mock()
        mock_response.text = "This is a test transcription."
        self.mock_client.audio.transcriptions.create.return_value = mock_response
        
        # Create test audio
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        # Start transcription
        self.engine.transcribe(test_audio)
        
        # Get result
        result = self.engine.get_result(timeout=0.1)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "This is a test transcription.")
        self.assertTrue(result["success"])
        self.assertTrue(result["is_final"])
        self.assertIn("processing_time", result)
        
        # Verify API was called with correct parameters
        self.mock_client.audio.transcriptions.create.assert_called_once()
    
    def test_add_audio_chunk(self):
        """Test adding audio chunks for streaming transcription."""
        # Set up engine
        self.engine.start()  # Ensure engine is properly set up
        
        # Create test audio chunks
        chunk1 = np.zeros(8000, dtype=np.float32)  # 0.5 seconds
        chunk2 = np.zeros(8000, dtype=np.float32)  # 0.5 seconds
        
        # Set up mock API response
        mock_response = Mock()
        mock_response.text = "Streaming transcription test."
        self.mock_client.audio.transcriptions.create.return_value = mock_response
        
        # First chunk - should be added to buffer but not processed
        self.engine.add_audio_chunk(chunk1, is_last=False)
        
        # Verify buffer state
        self.assertEqual(len(self.engine.audio_buffer), 1)
        
        # Second chunk with is_last=True - should trigger processing
        self.engine.add_audio_chunk(chunk2, is_last=True)
        
        # Get result
        result = self.engine.get_result(timeout=0.1)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], "Streaming transcription test.")
        self.assertTrue(result["success"])
        self.assertTrue(result["is_final"])
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Set up test state
        self.engine._running = True
        self.engine._client = Mock()
        self.engine._websocket = Mock()
        self.engine.audio_buffer = [np.zeros(1000)]
        
        # Execute cleanup
        with patch('gc.collect') as mock_gc_collect:
            self.engine.cleanup()
            
            # Verify state was reset
            self.assertFalse(self.engine._running)
            self.assertIsNone(self.engine._client)
            self.assertEqual(len(self.engine.audio_buffer), 0)
            
            # Verify garbage collection was requested
            mock_gc_collect.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling during transcription."""
        # Set up engine
        self.engine.start()  # Ensure engine is properly set up
        
        # Set up mock API to raise an exception
        def raise_exception(*args, **kwargs):
            raise Exception("API error")
        
        self.mock_client.audio.transcriptions.create.side_effect = raise_exception
        
        # Create test audio
        test_audio = np.zeros(16000, dtype=np.float32)
        
        # Start transcription with error
        self.engine.transcribe(test_audio)
        
        # Get result
        result = self.engine.get_result(timeout=0.1)
        
        # Verify error result
        self.assertIsNotNone(result)
        self.assertFalse(result["success"])
        self.assertEqual(result["text"], "")
        self.assertIn("error", result)
    
    def test_empty_audio_handling(self):
        """Test handling of empty audio input."""
        # Set up engine
        self.engine.start()  # Ensure engine is properly set up
        
        # Create empty audio
        empty_audio = np.array([], dtype=np.float32)
        
        # Start transcription with empty audio
        self.engine.transcribe(empty_audio)
        
        # Get result
        result = self.engine.get_result(timeout=0.1)
        
        # Verify empty result handling
        self.assertIsNotNone(result)
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("info", result)
        self.assertEqual(result["info"], "Empty or silent audio")


if __name__ == "__main__":
    unittest.main()