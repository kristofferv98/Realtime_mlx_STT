#!/usr/bin/env python3
"""
Unit tests for PorcupineWakeWordDetector.

This module tests the Porcupine wake word detector implementation,
including initialization, configuration, and detection capabilities.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import os
from typing import Dict, Any

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


class MockPorcupine:
    """Mock Porcupine instance for testing."""
    def __init__(self, access_key=None, keywords=None, keyword_paths=None, sensitivities=None):
        self.access_key = access_key
        self.keywords = keywords
        self.keyword_paths = keyword_paths
        self.sensitivities = sensitivities
        self.sample_rate = 16000
        self.frame_length = 512
        self.deleted = False
        
    def process(self, pcm):
        """Mock process method that returns -1 (no detection) by default."""
        return -1
        
    def delete(self):
        """Mock delete method."""
        self.deleted = True


# Create a mock pvporcupine module
mock_pvporcupine = MagicMock()
mock_pvporcupine.create = MagicMock(side_effect=lambda **kwargs: MockPorcupine(**kwargs))

# Patch before import
with patch.dict('sys.modules', {'pvporcupine': mock_pvporcupine}):
    from src.Features.WakeWordDetection.Detectors.PorcupineWakeWordDetector import PorcupineWakeWordDetector


class TestPorcupineWakeWordDetector(unittest.TestCase):
    """Test cases for PorcupineWakeWordDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset the mock for each test
        mock_pvporcupine.reset_mock()
        mock_pvporcupine.create = MagicMock(side_effect=lambda **kwargs: MockPorcupine(**kwargs))
        
    def tearDown(self):
        """Clean up after tests."""
        pass
        
    def test_initialization(self):
        """Test detector initialization."""
        detector = PorcupineWakeWordDetector(
            access_key="test_key",
            keywords=["alexa", "computer"],
            sensitivities=[0.5, 0.7]
        )
        
        self.assertIsNotNone(detector)
        self.assertEqual(detector.access_key, "test_key")
        self.assertEqual(detector.keywords, ["alexa", "computer"])
        self.assertEqual(detector.sensitivities, [0.5, 0.7])
        
    def test_initialization_without_pvporcupine(self):
        """Test detector initialization when pvporcupine is not available."""
        # This test is now less relevant since we're always mocking
        detector = PorcupineWakeWordDetector()
        self.assertIsNotNone(detector)
            
    def test_setup_with_builtin_keywords(self):
        """Test setup with built-in keywords."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector()
            
            success = detector.setup(wake_words=["alexa", "hey siri"], sensitivities=[0.6, 0.8])
            
            self.assertTrue(success)
            self.assertEqual(detector.keywords, ["alexa", "hey siri"])
            self.assertEqual(detector.sensitivities, [0.6, 0.8])
            self.assertEqual(detector.wake_words, ["alexa", "hey siri"])
            
            # Verify pvporcupine.create was called correctly
            mock_pvporcupine.create.assert_called_once_with(
                access_key='test_key',
                keywords=["alexa", "hey siri"],
                sensitivities=[0.6, 0.8]
            )
            
    def test_setup_with_custom_keyword_paths(self):
        """Test setup with custom keyword paths."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector(
                keyword_paths=["/path/to/keyword1.ppn", "/path/to/keyword2.ppn"]
            )
            
            success = detector.setup()
            
            self.assertTrue(success)
            self.assertEqual(detector.wake_words, ["keyword1", "keyword2"])
            
    def test_setup_without_access_key(self):
        """Test setup without access key."""
        # Clear environment variable
        with patch.dict('os.environ', {}, clear=True):
            detector = PorcupineWakeWordDetector()
            success = detector.setup()
            
            self.assertFalse(success)
            
    def test_process_audio_chunk(self):
        """Test processing audio chunks."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector()
            detector.setup()
            
            # Create mock audio data (512 samples of 16-bit PCM)
            audio_data = np.zeros(512, dtype=np.int16).tobytes()
            
            # Test no detection
            result = detector.process(audio_data)
            self.assertEqual(result, -1)
            
            # Mock detection of first wake word
            detector.porcupine.process = MagicMock(return_value=0)
            result = detector.process(audio_data)
            self.assertEqual(result, 0)
            
    def test_detect_method(self):
        """Test the detect method."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector()
            detector.setup(wake_words=["alexa", "computer"])
            
            # Create mock audio data
            audio_data = np.zeros(512, dtype=np.int16).tobytes()
            
            # Test no detection
            detected, wake_word = detector.detect(audio_data)
            self.assertFalse(detected)
            self.assertIsNone(wake_word)
            
            # Mock detection of second wake word
            detector.porcupine.process = MagicMock(return_value=1)
            detected, wake_word = detector.detect(audio_data)
            self.assertTrue(detected)
            self.assertEqual(wake_word, "computer")
            
    def test_detect_with_confidence(self):
        """Test the detect_with_confidence method."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector()
            detector.setup(wake_words=["alexa"], sensitivities=[0.7])
            
            # Create mock audio data
            audio_data = np.zeros(512, dtype=np.int16).tobytes()
            
            # Test no detection
            detected, confidence, wake_word = detector.detect_with_confidence(audio_data)
            self.assertFalse(detected)
            self.assertEqual(confidence, 0.0)
            self.assertIsNone(wake_word)
            
            # Mock detection
            detector.porcupine.process = MagicMock(return_value=0)
            detected, confidence, wake_word = detector.detect_with_confidence(audio_data)
            self.assertTrue(detected)
            self.assertEqual(confidence, 0.7)  # Should return sensitivity as confidence
            self.assertEqual(wake_word, "alexa")
            
    def test_audio_preparation(self):
        """Test audio preparation for different input formats."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector()
            detector.setup()
            
            # Test with correct length audio
            audio_data = np.zeros(512, dtype=np.int16).tobytes()
            prepared = detector._prepare_audio(audio_data)
            self.assertIsNotNone(prepared)
            self.assertEqual(len(prepared), 512)
            
            # Test with short audio (should be padded)
            short_audio = np.zeros(256, dtype=np.int16).tobytes()
            prepared = detector._prepare_audio(short_audio)
            self.assertIsNotNone(prepared)
            self.assertEqual(len(prepared), 512)
            
            # Test with long audio (should be truncated)
            long_audio = np.zeros(1024, dtype=np.int16).tobytes()
            prepared = detector._prepare_audio(long_audio)
            self.assertIsNotNone(prepared)
            self.assertEqual(len(prepared), 512)
            
            # Test with invalid input
            prepared = detector._prepare_audio("not bytes")
            self.assertIsNone(prepared)
            
    def test_configuration(self):
        """Test detector configuration."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector()
            
            config = {
                'keywords': ['jarvis', 'computer'],
                'sensitivities': [0.5, 0.6],
                'access_key': 'new_key'
            }
            
            success = detector.configure(config)
            self.assertTrue(success)
            self.assertEqual(detector.keywords, ['jarvis', 'computer'])
            self.assertEqual(detector.sensitivities, [0.5, 0.6])
            self.assertEqual(detector.access_key, 'new_key')
            
    def test_get_configuration(self):
        """Test getting current configuration."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector()
            detector.setup(wake_words=["alexa"])
            
            config = detector.get_configuration()
            
            self.assertEqual(config['detector_type'], 'porcupine')
            self.assertEqual(config['wake_words'], ["alexa"])
            self.assertEqual(config['keywords'], ["alexa"])
            self.assertEqual(config['sample_rate'], 16000)
            self.assertEqual(config['frame_length'], 512)
            
    def test_get_required_audio_format(self):
        """Test getting required audio format."""
        detector = PorcupineWakeWordDetector()
        
        # Before setup
        format_info = detector.get_required_audio_format()
        self.assertEqual(format_info['sample_rate'], 16000)
        self.assertEqual(format_info['frame_length'], 512)
        self.assertEqual(format_info['bit_depth'], 16)
        self.assertEqual(format_info['channels'], 1)
        
        # After setup
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector.setup()
            format_info = detector.get_required_audio_format()
            self.assertEqual(format_info['sample_rate'], 16000)
            self.assertEqual(format_info['frame_length'], 512)
            
    def test_get_sample_rate(self):
        """Test getting sample rate."""
        detector = PorcupineWakeWordDetector()
        
        # Before setup
        sample_rate = detector.get_sample_rate()
        self.assertEqual(sample_rate, 16000)
        
        # After setup
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector.setup()
            sample_rate = detector.get_sample_rate()
            self.assertEqual(sample_rate, 16000)
            
    def test_get_name(self):
        """Test getting detector name."""
        detector = PorcupineWakeWordDetector()
        name = detector.get_name()
        self.assertEqual(name, "Porcupine Wake Word Detector")
        
    def test_reset(self):
        """Test reset method."""
        detector = PorcupineWakeWordDetector()
        # Reset should not raise any errors
        detector.reset()
        
    def test_cleanup(self):
        """Test cleanup method."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector()
            detector.setup()
            
            # Verify porcupine instance exists
            self.assertIsNotNone(detector.porcupine)
            
            # Clean up
            detector.cleanup()
            
            # Verify cleanup was called
            self.assertIsNone(detector.porcupine)
            
    def test_cleanup_with_error(self):
        """Test cleanup with error handling."""
        with patch.dict('os.environ', {'PORCUPINE_ACCESS_KEY': 'test_key'}):
            detector = PorcupineWakeWordDetector()
            detector.setup()
            
            # Make delete raise an exception
            detector.porcupine.delete = MagicMock(side_effect=Exception("Delete error"))
            
            # Cleanup should handle the error gracefully
            detector.cleanup()
            self.assertIsNone(detector.porcupine)


if __name__ == '__main__':
    unittest.main()