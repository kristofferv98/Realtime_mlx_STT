"""
Test for SileroVadDetector implementation.

This script validates the SileroVadDetector class functionality,
including initialization, configuration, and speech detection.
"""

import os
import sys
import unittest
import traceback
from io import BytesIO

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Check for required dependencies
SKIP_TESTS = False
SKIP_REASON = ""

try:
    import numpy as np
    import wave
    import torch  # Required for SileroVadDetector
    from src.Features.VoiceActivityDetection.Detectors import SileroVadDetector
except ImportError as e:
    SKIP_TESTS = True
    SKIP_REASON = f"Required dependency not available: {str(e)}"
    print(f"Warning: {SKIP_REASON}")
    traceback.print_exc()


class SileroVadDetectorTest(unittest.TestCase):
    """Test cases for SileroVadDetector"""
    
    @classmethod
    def setUpClass(cls):
        if SKIP_TESTS:
            raise unittest.SkipTest(SKIP_REASON)

    def setUp(self):
        """Set up the test environment"""
        if SKIP_TESTS:
            self.skipTest(SKIP_REASON)
        self.detector = SileroVadDetector(
            threshold=0.5,
            sample_rate=16000,
            window_size_samples=1536,
            use_onnx=True  # Use ONNX for faster tests
        )
        self.detector.setup()
        
        # Create a sample audio buffer with silence
        self.silence_data = np.zeros(16000, dtype=np.int16).tobytes()
        
        # Create a sample audio buffer with a sine wave (simulating speech)
        t = np.linspace(0, 1, 16000, False)
        tone = np.sin(2 * np.pi * 440 * t) * 32767 * 0.5
        self.tone_data = tone.astype(np.int16).tobytes()
        
        # Get the path to warmup audio for realistic test
        self.warmup_audio_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
            "RealtimeSTT", 
            "warmup_audio.wav"
        )

    def test_initialization(self):
        """Test if the detector initializes correctly"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.get_name(), "Silero VAD")
        self.assertEqual(self.detector.threshold, 0.5)
        self.assertEqual(self.detector.sample_rate, 16000)

    def test_configuration(self):
        """Test the configuration method"""
        config = {
            'threshold': 0.7,
            'window_size_samples': 2048
        }
        
        result = self.detector.configure(config)
        self.assertTrue(result)
        self.assertEqual(self.detector.threshold, 0.7)
        self.assertEqual(self.detector.window_size_samples, 2048)
        
        # Test invalid configuration
        invalid_config = {
            'threshold': 2.0  # Invalid value
        }
        result = self.detector.configure(invalid_config)
        self.assertTrue(result)  # Should still return True but log warning
        self.assertEqual(self.detector.threshold, 0.7)  # Should not change

    def test_detect_silence(self):
        """Test silence detection"""
        result, confidence = self.detector.detect_with_confidence(self.silence_data)
        self.assertFalse(result)
        self.assertLess(confidence, self.detector.threshold)

    def test_detect_tone(self):
        """Test tone detection - a simple tone might not be detected as speech"""
        # This might fail as a sine wave is not speech, but keeping for completeness
        result, confidence = self.detector.detect_with_confidence(self.tone_data)
        print(f"Tone detection confidence: {confidence}")
        # Not asserting the result as it depends on the model

    def test_detect_real_audio(self):
        """Test with real audio data"""
        if not os.path.exists(self.warmup_audio_path):
            self.skipTest("Warmup audio file not found")
        
        # Read the warmup audio file
        with wave.open(self.warmup_audio_path, 'rb') as wav_file:
            audio_data = wav_file.readframes(wav_file.getnframes())
        
        # Test detection
        result, confidence = self.detector.detect_with_confidence(audio_data)
        print(f"Real audio detection confidence: {confidence}")
        
        # Since warmup_audio.wav should contain speech, we expect detection
        self.assertTrue(result)
        self.assertGreaterEqual(confidence, self.detector.threshold)

    def test_reset(self):
        """Test reset method"""
        self.detector.reset()
        self.assertEqual(len(self.detector.speech_probs), 0)
        self.assertEqual(self.detector.triggered, False)

    def test_cleanup(self):
        """Test cleanup method"""
        self.detector.cleanup()
        self.assertIsNone(self.detector.model)


if __name__ == "__main__":
    unittest.main()