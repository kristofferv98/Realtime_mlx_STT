"""
Simple WebRTC VAD test that doesn't rely on the full architecture.

This test directly tests the WebRTC VAD functionality without the core module imports.
"""

import os
import sys
import unittest
import traceback

try:
    import numpy as np
    import webrtcvad
    SKIP_TESTS = False
except ImportError as e:
    print(f"Warning: Required dependency not available: {e}")
    traceback.print_exc()
    SKIP_TESTS = True


@unittest.skipIf(SKIP_TESTS, "Required dependencies not available")
class SimpleWebRtcVadTest(unittest.TestCase):
    """Direct WebRTC VAD tests without framework dependencies"""

    def setUp(self):
        """Set up the test environment"""
        if SKIP_TESTS:
            self.skipTest("Required dependencies not available")
        
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
        
        # Create sample audio data
        self.frame_duration_ms = 30
        self.sample_rate = 16000
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # Create silence buffer
        self.silence_data = np.zeros(self.frame_size, dtype=np.int16).tobytes()
        
        # Create tone buffer (sine wave)
        t = np.linspace(0, self.frame_duration_ms/1000, self.frame_size, False)
        tone = np.sin(2 * np.pi * 440 * t) * 32767 * 0.9
        self.tone_data = tone.astype(np.int16).tobytes()

    def test_vad_initialization(self):
        """Test WebRTC VAD initialization"""
        self.assertIsNotNone(self.vad)
        self.assertEqual(self.vad.get_mode(), 3)

    def test_set_mode(self):
        """Test setting WebRTC VAD mode"""
        self.vad.set_mode(0)
        self.assertEqual(self.vad.get_mode(), 0)
        
        self.vad.set_mode(3)
        self.assertEqual(self.vad.get_mode(), 3)
        
        with self.assertRaises(ValueError):
            self.vad.set_mode(4)  # Invalid mode

    def test_silence_detection(self):
        """Test that silence is correctly identified"""
        result = self.vad.is_speech(self.silence_data, self.sample_rate)
        self.assertFalse(result)

    def test_tone_detection(self):
        """Test that tone may be detected as speech"""
        # Pure tones might be detected as speech by WebRTC VAD
        result = self.vad.is_speech(self.tone_data, self.sample_rate)
        # Just print the result, don't assert since it may vary
        print(f"WebRTC VAD tone detection result: {result}")


if __name__ == "__main__":
    unittest.main()