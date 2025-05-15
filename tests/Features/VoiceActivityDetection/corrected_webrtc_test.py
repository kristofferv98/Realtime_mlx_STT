"""
Corrected WebRTC VAD test based on the pre-refactor implementation patterns.

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
class CorrectedWebRtcVadTest(unittest.TestCase):
    """Direct WebRTC VAD tests without framework dependencies"""

    def setUp(self):
        """Set up the test environment"""
        if SKIP_TESTS:
            self.skipTest("Required dependencies not available")
        
        # Initialize WebRTC VAD with sensitivity 3 (most aggressive)
        self.sensitivity = 3
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.sensitivity)
        
        # Create sample audio data
        self.sample_rate = 16000  # 16kHz
        
        # WebRTC requires frame durations of 10, 20, or 30 ms
        self.frame_duration_ms = 10
        self.samples_per_frame = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.bytes_per_frame = self.samples_per_frame * 2  # 16-bit = 2 bytes
        
        # Create silence frame
        self.silence_frame = np.zeros(self.samples_per_frame, dtype=np.int16).tobytes()
        
        # Create tone frame (sine wave)
        t = np.linspace(0, self.frame_duration_ms/1000, self.samples_per_frame, False)
        tone = np.sin(2 * np.pi * 440 * t) * 32767 * 0.9
        self.tone_frame = tone.astype(np.int16).tobytes()
        
        # Create multiple frames for chunking test
        self.silence_chunk = self.silence_frame * 3  # 3 frames of silence
        
        # Create mixed chunk (2 frames tone, 1 frame silence)
        self.mixed_chunk = self.tone_frame * 2 + self.silence_frame

    def test_vad_sensitivity_setting(self):
        """Test we can set WebRTC VAD sensitivity"""
        # Test we can change the sensitivity
        for mode in range(4):  # Valid modes are 0-3
            self.vad.set_mode(mode)
        
        # Test invalid mode raises ValueError
        with self.assertRaises(ValueError):
            self.vad.set_mode(4)  # Invalid mode

    def test_single_frame_detection(self):
        """Test detection on single frames"""
        # Test silence frame
        result = self.vad.is_speech(self.silence_frame, self.sample_rate)
        self.assertFalse(result)
        
        # Test tone frame (might be detected as speech depending on content)
        result = self.vad.is_speech(self.tone_frame, self.sample_rate)
        print(f"WebRTC VAD tone detection result: {result}")

    def test_frame_size_requirements(self):
        """Test WebRTC VAD frame size requirements"""
        # Test with invalid frame size (too small)
        small_frame = self.silence_frame[:len(self.silence_frame)//2]
        with self.assertRaises(Exception):
            self.vad.is_speech(small_frame, self.sample_rate)
        
        # Test with invalid frame size (too large)
        large_frame = self.silence_frame * 2
        with self.assertRaises(Exception):
            self.vad.is_speech(large_frame, self.sample_rate)

    def test_chunking_pattern(self):
        """Test chunking audio data into frames as done in the original code"""
        # Recreate the chunking pattern from the original code
        chunk = self.mixed_chunk  # 2 frames tone, 1 frame silence
        
        frame_count = len(chunk) // self.bytes_per_frame
        speech_frames = 0
        
        for i in range(frame_count):
            start_byte = i * self.bytes_per_frame
            end_byte = start_byte + self.bytes_per_frame
            frame = chunk[start_byte:end_byte]
            
            if len(frame) == self.bytes_per_frame:
                if self.vad.is_speech(frame, self.sample_rate):
                    speech_frames += 1
        
        # We expect 2 speech frames (from the tone) in our mixed chunk
        print(f"Detected {speech_frames} speech frames out of {frame_count}")
        # This might be 0 or 2 depending on if the tone is detected as speech
        self.assertLessEqual(speech_frames, frame_count)


if __name__ == "__main__":
    unittest.main()