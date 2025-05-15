"""
Test for WebRtcVadDetector implementation.

This script validates the WebRtcVadDetector class functionality,
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
    import webrtcvad
    from src.Features.VoiceActivityDetection.Detectors import WebRtcVadDetector
except ImportError as e:
    SKIP_TESTS = True
    SKIP_REASON = f"Required dependency not available: {str(e)}"
    print(f"Warning: {SKIP_REASON}")
    traceback.print_exc()


class WebRtcVadDetectorTest(unittest.TestCase):
    """Test cases for WebRtcVadDetector"""
    
    @classmethod
    def setUpClass(cls):
        if SKIP_TESTS:
            raise unittest.SkipTest(SKIP_REASON)

    def setUp(self):
        """Set up the test environment"""
        if SKIP_TESTS:
            self.skipTest(SKIP_REASON)
        self.detector = WebRtcVadDetector(
            aggressiveness=3,
            sample_rate=16000,
            frame_duration_ms=30,
            history_size=5,
            speech_threshold=0.6
        )
        self.detector.setup()
        
        # Create a sample audio buffer with silence
        self.silence_data = np.zeros(480, dtype=np.int16).tobytes()  # 30ms at 16kHz
        
        # Create a sample audio buffer with a sine wave (simulating speech)
        t = np.linspace(0, 30/1000, 480, False)  # 30ms at 16kHz
        tone = np.sin(2 * np.pi * 440 * t) * 32767 * 0.9  # High amplitude for detection
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
        self.assertEqual(self.detector.get_name(), "WebRTC VAD")
        self.assertEqual(self.detector.aggressiveness, 3)
        self.assertEqual(self.detector.sample_rate, 16000)
        self.assertEqual(self.detector.frame_duration_ms, 30)

    def test_configuration(self):
        """Test the configuration method"""
        config = {
            'aggressiveness': 2,
            'speech_threshold': 0.7
        }
        
        result = self.detector.configure(config)
        self.assertTrue(result)
        self.assertEqual(self.detector.aggressiveness, 2)
        self.assertEqual(self.detector.speech_threshold, 0.7)
        
        # Test invalid configuration
        invalid_config = {
            'aggressiveness': 5  # Invalid value
        }
        result = self.detector.configure(invalid_config)
        self.assertTrue(result)  # Should still return True but log warning
        self.assertEqual(self.detector.aggressiveness, 2)  # Should not change

    def test_detect_silence(self):
        """Test silence detection"""
        result = self.detector.detect(self.silence_data)
        self.assertFalse(result)
        
        result, confidence = self.detector.detect_with_confidence(self.silence_data)
        self.assertFalse(result)
        self.assertLess(confidence, self.detector.speech_threshold)

    def test_detect_tone(self):
        """Test tone detection"""
        # For WebRTC, a pure tone might be detected as speech
        result, confidence = self.detector.detect_with_confidence(self.tone_data)
        print(f"WebRTC tone detection confidence: {confidence}")
        
        # Process multiple frames to build up history
        for _ in range(5):
            self.detector.detect_with_confidence(self.tone_data)
        
        result, confidence = self.detector.detect_with_confidence(self.tone_data)
        print(f"WebRTC tone detection after history: {confidence}")

    def test_detect_real_audio(self):
        """Test with real audio data"""
        if not os.path.exists(self.warmup_audio_path):
            self.skipTest("Warmup audio file not found")
        
        # Read the warmup audio file
        with wave.open(self.warmup_audio_path, 'rb') as wav_file:
            # Check if the audio format is compatible
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                self.skipTest("Audio file must be mono 16-bit")
                
            # Get a frame of audio
            frame_size = int(self.detector.sample_rate * self.detector.frame_duration_ms / 1000)
            audio_data = wav_file.readframes(frame_size)
            
            if len(audio_data) < frame_size * 2:  # 16-bit = 2 bytes per sample
                self.skipTest("Audio file too short")
        
            # Test detection
            speech_detected = False
            frames_processed = 0
            
            # Process several frames to get a reliable result
            while audio_data and frames_processed < 20:
                if len(audio_data) == frame_size * 2:  # Ensure frame is right size
                    result = self.detector.detect(audio_data)
                    if result:
                        speech_detected = True
                        break
                
                audio_data = wav_file.readframes(frame_size)
                frames_processed += 1
            
            # We expect the warmup audio to contain speech
            self.assertTrue(speech_detected)

    def test_reset(self):
        """Test reset method"""
        # Process some frames to build history
        for _ in range(3):
            self.detector.detect_with_confidence(self.tone_data)
            
        self.assertGreater(len(self.detector.history), 0)
        
        # Reset
        self.detector.reset()
        self.assertEqual(len(self.detector.history), 0)

    def test_get_configuration(self):
        """Test get_configuration method"""
        config = self.detector.get_configuration()
        self.assertEqual(config['aggressiveness'], self.detector.aggressiveness)
        self.assertEqual(config['sample_rate'], self.detector.sample_rate)
        self.assertEqual(config['frame_duration_ms'], self.detector.frame_duration_ms)
        self.assertEqual(config['history_size'], self.detector.history_size)
        self.assertEqual(config['speech_threshold'], self.detector.speech_threshold)
        self.assertEqual(config['detector_type'], "WebRTC VAD")


if __name__ == "__main__":
    unittest.main()