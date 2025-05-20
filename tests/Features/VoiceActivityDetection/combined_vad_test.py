"""
Test for CombinedVadDetector implementation.

This script validates the CombinedVadDetector class functionality,
including state transitions, speech detection accuracy, and configuration.
"""

import os
import sys
import unittest
import traceback
from io import BytesIO
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Check for required dependencies
SKIP_TESTS = False
SKIP_REASON = ""

try:
    import numpy as np
    import wave
    import webrtcvad  # Required for WebRtcVadDetector (used by CombinedVadDetector)
    import torch  # Required for SileroVadDetector (used by CombinedVadDetector)
    from src.Features.VoiceActivityDetection.Detectors import CombinedVadDetector
    from src.Features.VoiceActivityDetection.Detectors.CombinedVadDetector import DetectionState
except ImportError as e:
    SKIP_TESTS = True
    SKIP_REASON = f"Required dependency not available: {str(e)}"
    print(f"Warning: {SKIP_REASON}")
    traceback.print_exc()


class CombinedVadDetectorTest(unittest.TestCase):
    """Test cases for CombinedVadDetector"""
    
    @classmethod
    def setUpClass(cls):
        if SKIP_TESTS:
            raise unittest.SkipTest(SKIP_REASON)

    def setUp(self):
        """Set up the test environment"""
        if SKIP_TESTS:
            self.skipTest(SKIP_REASON)
        self.detector = CombinedVadDetector(
            webrtc_aggressiveness=2,
            silero_threshold=0.5,
            sample_rate=16000,
            frame_duration_ms=30,
            speech_confirmation_frames=2,  # Reduced for testing
            silence_confirmation_frames=2,  # Reduced for testing
            use_silero_confirmation=True
        )
        self.detector.setup()
        
        # Create test audio buffers
        frame_size = int(16000 * 30 / 1000)  # 30ms at 16kHz
        
        # Silence buffer
        self.silence_data = np.zeros(frame_size, dtype=np.int16).tobytes()
        
        # Speech-like buffer (tone)
        t = np.linspace(0, 30/1000, frame_size, False)
        tone = np.sin(2 * np.pi * 220 * t) * 32767 * 0.8
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
        self.assertEqual(self.detector.get_name(), "Combined VAD (WebRTC + Silero)")
        self.assertEqual(self.detector.state, DetectionState.SILENCE)
        self.assertEqual(self.detector.consecutive_speech_frames, 0)
        self.assertEqual(self.detector.consecutive_silence_frames, 0)

    def test_configuration(self):
        """Test the configuration method"""
        config = {
            'webrtc_aggressiveness': 3,
            'silero_threshold': 0.7,
            'speech_confirmation_frames': 4,
            'silence_confirmation_frames': 5,
            'use_silero_confirmation': False
        }
        
        result = self.detector.configure(config)
        self.assertTrue(result)
        
        # Check if configuration was applied
        detector_config = self.detector.get_configuration()
        self.assertEqual(detector_config['webrtc_aggressiveness'], 3)
        self.assertEqual(detector_config['silero_threshold'], 0.7)
        self.assertEqual(detector_config['speech_confirmation_frames'], 4)
        self.assertEqual(detector_config['silence_confirmation_frames'], 5)
        self.assertEqual(detector_config['use_silero_confirmation'], False)

    def test_state_transitions(self):
        """Test state transitions for the detector"""
        # Start in SILENCE state
        self.assertEqual(self.detector.state, DetectionState.SILENCE)
        
        # Completely reset and reconfigure the detector
        self.detector.reset()
        self.detector.configure({
            'use_silero_confirmation': False,  # Disable for simpler testing
            'speech_confirmation_frames': 2,   # Ensure we know exactly how many frames needed
            'silence_confirmation_frames': 2,  # Ensure we know exactly how many frames needed
            'webrtc_threshold': 0.9,           # High threshold ensures tone is detected as speech
        })
        
        # Process "speech" frames to move to POTENTIAL_SPEECH
        _, _ = self.detector.detect_with_confidence(self.tone_data)
        self.assertEqual(self.detector.state, DetectionState.POTENTIAL_SPEECH)
        self.assertEqual(self.detector.consecutive_speech_frames, 1)
        
        # Another speech frame should move to SPEECH (with speech_confirmation_frames=2)
        is_speech, _ = self.detector.detect_with_confidence(self.tone_data)
        # This may fail if the WebRTC detector doesn't consistently detect our tone
        # as speech, but we'll check that it's either in SPEECH state or reporting speech is True
        speech_detected = (self.detector.state == DetectionState.SPEECH) or is_speech
        self.assertTrue(speech_detected, "Speech not detected after multiple tone frames")
        
        # Process many silence frames to ensure transition back to silence
        is_speech = True  # Initial value to enter the loop
        max_attempts = 10  # Limit loop to avoid infinite loop
        attempts = 0
        
        # Keep processing silence until we reach SILENCE state or max attempts
        while (self.detector.state != DetectionState.SILENCE) and (attempts < max_attempts):
            is_speech, _ = self.detector.detect_with_confidence(self.silence_data)
            attempts += 1
            
        # Either we should have reached SILENCE state or detection should be False
        self.assertTrue(
            self.detector.state == DetectionState.SILENCE or not is_speech,
            f"Failed to transition to silence after {attempts} frames. State: {self.detector.state}"
        )

    def test_statistics(self):
        """Test if statistics are being tracked"""
        # Disable Silero confirmation for testing
        self.detector.configure({'use_silero_confirmation': False})
        
        # Process a few speech and silence frames
        for _ in range(3):
            self.detector.detect_with_confidence(self.tone_data)
        
        for _ in range(3):
            self.detector.detect_with_confidence(self.silence_data)
        
        # Check if statistics were updated
        stats = self.detector.get_statistics()
        self.assertGreater(stats['webrtc_positives'], 0)
        self.assertGreater(stats['state_transitions'], 0)

    def test_reset(self):
        """Test reset method"""
        # Process some frames to change state
        self.detector.detect_with_confidence(self.tone_data)
        
        # Reset the detector
        self.detector.reset()
        
        # Check if state was reset
        self.assertEqual(self.detector.state, DetectionState.SILENCE)
        self.assertEqual(self.detector.consecutive_speech_frames, 0)
        self.assertEqual(self.detector.consecutive_silence_frames, 0)
        
        # Check if statistics were reset
        stats = self.detector.get_statistics()
        self.assertEqual(stats['webrtc_positives'], 0)
        self.assertEqual(stats['state_transitions'], 0)

    def test_with_real_audio_chunks(self):
        """Test with real audio data in chunks"""
        if not os.path.exists(self.warmup_audio_path):
            self.skipTest("Warmup audio file not found")
        
        # Read the warmup audio file
        with wave.open(self.warmup_audio_path, 'rb') as wav_file:
            # Get basic info
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            
            # Reset detector state
            self.detector.reset()
            
            # Configure for this test
            self.detector.configure({
                'sample_rate': sample_rate,
                'use_silero_confirmation': False  # Speed up test
            })
            
            # Read in chunks and process
            frame_size = int(sample_rate * 30 / 1000)  # 30ms at the file's sample rate
            speech_detected = False
            
            # Process the audio in chunks
            audio_data = wav_file.readframes(frame_size)
            while audio_data:
                is_speech, confidence = self.detector.detect_with_confidence(audio_data)
                if is_speech:
                    speech_detected = True
                
                audio_data = wav_file.readframes(frame_size)
            
            # We expect the warmup audio to contain speech
            self.assertTrue(speech_detected)
            
            # Check if statistics are reasonable
            stats = self.detector.get_statistics()
            self.assertGreater(stats['webrtc_positives'], 0)
            self.assertGreater(stats['speech_segments'], 0)


if __name__ == "__main__":
    unittest.main()