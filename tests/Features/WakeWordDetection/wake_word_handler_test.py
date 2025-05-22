#!/usr/bin/env python3
"""
Unit tests for WakeWordCommandHandler.

This module tests the wake word command handler implementation,
including command handling, event processing, and state management.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import time
import numpy as np
import os
from collections import deque
from typing import List, Dict, Any

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.Core.Events.event_bus import EventBus
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Features.WakeWordDetection.Handlers.WakeWordCommandHandler import WakeWordCommandHandler, DetectorState
from src.Features.WakeWordDetection.Commands.ConfigureWakeWordCommand import ConfigureWakeWordCommand
from src.Features.WakeWordDetection.Commands.StartWakeWordDetectionCommand import StartWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.StopWakeWordDetectionCommand import StopWakeWordDetectionCommand
from src.Features.WakeWordDetection.Commands.DetectWakeWordCommand import DetectWakeWordCommand
from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
from src.Features.WakeWordDetection.Events.WakeWordDetectionStartedEvent import WakeWordDetectionStartedEvent
from src.Features.WakeWordDetection.Events.WakeWordDetectionStoppedEvent import WakeWordDetectionStoppedEvent
from src.Features.WakeWordDetection.Events.WakeWordTimeoutEvent import WakeWordTimeoutEvent
from src.Features.WakeWordDetection.Models.WakeWordConfig import WakeWordConfig
from src.Features.AudioCapture.Models.AudioChunk import AudioChunk
from src.Features.AudioCapture.Events.AudioChunkCapturedEvent import AudioChunkCapturedEvent
from src.Features.VoiceActivityDetection.Events.SpeechDetectedEvent import SpeechDetectedEvent
from src.Features.VoiceActivityDetection.Events.SilenceDetectedEvent import SilenceDetectedEvent


class TestWakeWordCommandHandler(unittest.TestCase):
    """Test cases for WakeWordCommandHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock event bus and command dispatcher
        self.event_bus = Mock(spec=EventBus)
        self.command_dispatcher = Mock(spec=CommandDispatcher)
        
        # Mock the detector
        self.mock_detector = Mock()
        self.mock_detector.configure.return_value = True
        self.mock_detector.detect.return_value = (False, None)
        self.mock_detector.detect_with_confidence.return_value = (False, 0.0, None)
        self.mock_detector.cleanup.return_value = None
        
        # Create handler with mocked detector
        self.handler = WakeWordCommandHandler(self.event_bus, self.command_dispatcher)
        self.handler.detectors['porcupine'] = self.mock_detector
        
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.handler, 'cleanup'):
            self.handler.cleanup()
            
    def test_initialization(self):
        """Test handler initialization."""
        # Create a fresh mock event bus for this test
        fresh_event_bus = Mock(spec=EventBus)
        handler = WakeWordCommandHandler(fresh_event_bus, self.command_dispatcher)
        
        self.assertIsNotNone(handler)
        self.assertEqual(handler.state, DetectorState.INACTIVE)
        self.assertFalse(handler.is_detecting)
        self.assertEqual(handler.active_detector_name, 'porcupine')
        self.assertIsInstance(handler.config, WakeWordConfig)
        self.assertFalse(handler.vad_subscribed)
        
        # Verify event subscription
        fresh_event_bus.subscribe.assert_called_once_with(
            AudioChunkCapturedEvent, 
            handler._on_audio_chunk_captured
        )
        
    def test_can_handle(self):
        """Test can_handle method."""
        # Should handle wake word commands
        self.assertTrue(self.handler.can_handle(ConfigureWakeWordCommand(WakeWordConfig())))
        self.assertTrue(self.handler.can_handle(StartWakeWordDetectionCommand()))
        self.assertTrue(self.handler.can_handle(StopWakeWordDetectionCommand()))
        self.assertTrue(self.handler.can_handle(DetectWakeWordCommand(
            AudioChunk(raw_data=b'', sample_rate=16000, channels=1, format='int16')
        )))
        
        # Should not handle other commands
        from src.Core.Commands.command import Command
        other_command = Mock(spec=Command)
        self.assertFalse(self.handler.can_handle(other_command))
        
    def test_handle_configure_wake_word(self):
        """Test handling ConfigureWakeWordCommand."""
        config = WakeWordConfig(
            detector_type='porcupine',
            wake_words=['alexa', 'computer'],
            sensitivities=[0.5, 0.7],
            access_key='test_key',
            buffer_duration=2.0
        )
        command = ConfigureWakeWordCommand(config)
        
        result = self.handler.handle(command)
        
        self.assertTrue(result)
        # Verify the config was updated (handler.handle modifies the handler's config)
        self.assertEqual(self.handler.config.detector_type, 'porcupine')
        self.assertEqual(self.handler.config.wake_words, ['alexa', 'computer'])
        self.assertEqual(self.handler.config.sensitivities, [0.5, 0.7])
        self.assertEqual(self.handler.config.access_key, 'test_key')
        self.assertEqual(self.handler.config.buffer_duration, 2.0)
        self.assertEqual(self.handler.active_detector_name, 'porcupine')
        
        # Verify detector configuration
        self.mock_detector.configure.assert_called_once()
        config_dict = self.mock_detector.configure.call_args[0][0]
        self.assertEqual(config_dict['keywords'], ['alexa', 'computer'])
        self.assertEqual(config_dict['sensitivities'], [0.5, 0.7])
        self.assertEqual(config_dict['access_key'], 'test_key')
        
    def test_handle_configure_with_unknown_detector(self):
        """Test configuration with unknown detector type."""
        # WakeWordConfig validates detector_type, so we need to set it after creation
        config = WakeWordConfig()
        config.detector_type = 'unknown_detector'  # Bypass validation
        command = ConfigureWakeWordCommand(config)
        
        result = self.handler.handle(command)
        
        self.assertFalse(result)
        
    def test_handle_start_wake_word_detection(self):
        """Test handling StartWakeWordDetectionCommand."""
        command = StartWakeWordDetectionCommand()
        
        result = self.handler.handle(command)
        
        self.assertTrue(result)
        self.assertTrue(self.handler.is_detecting)
        self.assertEqual(self.handler.state, DetectorState.WAKE_WORD)
        self.assertFalse(self.handler.wake_word_detected)
        
        # Verify event published
        self.event_bus.publish.assert_called()
        published_event = self.event_bus.publish.call_args[0][0]
        self.assertIsInstance(published_event, WakeWordDetectionStartedEvent)
        self.assertEqual(published_event.detector_type, 'porcupine')
        
        # Verify VAD disabled
        self.command_dispatcher.dispatch.assert_called()
        
    def test_handle_start_when_already_detecting(self):
        """Test starting detection when already active."""
        self.handler.is_detecting = True
        command = StartWakeWordDetectionCommand()
        
        result = self.handler.handle(command)
        
        self.assertTrue(result)  # Should return True but not restart
        
    def test_handle_stop_wake_word_detection(self):
        """Test handling StopWakeWordDetectionCommand."""
        # Start detection first
        self.handler.is_detecting = True
        self.handler.state = DetectorState.WAKE_WORD
        
        command = StopWakeWordDetectionCommand()
        result = self.handler.handle(command)
        
        self.assertTrue(result)
        self.assertFalse(self.handler.is_detecting)
        self.assertEqual(self.handler.state, DetectorState.INACTIVE)
        self.assertFalse(self.handler.wake_word_detected)
        
        # Verify event published
        self.event_bus.publish.assert_called()
        published_event = self.event_bus.publish.call_args[0][0]
        self.assertIsInstance(published_event, WakeWordDetectionStoppedEvent)
        self.assertEqual(published_event.reason, "user_requested")
        
    def test_handle_stop_when_not_detecting(self):
        """Test stopping detection when not active."""
        command = StopWakeWordDetectionCommand()
        
        result = self.handler.handle(command)
        
        self.assertTrue(result)  # Should return True even if not detecting
        
    def test_handle_detect_wake_word(self):
        """Test handling DetectWakeWordCommand."""
        audio_chunk = AudioChunk(
            raw_data=np.zeros(512, dtype=np.int16).tobytes(),
            sample_rate=16000,
            channels=1,
            format='int16',
            timestamp=0
        )
        command = DetectWakeWordCommand(
            audio_chunk=audio_chunk,
            return_confidence=True
        )
        
        # Mock detection result
        self.mock_detector.detect_with_confidence.return_value = (True, 0.8, "alexa")
        
        result = self.handler.handle(command)
        
        self.assertEqual(result['detected'], True)
        self.assertEqual(result['wake_word'], "alexa")
        self.assertEqual(result['confidence'], 0.8)
        
    def test_handle_detect_wake_word_without_confidence(self):
        """Test detecting wake word without confidence."""
        audio_chunk = AudioChunk(
            raw_data=np.zeros(512, dtype=np.int16).tobytes(),
            sample_rate=16000,
            channels=1,
            format='int16',
            timestamp=0
        )
        command = DetectWakeWordCommand(
            audio_chunk=audio_chunk,
            return_confidence=False
        )
        
        # Mock detection result
        self.mock_detector.detect.return_value = (True, "computer")
        
        result = self.handler.handle(command)
        
        self.assertEqual(result['detected'], True)
        self.assertEqual(result['wake_word'], "computer")
        self.assertNotIn('confidence', result)
        
    def test_handle_unsupported_command(self):
        """Test handling unsupported command type."""
        unsupported_command = Mock()
        
        with self.assertRaises(TypeError):
            self.handler.handle(unsupported_command)
            
    def test_audio_chunk_captured_when_detecting(self):
        """Test processing audio chunks during detection."""
        # Start detection
        self.handler.is_detecting = True
        self.handler.state = DetectorState.WAKE_WORD
        
        # Create audio event
        audio_chunk = AudioChunk(
            raw_data=np.zeros(512, dtype=np.int16).tobytes(),
            sample_rate=16000,
            channels=1,
            format='int16',
            timestamp=time.time()
        )
        event = AudioChunkCapturedEvent(audio_chunk)
        
        # Mock wake word detection
        self.mock_detector.detect_with_confidence.return_value = (True, 0.9, "alexa")
        
        # Process audio
        self.handler._on_audio_chunk_captured(event)
        
        # Verify wake word detected
        self.assertTrue(self.handler.wake_word_detected)
        self.assertEqual(self.handler.wake_word_name, "alexa")
        self.assertEqual(self.handler.state, DetectorState.LISTENING)
        
        # Verify event published
        self.event_bus.publish.assert_called()
        published_event = self.event_bus.publish.call_args[0][0]
        self.assertIsInstance(published_event, WakeWordDetectedEvent)
        
    def test_audio_chunk_captured_when_not_detecting(self):
        """Test that audio is ignored when not detecting."""
        self.handler.is_detecting = False
        
        audio_chunk = AudioChunk(
            raw_data=np.zeros(512, dtype=np.int16).tobytes(),
            sample_rate=16000,
            channels=1,
            format='int16',
            timestamp=0
        )
        event = AudioChunkCapturedEvent(audio_chunk)
        
        self.handler._on_audio_chunk_captured(event)
        
        # Should not process audio
        self.mock_detector.detect_with_confidence.assert_not_called()
        
    def test_wake_word_timeout(self):
        """Test wake word timeout after detection."""
        # Set up wake word detected state
        self.handler.is_detecting = True
        self.handler.wake_word_detected = True
        self.handler.wake_word_detected_time = time.time() - 10  # 10 seconds ago
        self.handler.wake_word_name = "alexa"
        self.handler.listening_for_speech = True
        self.handler.state = DetectorState.LISTENING
        self.handler.config.speech_timeout = 5.0  # 5 second timeout
        
        # Process audio chunk
        audio_chunk = AudioChunk(
            raw_data=np.zeros(512, dtype=np.int16).tobytes(),
            sample_rate=16000,
            channels=1,
            format='int16',
            timestamp=time.time()
        )
        event = AudioChunkCapturedEvent(audio_chunk)
        
        self.handler._on_audio_chunk_captured(event)
        
        # Verify timeout occurred
        self.assertFalse(self.handler.wake_word_detected)
        self.assertEqual(self.handler.state, DetectorState.WAKE_WORD)
        
        # Verify timeout event published
        timeout_event_published = False
        for call in self.event_bus.publish.call_args_list:
            if isinstance(call[0][0], WakeWordTimeoutEvent):
                timeout_event_published = True
                timeout_event = call[0][0]
                self.assertEqual(timeout_event.wake_word, "alexa")
                self.assertEqual(timeout_event.timeout_duration, 5.0)
                break
        self.assertTrue(timeout_event_published)
        
    def test_vad_subscription_management(self):
        """Test VAD event subscription/unsubscription."""
        # Initially not subscribed
        self.assertFalse(self.handler.vad_subscribed)
        
        # Enable VAD processing
        self.handler._enable_vad_processing()
        self.assertTrue(self.handler.vad_subscribed)
        
        # Verify subscriptions
        expected_calls = [
            call(SpeechDetectedEvent, self.handler._on_speech_detected),
            call(SilenceDetectedEvent, self.handler._on_silence_detected)
        ]
        for expected_call in expected_calls:
            self.assertIn(expected_call, self.event_bus.subscribe.call_args_list)
        
        # Disable VAD processing
        self.handler._disable_vad_processing()
        self.assertFalse(self.handler.vad_subscribed)
        
        # Verify unsubscriptions
        expected_unsub_calls = [
            call(SpeechDetectedEvent, self.handler._on_speech_detected),
            call(SilenceDetectedEvent, self.handler._on_silence_detected)
        ]
        for expected_call in expected_unsub_calls:
            self.assertIn(expected_call, self.event_bus.unsubscribe.call_args_list)
            
    def test_speech_detected_handling(self):
        """Test handling of speech detected event."""
        self.handler.state = DetectorState.LISTENING
        
        event = SpeechDetectedEvent(
            confidence=0.9,
            audio_timestamp=time.time()
        )
        self.handler._on_speech_detected(event)
        
        self.assertEqual(self.handler.state, DetectorState.RECORDING)
        
    def test_silence_detected_handling(self):
        """Test handling of silence detected event."""
        # Set up recording state
        self.handler.state = DetectorState.RECORDING
        self.handler.wake_word_detected = True
        self.handler.vad_subscribed = True
        
        event = SilenceDetectedEvent(
            audio_reference=[],
            speech_duration=2.5,
            timestamp=time.time()
        )
        self.handler._on_silence_detected(event)
        
        # Verify state reset
        self.assertEqual(self.handler.state, DetectorState.WAKE_WORD)
        self.assertFalse(self.handler.wake_word_detected)
        self.assertFalse(self.handler.listening_for_speech)
        
        # Verify VAD disabled
        self.assertFalse(self.handler.vad_subscribed)
        
    def test_silence_detected_wrong_state(self):
        """Test silence event ignored when not recording."""
        self.handler.state = DetectorState.WAKE_WORD
        initial_state = self.handler.state
        
        event = SilenceDetectedEvent(
            audio_reference=[],
            speech_duration=1.0,
            timestamp=time.time()
        )
        self.handler._on_silence_detected(event)
        
        # State should not change
        self.assertEqual(self.handler.state, initial_state)
        
    def test_get_buffered_audio(self):
        """Test getting buffered audio."""
        # Add some audio chunks to buffer
        chunks = [
            AudioChunk(
                raw_data=np.zeros(512, dtype=np.int16).tobytes(),
                sample_rate=16000,
                channels=1,
                format='int16',
                timestamp=i
            )
            for i in range(5)
        ]
        for chunk in chunks:
            self.handler.audio_buffer.append(chunk)
            
        buffered = self.handler._get_buffered_audio()
        
        self.assertIsNotNone(buffered)
        self.assertEqual(len(buffered), 5)
        self.assertEqual(buffered, chunks)
        
    def test_get_buffered_audio_empty(self):
        """Test getting buffered audio when empty."""
        buffered = self.handler._get_buffered_audio()
        self.assertIsNone(buffered)
        
    def test_cleanup(self):
        """Test handler cleanup."""
        # Set up some state
        self.handler.is_detecting = True
        self.handler.vad_subscribed = True
        self.handler.audio_buffer.append(AudioChunk(
            raw_data=b'data',
            sample_rate=16000,
            channels=1,
            format='int16',
            timestamp=0
        ))
        
        # Add a detector to clean up
        mock_detector = Mock()
        self.handler.detectors['test'] = mock_detector
        
        # Cleanup
        self.handler.cleanup()
        
        # Verify cleanup
        self.assertFalse(self.handler.is_detecting)
        self.assertEqual(self.handler.state, DetectorState.INACTIVE)
        self.assertFalse(self.handler.vad_subscribed)
        self.assertEqual(len(self.handler.audio_buffer), 0)
        self.assertEqual(len(self.handler.detectors), 0)
        
        # Verify detector cleanup called
        mock_detector.cleanup.assert_called_once()
        
        # Verify event unsubscription
        self.event_bus.unsubscribe.assert_called()
        
    def test_wake_word_detection_with_vad_buffer_clear(self):
        """Test wake word detection with VAD buffer clearing."""
        # Configure to exclude pre-wake-word audio
        self.handler.config.exclude_pre_wake_word_audio = True
        self.handler.is_detecting = True
        self.handler.state = DetectorState.WAKE_WORD
        
        # Create audio event
        audio_chunk = AudioChunk(
            raw_data=np.zeros(512, dtype=np.int16).tobytes(),
            sample_rate=16000,
            channels=1,
            format='int16',
            timestamp=time.time()
        )
        event = AudioChunkCapturedEvent(audio_chunk)
        
        # Mock wake word detection
        self.mock_detector.detect_with_confidence.return_value = (True, 0.9, "alexa")
        
        # Process audio
        self.handler._on_audio_chunk_captured(event)
        
        # Verify VAD buffer clear command dispatched
        clear_buffer_dispatched = False
        for call in self.command_dispatcher.dispatch.call_args_list:
            if hasattr(call[0][0], '__class__') and call[0][0].__class__.__name__ == 'ClearVadPreSpeechBufferCommand':
                clear_buffer_dispatched = True
                break
        self.assertTrue(clear_buffer_dispatched)


if __name__ == '__main__':
    unittest.main()