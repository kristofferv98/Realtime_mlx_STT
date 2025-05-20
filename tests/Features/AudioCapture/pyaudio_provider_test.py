"""
Test for the PyAudioInputProvider class.

This test verifies that the PyAudioInputProvider correctly initializes,
lists devices, and handles basic recording operations.
"""

import unittest
import time
from unittest.mock import MagicMock, patch

from src.Core.Events.event_bus import EventBus
from src.Features.AudioCapture.Providers.PyAudioInputProvider import PyAudioInputProvider
from src.Features.AudioCapture.Events.RecordingStateChangedEvent import RecordingState, RecordingStateChangedEvent
from src.Features.AudioCapture.Events.AudioChunkCapturedEvent import AudioChunkCapturedEvent

# All tests are now working with proper mocking of PyAudio dependencies


class PyAudioInputProviderTest(unittest.TestCase):
    """Test suite for PyAudioInputProvider."""
    
    def setUp(self):
        """Set up test environment."""
        # Create an event bus
        self.event_bus = EventBus()
        
        # Create tracking variables
        self.audio_chunks_received = 0
        self.last_state_change = None
        
        # Set up event listeners
        self.event_bus.subscribe(
            AudioChunkCapturedEvent,
            lambda event: self._on_audio_chunk(event)
        )
        self.event_bus.subscribe(
            RecordingStateChangedEvent,
            lambda event: self._on_state_changed(event)
        )
    
    def _on_audio_chunk(self, event):
        """Handle audio chunk events."""
        self.audio_chunks_received += 1
    
    def _on_state_changed(self, event):
        """Handle state change events."""
        self.last_state_change = (event.previous_state, event.current_state)
    
    @patch('pyaudio.PyAudio')
    def test_initialization(self, mock_pyaudio):
        """Test that the provider initializes correctly."""
        # Setup mock
        mock_instance = mock_pyaudio.return_value
        mock_instance.get_default_input_device_info.return_value = {
            'index': 0,
            'name': 'Test Microphone',
            'maxInputChannels': 2,
            'defaultSampleRate': 44100
        }
        
        # Create provider
        provider = PyAudioInputProvider(
            event_bus=self.event_bus,
            sample_rate=16000,
            chunk_size=512
        )
        
        # Test setup
        result = provider.setup()
        
        # Assertions
        self.assertTrue(result)
        self.assertEqual(provider.device_id, 0)
        self.assertEqual(provider.target_sample_rate, 16000)
        self.assertEqual(provider.chunk_size, 512)
        self.assertEqual(self.last_state_change, 
                         (RecordingState.INITIALIZED, RecordingState.INITIALIZED))
        
        # Verify PyAudio was initialized properly
        mock_pyaudio.assert_called_once()
        mock_instance.get_default_input_device_info.assert_called_once()
    
    @patch('pyaudio.PyAudio')
    def test_list_devices(self, mock_pyaudio):
        """Test that the provider can list devices."""
        # Setup mocks
        mock_instance = mock_pyaudio.return_value
        mock_instance.get_device_count.return_value = 2
        mock_instance.get_default_input_device_info.return_value = {'index': 0}
        
        # Mock device info
        def mock_get_device_info(index):
            if index == 0:
                return {
                    'index': 0,
                    'name': 'Default Microphone',
                    'maxInputChannels': 2,
                    'defaultSampleRate': 44100
                }
            else:
                return {
                    'index': 1,
                    'name': 'External Microphone',
                    'maxInputChannels': 1,
                    'defaultSampleRate': 48000
                }
        
        mock_instance.get_device_info_by_index.side_effect = mock_get_device_info
        mock_instance.is_format_supported.return_value = True
        
        # Create provider
        provider = PyAudioInputProvider(event_bus=self.event_bus)
        
        # Test list_devices
        devices = provider.list_devices()
        
        # Assertions
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]['name'], 'Default Microphone')
        self.assertEqual(devices[1]['name'], 'External Microphone')
        self.assertTrue(devices[0]['is_default'])
        self.assertFalse(devices[1]['is_default'])
    
    @patch('pyaudio.PyAudio')
    def test_start_stop_recording(self, mock_pyaudio):
        """Test that the provider can start and stop recording."""
        # Setup mocks
        mock_instance = mock_pyaudio.return_value
        mock_instance.get_default_input_device_info.return_value = {'index': 0}
        
        # Create a mock stream that returns real bytes data
        mock_stream = MagicMock()
        mock_stream.read.return_value = b'\x00\x00' * 512  # Simulate silence
        mock_instance.open.return_value = mock_stream
        
        # Patch read_chunk to return bytes instead of using the stream
        # This avoids threading issues and makes the test more reliable
        with patch.object(PyAudioInputProvider, 'read_chunk', return_value=b'\x00\x00' * 512):
            # Create provider
            provider = PyAudioInputProvider(event_bus=self.event_bus)
            provider.setup()
            
            # Reset state tracking
            self.last_state_change = None
            
            # Test start recording
            start_result = provider.start()
            
            # Verify start recording
            self.assertTrue(start_result)
            self.assertTrue(provider.is_recording)
            self.assertIsNotNone(provider.recording_thread)
            self.assertEqual(self.last_state_change[1], RecordingState.RECORDING)
            
            # Small delay to let recording thread run
            time.sleep(0.1)
            
            # Reset state tracking
            self.last_state_change = None
            
            # Test stop recording
            stop_result = provider.stop()
            
            # Verify stop recording
            self.assertTrue(stop_result)
            self.assertFalse(provider.is_recording)
            self.assertEqual(self.last_state_change[1], RecordingState.STOPPED)
            
            # Verify stream was closed
            mock_stream.close.assert_called_once()

    @patch('pyaudio.PyAudio')
    def test_read_chunk(self, mock_pyaudio):
        """Test that the provider can read audio chunks."""
        # Setup mocks
        mock_instance = mock_pyaudio.return_value
        mock_instance.get_default_input_device_info.return_value = {'index': 0}
        mock_stream = MagicMock()
        silence_bytes = b'\x00\x00' * 512  # Simulate silence
        mock_stream.read.return_value = silence_bytes
        mock_instance.open.return_value = mock_stream
        
        # Create provider with actual stream mocking only
        # Don't start the recording thread to avoid threading issues
        provider = PyAudioInputProvider(event_bus=self.event_bus)
        provider.setup()
        
        # Manually set the stream
        provider.stream = mock_stream
        provider.is_recording = True  # Pretend we're recording
        
        # Test read_chunk directly
        chunk = provider.read_chunk()
        
        # Assertions
        self.assertEqual(chunk, silence_bytes)
        self.assertEqual(len(chunk), 1024)  # 512 samples * 2 bytes per sample
        mock_stream.read.assert_called_with(512, exception_on_overflow=False)
        
        # Clean up
        provider.is_recording = False


if __name__ == '__main__':
    unittest.main()