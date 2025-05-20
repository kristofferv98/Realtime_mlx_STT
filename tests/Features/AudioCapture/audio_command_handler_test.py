"""
Test for the AudioCommandHandler class.

This test verifies that the AudioCommandHandler correctly processes audio commands
and interacts with audio providers.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.Core.Events.event_bus import EventBus
from src.Core.Common.Interfaces.audio_provider import IAudioProvider
from src.Features.AudioCapture.Handlers.AudioCommandHandler import AudioCommandHandler
from src.Features.AudioCapture.Commands.ListDevicesCommand import ListDevicesCommand
from src.Features.AudioCapture.Commands.SelectDeviceCommand import SelectDeviceCommand
from src.Features.AudioCapture.Commands.StartRecordingCommand import StartRecordingCommand
from src.Features.AudioCapture.Commands.StopRecordingCommand import StopRecordingCommand


class AudioCommandHandlerTest(unittest.TestCase):
    """Test suite for AudioCommandHandler."""
    
    def setUp(self):
        """Set up test environment."""
        # Create an event bus
        self.event_bus = EventBus()
        
        # Create mock providers
        self.mic_provider = MagicMock(spec=IAudioProvider)
        self.file_provider = MagicMock(spec=IAudioProvider)
        
        # Set up mock provider behavior
        self.mic_provider.setup.return_value = True
        self.mic_provider.start.return_value = True
        self.mic_provider.stop.return_value = True
        self.mic_provider.list_devices.return_value = [
            {'device_id': 0, 'name': 'Default Mic', 'is_default': True},
            {'device_id': 1, 'name': 'External Mic', 'is_default': False}
        ]
        
        self.file_provider.setup.return_value = True
        self.file_provider.start.return_value = True
        self.file_provider.stop.return_value = True
        self.file_provider.list_devices.return_value = [
            {'device_id': 2, 'name': 'Test File', 'is_default': True}
        ]
        
        # Create providers dictionary
        self.providers = {
            'microphone': self.mic_provider,
            'file': self.file_provider
        }
        
        # Create command handler
        self.handler = AudioCommandHandler(
            event_bus=self.event_bus,
            providers=self.providers,
            active_provider='microphone'
        )
    
    def test_handle_list_devices_command(self):
        """Test that the handler can list devices from all providers."""
        # Create command
        command = ListDevicesCommand()
        
        # Execute command
        devices = self.handler.handle(command)
        
        # Assertions
        self.assertEqual(len(devices), 3)  # Total from both providers
        
        # Verify both providers were called
        self.mic_provider.list_devices.assert_called_once()
        self.file_provider.list_devices.assert_called_once()
        
        # Check that provider info was added
        self.assertEqual(devices[0]['provider'], 'microphone')
        self.assertEqual(devices[2]['provider'], 'file')
    
    def test_handle_select_device_command(self):
        """Test that the handler can select a device."""
        # Create command for microphone device
        command = SelectDeviceCommand(device_id=1)
        
        # Execute command
        result = self.handler.handle(command)
        
        # Assertions
        self.assertTrue(result)
        self.assertEqual(self.handler.active_provider, 'microphone')
        
        # Try selecting a file device
        command = SelectDeviceCommand(device_id=2)
        result = self.handler.handle(command)
        
        # Verify provider was changed
        self.assertTrue(result)
        self.assertEqual(self.handler.active_provider, 'file')
        
        # Try selecting an invalid device
        command = SelectDeviceCommand(device_id=999)
        result = self.handler.handle(command)
        
        # Verify failure
        self.assertFalse(result)
    
    def test_handle_start_recording_command(self):
        """Test that the handler can start recording."""
        # Create command
        command = StartRecordingCommand(
            sample_rate=16000,
            chunk_size=512,
            device_id=None  # Use current active provider/device
        )
        
        # Execute command
        result = self.handler.handle(command)
        
        # Assertions
        self.assertTrue(result)
        
        # Verify the active provider was set up and started
        self.mic_provider.setup.assert_called_once()
        self.mic_provider.start.assert_called_once()
    
    def test_handle_stop_recording_command(self):
        """Test that the handler can stop recording."""
        # Create command
        command = StopRecordingCommand(
            save_recording=False,
            flush_buffer=False
        )
        
        # Execute command
        result = self.handler.handle(command)
        
        # Assertions
        self.assertTrue(result)
        
        # Verify the active provider was stopped
        self.mic_provider.stop.assert_called_once()
    
    def test_stop_recording_with_options(self):
        """Test stopping recording with additional options."""
        # Create command with options
        command = StopRecordingCommand(
            save_recording=True,
            output_path="test_output.wav",
            flush_buffer=True
        )
        
        # Add mock methods to provider
        self.mic_provider.save_recording = MagicMock(return_value=True)
        self.mic_provider.flush_buffer = MagicMock()
        
        # Execute command
        result = self.handler.handle(command)
        
        # Assertions
        self.assertTrue(result)
        self.mic_provider.stop.assert_called_once()
        self.mic_provider.flush_buffer.assert_called_once()
        self.mic_provider.save_recording.assert_called_once_with("test_output.wav")
    
    def test_handle_unknown_command(self):
        """Test that the handler rejects unknown commands."""
        # Create a mock command that's not supported
        command = MagicMock()
        
        # Execute command and check for exception
        with self.assertRaises(TypeError):
            self.handler.handle(command)
    
    def test_can_handle(self):
        """Test that the handler correctly identifies commands it can handle."""
        # Test supported commands
        self.assertTrue(self.handler.can_handle(ListDevicesCommand()))
        self.assertTrue(self.handler.can_handle(SelectDeviceCommand(device_id=0)))
        self.assertTrue(self.handler.can_handle(StartRecordingCommand()))
        self.assertTrue(self.handler.can_handle(StopRecordingCommand()))
        
        # Test unsupported command
        mock_command = MagicMock()
        self.assertFalse(self.handler.can_handle(mock_command))


if __name__ == '__main__':
    unittest.main()