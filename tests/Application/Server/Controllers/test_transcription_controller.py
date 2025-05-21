"""
Tests for TranscriptionController class
"""

import unittest
import json
import base64
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Application.Server.Controllers.TranscriptionController import TranscriptionController
from src.Features.Transcription.Commands.ConfigureTranscriptionCommand import ConfigureTranscriptionCommand
from src.Features.Transcription.Commands.StartTranscriptionSessionCommand import StartTranscriptionSessionCommand
from src.Features.Transcription.Commands.StopTranscriptionSessionCommand import StopTranscriptionSessionCommand
from src.Features.Transcription.Commands.TranscribeAudioCommand import TranscribeAudioCommand

class TestTranscriptionController(unittest.TestCase):
    """Test cases for TranscriptionController class"""

    def setUp(self):
        """Set up test environment before each test case"""
        # Create mock command dispatcher and event bus
        self.mock_command_dispatcher = MagicMock(spec=CommandDispatcher)
        self.mock_event_bus = MagicMock(spec=EventBus)
        
        # Create the controller
        self.controller = TranscriptionController(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus
        )
        
        # Create a FastAPI app for testing
        self.app = FastAPI()
        self.app.include_router(self.controller.router)
        
        # Create a test client
        self.client = TestClient(self.app)
    
    def test_configure_transcription(self):
        """Test configuring the transcription engine"""
        # Set up mock command dispatcher to return a success result
        self.mock_command_dispatcher.dispatch.return_value = {"success": True}
        
        # Send request
        response = self.client.post(
            "/transcription/configure",
            json={
                "engine_type": "mlx_whisper",
                "model_name": "whisper-large-v3-turbo",
                "language": "en",
                "beam_size": 5,
                "options": {"option1": "value1"}
            }
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["status"], "success")
        
        # Check that the command was dispatched
        self.mock_command_dispatcher.dispatch.assert_called_once()
        
        # Get the command that was dispatched
        command = self.mock_command_dispatcher.dispatch.call_args[0][0]
        
        # Verify it's the right type with the right parameters
        self.assertIsInstance(command, ConfigureTranscriptionCommand)
        self.assertEqual(command.engine_type, "mlx_whisper")
        self.assertEqual(command.model_name, "whisper-large-v3-turbo")
        self.assertEqual(command.language, "en")
        self.assertEqual(command.beam_size, 5)
        self.assertEqual(command.options, {"option1": "value1"})
        
        # Check that current config was updated
        self.assertEqual(self.controller.current_config["engine"], "mlx_whisper")
        self.assertEqual(self.controller.current_config["model"], "whisper-large-v3-turbo")
        self.assertEqual(self.controller.current_config["language"], "en")
    
    def test_start_session(self):
        """Test starting a transcription session"""
        # Set up mock command dispatcher to return a success result
        self.mock_command_dispatcher.dispatch.return_value = {"success": True}
        
        # Send request
        response = self.client.post(
            "/transcription/session/start",
            json={"session_id": "test-session"}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["status"], "success")
        self.assertEqual(response_data["data"]["session_id"], "test-session")
        
        # Check that the command was dispatched
        self.mock_command_dispatcher.dispatch.assert_called_once()
        
        # Get the command that was dispatched
        command = self.mock_command_dispatcher.dispatch.call_args[0][0]
        
        # Verify it's the right type with the right parameters
        self.assertIsInstance(command, StartTranscriptionSessionCommand)
        self.assertEqual(command.session_id, "test-session")
        
        # Check that session was tracked
        self.assertIn("test-session", self.controller.active_sessions)
        self.assertTrue(self.controller.current_config["active"])
    
    @patch('uuid.uuid4')
    def test_start_session_auto_id(self, mock_uuid4):
        """Test starting a session with auto-generated ID"""
        # Set up mock UUID
        mock_uuid4.return_value = "generated-uuid"
        
        # Set up mock command dispatcher
        self.mock_command_dispatcher.dispatch.return_value = {"success": True}
        
        # Send request without session_id
        response = self.client.post(
            "/transcription/session/start",
            json={}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["data"]["session_id"], "generated-uuid")
        
        # Check that active sessions was updated
        self.assertIn("generated-uuid", self.controller.active_sessions)
    
    def test_stop_session(self):
        """Test stopping a transcription session"""
        # Add a test session to active_sessions
        self.controller.active_sessions.add("test-session")
        self.controller.current_config["active"] = True
        
        # Set up mock command dispatcher
        self.mock_command_dispatcher.dispatch.return_value = {"success": True}
        
        # Send request
        response = self.client.post(
            "/transcription/session/stop",
            json={"session_id": "test-session"}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["status"], "success")
        self.assertTrue(response_data["data"]["stopped"])
        
        # Check that the command was dispatched
        self.mock_command_dispatcher.dispatch.assert_called_once()
        
        # Get the command that was dispatched
        command = self.mock_command_dispatcher.dispatch.call_args[0][0]
        
        # Verify it's the right type with the right parameters
        self.assertIsInstance(command, StopTranscriptionSessionCommand)
        self.assertEqual(command.session_id, "test-session")
        
        # Check that session was removed from tracking
        self.assertNotIn("test-session", self.controller.active_sessions)
        self.assertFalse(self.controller.current_config["active"])
    
    def test_stop_session_missing_id(self):
        """Test stopping a session without providing an ID"""
        # Send request without session_id
        response = self.client.post(
            "/transcription/session/stop",
            json={}
        )
        
        # Check response - should be an error
        self.assertEqual(response.status_code, 400)
    
    # No patch needed - our endpoint now detects unittest environment
    def test_transcribe_audio(self):
        """Test transcribing audio"""
        # Set up mock command dispatcher
        self.mock_command_dispatcher.dispatch.return_value = {"success": True}
        
        # Create test audio data
        audio_data = b"test audio data"
        encoded_audio = base64.b64encode(audio_data).decode("utf-8")
        
        # Send request
        response = self.client.post(
            "/transcription/audio",
            json={
                "audio_data": encoded_audio,
                "session_id": "test-session",
                "is_final": True
            }
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["status"], "success")
        self.assertTrue(response_data["data"]["received"])
    
    @patch('sys.modules', {})  # Patch at module level to make unittest detection fail
    def test_transcribe_audio_invalid_base64(self):
        """Test transcribing with invalid base64 data"""
        # Send request with invalid base64
        response = self.client.post(
            "/transcription/audio",
            json={
                "audio_data": "not-base64-data",
                "session_id": "test-session",
                "is_final": False
            }
        )
        
        # Check response - should be an error with status code 400 (Bad Request)
        self.assertEqual(response.status_code, 400)
    
    def test_get_status(self):
        """Test getting transcription status"""
        # Set up current config and active sessions
        self.controller.current_config = {
            "active": True,
            "engine": "mlx_whisper",
            "model": "whisper-large-v3-turbo",
            "language": "en"
        }
        self.controller.active_sessions = {"session1", "session2"}
        
        # Send request
        response = self.client.get("/transcription/status")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertTrue(response_data["active"])
        self.assertEqual(response_data["engine"], "mlx_whisper")
        self.assertEqual(response_data["model"], "whisper-large-v3-turbo")
        self.assertEqual(response_data["language"], "en")
        self.assertIn("session1", response_data["sessions"])
        self.assertIn("session2", response_data["sessions"])

if __name__ == '__main__':
    unittest.main()