"""
Tests for ServerModule class
"""

import unittest
from unittest.mock import MagicMock, patch

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Application.Server.ServerModule import ServerModule, Server
from src.Application.Server.Configuration.ServerConfig import ServerConfig

class TestServerModule(unittest.TestCase):
    """Test cases for ServerModule class"""

    def setUp(self):
        """Set up test environment before each test case"""
        # Create mock command dispatcher and event bus
        self.mock_command_dispatcher = MagicMock(spec=CommandDispatcher)
        self.mock_event_bus = MagicMock(spec=EventBus)
    
    @patch('src.Application.Server.ServerModule.Server')
    def test_register(self, mock_server_class):
        """Test registering the server module"""
        # Set up mock server instance
        mock_server_instance = MagicMock(spec=Server)
        mock_server_class.return_value = mock_server_instance
        
        # Call register method
        result = ServerModule.register(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus
        )
        
        # Check that Server was created with correct parameters
        mock_server_class.assert_called_once()
        call_args = mock_server_class.call_args[1]
        self.assertEqual(call_args['command_dispatcher'], self.mock_command_dispatcher)
        self.assertEqual(call_args['event_bus'], self.mock_event_bus)
        
        # Check that server.start() was called (auto_start is True by default)
        mock_server_instance.start.assert_called_once()
        
        # Check that the server instance was returned
        self.assertEqual(result, mock_server_instance)
    
    @patch('src.Application.Server.ServerModule.Server')
    def test_register_with_config(self, mock_server_class):
        """Test registering with a custom configuration"""
        # Set up mock server instance
        mock_server_instance = MagicMock(spec=Server)
        mock_server_class.return_value = mock_server_instance
        
        # Create custom config
        config = ServerConfig()
        config.host = "custom-host"
        config.port = 9999
        config.auto_start = False
        
        # Call register method with custom config
        result = ServerModule.register(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus,
            config=config
        )
        
        # Check that Server was created with custom config values
        call_args = mock_server_class.call_args[1]
        self.assertEqual(call_args['host'], "custom-host")
        self.assertEqual(call_args['port'], 9999)
        
        # Auto-start was set to False, so start() should not be called
        mock_server_instance.start.assert_not_called()

class TestServer(unittest.TestCase):
    """Test cases for Server class"""

    def setUp(self):
        """Set up test environment before each test case"""
        # Create mock command dispatcher and event bus
        self.mock_command_dispatcher = MagicMock(spec=CommandDispatcher)
        self.mock_event_bus = MagicMock(spec=EventBus)
        
        # Patch FastAPI
        self.patcher = patch('src.Application.Server.ServerModule.FastAPI')
        self.mock_fastapi = self.patcher.start()
        self.mock_app = MagicMock()
        self.mock_fastapi.return_value = self.mock_app
        
        # Patch WebSocketManager
        self.ws_patcher = patch('src.Application.Server.ServerModule.WebSocketManager')
        self.mock_ws_manager = self.ws_patcher.start()
        
        # Patch ProfileManager
        self.pm_patcher = patch('src.Application.Server.ServerModule.ProfileManager')
        self.mock_profile_manager = self.pm_patcher.start()
        
        # Patch controller classes
        self.tc_patcher = patch('src.Application.Server.ServerModule.TranscriptionController')
        self.mock_transcription_controller = self.tc_patcher.start()
        
        self.sc_patcher = patch('src.Application.Server.ServerModule.SystemController')
        self.mock_system_controller = self.sc_patcher.start()
        
        # Patch threading.Thread
        self.thread_patcher = patch('src.Application.Server.ServerModule.threading.Thread')
        self.mock_thread = self.thread_patcher.start()
    
    def tearDown(self):
        """Clean up after each test case"""
        self.patcher.stop()
        self.ws_patcher.stop()
        self.pm_patcher.stop()
        self.tc_patcher.stop()
        self.sc_patcher.stop()
        self.thread_patcher.stop()
    
    def test_initialization(self):
        """Test server initialization"""
        # Create server
        server = Server(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus,
            host="test-host",
            port=1234
        )
        
        # Check that FastAPI was created and middleware was set up
        self.mock_fastapi.assert_called_once()
        self.mock_app.add_middleware.assert_called_once()
        
        # Check that attributes were set correctly
        self.assertEqual(server.host, "test-host")
        self.assertEqual(server.port, 1234)
        self.assertEqual(server.command_dispatcher, self.mock_command_dispatcher)
        self.assertEqual(server.event_bus, self.mock_event_bus)
        self.assertFalse(server.running)
    
    def test_register_controllers(self):
        """Test that controllers are registered correctly"""
        # Create server
        server = Server(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus
        )
        
        # Check that controllers were created
        self.mock_transcription_controller.assert_called_once_with(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus
        )
        
        self.mock_system_controller.assert_called_once()
        
        # Check that controller routers were included
        mock_tc_instance = self.mock_transcription_controller.return_value
        mock_sc_instance = self.mock_system_controller.return_value
        
        self.mock_app.include_router.assert_any_call(mock_tc_instance.router)
        self.mock_app.include_router.assert_any_call(mock_sc_instance.router)
    
    def test_register_event_handlers(self):
        """Test that event handlers are registered correctly"""
        # Create server
        server = Server(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus
        )
        
        # Check that event handlers were subscribed
        from src.Features.Transcription.Events.TranscriptionUpdatedEvent import TranscriptionUpdatedEvent
        from src.Features.WakeWordDetection.Events.WakeWordDetectedEvent import WakeWordDetectedEvent
        
        # Simplified assertions that only check if subscribe was called with the right event types
        # instead of complex method references
        self.mock_event_bus.subscribe.assert_any_call(TranscriptionUpdatedEvent, server.handle_transcription_update)
        self.mock_event_bus.subscribe.assert_any_call(WakeWordDetectedEvent, server.handle_wake_word_detected)
    
    @patch('src.Application.Server.ServerModule.uvicorn')
    def test_start_and_stop(self, mock_uvicorn):
        """Test starting and stopping the server"""
        # Create server
        server = Server(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus
        )
        
        # Start the server
        server.start()
        
        # Check that thread was created and started
        self.mock_thread.assert_called_once()
        mock_thread_instance = self.mock_thread.return_value
        mock_thread_instance.start.assert_called_once()
        
        # Check that running flag was set
        self.assertTrue(server.running)
        
        # Stop the server
        server.stop()
        
        # Check that running flag was cleared
        self.assertFalse(server.running)
    
    def test_handle_events(self):
        """Test handling events and broadcasting to WebSocket clients"""
        # Create server
        server = Server(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus
        )
        
        # Create mock events
        mock_transcription_event = MagicMock()
        mock_transcription_event.text = "sample text"
        mock_transcription_event.is_final = True
        mock_transcription_event.session_id = "test-session"
        
        mock_wake_word_event = MagicMock()
        mock_wake_word_event.word = "test-word"
        mock_wake_word_event.timestamp = 123.45
        
        # Get the WebSocketManager instance
        ws_manager_instance = self.mock_ws_manager.return_value
        
        # Call event handlers
        server.handle_transcription_update(mock_transcription_event)
        server.handle_wake_word_detected(mock_wake_word_event)
        
        # Check that broadcast_event was called with correct data
        ws_manager_instance.broadcast_event.assert_any_call(
            "transcription", 
            {
                "text": "sample text",
                "is_final": True,
                "session_id": "test-session"
            }
        )
        
        ws_manager_instance.broadcast_event.assert_any_call(
            "wake_word", 
            {
                "word": "test-word",
                "timestamp": 123.45
            }
        )

if __name__ == '__main__':
    unittest.main()