"""
Tests for WebSocketManager class
"""

import unittest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.Application.Server.WebSocket.WebSocketManager import WebSocketManager

class TestWebSocketManager(unittest.TestCase):
    """Test cases for WebSocketManager class"""

    def setUp(self):
        """Set up test environment before each test case"""
        self.manager = WebSocketManager()
        
        # Create mock WebSocket
        self.mock_websocket = AsyncMock()
        self.mock_websocket.send_json = AsyncMock()
    
    def test_register_unregister(self):
        """Test registering and unregistering clients"""
        # Register client
        self.manager.register(self.mock_websocket)
        self.assertIn(self.mock_websocket, self.manager.active_connections)
        
        # Unregister client
        self.manager.unregister(self.mock_websocket)
        self.assertNotIn(self.mock_websocket, self.manager.active_connections)
    
    def test_unregister_nonexistent(self):
        """Test unregistering a client that isn't registered"""
        # Create a WebSocket that isn't registered
        mock_unregistered = AsyncMock()
        
        # Unregister it (should not raise errors)
        self.manager.unregister(mock_unregistered)
    
    async def async_test_send_personal_message(self):
        """Test sending a message to a specific client"""
        # Register client
        self.manager.register(self.mock_websocket)
        
        # Send message
        message = {"test": "message"}
        await self.manager.send_personal_message(message, self.mock_websocket)
        
        # Check that send_json was called
        self.mock_websocket.send_json.assert_called_once_with(message)
    
    def test_send_personal_message(self):
        """Run the async test for sending a personal message"""
        asyncio.run(self.async_test_send_personal_message())
    
    async def async_test_send_personal_message_error(self):
        """Test handling errors when sending a personal message"""
        # Register client
        self.manager.register(self.mock_websocket)
        
        # Set up mock to raise an exception
        self.mock_websocket.send_json.side_effect = Exception("Test error")
        
        # Send message (should not raise exception)
        message = {"test": "message"}
        await self.manager.send_personal_message(message, self.mock_websocket)
        
        # Check that client was unregistered
        self.assertNotIn(self.mock_websocket, self.manager.active_connections)
    
    def test_send_personal_message_error(self):
        """Run the async test for error handling in personal messages"""
        asyncio.run(self.async_test_send_personal_message_error())
    
    async def async_test_broadcast(self):
        """Test broadcasting a message to all connected clients"""
        # Create multiple clients
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        
        # Register clients
        self.manager.register(websocket1)
        self.manager.register(websocket2)
        
        # Broadcast message
        message = {"event": "test", "data": "value"}
        await self.manager._broadcast(message)
        
        # Check that send_json was called on both clients
        websocket1.send_json.assert_called_once_with(message)
        websocket2.send_json.assert_called_once_with(message)
    
    def test_broadcast(self):
        """Run the async test for broadcasting"""
        asyncio.run(self.async_test_broadcast())
    
    async def async_test_broadcast_error(self):
        """Test error handling during broadcasting"""
        # Create multiple clients
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        
        # Second client will raise an error
        websocket2.send_json.side_effect = Exception("Test error")
        
        # Register clients
        self.manager.register(websocket1)
        self.manager.register(websocket2)
        
        # Broadcast message
        message = {"event": "test", "data": "value"}
        await self.manager._broadcast(message)
        
        # Check that the error client was unregistered
        self.assertIn(websocket1, self.manager.active_connections)
        self.assertNotIn(websocket2, self.manager.active_connections)
    
    def test_broadcast_error(self):
        """Run the async test for error handling in broadcasting"""
        asyncio.run(self.async_test_broadcast_error())
    
    @patch('asyncio.get_event_loop')
    def test_broadcast_event(self, mock_get_event_loop):
        """Test the broadcast_event method"""
        # Set up mock event loop
        mock_loop = MagicMock()
        mock_get_event_loop.return_value = mock_loop
        
        # Call broadcast_event
        self.manager.broadcast_event("test_event", {"data": "value"})
        
        # Check that create_task was called
        mock_loop.create_task.assert_called_once()
        
        # The argument to create_task should be a coroutine from _broadcast
        # We can't easily check the content, but we can verify it was called
        self.assertEqual(mock_loop.create_task.call_count, 1)
    
    @patch('asyncio.get_event_loop')
    def test_broadcast_event_no_event_loop(self, mock_get_event_loop):
        """Test handling case when no event loop is available"""
        # Set up mock to raise RuntimeError
        mock_get_event_loop.side_effect = RuntimeError("No event loop")
        
        # Call broadcast_event - should not raise exception
        self.manager.broadcast_event("test_event", {"data": "value"})

if __name__ == '__main__':
    unittest.main()