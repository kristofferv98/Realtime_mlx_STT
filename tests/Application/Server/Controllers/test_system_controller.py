"""
Tests for SystemController class
"""

import unittest
import json
import time
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus
from src.Application.Server.Controllers.SystemController import SystemController
from src.Application.Server.Configuration.ProfileManager import ProfileManager

class TestSystemController(unittest.TestCase):
    """Test cases for SystemController class"""

    def setUp(self):
        """Set up test environment before each test case"""
        # Create mock command dispatcher and event bus
        self.mock_command_dispatcher = MagicMock(spec=CommandDispatcher)
        self.mock_event_bus = MagicMock(spec=EventBus)
        
        # Create mock profile manager
        self.mock_profile_manager = MagicMock(spec=ProfileManager)
        
        # Create the controller
        self.controller = SystemController(
            command_dispatcher=self.mock_command_dispatcher,
            event_bus=self.mock_event_bus,
            profile_manager=self.mock_profile_manager
        )
        
        # Set controller start time and version
        self.controller.start_time = time.time() - 100  # Pretend server started 100 seconds ago
        self.controller.version = "1.0.0"
        self.controller.active_features = ["transcription", "vad"]
        
        # Create a FastAPI app for testing
        self.app = FastAPI()
        self.app.include_router(self.controller.router)
        
        # Create a test client
        self.client = TestClient(self.app)
    
    def test_get_status(self):
        """Test getting system status"""
        # Send request
        response = self.client.get("/system/status")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["status"], "online")
        self.assertEqual(response_data["version"], "1.0.0")
        
        # Uptime should be around 100 seconds (give a little wiggle room)
        self.assertTrue(95 <= response_data["uptime"] <= 105)
        
        self.assertEqual(response_data["active_features"], ["transcription", "vad"])
    
    def test_get_info(self):
        """Test getting system information"""
        # Send request
        response = self.client.get("/system/info")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["version"], "1.0.0")
        self.assertIn("platform", response_data)
        self.assertIn("python_version", response_data)
        self.assertIn("cpu_count", response_data)
        self.assertIn("features", response_data)
    
    def test_list_profiles(self):
        """Test listing available profiles"""
        # Set up mock profile manager
        self.mock_profile_manager.list_profiles.return_value = [
            "profile1", "profile2", "vad-triggered"
        ]
        self.mock_profile_manager.PREDEFINED_PROFILES = {
            "default": "vad-triggered"
        }
        
        # Send request
        response = self.client.get("/system/profiles")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["profiles"], ["profile1", "profile2", "vad-triggered"])
        self.assertEqual(response_data["default"], "vad-triggered")
    
    def test_get_profile(self):
        """Test getting a specific profile"""
        # Set up mock profile manager
        test_profile = {
            "transcription": {"engine": "test-engine"},
            "vad": {"enabled": True}
        }
        self.mock_profile_manager.get_profile.return_value = test_profile
        
        # Send request
        response = self.client.get("/system/profiles/test-profile")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["name"], "test-profile")
        self.assertEqual(response_data["config"], test_profile)
        
        # Check that profile manager was called with correct name
        self.mock_profile_manager.get_profile.assert_called_once_with("test-profile")
    
    def test_get_nonexistent_profile(self):
        """Test getting a profile that doesn't exist"""
        # Set up mock profile manager
        self.mock_profile_manager.get_profile.return_value = None
        
        # Send request
        response = self.client.get("/system/profiles/nonexistent")
        
        # Check response - should be an error
        self.assertEqual(response.status_code, 404)
    
    def test_save_profile(self):
        """Test saving a profile"""
        # Set up mock profile manager
        self.mock_profile_manager.save_profile.return_value = True
        
        # Prepare profile data
        profile_data = {
            "name": "new-profile",
            "config": {
                "transcription": {"engine": "new-engine"},
                "vad": {"enabled": False}
            }
        }
        
        # Send request
        response = self.client.post("/system/profiles", json=profile_data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["status"], "success")
        self.assertTrue(response_data["data"]["saved"])
        
        # Check that profile manager was called with correct parameters
        self.mock_profile_manager.save_profile.assert_called_once_with(
            "new-profile", profile_data["config"]
        )
    
    def test_save_profile_failure(self):
        """Test handling a failure when saving a profile"""
        # Set up mock profile manager
        self.mock_profile_manager.save_profile.return_value = False
        
        # Prepare profile data
        profile_data = {
            "name": "failed-profile",
            "config": {"test": "data"}
        }
        
        # Send request
        response = self.client.post("/system/profiles", json=profile_data)
        
        # Check response - should be an error
        self.assertEqual(response.status_code, 400)
    
    def test_delete_profile(self):
        """Test deleting a profile"""
        # Set up mock profile manager
        self.mock_profile_manager.delete_profile.return_value = True
        
        # Send request
        response = self.client.delete("/system/profiles/profile-to-delete")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["status"], "success")
        self.assertTrue(response_data["data"]["deleted"])
        
        # Check that profile manager was called with correct name
        self.mock_profile_manager.delete_profile.assert_called_once_with("profile-to-delete")
    
    def test_delete_profile_failure(self):
        """Test handling a failure when deleting a profile"""
        # Set up mock profile manager
        self.mock_profile_manager.delete_profile.return_value = False
        
        # Send request
        response = self.client.delete("/system/profiles/cannot-delete")
        
        # Check response - should be an error
        self.assertEqual(response.status_code, 400)
    
    def test_start_system(self):
        """Test starting the system with a profile"""
        # Set up mock profile manager
        test_profile = {
            "transcription": {"engine": "test-engine"},
            "vad": {"enabled": True}
        }
        self.mock_profile_manager.get_profile.return_value = test_profile
        
        # Send request
        response = self.client.post(
            "/system/start",
            json={"profile": "test-profile"}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["status"], "success")
        self.assertTrue(response_data["data"]["started"])
        self.assertEqual(response_data["data"]["profile"], "test-profile")
        
        # Check that profile manager was called
        self.mock_profile_manager.get_profile.assert_called_once_with("test-profile")
    
    def test_start_system_nonexistent_profile(self):
        """Test starting with a profile that doesn't exist"""
        # Set up mock profile manager
        self.mock_profile_manager.get_profile.return_value = None
        
        # Send request
        response = self.client.post(
            "/system/start",
            json={"profile": "nonexistent-profile"}
        )
        
        # Check response - should be an error
        self.assertEqual(response.status_code, 404)
    
    def test_stop_system(self):
        """Test stopping the system"""
        # Send request
        response = self.client.post("/system/stop")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["status"], "success")
        self.assertTrue(response_data["data"]["stopped"])
    
    def test_update_config(self):
        """Test updating system configuration"""
        # Send request
        response = self.client.post(
            "/system/config",
            json={
                "transcription": {"engine": "updated-engine"},
                "vad": {"sensitivity": 0.8},
                "wake_word": {"enabled": True}
            }
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        
        response_data = response.json()
        self.assertEqual(response_data["status"], "success")
        self.assertTrue(response_data["data"]["updated"])

if __name__ == '__main__':
    unittest.main()