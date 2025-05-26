"""
Tests for ProfileManager class
"""

import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch

from src.Application.Server.Configuration.ProfileManager import ProfileManager

class TestProfileManager(unittest.TestCase):
    """Test cases for ProfileManager class"""

    def setUp(self):
        """Set up test environment before each test case"""
        # Create a temporary directory for profiles
        self.temp_dir = tempfile.mkdtemp()
        self.profile_manager = ProfileManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up after each test case"""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_predefined_profiles(self):
        """Test that predefined profiles are available"""
        # Check that predefined profiles exist
        self.assertIn("vad-triggered", ProfileManager.PREDEFINED_PROFILES)
        self.assertIn("wake-word", ProfileManager.PREDEFINED_PROFILES)
        
        # Check predefined profile structure (spot check a few fields)
        vad_profile = ProfileManager.PREDEFINED_PROFILES["vad-triggered"]
        self.assertTrue(vad_profile["vad"]["enabled"])
        self.assertFalse(vad_profile["wake_word"]["enabled"])
        
        wake_word_profile = ProfileManager.PREDEFINED_PROFILES["wake-word"]
        self.assertTrue(wake_word_profile["wake_word"]["enabled"])
        self.assertIn("jarvis", wake_word_profile["wake_word"]["words"])
    
    def test_get_predefined_profile(self):
        """Test retrieving a predefined profile"""
        profile = self.profile_manager.get_profile("wake-word-mlx")
        
        self.assertIsNotNone(profile)
        self.assertEqual(profile["transcription"]["engine"], "mlx_whisper")
        self.assertTrue(profile["wake_word"]["enabled"])
    
    def test_get_nonexistent_profile(self):
        """Test retrieving a non-existent profile"""
        profile = self.profile_manager.get_profile("non-existent-profile")
        
        self.assertIsNone(profile)
    
    def test_save_and_get_profile(self):
        """Test saving and retrieving a custom profile"""
        profile_data = {
            "transcription": {
                "engine": "custom-engine",
                "model": "custom-model"
            },
            "wake_word": {
                "enabled": True,
                "words": ["computer"]
            }
        }
        
        # Save the profile
        success = self.profile_manager.save_profile("custom-profile", profile_data)
        self.assertTrue(success)
        
        # Verify the file was created
        profile_path = os.path.join(self.temp_dir, "custom-profile.json")
        self.assertTrue(os.path.exists(profile_path))
        
        # Get the profile back
        retrieved_profile = self.profile_manager.get_profile("custom-profile")
        self.assertEqual(retrieved_profile, profile_data)
    
    def test_save_predefined_profile(self):
        """Test that saving over a predefined profile fails"""
        profile_data = {"test": "data"}
        
        # Try to save over a predefined profile
        success = self.profile_manager.save_profile("vad-triggered", profile_data)
        self.assertFalse(success)
        
        # The predefined profile should remain unchanged
        profile = self.profile_manager.get_profile("vad-triggered")
        self.assertNotEqual(profile, profile_data)
    
    def test_list_profiles(self):
        """Test listing all available profiles"""
        # Create some custom profiles
        self.profile_manager.save_profile("custom1", {"test": "data1"})
        self.profile_manager.save_profile("custom2", {"test": "data2"})
        
        profiles = self.profile_manager.list_profiles()
        
        # Should include both predefined and custom profiles
        self.assertIn("vad-triggered", profiles)
        self.assertIn("wake-word-mlx", profiles)
        self.assertIn("custom1", profiles)
        self.assertIn("custom2", profiles)
    
    def test_delete_profile(self):
        """Test deleting a custom profile"""
        # Create a custom profile
        self.profile_manager.save_profile("custom-profile", {"test": "data"})
        
        # Verify it exists
        self.assertIsNotNone(self.profile_manager.get_profile("custom-profile"))
        
        # Delete it
        success = self.profile_manager.delete_profile("custom-profile")
        self.assertTrue(success)
        
        # Verify it's gone
        self.assertIsNone(self.profile_manager.get_profile("custom-profile"))
    
    def test_delete_predefined_profile(self):
        """Test that deleting a predefined profile fails"""
        # Try to delete a predefined profile
        success = self.profile_manager.delete_profile("vad-triggered")
        self.assertFalse(success)
        
        # The predefined profile should still be available
        self.assertIsNotNone(self.profile_manager.get_profile("vad-triggered"))
    
    def test_delete_nonexistent_profile(self):
        """Test deleting a non-existent profile"""
        success = self.profile_manager.delete_profile("non-existent-profile")
        self.assertFalse(success)

if __name__ == '__main__':
    unittest.main()