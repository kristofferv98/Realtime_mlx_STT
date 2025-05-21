"""
Tests for ServerConfig class
"""

import unittest
import os
import tempfile
import json
from unittest.mock import patch

from src.Application.Server.Configuration.ServerConfig import ServerConfig

class TestServerConfig(unittest.TestCase):
    """Test cases for ServerConfig class"""

    def setUp(self):
        """Set up test environment before each test case"""
        # Clear environment variables that might affect the tests
        for var in ['STT_SERVER_HOST', 'STT_SERVER_PORT', 'STT_SERVER_DEBUG', 
                   'STT_SERVER_AUTO_START', 'STT_SERVER_CORS_ORIGINS']:
            if var in os.environ:
                del os.environ[var]
    
    def test_default_config(self):
        """Test that default configuration is correctly initialized"""
        config = ServerConfig()
        
        # Check default values
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8080)
        self.assertFalse(config.debug)
        self.assertTrue(config.auto_start)
        self.assertEqual(config.cors_origins, ["*"])
        self.assertFalse(config.auth_enabled)
        self.assertIsNone(config.auth_token)
        self.assertEqual(config.profiles_directory, "profiles/")
        self.assertEqual(config.default_profile, "default")
    
    def test_config_from_env(self):
        """Test loading configuration from environment variables"""
        # Set environment variables
        os.environ['STT_SERVER_HOST'] = "0.0.0.0"
        os.environ['STT_SERVER_PORT'] = "9000"
        os.environ['STT_SERVER_DEBUG'] = "true"
        os.environ['STT_SERVER_AUTO_START'] = "false"
        os.environ['STT_SERVER_CORS_ORIGINS'] = "http://localhost:3000,http://app.example.com"
        os.environ['STT_SERVER_AUTH_ENABLED'] = "true"
        os.environ['STT_SERVER_AUTH_TOKEN'] = "secret-token"
        os.environ['STT_SERVER_PROFILES_DIR'] = "custom-profiles/"
        os.environ['STT_SERVER_DEFAULT_PROFILE'] = "custom-default"
        
        config = ServerConfig.from_env()
        
        # Check values from environment variables
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 9000)
        self.assertTrue(config.debug)
        self.assertFalse(config.auto_start)
        self.assertEqual(config.cors_origins, ["http://localhost:3000", "http://app.example.com"])
        self.assertTrue(config.auth_enabled)
        self.assertEqual(config.auth_token, "secret-token")
        self.assertEqual(config.profiles_directory, "custom-profiles/")
        self.assertEqual(config.default_profile, "custom-default")
    
    def test_invalid_port_in_env(self):
        """Test handling of invalid port value in environment"""
        os.environ['STT_SERVER_PORT'] = "invalid-port"
        
        config = ServerConfig.from_env()
        
        # Should fall back to default port
        self.assertEqual(config.port, 8080)
    
    def test_config_from_file(self):
        """Test loading configuration from a JSON file"""
        # Create a temporary config file
        config_data = {
            "server": {
                "host": "1.2.3.4",
                "port": 5000,
                "debug": True,
                "auto_start": False,
                "cors_origins": ["https://example.com"],
                "auth_enabled": True,
                "auth_token": "file-token",
                "profiles_directory": "file-profiles/",
                "default_profile": "file-default"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(config_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            config = ServerConfig.from_file(temp_file_path)
            
            # Check values from file
            self.assertEqual(config.host, "1.2.3.4")
            self.assertEqual(config.port, 5000)
            self.assertTrue(config.debug)
            self.assertFalse(config.auto_start)
            self.assertEqual(config.cors_origins, ["https://example.com"])
            self.assertTrue(config.auth_enabled)
            self.assertEqual(config.auth_token, "file-token")
            self.assertEqual(config.profiles_directory, "file-profiles/")
            self.assertEqual(config.default_profile, "file-default")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_file_not_found(self):
        """Test handling of missing configuration file"""
        non_existent_file = "/tmp/non_existent_config_file.json"
        
        # Make sure the file doesn't exist
        if os.path.exists(non_existent_file):
            os.unlink(non_existent_file)
        
        # Should fall back to environment-based config
        with patch.object(ServerConfig, 'from_env') as mock_from_env:
            mock_from_env.return_value = "mock_config"
            result = ServerConfig.from_file(non_existent_file)
            
            mock_from_env.assert_called_once()
            self.assertEqual(result, "mock_config")
    
    def test_to_dict(self):
        """Test converting config to dictionary"""
        config = ServerConfig()
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict["host"], "127.0.0.1")
        self.assertEqual(config_dict["port"], 8080)
        self.assertFalse(config_dict["debug"])
        self.assertTrue(config_dict["auto_start"])
        self.assertEqual(config_dict["cors_origins"], ["*"])
    
    def test_save_to_file(self):
        """Test saving configuration to a file"""
        config = ServerConfig()
        config.host = "test-host"
        config.port = 1234
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Delete the file so we can test directory creation
            os.unlink(temp_file_path)
            
            success = config.save_to_file(temp_file_path)
            
            # Check save was successful
            self.assertTrue(success)
            self.assertTrue(os.path.exists(temp_file_path))
            
            # Read the file back and check contents
            with open(temp_file_path, 'r') as f:
                saved_data = json.load(f)
                
            self.assertEqual(saved_data["server"]["host"], "test-host")
            self.assertEqual(saved_data["server"]["port"], 1234)
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

if __name__ == '__main__':
    unittest.main()