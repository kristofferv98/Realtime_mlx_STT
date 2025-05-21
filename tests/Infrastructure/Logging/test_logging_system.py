#!/usr/bin/env python3
"""
Tests for the centralized logging system.

Tests the functionality of the logging configuration system, including:
- Configuration creation and application
- Environment variable integration
- File handlers and rotation
- Runtime configuration via the control server
"""

import os
import sys
import logging
import unittest
import tempfile
import time
import json
import socket
import shutil
import threading
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.Infrastructure.Logging import LoggingModule, LoggingConfig, LoggingConfigurer, LogLevel
from src.Infrastructure.Logging.LoggingControlServer import LoggingControlServer


class LoggingSystemTests(unittest.TestCase):
    """Tests for the centralized logging system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temp directory for log files
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test.log")
        
        # Reset any existing configuration
        LoggingConfigurer._current_config = None
        LoggingConfigurer._active_handlers = {}
        
        # Stop any running control server
        LoggingModule.stop_control_server()
        
        # Record initial loggers to restore after test
        self.root_logger = logging.getLogger()
        self.root_level = self.root_logger.level
        self.root_handlers = list(self.root_logger.handlers)
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)
        
        # Stop any running control server
        LoggingModule.stop_control_server()
        
        # Restore root logger config
        self.root_logger.setLevel(self.root_level)
        self.root_logger.handlers = self.root_handlers
        
    def test_log_level_comparison(self):
        """Test comparison operators for LogLevel enum."""
        # Test direct enum comparisons
        self.assertTrue(LogLevel.DEBUG < LogLevel.INFO)
        self.assertTrue(LogLevel.INFO < LogLevel.WARNING)
        self.assertTrue(LogLevel.WARNING < LogLevel.ERROR)
        self.assertTrue(LogLevel.ERROR < LogLevel.CRITICAL)
        
        # Test greater than
        self.assertTrue(LogLevel.CRITICAL > LogLevel.ERROR)
        
        # Test less than or equal
        self.assertTrue(LogLevel.DEBUG <= LogLevel.DEBUG)
        self.assertTrue(LogLevel.DEBUG <= LogLevel.INFO)
        
        # Test greater than or equal
        self.assertTrue(LogLevel.ERROR >= LogLevel.ERROR)
        self.assertTrue(LogLevel.ERROR >= LogLevel.WARNING)
        
        # Test min/max functions
        self.assertEqual(min(LogLevel.INFO, LogLevel.DEBUG), LogLevel.DEBUG)
        self.assertEqual(max(LogLevel.INFO, LogLevel.DEBUG), LogLevel.INFO)
    
    def test_basic_configuration(self):
        """Test basic logging configuration creation and application."""
        # Configure logging
        config = LoggingModule.initialize(
            console_level="INFO",
            file_enabled=True,
            file_path=self.log_file
        )
        
        # Verify configuration
        self.assertEqual(config.console_level, LogLevel.INFO)
        self.assertTrue(config.file_enabled)
        self.assertEqual(config.file_path, self.log_file)
        
        # Get a test logger
        logger = LoggingModule.get_logger("test_logger")
        
        # Log some messages
        logger.debug("Debug message that should not appear")
        logger.info("Info message that should appear")
        logger.warning("Warning message that should appear")
        
        # Verify log file was created
        self.assertTrue(os.path.exists(self.log_file))
        
        # Read log file content
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        # Verify content
        self.assertNotIn("Debug message", content)
        self.assertIn("Info message", content)
        self.assertIn("Warning message", content)
        
    def test_log_levels(self):
        """Test different log levels."""
        # Configure logging
        config = LoggingModule.initialize(
            console_level="DEBUG",
            file_enabled=True,
            file_path=self.log_file,
            feature_levels={
                "feature1": LogLevel.INFO,
                "feature2": LogLevel.DEBUG
            }
        )
        
        # Get loggers
        logger1 = LoggingModule.get_logger("realtimestt.features.feature1")
        logger2 = LoggingModule.get_logger("realtimestt.features.feature2")
        
        # Log messages
        logger1.debug("Debug message from feature1")
        logger1.info("Info message from feature1")
        logger2.debug("Debug message from feature2")
        logger2.info("Info message from feature2")
        
        # Verify log file was created
        self.assertTrue(os.path.exists(self.log_file))
        
        # Read log file content
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        # Verify content
        self.assertNotIn("Debug message from feature1", content)
        self.assertIn("Info message from feature1", content)
        self.assertIn("Debug message from feature2", content)
        self.assertIn("Info message from feature2", content)
        
    def test_namespace_standardization(self):
        """Test logger namespace standardization."""
        # Configure logging
        LoggingModule.initialize()
        
        # Get loggers with different naming patterns
        logger1 = LoggingModule.get_logger("src.Features.AudioCapture.AudioCaptureModule")
        logger2 = LoggingModule.get_logger("src.Core.Commands.command_dispatcher")
        logger3 = LoggingModule.get_logger("test_logger")
        logger4 = LoggingModule.get_logger("realtimestt.features.customfeature")
        
        # Verify names
        self.assertTrue(logger1.name.startswith("realtimestt.features.audiocapture"))
        self.assertTrue(logger2.name.startswith("realtimestt.core.commands"))
        self.assertTrue(logger3.name.startswith("realtimestt.test_logger"))
        self.assertEqual(logger4.name, "realtimestt.features.customfeature")
        
    def test_log_rotation(self):
        """Test log rotation functionality."""
        # Configure logging with small max_bytes to trigger rotation
        config = LoggingModule.initialize(
            console_level="INFO",
            file_enabled=True,
            file_path=self.log_file,
            rotation_enabled=True,
            max_bytes=100,  # Very small to ensure rotation
            backup_count=3
        )
        
        # Get a test logger
        logger = LoggingModule.get_logger("test_logger")
        
        # Log enough data to trigger rotation multiple times
        for i in range(10):
            logger.info(f"Test message {i} with enough data to exceed 100 bytes easily when formatted with timestamp and logger name")
            
        # Verify backup files were created
        self.assertTrue(os.path.exists(self.log_file))
        self.assertTrue(os.path.exists(f"{self.log_file}.1"))
        self.assertTrue(os.path.exists(f"{self.log_file}.2"))
        self.assertTrue(os.path.exists(f"{self.log_file}.3"))
        
        # Verify we don't have more backups than specified
        self.assertFalse(os.path.exists(f"{self.log_file}.4"))
        
    def test_environment_variables(self):
        """Test environment variable configuration."""
        # Set environment variables
        os.environ['REALTIMESTT_LOG_LEVEL'] = 'DEBUG'
        os.environ['REALTIMESTT_LOG_FILE'] = 'true'
        os.environ['REALTIMESTT_LOG_PATH'] = self.log_file
        os.environ['REALTIMESTT_LOG_ROTATION'] = 'true'
        os.environ['REALTIMESTT_LOG_FEATURE_TESTFEATURE'] = 'INFO'
        
        try:
            # Apply environment configuration
            config = LoggingModule.initialize_from_env()
            
            # Verify configuration
            self.assertEqual(config.root_level, LogLevel.DEBUG)
            self.assertTrue(config.file_enabled)
            self.assertEqual(config.file_path, self.log_file)
            self.assertTrue(config.rotation_enabled)
            self.assertEqual(config.feature_levels.get('testfeature'), LogLevel.INFO)
            
            # Test logging
            logger = LoggingModule.get_logger("realtimestt.features.testfeature")
            logger.debug("Debug message")
            logger.info("Info message")
            
            # Verify log file
            with open(self.log_file, 'r') as f:
                content = f.read()
                
            self.assertNotIn("Debug message", content)
            self.assertIn("Info message", content)
            
        finally:
            # Clean up environment
            del os.environ['REALTIMESTT_LOG_LEVEL']
            del os.environ['REALTIMESTT_LOG_FILE']
            del os.environ['REALTIMESTT_LOG_PATH']
            del os.environ['REALTIMESTT_LOG_ROTATION']
            del os.environ['REALTIMESTT_LOG_FEATURE_TESTFEATURE']
            
    def test_runtime_configuration(self):
        """Test runtime configuration via the control server."""
        # Start the logging system with control server
        LoggingModule.initialize(
            console_level="INFO",
            file_enabled=True,
            file_path=self.log_file,
            start_control_server=True,
            control_server_port=50555  # Use a custom port to avoid conflicts
        )
        
        # Get a test logger
        logger = LoggingModule.get_logger("test_runtime_config")
        
        # Log initial messages
        logger.debug("Initial debug message")
        logger.info("Initial info message")
        
        # Verify the debug message doesn't appear
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertNotIn("Initial debug message", content)
            self.assertIn("Initial info message", content)
        
        # Change log level via UDP packet
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command = {
            'action': 'set_level',
            'target': 'test_runtime_config',
            'level': 'DEBUG',
            'timestamp': time.time()
        }
        message = json.dumps(command).encode('utf-8')
        sock.sendto(message, ('127.0.0.1', 50555))
        
        # Give the server time to process
        time.sleep(0.5)
        
        # Log more messages
        logger.debug("After-change debug message")
        logger.info("After-change info message")
        
        # Verify both messages appear now
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn("After-change debug message", content)
            self.assertIn("After-change info message", content)
            
        # Verify control server can be stopped
        self.assertTrue(LoggingModule.stop_control_server())
        self.assertFalse(LoggingModule.is_control_server_running())
        
    def test_predefined_configs(self):
        """Test predefined development and production configurations."""
        # Test development config
        dev_config = LoggingModule.create_development_config(start_control_server=False)
        self.assertEqual(dev_config.root_level, LogLevel.DEBUG)
        self.assertEqual(dev_config.console_level, LogLevel.DEBUG)
        self.assertTrue(dev_config.file_enabled)
        
        # Clean up after dev config
        LoggingConfigurer._current_config = None
        LoggingConfigurer._active_handlers = {}
        
        # Test production config
        prod_config = LoggingModule.create_production_config(start_control_server=False)
        self.assertEqual(prod_config.root_level, LogLevel.INFO)
        self.assertEqual(prod_config.console_level, LogLevel.INFO)
        self.assertTrue(prod_config.file_enabled)
        self.assertTrue(prod_config.rotation_enabled)
        

if __name__ == '__main__':
    unittest.main()