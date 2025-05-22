#!/usr/bin/env python3
"""
Unit tests for ProgressBarManager.

This module tests the progress bar manager implementation,
including enabling/disabling progress bars and environment variable handling.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


class TestProgressBarManager(unittest.TestCase):
    """Test cases for ProgressBarManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Store original environment variables
        self.original_env = {}
        self.env_vars = ['TQDM_DISABLE', 'HF_HUB_DISABLE_PROGRESS_BARS', 'DISABLE_PROGRESS_BARS']
        for var in self.env_vars:
            self.original_env[var] = os.environ.get(var)
            
        # Clear environment variables BEFORE importing
        for var in self.env_vars:
            if var in os.environ:
                del os.environ[var]
                
        # Mock tqdm BEFORE importing
        self.tqdm_mock = MagicMock()
        self.tqdm_mock.tqdm = MagicMock()
        self.tqdm_mock.tqdm.disable = False
        self.tqdm_mock.auto = MagicMock()
        self.tqdm_mock.auto.tqdm = MagicMock()
        self.tqdm_mock.auto.tqdm.disable = False
        
        # Patch tqdm module before importing ProgressBarManager
        self.tqdm_patcher = patch.dict('sys.modules', {'tqdm': self.tqdm_mock})
        self.tqdm_patcher.start()
        
        # Remove ProgressBarManager from sys.modules if it exists
        if 'src.Infrastructure.ProgressBar.ProgressBarManager' in sys.modules:
            del sys.modules['src.Infrastructure.ProgressBar.ProgressBarManager']
        
        # Import after patching and clearing env
        from src.Infrastructure.ProgressBar.ProgressBarManager import ProgressBarManager
        self.ProgressBarManager = ProgressBarManager
        
        # Reset class state AND clear any env vars set during import
        self.ProgressBarManager._initialized = False
        self.ProgressBarManager._disabled = False
        
        # Clear environment variables again after import
        for var in self.env_vars:
            if var in os.environ:
                del os.environ[var]
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop patcher
        self.tqdm_patcher.stop()
        
        # Restore original environment variables
        for var, value in self.original_env.items():
            if value is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = value
                
    def test_initialization_default(self):
        """Test default initialization without environment variables."""
        result = self.ProgressBarManager.initialize()
        
        self.assertFalse(result)  # Should be enabled by default
        self.assertTrue(self.ProgressBarManager._initialized)
        self.assertFalse(self.ProgressBarManager._disabled)
        self.assertFalse(self.tqdm_mock.tqdm.disable)
        
    def test_initialization_with_disabled_parameter(self):
        """Test initialization with explicit disabled parameter."""
        result = self.ProgressBarManager.initialize(disabled=True)
        
        self.assertTrue(result)  # Should be disabled
        self.assertTrue(self.ProgressBarManager._disabled)
        self.assertTrue(self.tqdm_mock.tqdm.disable)
        self.assertEqual(os.environ.get('TQDM_DISABLE'), '1')
        self.assertEqual(os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS'), '1')
        
    def test_initialization_with_enabled_parameter(self):
        """Test initialization with explicit enabled parameter."""
        result = self.ProgressBarManager.initialize(disabled=False)
        
        self.assertFalse(result)  # Should be enabled
        self.assertFalse(self.ProgressBarManager._disabled)
        self.assertFalse(self.tqdm_mock.tqdm.disable)
        self.assertNotIn('TQDM_DISABLE', os.environ)
        self.assertNotIn('HF_HUB_DISABLE_PROGRESS_BARS', os.environ)
        
    def test_initialization_with_tqdm_disable_env(self):
        """Test initialization with TQDM_DISABLE environment variable."""
        os.environ['TQDM_DISABLE'] = '1'
        
        result = self.ProgressBarManager.initialize()
        
        self.assertTrue(result)  # Should be disabled
        self.assertTrue(self.ProgressBarManager._disabled)
        self.assertTrue(self.tqdm_mock.tqdm.disable)
        
    def test_initialization_with_hf_hub_disable_env(self):
        """Test initialization with HF_HUB_DISABLE_PROGRESS_BARS environment variable."""
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'true'
        
        result = self.ProgressBarManager.initialize()
        
        self.assertTrue(result)  # Should be disabled
        self.assertTrue(self.ProgressBarManager._disabled)
        self.assertTrue(self.tqdm_mock.tqdm.disable)
        
    def test_initialization_with_disable_progress_bars_env(self):
        """Test initialization with DISABLE_PROGRESS_BARS environment variable."""
        os.environ['DISABLE_PROGRESS_BARS'] = 'yes'
        
        result = self.ProgressBarManager.initialize()
        
        self.assertTrue(result)  # Should be disabled
        self.assertTrue(self.ProgressBarManager._disabled)
        self.assertTrue(self.tqdm_mock.tqdm.disable)
        
    def test_explicit_parameter_overrides_env(self):
        """Test that explicit parameter overrides environment variables."""
        os.environ['TQDM_DISABLE'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        
        result = self.ProgressBarManager.initialize(disabled=False)
        
        self.assertFalse(result)  # Should be enabled despite env vars
        self.assertFalse(self.ProgressBarManager._disabled)
        self.assertFalse(self.tqdm_mock.tqdm.disable)
        # Environment variables should be cleared
        self.assertNotIn('TQDM_DISABLE', os.environ)
        self.assertNotIn('HF_HUB_DISABLE_PROGRESS_BARS', os.environ)
        
    def test_disable_method(self):
        """Test disable method."""
        # Initialize first as enabled
        self.ProgressBarManager.initialize(disabled=False)
        self.assertFalse(self.ProgressBarManager._disabled)
        
        # Disable
        result = self.ProgressBarManager.disable()
        
        self.assertTrue(result)
        self.assertTrue(self.ProgressBarManager._disabled)
        self.assertTrue(self.tqdm_mock.tqdm.disable)
        self.assertEqual(os.environ.get('TQDM_DISABLE'), '1')
        self.assertEqual(os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS'), '1')
        
    def test_enable_method(self):
        """Test enable method."""
        # Initialize first as disabled
        self.ProgressBarManager.initialize(disabled=True)
        self.assertTrue(self.ProgressBarManager._disabled)
        
        # Enable
        result = self.ProgressBarManager.enable()
        
        self.assertFalse(result)
        self.assertFalse(self.ProgressBarManager._disabled)
        self.assertFalse(self.tqdm_mock.tqdm.disable)
        self.assertNotIn('TQDM_DISABLE', os.environ)
        self.assertNotIn('HF_HUB_DISABLE_PROGRESS_BARS', os.environ)
        
    def test_is_disabled(self):
        """Test is_disabled method."""
        # Test when not initialized
        self.assertFalse(self.ProgressBarManager.is_disabled())  # Should initialize with default
        
        # Test when disabled
        self.ProgressBarManager.disable()
        self.assertTrue(self.ProgressBarManager.is_disabled())
        
        # Test when enabled
        self.ProgressBarManager.enable()
        self.assertFalse(self.ProgressBarManager.is_disabled())
        
    def test_auto_initialization_on_disable(self):
        """Test that disable auto-initializes if not initialized."""
        self.assertFalse(self.ProgressBarManager._initialized)
        
        self.ProgressBarManager.disable()
        
        self.assertTrue(self.ProgressBarManager._initialized)
        self.assertTrue(self.ProgressBarManager._disabled)
        
    def test_auto_initialization_on_enable(self):
        """Test that enable auto-initializes if not initialized."""
        self.assertFalse(self.ProgressBarManager._initialized)
        
        self.ProgressBarManager.enable()
        
        self.assertTrue(self.ProgressBarManager._initialized)
        self.assertFalse(self.ProgressBarManager._disabled)
        
    def test_tqdm_configuration_error_handling(self):
        """Test that tqdm configuration errors are handled gracefully."""
        # Make tqdm.tqdm.disable raise an exception when set
        type(self.tqdm_mock.tqdm).disable = property(
            lambda self: False,
            lambda self, value: (_ for _ in ()).throw(RuntimeError("Cannot set disable"))
        )
        
        # Should not raise, just log warning
        result = self.ProgressBarManager.initialize(disabled=True)
        
        # Should still set environment variables as fallback
        self.assertTrue(result)
        self.assertEqual(os.environ.get('TQDM_DISABLE'), '1')
        self.assertEqual(os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS'), '1')
        
    def test_environment_variable_values(self):
        """Test various environment variable values."""
        # Note: The implementation checks for presence of TQDM_DISABLE or HF_HUB_DISABLE_PROGRESS_BARS
        # regardless of value, so any value (including '0') is treated as True
        test_cases = [
            ('1', True),
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('yes', True),
            ('Yes', True),
            ('YES', True),
            ('0', True),      # Presence of variable means disabled
            ('false', True),  # Presence of variable means disabled
            ('no', True),     # Presence of variable means disabled
            ('', True),       # Presence of variable means disabled
        ]
        
        for value, expected in test_cases:
            # Clear env
            for var in self.env_vars:
                if var in os.environ:
                    del os.environ[var]
                    
            # Reset state
            self.ProgressBarManager._initialized = False
            self.ProgressBarManager._disabled = False
            
            # Set env and test
            os.environ['TQDM_DISABLE'] = value
            result = self.ProgressBarManager.initialize()
            
            self.assertEqual(result, expected, 
                           f"Failed for TQDM_DISABLE='{value}': expected {expected}, got {result}")
                           
    def test_disable_progress_bars_values(self):
        """Test DISABLE_PROGRESS_BARS environment variable values."""
        # DISABLE_PROGRESS_BARS does check the actual value
        test_cases = [
            ('1', True),
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('yes', True),
            ('Yes', True),
            ('YES', True),
            ('0', False),
            ('false', False),
            ('no', False),
            ('', False),
        ]
        
        for value, expected in test_cases:
            # Clear env
            for var in self.env_vars:
                if var in os.environ:
                    del os.environ[var]
                    
            # Reset state
            self.ProgressBarManager._initialized = False
            self.ProgressBarManager._disabled = False
            
            # Set env and test
            os.environ['DISABLE_PROGRESS_BARS'] = value
            result = self.ProgressBarManager.initialize()
            
            self.assertEqual(result, expected, 
                           f"Failed for DISABLE_PROGRESS_BARS='{value}': expected {expected}, got {result}")
            
    def test_multiple_env_vars_precedence(self):
        """Test when multiple environment variables are set."""
        # All set to disable
        os.environ['DISABLE_PROGRESS_BARS'] = '1'
        os.environ['TQDM_DISABLE'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        
        result = self.ProgressBarManager.initialize()
        self.assertTrue(result)
        
        # Mixed values - any true value should disable
        os.environ['DISABLE_PROGRESS_BARS'] = '0'
        os.environ['TQDM_DISABLE'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
        
        self.ProgressBarManager._initialized = False
        result = self.ProgressBarManager.initialize()
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()