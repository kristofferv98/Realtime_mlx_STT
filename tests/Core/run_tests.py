#!/usr/bin/env python3
"""
Run all Core infrastructure tests.
"""

import os
import sys
import unittest
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def discover_and_run_tests():
    """Discover and run all tests in the Core directory."""
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Discover tests in subdirectories
    for subdir in ['Events', 'Commands']:
        subdir_path = os.path.join(test_dir, subdir)
        if os.path.exists(subdir_path):
            discovered_tests = loader.discover(
                subdir_path,
                pattern='test_*.py',
                top_level_dir=test_dir
            )
            suite.addTests(discovered_tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = discover_and_run_tests()
    sys.exit(0 if success else 1)