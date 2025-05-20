#!/usr/bin/env python3
"""
Run all wake word detection tests.
"""

import os
import sys
import unittest
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import test modules
try:
    # Will be implemented in the future
    from tests.Features.WakeWordDetection.porcupine_detector_test import PorcupineDetectorTest
    from tests.Features.WakeWordDetection.wake_word_handler_test import WakeWordHandlerTest
except ImportError:
    print("One or more test modules could not be imported. Some tests will be skipped.")


if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests to the suite
    loader = unittest.TestLoader()
    
    # Add all the test cases
    # for test_case in [PorcupineDetectorTest, WakeWordHandlerTest]:
    #     suite.addTests(loader.loadTestsFromTestCase(test_case))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())