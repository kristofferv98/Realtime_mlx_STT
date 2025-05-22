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
    from tests.Features.WakeWordDetection.porcupine_detector_test import TestPorcupineWakeWordDetector
    from tests.Features.WakeWordDetection.wake_word_handler_test import TestWakeWordCommandHandler
except ImportError as e:
    print(f"One or more test modules could not be imported: {e}")
    print("Some tests will be skipped.")


if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests to the suite
    loader = unittest.TestLoader()
    
    # Add all the test cases if they were imported successfully
    test_cases = []
    
    try:
        test_cases.append(TestPorcupineWakeWordDetector)
    except NameError:
        print("TestPorcupineWakeWordDetector could not be imported")
        
    try:
        test_cases.append(TestWakeWordCommandHandler)
    except NameError:
        print("TestWakeWordCommandHandler could not be imported")
    
    for test_case in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())