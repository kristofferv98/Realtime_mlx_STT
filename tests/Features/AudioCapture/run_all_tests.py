"""
Run all tests for the AudioCapture feature.

This script discovers and runs all tests for the AudioCapture feature.
"""

import unittest
import sys
import logging
import os

# Set up logging to show info messages during tests
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   stream=sys.stdout)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def run_tests():
    """Discover and run all tests in the AudioCapture feature."""
    print("\n=== Running AudioCapture Feature Tests ===\n")
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover tests in current directory
    suite = loader.discover(start_dir=current_dir, pattern='*_test.py')
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())