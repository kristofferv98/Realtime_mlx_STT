#!/usr/bin/env python3
"""
Run tests for the centralized logging system.
"""

import os
import sys
import unittest

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import test modules
from test_logging_system import LoggingSystemTests


def run_tests():
    """Run all tests for the logging system."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(LoggingSystemTests)
    
    # Run tests with verbose output
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Return success/failure
    return len(result.errors) == 0 and len(result.failures) == 0


if __name__ == '__main__':
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)