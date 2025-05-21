#!/usr/bin/env python
"""
Test runner for Application tests

This script discovers and runs tests for the Application layer components.
"""

import unittest
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ApplicationTests")

def run_tests(component=None, test_name=None, verbose=False):
    """
    Run tests for Application components.
    
    Args:
        component: Optional specific component to test
        test_name: Optional specific test to run
        verbose: Whether to enable verbose output
    """
    # Set up the test discovery pattern
    pattern = 'test_*.py'
    start_dir = os.path.dirname(os.path.abspath(__file__))
    
    if component:
        start_dir = os.path.join(start_dir, component)
        logger.info(f"Running tests for specific component: {component}")
    
    if test_name:
        pattern = f'{test_name}.py'
        logger.info(f"Running specific test: {test_name}")
    
    # Discover and run tests
    test_suite = unittest.defaultTestLoader.discover(start_dir, pattern=pattern)
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    
    logger.info(f"Starting test execution from {start_dir} with pattern {pattern}")
    result = test_runner.run(test_suite)
    
    # Report results
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    
    return len(result.errors) + len(result.failures)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Application components")
    parser.add_argument("-c", "--component", help="Specific component to test (e.g. Server)")
    parser.add_argument("-t", "--test", help="Specific test to run (e.g. test_server_module)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run tests
    exit_code = run_tests(args.component, args.test, args.verbose)
    
    # Exit with non-zero code if any tests failed
    sys.exit(exit_code)