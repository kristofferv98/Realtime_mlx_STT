#!/usr/bin/env python3
"""
Test runner script for the Realtime_mlx_STT project.

This script runs tests for the project, with options to:
- Run all tests
- Run tests for a specific feature
- Run a specific test file
"""

import os
import sys
import unittest
import argparse
from pathlib import Path


def get_test_suite(feature=None, test_file=None):
    """
    Get a test suite based on the provided parameters.
    
    Args:
        feature: Optional specific feature to test
        test_file: Optional specific test file to run
        
    Returns:
        unittest.TestSuite: The test suite to run
    """
    if test_file:
        # Run a specific test file
        if not test_file.endswith('.py'):
            test_file += '.py'
        
        # If a full path is provided, use it
        if os.path.exists(test_file):
            test_path = test_file
        else:
            # Try to find it in the features directory
            feature_dir = os.path.join(os.path.dirname(__file__), 'Features')
            
            if feature:
                # Look in the specific feature directory
                test_path = os.path.join(feature_dir, feature, test_file)
                if not os.path.exists(test_path):
                    print(f"Error: Test file {test_file} not found in feature {feature}")
                    return None
            else:
                # Look in all feature directories
                found = False
                for feature_name in os.listdir(feature_dir):
                    feature_path = os.path.join(feature_dir, feature_name)
                    if os.path.isdir(feature_path):
                        test_path = os.path.join(feature_path, test_file)
                        if os.path.exists(test_path):
                            found = True
                            break
                
                if not found:
                    print(f"Error: Test file {test_file} not found in any feature directory")
                    return None
        
        # Load the test file
        return unittest.defaultTestLoader.discover(os.path.dirname(test_path), 
                                                  pattern=os.path.basename(test_path))
    
    elif feature:
        # Run all tests for a specific feature or infrastructure component
        if feature.lower() == 'infrastructure' or feature.lower() == 'infra':
            # Special case for Infrastructure tests
            infra_dir = os.path.join(os.path.dirname(__file__), 'Infrastructure')
            if not os.path.exists(infra_dir):
                print(f"Error: Infrastructure directory not found")
                return None
                
            return unittest.defaultTestLoader.discover(infra_dir)
        else:
            # Normal feature test
            feature_dir = os.path.join(os.path.dirname(__file__), 'Features', feature)
            if not os.path.exists(feature_dir):
                print(f"Error: Feature directory {feature} not found")
                return None
                
            return unittest.defaultTestLoader.discover(feature_dir)
    
    else:
        # Run all tests
        return unittest.defaultTestLoader.discover(os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description='Run tests for Realtime_mlx_STT')
    parser.add_argument('--feature', '-f', help='Specific feature to test (e.g., AudioCapture, VoiceActivityDetection, Infrastructure)')
    parser.add_argument('--test', '-t', help='Specific test file to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Add the project root to the path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    
    # Get and run the test suite
    suite = get_test_suite(args.feature, args.test)
    if suite:
        verbosity = 2 if args.verbose else 1
        result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
        return 0 if result.wasSuccessful() else 1
    else:
        return 1
    

if __name__ == '__main__':
    sys.exit(main())