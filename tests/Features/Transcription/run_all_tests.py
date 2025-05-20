#!/usr/bin/env python3
"""
Run all tests for the Transcription feature.

This script runs all available tests for the Transcription feature.
"""

import os
import sys
import argparse
import subprocess
import time
import signal

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)


def run_simple_test():
    """Run the simple import test."""
    print(f"Running simple import test...")
    
    cmd = [
        sys.executable,
        "-m",
        "tests.Features.Transcription.simple_test"
    ]
    
    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True)
    duration = time.time() - start_time
    
    success = process.returncode == 0
    status = "✓ PASSED" if success else "✗ FAILED"
    
    print(f"  Simple import test: {status} (took {duration:.2f}s)")
    
    if not success:
        print("\nTest output:")
        print(process.stdout.decode('utf-8'))
        print(process.stderr.decode('utf-8'))
    
    print()
    
    return success


def run_direct_engine_test():
    """Run the direct MLX engine test."""
    print(f"Running direct MLX engine test...")
    
    cmd = [
        sys.executable,
        "-m",
        "tests.Features.Transcription.direct_mlx_engine_test"
    ]
    
    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True)
    duration = time.time() - start_time
    
    success = process.returncode == 0
    status = "✓ PASSED" if success else "✗ FAILED"
    
    print(f"  Direct MLX engine test: {status} (took {duration:.2f}s)")
    
    if not success:
        print("\nTest output:")
        print(process.stdout.decode('utf-8'))
        print(process.stderr.decode('utf-8'))
    
    print()
    
    return success


def run_openai_engine_test():
    """Run the OpenAI transcription engine test."""
    print(f"Running OpenAI transcription engine test...")
    
    cmd = [
        sys.executable,
        "-m",
        "tests.Features.Transcription.openai_transcription_engine_test"
    ]
    
    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True)
    duration = time.time() - start_time
    
    success = process.returncode == 0
    status = "✓ PASSED" if success else "✗ FAILED"
    
    print(f"  OpenAI transcription engine test: {status} (took {duration:.2f}s)")
    
    if not success:
        print("\nTest output:")
        print(process.stdout.decode('utf-8'))
        print(process.stderr.decode('utf-8'))
    
    print()
    
    return success


def run_real_transcription_test(run_real_tests=False):
    """Run the real-world transcription test (only if explicitly requested)."""
    if not run_real_tests:
        print(f"Skipping real transcription test (use --run-real-tests to run)")
        print()
        return True
        
    print(f"Running real-world transcription test...")
    
    cmd = [
        sys.executable,
        "-m",
        "tests.Features.Transcription.real_transcription_test"
    ]
    
    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True)
    duration = time.time() - start_time
    
    success = process.returncode == 0
    status = "✓ PASSED" if success else "✗ FAILED"
    
    print(f"  Real-world transcription test: {status} (took {duration:.2f}s)")
    
    # Always show output for real transcription test
    print("\nTest output:")
    print(process.stdout.decode('utf-8'))
    if not success:
        print(process.stderr.decode('utf-8'))
    
    print()
    
    return success


def main():
    """Run all Transcription tests."""
    parser = argparse.ArgumentParser(description="Run all Transcription tests")
    parser.add_argument("--audio-file", type=str, 
                      default="/Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/bok_konge01.mp3",
                      help="Path to the audio file for transcription examples")
    parser.add_argument("--language", type=str, default="no",
                      help="Language code for transcription (default: 'no' for Norwegian)")
    parser.add_argument("--model", type=str, default="whisper-large-v3-turbo",
                      help="Model name (default: whisper-large-v3-turbo)")
    parser.add_argument("--compute-type", type=str, default="float16",
                      choices=["float16", "float32"],
                      help="Computation precision (default: float16)")
    parser.add_argument("--beam-size", type=int, default=1,
                      help="Beam search size for inference (default: 1)")
    parser.add_argument("--quiet", action="store_true",
                      help="Run with minimal output")
    parser.add_argument("--run-real-tests", action="store_true",
                      help="Run real-world tests that may take longer and use actual models")
    
    args = parser.parse_args()
    
    print("Running Transcription feature tests...")
    print("=" * 60)
    print()
    
    all_passed = True
    start_time = time.time()
    
    # Run simple import test
    simple_success = run_simple_test()
    all_passed = all_passed and simple_success
    
    # Run direct engine test
    direct_engine_success = run_direct_engine_test()
    all_passed = all_passed and direct_engine_success
    
    # Run OpenAI engine test
    openai_engine_success = run_openai_engine_test()
    all_passed = all_passed and openai_engine_success
    
    # Run real transcription test (only if requested)
    real_success = run_real_transcription_test(args.run_real_tests)
    all_passed = all_passed and real_success
    
    # Summary
    total_duration = time.time() - start_time
    print("=" * 60)
    final_status = "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
    print(f"Transcription tests complete: {final_status}")
    print(f"Total time: {total_duration:.2f}s")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())