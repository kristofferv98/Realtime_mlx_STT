#!/usr/bin/env python3
"""
STTClient Example - Modern API Usage

This shows how to use the client-based API which manages
sessions internally for a cleaner interface.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_mlx_stt import STTClient, create_client


def example_1_basic():
    """Basic client usage."""
    print("=== Example 1: Basic Client Usage ===")
    
    # Create client
    client = STTClient()
    
    # Transcribe for 10 seconds
    print("Listening for 10 seconds...\n")
    
    for result in client.transcribe(duration=10):
        print(f"📝 {result.text}")
        print(f"   Confidence: {result.confidence:.2f}\n")
    
    print("Done!\n")


def example_2_streaming():
    """Streaming with context manager."""
    print("=== Example 2: Streaming Context Manager ===")
    print("Say 'stop' to end the stream.\n")
    
    client = STTClient()
    
    with client.stream() as stream:
        for result in stream:
            print(f">>> {result.text}")
            
            # Stop on keyword
            if "stop" in result.text.lower():
                print("\nStop word detected!")
                break
    
    print("Stream closed.\n")


def example_3_openai():
    """Using OpenAI engine."""
    print("=== Example 3: OpenAI Engine ===")
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping example.")
        print("   Set it with: export OPENAI_API_KEY='your-key'\n")
        return
    
    # Create client with API key
    client = create_client(
        openai_api_key=api_key,
        default_engine="openai"
    )
    
    print("Using OpenAI for transcription (5 seconds)...\n")
    
    for result in client.transcribe(duration=5):
        print(f"🌐 {result.text}\n")
    
    print("OpenAI example done.\n")


def example_4_wake_word():
    """Wake word detection."""
    print("=== Example 4: Wake Word Mode ===")
    
    # Check for API key
    if not os.environ.get('PORCUPINE_ACCESS_KEY'):
        print("⚠️  PORCUPINE_ACCESS_KEY not set. Skipping example.")
        print("   Get your free key at: https://picovoice.ai/\n")
        return
    
    print("Say 'jarvis' to activate...\n")
    
    # Create client
    client = STTClient()
    
    # Define callbacks
    def on_wake(word, confidence):
        print(f"✨ Wake word '{word}' detected! (confidence: {confidence:.2f})")
        print("   Listening for command...\n")
    
    def on_transcription(result):
        print(f"   Command: {result.text}\n")
        print("Say 'jarvis' again...")
    
    # Start wake word mode
    client.start_wake_word(
        wake_word="jarvis",
        on_wake=on_wake,
        on_transcription=on_transcription
    )
    
    # Run for 20 seconds
    import time
    time.sleep(20)
    
    client.stop()
    print("Wake word mode stopped.\n")


def example_5_convenience():
    """Using convenience features."""
    print("=== Example 5: Convenience Features ===")
    
    # Create client with settings
    client = STTClient(
        default_language="en",
        verbose=False
    )
    
    # List devices
    print("Audio devices:")
    for device in client.list_devices():
        print(f"  [{device.index}] {device.name}")
    
    # Set language
    client.set_language("no")  # Norwegian
    
    print("\nTranscribing in Norwegian (5 seconds)...\n")
    
    for result in client.transcribe(duration=5):
        print(f"🇳🇴 {result.text}\n")


def main():
    """Run all examples."""
    print("STTClient Examples")
    print("=" * 50)
    print("\nThis demonstrates the modern client-based API.\n")
    
    # Basic usage
    example_1_basic()
    
    # Streaming
    print("Press Enter to continue to streaming example...")
    input()
    example_2_streaming()
    
    # OpenAI
    example_3_openai()
    
    # Wake word
    example_4_wake_word()
    
    # Convenience features
    example_5_convenience()
    
    print("All examples completed!")


if __name__ == "__main__":
    # Run specific example or all
    import argparse
    
    parser = argparse.ArgumentParser(description="STTClient examples")
    parser.add_argument('--example', type=int, choices=[1,2,3,4,5],
                        help='Run specific example (1-5)')
    
    args = parser.parse_args()
    
    if args.example:
        examples = {
            1: example_1_basic,
            2: example_2_streaming,
            3: example_3_openai,
            4: example_4_wake_word,
            5: example_5_convenience
        }
        examples[args.example]()
    else:
        main()