#!/usr/bin/env python3
"""Debug OpenAI transcription."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_mlx_stt import STTClient

def main():
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not set!")
        return
    
    print("Creating client with verbose mode...")
    client = STTClient(
        openai_api_key=api_key,
        default_engine="openai",
        verbose=True  # Enable verbose logging
    )
    
    print("\nTranscribing for 10 seconds (say something!)...\n")
    
    results = []
    for result in client.transcribe(duration=10, engine="openai"):
        print(f"Result: {result.text} (confidence: {result.confidence})")
        results.append(result)
    
    if not results:
        print("\nNo transcription results received!")
    else:
        print(f"\nReceived {len(results)} transcription(s)")
    
    print("\nTesting completed.")

if __name__ == "__main__":
    main()