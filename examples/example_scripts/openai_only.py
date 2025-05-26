#!/usr/bin/env python3
"""
OpenAI-only transcription example.

This example demonstrates using only the OpenAI engine for transcription,
without local MLX processing.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_mlx_stt import STTClient, create_client


def main():
    print("OpenAI Transcription Example")
    print("=" * 50)
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        return
    
    print("\n✅ OpenAI API key found")
    print("Creating OpenAI-only client...\n")
    
    # Create client configured for OpenAI
    client = create_client(
        openai_api_key=api_key,
        default_engine="openai",  # Use OpenAI by default
        default_model="whisper-1",  # OpenAI's whisper model
        verbose=False  # Set to True for debugging
    )
    
    print("Ready to transcribe using OpenAI!")
    print("-" * 50)
    
    # Example 1: Fixed duration transcription
    print("\n📝 Transcribing for 10 seconds...")
    print("(Speak clearly into your microphone)\n")
    
    transcription_count = 0
    for result in client.transcribe(duration=10, engine="openai"):
        transcription_count += 1
        print(f"🎙️  {result.text}")
        print(f"   (Confidence: {result.confidence:.2f})\n")
    
    if transcription_count == 0:
        print("No speech detected in the last 10 seconds.\n")
    else:
        print(f"Received {transcription_count} transcription(s).\n")
    
    # Example 2: Streaming with stop word
    print("-" * 50)
    print("\n🔄 Streaming mode (say 'goodbye' to stop)...")
    print("(Speak continuously, transcriptions will appear as you talk)\n")
    
    with client.stream(engine="openai") as stream:
        for result in stream:
            print(f">>> {result.text}")
            
            # Stop on keyword
            if "goodbye" in result.text.lower():
                print("\n👋 Goodbye detected, stopping stream.")
                break
    
    print("\n✅ OpenAI transcription example completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()