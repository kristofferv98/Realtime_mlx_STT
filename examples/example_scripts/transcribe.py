#!/usr/bin/env python3
"""
Realtime MLX STT - Main Example

This example shows the recommended way to use the library with
the session-based API that provides proper state management.
"""

import os
import sys
import time
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_mlx_stt import TranscriptionSession, ModelConfig, VADConfig, WakeWordConfig


def main():
    parser = argparse.ArgumentParser(
        description="Realtime speech transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (VAD-triggered transcription)
  python transcribe.py
  
  # Use OpenAI instead of local MLX
  python transcribe.py --engine openai --model gpt-4o-transcribe
  
  # Force Norwegian language
  python transcribe.py --language no
  
  # Wake word mode
  python transcribe.py --wake-word jarvis
  
  # High sensitivity VAD
  python transcribe.py --vad-sensitivity 0.8
  
  # List audio devices
  python transcribe.py --list-devices
"""
    )
    
    # Model options
    parser.add_argument('--engine', choices=['mlx_whisper', 'openai'], 
                        default='mlx_whisper', help='Transcription engine')
    parser.add_argument('--model', default='whisper-large-v3-turbo',
                        help='Model name (default: whisper-large-v3-turbo)')
    parser.add_argument('--language', help='Language code (e.g., en, no, es)')
    
    # VAD options
    parser.add_argument('--vad-sensitivity', type=float, default=0.6,
                        help='VAD sensitivity 0.0-1.0 (default: 0.6)')
    parser.add_argument('--vad-min-speech', type=float, default=0.25,
                        help='Minimum speech duration in seconds (default: 0.25)')
    
    # Wake word options
    parser.add_argument('--wake-word', help='Enable wake word mode with specified word')
    parser.add_argument('--wake-sensitivity', type=float, default=0.7,
                        help='Wake word sensitivity 0.0-1.0 (default: 0.7)')
    
    # Other options
    parser.add_argument('--device', type=int, help='Audio device index')
    parser.add_argument('--list-devices', action='store_true', 
                        help='List available audio devices and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        from realtime_mlx_stt import list_audio_devices
        print("Available audio devices:")
        for device in list_audio_devices():
            default = " (DEFAULT)" if device.is_default else ""
            print(f"  [{device.index}] {device.name}{default}")
        return
    
    # Check OpenAI key if needed
    if args.engine == 'openai' and not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable required for OpenAI engine")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Check Porcupine key if using wake word
    if args.wake_word and not os.environ.get('PORCUPINE_ACCESS_KEY'):
        print("Error: PORCUPINE_ACCESS_KEY environment variable required for wake word")
        print("Get your free key at: https://picovoice.ai/")
        return
    
    print("Realtime MLX STT")
    print("=" * 50)
    
    # Configure model
    model_config = ModelConfig(
        engine=args.engine,
        model=args.model,
        language=args.language
    )
    
    # Configure VAD
    vad_config = VADConfig(
        sensitivity=args.vad_sensitivity,
        min_speech_duration=args.vad_min_speech
    )
    
    # Configure wake word if specified
    wake_word_config = None
    if args.wake_word:
        wake_word_config = WakeWordConfig(
            words=[args.wake_word.lower()],
            sensitivity=args.wake_sensitivity
        )
    
    # Print configuration
    print(f"Engine: {args.engine}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language or 'auto-detect'}")
    if args.wake_word:
        print(f"Wake word: {args.wake_word}")
    print(f"VAD sensitivity: {args.vad_sensitivity}")
    print()
    
    # State tracking
    transcription_count = 0
    wake_word_active = False
    
    # Define callbacks
    def on_transcription(result):
        nonlocal transcription_count, wake_word_active
        transcription_count += 1
        
        if args.wake_word and not wake_word_active:
            # In wake word mode but wake word not detected
            return
            
        print(f"[{transcription_count}] {result.text}")
        if args.verbose:
            print(f"    Confidence: {result.confidence:.2f}")
        
        # Reset wake word state
        if args.wake_word:
            wake_word_active = False
            print(f"\n(Say '{args.wake_word}' to transcribe again)")
    
    def on_speech_start(timestamp):
        if args.verbose:
            print("🎤 Speech detected...")
    
    def on_speech_end(timestamp):
        if args.verbose:
            print("   Processing...")
    
    def on_wake_word(word, confidence):
        nonlocal wake_word_active
        wake_word_active = True
        print(f"\n✨ Wake word '{word}' detected! Listening...")
    
    def on_error(error):
        print(f"❌ Error: {error}")
    
    # Create session
    session = TranscriptionSession(
        model=model_config,
        vad=vad_config,
        wake_word=wake_word_config,
        device_id=args.device,
        on_transcription=on_transcription,
        on_speech_start=on_speech_start if args.verbose else None,
        on_speech_end=on_speech_end if args.verbose else None,
        on_wake_word=on_wake_word if args.wake_word else None,
        on_error=on_error,
        verbose=args.verbose
    )
    
    # Start session
    if not session.start():
        print("Failed to start session")
        return
    
    # Instructions
    if args.wake_word:
        print(f"Say '{args.wake_word}' to start transcribing.")
    else:
        print("Listening... (Press Ctrl+C to stop)")
    print()
    
    # Run until interrupted
    try:
        while session.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    # Stop session
    session.stop()
    
    # Summary
    print(f"\nTranscribed {transcription_count} segments.")


if __name__ == "__main__":
    main()