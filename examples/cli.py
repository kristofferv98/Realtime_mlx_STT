#!/usr/bin/env python3
"""
Realtime MLX STT - Interactive CLI

A user-friendly command-line interface for exploring the Realtime MLX STT library.
"""

import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_mlx_stt import STTClient, create_client, list_audio_devices


class STTDemo:
    """Interactive STT demonstration CLI."""
    
    def __init__(self):
        self.client: Optional[STTClient] = None
        self.openai_available = bool(os.environ.get('OPENAI_API_KEY'))
        self.porcupine_available = bool(os.environ.get('PORCUPINE_ACCESS_KEY'))
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def print_header(self, title: str):
        """Print a formatted header."""
        self.clear_screen()
        print("🎙️  Realtime MLX STT - Interactive Demo")
        print("=" * 50)
        print(f"\n{title}")
        print("-" * 50)
    
    def print_menu(self):
        """Print the main menu."""
        self.print_header("Main Menu")
        print("\n1. 🎯 Quick Start (10-second transcription)")
        print("2. ⏺️  Continuous Transcription (with stop word)")
        print("3. 🌐 OpenAI Transcription" + (" ✅" if self.openai_available else " ❌ (API key required)"))
        print("4. 🗣️  Wake Word Detection" + (" ✅" if self.porcupine_available else " ❌ (API key required)"))
        print("5. 🎛️  Audio Device Selection")
        print("6. 🌍 Language Settings")
        print("7. ℹ️  About & Help")
        print("8. 🚪 Exit")
        print("\n" + "-" * 50)
    
    def quick_transcription(self):
        """Run a quick 10-second transcription."""
        self.print_header("Quick Transcription")
        print("\n🎯 Transcribing for 10 seconds...")
        print("💡 Tip: Speak clearly into your microphone\n")
        
        if not self.client:
            self.client = STTClient()
        
        count = 0
        for result in self.client.transcribe(duration=10):
            count += 1
            print(f"📝 {result.text}")
            if result.confidence < 1.0:
                print(f"   (confidence: {result.confidence:.2f})")
            print()
        
        if count == 0:
            print("🔇 No speech detected. Make sure your microphone is working.")
        else:
            print(f"\n✅ Captured {count} transcription(s)")
        
        input("\nPress Enter to continue...")
    
    def continuous_transcription(self):
        """Run continuous transcription with stop word."""
        self.print_header("Continuous Transcription")
        print("\n⏺️  Streaming transcription mode")
        print("💡 Say 'stop recording' to end\n")
        
        if not self.client:
            self.client = STTClient()
        
        print("Listening...\n")
        
        with self.client.stream() as stream:
            for result in stream:
                print(f">>> {result.text}")
                
                if "stop recording" in result.text.lower():
                    print("\n🛑 Stop command detected!")
                    break
        
        print("\n✅ Streaming ended")
        input("\nPress Enter to continue...")
    
    def openai_transcription(self):
        """Run OpenAI-powered transcription."""
        if not self.openai_available:
            self.print_header("OpenAI Transcription")
            print("\n❌ OpenAI API key not found!")
            print("\nTo use OpenAI transcription:")
            print("1. Get an API key from https://platform.openai.com/")
            print("2. Set it: export OPENAI_API_KEY='your-key-here'")
            input("\nPress Enter to continue...")
            return
        
        self.print_header("OpenAI Transcription")
        print("\n🌐 Using OpenAI Whisper API")
        print("⏱️  Transcribing for 15 seconds...\n")
        
        if not self.client:
            self.client = STTClient(openai_api_key=os.environ.get('OPENAI_API_KEY'))
        
        count = 0
        for result in self.client.transcribe(duration=15, engine="openai"):
            count += 1
            print(f"🌐 {result.text}\n")
        
        if count == 0:
            print("🔇 No speech detected")
        else:
            print(f"✅ Received {count} transcription(s) via OpenAI")
        
        input("\nPress Enter to continue...")
    
    def wake_word_demo(self):
        """Run wake word detection demo."""
        if not self.porcupine_available:
            self.print_header("Wake Word Detection")
            print("\n❌ Porcupine API key not found!")
            print("\nTo use wake word detection:")
            print("1. Get a free key from https://picovoice.ai/")
            print("2. Set it: export PORCUPINE_ACCESS_KEY='your-key-here'")
            input("\nPress Enter to continue...")
            return
        
        self.print_header("Wake Word Detection")
        print("\n🗣️  Wake word: 'Jarvis'")
        print("💡 Say 'Jarvis' followed by your command")
        print("⏱️  Running for 30 seconds...\n")
        
        if not self.client:
            self.client = STTClient(porcupine_api_key=os.environ.get('PORCUPINE_ACCESS_KEY'))
        
        activations = 0
        
        def on_wake(word, confidence):
            nonlocal activations
            activations += 1
            print(f"\n✨ Wake word detected! (confidence: {confidence:.2f})")
            print("   Listening for command...\n")
        
        def on_command(result):
            print(f"   📝 Command: {result.text}\n")
            print("Say 'Jarvis' again...")
        
        self.client.start_wake_word(
            wake_word="jarvis",
            on_wake=on_wake,
            on_transcription=on_command
        )
        
        time.sleep(30)
        self.client.stop()
        
        print(f"\n✅ Session ended. Wake word activated {activations} time(s)")
        input("\nPress Enter to continue...")
    
    def select_audio_device(self):
        """Select audio input device."""
        self.print_header("Audio Device Selection")
        
        devices = list_audio_devices()
        
        print("\n📱 Available audio devices:\n")
        for device in devices:
            marker = "→" if device.is_default else " "
            print(f"{marker} [{device.index}] {device.name}")
            if device.channels:
                print(f"     Channels: {device.channels}, Rate: {device.sample_rate}Hz")
        
        print(f"\n💡 Current device: {getattr(self.client.config if self.client else None, 'default_device', 'System default')}")
        
        try:
            choice = input("\nEnter device number (or Enter for default): ").strip()
            if choice:
                device_idx = int(choice)
                if not self.client:
                    self.client = STTClient(device_index=device_idx)
                else:
                    self.client.set_device(device_idx)
                print(f"\n✅ Selected device [{device_idx}]")
            else:
                print("\n✅ Using system default")
        except ValueError:
            print("\n❌ Invalid selection")
        
        input("\nPress Enter to continue...")
    
    def language_settings(self):
        """Configure language settings."""
        self.print_header("Language Settings")
        
        languages = {
            "1": ("en", "English"),
            "2": ("es", "Spanish"),
            "3": ("fr", "French"),
            "4": ("de", "German"),
            "5": ("it", "Italian"),
            "6": ("pt", "Portuguese"),
            "7": ("nl", "Dutch"),
            "8": ("ru", "Russian"),
            "9": ("zh", "Chinese"),
            "10": ("ja", "Japanese"),
            "11": ("ko", "Korean"),
            "12": ("ar", "Arabic"),
            "13": ("hi", "Hindi"),
            "14": ("sv", "Swedish"),
            "15": ("no", "Norwegian"),
            "16": (None, "Auto-detect")
        }
        
        print("\n🌍 Select transcription language:\n")
        for key, (code, name) in languages.items():
            print(f"{key:>2}. {name}")
        
        current = getattr(self.client.config if self.client else None, 'default_language', None)
        current_name = "Auto-detect" if current is None else current
        print(f"\n💡 Current: {current_name}")
        
        choice = input("\nEnter number (or Enter to keep current): ").strip()
        
        if choice in languages:
            code, name = languages[choice]
            if not self.client:
                self.client = STTClient(default_language=code)
            else:
                self.client.set_language(code)
            print(f"\n✅ Language set to: {name}")
        
        input("\nPress Enter to continue...")
    
    def show_help(self):
        """Show help and about information."""
        self.print_header("About & Help")
        
        print("\n📚 Realtime MLX STT")
        print("High-performance speech-to-text for Apple Silicon\n")
        
        print("🚀 Features:")
        print("• Local transcription using MLX Whisper")
        print("• Cloud transcription via OpenAI API")
        print("• Voice activity detection (VAD)")
        print("• Wake word detection")
        print("• Real-time streaming")
        print("• Multi-language support\n")
        
        print("⚙️  Requirements:")
        print("• Apple Silicon Mac (M1/M2/M3)")
        print("• Python 3.9+")
        print("• Working microphone\n")
        
        print("🔑 Optional API Keys:")
        print("• OPENAI_API_KEY - For cloud transcription")
        print("• PORCUPINE_ACCESS_KEY - For wake words\n")
        
        print("📖 Documentation:")
        print("https://github.com/kristofferv98/Realtime_mlx_STT")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Run the interactive CLI."""
        while True:
            self.print_menu()
            
            choice = input("Select an option (1-8): ").strip()
            
            if choice == "1":
                self.quick_transcription()
            elif choice == "2":
                self.continuous_transcription()
            elif choice == "3":
                self.openai_transcription()
            elif choice == "4":
                self.wake_word_demo()
            elif choice == "5":
                self.select_audio_device()
            elif choice == "6":
                self.language_settings()
            elif choice == "7":
                self.show_help()
            elif choice == "8":
                self.clear_screen()
                print("👋 Thanks for using Realtime MLX STT!")
                break
            else:
                print("\n❌ Invalid option. Please try again.")
                time.sleep(1)


def main():
    """Main entry point."""
    try:
        demo = STTDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()