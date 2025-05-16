#!/usr/bin/env python3
"""
Real-world transcription test using an actual audio file.

This test verifies that the Transcription feature can properly load, process,
and transcribe a real MP3 file on the system.
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Any, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TranscriptionRealTest")

# Add project root to path to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus

# Feature imports
from src.Features.Transcription.TranscriptionModule import TranscriptionModule


class TranscriptionRealFileTest:
    """Test class for transcribing a real audio file."""

    def __init__(self, audio_file_path: str, output_path: Optional[str] = None,
                 language: Optional[str] = "no", log_to_console: bool = True):
        """
        Initialize the test with file paths.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            output_path: Path to save the transcription result (optional)
            language: Language code for transcription (default: "no" for Norwegian)
            log_to_console: Whether to log results to console
        """
        self.audio_file_path = audio_file_path
        
        # Setup output path
        if output_path is None:
            base_dir = os.path.dirname(audio_file_path)
            filename = os.path.splitext(os.path.basename(audio_file_path))[0]
            self.output_path = os.path.join(base_dir, f"{filename}_transcription.txt")
        else:
            self.output_path = output_path
            
        self.language = language
        self.log_to_console = log_to_console
        
        # Initialize system components
        self.command_dispatcher = CommandDispatcher()
        self.event_bus = EventBus()
        
        # Test state
        self.transcription_results = []
        self.transcription_complete = False
        self.performance_metrics = {
            "start_time": 0,
            "end_time": 0,
            "processing_time": 0,
            "audio_duration": 0,
            "realtime_factor": 0
        }
        
        # Setup logging file handler
        self.setup_logging()
        
    def setup_logging(self):
        """Setup file logging."""
        log_path = self.output_path.replace('.txt', '_log.txt')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    def setup_transcription(self):
        """Register and configure the Transcription feature."""
        logger.info(f"Registering Transcription feature")
        
        # Register the feature
        self.handler = TranscriptionModule.register(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus,
            default_engine="mlx_whisper",
            default_model="whisper-large-v3-turbo",
            default_language=self.language
        )
        
        # Configure for batch processing (non-streaming)
        TranscriptionModule.configure(
            command_dispatcher=self.command_dispatcher,
            engine_type="mlx_whisper",
            streaming=False,  # Use batch mode for higher accuracy
            language=self.language,
            compute_type="float16",
            beam_size=2  # Use beam search for better accuracy
        )
        
        logger.info(f"Transcription feature registered and configured")
        
        # Setup event handlers
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Setup event handlers for transcription events."""
        # Handler for transcription updates
        def on_transcription_updated(session_id, text, is_final, confidence):
            log_message = f"Transcription update: {'FINAL' if is_final else 'partial'}, " \
                          f"confidence: {confidence:.2f}"
            logger.info(log_message)
            
            if self.log_to_console:
                print(log_message)
                print(f"Text: {text}")
                
            self.transcription_results.append({
                "session_id": session_id,
                "text": text,
                "is_final": is_final,
                "confidence": confidence,
                "timestamp": time.time()
            })
            
            if is_final:
                self.transcription_complete = True
                self.performance_metrics["end_time"] = time.time()
                self.performance_metrics["processing_time"] = (
                    self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
                )
        
        # Handler for transcription errors
        def on_transcription_error(session_id, error_message, error_type):
            error_log = f"Transcription error ({error_type}): {error_message}"
            logger.error(error_log)
            if self.log_to_console:
                print(f"ERROR: {error_log}")
        
        # Subscribe to events
        TranscriptionModule.on_transcription_updated(self.event_bus, on_transcription_updated)
        TranscriptionModule.on_transcription_error(self.event_bus, on_transcription_error)
        
        logger.info("Event handlers registered")
    
    def load_audio(self):
        """
        Load and preprocess the audio file.
        
        Returns:
            tuple: (audio_data, sample_rate, duration)
        """
        try:
            logger.info(f"Loading audio file: {self.audio_file_path}")
            
            # Import here to avoid requiring these dependencies for the whole project
            import librosa
            import numpy as np
            
            # Load audio file
            logger.info("Loading audio using librosa...")
            audio_data, sample_rate = librosa.load(
                self.audio_file_path, 
                sr=16000,  # Resample to 16kHz
                mono=True   # Convert to mono
            )
            
            # Calculate duration
            duration = len(audio_data) / sample_rate
            logger.info(f"Audio loaded: {duration:.2f} seconds, {sample_rate} Hz")
            
            # Ensure correct format and normalization
            audio_data = audio_data.astype(np.float32)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
                
            # Store metrics
            self.performance_metrics["audio_duration"] = duration
                
            return audio_data, sample_rate, duration
            
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}", exc_info=True)
            if self.log_to_console:
                print(f"ERROR: Could not load audio file: {str(e)}")
            raise
    
    def transcribe_file(self):
        """Transcribe the audio file and measure performance."""
        try:
            # Load audio
            audio_data, sample_rate, duration = self.load_audio()
            
            logger.info(f"Starting transcription of {duration:.2f} seconds of audio...")
            self.performance_metrics["start_time"] = time.time()
            
            # Transcribe the file
            result = TranscriptionModule.transcribe_file(
                command_dispatcher=self.command_dispatcher,
                file_path=self.audio_file_path,
                language=self.language,
                streaming=False  # Use batch mode for better accuracy
            )
            
            # In case the event handler didn't fire, store the result
            if not self.transcription_results:
                # Handle different result types
                if isinstance(result, dict):
                    self.transcription_results.append({
                        "session_id": "direct",
                        "text": result.get("text", ""),
                        "is_final": True,
                        "confidence": result.get("confidence", 1.0),
                        "timestamp": time.time()
                    })
                elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    # If result is a list, use the first item
                    self.transcription_results.append({
                        "session_id": "direct",
                        "text": result[0].get("text", ""),
                        "is_final": True,
                        "confidence": result[0].get("confidence", 1.0),
                        "timestamp": time.time()
                    })
                else:
                    # Fallback for unexpected result format
                    self.transcription_results.append({
                        "session_id": "direct",
                        "text": str(result) if result else "",
                        "is_final": True,
                        "confidence": 1.0,
                        "timestamp": time.time()
                    })
                
            # Calculate realtime factor
            if self.performance_metrics["processing_time"] > 0 and duration > 0:
                self.performance_metrics["realtime_factor"] = (
                    self.performance_metrics["processing_time"] / duration
                )
                
            # Log performance
            logger.info(f"Transcription completed in {self.performance_metrics['processing_time']:.2f} seconds")
            logger.info(f"Realtime factor: {self.performance_metrics['realtime_factor']:.2f}x")
            
            if self.log_to_console:
                print(f"\nTranscription completed in {self.performance_metrics['processing_time']:.2f} seconds")
                print(f"Realtime factor: {self.performance_metrics['realtime_factor']:.2f}x realtime")
                
            return True
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}", exc_info=True)
            if self.log_to_console:
                print(f"ERROR: Transcription failed: {str(e)}")
            return False
    
    def save_results(self):
        """Save transcription results to file."""
        try:
            # Get final transcription (should be the last final result)
            final_result = None
            for result in reversed(self.transcription_results):
                if result["is_final"]:
                    final_result = result
                    break
                    
            if not final_result and self.transcription_results:
                final_result = self.transcription_results[-1]
                
            if not final_result:
                logger.warning("No transcription results to save")
                return False
                
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
                
            # Save the transcription text
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(final_result["text"])
                
            # Save detailed results including metrics
            detailed_path = self.output_path.replace('.txt', '_detailed.json')
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "transcription_results": self.transcription_results,
                    "performance_metrics": self.performance_metrics,
                    "file_info": {
                        "path": self.audio_file_path,
                        "duration": self.performance_metrics["audio_duration"],
                        "language": self.language
                    }
                }, f, indent=2)
                
            logger.info(f"Results saved to {self.output_path} and {detailed_path}")
            if self.log_to_console:
                print(f"\nResults saved to:")
                print(f"  - {self.output_path} (plain text)")
                print(f"  - {detailed_path} (detailed JSON)")
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            if self.log_to_console:
                print(f"ERROR: Could not save results: {str(e)}")
            return False
    
    def run_test(self):
        """Run the complete test."""
        try:
            logger.info(f"Starting real-world transcription test")
            logger.info(f"Audio file: {self.audio_file_path}")
            logger.info(f"Language: {self.language}")
            
            if self.log_to_console:
                print(f"Starting real-world transcription test")
                print(f"Audio file: {self.audio_file_path}")
                print(f"Language: {self.language}")
                print(f"Output path: {self.output_path}")
                print("-" * 50)
            
            # Setup
            self.setup_transcription()
            
            # Execute
            success = self.transcribe_file()
            
            # Save results
            if success:
                self.save_results()
                
                # Print final transcription
                if self.log_to_console:
                    final_text = None
                    for result in reversed(self.transcription_results):
                        if result["is_final"]:
                            final_text = result["text"]
                            break
                    
                    if final_text:
                        print("\nFinal Transcription:")
                        print("-" * 50)
                        print(final_text)
                        print("-" * 50)
            
            logger.info("Test completed")
            
            # Final summary
            if self.log_to_console:
                status = "SUCCESS" if success else "FAILED"
                print(f"\nTest completed: {status}")
                
            return success
            
        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}", exc_info=True)
            if self.log_to_console:
                print(f"TEST FAILED: {str(e)}")
            return False


def main():
    """Run the transcription test."""
    parser = argparse.ArgumentParser(description="Transcription Real File Test")
    parser.add_argument("--audio-file", type=str, 
                      default="/Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/bok_konge01.mp3",
                      help="Path to the audio file to transcribe")
    parser.add_argument("--output-path", type=str, default=None,
                      help="Path to save the transcription result")
    parser.add_argument("--language", type=str, default="no",
                      help="Language code for transcription (default: 'no' for Norwegian)")
    parser.add_argument("--quiet", action="store_true",
                      help="Run without console output")
    
    args = parser.parse_args()
    
    test = TranscriptionRealFileTest(
        audio_file_path=args.audio_file,
        output_path=args.output_path,
        language=args.language,
        log_to_console=not args.quiet
    )
    
    success = test.run_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()