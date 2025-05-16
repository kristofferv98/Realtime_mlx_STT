#!/usr/bin/env python3
"""
Real-world end-to-end transcription test using the actual MLX model.

This test fully exercises the Transcription feature with a real audio file,
using the actual MLX-optimized Whisper model without any mocking.
It is designed for environments with high-performance Apple Silicon hardware.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import json
import signal
from typing import Optional, Dict, Any, List
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RealWorldTranscriptionTest")

# Add project root to path to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Core imports
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Core.Events.event_bus import EventBus

# Feature imports
from src.Features.Transcription.TranscriptionModule import TranscriptionModule
from src.Features.Transcription.Models.TranscriptionConfig import TranscriptionConfig
from src.Features.Transcription.Models.TranscriptionResult import TranscriptionResult


class RealWorldTranscriptionTest:
    """Real-world end-to-end test using the actual MLX Whisper model."""

    def __init__(self, 
                audio_file_path: str, 
                output_path: Optional[str] = None,
                language: Optional[str] = None, 
                model_name: str = "whisper-large-v3-turbo",
                compute_type: str = "float16",
                beam_size: int = 1,
                log_to_console: bool = True):
        """
        Initialize the test with file paths and configuration options.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            output_path: Path to save the transcription result (optional)
            language: Language code for transcription (default: auto-detect)
            model_name: Model name to use (default: whisper-large-v3-turbo)
            compute_type: Computation precision (default: float16)
            beam_size: Beam search size (default: 1 for greedy decoding)
            log_to_console: Whether to log results to console
        """
        self.audio_file_path = audio_file_path
        
        # Setup output path
        if output_path is None:
            base_dir = os.path.dirname(audio_file_path)
            filename = os.path.splitext(os.path.basename(audio_file_path))[0]
            self.output_path = os.path.join(base_dir, f"{filename}_transcription_real.txt")
        else:
            self.output_path = output_path
            
        # Configuration options
        self.language = language
        self.model_name = model_name
        self.compute_type = compute_type
        self.beam_size = beam_size
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
            "realtime_factor": 0,
            "model_name": self.model_name,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size
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
        """Register and configure the Transcription feature with actual MLX engine."""
        logger.info(f"Registering Transcription feature with model={self.model_name}")
        
        # Register the feature
        self.handler = TranscriptionModule.register(
            command_dispatcher=self.command_dispatcher,
            event_bus=self.event_bus,
            default_engine="mlx_whisper",
            default_model=self.model_name,
            default_language=self.language
        )
        
        # Configure for batch processing with specific settings
        TranscriptionModule.configure(
            command_dispatcher=self.command_dispatcher,
            engine_type="mlx_whisper",
            model_name=self.model_name,
            language=self.language,
            streaming=False,  # Use batch mode for higher accuracy
            compute_type=self.compute_type,
            beam_size=self.beam_size,
            chunk_duration_ms=30000,  # Use larger chunks for batch processing
            chunk_overlap_ms=200
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
            
            logger.info(f"Starting transcription of {duration:.2f} seconds of audio with {self.model_name}")
            self.performance_metrics["start_time"] = time.time()
            
            # Add a safety timeout for the transcription process
            # This helps prevent the test from running indefinitely
            transcription_max_time = max(180.0, duration * 10)  # At least 3 minutes or 10x audio duration
            logger.info(f"Setting transcription maximum time to {transcription_max_time:.1f} seconds")
            
            # Key difference: use the actual MLX engine with full batch processing
            # This will download and run the full model (large resource usage)
            logger.info(f"Using MLX engine with model={self.model_name}, compute_type={self.compute_type}")
            
            # Use batch transcribe mode for better accuracy
            result = TranscriptionModule.transcribe_file(
                command_dispatcher=self.command_dispatcher,
                file_path=self.audio_file_path,
                language=self.language,
                compute_type=self.compute_type,
                beam_size=self.beam_size,
                streaming=False,  # Use batch mode for better accuracy
                timeout=transcription_max_time  # Dynamic timeout based on audio length
            )
            
            # In case the event handler didn't fire, store the result
            if not self.transcription_results:
                logger.info(f"No event-based results received, using direct result: {result}")
                
                # Handle different result types
                if isinstance(result, dict):
                    text = result.get("text", "")
                    success = result.get("success", False)
                    error = result.get("error", None)
                    
                    # Check for error condition
                    if error:
                        logger.error(f"Transcription error: {error}")
                        if self.log_to_console:
                            print(f"ERROR: Transcription failed: {error}")
                            
                        # Add error result for tracking
                        self.transcription_results.append({
                            "session_id": "direct",
                            "text": "",
                            "is_final": True,
                            "confidence": 0.0,
                            "timestamp": time.time(),
                            "error": error,
                            "success": False
                        })
                        
                        # Ensure we have an end time for metrics
                        if not self.performance_metrics["end_time"]:
                            self.performance_metrics["end_time"] = time.time()
                            self.performance_metrics["processing_time"] = (
                                self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
                            )
                            
                        return False
                    
                    # Add normal result for tracking - check for empty text as a warning
                    if not text and success:
                        logger.warning("Transcription returned empty text despite success flag")
                        
                    self.transcription_results.append({
                        "session_id": "direct",
                        "text": text,
                        "is_final": True,
                        "confidence": result.get("confidence", 1.0),
                        "timestamp": time.time(),
                        "token_count": result.get("token_count", 0),
                        "success": success
                    })
                    
                elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                    # If result is a list, use the first item
                    text = result[0].get("text", "")
                    success = result[0].get("success", False)
                    
                    self.transcription_results.append({
                        "session_id": "direct",
                        "text": text,
                        "is_final": True,
                        "confidence": result[0].get("confidence", 1.0),
                        "timestamp": time.time(),
                        "token_count": result[0].get("token_count", 0),
                        "success": success
                    })
                    
                    # Check for empty result as warning
                    if not text and success:
                        logger.warning("Transcription returned empty text despite success flag")
                        
                else:
                    # Fallback for unexpected result format
                    logger.warning(f"Received unexpected result format: {type(result)}")
                    result_str = str(result) if result else ""
                    
                    self.transcription_results.append({
                        "session_id": "direct",
                        "text": result_str,
                        "is_final": True,
                        "confidence": 1.0,
                        "timestamp": time.time(),
                        "success": bool(result_str)  # Consider successful if we got some text
                    })
                    
            # Ensure we have an end time
            if not self.performance_metrics["end_time"]:
                self.performance_metrics["end_time"] = time.time()
                self.performance_metrics["processing_time"] = (
                    self.performance_metrics["end_time"] - self.performance_metrics["start_time"]
                )
                
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
                
            # Validate success criteria - we need at least one result with text
            has_successful_transcription = False
            for result in self.transcription_results:
                if result.get("text") and not result.get("error"):
                    has_successful_transcription = True
                    break
                    
            if not has_successful_transcription:
                logger.warning("No successful transcription with text was produced")
                if self.log_to_console:
                    print("WARNING: No successful transcription with text was produced")
                    
            return has_successful_transcription
            
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
                    },
                    "test_info": {
                        "test_time": datetime.now().isoformat(),
                        "model_name": self.model_name,
                        "compute_type": self.compute_type,
                        "beam_size": self.beam_size
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
            logger.info(f"Starting real-world transcription test with full MLX model")
            logger.info(f"Audio file: {self.audio_file_path}")
            logger.info(f"Language: {self.language or 'auto'}")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Compute type: {self.compute_type}")
            logger.info(f"Beam size: {self.beam_size}")
            
            if self.log_to_console:
                print(f"Starting real-world transcription test with full MLX model")
                print(f"Audio file: {self.audio_file_path}")
                print(f"Language: {self.language or 'auto'}")
                print(f"Model: {self.model_name}")
                print(f"Compute type: {self.compute_type}")
                print(f"Beam size: {self.beam_size}")
                print(f"Output path: {self.output_path}")
                print("-" * 80)
            
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
                        if result.get("is_final") and result.get("text"):
                            final_text = result["text"]
                            break
                    
                    if final_text:
                        print("\nFinal Transcription:")
                        print("-" * 80)
                        print(final_text)
                        print("-" * 80)
                    else:
                        print("\nNo transcription text was produced!")
                        # Print error if any
                        for result in self.transcription_results:
                            if result.get("error"):
                                print(f"Error: {result['error']}")
                                break
            
            logger.info("Test completed")
            
            # Final summary with more details
            if self.log_to_console:
                if success:
                    status = "SUCCESS"
                    token_count = 0
                    for result in self.transcription_results:
                        token_count = max(token_count, result.get("token_count", 0))
                    
                    print(f"\nTest completed: {status}")
                    print(f"Performance:")
                    print(f"- Processing time: {self.performance_metrics['processing_time']:.2f}s")
                    print(f"- Audio duration: {self.performance_metrics['audio_duration']:.2f}s")
                    print(f"- Realtime factor: {self.performance_metrics['realtime_factor']:.3f}x realtime")
                    if token_count > 0:
                        print(f"- Tokens generated: {token_count}")
                else:
                    print(f"\nTest completed: FAILED")
                    print("- No valid transcription was produced")
                    # Print diagnostic info
                    print("- Possible causes:")
                    print("  * Model initialization failed")
                    print("  * Audio processing error")
                    print("  * Transcription engine error")
                    print("  * Check logs for detailed error information")
                
            return success
            
        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}", exc_info=True)
            if self.log_to_console:
                print(f"TEST FAILED: {str(e)}")
            return False


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    print("\n\nERROR: Test timed out! The test has been running too long.")
    print("This could be due to model download issues, transcription hanging, or other problems.")
    print("Check the logs for more details.")
    # Exit with error code for timeout
    sys.exit(2)

def main():
    """Run the real-world transcription test."""
    parser = argparse.ArgumentParser(description="Real-World Transcription Test")
    parser.add_argument("--audio-file", type=str, 
                      default="/Users/kristoffervatnehol/Code/projects/Realtime_mlx_STT/bok_konge01.mp3",
                      help="Path to the audio file to transcribe")
    parser.add_argument("--output-path", type=str, default=None,
                      help="Path to save the transcription result")
    parser.add_argument("--language", type=str, default=None,
                      help="Language code for transcription (default: auto-detect)")
    parser.add_argument("--model", type=str, default="whisper-large-v3-turbo",
                      help="Model name (default: whisper-large-v3-turbo)")
    parser.add_argument("--compute-type", type=str, default="float16",
                      choices=["float16", "float32"],
                      help="Computation precision (default: float16)")
    parser.add_argument("--beam-size", type=int, default=1,
                      help="Beam search size for inference (default: 1 for greedy)")
    parser.add_argument("--quiet", action="store_true",
                      help="Run without console output")
    parser.add_argument("--timeout", type=int, default=300,
                      help="Test timeout in seconds (default: 300)")
    
    args = parser.parse_args()
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    
    try:
        test = RealWorldTranscriptionTest(
            audio_file_path=args.audio_file,
            output_path=args.output_path,
            language=args.language,
            model_name=args.model,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            log_to_console=not args.quiet
        )
        
        success = test.run_test()
        
        # Disable the alarm
        signal.alarm(0)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
    except Exception as e:
        # Disable the alarm
        signal.alarm(0)
        print(f"\n\nERROR: Test failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()