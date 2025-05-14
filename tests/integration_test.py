#!/usr/bin/env python3
"""
Integration Test for Realtime_mlx_STT

This script tests the integration between the audio processing pipeline
and the MLX-optimized transcriber for real-time speech-to-text transcription.
"""

import os
import sys
import time
import argparse
import logging
import threading

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RealtimeSTT.audio_input import AudioInput
from RealtimeSTT.mlx_transcriber import MLXTranscriber

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("integration_test")

class SimpleSTTRecorder:
    """
    A simplified recorder that integrates AudioInput with MLXTranscriber.
    This demonstrates the basic integration between the components.
    """
    def __init__(self, input_device_index=None, debug_mode=False):
        # Initialize audio input
        self.audio_input = AudioInput(
            input_device_index=input_device_index,
            debug_mode=debug_mode
        )
        
        # Initialize transcriber
        self.transcriber = MLXTranscriber(
            realtime_mode=True,
            any_lang=False,
            quick=True
        )
        
        # Control flags
        self.running = False
        self.recording = False
        self.recorded_audio = []
        
        # Threading
        self.record_thread = None
        self.process_thread = None
    
    def start(self):
        """Start the recorder and transcriber."""
        logger.info("Starting recorder and transcriber")
        
        # Start the audio input
        if not self.audio_input.setup():
            logger.error("Failed to set up audio input")
            return False
        
        # Start the transcriber
        if not self.transcriber.start():
            logger.error("Failed to start transcriber")
            self.audio_input.cleanup()
            return False
        
        # Set control flags
        self.running = True
        self.recording = False
        
        # Start the recording thread
        self.record_thread = threading.Thread(
            target=self._record_thread_func,
            name="RecordThread",
            daemon=True
        )
        self.record_thread.start()
        
        # Start the processing thread
        self.process_thread = threading.Thread(
            target=self._process_thread_func,
            name="ProcessThread",
            daemon=True
        )
        self.process_thread.start()
        
        logger.info("Recorder and transcriber started successfully")
        return True
    
    def stop(self):
        """Stop the recorder and transcriber."""
        logger.info("Stopping recorder and transcriber")
        
        # Set control flags
        self.running = False
        self.recording = False
        
        # Wait for threads to finish
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        # Clean up resources
        self.audio_input.cleanup()
        self.transcriber.cleanup()
        
        logger.info("Recorder and transcriber stopped")
    
    def start_recording(self):
        """Start recording audio."""
        logger.info("Starting audio recording")
        self.recording = True
        self.recorded_audio = []
    
    def stop_recording(self):
        """Stop recording audio and process the final result."""
        if not self.recording:
            return
        
        logger.info("Stopping audio recording")
        self.recording = False
        
        # Process any remaining audio
        if self.recorded_audio:
            import numpy as np
            audio_data = np.concatenate(self.recorded_audio)
            logger.info(f"Processing final audio chunk of size {len(audio_data)}")
            self.transcriber.add_audio_chunk(audio_data, is_last=True)
            self.recorded_audio = []
    
    def _record_thread_func(self):
        """Thread function for recording audio."""
        import numpy as np
        import struct
        
        logger.info("Record thread started")
        
        while self.running:
            # Check if we're recording
            if not self.recording:
                time.sleep(0.1)
                continue
            
            # Read audio data
            try:
                raw_data = self.audio_input.read_chunk()
                
                # Convert to numpy array
                fmt = f"{self.audio_input.chunk_size}h"
                pcm_data = np.array(struct.unpack(fmt, raw_data)) / 32768.0
                
                # Resample if needed
                if (self.audio_input.device_sample_rate != self.audio_input.target_samplerate and
                    self.audio_input.resample_to_target):
                    pcm_data = self.audio_input.resample_audio(
                        pcm_data,
                        self.audio_input.target_samplerate,
                        self.audio_input.device_sample_rate
                    )
                
                # Add to buffer
                self.recorded_audio.append(pcm_data)
                
                # Submit for processing
                self.transcriber.add_audio_chunk(pcm_data, is_last=False)
                
            except Exception as e:
                logger.error(f"Error in record thread: {e}")
                time.sleep(0.1)
        
        logger.info("Record thread stopped")
    
    def _process_thread_func(self):
        """Thread function for processing transcription results."""
        logger.info("Process thread started")
        
        while self.running:
            # Check for transcription results
            result = self.transcriber.get_result(timeout=0.5)
            if result:
                logger.info(f"Transcription result: {result['text']}")
                logger.info(f"Processing time: {result['processing_time']:.2f}s")
        
        logger.info("Process thread stopped")

def test_microphone_transcription(input_device=None, duration=10):
    """
    Test transcription from the microphone.
    
    Args:
        input_device (int): Index of the input device to use
        duration (int): How long to record in seconds
    """
    logger.info("Starting microphone transcription test")
    logger.info(f"Using input device: {input_device if input_device is not None else 'default'}")
    logger.info(f"Recording duration: {duration} seconds")
    
    # Create the recorder
    recorder = SimpleSTTRecorder(
        input_device_index=input_device,
        debug_mode=True
    )
    
    try:
        # Start the recorder
        if not recorder.start():
            logger.error("Failed to start recorder")
            return
        
        # Start recording
        logger.info("Starting recording... Speak now!")
        recorder.start_recording()
        
        # Wait for the specified duration
        time.sleep(duration)
        
        # Stop recording
        logger.info("Stopping recording...")
        recorder.stop_recording()
        
        # Wait for final processing
        logger.info("Waiting for final transcription...")
        time.sleep(5)
        
    finally:
        # Clean up
        recorder.stop()
    
    logger.info("Microphone transcription test completed")

def list_audio_devices():
    """List available audio input devices."""
    audio_input = AudioInput()
    audio_input.list_devices()

def main():
    parser = argparse.ArgumentParser(description="Test Realtime_mlx_STT integration")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index to use")
    parser.add_argument("--duration", type=int, default=10,
                        help="Recording duration in seconds")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    test_microphone_transcription(
        input_device=args.device,
        duration=args.duration
    )

if __name__ == "__main__":
    main()