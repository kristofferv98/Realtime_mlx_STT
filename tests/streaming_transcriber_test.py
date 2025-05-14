#!/usr/bin/env python3
"""
Streaming Transcriber Test

This script tests the streaming transcription capabilities of the MLX-optimized
whisper-large-v3-turbo model in the Realtime_mlx_STT library.
"""

import os
import sys
import time
import json
import numpy as np
import argparse
import logging
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RealtimeSTT.whisper_turbo import StreamingTranscriber, create_streaming_transcriber
from RealtimeSTT.mlx_transcriber import MLXTranscriber

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streaming_transcriber_test")

def test_direct_streaming(audio_file, chunk_size=4000, overlap=2000, buffer_size=16000, any_lang=False):
    """
    Test the StreamingTranscriber directly (without using MLXTranscriber).
    
    Args:
        audio_file (str): Path to an audio file for testing
        chunk_size (int): Size of audio chunks to feed to the transcriber
        overlap (int): Overlap between processing segments
        buffer_size (int): Buffer size for the streaming transcriber
        any_lang (bool): Whether to use language detection
        
    Returns:
        dict: Test results with transcription and timing data
    """
    logger.info(f"Testing direct StreamingTranscriber with file: {audio_file}")
    logger.info(f"Parameters: chunk_size={chunk_size}, overlap={overlap}, buffer_size={buffer_size}")
    
    # Load the audio file
    audio_data, sample_rate = sf.read(audio_file)
    
    # Convert to mono if needed
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Convert to float32 if needed
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Normalize if needed
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Create the streaming transcriber
    start_time = time.time()
    logger.info("Creating streaming transcriber...")
    transcriber = create_streaming_transcriber(buffer_size=buffer_size, overlap=overlap)
    model_load_time = time.time() - start_time
    logger.info(f"Streaming transcriber created in {model_load_time:.2f} seconds")
    
    # Process audio in chunks
    chunks = []
    total_chunks = len(audio_data) // chunk_size + 1
    
    results = {
        'partial_results': [],
        'final_result': None,
        'timings': {
            'model_load_time': model_load_time,
            'chunk_processing_times': [],
            'total_processing_time': 0,
            'audio_duration': len(audio_data) / sample_rate,
            'realtime_factor': 0
        }
    }
    
    start_time = time.time()
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        is_last = (i + chunk_size >= len(audio_data))
        
        logger.info(f"Processing chunk {i // chunk_size + 1}/{total_chunks}" +
                  f" ({len(chunk)} samples, {'last' if is_last else 'not last'})")
        
        # Process the chunk
        chunk_start = time.time()
        result = transcriber.process_chunk(
            chunk, 
            is_last=is_last,
            language=None if any_lang else 'en'
        )
        chunk_time = time.time() - chunk_start
        
        # Save timing information
        results['timings']['chunk_processing_times'].append(chunk_time)
        
        # Save partial results
        if result['text']:
            logger.info(f"Partial result: {result['text'][:50]}...")
            results['partial_results'].append({
                'chunk': i // chunk_size + 1,
                'text': result['text'],
                'new_text': result.get('new_text', ''),
                'is_final': result['is_final'],
                'time': chunk_time
            })
        
        # Save final result if this is the last chunk
        if is_last:
            results['final_result'] = result['text']
    
    # Calculate total processing time and realtime factor
    total_time = time.time() - start_time
    results['timings']['total_processing_time'] = total_time
    results['timings']['realtime_factor'] = total_time / results['timings']['audio_duration']
    
    # Display results
    logger.info(f"Streaming transcription completed in {total_time:.2f} seconds")
    logger.info(f"Audio duration: {results['timings']['audio_duration']:.2f} seconds")
    logger.info(f"Realtime factor: {results['timings']['realtime_factor']:.2f}x")
    logger.info(f"Final result: {results['final_result']}")
    
    return results

def test_mlx_streaming(audio_file, chunk_size=4000, any_lang=False, quick=True):
    """
    Test streaming transcription using the MLXTranscriber.
    
    Args:
        audio_file (str): Path to an audio file for testing
        chunk_size (int): Size of audio chunks in samples
        any_lang (bool): Whether to use language detection
        quick (bool): Whether to use quick (parallel) processing
    """
    logger.info(f"Testing MLX streaming transcription with file: {audio_file}")
    
    # Create the transcriber
    transcriber = MLXTranscriber(
        realtime_mode=True,
        any_lang=any_lang,
        quick=quick
    )
    
    # Start the transcriber
    transcriber.start()
    
    results = {
        'partial_results': [],
        'final_result': None,
        'timings': {
            'chunk_processing_times': [],
            'total_processing_time': 0,
            'audio_duration': 0,
            'realtime_factor': 0
        }
    }
    
    try:
        # Load the audio file
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Set audio duration
        results['timings']['audio_duration'] = len(audio_data) / sample_rate
        
        # Process audio in chunks to simulate streaming
        start_time = time.time()
        total_chunks = len(audio_data) // chunk_size + 1
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            is_last = (i + chunk_size >= len(audio_data))
            
            chunk_idx = i // chunk_size + 1
            logger.info(f"Processing chunk {chunk_idx}/{total_chunks}" +
                      f" ({len(chunk)} samples, {'last' if is_last else 'not last'})")
            
            # Add the chunk to the transcriber
            chunk_start = time.time()
            transcriber.add_audio_chunk(chunk, is_last=is_last, request_id=str(i))
            
            # Check for results
            result = None
            max_wait = 5  # wait up to 5 seconds for result
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait:
                result = transcriber.get_result(timeout=0.1)
                if result:
                    break
                time.sleep(0.05)
            
            chunk_time = time.time() - chunk_start
            results['timings']['chunk_processing_times'].append(chunk_time)
            
            if result:
                logger.info(f"Result from chunk {chunk_idx}: {result['text'][:50]}...")
                results['partial_results'].append({
                    'chunk': chunk_idx,
                    'text': result['text'],
                    'new_text': result.get('new_text', ''),
                    'is_final': result.get('is_final', False),
                    'time': chunk_time
                })
                
                if result.get('is_final', False):
                    results['final_result'] = result['text']
            else:
                logger.warning(f"No result received for chunk {chunk_idx}")
            
            # Small delay to simulate real-time processing
            time.sleep(0.1)
        
        # Wait for final result if not received
        if not results['final_result']:
            logger.info("Waiting for final result...")
            max_wait = 10
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait:
                result = transcriber.get_result(timeout=0.5)
                if result and result.get('is_final', False):
                    results['final_result'] = result['text']
                    break
                time.sleep(0.1)
        
        # Calculate metrics
        total_time = time.time() - start_time
        results['timings']['total_processing_time'] = total_time
        results['timings']['realtime_factor'] = total_time / results['timings']['audio_duration']
        
        logger.info(f"MLX streaming transcription completed in {total_time:.2f} seconds")
        logger.info(f"Audio duration: {results['timings']['audio_duration']:.2f} seconds")
        logger.info(f"Realtime factor: {results['timings']['realtime_factor']:.2f}x")
        
        if results['final_result']:
            logger.info(f"Final result: {results['final_result']}")
        else:
            logger.warning("No final result received")
            
        return results
            
    finally:
        # Clean up
        transcriber.stop()
        transcriber.cleanup()

def plot_results(results, output_file=None):
    """
    Plot streaming transcription results.
    
    Args:
        results (dict): Results dictionary from test functions
        output_file (str): Path to save the plot (optional)
    """
    if not results.get('partial_results'):
        logger.warning("No partial results to plot")
        return
    
    # Extract data
    chunks = [r['chunk'] for r in results['partial_results']]
    times = [r['time'] for r in results['partial_results']]
    text_lengths = [len(r['text']) for r in results['partial_results']]
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot processing time per chunk
    ax1.bar(chunks, times, alpha=0.7)
    ax1.axhline(y=results['timings']['audio_duration'] / len(chunks), 
               color='r', linestyle='--', label='Realtime threshold')
    ax1.set_xlabel('Chunk')
    ax1.set_ylabel('Processing Time (s)')
    ax1.set_title('Processing Time per Chunk')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot text length
    ax2.plot(chunks, text_lengths, marker='o', alpha=0.7)
    ax2.set_xlabel('Chunk')
    ax2.set_ylabel('Transcript Length (chars)')
    ax2.set_title('Transcript Length by Chunk')
    ax2.grid(True, alpha=0.3)
    
    # Add overall statistics
    plt.figtext(0.5, 0.01, 
               f"Total processing time: {results['timings']['total_processing_time']:.2f}s | " +
               f"Audio duration: {results['timings']['audio_duration']:.2f}s | " +
               f"Realtime factor: {results['timings']['realtime_factor']:.2f}x",
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test the Streaming Transcriber")
    parser.add_argument("--audio", type=str, default="../RealtimeSTT/warmup_audio.wav",
                        help="Path to an audio file for testing")
    parser.add_argument("--chunk-size", type=int, default=4000,
                        help="Size of audio chunks to process")
    parser.add_argument("--overlap", type=int, default=2000,
                        help="Overlap between processing segments")
    parser.add_argument("--buffer-size", type=int, default=16000,
                        help="Buffer size for the streaming transcriber")
    parser.add_argument("--any-lang", action="store_true",
                        help="Use language detection")
    parser.add_argument("--no-quick", action="store_true",
                        help="Disable quick (parallel) processing")
    parser.add_argument("--use-mlx-transcriber", action="store_true",
                        help="Use MLXTranscriber instead of direct StreamingTranscriber")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results as JSON")
    parser.add_argument("--plot", type=str, default=None,
                        help="Path to save results plot")
    
    args = parser.parse_args()
    
    audio_file = args.audio
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    logger.info("Starting streaming transcription test")
    
    if args.use_mlx_transcriber:
        results = test_mlx_streaming(
            audio_file,
            chunk_size=args.chunk_size,
            any_lang=args.any_lang,
            quick=not args.no_quick
        )
    else:
        results = test_direct_streaming(
            audio_file,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            buffer_size=args.buffer_size,
            any_lang=args.any_lang
        )
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Generate plot if requested
    if args.plot:
        plot_results(results, args.plot)

if __name__ == "__main__":
    main()