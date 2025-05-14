#!/usr/bin/env python3
"""
Benchmark the MLX-optimized Whisper model.

This script compares the performance of:
1. Direct whisper_turbo.py transcription
2. MLXTranscriber with batch mode
3. MLXTranscriber with streaming mode

It also compares the optimized NumPy array approach with the original
temporary file approach.
"""

import os
import sys
import time
import numpy as np
import logging
import argparse
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('benchmark_mlx')

def load_audio(file_path):
    """Load audio from file and prepare for processing."""
    try:
        import soundfile as sf
        audio, sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy import signal
            audio = signal.resample_poly(audio, 16000, sample_rate)
            logger.info(f"Resampled audio from {sample_rate}Hz to 16000Hz")
        
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        return audio, len(audio) / 16000.0  # Return audio and duration in seconds
    
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return None, 0.0

def benchmark_whisper_turbo_direct(audio_data, name):
    """Benchmark direct whisper_turbo.py transcription with NumPy array."""
    from RealtimeSTT.whisper_turbo import transcribe
    
    logger.info(f"[{name}] Testing direct whisper_turbo transcription with NumPy array...")
    
    start_time = time.time()
    text = transcribe(audio_data, quick=True)
    transcription_time = time.time() - start_time
    
    logger.info(f"[{name}] Transcription time: {transcription_time:.4f}s")
    logger.info(f"[{name}] First 100 chars: '{text[:100]}...'")
    
    return {
        "method": "whisper_turbo_direct",
        "transcription_time": transcription_time,
        "transcription_result_preview": text[:100],
        "full_text": text
    }

def benchmark_mlx_transcriber_batch(audio_data, name):
    """Benchmark MLXTranscriber in batch mode."""
    from RealtimeSTT.mlx_transcriber import MLXTranscriber
    
    logger.info(f"[{name}] Testing MLXTranscriber in batch mode...")
    
    transcriber = MLXTranscriber(realtime_mode=False, quick=True)
    transcriber.start()
    
    try:
        start_time = time.time()
        transcriber.transcribe(audio_data)
        
        result = None
        max_wait = 60  # seconds
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait:
            result = transcriber.get_result(timeout=0.5)
            if result:
                break
            logger.info(f"[{name}] Waiting for transcription result...")
        
        processing_time = time.time() - start_time
        
        if result:
            logger.info(f"[{name}] Total time: {processing_time:.4f}s")
            logger.info(f"[{name}] Model time: {result['processing_time']:.4f}s")
            logger.info(f"[{name}] First 100 chars: '{result['text'][:100]}...'")
            
            return {
                "method": "mlx_transcriber_batch",
                "total_time": processing_time,
                "model_time": result['processing_time'],
                "transcription_result_preview": result['text'][:100],
                "full_text": result['text']
            }
        else:
            logger.error(f"[{name}] No transcription result received within timeout")
            return {
                "method": "mlx_transcriber_batch",
                "error": "Timeout waiting for result"
            }
    
    finally:
        transcriber.cleanup()

def benchmark_mlx_transcriber_streaming(audio_data, name, chunk_size=4000):
    """
    Benchmark MLXTranscriber in streaming mode.
    
    This simulates real-time audio processing by feeding the audio in chunks.
    """
    from RealtimeSTT.mlx_transcriber import MLXTranscriber
    
    logger.info(f"[{name}] Testing MLXTranscriber in streaming mode (chunk_size={chunk_size})...")
    
    transcriber = MLXTranscriber(realtime_mode=True, quick=True)
    transcriber.start()
    
    chunks = []
    for i in range(0, len(audio_data), chunk_size):
        chunks.append(audio_data[i:i+chunk_size])
    
    logger.info(f"[{name}] Split audio into {len(chunks)} chunks")
    
    try:
        results = []
        start_time = time.time()
        
        # Process all chunks except the last one
        for i, chunk in enumerate(chunks[:-1]):
            logger.info(f"[{name}] Processing chunk {i+1}/{len(chunks)}")
            transcriber.add_audio_chunk(chunk, is_last=False)
            
            # Check for intermediate results (would be used in real-time application)
            result = transcriber.get_result(timeout=0.1)
            if result:
                results.append({
                    "chunk": i+1,
                    "time": time.time() - start_time,
                    "text": result['text'],
                    "is_final": False
                })
                logger.info(f"[{name}] Intermediate result after chunk {i+1}: '{result['text'][:50]}...'")
            
            # Simulate real-time processing delay
            time.sleep(chunk_size / 16000 * 0.1)  # Sleep for 10% of chunk duration
        
        # Process the last chunk
        logger.info(f"[{name}] Processing final chunk {len(chunks)}/{len(chunks)}")
        transcriber.add_audio_chunk(chunks[-1], is_last=True)
        
        # Wait for final result
        final_result = None
        max_wait = 60  # seconds
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait:
            result = transcriber.get_result(timeout=0.5)
            if result:
                if final_result is None or len(result['text']) > len(final_result['text']):
                    final_result = result
                    results.append({
                        "chunk": len(chunks),
                        "time": time.time() - start_time,
                        "text": result['text'],
                        "is_final": True
                    })
                    logger.info(f"[{name}] Final result: '{result['text'][:100]}...'")
            else:
                if final_result is not None:
                    break
                logger.info(f"[{name}] Waiting for final transcription result...")
        
        total_time = time.time() - start_time
        
        if final_result:
            logger.info(f"[{name}] Total streaming time: {total_time:.4f}s")
            
            return {
                "method": "mlx_transcriber_streaming",
                "total_time": total_time,
                "chunk_size": chunk_size,
                "num_chunks": len(chunks),
                "intermediate_results": len(results) - 1,  # Exclude final result
                "transcription_result_preview": final_result['text'][:100],
                "full_text": final_result['text'],
                "result_timeline": results
            }
        else:
            logger.error(f"[{name}] No final transcription result received")
            return {
                "method": "mlx_transcriber_streaming",
                "error": "No final result"
            }
    
    finally:
        transcriber.cleanup()

def run_benchmarks(audio_files, output_file=None):
    """Run all benchmarks on the provided audio files."""
    results = {}
    
    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        logger.info(f"Processing file: {file_name}")
        
        audio_data, duration = load_audio(file_path)
        if audio_data is None:
            logger.error(f"Skipping {file_name} due to loading error")
            continue
        
        file_results = {
            "file_name": file_name,
            "duration": duration,
            "sample_count": len(audio_data),
            "tests": []
        }
        
        # Run direct whisper_turbo test
        direct_result = benchmark_whisper_turbo_direct(audio_data, file_name)
        file_results["tests"].append(direct_result)
        
        # Run MLXTranscriber batch test
        batch_result = benchmark_mlx_transcriber_batch(audio_data, file_name)
        file_results["tests"].append(batch_result)
        
        # Run MLXTranscriber streaming tests with different chunk sizes
        for chunk_size in [4000, 8000, 16000]:
            streaming_result = benchmark_mlx_transcriber_streaming(
                audio_data, file_name, chunk_size=chunk_size
            )
            file_results["tests"].append(streaming_result)
        
        results[file_name] = file_results
    
    # Calculate summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "files_tested": len(audio_files),
        "realtime_factors": {}
    }
    
    for file_name, file_results in results.items():
        duration = file_results["duration"]
        
        for test in file_results["tests"]:
            method = test.get("method", "unknown")
            
            if "error" in test:
                continue
                
            if "total_time" in test:
                time_value = test["total_time"]
            elif "transcription_time" in test:
                time_value = test["transcription_time"]
            else:
                continue
                
            realtime_factor = time_value / duration
            
            if method not in summary["realtime_factors"]:
                summary["realtime_factors"][method] = []
                
            summary["realtime_factors"][method].append(realtime_factor)
    
    # Calculate averages
    for method, factors in summary["realtime_factors"].items():
        summary["realtime_factors"][method] = {
            "values": factors,
            "average": sum(factors) / len(factors),
            "min": min(factors),
            "max": max(factors)
        }
    
    results["summary"] = summary
    
    # Save results if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX-optimized Whisper transcription")
    parser.add_argument("--audio", "-a", type=str, nargs="+", required=True,
                      help="Path(s) to audio file(s) for benchmarking")
    parser.add_argument("--output", "-o", type=str, default=None,
                      help="Output file for benchmark results (JSON)")
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmarks(args.audio, args.output)
    
    # Print summary
    summary = results["summary"]
    logger.info("Benchmark Summary:")
    logger.info(f"Files tested: {summary['files_tested']}")
    logger.info("Realtime factors (lower is better):")
    
    for method, stats in summary["realtime_factors"].items():
        logger.info(f"  {method}: avg={stats['average']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")

if __name__ == "__main__":
    main()