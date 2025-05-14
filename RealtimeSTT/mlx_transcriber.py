"""
MLX Transcriber Module

This module provides an interface for the MLX-optimized Whisper large-v3-turbo model.
It bridges the RealtimeSTT audio processing pipeline with the high-performance 
transcription capabilities of MLX on Apple Silicon hardware.

Author: Kristoffer Vatnehol
"""

import os
import queue
import threading
import time
import logging
import numpy as np
import mlx.core as mx
from functools import lru_cache
from huggingface_hub import snapshot_download
import json
from .whisper_turbo import Transcriber, StreamingTranscriber, log_mel_spectrogram, create_streaming_transcriber

# Configure logging
logger = logging.getLogger("realtimestt.mlx_transcriber")

class MLXTranscriber:
    """
    Provides an interface to the MLX-optimized Whisper large-v3-turbo model.
    
    This class serves as an adapter between RealtimeSTT's audio processing pipeline
    and the MLX-optimized Whisper model implementation. It manages transcription
    requests in a separate thread to avoid blocking the main audio processing loop.
    
    Attributes:
        model_path (str): Path or identifier for the Whisper model
        realtime_mode (bool): Whether to use streaming mode for real-time transcription
        device (str): Computation device to use (auto-detected for Apple Silicon)
        any_lang (bool): Whether to detect language automatically
        quick (bool): Whether to use parallel processing for fast transcription
        transcriber (Transcriber): The underlying MLX Whisper model
    """
    def __init__(self, 
                 model_path="openai/whisper-large-v3-turbo",
                 realtime_mode=True,
                 device="auto",
                 any_lang=False,
                 quick=True,
                 language=None):
        """
        Initialize the MLX transcriber.
        
        Args:
            model_path (str): Path or identifier for the Whisper model
            realtime_mode (bool): Whether to use streaming mode for real-time transcription
            device (str): Computation device to use (auto-detected for Apple Silicon)
            any_lang (bool): Whether to detect language automatically
            quick (bool): Whether to use parallel processing for fast transcription
            language (str, optional): Language code (e.g., 'en', 'fr', 'de', etc.)
        """
        self.model_path = model_path
        self.realtime_mode = realtime_mode
        self.device = device
        self.any_lang = any_lang
        self.quick = quick
        self.language_code = language if not any_lang else None
        
        # Queues for thread-safe data passing
        self.transcription_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Threading control
        self.running = False
        self.worker_thread = None
        
        # The underlying transcriber models (initialized lazily)
        self.transcriber = None  # Batch transcriber
        self.streaming_transcriber = None  # Streaming transcriber
        
        # Audio buffer for real-time processing
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Request ID tracking for streaming mode
        self.current_request_id = None
        
        logger.info(f"MLXTranscriber initialized with model: {model_path}, realtime_mode: {realtime_mode}")

    def _load_model(self):
        """
        Load the MLX-optimized Whisper model.
        
        This loads the model weights and prepares the transcriber for use.
        It's called lazily when the transcriber is first used.
        
        For realtime mode, it loads both the batch transcriber and the
        streaming transcriber. For batch mode, it only loads the batch transcriber.
        
        Returns:
            bool: True if model loading was successful, False otherwise
        """
        logger.info(f"Loading MLX-optimized Whisper model from {self.model_path}")
        try:
            # Download model if needed
            path_hf = snapshot_download(
                repo_id=self.model_path,
                allow_patterns=["config.json", "model.safetensors"]
            )
            
            # Load configuration
            with open(f'{path_hf}/config.json', 'r') as fp:
                cfg = json.load(fp)
            
            # Load weights
            weights = [(k.replace("embed_positions.weight", "positional_embedding"), 
                      v.swapaxes(1, 2) if ('conv' in k and v.ndim==3) else v) 
                      for k, v in mx.load(f'{path_hf}/model.safetensors').items()]
            
            # Initialize batch transcriber
            self.transcriber = Transcriber(cfg)
            self.transcriber.load_weights(weights, strict=False)
            self.transcriber.eval()
            mx.eval(self.transcriber)
            
            # Initialize streaming transcriber if in realtime mode
            if self.realtime_mode:
                logger.info("Creating streaming transcriber for real-time mode")
                self.streaming_transcriber = StreamingTranscriber(cfg, buffer_size=16000, overlap=2000)
                self.streaming_transcriber.load_weights(weights, strict=False)
                self.streaming_transcriber.eval()
                mx.eval(self.streaming_transcriber)
            
            logger.info("MLX transcriber model(s) loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load MLX transcriber model: {e}")
            return False

    def start(self):
        """
        Start the transcription worker thread.
        
        This initializes the worker thread that handles transcription requests
        without blocking the main thread.
        """
        if self.running:
            logger.warning("Transcription worker already running")
            return False
        
        # Create and start the worker thread
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._transcription_worker,
            name="MLXTranscriber_Worker",
            daemon=True
        )
        self.worker_thread.start()
        logger.info("Transcription worker thread started")
        return True

    def stop(self):
        """
        Stop the transcription worker thread.
        
        This signals the worker thread to terminate and waits for it to complete.
        """
        if not self.running:
            logger.warning("Transcription worker not running")
            return
        
        # Signal the worker to stop
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                logger.warning("Transcription worker thread did not terminate gracefully")
            else:
                logger.info("Transcription worker thread stopped")
            self.worker_thread = None

    def add_audio_chunk(self, audio_chunk, is_last=False, request_id=None):
        """
        Add an audio chunk to the transcription buffer.
        
        In real-time mode, this adds the chunk directly to the transcription queue
        for streaming processing. In batch mode, it accumulates chunks and only
        processes when is_last=True.
        
        Args:
            audio_chunk (numpy.ndarray): Audio data as a numpy array
            is_last (bool): Whether this is the last chunk in a sequence
            request_id (any): Optional identifier for the request
        """
        if not self.realtime_mode:
            # In batch mode, accumulate chunks and process only when complete
            with self.buffer_lock:
                self.audio_buffer.append(audio_chunk)
                
                # Only process when we have the full audio
                if is_last and self.audio_buffer:
                    # Concatenate all chunks
                    audio_data = np.concatenate(self.audio_buffer)
                    self.audio_buffer = []
                    self.transcribe(audio_data, request_id=request_id)
        else:
            # In streaming mode, submit each chunk directly for incremental processing
            self.transcription_queue.put({
                'audio': audio_chunk,
                'is_last': is_last,
                'request_id': request_id,
                'streaming': True
            })
            logger.debug(f"Submitted streaming audio chunk of size {len(audio_chunk)} for transcription")

    def transcribe(self, audio_data, request_id=None):
        """
        Submit audio data for batch transcription.
        
        Args:
            audio_data (numpy.ndarray): Audio data to transcribe
            request_id (any): Optional identifier for the request
        """
        # Make sure we have enough data for processing
        if len(audio_data) < 512:  # Minimum size for processing
            logger.debug("Audio chunk too small for transcription, skipping")
            return
        
        # Submit to the worker queue
        self.transcription_queue.put({
            'audio': audio_data,
            'request_id': request_id,
            'streaming': False
        })
        logger.debug(f"Submitted batch audio chunk of size {len(audio_data)} for transcription")

    def get_result(self, timeout=0.1):
        """
        Get a transcription result if available.
        
        Args:
            timeout (float): How long to wait for a result
            
        Returns:
            dict or None: Transcription result or None if no result is available
        """
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _transcription_worker(self):
        """
        Worker thread that processes transcription requests.
        
        This runs in a separate thread to avoid blocking the main thread
        during model inference.
        """
        logger.info("Transcription worker starting")
        
        # Load the model on first run
        if self.transcriber is None:
            if not self._load_model():
                logger.error("Failed to load transcription model, worker exiting")
                self.running = False
                return
        
        # Process transcription requests until stopped
        while self.running:
            try:
                # Get the next item to process
                try:
                    item = self.transcription_queue.get(timeout=0.1)
                    
                    # Check if this is a streaming or batch request
                    if isinstance(item, dict) and 'streaming' in item and item['streaming']:
                        # Streaming mode
                        audio_data = item['audio']
                        is_last = item.get('is_last', False)
                        request_id = item.get('request_id', None)
                        
                        # Only update current_request_id if different and this is a new request
                        if request_id is not None and (self.current_request_id != request_id):
                            logger.debug(f"New streaming request ID: {request_id}")
                            self.current_request_id = request_id
                            
                            # Reset streaming state when starting a new request
                            if self.streaming_transcriber is not None:
                                self.streaming_transcriber.reset()
                        
                        # Process with streaming transcriber
                        if self.streaming_transcriber is None:
                            logger.error("Streaming transcriber not initialized but received streaming request")
                            continue
                            
                        # Process the chunk
                        start_time = time.time()
                        result = self.streaming_transcriber.process_chunk(
                            audio_data, 
                            is_last=is_last,
                            language=None if self.any_lang else self.language_code
                        )
                        processing_time = time.time() - start_time
                        
                        # Add processing time and request ID to result
                        result['processing_time'] = processing_time
                        result['request_id'] = request_id
                        
                        # Put result in queue
                        self.results_queue.put(result)
                        
                        if result.get('new_text'):
                            logger.debug(f"Streaming transcription update in {processing_time:.2f}s: {result['new_text'][:30]}...")
                        
                    else:
                        # Batch mode
                        if isinstance(item, dict):
                            audio_data = item['audio']
                            request_id = item.get('request_id', None)
                        else:
                            # Handle legacy format for backward compatibility
                            audio_data = item
                            request_id = None
                        
                        # Process the audio with batch transcriber
                        start_time = time.time()
                        text = self._process_audio(audio_data)
                        processing_time = time.time() - start_time
                        
                        # Put the result in the results queue
                        if text:
                            self.results_queue.put({
                                'text': text,
                                'processing_time': processing_time,
                                'is_final': True,
                                'request_id': request_id
                            })
                            logger.debug(f"Batch transcription completed in {processing_time:.2f}s: {text[:30]}...")
                
                except queue.Empty:
                    continue
                
            except Exception as e:
                logger.exception(f"Error in transcription worker: {e}")
        
        logger.info("Transcription worker stopped")

    def _process_audio(self, audio_data):
        """
        Process audio data through the MLX Whisper model.
        
        Args:
            audio_data (numpy.ndarray or mx.array): Audio data to transcribe
            
        Returns:
            str: Transcribed text
        """
        try:
            # Convert NumPy array to MLX array if necessary
            if isinstance(audio_data, np.ndarray):
                # Ensure correct dtype and normalization
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Normalize if not already in [-1, 1] range
                max_val = np.max(np.abs(audio_data))
                if max_val > 0 and max_val > 1.0:
                    audio_data = audio_data / max_val
                
                # Convert to MLX array
                audio_data = mx.array(audio_data)
            elif not isinstance(audio_data, mx.array):
                # If not numpy or MLX array, attempt conversion
                audio_data = mx.array(audio_data)
            
            # Process with the transcriber model
            text = self.transcriber(
                audio_data, 
                self.any_lang, 
                self.quick,
                language=self.language_code
            )
            return text
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def cleanup(self):
        """
        Clean up resources and stop the transcriber.
        
        This ensures all threads are stopped, data is cleared,
        and memory is released.
        """
        self.stop()
        
        # Clear MLX cache to prevent memory leaks
        try:
            import mlx.core as mx
            mx.clear_all_device_arrays()
        except Exception as e:
            logger.debug(f"Failed to clear MLX device arrays: {e}")
            
        # Reset streaming state
        if self.streaming_transcriber:
            try:
                self.streaming_transcriber.reset()
            except:
                pass
        
        # Clear any remaining data
        with self.buffer_lock:
            self.audio_buffer = []
        
        # Clear queues
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except:
                pass
                
        while not self.results_queue.empty():
            try:
                self.results_queue.get_nowait()
            except:
                pass
        
        # Explicitly clear models to help with garbage collection
        self.transcriber = None
        self.streaming_transcriber = None
        self.current_request_id = None
        
        # Force garbage collection
        try:
            import gc
            gc.collect()
        except:
            pass
            
        logger.info("MLX transcriber cleaned up")