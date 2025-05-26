"""
OpenAITranscriptionEngine implementation for real-time speech-to-text transcription.

This module implements the ITranscriptionEngine interface using OpenAI's gpt-4o-transcribe
model for cloud-based speech-to-text conversion.
"""

import os
import tempfile
import threading
import time
from queue import Queue, Empty
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import requests
import soundfile as sf

# Infrastructure imports
from src.Infrastructure.Logging import LoggingModule

from src.Core.Common.Interfaces.transcription_engine import ITranscriptionEngine


class OpenAITranscriptionEngine(ITranscriptionEngine):
    """OpenAI GPT-4o-transcribe implementation of the transcription engine interface."""
    
    def __init__(self, 
                 model_name="gpt-4o-transcribe", 
                 language=None, 
                 api_key=None,
                 streaming=True, 
                 **kwargs):
        """
        Initialize the OpenAI transcription engine.
        
        Args:
            model_name: The OpenAI model to use for transcription (default: "gpt-4o-transcribe")
            language: Language code or None for auto-detection
            api_key: OpenAI API key (will fall back to env var if None)
            streaming: Whether to use streaming mode
            **kwargs: Additional configuration options
        """
        self.logger = LoggingModule.get_logger(__name__)
        
        # Configuration
        self.model_name = model_name
        self.language = language
        self.streaming = streaming
        self.sample_rate = kwargs.get('sample_rate', 16000)
        
        # API key handling
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = os.environ.get('OPENAI_API_KEY')
            if self.api_key is None:
                self.logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Audio buffer for streaming
        self.audio_buffer = []
        
        # Thread safety
        self.lock = threading.RLock()
        self.result_queue = Queue()
        self.current_result = None
        self.result_ready = threading.Event()
        self.is_processing = False
        
        # State tracking
        self._running = False
        self._client = None
        self._websocket = None
        
        self.logger.info(f"Initialized OpenAITranscriptionEngine with model={model_name}, language={language}")
    
    def start(self) -> bool:
        """
        Initialize and verify API connectivity.
        
        Returns:
            bool: True if the engine started successfully
        """
        with self.lock:
            try:
                self.logger.info(f"Starting OpenAITranscriptionEngine with model={self.model_name}")
                
                # Check for API key
                if not self.api_key:
                    self.logger.error("Failed to start OpenAITranscriptionEngine: No API key provided")
                    return False
                
                # Test API connectivity
                response = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.logger.info("Successfully connected to OpenAI API")
                    self._running = True
                    
                    # Import here to delay until necessary
                    try:
                        import openai
                        self._client = openai.OpenAI(api_key=self.api_key)
                    except ImportError:
                        self.logger.error("Failed to import openai package. Install with 'uv pip install openai'")
                        self._running = False
                        return False
                    
                    return True
                else:
                    self.logger.error(f"Failed to connect to OpenAI API: {response.status_code}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error starting OpenAITranscriptionEngine: {e}", exc_info=True)
                return False
    
    def transcribe(self, audio: np.ndarray) -> None:
        """
        Transcribe complete audio segment using OpenAI API.
        
        Args:
            audio: Audio data as numpy array
        """
        with self.lock:
            if not self.is_running():
                self.logger.warning("Cannot transcribe - engine not running")
                return
            
            self.is_processing = True
            self.result_ready.clear()
            
            # Process in a separate thread to avoid blocking
            processing_thread = threading.Thread(
                target=self._process_audio,
                args=(audio, True),
                daemon=True
            )
            processing_thread.start()
    
    def add_audio_chunk(self, audio_chunk: np.ndarray, is_last: bool = False) -> None:
        """
        Add an audio chunk for streaming transcription.
        
        Args:
            audio_chunk: Audio data chunk as numpy array
            is_last: Whether this is the last chunk in the stream
        """
        with self.lock:
            if not self.is_running():
                self.logger.warning("Cannot add audio chunk - engine not running")
                return
            
            # Add to buffer
            self.audio_buffer.append(audio_chunk)
            
            # Process if it's the last chunk or we have enough accumulated audio
            # Adjust chunk accumulation based on your use case
            if is_last or len(self.audio_buffer) * len(audio_chunk) >= self.sample_rate * 2:  # 2 seconds of audio
                audio_data = np.concatenate(self.audio_buffer)
                self.audio_buffer = []  # Clear buffer after processing
                
                self.is_processing = True
                self.result_ready.clear()
                
                # Process in a separate thread
                processing_thread = threading.Thread(
                    target=self._process_audio,
                    args=(audio_data, is_last),
                    daemon=True
                )
                processing_thread.start()
    
    def _process_audio(self, audio_data, is_final=False):
        """
        Process audio with appropriate API call based on mode.
        
        Args:
            audio_data: Audio data to transcribe
            is_final: Whether this is the final chunk of audio
        """
        try:
            # Ensure we have a valid API client
            if not self._client:
                self.logger.error("OpenAI client not initialized")
                self._handle_error("OpenAI client not initialized", is_final)
                return
            
            # Check for empty or invalid audio input
            if isinstance(audio_data, np.ndarray) and (audio_data.size == 0 or np.all(audio_data == 0)):
                self.logger.warning("Empty or silent audio chunk received, returning empty result")
                self._handle_empty_result(is_final)
                return
                
            start_time = time.time()
            
            # Log audio information
            if isinstance(audio_data, np.ndarray):
                duration_seconds = len(audio_data) / self.sample_rate if self.sample_rate > 0 else 0
                
                self.logger.info(f"Processing audio array: shape={audio_data.shape}, samples={len(audio_data)}, "
                               f"duration={duration_seconds:.2f}s")
                
                # Save a copy of the audio chunk for debugging
                try:
                    debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    timestamp = int(time.time())
                    debug_path = os.path.join(debug_dir, f"openai_{timestamp}.wav")
                    sf.write(debug_path, audio_data, self.sample_rate)
                    self.logger.info(f"DEBUGGING: Saved audio chunk to: {debug_path}")
                except Exception as e:
                    self.logger.error(f"Error saving debug audio chunk: {e}")
            
            # Prepare audio (ensure proper sample rate, format, etc.)
            processed_audio = self._prepare_audio(audio_data)
            
            # Use appropriate method based on streaming mode
            if self.streaming and not is_final:
                # WebSocket streaming is complex and requires setting up a persistent connection
                # For simplicity, using standard transcription for now even in streaming mode
                result = self._handle_standard_transcription(processed_audio)
            else:
                # Use standard API
                result = self._handle_standard_transcription(processed_audio)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Format result
            transcription_result = {
                "text": result,
                "is_final": is_final,
                "language": self.language,
                "processing_time": processing_time,
                "confidence": 1.0,  # OpenAI doesn't provide confidence scores
                "success": True
            }
            
            # Update current result and signal completion
            with self.lock:
                self.current_result = transcription_result
                self.is_processing = False
                self.result_ready.set()
                self.result_queue.put(transcription_result)
            
            self.logger.info(f"Processed audio in {processing_time:.2f}s, text length: {len(result)}")
                
        except Exception as e:
            self.logger.error(f"Error processing audio with OpenAI: {e}", exc_info=True)
            self._handle_error(str(e), is_final)
    
    def _handle_standard_transcription(self, audio_data):
        """
        Handle standard transcription via REST API.
        
        Args:
            audio_data: Audio data to transcribe or path to audio file
            
        Returns:
            str: Transcribed text
        """
        import tempfile
        import os
        
        try:
            # If audio_data is a file path, use it directly
            if isinstance(audio_data, str):
                self.logger.info(f"Using existing audio file: {audio_data}, language: {self.language}")
                
                # Open the file for sending to the API
                with open(audio_data, "rb") as audio_file:
                    # Make the API call
                    response = self._client.audio.transcriptions.create(
                        model=self.model_name,
                        file=audio_file,
                        response_format="text",
                        language=self.language
                    )
                    
                    # Handle the response
                    if hasattr(response, 'text'):
                        return response.text
                    else:
                        return str(response)
            
            # Otherwise, it's numpy array data that needs to be written to a file
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                    # Write the audio data to a temporary file
                    sf.write(temp_file.name, audio_data, self.sample_rate, format='WAV', subtype='PCM_16')
                    temp_file.flush()
                    
                    # Open the file for sending to the API
                    with open(temp_file.name, "rb") as audio_file:
                        # Make the API call
                        response = self._client.audio.transcriptions.create(
                            model=self.model_name,
                            file=audio_file,
                            response_format="text",
                            language=self.language
                        )
                        
                        # Handle the response
                        if hasattr(response, 'text'):
                            return response.text
                        else:
                            return str(response)
                
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            raise
    
    def _handle_websocket_streaming(self, audio_data):
        """
        Handle streaming audio via WebSocket API.
        Note: This is a placeholder. Full WebSocket implementation would be more complex.
        
        Args:
            audio_data: Audio data chunk
            
        Returns:
            str: Partial transcription
        """
        self.logger.warning("WebSocket streaming not fully implemented yet, using standard API")
        return self._handle_standard_transcription(audio_data)
    
    def _prepare_audio(self, audio_data):
        """
        Prepare audio for the OpenAI API.
        
        Args:
            audio_data: Raw audio data (numpy array) or file path (str)
            
        Returns:
            np.ndarray or str: Properly formatted audio data or file path
        """
        # If audio_data is a string (file path), return it as is
        if isinstance(audio_data, str):
            self.logger.info(f"Using file path directly: {audio_data}")
            return audio_data
            
        # Handle numpy arrays
        if isinstance(audio_data, np.ndarray):
            # Convert sample rate if needed
            if self.sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=self.sample_rate,
                    target_sr=16000
                )
            
            # Ensure proper format and normalization
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Normalize to [-1, 1] range if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 0 and max_val > 1.0:
                audio_data = audio_data / max_val
                
            return audio_data
        
        # Handle unsupported types
        raise TypeError(f"Unsupported audio data type: {type(audio_data).__name__}")
    
    def _handle_empty_result(self, is_final):
        """
        Handle empty or silent audio with an empty result.
        
        Args:
            is_final: Whether this is the final audio chunk
        """
        with self.lock:
            empty_result = {
                "text": "",
                "is_final": is_final,
                "language": self.language,
                "processing_time": 0.0,
                "confidence": 0.0,
                "success": True,
                "info": "Empty or silent audio"
            }
            self.current_result = empty_result
            self.is_processing = False
            self.result_ready.set()
            self.result_queue.put(empty_result)
    
    def _handle_error(self, error_message, is_final):
        """
        Handle processing errors with an error result.
        
        Args:
            error_message: Error message
            is_final: Whether this is the final audio chunk
        """
        with self.lock:
            error_result = {
                "error": error_message,
                "is_final": is_final,
                "text": "",
                "success": False
            }
            self.current_result = error_result
            self.is_processing = False
            self.result_ready.set()
            self.result_queue.put(error_result)
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get the transcription result (blocking with timeout).
        
        Args:
            timeout: Maximum time to wait for a result in seconds
            
        Returns:
            Optional[Dict[str, Any]]: Transcription result or None if not available
        """
        try:
            # Wait for result
            if self.is_processing:
                if not self.result_ready.wait(timeout):
                    return None
            
            # Get result from queue if available
            try:
                return self.result_queue.get(block=False)
            except Empty:
                return self.current_result
                
        except Exception as e:
            self.logger.error(f"Error getting result: {e}")
            return {"error": str(e)}
    
    def cleanup(self) -> None:
        """Release resources used by the transcription engine."""
        with self.lock:
            self.logger.info("Cleaning up OpenAITranscriptionEngine")
            
            # Close WebSocket connection if it exists
            if self._websocket:
                try:
                    self._websocket.close()
                except:
                    pass
                self._websocket = None
            
            # Clear references
            self._client = None
            self.audio_buffer = []
            
            # Clear result tracking
            self.current_result = None
            self.result_ready.clear()
            self._running = False
            
            # Attempt to force garbage collection
            import gc
            gc.collect()
    
    def is_running(self) -> bool:
        """
        Check if the transcription engine is currently running.
        
        Returns:
            bool: True if the engine is running
        """
        return self._running and self._client is not None
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Update the engine configuration.
        
        Args:
            config: New configuration settings
            
        Returns:
            bool: True if the configuration was updated successfully
        """
        with self.lock:
            try:
                # Update language if specified
                if 'language' in config:
                    self.language = config['language']
                
                # Update model if specified
                if 'model_name' in config:
                    self.model_name = config['model_name']
                
                # Update API key if specified
                if 'api_key' in config:
                    self.api_key = config['api_key']
                    # Reinitialize client with new API key
                    if self._client and self.api_key:
                        import openai
                        self._client = openai.OpenAI(api_key=self.api_key)
                
                # Update streaming mode
                if 'streaming' in config:
                    self.streaming = config['streaming']
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error configuring engine: {e}")
                return False