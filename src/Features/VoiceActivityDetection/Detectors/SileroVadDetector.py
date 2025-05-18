"""
SileroVadDetector implementation of IVoiceActivityDetector.

This module provides an implementation of voice activity detection using the 
Silero VAD model, which offers higher accuracy compared to rule-based approaches.
"""

import logging
import os
import struct
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import torch

from src.Core.Common.Interfaces.voice_activity_detector import IVoiceActivityDetector


class SileroVadDetector(IVoiceActivityDetector):
    """
    Silero-based voice activity detector.
    
    This implementation uses the Silero VAD model, an ML-based approach that
    offers higher accuracy for speech detection. It is more computationally
    intensive than WebRTC VAD but provides more nuanced detection capabilities.
    
    The detector uses a pretrained model from the Torch Hub and provides
    confidence scores for detections.
    """
    
    DEFAULT_MODEL = "silero_vad"
    DEFAULT_MODEL_URL = "https://huggingface.co/snakers4/silero-vad/resolve/master/silero_vad.onnx"
    FALLBACK_MODEL_URL = "https://huggingface.co/onnx-community/silero-vad/resolve/main/silero_vad.onnx"
    
    def __init__(self, 
                 threshold: float = 0.5,
                 sample_rate: int = 16000,
                 window_size_samples: int = 1536,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 100,
                 use_onnx: bool = True):
        """
        Initialize the Silero VAD detector.
        
        Args:
            threshold: Speech probability threshold (0.0-1.0)
            sample_rate: Audio sample rate in Hz (must be 8000 or 16000)
            window_size_samples: Sliding window size in samples
            min_speech_duration_ms: Minimum speech segment duration in ms
            min_silence_duration_ms: Minimum silence segment duration in ms
            use_onnx: Whether to use ONNX model (faster) instead of PyTorch
        """
        self.logger = logging.getLogger(__name__)
        
        # Validate sample rate
        valid_sample_rates = [8000, 16000]
        if sample_rate not in valid_sample_rates:
            raise ValueError(f"Sample rate must be one of {valid_sample_rates}, got {sample_rate}")
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        self.model = None
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_size_samples = window_size_samples
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.use_onnx = use_onnx
        
        # RNN state for the ONNX model (128 is the state size used by Silero VAD)
        self.h_state = np.zeros((2, 1, 128), dtype=np.float32)
        self.c_state = np.zeros((2, 1, 128), dtype=np.float32)
        
        # Initialize speech/silence tracking
        self.reset_state()
        
        # Prepare cache directory for model
        self.cache_dir = os.path.expanduser("~/.cache/silero_vad")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def reset_state(self):
        """Reset internal state for speech tracking"""
        self.speech_probs = []
        self.vad_buffer = np.array([])
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.speech_start = 0
        self.speech_end = 0
        
        # Reset RNN states
        if hasattr(self, 'h_state'):
            self.h_state = np.zeros((2, 1, 128), dtype=np.float32)
        if hasattr(self, 'c_state'):
            self.c_state = np.zeros((2, 1, 128), dtype=np.float32)
    
    def setup(self) -> bool:
        """
        Initialize the Silero VAD model.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Delete the downloaded ONNX model if present to force redownload
            model_path = os.path.join(self.cache_dir, "silero_vad.onnx")
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    self.logger.info(f"Removed existing model file: {model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove existing model file: {e}")
            
            if self.use_onnx:
                self._setup_onnx_model()
            else:
                self._setup_torch_model()
            
            self.logger.info(f"Initialized Silero VAD with threshold {self.threshold}")
            self.reset_state()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Silero VAD: {e}")
            return False
    
    def _setup_torch_model(self):
        """Set up PyTorch Silero VAD model"""
        # Check if we have torch
        if not torch:
            raise ImportError("PyTorch not available. Install with 'pip install torch'")
            
        # Load model from torch hub
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                     model='silero_vad',
                                     force_reload=False,
                                     onnx=False)
        
        # Get preprocessing function
        (get_speech_timestamps, 
         _, 
         _, 
         _, 
         _) = utils
        
        self.model = model
        self.get_speech_timestamps = get_speech_timestamps
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _setup_onnx_model(self):
        """Set up ONNX Silero VAD model for faster inference"""
        try:
            import onnxruntime as ort
            
            model_path = os.path.join(self.cache_dir, "silero_vad.onnx")
            
            # Download model if not exists
            if not os.path.exists(model_path):
                self.logger.info("Downloading Silero VAD ONNX model...")
                import urllib.request
                try:
                    urllib.request.urlretrieve(self.DEFAULT_MODEL_URL, model_path)
                except Exception as e:
                    self.logger.warning(f"Failed to download from primary URL: {e}, trying fallback URL")
                    urllib.request.urlretrieve(self.FALLBACK_MODEL_URL, model_path)
            
            # Initialize ONNX runtime session
            self.ort_session = ort.InferenceSession(model_path)
            
            # Log model information for debugging
            input_names = [input.name for input in self.ort_session.get_inputs()]
            output_names = [output.name for output in self.ort_session.get_outputs()]
            self.logger.info(f"Model input names: {input_names}")
            self.logger.info(f"Model output names: {output_names}")
            
            # Check if model uses state, h0/c0, or another format
            self.model_format = 'unknown'
            if 'state' in input_names:
                self.model_format = 'state'
            elif 'h0' in input_names and 'c0' in input_names:
                self.model_format = 'h0_c0'
            
            self.logger.info(f"Detected model format: {self.model_format}")
            
            # We'll implement custom speech timestamps function when using ONNX
            self.logger.info("Successfully loaded Silero VAD ONNX model")
            
        except ImportError:
            self.logger.warning("ONNX Runtime not available. Falling back to PyTorch model")
            self.use_onnx = False
            self._setup_torch_model()
    
    def detect(self, audio_data: bytes, sample_rate: Optional[int] = None) -> bool:
        """
        Detect if the provided audio data contains speech.
        
        Args:
            audio_data: Raw audio data as bytes (must be 16-bit PCM)
            sample_rate: Sample rate of the audio data (optional, uses default if None)
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        is_speech, _ = self.detect_with_confidence(audio_data, sample_rate)
        return is_speech
    
    def detect_with_confidence(self, audio_data: bytes, 
                               sample_rate: Optional[int] = None) -> Tuple[bool, float]:
        """
        Detect if the provided audio data contains speech and return confidence level.
        
        Args:
            audio_data: Raw audio data as bytes (must be 16-bit PCM)
            sample_rate: Sample rate of the audio data (optional, uses default if None)
            
        Returns:
            Tuple[bool, float]: (speech_detected, confidence_score)
        """
        if self.model is None:
            if not self.setup():
                return False, 0.0
        
        # Use provided sample rate or default
        rate = sample_rate if sample_rate is not None else self.sample_rate
        
        # Validate sample rate
        valid_sample_rates = [8000, 16000]
        if rate not in valid_sample_rates:
            self.logger.warning(f"Invalid sample rate {rate}, using {self.sample_rate}")
            rate = self.sample_rate
        
        try:
            # Convert audio bytes to float tensor
            audio_array = self._bytes_to_audio_array(audio_data)
            
            # Detect speech in the current frame
            if self.use_onnx:
                speech_prob = self._predict_onnx(audio_array)
            else:
                speech_prob = self._predict_torch(audio_array)
            
            # Apply threshold
            is_speech = speech_prob >= self.threshold
            
            return is_speech, speech_prob
            
        except Exception as e:
            self.logger.error(f"Error in speech detection: {e}")
            return False, 0.0
    
    def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        # Convert bytes to numpy array assuming 16-bit PCM
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Normalize to [-1, 1]
        audio_array = audio_array / 32768.0
        
        return audio_array
    
    def _predict_torch(self, audio_array: np.ndarray) -> float:
        """Get speech probability using PyTorch model"""
        # Convert to tensor
        tensor = torch.from_numpy(audio_array).unsqueeze(0)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(tensor, self.sample_rate).item()
        
        return speech_prob
    
    def _predict_onnx(self, audio_array: np.ndarray) -> float:
        """Get speech probability using ONNX model"""
        # Ensure proper shape for ONNX input
        tensor = audio_array.reshape(1, -1)
        
        try:
            if hasattr(self, 'model_format'):
                # Create inputs based on detected model format
                if self.model_format == 'state':
                    ort_inputs = {
                        'input': tensor.astype(np.float32),
                        'sr': np.array([self.sample_rate], dtype=np.int64),
                        'state': np.zeros((4, 1, 128), dtype=np.float32)  # Initial state
                    }
                elif self.model_format == 'h0_c0':
                    ort_inputs = {
                        'input': tensor.astype(np.float32),
                        'sr': np.array([self.sample_rate], dtype=np.int64),
                        'h0': self.h_state,
                        'c0': self.c_state
                    }
                else:  # Use basic input format as fallback
                    ort_inputs = {
                        'input': tensor.astype(np.float32),
                        'sr': np.array([self.sample_rate], dtype=np.int64)
                    }
            else:  # Use basic input format as fallback
                ort_inputs = {
                    'input': tensor.astype(np.float32),
                    'sr': np.array([self.sample_rate], dtype=np.int64)
                }
            
            # Get output names
            output_names = [output.name for output in self.ort_session.get_outputs()]
            
            # Run ONNX inference
            ort_outs = self.ort_session.run(output_names, ort_inputs)
            
            # Get speech probability from first output
            speech_prob = ort_outs[0][0].item()
            
            # Update state if available
            if hasattr(self, 'model_format') and len(ort_outs) > 1:
                if self.model_format == 'state' and output_names[1] == 'state':
                    state_out = ort_outs[1]
                    if state_out.shape[0] == 4:  # Expected shape for state
                        # Update our state with the model's output state
                        pass  # Not updating for now to avoid further issues
                elif self.model_format == 'h0_c0' and len(output_names) > 2:
                    if 'hn' in output_names and 'cn' in output_names:
                        hn_idx = output_names.index('hn')
                        cn_idx = output_names.index('cn')
                        # Update our state with the model's output state
                        if hn_idx < len(ort_outs) and cn_idx < len(ort_outs):
                            self.h_state = ort_outs[hn_idx]
                            self.c_state = ort_outs[cn_idx]
            
        except Exception as e:
            self.logger.error(f"Failed to run ONNX inference: {e}")
            # Fallback to a default value
            speech_prob = 0.0
        
        return speech_prob
    
    def get_speech_timestamps(self, audio_data: bytes, 
                              return_seconds: bool = False) -> List[Dict[str, Union[int, float]]]:
        """
        Get timestamps of speech segments in the audio.
        
        Args:
            audio_data: Raw audio data as bytes (must be 16-bit PCM)
            return_seconds: If True, return timestamps in seconds, otherwise in samples
            
        Returns:
            List of dicts with start and end timestamps of speech segments
        """
        if self.model is None:
            if not self.setup():
                return []
        
        try:
            # Convert audio bytes to float tensor
            audio_array = self._bytes_to_audio_array(audio_data)
            
            if self.use_onnx:
                # We'll implement our custom timestamp function for ONNX
                # Since we can't use the torch utilities directly
                return self._get_speech_timestamps_onnx(
                    audio_array, 
                    self.threshold,
                    self.window_size_samples,
                    self.min_silence_duration_ms,
                    self.min_speech_duration_ms,
                    return_seconds
                )
            else:
                # Use torch hub provided functions
                timestamps = self.get_speech_timestamps(
                    audio_array, 
                    self.model,
                    threshold=self.threshold,
                    sampling_rate=self.sample_rate,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    window_size_samples=self.window_size_samples,
                    return_seconds=return_seconds
                )
                
                return timestamps
                
        except Exception as e:
            self.logger.error(f"Error getting speech timestamps: {e}")
            return []
    
    def _get_speech_timestamps_onnx(self, audio_array: np.ndarray, 
                                    threshold: float,
                                    window_size_samples: int,
                                    min_silence_duration_ms: int,
                                    min_speech_duration_ms: int,
                                    return_seconds: bool) -> List[Dict[str, Union[int, float]]]:
        """Custom implementation of speech timestamp detection for ONNX model"""
        # Implementation specific to ONNX model
        # This is a simplified version of the PyTorch implementation
        
        # Process audio in chunks of window_size_samples
        num_samples = len(audio_array)
        timestamps = []
        
        min_silence_samples = int(min_silence_duration_ms * self.sample_rate / 1000)
        min_speech_samples = int(min_speech_duration_ms * self.sample_rate / 1000)
        
        # Process audio in windows
        speech_start = None
        in_speech = False
        silence_counter = 0
        
        for i in range(0, num_samples, window_size_samples):
            chunk = audio_array[i:i + window_size_samples]
            
            # Skip if chunk is too small
            if len(chunk) < window_size_samples:
                if len(chunk) < window_size_samples // 2:
                    break
                # Pad with zeros if needed
                pad_size = window_size_samples - len(chunk)
                chunk = np.pad(chunk, (0, pad_size), 'constant')
            
            # Get speech probability for this chunk
            ort_inputs = {
                'input': chunk.reshape(1, -1).astype(np.float32),
                'sr': np.array([self.sample_rate], dtype=np.int64)
            }
            
            ort_outs = self.ort_session.run(None, ort_inputs)
            speech_prob = ort_outs[0][0].item()
            
            # Apply threshold logic
            if not in_speech and speech_prob >= threshold:
                in_speech = True
                speech_start = i
                silence_counter = 0
            elif in_speech:
                if speech_prob >= threshold:
                    silence_counter = 0
                else:
                    silence_counter += window_size_samples
                    
                    if silence_counter >= min_silence_samples:
                        # End of speech detected
                        in_speech = False
                        
                        # Only add if speech segment is long enough
                        speech_end = i
                        if speech_end - speech_start >= min_speech_samples:
                            if return_seconds:
                                timestamps.append({
                                    'start': speech_start / self.sample_rate,
                                    'end': speech_end / self.sample_rate
                                })
                            else:
                                timestamps.append({
                                    'start': speech_start,
                                    'end': speech_end
                                })
                        
                        speech_start = None
        
        # Handle case where speech continues until the end
        if in_speech and speech_start is not None:
            speech_end = num_samples
            if speech_end - speech_start >= min_speech_samples:
                if return_seconds:
                    timestamps.append({
                        'start': speech_start / self.sample_rate,
                        'end': speech_end / self.sample_rate
                    })
                else:
                    timestamps.append({
                        'start': speech_start,
                        'end': speech_end
                    })
        
        return timestamps
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the voice activity detector with the provided parameters.
        
        Args:
            config: Dictionary of configuration parameters, including:
                - threshold: Speech probability threshold
                - sample_rate: Audio sample rate in Hz
                - window_size_samples: Window size in samples
                - min_speech_duration_ms: Minimum speech segment duration
                - min_silence_duration_ms: Minimum silence segment duration
            
        Returns:
            bool: True if configuration was successful, False otherwise
        """
        try:
            # Handle threshold
            if 'threshold' in config:
                threshold = config['threshold']
                if not 0.0 <= threshold <= 1.0:
                    self.logger.warning(f"Invalid threshold {threshold}, must be between 0.0 and 1.0")
                else:
                    self.threshold = threshold
            
            # Handle sample rate
            if 'sample_rate' in config:
                sample_rate = config['sample_rate']
                valid_sample_rates = [8000, 16000]
                if sample_rate not in valid_sample_rates:
                    self.logger.warning(f"Invalid sample rate {sample_rate}, must be one of {valid_sample_rates}")
                else:
                    self.sample_rate = sample_rate
            
            # Handle window size
            if 'window_size_samples' in config:
                window_size = config['window_size_samples']
                if window_size < 512:
                    self.logger.warning(f"Window size {window_size} is too small, minimum is 512")
                else:
                    self.window_size_samples = window_size
            
            # Handle min speech duration
            if 'min_speech_duration_ms' in config:
                min_speech = config['min_speech_duration_ms']
                if min_speech < 0:
                    self.logger.warning(f"Invalid min speech duration {min_speech}, must be >= 0")
                else:
                    self.min_speech_duration_ms = min_speech
            
            # Handle min silence duration
            if 'min_silence_duration_ms' in config:
                min_silence = config['min_silence_duration_ms']
                if min_silence < 0:
                    self.logger.warning(f"Invalid min silence duration {min_silence}, must be >= 0")
                else:
                    self.min_silence_duration_ms = min_silence
            
            # Reset state with new config
            self.reset_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring Silero VAD: {e}")
            return False
    
    def reset(self) -> None:
        """
        Reset the internal state of the voice activity detector.
        """
        self.reset_state()
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get the current configuration of the voice activity detector.
        
        Returns:
            Dict[str, Any]: Dictionary of current configuration parameters
        """
        return {
            'threshold': self.threshold,
            'sample_rate': self.sample_rate,
            'window_size_samples': self.window_size_samples,
            'min_speech_duration_ms': self.min_speech_duration_ms,
            'min_silence_duration_ms': self.min_silence_duration_ms,
            'detector_type': self.get_name(),
            'use_onnx': self.use_onnx
        }
    
    def get_name(self) -> str:
        """
        Get the name of the voice activity detector implementation.
        
        Returns:
            str: Name of the detector
        """
        return "Silero VAD"
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the voice activity detector.
        """
        self.model = None
        if hasattr(self, 'ort_session'):
            delattr(self, 'ort_session')
        self.reset_state()