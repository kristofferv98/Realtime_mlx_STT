"""
SileroVadDetector implementation of IVoiceActivityDetector.

This module provides an implementation of voice activity detection using the 
Silero VAD model, which offers higher accuracy compared to rule-based approaches.
"""

import os
import struct
import shutil 
import time
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import torch

from src.Core.Common.Interfaces.voice_activity_detector import IVoiceActivityDetector
from src.Infrastructure.Logging import LoggingModule


class SileroVadDetector(IVoiceActivityDetector):
    """
    Silero-based voice activity detector.
    
    This implementation uses the Silero VAD model, an ML-based approach that
    offers higher accuracy for speech detection. It is more computationally
    intensive than WebRTC VAD but provides more nuanced detection capabilities.
    
    The detector uses a pretrained model from the Torch Hub and provides
    confidence scores for detections.
    """
    
    # Silero VAD model information
    DEFAULT_MODEL = "silero_vad"
    TORCH_REPO = "snakers4/silero-vad"  # Official Silero VAD repository
    
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
        self.logger = LoggingModule.get_logger(__name__)
        
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
        
        # Reset RNN states for all possible model formats
        if hasattr(self, 'h_state'):
            self.h_state = np.zeros((2, 1, 128), dtype=np.float32)
        if hasattr(self, 'c_state'):
            self.c_state = np.zeros((2, 1, 128), dtype=np.float32)
        if hasattr(self, 'state'):
            self.state = np.zeros((4, 1, 128), dtype=np.float32)
        
        # Reset model format information if initialization failed
        if not hasattr(self, 'model_format'):
            self.model_format = 'unknown'
            self.logger.info("Model format not detected yet, will be determined during first inference")
            
        # Log reset for debugging
        self.logger.debug("Reset VAD internal state")
    
    def setup(self) -> bool:
        """
        Initialize the Silero VAD model.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Check for required directories
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Model path
            model_path = os.path.join(self.cache_dir, "silero_vad.onnx")
            
            # Check if we should use existing model if present
            use_existing = True
            if os.path.exists(model_path):
                if os.path.getsize(model_path) < 10000:  # File too small, likely corrupted
                    self.logger.warning(f"Existing model file too small ({os.path.getsize(model_path)} bytes), forcing redownload")
                    use_existing = False
                else:
                    self.logger.info(f"Existing model file found: {model_path} ({os.path.getsize(model_path)} bytes)")
            
            # Force redownload if needed
            if not use_existing and os.path.exists(model_path):
                try:
                    # Backup the model before removing it
                    backup_name = f"silero_vad.onnx.backup.{int(time.time())}"
                    backup_path = os.path.join(self.cache_dir, backup_name)
                    shutil.copy2(model_path, backup_path)
                    self.logger.info(f"Backed up existing model to {backup_path}")
                    
                    # Remove the existing model
                    os.remove(model_path)
                    self.logger.info(f"Removed existing model file: {model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to backup/remove existing model file: {e}")
            
            # Setup the appropriate model implementation
            if self.use_onnx:
                try:
                    import onnxruntime
                    self.logger.info(f"Using ONNX Runtime version: {onnxruntime.__version__}")
                    self._setup_onnx_model()
                except ImportError as e:
                    self.logger.warning(f"ONNX Runtime not available ({e}), falling back to PyTorch model")
                    self.use_onnx = False
                    self._setup_torch_model()
            else:
                try:
                    self.logger.info(f"Using PyTorch version: {torch.__version__}")
                    self._setup_torch_model()
                except Exception as e:
                    self.logger.error(f"Failed to setup PyTorch model: {e}")
                    # Try ONNX as fallback if PyTorch fails
                    try:
                        self.logger.info("Trying ONNX model as fallback")
                        self.use_onnx = True
                        self._setup_onnx_model()
                    except Exception as e2:
                        self.logger.error(f"Failed to setup ONNX model as fallback: {e2}")
                        return False
            
            # Reset state and return success
            self.logger.info(f"Successfully initialized Silero VAD (using {'ONNX' if self.use_onnx else 'PyTorch'}) with threshold {self.threshold}")
            self.reset_state()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Silero VAD: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _setup_torch_model(self):
        """Set up PyTorch Silero VAD model"""
        # Check if we have torch
        if not torch:
            raise ImportError("PyTorch not available. Install with 'pip install torch'")
        
        # Create cache directory
        torch_hub_dir = os.path.join(self.cache_dir, "torch_hub")
        os.makedirs(torch_hub_dir, exist_ok=True)
        
        # Suppress all stdout during model loading to avoid "Using cache found" messages
        import io
        import contextlib
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            torch.hub.set_dir(torch_hub_dir)
            
            self.logger.info(f"Loading Silero VAD model from PyTorch Hub...")
            
            # Load model from torch hub using the same approach as KoljaB/RealtimeSTT
            model, utils = torch.hub.load(
                repo_or_dir=self.TORCH_REPO,
                model=self.DEFAULT_MODEL,
                force_reload=False,
                onnx=False,
                trust_repo=True,
                verbose=False
            )
        finally:
            sys.stdout = old_stdout
        
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
        
        self.logger.info("Successfully loaded Silero VAD PyTorch model")
    
    def _setup_onnx_model(self):
        """Set up ONNX Silero VAD model for faster inference"""
        try:
            import onnxruntime as ort
            
            # Create model directory with Parents if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            torch_hub_dir = os.path.join(self.cache_dir, "torch_hub")
            os.makedirs(torch_hub_dir, exist_ok=True)
            
            # Suppress all stdout during model loading to avoid "Using cache found" messages
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                torch.hub.set_dir(torch_hub_dir)
                
                # Load model directly from PyTorch hub
                self.logger.info("Loading Silero VAD model from PyTorch Hub...")
                repo_dir = os.path.join(torch_hub_dir, "snakers4_silero-vad_master")
                
                # If not already downloaded, get it from hub with ONNX=True
                if not os.path.exists(repo_dir):
                    self.logger.info("Downloading model from PyTorch Hub...")
                    model, utils = torch.hub.load(
                        repo_or_dir=self.TORCH_REPO,
                        model=self.DEFAULT_MODEL,
                        verbose=False,
                        onnx=True,
                        trust_repo=True
                    )
            finally:
                sys.stdout = old_stdout
            
            # Find ONNX model path in hub directory
            expected_model_paths = [
                os.path.join(repo_dir, "files", "silero_vad.onnx"),
                os.path.join(repo_dir, "silero_vad.onnx"),
                os.path.join(torch_hub_dir, "silero_vad.onnx")
            ]
            
            model_path = None
            for path in expected_model_paths:
                if os.path.exists(path) and os.path.getsize(path) > 100000:
                    model_path = path
                    self.logger.info(f"Found ONNX model at: {model_path}")
                    break
            
            # If no model found, try to download directly or export from PyTorch
            if not model_path:
                self.logger.warning("ONNX model not found in expected locations")
                self.logger.warning("Falling back to PyTorch model")
                self.use_onnx = False
                self._setup_torch_model()
                return
            
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
            self.logger.info("Successfully loaded Silero VAD ONNX model")
            
        except ImportError as e:
            self.logger.warning(f"ONNX Runtime not available ({e}). Falling back to PyTorch model")
            self.use_onnx = False
            self._setup_torch_model()
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            self.logger.warning("Falling back to PyTorch model")
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
        
        # Silero VAD expects audio data in chunks of specific sizes:
        # - 256 samples for 8kHz audio
        # - 512 samples for 16kHz audio
        expected_samples = 512 if self.sample_rate == 16000 else 256
        
        # If exact size, return as is
        if len(audio_array) == expected_samples:
            return audio_array
        
        # If too large, truncate to the expected size
        if len(audio_array) > expected_samples:
            self.logger.debug(f"Truncating audio from {len(audio_array)} to {expected_samples} samples")
            return audio_array[:expected_samples]
        
        # If too small, pad with zeros
        if len(audio_array) < expected_samples:
            self.logger.debug(f"Padding audio from {len(audio_array)} to {expected_samples} samples")
            padded = np.zeros(expected_samples, dtype=np.float32)
            padded[:len(audio_array)] = audio_array
            return padded
    
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
            # Get input and output names from the model
            input_names = [input.name for input in self.ort_session.get_inputs()]
            output_names = [output.name for output in self.ort_session.get_outputs()]
            
            # Create inputs dict based on the required inputs for the model
            ort_inputs = {}
            
            # Common inputs for all model formats
            if 'input' in input_names:
                ort_inputs['input'] = tensor.astype(np.float32)
            
            if 'sr' in input_names:
                ort_inputs['sr'] = np.array([self.sample_rate], dtype=np.int64)
            
            # Handle different model formats
            if hasattr(self, 'model_format'):
                if self.model_format == 'state' and 'state' in input_names:
                    # Some models expect a single state tensor with shape (4, 1, 128)
                    state_shape = (4, 1, 128)  # Standard shape for Silero VAD state
                    
                    # Initialize state if not already done
                    if not hasattr(self, 'state') or self.state is None:
                        self.state = np.zeros(state_shape, dtype=np.float32)
                    
                    ort_inputs['state'] = self.state
                
                elif self.model_format == 'h0_c0':
                    # Initialize states if not already done
                    if not hasattr(self, 'h_state') or self.h_state is None:
                        self.h_state = np.zeros((2, 1, 128), dtype=np.float32)
                    
                    if not hasattr(self, 'c_state') or self.c_state is None:
                        self.c_state = np.zeros((2, 1, 128), dtype=np.float32)
                    
                    # Add states to inputs
                    if 'h0' in input_names:
                        ort_inputs['h0'] = self.h_state
                    
                    if 'c0' in input_names:
                        ort_inputs['c0'] = self.c_state
            
            # Run inference with the prepared inputs
            ort_outs = self.ort_session.run(None, ort_inputs)
            
            # Get speech probability from first output (probability should always be first)
            if len(ort_outs) > 0 and ort_outs[0] is not None:
                speech_prob = float(ort_outs[0].item() if ort_outs[0].size == 1 else ort_outs[0][0].item())
            else:
                self.logger.warning("No valid output from ONNX inference")
                speech_prob = 0.0
            
            # Update state if available in outputs
            if hasattr(self, 'model_format') and len(ort_outs) > 1:
                output_dict = {name: ort_outs[i] for i, name in enumerate(output_names) if i < len(ort_outs)}
                
                if self.model_format == 'state' and 'state' in output_dict:
                    state_out = output_dict['state']
                    # Update state if shape matches our expected state shape
                    if state_out.shape == self.state.shape:
                        self.state = state_out
                        self.logger.debug("Updated RNN state")
                
                elif self.model_format == 'h0_c0':
                    if 'hn' in output_dict and 'cn' in output_dict:
                        hn_out = output_dict['hn']
                        cn_out = output_dict['cn']
                        
                        # Update states if shapes match
                        if hn_out.shape == self.h_state.shape and cn_out.shape == self.c_state.shape:
                            self.h_state = hn_out
                            self.c_state = cn_out
                            self.logger.debug("Updated RNN h and c states")
            
            return speech_prob
            
        except Exception as e:
            self.logger.error(f"Failed to run ONNX inference: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Fallback to a default value
            return 0.0
    
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
        # Save current RNN state to restore at the end
        # This ensures that this function doesn't affect ongoing detection
        saved_states = {}
        if hasattr(self, 'h_state'):
            saved_states['h_state'] = self.h_state.copy() if self.h_state is not None else None
        if hasattr(self, 'c_state'):
            saved_states['c_state'] = self.c_state.copy() if self.c_state is not None else None
        if hasattr(self, 'state'):
            saved_states['state'] = self.state.copy() if self.state is not None else None
        
        try:
            # Process audio in chunks of window_size_samples
            num_samples = len(audio_array)
            timestamps = []
            
            min_silence_samples = int(min_silence_duration_ms * self.sample_rate / 1000)
            min_speech_samples = int(min_speech_duration_ms * self.sample_rate / 1000)
            
            # Reset RNN states for clean processing
            self.reset_state()
            
            # Process audio in windows
            speech_start = None
            in_speech = False
            silence_counter = 0
            speech_probs = []  # Keep track of speech probabilities for debugging
            
            for i in range(0, num_samples, window_size_samples):
                chunk = audio_array[i:i + window_size_samples]
                
                # Skip if chunk is too small
                if len(chunk) < window_size_samples:
                    if len(chunk) < window_size_samples // 2:
                        break
                    # Pad with zeros if needed
                    pad_size = window_size_samples - len(chunk)
                    chunk = np.pad(chunk, (0, pad_size), 'constant')
                
                # Get speech probability for this chunk using our existing method
                speech_prob = self._predict_onnx(chunk)
                speech_probs.append(speech_prob)
                
                # Apply threshold logic
                if not in_speech and speech_prob >= threshold:
                    in_speech = True
                    speech_start = i
                    silence_counter = 0
                    self.logger.debug(f"Speech start detected at {i} samples with prob {speech_prob:.4f}")
                elif in_speech:
                    if speech_prob >= threshold:
                        silence_counter = 0
                    else:
                        silence_counter += window_size_samples
                        
                        if silence_counter >= min_silence_samples:
                            # End of speech detected
                            in_speech = False
                            speech_end = i
                            
                            self.logger.debug(f"Speech end detected at {speech_end} samples")
                            
                            # Only add if speech segment is long enough
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
                                self.logger.debug(f"Added valid speech segment: {speech_start}-{speech_end} samples")
                            else:
                                self.logger.debug(f"Discarded short speech segment: {speech_end - speech_start} samples")
                            
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
                    self.logger.debug(f"Added speech segment continuing to end: {speech_start}-{speech_end} samples")
            
            # Log summary for debugging
            avg_prob = sum(speech_probs) / len(speech_probs) if speech_probs else 0
            self.logger.debug(f"Speech timestamps: {len(timestamps)} segments found, avg prob: {avg_prob:.4f}")
            
            return timestamps
            
        except Exception as e:
            self.logger.error(f"Error getting speech timestamps: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
        finally:
            # Restore saved states
            for key, value in saved_states.items():
                if value is not None:
                    setattr(self, key, value)
    
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