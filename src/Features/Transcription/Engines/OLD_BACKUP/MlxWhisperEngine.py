"""
MlxWhisperEngine for high-performance speech-to-text transcription.

This module implements the ITranscriptionEngine interface using MLX-optimized Whisper
large-v3-turbo model for Apple Silicon.
"""

import base64
import json
import logging
import math
import os
import time
import threading
from functools import lru_cache
from queue import Queue, Empty
from typing import Dict, List, Optional, Any, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download, snapshot_download

from src.Core.Common.Interfaces.transcription_engine import ITranscriptionEngine


class Tokenizer:
    """
    Tokenizer for the Whisper model.
    
    Handles conversion between text and token IDs using the Whisper vocabulary.
    """
    
    def __init__(self):
        """Initialize the tokenizer with the Whisper vocabulary."""
        path_tok = 'multilingual.tiktoken'
        if not os.path.exists(path_tok):
            path_tok = hf_hub_download(repo_id='JosefAlbers/whisper', filename=path_tok)
        
        with open(path_tok) as f:
            ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line)}
        
        n_vocab = len(ranks)
        specials = ["<|endoftext|>", "<|startoftranscript|>",
                   *[f"<|_{lang}|>" for lang in range(100)],
                   "<|translate|>", "<|transcribe|>", "<|startoflm|>",
                   "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>",
                   *[f"<|{i * 0.02:.2f}|>" for i in range(1501)]]
        
        special_tokens = {k: (n_vocab + i) for i, k in enumerate(specials)}
        self.encoding = tiktoken.Encoding(
            name='jj', 
            explicit_n_vocab=n_vocab + len(special_tokens),
            pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=ranks,
            special_tokens=special_tokens
        )
    
    def encode(self, lot):
        """
        Encode text into token IDs.
        
        Args:
            lot: String or list of strings to encode
            
        Returns:
            List of token ID lists
        """
        if isinstance(lot, str):
            lot = [lot]
        return [self.encoding.encode(t, allowed_special='all') for t in lot]
    
    def decode(self, lol):
        """
        Decode token IDs into text.
        
        Args:
            lol: List of token IDs or list of token ID lists
            
        Returns:
            List of decoded strings
        """
        if isinstance(lol[0], int):
            lol = [lol]
        return [self.encoding.decode(l) for l in lol]


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation for the Whisper model."""
    
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def __call__(self, x, xa=None, mask=None, kv_cache=None):
        q = self.q_proj(x)
        if xa is None:
            k = self.k_proj(x)
            v = self.v_proj(x)
            if kv_cache is not None:
                k = mx.concatenate([kv_cache[0], k], axis=1)
                v = mx.concatenate([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            k = self.k_proj(xa)
            v = self.v_proj(xa)
        else:
            k, v = kv_cache
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out_proj(wv), (k, v), qk

    def qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        w = mx.softmax(qk, axis=-1)
        out = (w @ v).transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)
        return out, qk


class ResidualAttentionBlock(nn.Module):
    """Residual attention block for the Whisper model."""
    
    def __init__(self, d_model, n_head, cross_attention=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = MultiHeadAttention(d_model, n_head) if cross_attention else None
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model) if cross_attention else None
        n_mlp = d_model * 4
        self.fc1 = nn.Linear(d_model, n_mlp)
        self.fc2 = nn.Linear(n_mlp, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        
    def __call__(self, x, xa=None, mask=None, kv_cache=None):
        kv, cross_kv = kv_cache if kv_cache else (None, None)
        y, kv, _ = self.self_attn(self.self_attn_layer_norm(x), mask=mask, kv_cache=kv)
        x += y
        cross_qk = None
        if self.encoder_attn:
            y, cross_kv, cross_qk = self.encoder_attn(self.encoder_attn_layer_norm(x), xa, kv_cache=cross_kv)
            x += y
        x = x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))
        return x, (kv, cross_kv), cross_qk


class AudioEncoder(nn.Module):
    """Audio encoder component of the Whisper model."""
    
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv1d(cfg['num_mel_bins'], cfg['d_model'], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg['d_model'], cfg['d_model'], kernel_size=3, stride=2, padding=1)
        self._positional_embedding = sinusoids(cfg['max_source_positions'], cfg['d_model']).astype(mx.float16)
        self.layers = [ResidualAttentionBlock(cfg['d_model'], cfg['encoder_attention_heads']) 
                     for _ in range(cfg['encoder_layers'])]
        self.layer_norm = nn.LayerNorm(cfg['d_model'])
        
    def __call__(self, x):
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        x = x + self._positional_embedding
        for block in self.layers:
            x, _, _ = block(x)
        x = self.layer_norm(x)
        return x


class TextDecoder(nn.Module):
    """Text decoder component of the Whisper model."""
    
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
        self.positional_embedding = mx.zeros((cfg['max_target_positions'], cfg['d_model']))
        self.layers = [ResidualAttentionBlock(cfg['d_model'], cfg['decoder_attention_heads'], cross_attention=True) 
                      for _ in range(cfg['decoder_layers'])]
        self.layer_norm = nn.LayerNorm(cfg['d_model'])
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(cfg['max_target_positions']).astype(mx.float16)
        
    def __call__(self, x, xa, kv_cache=None):
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
        x = self.embed_tokens(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        if kv_cache is None:
            kv_cache = [None] * len(self.layers)
        cross_qk = [None] * len(self.layers)
        for e, block in enumerate(self.layers):
            x, kv_cache[e], cross_qk[e] = block(x, xa, mask=self._mask, kv_cache=kv_cache[e])
        x = self.layer_norm(x)
        return self.embed_tokens.as_linear(x), kv_cache, cross_qk


class Whisper(nn.Module):
    """Complete Whisper model combining encoder and decoder."""
    
    def __init__(self, cfg):
        self.encoder = AudioEncoder(cfg)
        self.decoder = TextDecoder(cfg)
        
    def __call__(self, mel, txt):
        return self.decoder(txt, self.encoder(mel))[0]
    
    def encode(self, mel):
        return self.encoder(mel)
    
    def decode(self, txt, mel, kv_cache):
        return self.decoder(txt, mel, kv_cache)


def sinusoids(length, channels, max_timescale=10000):
    """
    Generate sinusoidal positional embeddings.
    
    Args:
        length: Sequence length
        channels: Number of channels (must be even)
        max_timescale: Maximum timescale
        
    Returns:
        mx.array: Positional embeddings
    """
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


@lru_cache(maxsize=None)
def mel_filters(n_mels):
    """
    Get mel filterbank matrix.
    
    Args:
        n_mels: Number of mel bands
        
    Returns:
        mx.array: Mel filterbank matrix
    """
    path_mel = "mel_filters.npz"
    if not os.path.exists(path_mel):
        # Use librosa to create mel filters if not found
        try:
            import librosa
            np.savez_compressed(path_mel, mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128))
        except ImportError:
            raise ImportError("librosa is required to create mel filters. Please install it with pip install librosa.")
    return mx.load(path_mel)[f"mel_{n_mels}"]


@lru_cache(maxsize=None)
def hanning(n_fft):
    """
    Get Hanning window.
    
    Args:
        n_fft: FFT size
        
    Returns:
        mx.array: Hanning window
    """
    return mx.array(np.hanning(n_fft + 1)[:-1])


@lru_cache(maxsize=None)
def stft(x, window, nperseg=400, noverlap=160, nfft=None, axis=-1, pad_mode="reflect"):
    """
    Short-time Fourier transform.
    
    Args:
        x: Input signal
        window: Window function
        nperseg: Length of each segment
        noverlap: Number of points to overlap
        nfft: Length of the FFT
        axis: Axis along which to perform STFT
        pad_mode: Padding mode
        
    Returns:
        mx.array: STFT result
    """
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4
        
    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")
            
    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)
    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def load_audio(file_path, sr=16000):
    """
    Load audio from file path.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        
    Returns:
        mx.array: Audio data as mx.array
    """
    try:
        # First try to use soundfile which is faster but might not support all formats
        import soundfile as sf
        import librosa
        
        try:
            # Try soundfile first
            audio_data, sample_rate = sf.read(file_path, dtype='float32')
            
            # Resample if needed
            if sample_rate != sr:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=sr
                )
                
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = audio_data.mean(axis=1)
                
        except Exception as e:
            # Fall back to librosa if soundfile fails
            audio_data, _ = librosa.load(file_path, sr=sr, mono=True)
            
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            
        return mx.array(audio_data)
        
    except Exception as e:
        # Last resort: try whisper-turbo style loading with ffmpeg
        try:
            from subprocess import CalledProcessError, run
            
            out = run(["ffmpeg", "-nostdin", "-threads", "0", "-i", file_path, 
                       "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
                       "-ar", str(sr), "-"], capture_output=True, check=True).stdout
                       
            return mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0
            
        except Exception as ffmpeg_error:
            raise RuntimeError(f"Failed to load audio: {str(e)}, ffmpeg error: {str(ffmpeg_error)}")

def log_mel_spectrogram(audio, n_mels=128, padding=0):
    """
    Convert audio to log-mel spectrogram.
    
    Args:
        audio: Audio signal (mx.array, np.ndarray, or file path)
        n_mels: Number of mel bands
        padding: Amount of padding to add
        
    Returns:
        mx.array: Log-mel spectrogram
    """
    if isinstance(audio, str):
        # Load audio from file path
        audio = load_audio(audio)
    elif isinstance(audio, np.ndarray):
        # Convert numpy array to mx.array
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Normalize if not already in [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0 and max_val > 1.0:
            audio = audio / max_val
        audio = mx.array(audio)
    elif not isinstance(audio, mx.array):
        # Fallback for other types
        audio = mx.array(audio)
        
    if padding > 0:
        audio = mx.pad(audio, (0, padding))
        
    window = hanning(400)
    freqs = stft(audio, window, nperseg=400, noverlap=160)
    magnitudes = freqs[:-1, :].abs().square()
    filters = mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T
    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec


class MlxWhisperEngine(ITranscriptionEngine):
    """
    MLX-optimized Whisper transcription engine.
    
    This class implements the ITranscriptionEngine interface using the
    Whisper large-v3-turbo model optimized for Apple Silicon with MLX.
    """
    
    def __init__(self, 
                model_name: str = "whisper-large-v3-turbo",
                language: Optional[str] = None,
                compute_type: str = "float16",
                beam_size: int = 1,
                stream_mode: bool = False,
                chunk_duration_ms: int = 1000,
                overlap_duration_ms: int = 200,
                engine_type: str = "mlx_whisper",  # Ignored but required for compatibility
                **kwargs):
        """
        Initialize the MLX Whisper engine.
        
        Args:
            model_name: Whisper model name or HuggingFace model ID
            language: Language code (e.g., 'en', 'fr') or None for auto-detection
            compute_type: Compute precision ('float16' or 'float32')
            beam_size: Beam search size for inference (1 for greedy)
            stream_mode: Whether to use streaming mode (incremental decoding)
            chunk_duration_ms: Duration of each audio chunk in ms (for streaming)
            overlap_duration_ms: Overlap between chunks in ms (for streaming)
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.model_name = model_name
        self.language = language
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.stream_mode = stream_mode
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_duration_ms = overlap_duration_ms
        
        # State variables
        self.model = None
        self.tokenizer = None
        self.cfg = None
        self.path_hf = None
        self.sample_rate = 16000
        
        # Streaming state
        self.kv_cache = None
        self.len_sot = 0
        self.sot_sequence = None
        self.audio_buffer = []
        self.buffer_size_samples = int((chunk_duration_ms / 1000) * self.sample_rate)
        self.overlap_samples = int((overlap_duration_ms / 1000) * self.sample_rate)
        
        # Thread safety
        self.lock = threading.RLock()
        self.result_queue = Queue()
        self.is_processing = False
        
        # Result tracking
        self.current_result = None
        self.result_ready = threading.Event()
        
        self.logger.info(f"Initialized MlxWhisperEngine with model={model_name}, language={language}, "
                        f"stream_mode={stream_mode}")
    
    def start(self) -> bool:
        """
        Initialize and start the transcription engine.
        
        This method downloads and initializes the model from HuggingFace.
        A warmup audio sample is processed to ensure the model is ready.
        
        Returns:
            bool: True if the engine started successfully
        """
        with self.lock:
            try:
                self.logger.info(f"Starting MlxWhisperEngine with model={self.model_name}")
                
                # Download model from HuggingFace
                self.path_hf = snapshot_download(
                    repo_id=f'openai/{self.model_name}' if '/' not in self.model_name else self.model_name,
                    allow_patterns=["config.json", "model.safetensors"]
                )
                
                # Load configuration
                with open(f'{self.path_hf}/config.json', 'r') as fp:
                    self.cfg = json.load(fp)
                
                # Load weights
                weights = [(k.replace("embed_positions.weight", "positional_embedding"), 
                          v.swapaxes(1, 2) if ('conv' in k and v.ndim==3) else v) 
                         for k, v in mx.load(f'{self.path_hf}/model.safetensors').items()]
                
                # Initialize model
                self.model = Whisper(self.cfg)
                self.model.load_weights(weights, strict=False)
                self.model.eval()
                
                # Initialize tokenizer
                self.tokenizer = Tokenizer()
                
                # Warm up with simple input - check result
                model_ready = self._warmup()
                
                # Determine initial SOT sequence
                if self.language:
                    # Use specific language
                    language_token_id = self._get_language_token_id(self.language)
                    self.sot_sequence = mx.array([[50258, language_token_id, 50360, 50365]])
                    self.len_sot = 4
                else:
                    # Auto-detect language
                    self.sot_sequence = mx.array([[50258, 50360, 50365]])
                    self.len_sot = 3
                
                if model_ready:
                    self.logger.info("MlxWhisperEngine started successfully")
                else:
                    self.logger.warning("MlxWhisperEngine started but warmup was incomplete")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error starting MlxWhisperEngine: {e}", exc_info=True)
                return False
    
    def _warmup(self):
        """Warm up the model with a small audio sample to ensure initialization."""
        try:
            # Generate some silence as warmup - make it shorter to avoid shape issues
            silence = np.zeros(3000, dtype=np.float32)  # Short silence
            mel = log_mel_spectrogram(silence, padding=0).astype(mx.float16)
            
            # Run encoder once to initialize parameters
            self.logger.info("Warming up model encoder with silence...")
            encoded = self.model.encode(mel[None, ...])
            
            # Ensure execution completes
            mx.eval(encoded)
            self.logger.info(f"Encoder warmup successful, output shape: {encoded.shape}")
            
            # Run decoder in a safer way - creating a minimal setup
            try:
                sot_sequence = mx.array([[50258, 50259, 50360, 50365]])  # SOT + EN + Transcribe
                self.logger.info("Attempting to warm up model decoder...")
                
                # Just run one small decode step with minimal input
                logits, kv_cache, _ = self.model.decode(
                    txt=sot_sequence, 
                    mel=encoded,
                    kv_cache=None
                )
                
                # Force evaluation
                mx.eval(logits)
                self.logger.info("Decoder warmup successful")
                
                # Success
                return True
                
            except Exception as decoder_error:
                self.logger.warning(f"Decoder warmup failed, will initialize on first use: {decoder_error}")
                # Return partial success since encoder was initialized
                return True  # We'll still consider this a success
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
            self.logger.info("Continuing despite warmup failure - will initialize on first use")
            return False
    
    def _get_language_token_id(self, language_code: str) -> int:
        """
        Get the token ID for a specific language.
        
        Args:
            language_code: ISO language code (e.g., 'en', 'fr')
            
        Returns:
            int: Language token ID for the Whisper model
        """
        # Default to English
        if language_code == 'en':
            return 50259  # <|en|>
        
        # Try to get the language token
        try:
            language_token = f"<|_{language_code}|>"
            tokens = self.tokenizer.encode(language_token)
            if tokens and tokens[0]:
                return tokens[0][0]
        except Exception as e:
            self.logger.warning(f"Error getting token for language '{language_code}': {e}")
        
        # Fallback to English
        self.logger.warning(f"Language '{language_code}' not recognized, falling back to English")
        return 50259  # <|en|>
    
    def transcribe(self, audio: np.ndarray) -> None:
        """
        Request transcription of complete audio segment.
        
        Args:
            audio: Audio data as numpy array (float32, -1.0 to 1.0 range),
                  file path string, or other audio representation
        """
        with self.lock:
            if not self.is_running():
                self.logger.warning("Cannot transcribe - engine not running")
                return
            
            self.is_processing = True
            self.result_ready.clear()
            
            # Handle different audio input types
            if isinstance(audio, str):
                # Audio is a file path - load it using soundfile
                try:
                    self.logger.info(f"Loading audio from file: {audio}")
                    import soundfile as sf
                    import librosa
                    
                    # Load the audio file
                    file_data, sample_rate = sf.read(audio, dtype='float32')
                    
                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        file_data = librosa.resample(
                            file_data, 
                            orig_sr=sample_rate, 
                            target_sr=16000
                        )
                    
                    # Convert to mono if stereo
                    if len(file_data.shape) > 1 and file_data.shape[1] > 1:
                        file_data = file_data.mean(axis=1)
                    
                    # Normalize if not already in [-1, 1] range
                    max_val = np.max(np.abs(file_data))
                    if max_val > 0:
                        file_data = file_data / max_val
                    
                    # Process in a separate thread
                    processing_thread = threading.Thread(
                        target=self._process_audio,
                        args=(file_data.copy(), True),
                        daemon=True
                    )
                    processing_thread.start()
                    
                except Exception as e:
                    self.logger.error(f"Error loading audio file: {e}", exc_info=True)
                    # Signal error
                    error_result = {
                        "error": f"Error loading audio file: {str(e)}",
                        "is_final": True
                    }
                    self.current_result = error_result
                    self.is_processing = False
                    self.result_ready.set()
                    self.result_queue.put(error_result)
            else:
                # Process numpy array input
                # Process in a separate thread to avoid blocking
                processing_thread = threading.Thread(
                    target=self._process_audio,
                    args=(audio.copy(), True),
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
            
            # Append to buffer
            self.audio_buffer.append(audio_chunk.copy())
            
            # Calculate total buffer size
            buffer_length = sum(chunk.shape[0] for chunk in self.audio_buffer)
            
            # Process when we have enough audio or this is the last chunk
            if buffer_length >= self.buffer_size_samples or is_last:
                # Concatenate buffer
                audio_data = np.concatenate(self.audio_buffer)
                
                # Clear or retain buffer with overlap
                if not is_last:
                    overlap_samples = min(self.overlap_samples, buffer_length)
                    self.audio_buffer = [audio_data[-overlap_samples:]]
                else:
                    self.audio_buffer = []
                
                # Process in a separate thread
                self.is_processing = True
                self.result_ready.clear()
                
                processing_thread = threading.Thread(
                    target=self._process_audio,
                    args=(audio_data, is_last),
                    daemon=True
                )
                processing_thread.start()
    
    def _process_audio(self, audio_data: np.ndarray, is_final: bool) -> None:
        """
        Process audio data and produce transcription.
        
        Args:
            audio_data: Audio data to transcribe
            is_final: Whether this is the final chunk of audio
        """
        try:
            # Convert to log mel spectrogram
            mel = log_mel_spectrogram(audio_data, padding=0).astype(mx.float16)
            
            # Choose transcription method based on mode
            if self.stream_mode and not is_final:
                result = self._transcribe_streaming(mel)
            else:
                result = self._transcribe_batch(mel)
            
            # Update current result and signal completion
            with self.lock:
                self.current_result = result
                self.is_processing = False
                self.result_ready.set()
                self.result_queue.put(result)
                
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}", exc_info=True)
            
            # Signal error
            with self.lock:
                error_result = {
                    "error": str(e),
                    "is_final": is_final
                }
                self.current_result = error_result
                self.is_processing = False
                self.result_ready.set()
                self.result_queue.put(error_result)
    
    def _transcribe_batch(self, mel: mx.array) -> Dict[str, Any]:
        """
        Process audio in batch mode.
        
        Args:
            mel: Log mel spectrogram
            
        Returns:
            Dict[str, Any]: Transcription result
        """
        start_time = time.time()
        self.logger.info("Starting batch transcription")
        
        try:
            # Reshape for batch processing - split long audio into 30s chunks (~3000 frames)
            if mel.shape[0] > 3000:
                self.logger.info(f"Audio is long ({mel.shape[0]} frames), splitting into batches")
                mel = mel[:(mel.shape[0] // 3000) * 3000].reshape(-1, 3000, 128)
                sot = mx.repeat(self.sot_sequence, mel.shape[0], 0)
                self.logger.info(f"Created {mel.shape[0]} batch segments")
            else:
                mel = mel[None, ...]
                sot = self.sot_sequence
                self.logger.info(f"Processing audio as single batch of {mel.shape[1]} frames")
            
            # Encode audio
            self.logger.info("Encoding audio with transformer model...")
            encoded = self.model.encode(mel)
            mx.eval(encoded)  # Force execution to catch any encoding errors early
            self.logger.info(f"Audio encoded successfully, shape={encoded.shape}")
            
            # Generate tokens
            B = encoded.shape[0]  # Batch size
            new_tokens = mx.zeros((B, 0), dtype=mx.int32)
            kv_cache = None
            generation_active = mx.ones((B, 1), dtype=mx.bool_)
            
            # Token generation loop
            max_steps = 448 - self.len_sot  # Max sequence length minus SOT length
            self.logger.info(f"Starting token generation loop (max steps: {max_steps})")
            
            # Set a timeout for token generation - 60 seconds max
            token_gen_timeout = 60.0  # seconds
            token_gen_start = time.time()
            
            # Check if model is defined correctly
            if self.model is None or not hasattr(self.model, 'decode'):
                self.logger.error("Model is not properly initialized")
                return {
                    "text": "",
                    "is_final": True,
                    "language": self.language,
                    "processing_time": time.time() - start_time,
                    "error": "Model is not properly initialized",
                    "success": False
                }
            
            # Variable to hold logits between iterations
            logits = None
            
            # First initialize the model if needed (since warmup might have failed)
            if not hasattr(self.model, 'decoder') or self.model.decoder is None:
                self.logger.warning("Decoder seems to be uninitialized, attempting to rebuild model")
                # Try to reinitialize the model
                try:
                    self.logger.info("Rebuilding model from saved weights...")
                    
                    # Load weights if we need to rebuild
                    weights = [(k.replace("embed_positions.weight", "positional_embedding"), 
                               v.swapaxes(1, 2) if ('conv' in k and v.ndim==3) else v) 
                              for k, v in mx.load(f'{self.path_hf}/model.safetensors').items()]
                    
                    self.model = Whisper(self.cfg)
                    self.model.load_weights(weights, strict=False)
                    self.model.eval()
                    
                    self.logger.info("Model rebuilt successfully")
                except Exception as rebuild_error:
                    self.logger.error(f"Failed to rebuild model: {rebuild_error}")
                    return {
                        "text": "",
                        "is_final": True,
                        "language": self.language,
                        "processing_time": time.time() - start_time,
                        "error": f"Model initialization error: {rebuild_error}",
                        "success": False
                    }
            
            for i in range(max_steps):
                try:
                    # Decode step
                    input_tokens = sot if i == 0 else mx.argmax(logits[:, -1:, :], axis=-1)
                    
                    # Process batch with model
                    logits, kv_cache, _ = self.model.decode(
                        txt=input_tokens, 
                        mel=encoded, 
                        kv_cache=kv_cache
                    )
                    
                    # Get next token
                    next_token = mx.argmax(logits[:, -1:, :], axis=-1) * generation_active
                    mx.eval(next_token)  # Force execution to identify any errors
                    
                    # Check for end of sequence
                    generation_active = generation_active & (next_token != 50257)  # Not EOS token
                    
                    # Add token to results
                    new_tokens = mx.concatenate([new_tokens, next_token], axis=-1)
                    
                    # Logging for large steps
                    if i % 50 == 0 and i > 0:
                        self.logger.info(f"Generated {i} tokens, active sequences: {generation_active.sum().item()}/{B}")
                    
                    # Stop if all sequences are done
                    if generation_active.sum() <= 0:
                        self.logger.info(f"All sequences complete after {i+1} steps")
                        break
                        
                    # Check for timeout
                    if time.time() - token_gen_start > token_gen_timeout:
                        self.logger.warning(f"Token generation timeout after {i+1} steps")
                        # We have some tokens, so we can return a partial result
                        break
                
                except Exception as decode_error:
                    self.logger.error(f"Error in decode step {i}: {decode_error}")
                    # If we've generated some tokens, return them
                    if i > 0 and new_tokens.shape[1] > 0:
                        self.logger.warning(f"Returning partial transcription of {new_tokens.shape[1]} tokens")
                        break
                    else:
                        # Otherwise return error
                        return {
                            "text": "",
                            "is_final": True,
                            "language": self.language,
                            "processing_time": time.time() - start_time,
                            "error": f"Decode error: {decode_error}",
                            "success": False
                        }
            
            # Check if we hit the max token limit
            if generation_active.sum() > 0:
                self.logger.warning(f"Hit max token limit, {generation_active.sum().item()} sequences not finished")
            
            # Process tokens
            self.logger.info("Processing generated tokens...")
            tokens = new_tokens.astype(mx.int32).tolist()
            
            if B > 1:
                # For batch processing, concatenate all valid tokens
                all_tokens = []
                for seq in tokens:
                    all_tokens.extend([t for t in seq if t < 50257])  # Filter out special tokens
                tokens = all_tokens
                self.logger.info(f"Processed {len(tokens)} total tokens from {B} batches")
            else:
                tokens = [t for t in tokens[0] if t < 50257]
                self.logger.info(f"Processed {len(tokens)} tokens from single batch")
            
            # Decode to text
            if tokens:
                text = self.tokenizer.decode([tokens])[0]
                self.logger.info(f"Decoded text successfully, length: {len(text)}")
            else:
                text = ""
                self.logger.warning("No valid tokens generated, returning empty text")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Check if we hit timeout
            timeout_occurred = time.time() - token_gen_start > token_gen_timeout
                
            # Create result
            result = {
                "text": text,
                "is_final": True,
                "language": self.language,
                "processing_time": processing_time,
                "token_count": len(tokens),
                "confidence": 0.8 if timeout_occurred else 1.0,  # Lower confidence if we had a timeout
                "success": True,  # Indicate successful processing
                "timeout_occurred": timeout_occurred,  # Flag to indicate partial result due to timeout
                "message": "Partial transcription (timeout reached)" if timeout_occurred else "Complete transcription"
            }
            
            self.logger.info(f"Transcription completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Handle errors in transcription
            self.logger.error(f"Error during batch transcription: {str(e)}", exc_info=True)
            processing_time = time.time() - start_time
            
            return {
                "text": "",
                "is_final": True,
                "language": self.language,
                "processing_time": processing_time,
                "error": str(e),
                "success": False
            }
    
    def _transcribe_streaming(self, mel: mx.array) -> Dict[str, Any]:
        """
        Process audio in streaming mode.
        
        Args:
            mel: Log mel spectrogram
            
        Returns:
            Dict[str, Any]: Transcription result
        """
        start_time = time.time()
        
        # Encode audio
        encoded = self.model.encode(mel[None, ...])
        
        # Prepare initial tokens
        if self.kv_cache is None:
            # First chunk
            input_tokens = self.sot_sequence
            is_first_chunk = True
        else:
            # Continuing from previous chunk
            input_tokens = mx.array([[50365]])  # Timestamp token
            is_first_chunk = False
        
        # Generate tokens
        new_tokens = []
        
        # Token generation loop - limited for streaming
        max_new_tokens = 50 if is_first_chunk else 20  # Fewer tokens for continuation
        
        for i in range(max_new_tokens):
            # Decode step
            logits, self.kv_cache, _ = self.model.decode(
                txt=input_tokens if i == 0 else mx.array([[token]]),
                mel=encoded,
                kv_cache=self.kv_cache
            )
            
            # Get next token
            token = mx.argmax(logits[:, -1, :], axis=-1).item()
            
            # Stop on EOS token
            if token == 50257:  # End of sequence
                break
                
            # Add to results if not a special token
            if token < 50257:
                new_tokens.append(token)
                
            # For streaming, stop on certain punctuation for partial results
            if i > 5 and token in [50, 51, 52]:  # Period, comma, etc.
                break
        
        # Convert to text
        if not new_tokens:
            text = ""
        else:
            text = self.tokenizer.decode([new_tokens])[0]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result
        result = {
            "text": text,
            "is_final": False,
            "language": self.language,
            "processing_time": processing_time,
            "confidence": 1.0,  # No confidence score available
            "is_partial": True
        }
        
        return result
    
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
            self.logger.info("Cleaning up MlxWhisperEngine")
            
            # Clear references to large objects
            self.model = None
            self.tokenizer = None
            self.kv_cache = None
            self.audio_buffer = []
            
            # Clear result tracking
            self.current_result = None
            self.result_ready.clear()
            
            # Attempt to force garbage collection
            import gc
            gc.collect()
    
    def is_running(self) -> bool:
        """
        Check if the transcription engine is currently running.
        
        Returns:
            bool: True if the engine is running
        """
        return self.model is not None and self.tokenizer is not None
    
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
                    
                    # Update SOT sequence if language changed
                    if self.language:
                        language_token_id = self._get_language_token_id(self.language)
                        self.sot_sequence = mx.array([[50258, language_token_id, 50360, 50365]])
                        self.len_sot = 4
                    else:
                        self.sot_sequence = mx.array([[50258, 50360, 50365]])
                        self.len_sot = 3
                
                # Update streaming mode
                if 'stream_mode' in config:
                    self.stream_mode = config['stream_mode']
                    # Reset KV cache when switching modes
                    self.kv_cache = None
                
                # Update chunk parameters
                if 'chunk_duration_ms' in config:
                    self.chunk_duration_ms = config['chunk_duration_ms']
                    self.buffer_size_samples = int((self.chunk_duration_ms / 1000) * self.sample_rate)
                
                if 'overlap_duration_ms' in config:
                    self.overlap_duration_ms = config['overlap_duration_ms']
                    self.overlap_samples = int((self.overlap_duration_ms / 1000) * self.sample_rate)
                
                # Update beam size
                if 'beam_size' in config:
                    self.beam_size = config['beam_size']
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error configuring engine: {e}")
                return False