"""
DirectMlxWhisperEngine for high-performance speech-to-text transcription.

This module implements the ITranscriptionEngine interface using a direct MLX-optimized Whisper
model for Apple Silicon without process isolation.
"""

import base64
import json
import logging
import math
import os
import threading
import time
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
        # Handle empty lists to prevent "list index out of range" error
        if not lol:
            return [""]
            
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
        super().__init__()
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
    logger = logging.getLogger(__name__)
    logger.info(f"Loading audio from file: {file_path}")
    
    # TEMPORARY DEBUG: Extra file validation
    import os
    if not os.path.exists(file_path):
        logger.error(f"Audio file does not exist: {file_path}")
        # Return empty array instead of raising exception for debugging
        return mx.array(np.zeros(1, dtype=np.float32))
    
    try:
        # First try to use soundfile which is faster but might not support all formats
        import soundfile as sf
        import librosa
        
        try:
            # Try soundfile first
            logger.info(f"Loading with soundfile: {file_path}")
            audio_data, sample_rate = sf.read(file_path, dtype='float32')
            
            logger.info(f"Loaded audio: shape={audio_data.shape}, sample_rate={sample_rate}, "
                      f"min={np.min(audio_data):.5f}, max={np.max(audio_data):.5f}")
            
            # Resample if needed
            if sample_rate != sr:
                logger.info(f"Resampling from {sample_rate}Hz to {sr}Hz")
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=sr
                )
                
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                logger.info(f"Converting from {audio_data.shape[1]} channels to mono")
                audio_data = audio_data.mean(axis=1)
                
        except Exception as e:
            # Fall back to librosa if soundfile fails
            logger.warning(f"Soundfile failed ({str(e)}), falling back to librosa")
            audio_data, _ = librosa.load(file_path, sr=sr, mono=True)
            logger.info(f"Loaded with librosa: shape={audio_data.shape}, "
                      f"min={np.min(audio_data):.5f}, max={np.max(audio_data):.5f}")
            
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            if max_val > 1.0:
                logger.info(f"Normalizing audio (max_val={max_val:.5f})")
                audio_data = audio_data / max_val
            
        result_array = mx.array(audio_data)
        logger.info(f"Converted to MLX array: shape={result_array.shape}")
        return result_array
        
    except Exception as e:
        # Last resort: try ffmpeg style loading
        logger.warning(f"Standard loading failed ({str(e)}), trying ffmpeg")
        try:
            from subprocess import CalledProcessError, run
            
            out = run(["ffmpeg", "-nostdin", "-threads", "0", "-i", file_path, 
                       "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
                       "-ar", str(sr), "-"], capture_output=True, check=True).stdout
                       
            result = mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0
            logger.info(f"Loaded with ffmpeg: shape={result.shape}")
            return result
            
        except Exception as ffmpeg_error:
            logger.error(f"Failed to load audio: {str(e)}, ffmpeg error: {str(ffmpeg_error)}")
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
    logger = logging.getLogger(__name__)
    
    # TEMPORARY DEBUG: Log detailed information about input
    if isinstance(audio, str):
        logger.info(f"log_mel_spectrogram processing file: {audio}")
        try:
            import os
            if os.path.exists(audio):
                logger.info(f"File exists, size: {os.path.getsize(audio)} bytes")
            else:
                logger.warning(f"File does not exist: {audio}")
        except Exception as e:
            logger.error(f"Error checking file: {e}")
    elif isinstance(audio, np.ndarray):
        logger.info(f"log_mel_spectrogram input array: shape={audio.shape}, dtype={audio.dtype}")
        logger.info(f"Array stats: min={np.min(audio):.5f}, max={np.max(audio):.5f}, "
                   f"mean={np.mean(audio):.5f}, has_data={not np.allclose(audio, 0)}")
    else:
        logger.info(f"log_mel_spectrogram input type: {type(audio)}")
    
    # Handle empty inputs - return minimal valid spectrogram
    if isinstance(audio, np.ndarray) and (audio.size == 0 or np.all(audio == 0)):
        # Return minimal spectrogram with correct dimensions
        logger.warning("Empty audio data provided to log_mel_spectrogram")
        return mx.zeros((1, n_mels))
    
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
    
    # Handle empty mx arrays
    if hasattr(audio, 'size') and audio.size == 0:
        logging.getLogger(__name__).warning("Empty mx.array provided to log_mel_spectrogram")
        return mx.zeros((1, n_mels))
        
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


class Transcriber(nn.Module):
    """Transcription model using MLX-optimized Whisper."""
    
    def __init__(self, cfg):
        super().__init__()
        self.model = Whisper(cfg)
        self.tokenizer = Tokenizer()
        self.len_sot = 0
        
    def __call__(self, path_audio, any_lang, quick):
        """
        Transcribe audio using either parallel or recurrent mode.
        
        Args:
            path_audio: Audio path or data
            any_lang: Whether to auto-detect language
            quick: Whether to use parallel (quick) or recurrent mode
            
        Returns:
            str: Transcribed text
        """
        logger = logging.getLogger(__name__)
        raw = log_mel_spectrogram(path_audio).astype(mx.float16)
        sot = mx.array([[50258, 50360, 50365]]) if any_lang else mx.array([[50258, 50259, 50360, 50365]])
        self.len_sot = sot.shape[-1]
        
        # For short audio segments (common in VAD-triggered events), always use recurrent mode
        # This provides better results for short utterances
        if raw.shape[0] < 3000:
            logger.info(f"Short audio segment detected ({raw.shape[0]} frames) - forcing recurrent mode")
            txt = self.recurrent(raw, sot)
        else:
            # For longer segments, use the specified mode
            txt = self.parallel(raw, sot) if quick else self.recurrent(raw, sot)
            
        return txt
    
    def recurrent(self, raw, sot):
        """Process audio sequentially for higher accuracy."""
        logger = logging.getLogger(__name__)
        logger.info(f"recurrent: Input spectrogram shape={raw.shape}, length={len(raw)}")
        
        # Handle very short audio (less than 3000 frames)
        if len(raw) < 3000:
            logger.info(f"Short audio segment detected ({len(raw)} < 3000 samples), padding for processing")
            
            # Create padded version
            padded_raw = mx.zeros((3000, raw.shape[1]))
            padded_raw[:len(raw)] = raw
            
            # Process the single chunk
            logger.info("Processing padded segment as a single chunk")
            piece = self.step(padded_raw[None], sot)
            
            # Extract tokens
            new_tok = piece.astype(mx.int32)
            new_tok = [i for i in new_tok.tolist()[0] if i < 50257]
            
            if not new_tok:
                logger.warning("No tokens generated for short audio segment")
                return ""
                
            result = self.tokenizer.decode(new_tok)[0]
            logger.info(f"Short audio transcription result: {len(result)} characters")
            return result
        
        # Normal processing for longer audio
        new_tok, i = mx.zeros((1,0), dtype=mx.int32), 0
        chunk_count = 0
        
        while i+3000 < len(raw):
            chunk_count += 1
            logger.info(f"Processing chunk {chunk_count} starting at offset {i}")
            
            # Process current chunk
            piece = self.step(raw[i:i+3000][None], sot)
            
            # Determine next position
            arg_hop = mx.argmax(piece).item()
            hop = (piece[:,arg_hop].astype(mx.int32).item()-50365)*2
            
            # Add tokens to result
            new_tok = mx.concatenate([new_tok, piece[:,:arg_hop]], axis=-1)
            
            # Move to next chunk
            i += hop if hop > 0 else 3000
            logger.info(f"Moving to offset {i}" + (" (via timestamp)" if hop > 0 else ""))
        
        # Handle any remaining audio (less than 3000 frames)
        if i < len(raw) and len(raw) - i > 10:  # Only process if there's meaningful content left
            chunk_count += 1
            logger.info(f"Processing final chunk {chunk_count} with {len(raw) - i} frames")
            
            # Pad the final chunk to 3000 frames
            padded_chunk = mx.zeros((3000, raw.shape[1]))
            padded_chunk[:(len(raw) - i)] = raw[i:]
            
            # Process final chunk
            piece = self.step(padded_chunk[None], sot)
            
            # Add tokens to result
            arg_hop = mx.argmax(piece).item()
            new_tok = mx.concatenate([new_tok, piece[:,:arg_hop]], axis=-1)
        
        # Extract valid tokens
        new_tok = [i for i in new_tok.astype(mx.int32).tolist()[0] if i < 50257]
        logger.info(f"Generated {len(new_tok)} tokens from {chunk_count} chunks")
        
        # Handle empty token list
        if not new_tok:
            logger.warning("Empty token list after processing all chunks")
            return ""
            
        # Decode to text
        result = self.tokenizer.decode(new_tok)[0]
        logger.info(f"Decoded to {len(result)} characters")
        return result
    
    def parallel(self, raw, sot):
        """Process audio in parallel for faster processing."""
        logger = logging.getLogger(__name__)
        
        # Log the raw audio spectrogram shape
        logger.info(f"parallel: Input spectrogram shape={raw.shape}")
        
        # Special handling for short audio segments
        if raw.shape[0] < 3000:
            logger.info(f"Short audio segment detected ({raw.shape[0]} < 3000 samples), applying special handling")
            
            # For very short segments, we pad them to 3000 to ensure the model has enough context
            # This helps the model handle short speech segments better
            if raw.shape[0] > 0:
                padding_needed = 3000 - raw.shape[0]
                logger.info(f"Padding spectrogram with {padding_needed} frames")
                
                # Create padding (repeating the last frame or using zeros)
                if raw.shape[0] > 10:  # If we have at least some real data, repeat the last few frames
                    # Use the last 10 frames (or fewer if audio is shorter) and repeat them
                    last_frames = raw[-min(10, raw.shape[0]):]
                    padding = mx.concatenate([last_frames] * (padding_needed // last_frames.shape[0] + 1), axis=0)
                    padding = padding[:padding_needed]
                else:
                    # Create zero padding if we have very little data
                    padding = mx.zeros((padding_needed, raw.shape[1]))
                
                # Pad the original data
                raw = mx.concatenate([raw, padding], axis=0)
                logger.info(f"Padded spectrogram shape={raw.shape}")
            else:
                # If we have no data at all, create a minimal valid input
                raw = mx.zeros((3000, 128))
                logger.info("Created minimal valid input spectrogram")
            
            # Reshape to the expected 3D format with batch dimension
            raw = raw.reshape(-1, 3000, 128)
        else:
            # Normal processing for longer audio
            # Ensure input is multiple of 3000 and reshape
            raw = raw[:(raw.shape[0]//3000)*3000].reshape(-1, 3000, 128)
            logger.info(f"Reshaped spectrogram to {raw.shape}")
        
        # Safety check to prevent extremely large inputs - increased limit from 360 to 1000
        assert raw.shape[0] < 1000, f"Input too large: {raw.shape}"
        
        # If we have no valid chunks (though this shouldn't happen now), return empty string
        if raw.shape[0] == 0:
            logger.warning("No valid chunks found after preprocessing")
            return ""
        
        # Generate tokens
        logger.info(f"Processing {raw.shape[0]} chunks with model")
        sot = mx.repeat(sot, raw.shape[0], 0)
        new_tok = self.step(raw, sot)
        
        # Extract token indices
        arg_hop = mx.argmax(new_tok, axis=-1).tolist()
        new_tok = [i[:a] for i,a in zip(new_tok.astype(mx.int32).tolist(),arg_hop)]
        new_tok = [i for i in sum(new_tok, []) if i < 50257]
        
        # Log token count
        logger.info(f"Generated {len(new_tok)} tokens")
        
        # Handle empty token list
        if not new_tok:
            logger.warning("Empty token list after processing")
            return ""
        
        # Decode tokens to text
        result = self.tokenizer.decode(new_tok)[0]
        logger.info(f"Decoded to {len(result)} characters")
        return result
    
    def step(self, mel, txt):
        """Process a single step of transcription."""
        mel = self.model.encode(mel)
        kv_cache = None
        B = mel.shape[0]
        new_tok = mx.zeros((B,0), dtype=mx.int32)
        goon = mx.ones((B,1), dtype=mx.bool_)
        for i in range(449-self.len_sot):
            logits, kv_cache, _ = self.model.decode(
                txt=txt if i == 0 else mx.argmax(logits[:,-1:,:], axis=-1), 
                mel=mel, 
                kv_cache=kv_cache
            )
            txt = mx.argmax(logits[:,-1,:], axis=-1, keepdims=True) * goon
            mx.eval(txt)
            goon *= (txt != 50257)
            new_tok = mx.concatenate([new_tok, txt], axis=-1)
            if goon.sum() <= 0:
                break
        return new_tok


class DirectMlxWhisperEngine(ITranscriptionEngine):
    """Direct in-process MLX Whisper implementation without process isolation."""
    
    def __init__(self, model_name="whisper-large-v3-turbo", language=None, compute_type="float16", 
                beam_size=1, streaming=True, **kwargs):
        """
        Initialize the direct MLX Whisper engine.
        
        Args:
            model_name: Name of the Whisper model to use
            language: Language code or None for auto-detection
            compute_type: Compute precision ('float16' or 'float32')
            beam_size: Beam size for inference
            streaming: Whether to use streaming mode
            **kwargs: Additional configuration options
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.model_name = model_name
        self.language = language
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.streaming = streaming
        self.quick_mode = kwargs.get('quick_mode', True)  # Default to parallel/quick mode
        
        # Model state
        self.transcriber = None
        self.cfg = None
        self.weights = None
        self.model_path = None
        
        # Audio buffer for streaming
        self.audio_buffer = []
        self.sample_rate = 16000
        
        # Thread safety
        self.lock = threading.RLock()
        self.result_queue = Queue()
        self.current_result = None
        self.result_ready = threading.Event()
        self.is_processing = False
        
        self.logger.info(f"Initialized DirectMlxWhisperEngine with model={model_name}, language={language}")
    
    def start(self) -> bool:
        """
        Initialize and start the engine.
        
        Returns:
            bool: True if started successfully
        """
        with self.lock:
            try:
                self.logger.info(f"Starting DirectMlxWhisperEngine with model={self.model_name}")
                
                # Download model from HuggingFace
                self.model_path = snapshot_download(
                    repo_id=f'openai/{self.model_name}' if '/' not in self.model_name else self.model_name,
                    allow_patterns=["config.json", "model.safetensors"]
                )
                
                # Load configuration
                with open(f'{self.model_path}/config.json', 'r') as fp:
                    self.cfg = json.load(fp)
                
                # Load weights
                self.weights = [(k.replace("embed_positions.weight", "positional_embedding"), 
                              v.swapaxes(1, 2) if ('conv' in k and v.ndim==3) else v) 
                             for k, v in mx.load(f'{self.model_path}/model.safetensors').items()]
                
                # Initialize model
                self.transcriber = Transcriber(self.cfg)
                self.transcriber.load_weights(self.weights, strict=False)
                self.transcriber.eval()
                mx.eval(self.transcriber)
                
                self.logger.info("DirectMlxWhisperEngine started successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Error starting DirectMlxWhisperEngine: {e}", exc_info=True)
                return False
    
    def transcribe(self, audio: np.ndarray) -> None:
        """
        Transcribe complete audio segment.
        
        Args:
            audio: Audio data as numpy array or file path
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
            
            # For streaming mode, process immediately
            if is_last or len(self.audio_buffer) > 5:  # Process after accumulating enough chunks
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
    
    def _process_audio(self, audio, is_final=False):
        """
        Process audio data and produce transcription.
        
        Args:
            audio: Audio data to transcribe
            is_final: Whether this is the final chunk of audio
        """
        try:
            # Check for empty or invalid audio input
            if isinstance(audio, np.ndarray) and (audio.size == 0 or np.all(audio == 0)):
                self.logger.warning("Empty or silent audio chunk received, returning empty result")
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
                return
                
            start_time = time.time()
            
            # ENHANCED DEBUG: Log detailed information about the audio
            if isinstance(audio, str):
                self.logger.info(f"Processing audio file: {audio}")
                # Copy the file to the project directory for debugging
                try:
                    import shutil
                    import os.path
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
                    debug_path = os.path.join(base_dir, "transcribed_audio.wav")
                    shutil.copy2(audio, debug_path)
                    self.logger.info(f"DEBUGGING: Copied audio file to: {debug_path}")
                except Exception as e:
                    self.logger.error(f"Error copying debug audio file: {e}")
            elif isinstance(audio, np.ndarray):
                # For numpy arrays, log detailed information
                duration_seconds = len(audio) / self.sample_rate if self.sample_rate > 0 else 0
                expected_mel_frames = int(duration_seconds * 100)  # ~100 frames per second
                
                self.logger.info(f"Processing audio array: shape={audio.shape}, samples={len(audio)}, "
                               f"duration={duration_seconds:.2f}s, expected_mel_frames={expected_mel_frames}")
                
                # Log audio statistics for debugging
                if audio.size > 0:
                    self.logger.info(f"Audio stats: min={np.min(audio):.4f}, max={np.max(audio):.4f}, "
                                   f"mean={np.mean(audio):.4f}, non_zero={np.count_nonzero(audio)}")
                    
                # Save a copy of the audio chunk for debugging
                try:
                    import soundfile as sf
                    import os.path
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
                    debug_path = os.path.join(base_dir, "latest_speech.wav")
                    sf.write(debug_path, audio, self.sample_rate)
                    self.logger.info(f"DEBUGGING: Saved audio chunk to: {debug_path}")
                except Exception as e:
                    self.logger.error(f"Error saving debug audio chunk: {e}")
            else:
                # For other types
                self.logger.info(f"Processing audio of type: {type(audio).__name__}")
            
            # Process audio with simplified transcriber
            result = self.transcriber(
                path_audio=audio if isinstance(audio, str) else audio,
                any_lang=(self.language is None),
                quick=self.quick_mode
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Format result
            transcription_result = {
                "text": result,
                "is_final": is_final,
                "language": self.language,
                "processing_time": processing_time,
                "confidence": 1.0,
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
            self.logger.error(f"Error processing audio: {e}", exc_info=True)
            
            # Signal error
            with self.lock:
                error_result = {
                    "error": str(e),
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
            self.logger.info("Cleaning up DirectMlxWhisperEngine")
            
            # Clear references to large objects
            self.transcriber = None
            self.cfg = None
            self.weights = None
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
        return self.transcriber is not None and self.cfg is not None
    
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
                
                # Update streaming mode
                if 'streaming' in config:
                    self.streaming = config['streaming']
                
                # Update quick mode
                if 'quick_mode' in config:
                    self.quick_mode = config['quick_mode']
                
                # Update beam size
                if 'beam_size' in config:
                    self.beam_size = config['beam_size']
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error configuring engine: {e}")
                return False