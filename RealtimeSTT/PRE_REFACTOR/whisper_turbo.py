import base64
import glob
import json
import math
import os
import time
import logging
from functools import lru_cache
from subprocess import CalledProcessError, run

import fire
import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download, snapshot_download

# Configure logging
logger = logging.getLogger("realtimestt.whisper_turbo")

class Tokenizer:
    def __init__(self):
        path_tok = 'multilingual.tiktoken'
        if not os.path.exists(path_tok):
            path_tok = hf_hub_download(repo_id='JosefAlbers/whisper', filename=path_tok)
        with open(path_tok) as f:
            ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line)}
        n_vocab = len(ranks)
        specials = ["<|endoftext|>", "<|startoftranscript|>", *[f"<|_{lang}|>" for lang in range(100)], "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>", *[f"<|{i * 0.02:.2f}|>" for i in range(1501)]]
        special_tokens = {k:(n_vocab+i) for i,k in enumerate(specials)}
        self.encoding = tiktoken.Encoding(name='jj', explicit_n_vocab=n_vocab + len(special_tokens), pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", mergeable_ranks=ranks, special_tokens=special_tokens)
    def encode(self, lot):
        if isinstance(lot, str):
            lot = [lot]
        return [self.encoding.encode(t, allowed_special='all') for t in lot]
    def decode(self, lol):
        if isinstance(lol[0], int):
            lol = [lol]
        return [self.encoding.decode(l) for l in lol]

def load_audio(file, sr=16000):
    try:
        out = run(["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0

@lru_cache(maxsize=None)
def mel_filters(n_mels):
    path_mel = "mel_filters.npz"
    if not os.path.exists(path_mel):
        np.savez_compressed(path_mel, mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128))
    return mx.load(path_mel)[f"mel_{n_mels}"]

@lru_cache(maxsize=None)
def hanning(n_fft):
    return mx.array(np.hanning(n_fft + 1)[:-1])

@lru_cache(maxsize=None)
def stft(x, window, nperseg=400, noverlap=160, nfft=None, axis=-1, pad_mode="reflect"):
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

def log_mel_spectrogram(audio, n_mels=128, padding=480000):
    """
    Convert audio to log-mel spectrogram.
    
    Args:
        audio: Either a file path (str), NumPy array (np.ndarray), or MLX array (mx.array)
        n_mels: Number of mel bands
        padding: Amount of padding to add (set to 0 when audio is already chunked appropriately)
        
    Returns:
        mx.array: Log-mel spectrogram
    """
    if isinstance(audio, str):
        # Load audio from file path
        audio = load_audio(audio)
    elif isinstance(audio, np.ndarray):
        # Direct NumPy array support
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

def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)

class MultiHeadAttention(nn.Module):
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
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv1d(cfg['num_mel_bins'], cfg['d_model'], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cfg['d_model'], cfg['d_model'], kernel_size=3, stride=2, padding=1)
        self._positional_embedding = sinusoids(cfg['max_source_positions'], cfg['d_model']).astype(mx.float16)
        self.layers = [ResidualAttentionBlock(cfg['d_model'], cfg['encoder_attention_heads']) for _ in range(cfg['encoder_layers'])]
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
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
        self.positional_embedding = mx.zeros((cfg['max_target_positions'], cfg['d_model']))
        self.layers = [ResidualAttentionBlock(cfg['d_model'], cfg['decoder_attention_heads'], cross_attention=True) for _ in range(cfg['decoder_layers'])]
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
    def __init__(self, cfg):
        self.encoder = AudioEncoder(cfg)
        self.decoder = TextDecoder(cfg)
    def __call__(self, mel, txt):
        return self.decoder(txt, self.encoder(mel))[0]
    def encode(self, mel):
        return self.encoder(mel)
    def decode(self, txt, mel, kv_cache):
        return self.decoder(txt, mel, kv_cache)

class Transcriber(nn.Module):
    def __init__(self, cfg):
        self.model = Whisper(cfg)
        self.tokenizer = Tokenizer()
        self.len_sot = 0
    def __call__(self, audio_input, any_lang, quick, language=None):
        """
        Transcribe audio to text.
        
        Args:
            audio_input: Audio input as file path (str) or NumPy array (np.ndarray)
            any_lang: Whether to detect language automatically
            quick: Whether to use parallel processing for fast transcription
            language: Language code (e.g., 'en', 'fr', 'de', etc.) if any_lang is False
            
        Returns:
            str: Transcribed text
        """
        raw = log_mel_spectrogram(audio_input).astype(mx.float16)
        
        # Determine SOT (Start-Of-Transcript) sequence
        if any_lang:
            # Language auto-detection
            sot = mx.array([[50258, 50360, 50365]])  # <|startoftranscript|>, <|transcribe|>
        else:
            # Use specific language
            language_code = language or 'en'  # Default to English if not specified
            
            # Get language token ID
            # Default English token ID is 50259 (<|en|>)
            language_token_id = 50259  # Default to English
            
            if language_code != 'en':
                # Try to get the correct language token from the tokenizer
                try:
                    language_token = f"<|_{language_code}|>"
                    token_id = self.tokenizer.encoding.encode(language_token, allowed_special='all')[0]
                    if token_id:
                        language_token_id = token_id
                except:
                    logger.warning(f"Could not find token for language '{language_code}', defaulting to English")
            
            # Construct SOT with language token
            sot = mx.array([[50258, language_token_id, 50360, 50365]])  # <|startoftranscript|>, <|lang|>, <|transcribe|>
            
        self.len_sot = sot.shape[-1]
        txt = self.parallel(raw, sot) if quick else self.recurrent(raw, sot)
        return txt
    def recurrent(self, raw, sot):
        new_tok, i = mx.zeros((1,0), dtype=mx.int32), 0
        while i+3000 < len(raw):
            piece = self.step(raw[i:i+3000][None], sot)
            arg_hop = mx.argmax(piece).item()
            hop = (piece[:,arg_hop].astype(mx.int32).item()-50365)*2
            new_tok = mx.concatenate([new_tok, piece[:,:arg_hop]], axis=-1)
            i += hop if hop > 0 else 3000
        new_tok = [i for i in new_tok.astype(mx.int32).tolist()[0] if i < 50257]
        return self.tokenizer.decode(new_tok)[0]
    def parallel(self, raw, sot):
        raw = raw[:(raw.shape[0]//3000)*3000].reshape(-1, 3000, 128)
        assert raw.shape[0] < 360
        sot = mx.repeat(sot, raw.shape[0], 0)
        new_tok = self.step(raw, sot)
        arg_hop = mx.argmax(new_tok, axis=-1).tolist()
        new_tok = [i[:a] for i,a in zip(new_tok.astype(mx.int32).tolist(),arg_hop)]
        new_tok = [i for i in sum(new_tok, []) if i < 50257]
        return self.tokenizer.decode(new_tok)[0]
    def step(self, mel, txt):
        mel = self.model.encode(mel)
        kv_cache = None
        B = mel.shape[0]
        new_tok = mx.zeros((B,0), dtype=mx.int32)
        goon = mx.ones((B,1), dtype=mx.bool_)
        for i in range(449-self.len_sot):
            logits, kv_cache, _ = self.model.decode(txt=txt, mel=mel, kv_cache=kv_cache)
            txt = mx.argmax(logits[:,-1,:], axis=-1, keepdims=True) * goon
            mx.eval(txt)
            goon *= (txt != 50257)
            new_tok = mx.concatenate([new_tok, txt], axis=-1)
            if goon.sum() <= 0:
                break
        return new_tok

class StreamingTranscriber(nn.Module):
    """
    Streaming transcription module optimized for real-time audio.
    
    This class processes audio chunks incrementally, maintaining state
    between chunks for continuous transcription with minimal latency.
    
    Args:
        cfg (dict): Model configuration
        buffer_size (int): Size of audio buffer in samples
        overlap (int): Overlap between processing segments in samples
    """
    def __init__(self, cfg, buffer_size=16000, overlap=2000):
        super().__init__()
        self.model = Whisper(cfg)
        self.tokenizer = Tokenizer()
        
        # Streaming state
        self.audio_buffer = []
        self.text_tokens = []
        self.kv_cache = None
        self.buffer_size = buffer_size
        self.overlap = overlap
        self.sot_sequence = None
        
        # Initial prompt management
        self.any_lang = False
        self.language = None  # Store specific language code
        self.len_sot = 0
        
        # Create mask for attention
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(cfg['max_target_positions']).astype(mx.float16)
        
        logger.info(f"StreamingTranscriber initialized with buffer_size={buffer_size}, overlap={overlap}")
    
    def process_chunk(self, audio_chunk, is_last=False, language=None):
        """
        Process a new chunk of audio in streaming mode.
        
        Args:
            audio_chunk (np.ndarray): New audio data
            is_last (bool): Whether this is the final chunk
            language (str, optional): Force specific language (e.g., 'en' for English)
            
        Returns:
            dict: {
                'text': Transcribed text so far
                'is_final': Whether this is a final result
                'new_text': Text transcribed from this chunk
            }
        """
        # Handle language setting
        if language is not None:
            self.any_lang = False
            self.language = language
        elif language is None and self.sot_sequence is None:
            self.any_lang = True
            self.language = None
        
        # Convert and buffer audio chunk
        if isinstance(audio_chunk, np.ndarray):
            # Handle NumPy array
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            # Normalize if not already in [-1, 1] range
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 0 and max_val > 1.0:
                audio_chunk = audio_chunk / max_val
            self.audio_buffer.append(mx.array(audio_chunk))
        elif isinstance(audio_chunk, mx.array):
            self.audio_buffer.append(audio_chunk)
        else:
            raise TypeError(f"Unsupported audio chunk type: {type(audio_chunk)}")
        
        # Calculate total buffer length
        buffer_length = sum(chunk.shape[0] for chunk in self.audio_buffer)
        
        # Process when we have enough audio or this is the last chunk
        if buffer_length >= self.buffer_size or is_last:
            # Concatenate buffer
            audio_data = mx.concatenate(self.audio_buffer) if len(self.audio_buffer) > 1 else self.audio_buffer[0]
            
            # Process segment
            result = self._process_segment(audio_data, is_last)
            
            # Manage buffer for next chunk (keep overlap)
            if not is_last and buffer_length > self.overlap:
                overlap_samples = min(self.overlap, buffer_length)
                self.audio_buffer = [audio_data[-overlap_samples:]]
            elif is_last:
                # Clear buffer on last chunk
                self.audio_buffer = []
            
            return result
        else:
            # Not enough audio yet
            return {
                'text': self.tokenizer.decode([self.text_tokens])[0] if self.text_tokens else "",
                'is_final': False,
                'new_text': ""
            }
    
    def _process_segment(self, audio_data, is_final):
        """
        Process a segment of audio data.
        
        Args:
            audio_data (mx.array): Audio data to process
            is_final (bool): Whether this is the final segment
            
        Returns:
            dict: Processing results
        """
        # Convert to mel spectrogram
        mel = log_mel_spectrogram(audio_data, padding=0).astype(mx.float16)
        
        # Encode mel spectrogram
        mel_encoded = self.model.encode(mel[None, ...])  # Add batch dimension
        
        # Initialize SOT sequence if not already done
        if self.sot_sequence is None:
            # Start of transcript tokens
            if self.any_lang:
                # [<|startoftranscript|>, <|transcribe|>]
                self.sot_sequence = mx.array([[50258, 50365]])
                self.len_sot = 2
            else:
                # Use specific language token
                language_code = self.language or 'en'  # Default to English if not specified
                
                # Get language token ID
                # Default English token ID is 50259 (<|en|>)
                language_token_id = 50259  # Default to English
                
                if language_code != 'en':
                    # Try to get the correct language token from the tokenizer
                    try:
                        language_token = f"<|_{language_code}|>"
                        token_id = self.tokenizer.encoding.encode(language_token, allowed_special='all')[0]
                        if token_id:
                            language_token_id = token_id
                    except:
                        logger.warning(f"Could not find token for language '{language_code}', defaulting to English")
                
                # [<|startoftranscript|>, <|lang|>, <|transcribe|>]
                self.sot_sequence = mx.array([[50258, language_token_id, 50365]])
                self.len_sot = 3
        
        # Prepare initial tokens if this is the first segment
        if not self.text_tokens:
            prompt = self.sot_sequence
        else:
            # Use last few tokens as prompt
            context_tokens = self.text_tokens[-5:] if len(self.text_tokens) > 5 else self.text_tokens
            prompt = mx.array([context_tokens], dtype=mx.int32)
        
        # Decode with KV cache
        new_tokens = []
        current_prompt = prompt
        
        for i in range(100):  # Maximum new tokens to generate per segment
            logits, self.kv_cache, _ = self.model.decode(
                txt=current_prompt,
                mel=mel_encoded,
                kv_cache=self.kv_cache
            )
            
            # Get the next token
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            token_id = next_token.item()
            
            # Stop if end of text token
            if token_id == 50257:  # End of text token <|endoftext|>
                break
                
            new_tokens.append(token_id)
            current_prompt = next_token  # Next iteration uses this token
            
            # Early stopping for streaming (can be tuned)
            if not is_final and i > 10 and token_id in [50, 51, 52]:  # Period, comma, or other sentence break
                break
        
        # Update state
        old_token_count = len(self.text_tokens)
        self.text_tokens.extend(new_tokens)
        
        # Create result
        current_text = self.tokenizer.decode([self.text_tokens])[0] if self.text_tokens else ""
        previous_text = self.tokenizer.decode([self.text_tokens[:old_token_count]])[0] if old_token_count > 0 else ""
        new_text = current_text[len(previous_text):] if previous_text else current_text
        
        # If this is the final segment, reset state for future use
        if is_final:
            self.reset()
        
        return {
            'text': current_text,
            'is_final': is_final,
            'new_text': new_text
        }
    
    def reset(self):
        """Reset the streaming state."""
        self.audio_buffer = []
        self.text_tokens = []
        self.kv_cache = None
        self.sot_sequence = None
        # Don't reset language settings as they're set by the user


# Global variables for model caching
_cached_model = None
_cached_model_path = None
_cached_streaming_model = None

def transcribe(audio_input=None, any_lang=False, quick=False, model_path="openai/whisper-large-v3-turbo", language=None):
    """
    Transcribe audio using the MLX-optimized Whisper model.
    
    Args:
        audio_input: Audio input as file path (str) or NumPy array (np.ndarray)
        any_lang: Whether to detect language automatically
        quick: Whether to use parallel processing
        model_path: HuggingFace model ID or local path
        language: Language code (e.g., 'en', 'fr', 'de', etc.) if any_lang is False
        
    Returns:
        str: Transcribed text
    """
    global _cached_model, _cached_model_path
    
    if audio_input is None:
        return benchmark()
    
    # Check if we need to load the model
    if _cached_model is None or _cached_model_path != model_path:
        path_hf = snapshot_download(repo_id=model_path, 
                                  allow_patterns=["config.json", "model.safetensors"])
        with open(f'{path_hf}/config.json', 'r') as fp:
            cfg = json.load(fp)
        weights = [(k.replace("embed_positions.weight", "positional_embedding"), 
                  v.swapaxes(1, 2) if ('conv' in k and v.ndim==3) else v) 
                  for k, v in mx.load(f'{path_hf}/model.safetensors').items()]
        
        _cached_model = Transcriber(cfg)
        _cached_model.load_weights(weights, strict=False)
        _cached_model.eval()
        mx.eval(_cached_model)
        _cached_model_path = model_path
    
    # Use the cached model
    return _cached_model(audio_input=audio_input, any_lang=any_lang, quick=quick, language=language)

def create_streaming_transcriber(buffer_size=16000, overlap=2000, model_path="openai/whisper-large-v3-turbo"):
    """
    Create a streaming transcriber for incremental audio processing.
    
    Args:
        buffer_size (int): Size of audio buffer in samples
        overlap (int): Overlap between processing segments
        model_path (str): HuggingFace model ID or local path
        
    Returns:
        StreamingTranscriber: Initialized streaming transcriber
    """
    global _cached_streaming_model, _cached_model_path
    
    # Define a cache key based on parameters
    cache_key = (model_path, buffer_size, overlap)
    
    # Check if we already have a cached model with the same parameters
    if (_cached_streaming_model is not None and 
        hasattr(_cached_streaming_model, '_cache_key') and 
        _cached_streaming_model._cache_key == cache_key):
        # Return the cached model if parameters match
        return _cached_streaming_model
    
    # Download and load model
    path_hf = snapshot_download(repo_id=model_path, 
                              allow_patterns=["config.json", "model.safetensors"])
    with open(f'{path_hf}/config.json', 'r') as fp:
        cfg = json.load(fp)
    weights = [(k.replace("embed_positions.weight", "positional_embedding"), 
              v.swapaxes(1, 2) if ('conv' in k and v.ndim==3) else v) 
              for k, v in mx.load(f'{path_hf}/model.safetensors').items()]
    
    # Create streaming transcriber
    _cached_streaming_model = StreamingTranscriber(cfg, buffer_size=buffer_size, overlap=overlap)
    _cached_streaming_model.load_weights(weights, strict=False)
    _cached_streaming_model.eval()
    mx.eval(_cached_streaming_model)
    
    # Store the cache key for future comparison
    _cached_streaming_model._cache_key = cache_key
    
    return _cached_streaming_model

def benchmark():
    path_hf = snapshot_download(repo_id='JosefAlbers/exurb1a', allow_patterns=["*.mp3"])
    tics = {}
    for path_audio in sorted(glob.glob(f"{path_hf}/*.mp3")):
        for any_lang in [True, False]:
            for quick in [True, False]:
                tic = time.perf_counter()
                arg = f"{path_audio.split('/')[-1]} {any_lang=} {quick=}"
                print(f"--- {arg=}")
                print(transcribe(path_audio=path_audio, any_lang=any_lang, quick=quick))
                tic = f"{(time.perf_counter() - tic):.2f}"
                print(f"{tic=}")
                tics[arg] = tic
    return tics

def fire_main():
    fire.Fire(transcribe)

if __name__ == '__main__':
    fire.Fire(transcribe)

# benchmarks:
# 0_test.mp3 any_lang=True quick=True:    0.85
# 0_test.mp3 any_lang=True quick=False:   0.75
# 0_test.mp3 any_lang=False quick=True:   0.78
# 0_test.mp3 any_lang=False quick=False:  0.77
# 1_alive.mp3 any_lang=True quick=True:   7.10
# 1_alive.mp3 any_lang=True quick=False:  7.98
# 1_alive.mp3 any_lang=False quick=True:  6.57
# 1_alive.mp3 any_lang=False quick=False: 7.98
# 2_make.mp3 any_lang=True quick=True:    7.30
# 2_make.mp3 any_lang=True quick=False:   13.30
# 2_make.mp3 any_lang=False quick=True:   6.26
# 2_make.mp3 any_lang=False quick=False:  11.10
# 3_try.mp3 any_lang=True quick=True:     8.62
# 3_try.mp3 any_lang=True quick=False:    14.79
# 3_try.mp3 any_lang=False quick=True:    7.87
# 3_try.mp3 any_lang=False quick=False:   15.21
# 4_never.mp3 any_lang=True quick=True:   11.70
# 4_never.mp3 any_lang=True quick=False:  17.70
# 4_never.mp3 any_lang=False quick=True:  10.67
# 4_never.mp3 any_lang=False quick=False: 19.48
