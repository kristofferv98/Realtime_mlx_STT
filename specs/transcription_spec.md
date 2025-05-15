# Transcription Feature Specification

## Overview

The Transcription feature implements speech-to-text functionality using MLX-optimized Whisper large-v3-turbo model for Apple Silicon. It integrates with AudioCapture and VoiceActivityDetection features to provide a complete audio processing pipeline. The implementation follows vertical slice architecture principles with process isolation for stability and performance.

## Architecture

### Core Components

1. **ITranscriptionEngine Interface** (already implemented)
   - Located at: `src/Core/Common/Interfaces/transcription_engine.py`
   - Key methods: `start()`, `transcribe()`, `add_audio_chunk()`, `get_result()`, `cleanup()`, `is_running()`
   - Purpose: Abstract interface for any transcription implementation

2. **Directory Structure** (already set up)
   - Commands: `src/Features/Transcription/Commands/`
   - Engines: `src/Features/Transcription/Engines/`
   - Events: `src/Features/Transcription/Events/`
   - Handlers: `src/Features/Transcription/Handlers/`
   - Models: `src/Features/Transcription/Models/`

## Implementation Plan

### 1. Engine Implementation

**MlxWhisperEngine** (`src/Features/Transcription/Engines/MlxWhisperEngine.py`)
- Implements `ITranscriptionEngine`
- Responsibilities:
  - Load and initialize MLX Whisper model
  - Process audio data using mel spectrogram conversion
  - Generate transcriptions using transformer architecture
  - Manage streaming transcription with KV cache
  - Support both batch and streaming modes
- Key optimizations:
  - Memory management for Apple Silicon
  - Caching of model components and filters
  - Process isolation via multiprocessing

#### Core Implementation Examples from whisper_turbo.py

**Model Initialization and Loading:**
```python
def setup(self) -> bool:
    """Initialize the MLX Whisper model."""
    try:
        # Download model from HuggingFace
        self.path_hf = snapshot_download(
            repo_id='openai/whisper-large-v3-turbo', 
            allow_patterns=["config.json", "model.safetensors"]
        )
        
        # Load configuration and weights
        with open(f'{self.path_hf}/config.json', 'r') as fp:
            self.cfg = json.load(fp)
        
        weights = [(k.replace("embed_positions.weight", "positional_embedding"), 
                   v.swapaxes(1, 2) if ('conv' in k and v.ndim==3) else v) 
                  for k, v in mx.load(f'{self.path_hf}/model.safetensors').items()]
        
        # Initialize model components
        self.model = Whisper(self.cfg)
        self.model.load_weights(weights, strict=False)
        self.model.eval()
        self.tokenizer = Tokenizer()
        
        # Warm up with sample audio
        self._warmup()
        return True
    except Exception as e:
        print(f"Error initializing MLX Whisper engine: {e}")
        return False
```

**Audio Preprocessing: Log Mel Spectrogram Generation**
```python
@lru_cache(maxsize=None)
def log_mel_spectrogram(self, audio):
    """Convert audio to log mel spectrogram."""
    if isinstance(audio, str):
        audio = self.load_audio(audio)
    elif not isinstance(audio, mx.array):
        audio = mx.array(audio)
    
    if self.padding > 0:
        audio = mx.pad(audio, (0, self.padding))
        
    window = self.hanning(400)
    freqs = self.stft(audio, window, nperseg=400, noverlap=160)
    magnitudes = freqs[:-1, :].abs().square()
    filters = self.mel_filters(self.n_mels)
    mel_spec = magnitudes @ filters.T
    
    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec
```

**Batch Transcription (Parallel Processing):**
```python
def transcribe_parallel(self, raw, sot):
    """Process audio in parallel chunks for batch processing."""
    # Reshape audio into chunks
    raw = raw[:(raw.shape[0]//3000)*3000].reshape(-1, 3000, 128)
    
    # Prepare initial tokens
    sot = mx.repeat(sot, raw.shape[0], 0)
    
    # Encode audio with model
    mel = self.model.encode(raw)
    
    # Initialize variables
    B = mel.shape[0]
    new_tok = mx.zeros((B, 0), dtype=mx.int32)
    goon = mx.ones((B, 1), dtype=mx.bool_)
    kv_cache = None
    
    # Generate tokens sequentially
    for i in range(449 - sot.shape[-1]):
        logits, kv_cache, _ = self.model.decode(txt=sot if i==0 else txt, mel=mel, kv_cache=kv_cache)
        txt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True) * goon
        mx.eval(txt)
        goon *= (txt != 50257)  # Continue until EOS token
        new_tok = mx.concatenate([new_tok, txt], axis=-1)
        if goon.sum() <= 0:
            break
    
    # Extract results
    arg_hop = mx.argmax(new_tok, axis=-1).tolist()
    new_tok = [i[:a] for i, a in zip(new_tok.astype(mx.int32).tolist(), arg_hop)]
    new_tok = [i for i in sum(new_tok, []) if i < 50257]
    
    # Convert tokens to text
    return self.tokenizer.decode(new_tok)[0]
```

**Streaming Transcription (Recurrent Processing):**
```python
def transcribe_recurrent(self, raw, sot, kv_cache=None):
    """Process audio sequentially, maintaining context via KV cache."""
    # Initialize token buffer and position
    new_tok, i = mx.zeros((1, 0), dtype=mx.int32), 0
    
    # Process audio in chunks
    while i + 3000 < len(raw):
        # Encode chunk
        mel = self.model.encode(raw[i:i+3000][None])
        
        # Initial tokens or continue from previous
        txt = sot if i == 0 else mx.array([[50365]])  # Use timestamp token
        
        # Generate tokens
        for j in range(449 - (self.len_sot if i == 0 else 1)):
            logits, kv_cache, _ = self.model.decode(txt=txt, mel=mel, kv_cache=kv_cache)
            txt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(txt)
            
            # Exit on EOS token
            if txt.item() == 50257:
                break
                
            new_tok = mx.concatenate([new_tok, txt], axis=-1)
        
        # Find timestamp token to determine next chunk
        arg_hop = mx.argmax(mel[:, :, 0]).item()
        hop = (arg_hop - 50365) * 2 if arg_hop >= 50365 else 3000
        i += hop if hop > 0 else 3000
    
    # Process final tokens
    new_tok = [i for i in new_tok.astype(mx.int32).tolist()[0] if i < 50257]
    text = self.tokenizer.decode(new_tok)[0]
    
    return text, kv_cache
```

**Tokenizer Implementation:**
```python
class Tokenizer:
    def __init__(self):
        # Initialize tokenizer with the Whisper vocabulary
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
        if isinstance(lot, str):
            lot = [lot]
        return [self.encoding.encode(t, allowed_special='all') for t in lot]
    
    def decode(self, lol):
        if isinstance(lol[0], int):
            lol = [lol]
        return [self.encoding.decode(l) for l in lol]
```

**TranscriptionProcessManager** (`src/Features/Transcription/Engines/TranscriptionProcessManager.py`)
- Manages a separate process for transcription
- Uses pipe-based IPC for communication
- Handles process lifecycle (start, stop, monitoring)
- Provides thread-safe access to transcription results

#### Process Management Implementation
```python
class TranscriptionProcessManager:
    """Manages the transcription process in a separate Python process."""
    
    def __init__(self):
        """Initialize the process manager."""
        self.process = None
        self.parent_pipe = None
        self.child_pipe = None
        
    def start(self, engine_type="mlx_whisper", engine_config=None):
        """Start the transcription process with specified engine."""
        # Create pipes for communication
        self.parent_pipe, self.child_pipe = multiprocessing.Pipe()
        
        # Create and start the process
        self.process = multiprocessing.Process(
            target=self._run_transcription_process,
            args=(self.child_pipe, engine_type, engine_config),
            daemon=True
        )
        self.process.start()
        
        # Verify process started correctly
        if not self.parent_pipe.poll(10.0):  # Wait for ready signal
            self.stop()
            return False
            
        response = self.parent_pipe.recv()
        return response.get('success', False)
    
    def transcribe(self, audio_data, is_first_chunk=False, is_last_chunk=False, options=None):
        """Send audio data to transcription process."""
        if not self.is_running():
            return {"error": "Transcription process not running"}
            
        # Send transcription request
        self.parent_pipe.send({
            'command': 'TRANSCRIBE',
            'audio_data': audio_data,
            'is_first_chunk': is_first_chunk,
            'is_last_chunk': is_last_chunk,
            'options': options or {}
        })
        
        # Wait for response
        if self.parent_pipe.poll(30.0):  # 30 second timeout
            return self.parent_pipe.recv()
        else:
            return {"error": "Transcription timeout"}
    
    def configure(self, config):
        """Configure the transcription engine."""
        if not self.is_running():
            return False
            
        # Send configuration request
        self.parent_pipe.send({
            'command': 'CONFIGURE',
            'config': config
        })
        
        # Wait for response
        if self.parent_pipe.poll(10.0):  # 10 second timeout
            response = self.parent_pipe.recv()
            return response.get('success', False)
        else:
            return False
    
    def stop(self):
        """Stop the transcription process."""
        if self.is_running():
            # Send shutdown command
            self.parent_pipe.send({'command': 'SHUTDOWN'})
            
            # Wait for process to terminate
            self.process.join(5.0)
            
            # Force terminate if necessary
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2.0)
            
            # Cleanup
            self.parent_pipe.close()
            self.child_pipe.close()
            self.process = None
            self.parent_pipe = None
            self.child_pipe = None
            
        return True
    
    def is_running(self):
        """Check if transcription process is running."""
        return self.process is not None and self.process.is_alive()
    
    @staticmethod
    def _run_transcription_process(pipe, engine_type, engine_config):
        """Process function that runs the transcription engine."""
        # Create and initialize the appropriate engine
        engine = None
        
        try:
            if engine_type == "mlx_whisper":
                from src.Features.Transcription.Engines.MlxWhisperEngine import MlxWhisperEngine
                engine = MlxWhisperEngine(**(engine_config or {}))
                
            # Initialize the engine
            if engine and engine.start():
                # Signal ready
                pipe.send({'success': True})
                
                # Process commands until shutdown
                while True:
                    if pipe.poll(0.1):
                        command = pipe.recv()
                        
                        if command['command'] == 'TRANSCRIBE':
                            # Process audio data
                            try:
                                engine.add_audio_chunk(
                                    command['audio_data'],
                                    command['is_last_chunk']
                                )
                                
                                # Get result (blocks until available or timeout)
                                result = engine.get_result(
                                    timeout=command.get('options', {}).get('timeout', 5.0)
                                )
                                
                                # Send result back
                                pipe.send(result or {"error": "No result available"})
                            except Exception as e:
                                pipe.send({"error": str(e)})
                        
                        elif command['command'] == 'CONFIGURE':
                            # Configure the engine
                            try:
                                success = engine.configure(command['config'])
                                pipe.send({'success': success})
                            except Exception as e:
                                pipe.send({'success': False, 'error': str(e)})
                        
                        elif command['command'] == 'SHUTDOWN':
                            # Clean up and exit
                            break
            else:
                # Signal initialization failure
                pipe.send({'success': False, 'error': 'Failed to initialize engine'})
                
        except Exception as e:
            # Signal error
            pipe.send({'success': False, 'error': str(e)})
            
        finally:
            # Clean up
            if engine:
                engine.cleanup()
            pipe.close()
```

### 2. Models

**TranscriptionResult** (`src/Features/Transcription/Models/TranscriptionResult.py`)
- Properties: `text`, `confidence`, `language`, `is_final`, `segments`, `timestamps`
- Used to represent transcription outputs

**TranscriptionConfig** (`src/Features/Transcription/Models/TranscriptionConfig.py`)
- Properties for engine configuration: `model_name`, `compute_type`, `language`, `beam_size`
- Additional streaming options: `chunk_duration`, `overlap`, `realtime_factor`

**TranscriptionSession** (`src/Features/Transcription/Models/TranscriptionSession.py`)
- Tracks state for ongoing transcription sessions
- Properties: `session_id`, `start_time`, `audio_chunks`, `current_text`, `language`

### 3. Commands

**TranscribeAudioCommand** (`src/Features/Transcription/Commands/TranscribeAudioCommand.py`)
- Parameters: `audio_chunk`, `session_id`, `is_first_chunk`, `is_last_chunk`, `language`
- Requests transcription of a specific audio chunk

**ConfigureTranscriptionCommand** (`src/Features/Transcription/Commands/ConfigureTranscriptionCommand.py`)
- Parameters: `engine_type`, `model_name`, `language`, `beam_size`, `compute_type`, `options`
- Updates transcription engine configuration

**StartTranscriptionSessionCommand** (`src/Features/Transcription/Commands/StartTranscriptionSessionCommand.py`)
- Parameters: `session_id`, `language`, `config`
- Initializes a new transcription session

**StopTranscriptionSessionCommand** (`src/Features/Transcription/Commands/StopTranscriptionSessionCommand.py`)
- Parameters: `session_id`, `flush_remaining_audio`
- Finalizes and ends a transcription session

### 4. Events

**TranscriptionStartedEvent** (`src/Features/Transcription/Events/TranscriptionStartedEvent.py`)
- Properties: `session_id`, `timestamp`, `language`
- Published when transcription begins for a speech segment

**TranscriptionUpdatedEvent** (`src/Features/Transcription/Events/TranscriptionUpdatedEvent.py`)
- Properties: `session_id`, `text`, `is_final`, `confidence`, `timestamp`
- Published for both partial and final transcription results

**TranscriptionErrorEvent** (`src/Features/Transcription/Events/TranscriptionErrorEvent.py`)
- Properties: `session_id`, `error_message`, `error_type`, `timestamp`
- Published when transcription encounters an error

### 5. Handler Implementation

**TranscriptionCommandHandler** (`src/Features/Transcription/Handlers/TranscriptionCommandHandler.py`)
- Implements `ICommandHandler`
- Handles all transcription-related commands
- Manages transcription sessions and engine instances
- Publishes transcription events
- Coordinates with the process manager for isolated processing

### 6. Module Facade

**TranscriptionModule** (`src/Features/Transcription/TranscriptionModule.py`)
- Public static API for the Transcription feature
- Methods:
  - `register()`: Register with command dispatcher and event bus
  - `configure()`: Configure transcription engine
  - `transcribe_audio()`: Transcribe a specific audio chunk
  - `start_session()`: Start a new transcription session
  - `stop_session()`: End a transcription session
  - Event subscription methods for results and errors

## Integration with Other Features

### AudioCapture Integration
- Subscribes to `AudioChunkCapturedEvent`
- Routes audio chunks to active transcription sessions
- Handles proper audio format conversion

### VoiceActivityDetection Integration
- Subscribes to `SpeechDetectedEvent` and `SilenceDetectedEvent`
- Automatically starts transcription sessions when speech begins
- Finalizes transcription when speech ends
- Maintains session IDs consistent with VAD speech segments

## Technical Requirements

### Process Isolation
- Transcription engine runs in separate Python process
- Communication via pipe-based IPC
- Resource management (memory, CPU, Neural Engine)
- Graceful handling of process failures

### Performance Optimization
- Memory management for Apple Silicon
- Efficient audio buffer handling
- Cached preprocessing for mel spectrograms
- KV cache management for streaming transcription
- Adaptive chunk sizing based on latency measurements

### Error Handling
- Graceful recovery from model errors
- Timeout mechanisms for unresponsive transcription
- Proper resource cleanup
- Detailed error reporting

## Implementation Phases

### Phase 1: Basic Infrastructure
- Implement `TranscriptionProcessManager`
- Create basic `MlxWhisperEngine` (non-streaming)
- Define data models and commands
- Implement command handler with process isolation

### Phase 2: Batch Transcription
- Complete implementation of batch transcription
- Add integration with VAD for complete speech segments
- Implement proper error handling and recovery

### Phase 3: Streaming Transcription
- Add streaming capability with KV cache management
- Implement partial results publishing
- Optimize for low latency
- Add chunking and context management

### Phase 4: Optimization and Testing
- Performance benchmarking
- Memory usage optimization
- Latency reduction
- Comprehensive unit and integration tests

## Testing Strategy

### Unit Tests
- Test each component in isolation with mocks
- Verify command and event handling
- Test process isolation utilities

### Integration Tests
- End-to-end tests with audio input to text output
- Test with various audio conditions
- Test integration with other features

## Acceptance Criteria

1. Successfully transcribe speech with >95% accuracy on clean audio
2. Support both batch and streaming transcription modes
3. Process audio in near real-time (RTF <1.0 on M1/M2 chips)
4. Seamlessly integrate with AudioCapture and VoiceActivityDetection
5. Handle errors gracefully with proper reporting
6. Manage resources efficiently without memory leaks
7. Support configuration changes at runtime