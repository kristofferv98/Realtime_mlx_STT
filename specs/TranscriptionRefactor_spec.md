# Transcription Feature Refactoring Specification

## Overview

This document outlines a plan to refactor the Transcription feature in the Realtime_mlx_STT system. The current implementation uses process isolation with IPC (Inter-Process Communication), which has been identified as the source of significant issues including deadlocks, data serialization problems, and unreliable transcription results. The refactoring will replace this approach with a direct, in-process implementation based on the successful `simple_mlx_whisper_test.py` script.

## Current Issues

1. **Process Isolation Problems**: 
   - Inter-process communication causing serialization errors with large audio/tensor data
   - Deadlocks and synchronization issues between parent and child processes
   - Timeouts during model initialization or inference
   - Corrupted transcription results despite successful process execution

2. **Performance Impact**:
   - Additional overhead from process management
   - Latency introduced by IPC mechanisms
   - Inefficient memory usage due to data duplication across processes

## Proposed Solution

Refactor the Transcription feature to use direct, in-process model execution while preserving the existing vertical slice architecture and event-based communication patterns. The core MLX Whisper implementation from `simple_mlx_whisper_test.py` will replace the current process-isolated approach.

### Key Components to Modify

1. **MlxWhisperEngine**:
   - Remove process isolation mechanisms
   - Integrate the simplified Transcriber implementation
   - Maintain compatibility with existing interfaces

2. **TranscriptionProcessManager**:
   - Convert to a simpler TranscriptionManager without process isolation
   - Retain command handling but execute directly in the same process
   - Maintain existing interface for backward compatibility

3. **Event Handling**:
   - Ensure proper integration with VAD events (SilenceDetectedEvent)
   - Maintain subscription interfaces for TranscriptionUpdatedEvent

## Implementation Plan

### Phase 1: Core Transcription Engine Refactoring

1. Create a new `DirectMlxWhisperEngine` class based on `simple_mlx_whisper_test.py`:
   ```python
   class DirectMlxWhisperEngine(ITranscriptionEngine):
       """Direct in-process MLX Whisper implementation without process isolation."""
       
       def __init__(self, model_name="whisper-large-v3-turbo", language=None, compute_type="float16", beam_size=1, **kwargs):
           # Initialize attributes
           self.model_name = model_name
           self.language = language
           self.compute_type = compute_type
           self.beam_size = beam_size
           self.quick_mode = kwargs.get('quick_mode', True)  # Default to quick/parallel processing
           
           # Model state
           self.transcriber = None
           self.cfg = None
           self.weights = None
           self.model_path = None
       
       def start(self):
           """Initialize the model."""
           # Download model from HuggingFace
           self.model_path = snapshot_download(
               repo_id=f'openai/{self.model_name}',
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
           
           return True
       
       def transcribe(self, audio):
           """Transcribe complete audio."""
           return self._process_audio(audio, is_final=True)
       
       def add_audio_chunk(self, audio_chunk, is_last=False):
           """Process audio chunk."""
           return self._process_audio(audio_chunk, is_final=is_last)
       
       def _process_audio(self, audio, is_final=False):
           """Core processing method."""
           start_time = time.time()
           
           # Process audio using the simplified implementation
           result = self.transcriber(
               path_audio=audio if isinstance(audio, str) else audio,
               any_lang=(self.language is None),
               quick=self.quick_mode
           )
           
           processing_time = time.time() - start_time
           
           # Format result
           return {
               "text": result,
               "is_final": is_final,
               "language": self.language,
               "processing_time": processing_time,
               "confidence": 1.0,
               "success": True
           }
       
       # Other ITranscriptionEngine interface methods
   ```

2. Replace the existing process manager with a direct manager:
   ```python
   class DirectTranscriptionManager:
       """Simplified transcription manager without process isolation."""
       
       def __init__(self):
           self.engine = None
           self.logger = logging.getLogger(__name__)
       
       def start(self, engine_type="mlx_whisper", engine_config=None):
           """Start the transcription engine."""
           config = engine_config or {}
           
           if engine_type == "mlx_whisper":
               self.engine = DirectMlxWhisperEngine(**config)
               return self.engine.start()
           else:
               self.logger.error(f"Unsupported engine type: {engine_type}")
               return False
       
       def transcribe(self, audio_data, is_first_chunk=False, is_last_chunk=False, options=None):
           """Process transcription request."""
           if not self.is_running():
               return {"error": "Transcription engine not running"}
           
           try:
               if is_first_chunk and is_last_chunk:
                   # Complete audio file
                   return self.engine.transcribe(audio_data)
               else:
                   # Streaming audio chunk
                   return self.engine.add_audio_chunk(audio_data, is_last=is_last_chunk)
           except Exception as e:
               self.logger.error(f"Error in transcription: {e}", exc_info=True)
               return {"error": str(e)}
       
       # Other manager methods
   ```

### Phase 2: Integration with Event System

1. Modify `TranscriptionModule` to use the direct implementation:
   ```python
   # Update import
   from src.Features.Transcription.Engines.DirectTranscriptionManager import DirectTranscriptionManager
   
   # Replace TranscriptionProcessManager instance creation
   self.process_manager = DirectTranscriptionManager()
   ```

2. Add a dedicated handler for VAD events:
   ```python
   def _on_silence_detected(self, session_id, audio_reference, duration):
       """Handle silence detection events for transcription."""
       if not audio_reference or len(audio_reference) == 0:
           return
       
       # Process the complete speech segment
       self.transcribe_audio(
           audio_data=audio_reference,
           is_first_chunk=True,
           is_last_chunk=True,
           language=self.config.language
       )
   ```

### Phase 3: Testing and Validation

1. Update unit tests to work with the new implementation
2. Create integration tests that verify:
   - Complete file transcription
   - Streaming audio transcription
   - VAD integration
   - Event publication
3. Benchmark performance against the previous implementation

## Interface Compatibility

The refactored implementation will maintain the same public APIs:

- `TranscriptionModule.transcribe_audio()`
- `TranscriptionModule.transcribe_file()`
- `TranscriptionModule.on_transcription_updated()`

This ensures backward compatibility with existing code while improving the internal implementation.

## Performance Expectations

Based on the `simple_mlx_whisper_test.py` results:

- **Quick mode**: ~4.5 seconds for a 3-minute audio file (~0.025x realtime)
- **Recurrent mode**: ~7.8 seconds for a 3-minute audio file (~0.043x realtime)

Both modes provide high-quality transcription, with recurrent mode offering slightly better accuracy at the cost of processing time.

## Recommendations

1. Use `quick=True` (parallel processing) for real-time applications where latency is critical
2. Use `quick=False` (recurrent processing) for offline processing where accuracy is more important
3. Maintain singleton model instances to avoid redundant loading of large model files
4. Consider separating model loading from transcription to reduce initial latency

## Conclusion

The proposed refactoring eliminates the problematic process isolation mechanism while preserving the system's architecture and event-based communication patterns. By adopting the direct implementation approach from `simple_mlx_whisper_test.py`, we can achieve reliable transcription performance with significantly improved stability.