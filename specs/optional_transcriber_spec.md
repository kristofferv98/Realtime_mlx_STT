# Optional GPT-4o-transcribe Implementation Specification

## Overview

This specification outlines the implementation of OpenAI's GPT-4o-transcribe as an optional transcription engine alternative to the current MLX-optimized Whisper model in the Realtime_mlx_STT system. The implementation will maintain all current functionality while providing a cloud-based alternative for users who prefer OpenAI's state-of-the-art transcription capabilities.

## Goals

1. Implement a new `OpenAITranscriptionEngine` that follows the `ITranscriptionEngine` interface
2. Support both the full `gpt-4o-transcribe` model and the lighter `gpt-4o-mini-transcribe` for different use cases
3. Maintain complete compatibility with the existing audio pipeline and event system
4. Support both real-time streaming transcription and batch transcription
5. Allow explicit user choice between transcription engines without automatic fallbacks

## Design

### Component Structure

The implementation will follow the established vertical slice architecture pattern and be organized as follows:

```
src/
└── Features/
    └── Transcription/
        ├── Engines/
        │   ├── __init__.py                     (updated to include new engine)
        │   ├── DirectMlxWhisperEngine.py       (existing)
        │   ├── DirectTranscriptionManager.py   (existing) 
        │   └── OpenAITranscriptionEngine.py    (new)
        ├── Models/
        │   └── TranscriptionConfig.py          (updated to include OpenAI options)
        ├── TranscriptionModule.py              (updated to support OpenAI engine selection)
        └── Handlers/
            └── TranscriptionCommandHandler.py  (updated to support OpenAI engine)
```

### OpenAITranscriptionEngine Class

The new engine will implement the `ITranscriptionEngine` interface and support both streaming and batch transcription modes:

```python
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
            model_name: "gpt-4o-transcribe" or "gpt-4o-mini-transcribe"
            language: Language code or None for auto-detection
            api_key: OpenAI API key (will fall back to env var if None)
            streaming: Whether to use streaming mode
            **kwargs: Additional configuration options
        """
        # Implementation details
        
    # Interface method implementations
    def start(self) -> bool:
        """Initialize API client and verify connectivity."""
        
    def transcribe(self, audio: np.ndarray) -> None:
        """Transcribe complete audio segment using OpenAI API."""
        
    def add_audio_chunk(self, audio_chunk: np.ndarray, is_last: bool = False) -> None:
        """Add audio chunk for streaming transcription."""
        
    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get transcription result with timeout."""
        
    def cleanup(self) -> None:
        """Close connections and clean up resources."""
        
    def is_running(self) -> bool:
        """Check if the engine is connected and ready."""
        
    # Helper methods
    def _process_audio(self, audio_data, is_final=False):
        """Process audio with appropriate API call based on mode."""
        
    def _handle_websocket_streaming(self, audio_data):
        """Handle streaming audio via WebSocket API."""
        
    def _handle_standard_transcription(self, audio_data):
        """Handle standard transcription via REST API."""
```

### Implementation Details

#### API Integration

The implementation will use the official OpenAI Python client library for API calls. For WebSocket streaming, it will use the `websocket-client` library to implement real-time streaming:

```python
# For standard transcription
async def transcribe_audio_file(self, audio_data):
    """Transcribe audio using standard API."""
    import tempfile
    import soundfile as sf
    
    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        sf.write(temp_file.name, audio_data, 16000, format='WAV', subtype='PCM_16')
        temp_file.flush()
        
        # Transcribe with appropriate model
        with open(temp_file.name, "rb") as audio_file:
            client = openai.OpenAI(api_key=self.api_key)
            result = client.audio.transcriptions.create(
                model=self.model_name,
                file=audio_file,
                response_format="text"
            )
            
    return result

# For WebSocket streaming implementation
def setup_websocket_connection(self):
    """Set up WebSocket connection for real-time transcription."""
    # Implementation details for WebSocket setup
```

#### Configuration Updates

The `TranscriptionConfig` model will be updated to include OpenAI-specific options:

```python
@dataclass
class TranscriptionConfig:
    """Configuration for transcription engines."""
    
    # Existing fields
    model: str = "whisper-large-v3-turbo"
    language: Optional[str] = None
    sample_rate: int = 16000
    compute_type: str = "float16"
    streaming: bool = True
    
    # New fields
    engine_type: str = "mlx"  # Options: "mlx", "openai"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-transcribe"  # or "gpt-4o-mini-transcribe"
```

#### API Connectivity Verification

The engine will verify API connectivity and provide clear error messages if there are issues, but will not automatically fall back to local models:

```python
def start(self) -> bool:
    """Initialize and verify API connectivity."""
    try:
        # Test API connectivity
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=5
        )
        if response.status_code == 200:
            self.logger.info("Successfully connected to OpenAI API")
            return True
        else:
            self.logger.error(f"Failed to connect to OpenAI API: {response.status_code}")
            return False
    except Exception as e:
        self.logger.error(f"Error connecting to OpenAI API: {e}")
        return False
```

#### Audio Format Handling

The engine will ensure audio is properly formatted for OpenAI's API requirements:

```python
def _prepare_audio(self, audio_data):
    """
    Prepare audio for the OpenAI API.
    
    Args:
        audio_data: Raw audio data (numpy array)
        
    Returns:
        Properly formatted audio file or data
    """
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
```

### Integration with User Interface

The examples will be updated to support using the OpenAI engine as an option:

```python
# Example script modification
parser.add_argument('--engine', type=str, default='mlx', choices=['mlx', 'openai'],
                   help='Transcription engine to use (mlx or openai)')
parser.add_argument('--openai-model', type=str, default='gpt-4o-mini-transcribe',
                   choices=['gpt-4o-transcribe', 'gpt-4o-mini-transcribe'],
                   help='OpenAI model to use')
```

## Implementation Plan

1. **Phase 1: Core Engine Implementation**
   - Create `OpenAITranscriptionEngine.py` with REST API integration
   - Update configuration models and module initialization
   - Implement standard (non-streaming) transcription capability
   - Add unit tests

2. **Phase 2: Streaming Implementation**
   - Add WebSocket-based real-time transcription
   - Implement audio chunking and buffer management
   - Add streaming transcription test cases

3. **Phase 3: Integration and Examples**
   - Update example scripts to support OpenAI engine option
   - Create new example demonstrating OpenAI-specific features
   - Document API key setup and management

## Technical Requirements

1. **Dependencies:**
   - `openai>=1.3.0` for API access
   - `websocket-client>=1.4.0` for WebSocket streaming
   - `requests>=2.28.0` for REST API calls

2. **Environment Configuration:**
   - Support for API key via:
     - Direct configuration parameter
     - Environment variable (OPENAI_API_KEY)
     - Configuration file (~/.openai/config)

3. **Error Handling:**
   - Clear error messaging for connectivity or authentication issues
   - Appropriate handling of API errors and timeouts
   - Rate limiting handling with exponential backoff

## Security Considerations

1. **API Key Management:**
   - Never log API keys in debug output
   - Store API keys in environment variables, not in code
   - Support for user-provided API keys at runtime

2. **Data Privacy:**
   - Warn users that audio data will be sent to OpenAI servers
   - Implement option to disable sending debug audio files
   - Document data retention policies of OpenAI API

## Testing Strategy

1. **Unit Tests:**
   - Mock OpenAI API responses for deterministic testing
   - Test error handling with simulated connectivity failures
   - Verify correct audio format conversion

2. **Integration Tests:**
   - End-to-end tests with actual API calls (using test API key)
   - Comparison tests between local and OpenAI transcription results
   - Performance benchmarks for latency and accuracy

## Future Extensions

1. **Additional Models:**
   - Support for future OpenAI speech models as they become available
   - Extensible architecture for other cloud transcription services

2. **Advanced Features:**
   - Support for speaker diarization if/when available
   - Multi-language mode support
   - Custom prompt engineering for domain-specific transcription

## Conclusion

This implementation will provide users with a flexible choice between local MLX-optimized Whisper processing and OpenAI's cloud-based GPT-4o-transcribe, while maintaining all the existing functionality and architecture of the system. The design emphasizes user control through explicit engine selection and full compatibility with the current event-driven system.