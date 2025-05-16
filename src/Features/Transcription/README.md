# Transcription Feature

## Overview

The Transcription feature provides high-performance speech-to-text capabilities optimized for Apple Silicon. It converts audio data into text using the MLX-optimized Whisper large-v3-turbo model, supporting both batch processing for accuracy and streaming for real-time applications. The feature uses process isolation for stability, provides a robust event system for notifications, and offers a clean command-based API for integration.

## Directory Structure

```
Transcription/
├── Commands/                  # Command definitions
│   ├── ConfigureTranscriptionCommand.py   # Configure engine settings
│   ├── StartTranscriptionSessionCommand.py  # Start a session
│   ├── StopTranscriptionSessionCommand.py   # End a session
│   └── TranscribeAudioCommand.py           # Process audio data
├── Engines/                   # Transcription engine implementations
│   ├── DirectMlxWhisperEngine.py      # Direct MLX-optimized Whisper engine
│   ├── DirectTranscriptionManager.py  # Direct in-process manager
│   └── OLD_BACKUP/                    # Legacy implementations (backup)
├── Events/                    # Event definitions
│   ├── TranscriptionStartedEvent.py    # Transcription begins
│   ├── TranscriptionUpdatedEvent.py    # Text is updated
│   └── TranscriptionErrorEvent.py      # Error occurred
├── Handlers/                  # Command handlers
│   └── TranscriptionCommandHandler.py  # Processes all transcription commands
├── Models/                    # Domain models
│   ├── TranscriptionConfig.py          # Engine configuration
│   ├── TranscriptionResult.py          # Transcription output
│   └── TranscriptionSession.py         # Session state management
├── TranscriptionModule.py     # Feature registration and facade
└── README.md                  # This documentation
```

## Key Components

### Models

#### `TranscriptionResult`

Represents the output of a transcription operation with metadata.

```python
@dataclass
class TranscriptionResult:
    text: str                             # Transcribed text
    is_final: bool                        # Whether this is a final result
    session_id: str                       # Session identifier
    timestamp: float                      # Timestamp in milliseconds
    language: Optional[str] = None        # Language code (e.g., 'en')
    confidence: float = 1.0               # Confidence score (0.0-1.0)
    segments: List[TranscriptionSegment] = field(default_factory=list)  # Time-aligned segments
    processing_time: Optional[float] = None  # Processing time in ms
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    # Methods
    @property
    def has_segments(self) -> bool: ...   # Whether segments are available
    def to_dict(self) -> Dict[str, Any]: ...  # Convert to dictionary
```

#### `TranscriptionConfig`

Configures the behavior of transcription engines and sessions.

```python
@dataclass
class TranscriptionConfig:
    engine_type: str = "mlx_whisper"      # Engine selection
    model_name: str = "whisper-large-v3-turbo"  # Model to use
    language: Optional[str] = None        # Language code or None for auto
    compute_type: Literal["default", "float16", "float32"] = "float16"  # Precision
    beam_size: int = 1                    # Beam search size
    streaming: bool = True                # Enable streaming mode
    chunk_duration_ms: int = 1000         # Chunk size in milliseconds
    chunk_overlap_ms: int = 200           # Overlap in milliseconds
    realtime_factor: float = 0.5          # Target processing speed
    max_context_length: int = 128         # Context token limit
    options: Dict[str, Any] = field(default_factory=dict)  # Engine-specific options
    
    # Methods
    @property
    def chunk_duration_samples(self) -> int: ...  # Convert ms to samples
    @property
    def chunk_overlap_samples(self) -> int: ...   # Convert ms to samples
    def to_dict(self) -> Dict[str, Any]: ...      # Convert to dictionary
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TranscriptionConfig': ...
```

#### `TranscriptionSession`

Manages the state of an ongoing transcription session.

```python
@dataclass
class TranscriptionSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique ID
    start_time: float = field(default_factory=time.time)  # Start timestamp
    last_activity_time: float = field(default_factory=time.time)  # Last update
    config: TranscriptionConfig = field(default_factory=TranscriptionConfig)  # Settings
    is_active: bool = True                # Whether session is active
    language: Optional[str] = None        # Detected language
    detected_language_confidence: float = 0.0  # Language detection confidence
    current_text: str = ""                # Current transcription text
    results: List[TranscriptionResult] = field(default_factory=list)  # All results
    _audio_chunks: List[np.ndarray] = field(default_factory=list)  # Stored audio
    _total_audio_duration_ms: float = 0.0  # Total audio duration
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    # Methods
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> None: ...  # Add audio
    def get_combined_audio(self) -> np.ndarray: ...  # Get all audio
    def get_latest_audio(self, duration_ms: Optional[float] = None) -> np.ndarray: ...
    def clear_audio_buffer(self) -> None: ...  # Clear stored audio
    def add_result(self, result: TranscriptionResult) -> None: ...  # Add result
    def close(self) -> None: ...  # Mark as inactive
    @property
    def duration_ms(self) -> float: ...  # Total duration
    @property
    def idle_time(self) -> float: ...  # Time since last activity
    @property
    def audio_sample_count(self) -> int: ...  # Total samples
    def to_dict(self) -> Dict[str, Any]: ...  # Convert to dictionary
```

### Commands

#### `TranscribeAudioCommand`

Command to transcribe an audio chunk.

```python
@dataclass
class TranscribeAudioCommand(Command):
    audio_chunk: np.ndarray              # Audio data as numpy array
    session_id: str                      # Session identifier
    is_first_chunk: bool = False         # Whether this is the first chunk
    is_last_chunk: bool = False          # Whether this is the final chunk
    timestamp_ms: float = 0.0            # Audio timestamp in ms
    language: Optional[str] = None       # Language code or None for auto
    options: Dict[str, Any] = field(default_factory=dict)  # Additional options
```

**Returns:** `Dict[str, Any]` - Transcription result dictionary

#### `ConfigureTranscriptionCommand`

Command to configure the transcription engine settings.

```python
@dataclass
class ConfigureTranscriptionCommand(Command):
    engine_type: str = "mlx_whisper"     # Engine type to use
    model_name: str = "whisper-large-v3-turbo"  # Model name
    language: Optional[str] = None       # Language code or None for auto
    beam_size: int = 1                   # Beam search size
    compute_type: Literal["default", "float16", "float32"] = "float16"  # Precision
    streaming: bool = True               # Enable streaming mode
    chunk_duration_ms: int = 1000        # Chunk size in milliseconds
    chunk_overlap_ms: int = 200          # Overlap in milliseconds
    options: Dict[str, Any] = field(default_factory=dict)  # Engine-specific options
```

**Returns:** `bool` - True if configuration was successful

#### `StartTranscriptionSessionCommand`

Command to start a new transcription session.

```python
@dataclass
class StartTranscriptionSessionCommand(Command):
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Session ID
    language: Optional[str] = None       # Language code or None for auto
    streaming: bool = True               # Enable streaming mode
    config: Dict[str, Any] = field(default_factory=dict)  # Additional configuration
```

**Returns:** `Dict[str, Any]` - Session information

#### `StopTranscriptionSessionCommand`

Command to stop an ongoing transcription session.

```python
@dataclass
class StopTranscriptionSessionCommand(Command):
    session_id: str                      # Session ID to stop
    flush_remaining_audio: bool = True   # Process remaining audio
    save_results: bool = False           # Save results to file
    output_path: Optional[str] = None    # Path to save results
```

**Returns:** `Dict[str, Any]` - Operation result

### Events

#### `TranscriptionStartedEvent`

Event published when transcription begins.

```python
class TranscriptionStartedEvent(Event):
    session_id: str                      # Session identifier
    language: Optional[str] = None       # Language code
    audio_timestamp: float = 0.0         # Audio timestamp in ms
```

#### `TranscriptionUpdatedEvent`

Event published when transcription text is updated.

```python
class TranscriptionUpdatedEvent(Event):
    session_id: str                      # Session identifier
    text: str                            # Transcribed text
    is_final: bool                       # Whether this is a final result
    confidence: float = 1.0              # Confidence score (0.0-1.0)
    language: Optional[str] = None       # Language code
    audio_timestamp: float = 0.0         # Audio timestamp in ms
    processing_time: Optional[float] = None  # Processing time in ms
    segments: Optional[List[Dict[str, Any]]] = None  # Time-aligned segments
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
```

#### `TranscriptionErrorEvent`

Event published when transcription encounters an error.

```python
class TranscriptionErrorEvent(Event):
    session_id: str                      # Session identifier
    error_message: str                   # Human-readable error message
    error_type: str                      # Type of error
    audio_timestamp: float = 0.0         # Audio timestamp in ms
    details: Optional[Dict[str, Any]] = None  # Additional error details
    recovery_attempted: bool = False     # Whether recovery was attempted
    recovery_successful: bool = False    # Whether recovery was successful
```

### Engines

#### `DirectMlxWhisperEngine`

Direct MLX-optimized Whisper engine for Apple Silicon without process isolation.

```python
class DirectMlxWhisperEngine(ITranscriptionEngine):
    def __init__(self, 
                model_name: str = "whisper-large-v3-turbo",
                language: Optional[str] = None,
                compute_type: str = "float16",
                beam_size: int = 1,
                **kwargs): ...
                
    # ITranscriptionEngine methods
    def start(self) -> bool: ...                          # Initialize engine
    def transcribe(self, audio_data: Any) -> Dict[str, Any]: ...  # Complete audio transcription
    def add_audio_chunk(self, audio_chunk: Union[np.ndarray, mx.array], is_last: bool = False) -> Dict[str, Any]: ...  # Streaming
    def cleanup(self) -> None: ...                        # Release resources
    def is_running(self) -> bool: ...                     # Check status
    
    # Additional methods
    def configure(self, config: Dict[str, Any]) -> bool: ...  # Update settings
    def _process_audio(self, audio: Union[str, np.ndarray, mx.array], is_final: bool = False) -> Dict[str, Any]: ...  # Core processing
```

#### `DirectTranscriptionManager`

Manages the transcription engine in the same process.

```python
class DirectTranscriptionManager:
    def __init__(self): ...
    
    def start(self, engine_type: str = "mlx_whisper", 
             engine_config: Optional[Dict[str, Any]] = None) -> bool: ...  # Start engine
    def transcribe(self, audio_data: Any, 
                  is_first_chunk: bool = False,
                  is_last_chunk: bool = False, 
                  options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...  # Process audio
    def configure(self, config: Dict[str, Any]) -> bool: ...  # Update settings
    def stop(self) -> bool: ...                               # Stop engine
    def is_running(self) -> bool: ...                         # Check status
```

### Module Facade

#### `TranscriptionModule`

Provides a simple interface for registering and using the Transcription feature.

```python
class TranscriptionModule:
    @staticmethod
    def register(command_dispatcher: CommandDispatcher,
                event_bus: IEventBus,
                default_engine: str = "mlx_whisper",
                default_model: str = "whisper-large-v3-turbo",
                default_language: Optional[str] = None) -> TranscriptionCommandHandler: ...
                
    @staticmethod
    def configure(command_dispatcher: CommandDispatcher,
                 engine_type: str = "mlx_whisper",
                 model_name: str = "whisper-large-v3-turbo",
                 language: Optional[str] = None,
                 streaming: bool = True,
                 **kwargs) -> bool: ...
    
    @staticmethod
    def start_session(command_dispatcher: CommandDispatcher,
                     session_id: Optional[str] = None,
                     language: Optional[str] = None,
                     streaming: bool = True,
                     **kwargs) -> Dict[str, Any]: ...
    
    @staticmethod
    def stop_session(command_dispatcher: CommandDispatcher,
                    session_id: str,
                    flush_remaining_audio: bool = True,
                    save_results: bool = False,
                    output_path: Optional[str] = None) -> Dict[str, Any]: ...
    
    @staticmethod
    def transcribe_audio(command_dispatcher: CommandDispatcher,
                        audio_data: np.ndarray,
                        session_id: Optional[str] = None,
                        is_first_chunk: bool = False,
                        is_last_chunk: bool = False,
                        language: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]: ...
    
    @staticmethod
    def transcribe_file(command_dispatcher: CommandDispatcher,
                       file_path: str,
                       language: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]: ...
    
    @staticmethod
    def register_vad_integration(event_bus: IEventBus,
                               transcription_handler: TranscriptionCommandHandler,
                               session_id: Optional[str] = None,
                               auto_start_on_speech: bool = True) -> None: ...
    
    @staticmethod
    def on_transcription_started(event_bus: IEventBus,
                               handler: Callable[[str, Optional[str], float], None]) -> None: ...
    
    @staticmethod
    def on_transcription_updated(event_bus: IEventBus,
                               handler: Callable[[str, str, bool, float], None]) -> None: ...
    
    @staticmethod
    def on_transcription_error(event_bus: IEventBus,
                              handler: Callable[[str, str, str], None]) -> None: ...
```

## Usage Examples

### Basic Registration and Setup

```python
from src.Core.Events.event_bus import EventBus
from src.Core.Commands.command_dispatcher import CommandDispatcher
from src.Features.Transcription.TranscriptionModule import TranscriptionModule

# Create core infrastructure
event_bus = EventBus()
command_dispatcher = CommandDispatcher()

# Register the Transcription feature
handler = TranscriptionModule.register(
    command_dispatcher=command_dispatcher,
    event_bus=event_bus,
    default_engine="mlx_whisper",
    default_model="whisper-large-v3-turbo",
    default_language=None  # Auto-detect language
)
```

### Configuring Transcription

```python
# Configure transcription engine
TranscriptionModule.configure(
    command_dispatcher=command_dispatcher,
    engine_type="mlx_whisper",
    model_name="whisper-large-v3-turbo",
    language="en",  # Set to English
    streaming=True,
    chunk_duration_ms=1200,  # Larger chunks for better context
    chunk_overlap_ms=300,    # More overlap for better continuity
    beam_size=2,             # Use beam search for better accuracy
    compute_type="float16"   # Precision for computation
)
```

### Transcribing Audio Data

```python
# Get audio data from somewhere (e.g., AudioCapture feature)
import numpy as np

# Create sample audio data (silence)
audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz

# Transcribe audio
result = TranscriptionModule.transcribe_audio(
    command_dispatcher=command_dispatcher,
    audio_data=audio_data,
    language="en",
    is_first_chunk=True,
    is_last_chunk=True
)

print(f"Transcription: {result['text']}")
print(f"Is Final: {result['is_final']}")
print(f"Confidence: {result.get('confidence', 1.0)}")
```

### Working with Transcription Sessions

```python
# Start a session
session_result = TranscriptionModule.start_session(
    command_dispatcher=command_dispatcher,
    language="en",
    streaming=True,
    chunk_duration_ms=800,  # Session-specific settings
    chunk_overlap_ms=150
)

session_id = session_result["session_id"]

# Transcribe multiple audio chunks in the session
for i in range(5):
    # Get or create some audio data
    audio_chunk = np.zeros(8000, dtype=np.float32)  # 0.5 seconds at 16kHz
    
    # Transcribe as part of the session
    result = TranscriptionModule.transcribe_audio(
        command_dispatcher=command_dispatcher,
        audio_data=audio_chunk,
        session_id=session_id,
        is_first_chunk=(i == 0),
        is_last_chunk=(i == 4)
    )
    
    # Print partial results
    if 'text' in result:
        print(f"Chunk {i+1} transcription: {result['text']}")

# Stop the session
TranscriptionModule.stop_session(
    command_dispatcher=command_dispatcher,
    session_id=session_id,
    flush_remaining_audio=True,
    save_results=True,
    output_path="transcription_results.json"
)
```

### Transcribing a File

```python
# Transcribe an audio file
result = TranscriptionModule.transcribe_file(
    command_dispatcher=command_dispatcher,
    file_path="speech_sample.wav",
    language=None,  # Auto-detect language
    streaming=False  # Use batch mode for better accuracy
)

print(f"File Transcription: {result['text']}")
```

### Handling Transcription Events

```python
# Define event handlers
def on_transcription_start(session_id, language, timestamp):
    print(f"Transcription started: Session ID = {session_id}")
    print(f"Language: {language or 'Auto-detecting'}")
    print(f"Timestamp: {timestamp:.2f}ms")

def on_transcription_update(session_id, text, is_final, confidence):
    status = "FINAL" if is_final else "Partial"
    print(f"[{status}] [{confidence:.2f}]: {text}")

def on_transcription_error(session_id, error_message, error_type):
    print(f"Transcription Error ({error_type}): {error_message}")
    print(f"Session ID: {session_id}")

# Subscribe to events
TranscriptionModule.on_transcription_started(event_bus, on_transcription_start)
TranscriptionModule.on_transcription_updated(event_bus, on_transcription_update)
TranscriptionModule.on_transcription_error(event_bus, on_transcription_error)
```

## Integration with Other Features

The Transcription feature integrates with other features through:

1. **Processing Audio**: Receives audio data from the AudioCapture feature via `AudioChunkCapturedEvent` subscription.

2. **Speech Detection Integration**: Works with the VoiceActivityDetection feature to process only speech segments.

3. **Event-Based Architecture**: Publishes events that applications can subscribe to for receiving transcription results.

4. **Command-Mediator Pattern**: All actions are performed through commands, decoupling the feature from other components.

Example integration with AudioCapture and VoiceActivityDetection:

```python
# Initialize core components
event_bus = EventBus()
command_dispatcher = CommandDispatcher()

# Register features
audio_handler = AudioCaptureModule.register(command_dispatcher, event_bus)
vad_handler = VadModule.register(command_dispatcher, event_bus)
transcription_handler = TranscriptionModule.register(command_dispatcher, event_bus)

# Track active transcription sessions
active_sessions = {}

# Handle speech detection (start transcription)
def on_speech_detected(confidence, timestamp, speech_id):
    # Start a new transcription session when speech begins
    result = TranscriptionModule.start_session(
        command_dispatcher,
        session_id=speech_id,
        streaming=True
    )
    active_sessions[speech_id] = result.get('session_id')

# Handle silence detection (end transcription)
def on_silence_detected(speech_duration, start_time, end_time, speech_id):
    # Stop transcription session when speech ends
    if speech_id in active_sessions:
        TranscriptionModule.stop_session(
            command_dispatcher,
            session_id=active_sessions[speech_id],
            flush_remaining_audio=True
        )
        del active_sessions[speech_id]

# Handle audio chunk processing
def on_audio_chunk(audio_chunk):
    # If associated with active speech, send to transcription
    speech_id = getattr(audio_chunk, 'speech_id', None)
    if speech_id and speech_id in active_sessions:
        TranscriptionModule.transcribe_audio(
            command_dispatcher,
            audio_data=audio_chunk.numpy_data,
            session_id=active_sessions[speech_id],
            timestamp_ms=audio_chunk.timestamp
        )

# Handle transcription results
def on_transcription_updated(session_id, text, is_final, confidence):
    print(f"Transcription: {text}")
    if is_final:
        print("Final transcription completed.")

# Subscribe to events
VadModule.on_speech_detected(event_bus, on_speech_detected)
VadModule.on_silence_detected(event_bus, on_silence_detected)
AudioCaptureModule.on_audio_chunk_captured(event_bus, on_audio_chunk)
TranscriptionModule.on_transcription_updated(event_bus, on_transcription_updated)

# Start audio capture
AudioCaptureModule.start_recording(command_dispatcher)
```

## Advanced Features

### Direct MLX Integration

The Transcription feature now runs the MLX-optimized Whisper model directly in-process to:

1. **Improve Performance**: Eliminate IPC overhead for 2-3x faster processing
2. **Enhance Reliability**: Avoid serialization issues with large tensors
3. **Reduce Complexity**: Simplify the architecture and error handling
4. **Support Two Processing Modes**: Quick/parallel (faster) and recurrent/sequential (more accurate)

### VAD Integration

The new `register_vad_integration` method provides automatic transcription of complete speech segments:

```python
# Simple VAD integration with one method call
TranscriptionModule.register_vad_integration(
    event_bus=event_bus,
    transcription_handler=transcription_handler,
    session_id=None,  # Auto-generate unique session for each speech segment
    auto_start_on_speech=True  # Start a new session when speech is detected
)
```

This integration:
1. **Detects Complete Sentences**: Transcribes complete speech segments
2. **Improves Accuracy**: Processes speech-only segments for better results
3. **Reduces Processing**: Avoids transcribing silence or background noise
4. **Simplifies Integration**: Replaces dozens of lines of manual integration code

### Streaming Transcription

The `DirectMlxWhisperEngine` provides streaming transcription with:

1. **Incremental Processing**: Process audio chunks as they arrive
2. **Context Preservation**: Maintain context between chunks
3. **KV Cache Management**: Efficient key-value cache for transformer layers
4. **Early Results**: Generate partial transcriptions before speech ends

### Performance Optimization

Optimization techniques used in the engine:

1. **Mixed Precision**: Using float16 for faster computation on Apple Silicon
2. **Cached Operations**: LRU caching for expensive operations like mel filters
3. **Efficient Audio Preprocessing**: Optimized mel spectrogram generation
4. **Memory Management**: Careful buffer management and cleanup
5. **Parallel Processing**: Batch processing of audio segments in parallel

## Dependencies

- **MLX**: Apple's ML framework optimized for Apple Silicon
- **NumPy**: For audio data processing
- **Tiktoken**: For tokenization
- **HuggingFace Hub**: For model downloading and management
- **Core Module**: For event bus, command dispatcher, and interface definitions
- **AudioCapture Feature**: For audio input (consumed through events)
- **VoiceActivityDetection Feature**: For speech detection (integrated through events)