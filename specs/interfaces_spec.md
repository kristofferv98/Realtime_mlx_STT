# Core Infrastructure: Interfaces Specification

This document outlines the design and implementation plan for the core interfaces in the Realtime_mlx_STT project as part of the vertical slice architecture refactoring.

## Overview

The core interfaces will provide a clear contract between different components of the system, enabling:

1. **Loose coupling** between features
2. **Better testability** through dependency injection
3. **Easier extension** with new implementations
4. **Standardized communication** between vertical slices

## Key Interfaces

### 1. IAudioProvider

The `IAudioProvider` interface defines the contract for any component that provides audio data to the system.

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

class IAudioProvider(ABC):
    """
    Interface for audio input providers that capture audio data from various sources.
    
    Implementations might include microphone input, file input, or network streams.
    """
    
    @abstractmethod
    def setup(self) -> bool:
        """
        Initialize the audio provider and prepare it for recording.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def read_chunk(self) -> bytes:
        """
        Read a single chunk of audio data.
        
        Returns:
            bytes: Raw audio data as bytes
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Get the sample rate of the audio data.
        
        Returns:
            int: Sample rate in Hz
        """
        pass
    
    @abstractmethod
    def get_chunk_size(self) -> int:
        """
        Get the size of each audio chunk in samples.
        
        Returns:
            int: Chunk size in samples
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources used by the audio provider.
        """
        pass
    
    @abstractmethod
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        List available audio input devices.
        
        Returns:
            List[Dict[str, Any]]: List of device information dictionaries
        """
        pass
```

### 2. IVoiceActivityDetector

The `IVoiceActivityDetector` interface defines the contract for voice activity detection components.

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, Tuple

class IVoiceActivityDetector(ABC):
    """
    Interface for voice activity detection components that identify speech in audio.
    
    Implementations might include WebRTC VAD, Silero VAD, or combined approaches.
    """
    
    @abstractmethod
    def setup(self) -> bool:
        """
        Initialize the VAD component.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Determine if an audio chunk contains speech.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state of the VAD component.
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the VAD component.
        
        Returns:
            Dict[str, Any]: Configuration parameters
        """
        pass
    
    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> bool:
        """
        Update the configuration of the VAD component.
        
        Args:
            config: Configuration parameters to update
            
        Returns:
            bool: True if configuration was successfully updated
        """
        pass
```

### 3. ITranscriptionEngine

The `ITranscriptionEngine` interface defines the contract for speech-to-text transcription components.

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, List, Union

class ITranscriptionEngine(ABC):
    """
    Interface for transcription engines that convert audio to text.
    
    Implementations might include MLX-optimized Whisper, other local models,
    or remote API-based transcription services.
    """
    
    @abstractmethod
    def start(self) -> bool:
        """
        Initialize and start the transcription engine.
        
        Returns:
            bool: True if the engine started successfully
        """
        pass
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> None:
        """
        Request transcription of complete audio segment.
        
        Args:
            audio: Audio data as numpy array (float32, -1.0 to 1.0 range)
        """
        pass
    
    @abstractmethod
    def add_audio_chunk(self, audio_chunk: np.ndarray, is_last: bool = False) -> None:
        """
        Add an audio chunk for streaming transcription.
        
        Args:
            audio_chunk: Audio data chunk as numpy array
            is_last: Whether this is the last chunk in the stream
        """
        pass
    
    @abstractmethod
    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get the transcription result (blocking with timeout).
        
        Args:
            timeout: Maximum time to wait for a result in seconds
            
        Returns:
            Optional[Dict[str, Any]]: Transcription result or None if not available
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources used by the transcription engine.
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the transcription engine is currently running.
        
        Returns:
            bool: True if the engine is running
        """
        pass
```

### 4. IWakeWordDetector

The `IWakeWordDetector` interface defines the contract for wake word detection components.

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

class IWakeWordDetector(ABC):
    """
    Interface for wake word detection components.
    
    Implementations might include Porcupine, OpenWakeWord, or custom detectors.
    """
    
    @abstractmethod
    def setup(self, wake_words: List[str], sensitivities: List[float]) -> bool:
        """
        Initialize the wake word detector with specified wake words.
        
        Args:
            wake_words: List of wake word names or paths to models
            sensitivities: List of sensitivity values for each wake word
            
        Returns:
            bool: True if setup was successful
        """
        pass
    
    @abstractmethod
    def process(self, audio_chunk: bytes) -> int:
        """
        Process an audio chunk to detect wake words.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            int: Index of detected wake word or -1 if none detected
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """
        Get the expected sample rate for the detector.
        
        Returns:
            int: Expected sample rate in Hz
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Release resources used by the wake word detector.
        """
        pass
```

### 5. IEventBus

The `IEventBus` interface defines the contract for the event bus that enables communication between features.

```python
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

class Event:
    """Base class for all events in the system."""
    pass

class IEventBus(ABC):
    """
    Interface for the event bus that enables communication between features.
    
    The event bus implements the publish-subscribe pattern, allowing components
    to publish events and subscribe to event types they're interested in.
    """
    
    @abstractmethod
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: The event to publish
        """
        pass
    
    @abstractmethod
    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: The type of event to subscribe to
            handler: The callback function to invoke when the event occurs
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: Type[Event], handler: Callable[[Event], None]) -> bool:
        """
        Unsubscribe from a specific event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            handler: The handler to remove
            
        Returns:
            bool: True if the handler was successfully unsubscribed
        """
        pass
    
    @abstractmethod
    def clear_subscriptions(self) -> None:
        """
        Clear all event subscriptions.
        """
        pass
```

### 6. ICommandHandler

The `ICommandHandler` interface defines the contract for command handlers that implement the mediator pattern.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar

T = TypeVar('T')

class Command:
    """Base class for all commands in the system."""
    pass

class ICommandHandler(Generic[T], ABC):
    """
    Interface for command handlers that process commands and produce results.
    
    The command handler implements the mediator pattern, decoupling command
    senders from command processors.
    """
    
    @abstractmethod
    def handle(self, command: Command) -> T:
        """
        Handle a command and produce a result.
        
        Args:
            command: The command to handle
            
        Returns:
            Result of the command execution
        """
        pass
```

## Implementation Plan

1. **Core Package Structure**
   - Create the `src/Core/Common/Interfaces` directory
   - Implement each interface in its own file

2. **Feature-Specific Interface Extensions**
   - Extend core interfaces with feature-specific interfaces where needed
   - Place these in the respective feature module's directory

3. **Concrete Implementations**
   - Create implementations of these interfaces for each feature
   - Place implementations in their respective feature modules

4. **Implementation of EventBus**
   - The EventBus is a critical component for decoupling features
   - Implement a thread-safe EventBus with proper error handling and logging

5. **Implementation of CommandDispatcher**
   - Create a central CommandDispatcher that routes commands to handlers
   - Register handlers during application initialization

6. **Migration Path**
   - First implement interfaces based on the current functionality
   - Create initial implementations that wrap existing classes
   - Gradually refactor existing code to use the new interfaces

## Next Steps

1. Create the Core directory structure
2. Implement the EventBus
3. Implement the CommandDispatcher
4. Create interfaces for each feature area
5. Begin implementing concrete classes that fulfill these interfaces