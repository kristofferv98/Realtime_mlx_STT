# Wake Word System Optimization Specification

## Problem Statement
The current implementation of the wake word detection system keeps the Voice Activity Detection (VAD) system continuously running, even when not actively listening for speech after a wake word. This is unnecessarily consuming computational resources, especially when the wake word has not been detected.

As observed in the logs, the VAD system processes audio chunks and produces numerous detection messages even when the wake word is not active, which is inefficient.

## Proposed Solution
Implement a two-stage activation system:
1. **Wake Word Listening Stage**: Only the wake word detector runs continuously
2. **Speech Processing Stage**: VAD is dynamically activated only after wake word detection, and deactivated once processing is complete

## Technical Requirements

### 1. Dynamic VAD Management
- Add capability to enable/disable VAD processing on demand
- Prevent VAD events from being processed until wake word is detected
- Properly cleanup VAD subscription when no longer needed

### 2. Event Subscription Optimization
- Modify event subscription logic to only subscribe to VAD events when needed
- Implement a cleaner subscription/unsubscription system

### 3. Resource Usage
- Minimize CPU and memory usage when in idle state
- Reduce the number of VAD event logs during idle periods
- Ensure no memory leaks from event subscriptions

## Implementation Plan

### Phase 1: Modify WakeWordCommandHandler

#### 1. Enhanced State Management
- Update `WakeWordCommandHandler` to include a specific flag for VAD activation
- Implement methods for enabling and disabling VAD processing

```python
# Add new method to WakeWordCommandHandler
def _enable_vad_processing(self) -> None:
    """Enable VAD processing by subscribing to VAD events."""
    if not self.vad_subscribed:
        self.event_bus.subscribe(SpeechDetectedEvent, self._on_speech_detected)
        self.event_bus.subscribe(SilenceDetectedEvent, self._on_silence_detected)
        self.vad_subscribed = True
        self.logger.info("VAD event processing enabled")

# Add new method to WakeWordCommandHandler
def _disable_vad_processing(self) -> None:
    """Disable VAD processing by unsubscribing from VAD events."""
    if self.vad_subscribed:
        self.event_bus.unsubscribe(SpeechDetectedEvent, self._on_speech_detected)
        self.event_bus.unsubscribe(SilenceDetectedEvent, self._on_silence_detected)
        self.vad_subscribed = False
        self.logger.info("VAD event processing disabled")
```

#### 2. Modify Wake Word Detection Logic
- Update the `_on_wake_word_detected` method to enable VAD only after wake word detection
- Update the `_on_silence_detected` method to disable VAD after speech processing

```python
# Modified _on_wake_word_detected method
def _on_wake_word_detected(self, wake_word: str, confidence: float) -> None:
    """Handle wake word detection."""
    self.logger.info(f"Wake word detected: {wake_word} (confidence: {confidence:.2f})")
    
    # Update state
    self.wake_word_detected = True
    self.wake_word_detected_time = time.time()
    self.wake_word_name = wake_word
    self.listening_for_speech = True
    self.state = DetectorState.LISTENING
    
    # Publish wake word detected event
    self.event_bus.publish(WakeWordDetectedEvent(
        wake_word=wake_word,
        confidence=confidence,
        audio_timestamp=self.last_audio_timestamp,
        detector_type=self.active_detector_name,
        audio_reference=self._get_buffered_audio()
    ))
    
    # Start listening for speech with VAD
    self.command_dispatcher.dispatch(
        ConfigureVadCommand(
            detector_type="combined",
            sensitivity=0.7
        )
    )
    
    # Enable VAD processing only after wake word detection
    self._enable_vad_processing()

# Modified _on_silence_detected method
def _on_silence_detected(self, event: SilenceDetectedEvent) -> None:
    """Handle silence detection event."""
    self.logger.info(f"Silence detected, speech duration: {event.speech_duration:.2f}s")
    
    # Only handle if we are currently recording
    if self.state == DetectorState.RECORDING:
        # Update state
        self.state = DetectorState.PROCESSING
        
        # Reset wake word detection state
        self.wake_word_detected = False
        self.wake_word_detected_time = 0
        self.wake_word_name = ""
        self.listening_for_speech = False
        
        # Disable VAD processing after speech is processed
        self._disable_vad_processing()
        
        # Return to wake word detection state after processing
        self.state = DetectorState.WAKE_WORD
```

#### 3. Update Handler Initialization and Cleanup
- Ensure handler is initialized with VAD disabled
- Modify startup and cleanup procedures to manage VAD correctly

```python
# Modified __init__ method
def __init__(self, event_bus: IEventBus, command_dispatcher: CommandDispatcher):
    # ... existing initialization code ...
    
    # VAD subscription state - start disabled
    self.vad_subscribed = False
    
    # Register for audio events - only audio is needed for wake word detection
    self.event_bus.subscribe(AudioChunkCapturedEvent, self._on_audio_chunk_captured)
    
    # Do NOT subscribe to VAD events here - will be subscribed dynamically

# Modified cleanup method
def cleanup(self) -> None:
    """Clean up resources used by the handler."""
    
    # Explicitly disable VAD processing
    self._disable_vad_processing()
    
    # ... rest of existing cleanup code ...
```

### Phase 2: Modify WakeWordTimeoutEvent Handling

- Ensure timeout properly disables VAD processing

```python
# Modified audio chunk processing with timeout handling
if self.wake_word_detected and self.listening_for_speech and self.state == DetectorState.LISTENING:
    timeout_duration = self.config.speech_timeout
    current_time = time.time()
    
    if current_time - self.wake_word_detected_time > timeout_duration:
        # Timeout occurred
        self.logger.info(f"Wake word timeout after {timeout_duration}s without speech")
        
        # Publish timeout event
        self.event_bus.publish(WakeWordTimeoutEvent(
            wake_word=self.wake_word_name,
            timeout_duration=timeout_duration
        ))
        
        # Reset state
        self.wake_word_detected = False
        self.wake_word_detected_time = 0
        self.wake_word_name = ""
        self.listening_for_speech = False
        
        # Disable VAD processing on timeout
        self._disable_vad_processing()
        
        # Return to wake word detection state
        self.state = DetectorState.WAKE_WORD
```

### Phase 3: Modify Example Code

- Update the `wake_word_detection.py` example to demonstrate the optimized approach
- Add flags to track whether VAD is active for debugging purposes

## Expected Benefits

1. **Reduced CPU Usage**
   - VAD processing only runs when needed
   - Overall reduction in audio processing when idle

2. **Improved Power Efficiency**
   - Especially important for battery-powered devices
   - Lower constant load when in wake word listening mode

3. **Cleaner Log Output**
   - Fewer VAD detection messages in the logs when not actively processing
   - Easier to identify important events

4. **Better Architecture**
   - Clearer separation between wake word detection and speech processing stages
   - More maintainable code with explicit state transitions

## Testing Plan

1. **Functionality Testing**
   - Verify wake word detection still works correctly
   - Confirm speech is still properly transcribed after optimization

2. **Performance Testing**
   - Measure CPU usage before and after optimization
   - Compare log volume to confirm reduction in message processing

3. **Regression Testing**
   - Test all existing examples to ensure backward compatibility
   - Check edge cases like back-to-back wake word detection