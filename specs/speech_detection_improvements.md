# Speech Detection and Transcription Improvements

## Current Issues

1. **Missing Beginning of Speech**
   - The Voice Activity Detection (VAD) system doesn't capture the beginning of speech
   - By the time VAD confirms speech is occurring, the user has already spoken several words
   - These initial words are lost and not included in the transcription
   - Example from user testing: "It seems like it doesn't start to record right away"

2. **Debug File Location**
   - Debug audio files being saved to both project root and debug directory
   - `TranscriptionCommandHandler.py` saves to root directory (line ~580)
   - `DirectMlxWhisperEngine.py` now correctly saves to debug directory (line ~965)

3. **VAD Sensitivity and Responsiveness**
   - VAD takes too long to detect and confirm speech
   - The system requires multiple frames (window of audio) before triggering
   - The delay creates a poor user experience as the beginning of sentences are cut off
   - No pre-speech buffering mechanism to capture audio spoken before detection

## Root Causes

1. **Architectural Limitations**
   - VAD inherently needs a window of audio frames to make a confident decision
   - Current implementation only starts a new transcription session after VAD confirms speech
   - No mechanism to include audio from before the detection point

2. **Buffer Management**
   - `VoiceActivityHandler._update_speech_state()` creates a new buffer at the moment speech is detected:
   ```python
   # Line 252
   self.speech_buffer = deque([audio_chunk], maxlen=self.buffer_limit)
   ```
   - The buffer only contains the current chunk that triggered detection, not previous chunks
   - While a large speech buffer exists (maxlen=10000), it's initialized only after speech is detected
   - No buffer exists to capture audio before speech detection

3. **VAD Configuration**
   - Combined detector may be optimized for accuracy but not responsiveness
   - Balancing sensitivity vs. false positives is challenging
   - The `CombinedVadDetector` uses a state machine with four states:
     - SILENCE → POTENTIAL_SPEECH → SPEECH → POTENTIAL_SILENCE
   - It requires multiple consecutive speech frames before confirming speech:
     ```python
     # From CombinedVadDetector.py
     speech_confirmation_frames: int = 3  # Requires 3 consecutive frames of speech
     ```
   - For extra accuracy, it further requires Silero VAD confirmation after WebRTC VAD detection

4. **Audio RMS Energy Level**
   - The system calculates and logs RMS (Root Mean Square) energy level of audio:
     ```python
     # In TranscriptionCommandHandler.py (lines 572-574)
     rms = np.sqrt(np.mean(np.square(audio_reference)))
     self.logger.info(f"Audio RMS energy level: {rms:.4f}")
     ```
   - This RMS value indicates audio amplitude/volume but is not currently used for detection
   - It's logged for informational purposes only and could potentially be used for improved detection

## Proposed Solutions

### 1. Implement Pre-Speech Buffer

Create a continuous circular buffer of recent audio chunks that:
- Always captures the last few seconds of audio, even before speech is detected
- Is prepended to the speech buffer when VAD triggers
- Maintains a configurable lookback window (e.g., 1-2 seconds)

Implementation in `VoiceActivityHandler`:
- Add a new `pre_speech_buffer` with a fixed size (e.g., 32 chunks = ~1 second at 32ms/chunk)
- Always add incoming audio to this buffer, regardless of speech detection
- When speech is detected, prepend all chunks from the pre-speech buffer to the speech buffer
- Configure the pre-speech buffer size via the `ConfigureVadCommand`

### 2. Fix Debug File Location

Change the file saving location in `TranscriptionCommandHandler.py` to use the debug directory (lines ~577-584), matching the pattern used in `DirectMlxWhisperEngine.py`:
- Ensure a consistent debug directory is used
- Add `os.makedirs(debug_dir, exist_ok=True)` to create directory if needed

### 3. Improve VAD Sensitivity and Responsiveness

- Fine-tune VAD parameters for faster detection:
  - Reduce `speech_confirmation_frames` in `CombinedVadDetector` (currently 3)
  - Lower the `silero_threshold` for earlier detection
  - Experiment with `silence_confirmation_frames` (currently 30) if needed
- Focus on ensuring that all pre-speech frames are included:
  - Use a generous pre-speech buffer size to capture all potential speech
  - Critical that no speech frames are lost at the beginning of utterances

## Implementation Plan

### Phase 1: Fix Debug File Location (Completed)
- Update `TranscriptionCommandHandler.py` to save debug files to the debug directory ✓
- Ensure consistent path construction between components ✓

### Phase 2: Implement Pre-Speech Buffer
1. Modify `VoiceActivityHandler.__init__()`:
   - Add a new `pre_speech_buffer = deque(maxlen=32)` (configurable size)
   - The current `speech_buffer` continues to collect speech after detection
   - Document the purpose of each buffer

2. Update `VoiceActivityHandler._on_audio_chunk_captured()`:
   - Add each incoming audio chunk to the pre-speech buffer, regardless of speech state:
   ```python
   # Add to pre-speech buffer - always maintain this circular buffer
   self.pre_speech_buffer.append(audio_chunk)
   ```
   
3. Modify `VoiceActivityHandler._update_speech_state()`:
   - When transitioning from silence to speech (line ~248), include pre-speech buffer contents:
   ```python
   # Create a new speech buffer with pre-speech buffer + current chunk
   self.speech_buffer = deque(list(self.pre_speech_buffer) + [audio_chunk], maxlen=self.buffer_limit)
   ```
   - Adjust speech start time to account for pre-speech buffer duration:
   ```python
   # Adjust speech start time to account for pre-speech buffer
   pre_speech_duration = sum(chunk.get_duration() for chunk in self.pre_speech_buffer)
   self.speech_start_time = current_time - pre_speech_duration
   ```

4. Add configuration options:
   - Update `ConfigureVadCommand` to accept a `pre_speech_buffer_size` parameter
   - Update the constructor to use this parameter
   - Provide documentation and validation for this parameter

### Phase 3: Tune VAD Sensitivity
1. Modify `CombinedVadDetector` parameters for faster detection:
   - Reduce `speech_confirmation_frames` from 3 to 2 for quicker speech detection:
   ```python
   # In CombinedVadDetector.__init__()
   speech_confirmation_frames: int = 2  # Reduced from 3
   ```
   - Adjust `silero_threshold` for better sensitivity:
   ```python
   silero_threshold: float = 0.4  # Reduced from 0.5 for earlier detection
   ```

2. Increase pre-speech buffer size to ensure complete capture:
   - Set a generous buffer size to accommodate different speech patterns:
   ```python
   # In VoiceActivityHandler.__init__()
   self.pre_speech_buffer_size = 64  # ~2 seconds at 32ms/chunk
   self.pre_speech_buffer = deque(maxlen=self.pre_speech_buffer_size)
   ```
   - Make this configurable via the `ConfigureVadCommand` for easy tuning

## Testing Plan

1. Test with varied speech patterns:
   - Quiet beginnings of sentences
   - Fast speech transitions
   - Varied volume levels

2. Measure improvement metrics:
   - Compare before/after recordings of the same speech
   - Measure latency between actual speech start and VAD detection
   - Calculate percentage of words captured before/after improvements

3. User Experience Validation:
   - Gather feedback on transcription completeness
   - Test with multiple speakers and speech patterns

## Conclusion

These improvements should significantly enhance the transcription experience by capturing the beginning of speech that was previously missed. By implementing a pre-speech buffer, the system can include audio from before the VAD trigger point, ensuring complete transcriptions even with the inherent delay in speech detection.

The pre-speech buffer approach provides a good balance between responsiveness and accuracy, allowing the system to maintain high detection accuracy while not losing the beginning of utterances.